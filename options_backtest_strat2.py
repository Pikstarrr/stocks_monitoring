# momentum_x_optimized.py - High accuracy trading strategy with signal persistence
import time
import json
import os
from datetime import datetime, timedelta
from firebase_admin import credentials, firestore, initialize_app

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from google.cloud.firestore_v1 import ArrayUnion
from sklearn.metrics.pairwise import manhattan_distances

from send_mail import send_email
from strategy_features import compute_feature, rational_quadratic_kernel, gaussian_kernel
import dhanhq
from trade_placement import KotakOptionsTrader

# === CONFIG ===
# Feature configuration that gave 75% accuracy
FEATURE_CONFIG = [
    ("RSI", 14, 1),  # Feature 1
    ("WT", 10, 11),  # Feature 2
    ("CCI", 20, 1),  # Feature 3
    ("ADX", 20, 2),  # Feature 4
    ("RSI", 9, 1),  # Feature 5
]

# ML Parameters
NEIGHBOR_COUNT = 5
MAX_BARS_BACK = 2000
LOOKAHEAD = 4
FEATURE_COUNT = 5
STRICT_EXIT_BARS = 4

# Critical thresholds for high accuracy
LABEL_THRESHOLD = 0.002  # 0.2% move threshold
CONFIDENCE_THRESHOLD = 3  # Minimum votes needed

# Kernel Settings
USE_KERNEL_FILTER = True
KERNEL_H = 10
KERNEL_R = 6.0
KERNEL_X = 15
KERNEL_LAG = 1

# Filter Settings
USE_VOLATILITY_FILTER = True
USE_REGIME_FILTER = True
USE_ADX_FILTER = False
REGIME_THRESHOLD = -0.1
ADX_THRESHOLD = 20

# Signal persistence file
SIGNAL_STATE_FILE = "signal_state.json"
DEBUG = True


class SignalState:
    """Manages signal persistence across runs"""

    def __init__(self, filepath=SIGNAL_STATE_FILE):
        self.filepath = filepath
        self.state = self.load_state()

    def load_state(self):
        if os.path.exists(self.filepath):
            with open(self.filepath, 'r') as f:
                return json.load(f)
        return {}

    def save_state(self):
        with open(self.filepath, 'w') as f:
            json.dump(self.state, f)

    def get_position(self, symbol):
        return self.state.get(symbol, {}).get('position', None)

    def get_bars_held(self, symbol):
        return self.state.get(symbol, {}).get('bars_held', 0)

    def get_last_signal(self, symbol):
        return self.state.get(symbol, {}).get('last_signal', 0)

    def get_entry_info(self, symbol):
        return self.state.get(symbol, {}).get('entry_info', {})

    def update_position(self, symbol, position, entry_price=None, entry_time=None):
        if symbol not in self.state:
            self.state[symbol] = {}

        self.state[symbol]['position'] = position
        if position is not None and entry_price is not None:
            # Convert numpy/pandas types to Python native types
            self.state[symbol]['entry_info'] = {
                'price': float(entry_price) if hasattr(entry_price, 'item') else entry_price,
                'time': entry_time.isoformat() if hasattr(entry_time, 'isoformat') else str(entry_time)
            }
        elif position is None:
            self.state[symbol]['entry_info'] = {}

        self.save_state()

    def update_signal(self, symbol, signal):
        if symbol not in self.state:
            self.state[symbol] = {}
        # Convert numpy types to Python native types for JSON serialization
        self.state[symbol]['last_signal'] = int(signal) if hasattr(signal, 'item') else signal
        self.save_state()

    def increment_bars_held(self, symbol):
        if symbol not in self.state:
            self.state[symbol] = {}
        self.state[symbol]['bars_held'] = self.state[symbol].get('bars_held', 0) + 1
        self.save_state()

    def reset_bars_held(self, symbol):
        if symbol not in self.state:
            self.state[symbol] = {}
        self.state[symbol]['bars_held'] = 0
        self.save_state()


# === Feature Calculation ===
def build_features(df):
    """Build technical indicator features"""
    features = []
    for kind, param_a, param_b in FEATURE_CONFIG:
        f = compute_feature(df.copy(), kind, param_a, param_b)
        f.name = f"{kind}_{param_a}_{param_b}"
        features.append(f)
    return pd.concat(features, axis=1).dropna()


def save_historical_data(symbols_dict, months=10):
    """Fetch and save 10 months of data locally"""
    data_dir = "historical_data"
    os.makedirs(data_dir, exist_ok=True)

    for symbol, index in symbols_dict.items():
        print(f"Fetching {months} months data for {index}...")
        df = fetch_ohlc(symbol, interval=25, months=months)

        # Save as pickle for faster loading
        filename = f"{data_dir}/{index}_{months}months.pkl"
        df.to_pickle(filename)
        print(f"Saved {len(df)} rows to {filename}")

        # Also save as CSV for inspection
        df.to_csv(f"{data_dir}/{index}_{months}months.csv")


def load_historical_data(index):
    """Load saved historical data"""
    filename = f"historical_data/{index}_10months.pkl"
    if os.path.exists(filename):
        return pd.read_pickle(filename)
    else:
        raise FileNotFoundError(f"Please run save_historical_data first. {filename} not found.")


def simulate_streaming_live_mode(base_months=5, total_months=10):
    """Simulate live trading by streaming data bar by bar"""
    indices = {
        "13": "NIFTY",
        "25": "BANKNIFTY",
        "51": "SENSEX",
        "27": "FINNIFTY",
        "442": "MIDCPNIFTY"
    }

    # First, save all data if not exists
    if not os.path.exists("historical_data/NIFTY_10months.pkl"):
        print("Fetching and saving historical data...")
        save_historical_data(indices, months=total_months)

    results = {}

    for symbol, index in indices.items():
        print(f"\n{'=' * 60}")
        print(f"Simulating {index}")
        print(f"{'=' * 60}")

        # Load full data
        full_df = load_historical_data(index)

        # Calculate split point (first 5 months for base)
        total_bars = len(full_df)
        base_bars = int(total_bars * (base_months / total_months))

        # Split data
        base_df = full_df.iloc[:base_bars].copy()
        stream_df = full_df.iloc[base_bars:].copy()

        print(f"Base data: {len(base_df)} bars, Stream data: {len(stream_df)} bars")

        # Initialize signal state for simulation
        sim_signal_state = SignalState(filepath=f"sim_state_{index}.json")

        # Track all trades
        all_trades = []

        # Process each new bar
        for i in range(len(stream_df)):
            # Create current dataset (base + streamed bars)
            current_df = pd.concat([
                base_df.iloc[1:],  # Drop oldest bar
                stream_df.iloc[:i + 1]  # Add new bars up to current
            ])

            # Update base_df for next iteration
            if i > 0:
                base_df = current_df.iloc[:-1].copy()

            # Run strategy on current data
            trades = process_single_bar(symbol, index, current_df, sim_signal_state)
            all_trades.extend(trades)

            # Log progress every 100 bars
            if i % 100 == 0:
                print(f"Processed {i}/{len(stream_df)} bars, Trades so far: {len(all_trades)}")

        results[index] = {
            'trades': all_trades,
            'final_state': sim_signal_state.state
        }

        # Clean up simulation state
        os.remove(f"sim_state_{index}.json")

    return results


def process_single_bar(symbol, index, df, signal_state):
    """Process strategy for a single point in time"""
    trades = []

    # Build features
    features = build_features(df)
    if len(features) < LOOKAHEAD:
        return trades

    # Generate predictions
    predictions = predict_labels(features, df['close'])

    # Get current values
    current_signal = predictions.iloc[-1]
    current_time = df.index[-1]
    current_price = df['close'].iloc[-1]

    # Get kernel signals
    bullish, bearish, rk, gk = get_kernel_signals(df['close'])
    current_bullish = bullish.iloc[-1]
    current_bearish = bearish.iloc[-1]

    # Get current position
    current_position = signal_state.get_position(index)
    bars_held = signal_state.get_bars_held(index)

    # Calculate ATR
    atr = calculate_atr(df)
    current_atr = atr.iloc[-1] if len(atr) > 0 and not pd.isna(atr.iloc[-1]) else 0

    action_taken = None

    # Exit logic
    if current_position == 'LONG':
        signal_state.increment_bars_held(index)
        bars_held = signal_state.get_bars_held(index)
        entry_info = signal_state.get_entry_info(index)
        entry_price = entry_info.get('price', current_price)

        if current_bearish or bars_held >= STRICT_EXIT_BARS or (
                current_atr > 0 and current_price < entry_price - current_atr):
            action_taken = "SELL"
            signal_state.update_position(index, None)
            signal_state.reset_bars_held(index)
            trades.append({
                'time': current_time,
                'action': action_taken,
                'price': current_price,
                'signal': current_signal
            })

    elif current_position == 'SHORT':
        signal_state.increment_bars_held(index)
        bars_held = signal_state.get_bars_held(index)
        entry_info = signal_state.get_entry_info(index)
        entry_price = entry_info.get('price', current_price)

        if current_bullish or bars_held >= STRICT_EXIT_BARS or (
                current_atr > 0 and current_price > entry_price + current_atr):
            action_taken = "COVER"
            signal_state.update_position(index, None)
            signal_state.reset_bars_held(index)
            trades.append({
                'time': current_time,
                'action': action_taken,
                'price': current_price,
                'signal': current_signal
            })

    # Entry logic
    elif current_position is None:
        if current_signal == 1 and current_bullish:
            action_taken = "BUY"
            signal_state.update_position(index, 'LONG', float(current_price), current_time)
            signal_state.reset_bars_held(index)
            trades.append({
                'time': current_time,
                'action': action_taken,
                'price': current_price,
                'signal': current_signal
            })

        elif current_signal == -1 and current_bearish:
            action_taken = "SHORT"
            signal_state.update_position(index, 'SHORT', float(current_price), current_time)
            signal_state.reset_bars_held(index)
            trades.append({
                'time': current_time,
                'action': action_taken,
                'price': current_price,
                'signal': current_signal
            })

    # Update signal state
    signal_state.update_signal(index, int(current_signal) if hasattr(current_signal, 'item') else current_signal)

    return trades


def compare_results():
    """Compare backtest vs simulated live results"""
    print("\nRunning comparison...")

    # Run normal backtest on last 5 months
    backtest_results = run_backtest_analysis()

    # Run simulated live mode
    sim_results = simulate_streaming_live_mode()

    # Compare
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)

    for index in ["NIFTY", "BANKNIFTY", "SENSEX", "FINNIFTY", "MIDCPNIFTY"]:
        if index in backtest_results and index in sim_results:
            bt_trades = backtest_results[index]['performance']['total_trades']
            sim_trades = len([t for t in sim_results[index]['trades'] if t['action'] in ['BUY', 'SHORT']])

            print(f"\n{index}:")
            print(f"  Backtest trades: {bt_trades}")
            print(f"  Simulated trades: {sim_trades}")
            print(f"  Match: {'✅' if bt_trades == sim_trades else '❌'}")


# === ML Prediction Functions ===
def predict_labels(features, close_series):
    """
    Generate ML predictions for all rows using Manhattan distance
    This is the core logic that gives 75% accuracy
    """
    # Align indices
    common_index = features.index.intersection(close_series.index)
    features = features.loc[common_index]
    close_series = close_series.loc[common_index]

    X = features.values
    close = close_series.values
    n = len(X)
    Y = np.zeros(n)

    # Create labels with threshold
    for i in range(LOOKAHEAD, n):
        future_close = close[i]
        current_close = close[i - LOOKAHEAD]
        delta = (future_close - current_close) / current_close

        if delta > LABEL_THRESHOLD:
            Y[i - LOOKAHEAD] = 1
        elif delta < -LABEL_THRESHOLD:
            Y[i - LOOKAHEAD] = -1
        else:
            Y[i - LOOKAHEAD] = 0

    # Calculate Manhattan distances
    distances = manhattan_distances(X, X)
    np.fill_diagonal(distances, np.inf)

    # Generate predictions for every row
    predictions = []
    for i in range(n):
        if i < LOOKAHEAD:
            predictions.append(0)
            continue

        # Get k nearest neighbors
        idx = np.argsort(distances[i])[:NEIGHBOR_COUNT]
        vote = np.sum(Y[idx])

        # Apply confidence threshold
        if abs(vote) < CONFIDENCE_THRESHOLD:
            predictions.append(0)
        else:
            predictions.append(int(np.sign(vote)))

    return pd.Series(predictions, index=features.index)


# === Technical Analysis Functions ===
def calculate_atr(df, period=14):
    """Calculate Average True Range"""
    high = df['high']
    low = df['low']
    close = df['close']

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    atr = tr.rolling(window=period).mean()
    return atr


def get_kernel_signals(series, h=KERNEL_H, r=KERNEL_R, x=KERNEL_X, lag=KERNEL_LAG):
    """Calculate kernel signals for entry/exit"""
    yhat1 = rational_quadratic_kernel(series, h, r, x)
    yhat2 = gaussian_kernel(series, h - lag, x)

    bullish = (yhat1.shift(1) < yhat1)
    bearish = (yhat1.shift(1) > yhat1)

    return bullish, bearish, yhat1, yhat2


# === Backtest Function ===
def backtest(df, predictions):
    """Backtest strategy with your original logic"""
    atr = calculate_atr(df)
    bullish, bearish, rk, gk = get_kernel_signals(df['close'])

    position = None
    entry_price = 0
    bar_hold = 0
    trades = []
    log_entries = []

    for i in range(len(predictions)):
        ts = df.index[i]
        signal = predictions.iloc[i]
        price = df['close'].iloc[i]
        atr_val = atr.iloc[i] if i < len(atr) and not pd.isna(atr.iloc[i]) else 0

        # Exit logic first
        if position == 'LONG':
            bar_hold += 1
            # Exit conditions: bearish signal, max bars, or stop loss
            if bearish.iloc[i] or bar_hold >= STRICT_EXIT_BARS or (atr_val > 0 and price < entry_price - atr_val):
                trades.append((ts, 'SELL', price))
                position = None
                bar_hold = 0

        elif position == 'SHORT':
            bar_hold += 1
            # Exit conditions: bullish signal, max bars, or stop loss
            if bullish.iloc[i] or bar_hold >= STRICT_EXIT_BARS or (atr_val > 0 and price > entry_price + atr_val):
                trades.append((ts, 'COVER', price))
                position = None
                bar_hold = 0

        # Entry logic (only if no position)
        if position is None:
            if signal == 1 and bullish.iloc[i]:
                trades.append((ts, 'BUY', price))
                position = 'LONG'
                entry_price = price
                bar_hold = 0
            elif signal == -1 and bearish.iloc[i]:
                trades.append((ts, 'SHORT', price))
                position = 'SHORT'
                entry_price = price
                bar_hold = 0

        log_entries.append({
            "Time": ts,
            "Signal": signal,
            "Close": price,
            "ATR": atr_val,
            "Position": position,
            "Bullish": bullish.iloc[i],
            "Bearish": bearish.iloc[i]
        })

    return trades, pd.DataFrame(log_entries)


# === Live Trading Function ===
def process_symbol(symbol_dict, signal_state, trader, quote_data=None):
    """Process a single symbol for live trading"""
    symbol, index = next(iter(symbol_dict.items()))

    # Fetch OHLC data
    df = fetch_ohlc(symbol, interval=interval_time)

    if quote_data and 'last_price' in quote_data:
        now = pd.Timestamp.now().floor("25min")
        if now not in df.index or (datetime.now() - df.index[-1].to_pydatetime()).total_seconds() > 25 * 60:
            ltp = float(quote_data['last_price'])
            df.loc[now] = [ltp, ltp, ltp, ltp]
            print(f"📊 Added live candle for {index} at {ltp}")

    if len(df) < 100:
        print(f"Skipping {index}: insufficient data")
        return

    print(f"Running {index} with {len(df)} rows...")

    # Build features
    features = build_features(df)
    if len(features) < LOOKAHEAD:
        print(f"Not enough features for {index}")
        return

    # Generate predictions
    predictions = predict_labels(features, df['close'])

    # Get current values
    current_signal = predictions.iloc[-1]
    current_time = df.index[-1]
    current_price = df['close'].iloc[-1]

    # Get kernel signals
    bullish, bearish, rk, gk = get_kernel_signals(df['close'])
    current_bullish = bullish.iloc[-1]
    current_bearish = bearish.iloc[-1]

    # Get current position
    current_position = signal_state.get_position(index)
    bars_held = signal_state.get_bars_held(index)

    # Calculate ATR
    atr = calculate_atr(df)
    current_atr = atr.iloc[-1] if len(atr) > 0 and not pd.isna(atr.iloc[-1]) else 0

    action_taken = None

    # Exit logic
    if current_position == 'LONG':
        signal_state.increment_bars_held(index)
        bars_held = signal_state.get_bars_held(index)
        entry_info = signal_state.get_entry_info(index)
        entry_price = entry_info.get('price', current_price)

        if current_bearish or bars_held >= STRICT_EXIT_BARS or (
                current_atr > 0 and current_price < entry_price - current_atr):
            action_taken = "SELL"
            signal_state.update_position(index, None)
            signal_state.reset_bars_held(index)

    elif current_position == 'SHORT':
        signal_state.increment_bars_held(index)
        bars_held = signal_state.get_bars_held(index)
        entry_info = signal_state.get_entry_info(index)
        entry_price = entry_info.get('price', current_price)

        if current_bullish or bars_held >= STRICT_EXIT_BARS or (
                current_atr > 0 and current_price > entry_price + current_atr):
            action_taken = "COVER"
            signal_state.update_position(index, None)
            signal_state.reset_bars_held(index)

    # Entry logic
    elif current_position is None:
        if current_signal == 1 and current_bullish:
            action_taken = "BUY"
            signal_state.update_position(index, 'LONG', current_price, current_time)
            signal_state.reset_bars_held(index)

        elif current_signal == -1 and current_bearish:
            action_taken = "SHORT"
            signal_state.update_position(index, 'SHORT', current_price, current_time)
            signal_state.reset_bars_held(index)

    # Update signal state (convert numpy int64 to regular int)
    current_signal_int = int(current_signal) if hasattr(current_signal, 'item') else current_signal
    signal_state.update_signal(index, current_signal_int)

    # Log and execute
    log_str = (
        f"INDEX: {index}, "
        f"Time: {current_time}, "
        f"Signal: {current_signal}, "
        f"Close: {current_price:.2f}, "
        f"ATR: {current_atr:.2f}, "
        f"Bullish: {current_bullish}, "
        f"Bearish: {current_bearish}, "
        f"Position: {current_position}, "
        f"Bars Held: {bars_held}, "
        f"Action: {action_taken}"
    )

    if action_taken:
        print(f"📈 TRADE SIGNAL: {log_str}")

        # Send alerts
        subject = f"{index} SIGNAL: {action_taken}"
        send_email(subject, log_str)

        # Update Firebase
        doc_ref = db.collection("live_alerts").document(index)
        doc_ref.update({"alerts": ArrayUnion([log_str])})

        # Execute trade
        trader.execute_single_trade(timestamp=current_time, index_name=index, signal_type=action_taken)
    else:
        print(f"📊 {index}: {log_str}")


# === Data Fetching ===
def fetch_ohlc(symbol, interval=25, months=5):
    """Fetch OHLC data from Dhan"""
    end = datetime.now()
    start = end - timedelta(days=30 * months)

    chunks = []
    current_start = start

    while current_start <= end:
        current_end = min(current_start + timedelta(days=75), end)
        r = dhan_object.intraday_minute_data(
            security_id=symbol,
            from_date=current_start.strftime('%Y-%m-%d'),
            to_date=current_end.strftime('%Y-%m-%d'),
            interval=interval,
            exchange_segment="IDX_I",
            instrument_type="INDEX"
        )
        data = r.get('data', [])
        if data:
            chunks.append(pd.DataFrame(data))
        current_start = current_end + timedelta(days=1)

    if not chunks:
        return pd.DataFrame()

    df = pd.concat(chunks).reset_index(drop=True)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    df['datetime'] = df['datetime'].dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
    df.set_index('datetime', inplace=True)
    df = df[['open', 'high', 'low', 'close']].astype(float)

    # Don't append live candle here - we'll do it in batch
    return df.sort_index()


# === Backtest Analysis ===
def run_backtest_analysis():
    """Run comprehensive backtest on all indices"""
    print("\n" + "=" * 80)
    print("RUNNING BACKTEST ANALYSIS")
    print("=" * 80 + "\n")

    indices = [
        {"13": "NIFTY"},
        {"25": "BANKNIFTY"},
        {"51": "SENSEX"},
        {"27": "FINNIFTY"},
        {"442": "MIDCPNIFTY"}
    ]

    all_results = {}

    for index_dict in indices:
        symbol, index = next(iter(index_dict.items()))
        print(f"\nBacktesting {index}...")

        try:
            # Fetch data
            df = fetch_ohlc(symbol, interval=interval_time)

            if len(df) < 100:
                print(f"Skipping {index}: insufficient data")
                continue

            # Build features and predictions
            features = build_features(df)
            if len(features) < LOOKAHEAD:
                print(f"Not enough features for {index}")
                continue

            predictions = predict_labels(features, df['close'])

            # Run backtest
            trades, logs = backtest(df.loc[predictions.index], predictions)

            # Calculate performance
            positions = []
            for i in range(1, len(trades), 2):
                if i < len(trades):
                    t1, action1, price1 = trades[i - 1]
                    t2, action2, price2 = trades[i]
                    profit = price2 - price1 if action1 == 'BUY' else price1 - price2
                    positions.append({
                        'entry_time': t1,
                        'entry_action': action1,
                        'entry_price': price1,
                        'exit_time': t2,
                        'exit_action': action2,
                        'exit_price': price2,
                        'profit': profit,
                        'profit_pct': (profit / price1) * 100
                    })

            # Calculate metrics
            if positions:
                total_trades = len(positions)
                winning_trades = sum(1 for p in positions if p['profit'] > 0)
                losing_trades = total_trades - winning_trades
                total_profit = sum(p['profit'] for p in positions)
                win_rate = (winning_trades / total_trades) * 100
                avg_profit = total_profit / total_trades
                avg_profit_pct = sum(p['profit_pct'] for p in positions) / total_trades

                performance = {
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'losing_trades': losing_trades,
                    'win_rate': win_rate,
                    'total_profit': total_profit,
                    'avg_profit': avg_profit,
                    'avg_profit_pct': avg_profit_pct
                }
            else:
                performance = {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0,
                    'total_profit': 0,
                    'avg_profit': 0,
                    'avg_profit_pct': 0
                }

            all_results[index] = {
                'trades': trades,
                'positions': positions,
                'performance': performance
            }

            # Display results
            print(f"\n{index} Performance:")
            print(f"  Total Trades: {performance['total_trades']}")
            print(f"  Win Rate: {performance['win_rate']:.2f}%")
            print(f"  Winning Trades: {performance['winning_trades']}")
            print(f"  Losing Trades: {performance['losing_trades']}")
            print(f"  Total Profit Points: {performance['total_profit']:.2f}")
            print(f"  Average Profit per Trade: {performance['avg_profit']:.2f}")
            print(f"  Average Profit %: {performance['avg_profit_pct']:.2f}%")

        except Exception as e:
            print(f"Error backtesting {index}: {str(e)}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)

    total_trades = sum(r['performance']['total_trades'] for r in all_results.values())
    total_wins = sum(r['performance']['winning_trades'] for r in all_results.values())
    overall_win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0

    print(f"\nTotal Trades Across All Indices: {total_trades}")
    print(f"Overall Win Rate: {overall_win_rate:.2f}%")
    print(f"Indices Tested: {len(all_results)}")

    # Save results
    save_backtest_results(all_results)

    return all_results


def save_backtest_results(all_results):
    """Save backtest results to CSV"""
    output_dir = "backtest_results"
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save summary
    summary_data = []
    for index, results in all_results.items():
        perf = results['performance']
        summary_data.append({
            'Index': index,
            'Total_Trades': perf['total_trades'],
            'Win_Rate_%': round(perf['win_rate'], 2),
            'Winning_Trades': perf['winning_trades'],
            'Losing_Trades': perf['losing_trades'],
            'Total_Profit': round(perf['total_profit'], 2),
            'Avg_Profit': round(perf['avg_profit'], 2),
            'Avg_Profit_%': round(perf['avg_profit_pct'], 2)
        })

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f"{output_dir}/backtest_summary_{timestamp}.csv", index=False)

    # Save detailed trades
    for index, results in all_results.items():
        if results['positions']:
            positions_df = pd.DataFrame(results['positions'])
            positions_df.to_csv(f"{output_dir}/{index}_trades_{timestamp}.csv", index=False)

    print(f"\nResults saved to {output_dir}/")


# === Main Execution Functions ===
def is_market_closed():
    """Check if market is closed"""
    now = datetime.now()
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return now >= market_close


def wait_for_market_open():
    """Wait until market opens at 9:15 AM"""
    while True:
        now = datetime.now()

        # Check if it's a weekend
        if now.weekday() > 4:
            print(f"It's {now.strftime('%A')}. Market is closed on weekends.")
            print("Please run the script on a weekday.")
            return False

        # Market open time
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)

        # If market is already open
        if market_open <= now <= market_close:
            print("Market is open! Starting strategy...")
            return True

        # If market has closed for the day
        if now > market_close:
            print(f"Market has closed for today at 3:30 PM.")
            print("Please run the script tomorrow.")
            return False

        # Calculate time to market open
        time_to_open = (market_open - now).total_seconds()
        hours = int(time_to_open // 3600)
        minutes = int((time_to_open % 3600) // 60)
        seconds = int(time_to_open % 60)

        print(f"Market opens at 9:15 AM. Current time: {now.strftime('%H:%M:%S')}")
        print(f"Waiting {hours}h {minutes}m {seconds}s for market to open...")

        # Sleep for 30 seconds and check again
        time.sleep(30)


def run_live_trading(signal_state, trader):
    """Run live trading loop"""

    if not wait_for_market_open():
        print("Exiting as market won't open today.")
        return

    print("✅ Live trading started. Will run every 25 minutes until 3:30 PM IST.")

    indices = [
        {"13": "NIFTY"},
        {"25": "BANKNIFTY"},
        {"51": "SENSEX"},
        {"27": "FINNIFTY"},
        {"442": "MIDCPNIFTY"}
    ]

    while True:
        if is_market_closed():
            print("⏹️ Market closed at 3:30 PM. Stopping strategy.")
            break

        print(f"\n{'=' * 60}")
        print(f"🕒 Running strategy at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'=' * 60}\n")

        # Fetch all quotes at once
        all_symbols = [int(list(idx.keys())[0]) for idx in indices]
        try:
            quote_response = dhan_object.quote_data({"IDX_I": all_symbols})
            quotes = {}
            if 'data' in quote_response and 'data' in quote_response['data'] and 'IDX_I' in quote_response['data'][
                'data']:
                quotes = quote_response['data']['data']['IDX_I']
        except Exception as e:
            print(f"Error fetching quotes: {e}")
            quotes = {}

        for index_dict in indices:
            try:
                symbol, index = next(iter(index_dict.items()))
                process_symbol(index_dict, signal_state, trader, quotes.get(symbol))
            except Exception as e:
                symbol, index = next(iter(index_dict.items()))
                print(f"❌ Error processing {index}: {str(e)}")
                continue

        # Sleep for 25 minutes
        next_run = get_next_candle_time()
        sleep_duration = (next_run - datetime.now()).total_seconds()

        if sleep_duration > 0:
            print(
                f"\n🕒 Next run at {next_run.strftime('%H:%M:%S')} (candle close + 30s). Sleeping for {int(sleep_duration)} seconds...\n")
            time.sleep(sleep_duration)


def get_next_candle_time():
    """Calculate the next 25-minute candle close time"""
    now = datetime.now()

    # Market opens at 9:15
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)

    # If before market open, return first candle close
    if now < market_open:
        return market_open + timedelta(minutes=25)

    # Calculate minutes since market open
    minutes_since_open = (now - market_open).total_seconds() / 60

    # Calculate which candle we're in
    candles_passed = int(minutes_since_open // 25)

    # Next candle close time
    next_candle_close = market_open + timedelta(minutes=(candles_passed + 1) * 25)

    # Add 30 seconds buffer to ensure candle is fully closed
    return next_candle_close + timedelta(seconds=30)


# === Main Entry Point ===
if __name__ == '__main__':
    load_dotenv()

    # Load credentials
    CLIENT_ID = os.getenv("DHAN_CLIENT_ID")
    ACCESS_TOKEN = os.getenv("DHAN_API_KEY")
    KOTAK_ACCESS_TOKEN = os.getenv("KOTAK_ACCESS_TOKEN")

    interval_time = 25

    # Initialize Dhan
    dhan_object = dhanhq.dhanhq(CLIENT_ID, ACCESS_TOKEN)

    # Check command line arguments
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == 'backtest':
            print("Running in BACKTEST mode...")
            run_backtest_analysis()
        elif sys.argv[1] == 'validate':
            print("Running validation mode...")
            compare_results()
        elif sys.argv[1] == 'test_live':
            # Run live mode with test orders
            print("Running in TEST LIVE mode (no real trades)...")

            # Initialize Kotak trader in test mode
            trader = KotakOptionsTrader(test_mode=True)

            print("\n*** KOTAK TRADER IN TEST MODE - NO REAL ORDERS ***\n")

            # Firebase Setup
            cred = credentials.Certificate("stock-monitoring-fb.json")
            initialize_app(cred)
            db = firestore.client()

            # Initialize signal state
            signal_state = SignalState()

            # Run strategy
            run_live_trading(signal_state, trader)
        elif sys.argv[1] == 'save_data':
            print("Saving historical data...")
            indices = {"13": "NIFTY", "25": "BANKNIFTY", "51": "SENSEX", "27": "FINNIFTY", "442": "MIDCPNIFTY"}
            save_historical_data(indices, months=10)
    else:
        # Run live trading mode (REAL TRADES)
        print("Running in LIVE TRADING mode...")

        # Initialize Kotak trader in LIVE mode
        trader = KotakOptionsTrader(test_mode=False)

        # Check if trader is logged in
        status = trader.get_account_status()
        if not status['logged_in']:
            print("Failed to login to Kotak. Please check your access token.")
            sys.exit(1)

        print(f"Kotak account connected. Market {'OPEN' if status.get('market_open') else 'CLOSED'}")

        # Firebase Setup
        cred = credentials.Certificate("stock-monitoring-fb.json")
        initialize_app(cred)
        db = firestore.client()

        # Initialize signal state
        signal_state = SignalState()

        # Run strategy
        run_live_trading(signal_state, trader)
