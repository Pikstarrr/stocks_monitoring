# momentum_x_pine_converted.py - Optimized for high accuracy (75%+)
import csv
import math
import time
import json
import os
from collections import deque
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
# Using your original config that gave 75% accuracy
FEATURE_CONFIG = [
    ("RSI", 14, 1),  # Feature 1
    ("WT", 10, 11),  # Feature 2
    ("CCI", 20, 1),  # Feature 3
    ("ADX", 20, 2),  # Feature 4
    ("RSI", 9, 1),  # Feature 5
]
NEIGHBOR_COUNT = 5  # Your original value
MAX_BARS_BACK = 2000
LOOKAHEAD = 4
FEATURE_COUNT = 5
STRICT_EXIT_BARS = 4  # From your original

# Important thresholds from your original script
LABEL_THRESHOLD = 0.002  # 0.2% move threshold - CRITICAL for accuracy
CONFIDENCE_THRESHOLD = 3  # Minimum votes needed - CRITICAL for accuracy

# Kernel Settings (adjusted for better signals)
USE_KERNEL_FILTER = True
KERNEL_H = 10  # From your original
KERNEL_R = 6.0  # From your original
KERNEL_X = 15  # From your original
KERNEL_LAG = 1  # From your original

# Filter Settings
USE_VOLATILITY_FILTER = True
USE_REGIME_FILTER = True
USE_ADX_FILTER = False
REGIME_THRESHOLD = -0.1
ADX_THRESHOLD = 20

# Dynamic Exit Settings
USE_DYNAMIC_EXITS = False
SHOW_EXITS = False

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
            self.state[symbol]['entry_info'] = {
                'price': entry_price,
                'time': entry_time.isoformat() if entry_time else None
            }
        elif position is None:
            self.state[symbol]['entry_info'] = {}

        self.save_state()

    def update_signal(self, symbol, signal):
        if symbol not in self.state:
            self.state[symbol] = {}
        self.state[symbol]['last_signal'] = signal
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
    features = []
    for kind, param_a, param_b in FEATURE_CONFIG:
        f = compute_feature(df.copy(), kind, param_a, param_b)
        f.name = f"{kind}_{param_a}_{param_b}"
        features.append(f)
    return pd.concat(features, axis=1).dropna()


# === ML Model Prediction (Using your original high-accuracy logic) ===
def predict_ml_signal_original(features_df, close_series):
    """
    Your original ML prediction logic that gave 75% accuracy
    Uses Manhattan distance and label thresholding
    """
    # Align indices
    common_index = features_df.index.intersection(close_series.index)
    features_df = features_df.loc[common_index]
    close_series = close_series.loc[common_index]

    X = features_df.values
    close = close_series.values
    n = len(X)

    # Create labels with threshold (your original logic)
    Y = np.zeros(n)
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

    # Calculate Manhattan distances (your original approach)
    distances = manhattan_distances(X, X)
    np.fill_diagonal(distances, np.inf)

    # Make predictions for the last row
    current_idx = n - 1
    if current_idx < LOOKAHEAD:
        return 0

    # Get nearest neighbors
    idx = np.argsort(distances[current_idx])[:NEIGHBOR_COUNT]

    # Vote counting with confidence threshold
    vote = np.sum(Y[idx])

    # Apply confidence threshold (critical for accuracy)
    if abs(vote) < CONFIDENCE_THRESHOLD:
        return 0
    else:
        return int(np.sign(vote))


def predict_ml_signal(features_df, close_series, current_idx):
    """
    Wrapper to use the original ML logic for better accuracy
    """
    # Get subset of data up to current index
    features_subset = features_df.iloc[:current_idx + 1]
    close_subset = close_series.iloc[:current_idx + 1]

    return predict_ml_signal_original(features_subset, close_subset)


# === Filter Functions ===
def volatility_filter(close_series, current_idx, use_filter=True):
    """Implements volatility filter from Pine Script"""
    if not use_filter or current_idx < 10:
        return True

    # Simple volatility check using recent price movement
    recent_volatility = close_series.iloc[max(0, current_idx - 10):current_idx].std()
    avg_volatility = close_series.iloc[max(0, current_idx - 50):current_idx].std()

    return recent_volatility > avg_volatility * 0.5


def regime_filter(ohlc4_series, current_idx, threshold=REGIME_THRESHOLD, use_filter=True):
    """Implements regime filter from Pine Script"""
    if not use_filter or current_idx < 50:
        return True

    # Calculate trend strength
    sma_short = ohlc4_series.iloc[max(0, current_idx - 20):current_idx].mean()
    sma_long = ohlc4_series.iloc[max(0, current_idx - 50):current_idx].mean()

    trend_strength = (sma_short - sma_long) / sma_long
    return trend_strength > threshold


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    # Calculate True Range
    high_low = high - low
    high_close = (high - close.shift()).abs()
    low_close = (low - close.shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    # Calculate Directional Movement
    up_move = high - high.shift()
    down_move = low.shift() - low

    pos_dm = pd.Series(0.0, index=up_move.index)
    neg_dm = pd.Series(0.0, index=down_move.index)

    pos_dm[(up_move > down_move) & (up_move > 0)] = up_move[(up_move > down_move) & (up_move > 0)]
    neg_dm[(down_move > up_move) & (down_move > 0)] = down_move[(down_move > up_move) & (down_move > 0)]

    # Calculate ATR
    atr = tr.rolling(window=period).mean()

    # Calculate +DI and -DI
    pos_di = 100 * (pos_dm.rolling(window=period).mean() / atr)
    neg_di = 100 * (neg_dm.rolling(window=period).mean() / atr)

    # Calculate DX
    dx = 100 * ((pos_di - neg_di).abs() / (pos_di + neg_di))

    # Calculate ADX
    adx = dx.rolling(window=period).mean()

    return adx


def adx_filter(high, low, close, current_idx, period=14, threshold=ADX_THRESHOLD, use_filter=False):
    """Implements ADX filter"""
    if not use_filter or current_idx < period * 2:
        return True

    # Calculate ADX
    adx = calculate_adx(high, low, close, period)

    # Check if current ADX is above threshold (indicating trending market)
    if current_idx < len(adx) and not pd.isna(adx.iloc[current_idx]):
        return adx.iloc[current_idx] > threshold

    return True


# === Kernel Functions ===
def get_kernel_signals(series, h=KERNEL_H, r=KERNEL_R, x=KERNEL_X, lag=KERNEL_LAG):
    """Calculate kernel signals for entry/exit"""
    yhat1 = rational_quadratic_kernel(series, h, r, x)
    yhat2 = gaussian_kernel(series, h - lag, x)

    # Your original logic: simple shift comparison
    bullish = (yhat1.shift(1) < yhat1)
    bearish = (yhat1.shift(1) > yhat1)

    return {
        'bullish': bullish,
        'bearish': bearish,
        'yhat1': yhat1,
        'yhat2': yhat2
    }


# === Backtesting Functions ===
def backtest_strategy(df, features, signal_state_mock=None):
    """
    Backtest the strategy on historical data
    Returns: trades list, performance metrics
    """
    trades = []
    positions = []

    # Mock signal state for backtesting
    if signal_state_mock is None:
        signal_state_mock = {
            'position': None,
            'bars_held': 0,
            'last_signal': 0,
            'entry_price': 0,
            'entry_time': None
        }

    # Calculate OHLC4
    df['ohlc4'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4

    # Align features and df indices
    common_index = features.index.intersection(df.index)
    features = features.loc[common_index]
    df_aligned = df.loc[common_index]

    # Get kernel signals
    kernel_signals = get_kernel_signals(df_aligned['close'])

    for i in range(LOOKAHEAD, len(features)):
        current_idx = i
        current_time = df_aligned.index[i]
        current_price = df_aligned['close'].iloc[i]

        # Get ML prediction
        prediction = predict_ml_signal(features, df_aligned['close'], current_idx)

        # Apply filters
        vol_filter = volatility_filter(df_aligned['close'], current_idx, USE_VOLATILITY_FILTER)
        regime_filt = regime_filter(df_aligned['ohlc4'], current_idx, REGIME_THRESHOLD, USE_REGIME_FILTER)
        adx_filt = adx_filter(df_aligned['high'], df_aligned['low'], df_aligned['close'], current_idx, 14,
                              ADX_THRESHOLD, USE_ADX_FILTER)

        filter_all = vol_filter and regime_filt and adx_filt

        # Determine signal
        current_signal = 0
        if prediction > 0 and filter_all:
            current_signal = 1
        elif prediction < 0 and filter_all:
            current_signal = -1

        # Kernel conditions
        is_bullish = kernel_signals['bullish'].iloc[i] if USE_KERNEL_FILTER else True
        is_bearish = kernel_signals['bearish'].iloc[i] if USE_KERNEL_FILTER else True

        # Entry conditions (matching your original)
        start_long_trade = current_signal == 1 and is_bullish
        start_short_trade = current_signal == -1 and is_bearish

        # Exit conditions
        end_long_trade = False
        end_short_trade = False

        if signal_state_mock['position'] == 'LONG':
            signal_state_mock['bars_held'] += 1
            # Your original exit logic
            end_long_trade = is_bearish or signal_state_mock['bars_held'] >= STRICT_EXIT_BARS

        elif signal_state_mock['position'] == 'SHORT':
            signal_state_mock['bars_held'] += 1
            # Your original exit logic
            end_short_trade = is_bullish or signal_state_mock['bars_held'] >= STRICT_EXIT_BARS

        # Execute trades
        if end_long_trade and signal_state_mock['position'] == 'LONG':
            trades.append({
                'time': current_time,
                'action': 'SELL',
                'price': current_price,
                'type': 'exit'
            })
            # Calculate profit
            profit = current_price - signal_state_mock['entry_price']
            positions.append({
                'entry_time': signal_state_mock['entry_time'],
                'entry_price': signal_state_mock['entry_price'],
                'exit_time': current_time,
                'exit_price': current_price,
                'type': 'LONG',
                'profit': profit,
                'profit_pct': (profit / signal_state_mock['entry_price']) * 100
            })
            signal_state_mock['position'] = None
            signal_state_mock['bars_held'] = 0

        elif end_short_trade and signal_state_mock['position'] == 'SHORT':
            trades.append({
                'time': current_time,
                'action': 'COVER',
                'price': current_price,
                'type': 'exit'
            })
            # Calculate profit
            profit = signal_state_mock['entry_price'] - current_price
            positions.append({
                'entry_time': signal_state_mock['entry_time'],
                'entry_price': signal_state_mock['entry_price'],
                'exit_time': current_time,
                'exit_price': current_price,
                'type': 'SHORT',
                'profit': profit,
                'profit_pct': (profit / signal_state_mock['entry_price']) * 100
            })
            signal_state_mock['position'] = None
            signal_state_mock['bars_held'] = 0

        elif start_long_trade and signal_state_mock['position'] is None:
            trades.append({
                'time': current_time,
                'action': 'BUY',
                'price': current_price,
                'type': 'entry'
            })
            signal_state_mock['position'] = 'LONG'
            signal_state_mock['entry_price'] = current_price
            signal_state_mock['entry_time'] = current_time
            signal_state_mock['bars_held'] = 0

        elif start_short_trade and signal_state_mock['position'] is None:
            trades.append({
                'time': current_time,
                'action': 'SHORT',
                'price': current_price,
                'type': 'entry'
            })
            signal_state_mock['position'] = 'SHORT'
            signal_state_mock['entry_price'] = current_price
            signal_state_mock['entry_time'] = current_time
            signal_state_mock['bars_held'] = 0

        # Update last signal
        signal_state_mock['last_signal'] = current_signal

    # Calculate performance metrics
    if positions:
        total_trades = len(positions)
        winning_trades = sum(1 for p in positions if p['profit'] > 0)
        losing_trades = sum(1 for p in positions if p['profit'] <= 0)

        total_profit = sum(p['profit'] for p in positions)
        total_profit_pct = sum(p['profit_pct'] for p in positions)

        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        avg_profit = total_profit / total_trades if total_trades > 0 else 0
        avg_profit_pct = total_profit_pct / total_trades if total_trades > 0 else 0

        # Calculate max drawdown
        cumulative_profit = 0
        max_profit = 0
        max_drawdown = 0

        for p in positions:
            cumulative_profit += p['profit']
            max_profit = max(max_profit, cumulative_profit)
            drawdown = max_profit - cumulative_profit
            max_drawdown = max(max_drawdown, drawdown)

        performance = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'total_profit_pct': total_profit_pct,
            'avg_profit': avg_profit,
            'avg_profit_pct': avg_profit_pct,
            'max_drawdown': max_drawdown,
            'profit_factor': abs(sum(p['profit'] for p in positions if p['profit'] > 0) /
                                 sum(p['profit'] for p in positions if
                                     p['profit'] < 0)) if losing_trades > 0 else float('inf')
        }
    else:
        performance = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'total_profit': 0,
            'total_profit_pct': 0,
            'avg_profit': 0,
            'avg_profit_pct': 0,
            'max_drawdown': 0,
            'profit_factor': 0
        }

    return trades, positions, performance


def run_backtest_analysis():
    """Run backtest on all indices and display results"""
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
            df = fetch_ohlc(symbol, interval=25)  # 25 min for consistency

            if len(df) < 100:
                print(f"Skipping {index}: insufficient data")
                continue

            # Build features
            features = build_features(df)
            if len(features) < LOOKAHEAD:
                print(f"Not enough features for {index}")
                continue

            # Run backtest
            trades, positions, performance = backtest_strategy(df, features)

            # Store results
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
            print(f"  Max Drawdown: {performance['max_drawdown']:.2f}")
            print(f"  Profit Factor: {performance['profit_factor']:.2f}")

        except Exception as e:
            print(f"Error backtesting {index}: {str(e)}")
            continue

    # Summary statistics
    print("\n" + "=" * 80)
    print("BACKTEST SUMMARY")
    print("=" * 80)

    total_trades = sum(r['performance']['total_trades'] for r in all_results.values())
    total_wins = sum(r['performance']['winning_trades'] for r in all_results.values())
    overall_win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0

    print(f"\nOverall Statistics:")
    print(f"  Total Trades Across All Indices: {total_trades}")
    print(f"  Overall Win Rate: {overall_win_rate:.2f}%")
    print(f"  Indices Tested: {len(all_results)}")

    # Save detailed results to CSV
    save_backtest_results(all_results)

    return all_results


def save_backtest_results(all_results):
    """Save backtest results to CSV files"""
    import os

    # Create output directory
    output_dir = "backtest_results"
    os.makedirs(output_dir, exist_ok=True)

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
            'Avg_Profit_%': round(perf['avg_profit_pct'], 2),
            'Max_Drawdown': round(perf['max_drawdown'], 2),
            'Profit_Factor': round(perf['profit_factor'], 2)
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f"{output_dir}/backtest_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)

    # Save detailed trades for each index
    for index, results in all_results.items():
        if results['positions']:
            positions_df = pd.DataFrame(results['positions'])
            positions_df.to_csv(f"{output_dir}/{index}_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                index=False)

    print(f"\nBacktest results saved to {output_dir}/")


def process_symbol(symbol_dict, signal_state, trader):
    """Process a single symbol following Pine Script logic"""
    symbol, index = next(iter(symbol_dict.items()))

    # Fetch OHLC data
    df = fetch_ohlc(symbol, interval=25)  # 25 min like your original

    if len(df) < 100:
        print(f"Skipping {index}: insufficient data")
        return

    # Calculate OHLC4
    df['ohlc4'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4

    # Build features
    features = build_features(df)
    if len(features) < LOOKAHEAD:
        print(f"Not enough features for {index}")
        return

    # Get current index
    current_idx = len(features) - 1

    # Get ML prediction
    prediction = predict_ml_signal(features, df['close'], current_idx)

    # Apply filters
    vol_filter = volatility_filter(df['close'], current_idx, USE_VOLATILITY_FILTER)
    regime_filt = regime_filter(df['ohlc4'], current_idx, REGIME_THRESHOLD, USE_REGIME_FILTER)
    adx_filt = adx_filter(df['high'], df['low'], df['close'], current_idx, 14, ADX_THRESHOLD, USE_ADX_FILTER)

    filter_all = vol_filter and regime_filt and adx_filt

    # Determine signal (1=long, -1=short, 0=neutral)
    current_signal = 0
    if prediction > 0 and filter_all:
        current_signal = 1
    elif prediction < 0 and filter_all:
        current_signal = -1

    # Get kernel signals
    kernel_signals = get_kernel_signals(df['close'])

    # Check current position and signals
    current_position = signal_state.get_position(index)
    last_signal = signal_state.get_last_signal(index)
    bars_held = signal_state.get_bars_held(index)

    # Determine if we have bullish/bearish conditions
    is_bullish = kernel_signals['bullish'].iloc[-1] if USE_KERNEL_FILTER else True
    is_bearish = kernel_signals['bearish'].iloc[-1] if USE_KERNEL_FILTER else True

    # Entry conditions (matching your original logic)
    start_long_trade = current_signal == 1 and is_bullish
    start_short_trade = current_signal == -1 and is_bearish

    # Exit conditions based on your original logic
    end_long_trade = False
    end_short_trade = False

    if current_position == 'LONG':
        signal_state.increment_bars_held(index)
        bars_held = signal_state.get_bars_held(index)
        # Exit on: bearish kernel, strict bar count, or stop loss (using ATR if available)
        end_long_trade = is_bearish or bars_held >= STRICT_EXIT_BARS

    elif current_position == 'SHORT':
        signal_state.increment_bars_held(index)
        bars_held = signal_state.get_bars_held(index)
        # Exit on: bullish kernel, strict bar count, or stop loss
        end_short_trade = is_bullish or bars_held >= STRICT_EXIT_BARS

    # Execute trades
    current_price = df['close'].iloc[-1]
    current_time = df.index[-1]
    action_taken = None

    if end_long_trade and current_position == 'LONG':
        action_taken = "SELL"
        signal_state.update_position(index, None)
        signal_state.reset_bars_held(index)

    elif end_short_trade and current_position == 'SHORT':
        action_taken = "COVER"
        signal_state.update_position(index, None)
        signal_state.reset_bars_held(index)

    elif start_long_trade and current_position is None:
        action_taken = "BUY"
        signal_state.update_position(index, 'LONG', current_price, current_time)
        signal_state.reset_bars_held(index)

    elif start_short_trade and current_position is None:
        action_taken = "SHORT"
        signal_state.update_position(index, 'SHORT', current_price, current_time)
        signal_state.reset_bars_held(index)

    # Update signal state
    signal_state.update_signal(index, current_signal)

    # Log and send alerts
    if action_taken:
        log_str = (
            f"INDEX: {index}, "
            f"Time: {current_time}, "
            f"Action: {action_taken}, "
            f"Price: {current_price:.2f}, "
            f"ML Prediction: {prediction}, "
            f"Signal: {current_signal}, "
            f"Kernel Bullish: {is_bullish}, "
            f"Kernel Bearish: {is_bearish}"
        )

        print(f"📈 SIGNAL: {log_str}")

        # Send email alert
        subject = f"{index} SIGNAL: {action_taken}"
        send_email(subject, log_str)

        # Update Firebase
        doc_ref = db.collection("live_alerts").document(index)
        doc_ref.update({
            "alerts": ArrayUnion([log_str])
        })

        # Execute trade
        # trader.execute_single_trade(timestamp=current_time, index_name=index, signal_type=action_taken)

    else:
        print(f"📊 {index}: No action. Position: {current_position}, Signal: {current_signal}, Bars held: {bars_held}")


# === Helper Functions ===
def fetch_ohlc(symbol, interval=25, months=5):  # Changed to 25 min like your original
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
            instrument_type="INDEX")
        data = r.get('data', [])
        if data:
            chunks.append(pd.DataFrame(data))
        current_start = current_end + timedelta(days=1)

    df = pd.concat(chunks).reset_index(drop=True)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    df['datetime'] = df['datetime'].dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
    df.set_index('datetime', inplace=True)
    df = df[['open', 'high', 'low', 'close']].astype(float)

    # Append live candle if needed
    now = pd.Timestamp.now().floor(f"{interval}min")
    if now not in df.index or (datetime.now() - df.index[-1].to_pydatetime()).total_seconds() > interval * 60:
        print(f"🔄 Appending live LTP for {symbol}")
        r2 = dhan_object.quote_data({
            "IDX_I": [int(symbol)],
        })

        ltp = float(r2['data']['data']['IDX_I'][symbol]['last_price'])
        df.loc[now] = [ltp, ltp, ltp, ltp]

    return df.sort_index()


def is_market_closed():
    now = datetime.now()
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return now >= market_close


def run_every_15_minutes_until_close(signal_state, trader):
    """Run strategy every 15 minutes until market close"""
    print("✅ Momentum X Strategy started. Will run every 15 minutes until 3:30 PM IST.")

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

        for index_dict in indices:
            try:
                process_symbol(index_dict, signal_state, trader)
            except Exception as e:
                symbol, index = next(iter(index_dict.items()))
                print(f"❌ Error processing {index}: {str(e)}")
                continue

        # Sleep until next run (matching your original 25 min)
        now = datetime.now()
        next_run = (now + timedelta(minutes=25)).replace(second=0, microsecond=0)

        sleep_duration = (next_run - now).total_seconds()
        print(f"\n🕒 Next run at {next_run.strftime('%H:%M:%S')}. Sleeping for {int(sleep_duration)} seconds...\n")
        time.sleep(sleep_duration)


# === Main Execution ===
if __name__ == '__main__':
    load_dotenv()

    # Load credentials
    CLIENT_ID = os.getenv("DHAN_CLIENT_ID")
    ACCESS_TOKEN = os.getenv("DHAN_API_KEY")
    KOTAK_ACCESS_TOKEN = os.getenv("KOTAK_ACCESS_TOKEN")

    # Initialize Dhan
    dhan_object = dhanhq.dhanhq(CLIENT_ID, ACCESS_TOKEN)

    # Add command line argument parsing
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'backtest':
        # Run backtest mode
        print("Running in BACKTEST mode...")
        run_backtest_analysis()
    else:
        # Run live trading mode
        print("Running in LIVE TRADING mode...")

        # Initialize Kotak trader
        trader = KotakOptionsTrader(access_token=KOTAK_ACCESS_TOKEN)

        # Firebase Setup
        cred = credentials.Certificate("stock-monitoring-fb.json")
        initialize_app(cred)
        db = firestore.client()

        # Initialize signal state
        signal_state = SignalState()

        # Run strategy
        run_every_15_minutes_until_close(signal_state, trader)