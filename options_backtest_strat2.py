# momentum_x_optimized.py - High accuracy trading strategy with signal persistence
import csv
import shutil
import time
import json
import os
from datetime import datetime, timedelta

from firebase_admin import credentials, firestore, initialize_app
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from google.cloud.firestore_v1 import ArrayUnion
from numba import jit

from send_mail import send_email
from strategy_features import compute_feature, rational_quadratic_kernel, gaussian_kernel
import dhanhq
from trade_placement import KotakOptionsTrader

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# === CONFIG ===
# Feature configuration - EXACT match to Pine Script
FEATURE_CONFIG = [
    ("RSI", 14, 1),  # Feature 1
    ("WT", 10, 11),  # Feature 2
    ("CCI", 20, 1),  # Feature 3
    ("ADX", 20, 2),  # Feature 4
    ("RSI", 9, 1),  # Feature 5
]

# ML Parameters - EXACT match to Pine Script inputs: close 8 5,000 5 1
NEIGHBOR_COUNT = 8  # "defval" in Pine Script
LUCKY_NUMBER = 3  # Not directly used for threshold
MAX_BARS_BACK = 5000  # Can be 5000 based on Pine input
LOOKAHEAD = 4
FEATURE_COUNT = 5
COLOR_COMPRESSION = 1
STRICT_EXIT_BARS = 25

# Critical thresholds - TUNED FOR ACCURACY
LABEL_THRESHOLD = 0.001  # Pine Script comment suggests this over 0
CONFIDENCE_THRESHOLD = 1  # Require strong consensus (at least 3 votes)

# Kernel Settings - EXACT match to Pine Script
USE_KERNEL_FILTER = True  # "Trade with Kernel" ✓
USE_KERNEL_SMOOTHING = True  # "Enhance Kernel Smoothing" ✓
KERNEL_H = 8  # Lookback Window
KERNEL_R = 8.0  # Relative Weighting
KERNEL_X = 25  # Regression Level
KERNEL_LAG = 2  # Lag

# Filter Settings - EXACT match to Pine Script DEFAULTS
USE_VOLATILITY_FILTER = True  # Pine default is True
USE_REGIME_FILTER = True  # Pine default is True
REGIME_THRESHOLD = -0.1

USE_ADX_FILTER = False  # Pine default is False
ADX_THRESHOLD = 20

# Additional Filters (Pine Script defaults)
USE_EMA_FILTER = False  # Pine default is False
EMA_PERIOD = 200  # Pine default is 200
USE_SMA_FILTER = False  # Pine default is False
SMA_PERIOD = 200

# Dynamic Exits
USE_DYNAMIC_EXITS = True  # Pine default is False

# Signal persistence file
SIGNAL_STATE_FILE = "signal_state.json"
DEBUG = True

CANDLE_INTERVAL_MINUTES = 15  # Change this to modify candle interval globally
CANDLE_COMPLETION_BUFFER = 5  # Seconds to wait after candle closes

# Add realistic execution price (slight slippage)
BACKTEST_SLIPPAGE = 0.02  # 0.02% slippage

MIN_CONSENSUS_RATIO = 0.6

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


class TradeRecorder:
    """Records trades during backtest"""

    def __init__(self):
        self.trades = []
        self.positions = []
        self.current_position = None
        self.entry_price = None
        self.entry_time = None

    def record_trade(self, timestamp, index_name, signal_type, price):
        # Ensure timestamp is a string
        if isinstance(timestamp, pd.Timestamp):
            timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
        else:
            timestamp_str = str(timestamp)

        self.trades.append({
            'timestamp': timestamp_str,
            'index': index_name,
            'action': signal_type,
            'price': price
        })

        # Track positions for P&L
        if signal_type in ['BUY', 'SHORT']:
            self.current_position = signal_type
            self.entry_price = price
            self.entry_time = timestamp_str  # Use string version
        elif signal_type in ['SELL', 'COVER']:
            if self.current_position:
                profit = price - self.entry_price if self.current_position == 'BUY' else self.entry_price - price
                self.positions.append({
                    'index': index_name,
                    'entry_time': self.entry_time,  # Already a string
                    'exit_time': timestamp_str,  # Use string version
                    'entry_price': self.entry_price,
                    'exit_price': price,
                    'type': self.current_position,
                    'profit': profit,
                    'profit_pct': (profit / self.entry_price) * 100
                })
            self.current_position = None


def replace_csv_content(source_filename, dest_filename):
    if not os.path.exists(source_filename):
        raise FileNotFoundError(f"Source file '{source_filename}' does not exist.")

    if not os.path.exists(dest_filename):
        raise FileNotFoundError(f"Destination file '{dest_filename}' does not exist.")

    shutil.copyfile(source_filename, dest_filename)
    return True


# === Feature Calculation ===
def build_features(df):
    """Build technical indicator features"""
    features = []
    for kind, param_a, param_b in FEATURE_CONFIG:
        f = compute_feature(df.copy(), kind, param_a, param_b)
        f.name = f"{kind}_{param_a}_{param_b}"

        # Validate features are in expected range
        if DEBUG and len(f) > 0:
            if f.min() < -1.5 or f.max() > 1.5:
                print(f"WARNING: Feature {f.name} out of expected range: [{f.min():.2f}, {f.max():.2f}]")

        features.append(f)
    return pd.concat(features, axis=1).dropna()


def save_historical_data(symbols_dict, start_date=None, end_date=None, months=10):
    """
    Fetch and save historical data locally

    Args:
        symbols_dict: Dictionary of symbol:index pairs
        start_date: Start date string (YYYY-MM-DD). If provided, ignores months param
        end_date: End date string (YYYY-MM-DD). If provided, ignores months param
        months: Number of months to fetch (used only if start_date/end_date not provided)
    """
    data_dir = "historical_data"
    os.makedirs(data_dir, exist_ok=True)

    # Determine date range
    if start_date and end_date:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        date_suffix = f"{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}"
        print(f"Fetching data from {start_date} to {end_date}")
    else:
        end = datetime.now()
        start = end - timedelta(days=30 * months)
        date_suffix = f"{months}months"
        print(f"Fetching last {months} months of data")

    for symbol, index in symbols_dict.items():
        print(f"Fetching data for {index}...")

        # Fetch data in chunks
        chunks = []
        current_start = start

        while current_start <= end:
            current_end = min(current_start + timedelta(days=75), end)

            try:
                r = dhan_object.intraday_minute_data(
                    security_id=symbol,
                    from_date=current_start.strftime('%Y-%m-%d'),
                    to_date=current_end.strftime('%Y-%m-%d'),
                    interval=CANDLE_INTERVAL_MINUTES,
                    exchange_segment="IDX_I",
                    instrument_type="INDEX"
                )
                data = r.get('data', [])
                if data:
                    chunks.append(pd.DataFrame(data))
            except Exception as e:
                print(f"  Error fetching {index} for {current_start.strftime('%Y-%m-%d')}: {e}")

            current_start = current_end + timedelta(days=1)

        if not chunks:
            print(f"  No data fetched for {index}")
            continue

        # Combine chunks
        df = pd.concat(chunks).reset_index(drop=True)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df['datetime'] = df['datetime'].dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
        df.set_index('datetime', inplace=True)
        df = df[['open', 'high', 'low', 'close']].astype(float)
        df = df.sort_index()

        # Save with date suffix
        filename = f"{data_dir}/{index}_{date_suffix}.pkl"
        df.to_pickle(filename)
        print(f"  Saved {len(df)} rows to {filename}")

        # Also save as CSV for inspection
        csv_filename = f"{data_dir}/{index}_{date_suffix}.csv"
        df.to_csv(csv_filename)
        print(f"  Also saved as {csv_filename}")


def load_historical_data(index, filename=None):
    """Load saved historical data with proper datetime parsing"""
    if filename is None:
        filename = f"historical_data/{index}_10months.pkl"

    if os.path.exists(filename):
        # Load CSV with datetime parsing - 'datetime' is the column name
        df = pd.read_csv(filename, parse_dates=['datetime'], index_col='datetime')

        # Ensure columns are float
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                df[col] = df[col].astype(float)

        return df
    else:
        raise FileNotFoundError(f"Please run save_historical_data first. {filename} not found.")


def validate_minimum_data_requirements(df, index_name):
    """Ensure we have minimum data for reliable predictions"""
    min_bars_required = max(100, LOOKAHEAD * 10)  # At least 10x lookahead period

    if len(df) < min_bars_required:
        if DEBUG:
            print(f"⚠️ {index_name}: Only {len(df)} bars available, need {min_bars_required} minimum")
        return False

    # Check if we have enough neighbors for ML
    features = build_features(df)
    if len(features) < NEIGHBOR_COUNT * 2:
        if DEBUG:
            print(f"⚠️ {index_name}: Insufficient features for ML predictions")
        return False

    return True


def calculate_adx(df, period=14):
    """Calculate ADX (Average Directional Index) matching Pine Script"""
    high = df['high']
    low = df['low']
    close = df['close']

    # Calculate True Range (TR)
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Calculate Directional Movement
    high_diff = high - high.shift(1)
    low_diff = low.shift(1) - low

    # +DM and -DM
    plus_dm = pd.Series(0.0, index=df.index)
    minus_dm = pd.Series(0.0, index=df.index)

    # When high_diff > low_diff and high_diff > 0, +DM = high_diff, else +DM = 0
    plus_dm[high_diff > low_diff] = high_diff[high_diff > low_diff]
    plus_dm[plus_dm < 0] = 0

    # When low_diff > high_diff and low_diff > 0, -DM = low_diff, else -DM = 0
    minus_dm[low_diff > high_diff] = low_diff[low_diff > high_diff]
    minus_dm[minus_dm < 0] = 0

    # Smooth using RMA (Running Moving Average) - Pine Script uses RMA for ADX
    def rma(series, period):
        """Pine Script's RMA (Running Moving Average)"""
        alpha = 1.0 / period
        return series.ewm(alpha=alpha, adjust=False).mean()

    # Smooth TR, +DM, -DM
    atr = rma(tr, period)
    plus_di = 100 * rma(plus_dm, period) / atr
    minus_di = 100 * rma(minus_dm, period) / atr

    # Calculate DX
    di_sum = plus_di + minus_di
    di_diff = abs(plus_di - minus_di)
    dx = 100 * di_diff / di_sum
    dx = dx.fillna(0)

    # Calculate ADX
    adx = rma(dx, period)

    return adx


def check_early_exit_signals(df, position, bars_held, entry_price):
    """Detect early signs that a trade is going wrong"""
    if bars_held == 0 or len(df) < 5:
        return False

    current_price = df['close'].iloc[-1]
    current_profit_pct = ((current_price - entry_price) / entry_price * 100) if position == 'LONG' else (
            (entry_price - current_price) / entry_price * 100)

    # Early exit conditions (without using stop loss)
    early_exit_conditions = []

    # 1. Momentum reversal in first 2 bars
    if bars_held <= 2:
        # Check if price is moving against us quickly
        if position == 'LONG':
            # Price made lower low than entry bar
            going_wrong = df['low'].iloc[-1] < df['low'].iloc[-bars_held - 1]
            momentum_against = df['close'].iloc[-1] < df['open'].iloc[-1]  # Red candle
        else:  # SHORT
            # Price made higher high than entry bar
            going_wrong = df['high'].iloc[-1] > df['high'].iloc[-bars_held - 1]
            momentum_against = df['close'].iloc[-1] > df['open'].iloc[-1]  # Green candle

        if going_wrong and momentum_against and current_profit_pct < 0:
            early_exit_conditions.append("early_momentum_reversal")

    # 2. Failed breakout pattern (price rejection)
    if 1 <= bars_held <= 3:
        # Check for price rejection patterns
        current_range = (df['high'].iloc[-1] - df['low'].iloc[-1]) / df['low'].iloc[-1] * 100

        if position == 'LONG':
            # Long wick on top (rejection at highs)
            wick_ratio = (df['high'].iloc[-1] - df['close'].iloc[-1]) / (df['high'].iloc[-1] - df['low'].iloc[-1])
            rejected = wick_ratio > 0.6 and current_range > 0.2
        else:  # SHORT
            # Long wick on bottom (rejection at lows)
            wick_ratio = (df['close'].iloc[-1] - df['low'].iloc[-1]) / (df['high'].iloc[-1] - df['low'].iloc[-1])
            rejected = wick_ratio > 0.6 and current_range > 0.2

        if rejected and current_profit_pct < -0.05:
            early_exit_conditions.append("price_rejection")

    # 3. Accelerating adverse movement
    if bars_held >= 2:
        # Check if losses are accelerating
        prev_close = df['close'].iloc[-2]
        prev_profit_pct = ((prev_close - entry_price) / entry_price * 100) if position == 'LONG' else (
                (entry_price - prev_close) / entry_price * 100)

        # If loss is getting worse each bar
        if current_profit_pct < -0.1 and current_profit_pct < prev_profit_pct:
            early_exit_conditions.append("accelerating_loss")

    if early_exit_conditions and DEBUG:
        print(f"  ⚠️ Early exit signals: {', '.join(early_exit_conditions)}")

    return len(early_exit_conditions) > 0


@jit(nopython=True)
def process_single_prediction_pine(X, Y, i, max_lookback, neighbor_count):
    """Process prediction using Pine Script's exact logic"""
    lastDistance = -1.0
    distances = np.zeros(neighbor_count)
    votes = np.zeros(neighbor_count, dtype=np.int32)
    count = 0

    # Pine Script iterates forward through historical data
    size = min(max_lookback, i)

    for loop_idx in range(size):
        # Pine Script's modulo: skip when loop_idx % 4 == 0
        if loop_idx % 4 == 0:
            continue

        # Historical index
        hist_idx = i - size + loop_idx
        if hist_idx >= i:
            continue

        # Lorentzian distance
        d = 0.0
        for k in range(X.shape[1]):
            d += np.log(1 + np.abs(X[i, k] - X[hist_idx, k]))

        # Pine Script: d >= lastDistance
        if d >= lastDistance:
            lastDistance = d

            if count < neighbor_count:
                distances[count] = d
                votes[count] = Y[hist_idx]
                count += 1
            else:
                # Shift arrays
                for idx in range(neighbor_count - 1):
                    distances[idx] = distances[idx + 1]
                    votes[idx] = votes[idx + 1]
                distances[neighbor_count - 1] = d
                votes[neighbor_count - 1] = Y[hist_idx]

                # Update threshold
                lastDistance = distances[int(neighbor_count * 3 / 4)]

    # Return sum of ALL votes
    return np.sum(votes[:count]), count


def predict_labels(features, close_series):
    """
    Generate ML predictions using Lorentzian distance (optimized)
    """
    # Align indices
    common_index = features.index.intersection(close_series.index)
    features = features.loc[common_index]
    close_series = close_series.loc[common_index]

    X = features.values.astype(np.float64)
    close = close_series.values
    n = len(X)

    # Create labels - with CORRECT direction
    Y = np.zeros(n, dtype=np.int32)
    for i in range(n - LOOKAHEAD):
        future_close = close[i + LOOKAHEAD]
        current_close = close[i]
        delta = (future_close - current_close) / current_close

        if delta > LABEL_THRESHOLD:
            Y[i] = 1  # Long when price goes UP
        elif delta < -LABEL_THRESHOLD:
            Y[i] = -1  # Short when price goes DOWN
        else:
            Y[i] = 0

    # Generate predictions
    predictions = np.zeros(n, dtype=np.int32)

    for i in range(LOOKAHEAD, n):
        max_lookback = min(i, MAX_BARS_BACK)

        if max_lookback < NEIGHBOR_COUNT:
            predictions[i] = 0
            continue

        # In predict_labels, update this section:
        vote_sum, vote_count = process_single_prediction_pine(
            X, Y, i, max_lookback, NEIGHBOR_COUNT
        )

        # ADD THIS: Require 60% consensus for high-quality signals

        if vote_count >= LUCKY_NUMBER:  # Need at least 3 votes
            consensus_ratio = abs(vote_sum) / vote_count
            if abs(vote_sum) >= CONFIDENCE_THRESHOLD and consensus_ratio >= MIN_CONSENSUS_RATIO:
                predictions[i] = int(np.sign(vote_sum))
            else:
                predictions[i] = 0
        else:
            predictions[i] = 0

    return pd.Series(predictions, index=features.index)


def calculate_volatility_filter(close, use_filter=True):
    """Pine Script volatility filter using ATR comparison"""
    if not use_filter:
        return True

    # Create a DataFrame for ATR calculation
    df_temp = pd.DataFrame({
        'high': close,
        'low': close,
        'close': close
    })

    # Pine Script uses 1 and 10 for volatility filter parameters
    recent_atr = calculate_atr(df_temp, period=1)
    historical_atr = calculate_atr(df_temp, period=10)

    # Return True if recent volatility > historical volatility
    if len(recent_atr) > 0 and len(historical_atr) > 0:
        return recent_atr.iloc[-1] > historical_atr.iloc[-1]
    return True


def calculate_regime_filter(close, threshold=-0.1, use_filter=True):
    """Pine Script regime filter - detects trending vs ranging markets"""
    if not use_filter:
        return True

    # Pine Script's regime filter uses normalized slope
    lookback = 50  # Typical lookback period

    if len(close) < lookback:
        return True

    # Calculate linear regression slope
    x = np.arange(lookback)
    y = close.iloc[-lookback:].values

    # Normalize the data
    y_mean = np.mean(y)
    if y_mean != 0:
        y_normalized = (y - y_mean) / y_mean
        slope = np.polyfit(x, y_normalized, 1)[0]
        return slope > threshold

    return True


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
    """Calculate kernel signals - MATCHING PINE SCRIPT EXACTLY"""
    yhat1 = rational_quadratic_kernel(series, h, r, x)
    yhat2 = gaussian_kernel(series, h - lag, x)

    if USE_KERNEL_SMOOTHING:
        # Smoothed mode uses crossovers
        bullish = yhat2 >= yhat1
        bearish = yhat2 <= yhat1
    else:
        # Non-smoothed mode uses RATE CHANGES
        # Pine Script: isBullishChange = isBullishRate and wasBearishRate
        if len(yhat1) < 3:
            return pd.Series([False] * len(series), index=series.index), pd.Series([False] * len(series),
                                                                                   index=series.index), yhat1, yhat2

        # Rate calculations matching Pine Script exactly
        wasBearishRate = yhat1.shift(2) > yhat1.shift(1)
        wasBullishRate = yhat1.shift(2) < yhat1.shift(1)
        isBearishRate = yhat1.shift(1) > yhat1
        isBullishRate = yhat1.shift(1) < yhat1

        # Changes (transitions) - this is what Pine Script uses for signals
        bullish = isBullishRate & wasBearishRate
        bearish = isBearishRate & wasBullishRate

    return bullish, bearish, yhat1, yhat2


def check_dynamic_exit_conditions(df, position, bars_held):
    """Enhanced dynamic exit conditions"""
    if not USE_DYNAMIC_EXITS:
        return False

    if len(df) < 10:
        return False

    current_idx = len(df) - 1

    # Get kernel signals
    _, _, yhat1, yhat2 = get_kernel_signals(df['close'])

    # Calculate momentum
    momentum = df['close'].pct_change(5).iloc[-1]  # 5-bar momentum

    if USE_KERNEL_SMOOTHING:
        if position == 'LONG':
            # Exit conditions for long - check for crossunder
            bearish_cross = (yhat2.iloc[current_idx] < yhat1.iloc[current_idx] and
                             yhat2.iloc[current_idx - 1] >= yhat1.iloc[current_idx - 1])

            # Additional exit: momentum reversal
            momentum_exit = momentum < -0.002  # -0.2% momentum threshold

            return (bearish_cross or momentum_exit) and bars_held > 0

        elif position == 'SHORT':
            # Exit conditions for short - check for crossover
            bullish_cross = (yhat2.iloc[current_idx] > yhat1.iloc[current_idx] and
                             yhat2.iloc[current_idx - 1] <= yhat1.iloc[current_idx - 1])

            # Additional exit: momentum reversal
            momentum_exit = momentum > 0.002  # +0.2% momentum threshold

            return (bullish_cross or momentum_exit) and bars_held > 0

    return False


def calculate_entry_quality(df, signal_type):
    """Only enter if we can realistically hit 0.1% profit"""

    # 1. Check recent volatility - need movement for profit
    recent_moves = df['close'].pct_change().iloc[-10:].abs()
    avg_move = recent_moves.mean() * 100  # Convert to percentage
    has_volatility = avg_move >= 0.05  # Need at least 0.05% average moves

    # 2. Check momentum alignment
    momentum_1 = (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2] * 100
    momentum_3 = (df['close'].iloc[-1] - df['close'].iloc[-4]) / df['close'].iloc[-4] * 100

    if signal_type == 'LONG':
        momentum_aligned = momentum_1 > 0 and momentum_3 > 0.05  # Rising momentum
    else:
        momentum_aligned = momentum_1 < 0 and momentum_3 < -0.05  # Falling momentum

    # 3. Check if recent bars had 0.1% moves (proving it's possible)
    recent_ranges = ((df['high'] - df['low']) / df['low'] * 100).iloc[-5:]
    can_move = recent_ranges.max() >= 0.15  # At least one bar had 0.15% range

    # 4. Kernel acceleration check
    _, _, yhat1, _ = get_kernel_signals(df['close'])
    if len(yhat1) >= 3:
        kernel_accel = yhat1.iloc[-1] - yhat1.iloc[-3]
        if signal_type == 'LONG':
            kernel_accelerating = kernel_accel > 0
        else:
            kernel_accelerating = kernel_accel < 0
    else:
        kernel_accelerating = True

    # All conditions must be true
    quality_score = has_volatility and momentum_aligned and can_move and kernel_accelerating

    # Debug logging
    if DEBUG and not quality_score:
        print(
            f"  Entry Quality Failed - Vol:{has_volatility}, Mom:{momentum_aligned}, Move:{can_move}, Kernel:{kernel_accelerating}")

    return quality_score


def predict_labels_single(features, close_series, current_idx):
    """Optimized prediction for single candle"""
    X = features.values.astype(np.float64)
    close = close_series.values
    n = len(X)

    # Create labels
    Y = np.zeros(n, dtype=np.int32)
    for i in range(n - LOOKAHEAD):
        future_close = close[i + LOOKAHEAD]
        current_close = close[i]
        delta = (future_close - current_close) / current_close

        if delta > LABEL_THRESHOLD:
            Y[i] = 1
        elif delta < -LABEL_THRESHOLD:
            Y[i] = -1

    # Only predict for current candle
    i = current_idx
    max_lookback = min(i, MAX_BARS_BACK)

    if max_lookback < NEIGHBOR_COUNT:
        return 0

    vote_sum, vote_count = process_single_prediction_pine(X, Y, i, max_lookback, NEIGHBOR_COUNT)

    if vote_count >= LUCKY_NUMBER:
        consensus_ratio = abs(vote_sum) / vote_count
        if abs(vote_sum) >= CONFIDENCE_THRESHOLD and consensus_ratio >= MIN_CONSENSUS_RATIO:
            return int(np.sign(vote_sum))
    return 0

def calculate_market_structure(df):
    """Identify market regime - only trade with it"""
    # Need enough data for all MAs
    if len(df) < 200:
        return 'NEUTRAL'

    try:
        sma20 = df['close'].rolling(20).mean()
        sma50 = df['close'].rolling(50).mean()
        sma200 = df['close'].rolling(200).mean()

        # Market structure
        bullish_structure = (
                df['close'].iloc[-1] > sma20.iloc[-1] > sma50.iloc[-1] > sma200.iloc[-1]
        )

        bearish_structure = (
                df['close'].iloc[-1] < sma20.iloc[-1] < sma50.iloc[-1] < sma200.iloc[-1]
        )

        # Additional strength check - MAs should be properly spaced
        if bullish_structure:
            # Check if MAs are spreading apart (strong trend)
            ma_spread = (sma20.iloc[-1] - sma200.iloc[-1]) / sma200.iloc[-1] * 100
            if ma_spread > 0.5:  # Strong bullish if spread > 0.5%
                return 'STRONG_BULLISH'
            else:
                return 'BULLISH'

        elif bearish_structure:
            # Check if MAs are spreading apart (strong trend)
            ma_spread = (sma200.iloc[-1] - sma20.iloc[-1]) / sma200.iloc[-1] * 100
            if ma_spread > 0.5:  # Strong bearish if spread > 0.5%
                return 'STRONG_BEARISH'
            else:
                return 'BEARISH'
        else:
            return 'NEUTRAL'

    except Exception as e:
        if DEBUG:
            print(f"Error calculating market structure: {e}")
        return 'NEUTRAL'


def process_symbol(symbol_dict, signal_state, trader, quote_data=None, mode='live', trade_recorder=None, window_df=None, precalc_data=None):
    """Process a single symbol for live trading - UNIFIED LOGIC"""
    symbol, index = next(iter(symbol_dict.items()))

    # Fetch data based on mode
    if mode == 'live':
        # Fetch OHLC data from API
        df = fetch_ohlc(symbol)
    else:
        # Use provided window for backtest
        df = window_df

    # Skip validation in backtest if precalc provided
    if not precalc_data:
        # Ensure we have consistent history window
        if len(df) > MAX_BARS_BACK:
            df = df.iloc[-MAX_BARS_BACK:].copy()

        # Validate minimum data requirements
        if not validate_minimum_data_requirements(df, index):
            return

    # Use pre-calculated or calculate fresh
    if precalc_data:
        features = precalc_data['features']
        current_signal = precalc_data['prediction']
        current_bullish = precalc_data['bullish']
        current_bearish = precalc_data['bearish']
        current_atr = precalc_data['atr']
    else:
        # Build features
        features = build_features(df)
        if len(features) < LOOKAHEAD:
            print(f"Not enough features for {index}")
            return

        # Generate predictions
        predictions = predict_labels(features, df['close'])
        current_signal = predictions.iloc[-1]

        # Get kernel signals
        bullish, bearish, rk, gk = get_kernel_signals(df['close'])
        current_bullish = bullish.iloc[-1]
        current_bearish = bearish.iloc[-1]

        # Calculate ATR
        atr = calculate_atr(df)
        current_atr = atr.iloc[-1] if len(atr) > 0 and not pd.isna(atr.iloc[-1]) else 0

    # Get current values
    current_time = df.index[-1].strftime('%Y-%m-%d %H:%M:%S') if isinstance(df.index[-1], pd.Timestamp) else str(df.index[-1])
    current_price = df['close'].iloc[-1]

    # Get current position
    current_position = signal_state.get_position(index)
    bars_held = signal_state.get_bars_held(index)

    action_taken = None

    # Calculate all filters (always fresh, they're fast)
    current_vol_filter = calculate_volatility_filter(df['close'], USE_VOLATILITY_FILTER)
    current_regime_filter = calculate_regime_filter(df['close'], REGIME_THRESHOLD, USE_REGIME_FILTER)

    # ADX Filter
    if USE_ADX_FILTER:
        adx = calculate_adx(df, period=14)  # Fixed: use period=14, not ADX_THRESHOLD
        current_adx_filter = adx.iloc[-1] > ADX_THRESHOLD if len(adx) > 0 else False
    else:
        current_adx_filter = True

    # EMA Filter
    if USE_EMA_FILTER:
        ema = df['close'].ewm(span=EMA_PERIOD, adjust=False).mean()
        ema_uptrend = df['close'].iloc[-1] > ema.iloc[-1]
        ema_downtrend = df['close'].iloc[-1] < ema.iloc[-1]
    else:
        ema_uptrend = True
        ema_downtrend = True

    # SMA Filter
    if USE_SMA_FILTER:
        sma = df['close'].rolling(window=SMA_PERIOD).mean()
        sma_uptrend = df['close'].iloc[-1] > sma.iloc[-1] if len(sma) > 0 and not pd.isna(sma.iloc[-1]) else True
        sma_downtrend = df['close'].iloc[-1] < sma.iloc[-1] if len(sma) > 0 and not pd.isna(sma.iloc[-1]) else True
    else:
        sma_uptrend = True
        sma_downtrend = True

    # Apply all filters
    all_filters = current_vol_filter and current_regime_filter and current_adx_filter

    # Directional filters for entries
    long_filters = all_filters and ema_uptrend and sma_uptrend
    short_filters = all_filters and ema_downtrend and sma_downtrend

    # ===== SIGNAL PERSISTENCE LOGIC =====
    # Get previous signal for persistence check
    previous_signal = signal_state.get_last_signal(index)

    # Apply signal persistence logic (Pine Script's nz(signal[1]))
    if all_filters:  # Only update signal if filters pass
        if current_signal != 0:
            persisted_signal = current_signal
        else:
            persisted_signal = previous_signal
    else:
        persisted_signal = previous_signal

    # ===== END SIGNAL PERSISTENCE LOGIC =====

    # Exit logic - MORE LIKE PINE SCRIPT
    if current_position == 'LONG':
        signal_state.increment_bars_held(index)
        bars_held = signal_state.get_bars_held(index)

        # Get entry price
        entry_info = signal_state.get_entry_info(index)
        entry_price = entry_info.get('price', current_price)

        # Calculate profit percentage
        profit_pct = ((current_price - entry_price) / entry_price) * 100

        # Exit conditions:
        # 1. Hit 0.1% profit target
        # 2. Hit 0.05% stop loss (NEW!)
        # 3. Held for exactly 4 bars (strict exit)
        # 4. New opposite signal appears before 4 bars
        # 5. Dynamic exit (if enabled)

        force_exit_at_315 = False
        if hasattr(df.index[-1], 'time'):
            current_candle_time = df.index[-1].time()
            # Exit at 3:15 PM or later
            if current_candle_time >= pd.Timestamp("15:15:00").time():
                force_exit_at_315 = True
                if DEBUG:
                    print(f"  🕐 Forcing exit at 3:15 PM for intraday square-off")

        profit_target_hit = profit_pct >= 0.1  # 0.1% profit target
        # early_exit_signal = check_early_exit_signals(df, 'LONG', bars_held, entry_price)
        # stop_loss_hit = profit_pct <= -0.1  # 0.05% stop loss - TIGHT!
        strict_exit = bars_held >= STRICT_EXIT_BARS
        opposite_signal = persisted_signal == -1 and current_bearish and short_filters
        dynamic_exit = check_dynamic_exit_conditions(df, 'LONG', bars_held) if USE_DYNAMIC_EXITS else False

        if force_exit_at_315 or profit_target_hit or strict_exit or opposite_signal or dynamic_exit:
            action_taken = "SELL"
            signal_state.update_position(index, None)
            signal_state.reset_bars_held(index)

    elif current_position == 'SHORT':
        signal_state.increment_bars_held(index)
        bars_held = signal_state.get_bars_held(index)

        # Get entry price
        entry_info = signal_state.get_entry_info(index)
        entry_price = entry_info.get('price', current_price)

        # Calculate profit percentage (for SHORT, profit is when price goes DOWN)
        profit_pct = ((entry_price - current_price) / entry_price) * 100

        force_exit_at_315 = False
        if hasattr(df.index[-1], 'time'):
            current_candle_time = df.index[-1].time()
            # Exit at 3:15 PM or later
            if current_candle_time >= pd.Timestamp("15:15:00").time():
                force_exit_at_315 = True
                if DEBUG:
                    print(f"  🕐 Forcing exit at 3:15 PM for intraday square-off")

        # Exit conditions with profit target and stop loss
        profit_target_hit = profit_pct >= 0.1  # 0.1% profit target
        # early_exit_signal = check_early_exit_signals(df, 'SHORT', bars_held, entry_price)
        strict_exit = bars_held >= STRICT_EXIT_BARS
        opposite_signal = persisted_signal == 1 and current_bullish and long_filters
        dynamic_exit = check_dynamic_exit_conditions(df, 'SHORT', bars_held) if USE_DYNAMIC_EXITS else False

        if force_exit_at_315 or strict_exit or opposite_signal or dynamic_exit or profit_target_hit:
            action_taken = "COVER"
            signal_state.update_position(index, None)
            signal_state.reset_bars_held(index)

    # Entry logic - MORE LIKE PINE SCRIPT
    if current_position is None:

        # NEW: Check if we should skip this candle for entries
        if should_skip_candle_for_entry(df.index[-1]):
            if DEBUG:
                print(f"  ⏭️ Skipping entry signals for {index} at {current_time} (first/last candle)")
            # Still update the signal state but don't take action
            signal_state.update_signal(index,
                                       int(persisted_signal) if hasattr(persisted_signal, 'item') else persisted_signal)
            return

        # ADD THIS: Check market structure first
        market_structure = calculate_market_structure(df)

        # Check entry quality first!
        long_quality = calculate_entry_quality(df, 'LONG')
        short_quality = calculate_entry_quality(df, 'SHORT')

        # ADD THIS: Market structure alignment check
        can_go_long = market_structure in ['BULLISH', 'STRONG_BULLISH']
        can_go_short = market_structure in ['BEARISH', 'STRONG_BEARISH']

        if persisted_signal == 1 and current_bullish and long_filters and long_quality and can_go_long:
            action_taken = "BUY"
            signal_state.update_position(index, 'LONG', float(current_price), current_time)
            signal_state.reset_bars_held(index)

            # Log market structure
            if DEBUG:
                print(f"  Entering LONG in {market_structure} market")

        elif persisted_signal == -1 and current_bearish and short_filters and short_quality and can_go_short:
            action_taken = "SHORT"
            signal_state.update_position(index, 'SHORT', float(current_price), current_time)
            signal_state.reset_bars_held(index)

            # Log market structure
            if DEBUG:
                print(f"  Entering SHORT in {market_structure} market")

            # ADD THIS: Debug why trades are skipped
        elif DEBUG and (persisted_signal != 0):
            if persisted_signal == 1 and not can_go_long:
                print(f"  LONG signal rejected - Market structure: {market_structure}")
            elif persisted_signal == -1 and not can_go_short:
                print(f"  SHORT signal rejected - Market structure: {market_structure}")

    # Update signal state with PERSISTED signal
    signal_state.update_signal(index, int(persisted_signal) if hasattr(persisted_signal, 'item') else persisted_signal)

    # Log and execute
    log_str = (
        f"INDEX: {index}, "
        f"Time: {current_time}, "
        f"Signal: {persisted_signal}, "  # Changed to show persisted signal
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

        if mode == 'live':
            # Send alerts
            subject = f"{index} SIGNAL: {action_taken}"
            send_email(subject, log_str)

            # Update Firebase
            doc_ref = db.collection("live_alerts").document(index)
            doc_ref.update({"alerts": ArrayUnion([log_str])})

            # Execute trade
            trader.execute_single_trade(timestamp=current_time, index_name=index, signal_type=action_taken)

        elif mode == 'backtest' and trade_recorder:

            if action_taken in ['BUY', 'SHORT']:
                execution_price = current_price * (1 + BACKTEST_SLIPPAGE / 100)
            else:  # SELL, COVER
                execution_price = current_price * (1 - BACKTEST_SLIPPAGE / 100)
            # Record trade for backtest
            trade_recorder.record_trade(current_time, index, action_taken, execution_price)
    else:
        if DEBUG and mode == 'live':  # Only log in live mode to reduce noise
            print(f"📊 {index}: {log_str}")


# === Data Fetching ===
def fetch_ohlc(symbol, interval=CANDLE_INTERVAL_MINUTES, months=10):
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


def process_index_backtest(index_dict, backtest_state_lock):
    """Optimized backtest processing"""
    symbol, index = next(iter(index_dict.items()))
    index_state = SignalState(filepath=f"backtest_signal_state_{index}.json")
    trade_recorder = TradeRecorder()

    print(f"\n🔄 Starting backtest for {index}...")

    # Load all data once
    input_df = pd.read_csv(f'testing_data/{index}_input.csv')
    test_df = pd.read_csv(f'testing_data/{index}_test.csv')
    all_data = pd.concat([input_df, test_df], ignore_index=True)
    all_data['datetime'] = pd.to_datetime(all_data['datetime'])
    all_data.set_index('datetime', inplace=True)

    # Pre-calculate everything
    print(f"Pre-calculating features for {index}...")
    all_features = build_features(all_data)
    all_atr = calculate_atr(all_data)
    all_bullish, all_bearish, all_rk, all_gk = get_kernel_signals(all_data['close'])

    test_start_idx = len(input_df)

    for i in range(test_start_idx, len(all_data)):
        window_start = max(0, i - 4999)
        window_df = all_data.iloc[window_start:i + 1]

        # Prepare pre-calculated data
        precalc_data = {
            'features': all_features.iloc[window_start:i + 1],
            'prediction': predict_labels_single(
                all_features.iloc[window_start:i + 1],
                all_data['close'].iloc[window_start:i + 1],
                i - window_start
            ),
            'bullish': all_bullish.iloc[i],
            'bearish': all_bearish.iloc[i],
            'atr': all_atr.iloc[i] if i < len(all_atr) else 0
        }

        try:
            process_symbol(
                symbol_dict=index_dict,
                signal_state=index_state,
                trader=None,
                quote_data=None,
                mode='backtest',
                trade_recorder=trade_recorder,
                window_df=window_df,
                precalc_data=precalc_data
            )

            if (i - test_start_idx) % 100 == 0:
                print(f"  {index}: Processed {i - test_start_idx} candles...")

        except Exception as e:
            print(f"  ❌ Error processing {index} at {i}: {str(e)}")

    print(f"✅ Completed {index} backtest: {len(test_df)} candles processed")

    if os.path.exists(f"backtest_signal_state_{index}.json"):
        os.remove(f"backtest_signal_state_{index}.json")

    return index, trade_recorder


def run_backtest_analysis():
    """Run comprehensive backtest using parallel processing"""
    print("\n" + "=" * 80)
    print("RUNNING PARALLEL BACKTEST ANALYSIS")
    print("=" * 80 + "\n")

    indices = [
        # {"13": "NIFTY"},
        # {"25": "BANKNIFTY"},
        # {"51": "SENSEX"},
        # {"27": "FINNIFTY"},
        {"442": "MIDCPNIFTY"}
    ]

    # Thread lock for shared resources
    backtest_state_lock = threading.Lock()

    # Run backtests in parallel
    trade_recorders = {}

    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(process_index_backtest, index_dict, backtest_state_lock): index_dict
            for index_dict in indices
        }

        # Collect results as they complete
        for future in as_completed(future_to_index):
            index_dict = future_to_index[future]
            try:
                index_name, recorder = future.result()
                trade_recorders[index_name] = recorder
            except Exception as exc:
                symbol, index_name = next(iter(index_dict.items()))
                print(f"❌ {index_name} generated an exception: {exc}")

    # Analyze results
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS")
    print("=" * 80)

    all_results = {}
    for index, recorder in trade_recorders.items():
        if recorder.positions:
            total_trades = len(recorder.positions)
            winning_trades = sum(1 for p in recorder.positions if p['profit'] > 0)
            total_profit = sum(p['profit'] for p in recorder.positions)
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0

            all_results[index] = {
                'trades': recorder.trades,
                'positions': recorder.positions,
                'performance': {
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'losing_trades': total_trades - winning_trades,
                    'win_rate': win_rate,
                    'total_profit': total_profit,
                    'avg_profit': total_profit / total_trades if total_trades > 0 else 0,
                    'avg_profit_pct': sum(
                        p['profit_pct'] for p in recorder.positions) / total_trades if total_trades > 0 else 0
                }
            }

            print(f"\n{index} Performance:")
            print(f"  Total Trades: {total_trades}")
            print(f"  Win Rate: {win_rate:.2f}%")
            print(f"  Total Profit: {total_profit:.2f}")

    # Save results
    save_backtest_results(all_results)

    return all_results


def move_first_row(input_path, test_path):
    # Read all rows from input.csv
    with open(input_path, newline='') as infile:
        input_reader = list(csv.reader(infile))

    # If there's only the header or less, return False
    if len(input_reader) <= 1:
        return False

    header_input = input_reader[0]
    first_data_row = input_reader[1]
    remaining_input_rows = [header_input] + input_reader[2:]

    # Save the updated input.csv with the first data row removed
    with open(input_path, 'w', newline='') as infile:
        writer = csv.writer(infile)
        writer.writerows(remaining_input_rows)

    # Read all rows from test.csv
    with open(test_path, newline='') as testfile:
        test_reader = list(csv.reader(testfile))

    header_test = test_reader[0]
    test_data_rows = test_reader[1:]

    # Add the row from input and remove the first data row
    test_data_rows.append(first_data_row)
    if test_data_rows:
        test_data_rows.pop(0)

    # Save the updated test.csv
    with open(test_path, 'w', newline='') as testfile:
        writer = csv.writer(testfile)
        writer.writerow(header_test)
        writer.writerows(test_data_rows)

    return True


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

    print(f"✅ Live trading started. Will run every {CANDLE_INTERVAL_MINUTES} minutes until 3:30 PM IST.")

    # Wait for current candle to complete if starting mid-candle
    next_run = get_next_candle_time()
    wait_time = (next_run - datetime.now()).total_seconds()

    if wait_time > 0:
        print(f"\n⏳ Waiting for current candle to complete...")
        print(f"First run will be at {next_run.strftime('%H:%M:%S')} (candle close + 30s)")
        print(f"Waiting {int(wait_time)} seconds...\n")
        time.sleep(wait_time)

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

        # # Fetch all quotes at once
        # all_symbols = [int(list(idx.keys())[0]) for idx in indices]
        # try:
        #     quote_response = dhan_object.quote_data({"IDX_I": all_symbols})
        #     quotes = {}
        #     if 'data' in quote_response and 'data' in quote_response['data'] and 'IDX_I' in quote_response['data'][
        #         'data']:
        #         quotes = quote_response['data']['data']['IDX_I']
        # except Exception as e:
        #     print(f"Error fetching quotes: {e}")
        #     quotes = {}

        for index_dict in indices:
            try:
                # symbol, index = next(iter(index_dict.items()))
                process_symbol(index_dict, signal_state, trader, quote_data=None, mode='live')
            except Exception as e:
                symbol, index = next(iter(index_dict.items()))
                print(f"❌ Error processing {index}: {str(e)}")
                continue

        # Sleep for 25 minutes
        next_run = get_next_candle_time()
        sleep_duration = (next_run - datetime.now()).total_seconds()

        if sleep_duration > 0:
            print(f"\n🕒 Next run at {next_run.strftime('%H:%M:%S')} (candle close + {CANDLE_COMPLETION_BUFFER}s). "
                  f"Sleeping for {int(sleep_duration)} seconds...\n")
            time.sleep(sleep_duration)


def should_skip_candle_for_entry(timestamp):
    """Skip entry signals on first and last candles of the day"""
    if hasattr(timestamp, 'time'):
        candle_time = timestamp.time()
    else:
        candle_time = pd.Timestamp(timestamp).time()

    hour = candle_time.hour
    minute = candle_time.minute

    # Skip 9:15 candle (market open)
    if hour == 9 and minute == 15:
        return True

    # Skip 3:15 candle (near market close)
    if hour == 15 and minute == 15:
        return True

    return False


def get_next_candle_time():
    """Calculate the next 25-minute candle close time"""
    now = datetime.now()

    # Market opens at 9:15
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)

    # If before market open, return first candle close
    if now < market_open:
        return market_open + timedelta(minutes=CANDLE_INTERVAL_MINUTES)

    # Calculate minutes since market open
    minutes_since_open = (now - market_open).total_seconds() / 60

    # Calculate which candle we're in
    candles_passed = int(minutes_since_open // CANDLE_INTERVAL_MINUTES)

    # Next candle close time
    next_candle_close = market_open + timedelta(minutes=(candles_passed + 1) * CANDLE_INTERVAL_MINUTES)

    # Add 30 seconds buffer to ensure candle is fully closed
    return next_candle_close + timedelta(seconds=CANDLE_COMPLETION_BUFFER)


# === Main Entry Point ===
if __name__ == '__main__':
    load_dotenv()

    # Load credentials
    CLIENT_ID = os.getenv("DHAN_CLIENT_ID")
    ACCESS_TOKEN = os.getenv("DHAN_API_KEY")
    KOTAK_ACCESS_TOKEN = os.getenv("KOTAK_ACCESS_TOKEN")

    # Initialize Dhan
    dhan_object = dhanhq.dhanhq(CLIENT_ID, ACCESS_TOKEN)

    # Check command line arguments
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == 'backtest':
            print("Running in BACKTEST mode...")
            run_backtest_analysis()
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

            # 1. COVID Crash Period (Jan 2019 - Aug 2020)
            # save_historical_data(indices, start_date="2019-01-01", end_date="2020-08-31")
            #
            # # 2. Post-COVID Bull Run (Aug 2020 - Mar 2022)
            # save_historical_data(indices, start_date="2020-08-01", end_date="2022-03-31")
            #
            # # 3. 2022 Bear Market (Jun 2021 - Jan 2023)
            # save_historical_data(indices, start_date="2021-06-01", end_date="2023-01-31")
            #
            # # 4. Volatile Recovery (Apr 2022 - Nov 2023)
            # save_historical_data(indices, start_date="2022-04-01", end_date="2023-11-30")
            #
            # # 5. Recent Bull Market (Feb 2023 - Sep 2024)
            # save_historical_data(indices, start_date="2023-02-01", end_date="2024-09-30")
            #
            # # 6. Mixed Market Conditions (Oct 2018 - May 2020)
            # save_historical_data(indices, start_date="2018-10-01", end_date="2020-05-31")

            # Default - Now - 20 months
            # save_historical_data(indices, months=20)

            # Default - Now - 60 months
            save_historical_data(indices, months=60)

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