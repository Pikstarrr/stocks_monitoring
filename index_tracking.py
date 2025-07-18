# index_tracking.py - High accuracy trading strategy with signal persistence

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import firebase_admin
from firebase_admin import credentials, firestore
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google.cloud.firestore_v1 import ArrayUnion
from numba import jit, njit

import dhanhq
from send_mail import send_email
from strategy_features import compute_feature, rational_quadratic_kernel, gaussian_kernel
from trade_placement import KotakOptionsTrader

# === CONFIG ===

FEATURE_CONFIG = [
    ("RSI", 14, 1),  # Feature 1
    ("WT", 10, 11),  # Feature 2
    ("CCI", 20, 1),  # Feature 3
    ("ADX", 20, 2),  # Feature 4
    ("RSI", 9, 1),  # Feature 5
]

NEIGHBOR_COUNT = 8
LUCKY_NUMBER = 3
MAX_BARS_BACK = 5000
LOOKAHEAD = 4
STRICT_EXIT_BARS = 25

LABEL_THRESHOLD = 0.001
CONFIDENCE_THRESHOLD = 1

USE_KERNEL_FILTER = True
USE_KERNEL_SMOOTHING = True
KERNEL_H = 8
KERNEL_R = 8.0
KERNEL_X = 25
KERNEL_LAG = 2

USE_VOLATILITY_FILTER = True
USE_REGIME_FILTER = True
REGIME_THRESHOLD = -0.1
USE_ADX_FILTER = False
ADX_THRESHOLD = 20
USE_EMA_FILTER = False
EMA_PERIOD = 200
USE_SMA_FILTER = False
SMA_PERIOD = 200
USE_DYNAMIC_EXITS = True

SIGNAL_STATE_FILE = "signal_state.json"

# Logging flag for detailed output
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
                'price': float(entry_price),
                'time': entry_time.isoformat() if isinstance(entry_time, datetime) else str(entry_time)
            }
        elif position is None:
            self.state[symbol]['entry_info'] = {}
        self.save_state()

    def update_signal(self, symbol, signal):
        if symbol not in self.state:
            self.state[symbol] = {}
        self.state[symbol]['last_signal'] = int(signal)
        self.save_state()

    def increment_bars_held(self, symbol):
        if symbol not in self.state:
            self.state[symbol] = {}
        self.state[symbol]['bars_held'] = self.state.get(symbol, {}).get('bars_held', 0) + 1
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
        self.trades.append({
            'timestamp': timestamp,
            'index': index_name,
            'action': signal_type,
            'price': price
        })
        if signal_type in ['BUY', 'SHORT']:
            self.current_position = signal_type
            self.entry_price = price
            self.entry_time = timestamp
        elif signal_type in ['SELL', 'COVER']:
            if self.current_position:
                profit = price - self.entry_price if self.current_position == 'BUY' else self.entry_price - price
                self.positions.append({
                    'index': index_name,
                    'entry_time': self.entry_time,
                    'exit_time': timestamp,
                    'entry_price': self.entry_price,
                    'exit_price': price,
                    'type': self.current_position,
                    'profit': profit,
                    'profit_pct': (profit / self.entry_price) * 100
                })
                self.current_position = None


# === Optimized Feature and Indicator Calculations ===

@njit
def incremental_compute_rsi(prices, period):
    if len(prices) < period + 1:
        return 50.0  # Default if insufficient data
    deltas = np.diff(prices)
    up = np.maximum(deltas, 0)
    down = np.maximum(-deltas, 0)
    avg_up = np.mean(up[-period:])
    avg_down = np.mean(down[-period:])
    rs = avg_up / avg_down if avg_down != 0 else 0
    return 100 - 100 / (1 + rs)


def build_features_incremental(df, prev_features=None):
    """Build features incrementally"""
    if prev_features is None or len(prev_features) == 0:
        features = []
        for kind, param_a, param_b in FEATURE_CONFIG:
            f = compute_feature(df.copy(), kind, param_a, param_b)
            f.name = f"{kind}_{param_a}_{param_b}"
            features.append(f)
        return pd.concat(features, axis=1).dropna()
    else:
        new_row = pd.DataFrame(index=[df.index[-1]])
        for i, (kind, param_a, param_b) in enumerate(FEATURE_CONFIG):
            if kind == "RSI":
                new_val = incremental_compute_rsi(df['close'].values[- (param_a + 1):], param_a)
            else:
                recent_df = df.iloc[-max(param_a, param_b, 50):].copy()  # Buffer for safety
                new_val = compute_feature(recent_df, kind, param_a, param_b).iloc[-1]
            new_row[f"{kind}_{param_a}_{param_b}"] = new_val
        return pd.concat([prev_features, new_row])


@njit
def calculate_adx(high, low, close, period=14):
    if len(high) < period * 2:
        return np.zeros(len(high), dtype=np.float64)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    high_diff = high[1:] - high[:-1]
    low_diff = low[:-1] - low[1:]
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)

    def rma(arr, p):
        alpha = 1 / p
        res = np.zeros_like(arr, dtype=np.float64)
        res[0] = arr[0]
        for i in range(1, len(arr)):
            res[i] = alpha * arr[i] + (1 - alpha) * res[i - 1]
        return res

    atr = rma(tr, period)
    plus_di = 100 * rma(plus_dm, period) / atr
    minus_di = 100 * rma(minus_dm, period) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = rma(dx, period)
    pad = np.zeros(1, dtype=np.float64)
    return np.concatenate((pad, adx))


@njit
def calculate_atr(high, low, close, period=14):
    if len(high) < period + 1:
        return np.zeros(len(high), dtype=np.float64)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))[1:]
    atr = np.zeros(len(tr), dtype=np.float64)
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, len(tr)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    pad = np.zeros(len(high) - len(atr), dtype=np.float64)
    return np.concatenate((pad, atr))


def get_kernel_signals(series, h=KERNEL_H, r=KERNEL_R, x=KERNEL_X, lag=KERNEL_LAG):
    yhat1 = rational_quadratic_kernel(series, h, r, x)
    yhat2 = gaussian_kernel(series, h - lag, x)
    if USE_KERNEL_SMOOTHING:
        bullish = yhat2 >= yhat1
        bearish = yhat2 <= yhat1
    else:
        if len(yhat1) < 3:
            return pd.Series([False] * len(series), index=series.index), pd.Series([False] * len(series),
                                                                                   index=series.index), yhat1, yhat2
        wasBearishRate = yhat1.shift(2) > yhat1.shift(1)
        wasBullishRate = yhat1.shift(2) < yhat1.shift(1)
        isBearishRate = yhat1.shift(1) > yhat1
        isBullishRate = yhat1.shift(1) < yhat1
        bullish = isBullishRate & wasBearishRate
        bearish = isBearishRate & wasBullishRate
    return bullish, bearish, yhat1, yhat2


def calculate_volatility_filter(close, use_filter=True):
    if not use_filter:
        return True
    df_temp = pd.DataFrame({'high': close, 'low': close, 'close': close})
    recent_atr = calculate_atr(df_temp['high'].values, df_temp['low'].values, df_temp['close'].values, period=1)[-1]
    historical_atr = calculate_atr(df_temp['high'].values, df_temp['low'].values, df_temp['close'].values, period=10)[
        -1]
    return recent_atr > historical_atr


def calculate_regime_filter(close, threshold=REGIME_THRESHOLD, use_filter=True):
    if not use_filter:
        return True
    lookback = 50
    if len(close) < lookback:
        return True
    x = np.arange(lookback)
    y = close.iloc[-lookback:].values
    y_mean = np.mean(y)
    if y_mean != 0:
        y_normalized = (y - y_mean) / y_mean
    else:
        y_normalized = y
    slope = np.polyfit(x, y_normalized, 1)[0]
    return slope > threshold


def check_dynamic_exit_conditions(df, position, bars_held):
    if not USE_DYNAMIC_EXITS:
        return False
    if len(df) < 10:
        return False
    current_idx = len(df) - 1
    _, _, yhat1, yhat2 = get_kernel_signals(df['close'])
    momentum = df['close'].pct_change(5).iloc[-1]
    if USE_KERNEL_SMOOTHING:
        if position == 'LONG':
            bearish_cross = (
                        yhat2.iloc[current_idx] < yhat1.iloc[current_idx] and yhat2.iloc[current_idx - 1] >= yhat1.iloc[
                    current_idx - 1])
            momentum_exit = momentum < -0.002
            return (bearish_cross or momentum_exit) and bars_held > 0
        elif position == 'SHORT':
            bullish_cross = (
                        yhat2.iloc[current_idx] > yhat1.iloc[current_idx] and yhat2.iloc[current_idx - 1] <= yhat1.iloc[
                    current_idx - 1])
            momentum_exit = momentum > 0.002
            return (bullish_cross or momentum_exit) and bars_held > 0
    return False


def calculate_entry_quality(df, signal_type):
    recent_moves = df['close'].pct_change().iloc[-10:].abs()
    avg_move = recent_moves.mean() * 100
    has_volatility = avg_move >= 0.05
    momentum_1 = (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2] * 100 if len(df) >= 2 else 0
    momentum_3 = (df['close'].iloc[-1] - df['close'].iloc[-4]) / df['close'].iloc[-4] * 100 if len(df) >= 4 else 0
    if signal_type == 'LONG':
        momentum_aligned = momentum_1 > 0 and momentum_3 > 0.05
    else:
        momentum_aligned = momentum_1 < 0 and momentum_3 < -0.05
    recent_ranges = ((df['high'] - df['low']) / df['low'] * 100).iloc[-5:]
    can_move = recent_ranges.max() >= 0.15 if not recent_ranges.empty else False
    _, _, yhat1, _ = get_kernel_signals(df['close'])
    if len(yhat1) >= 3:
        kernel_accel = yhat1.iloc[-1] - yhat1.iloc[-3]
        kernel_accelerating = kernel_accel > 0 if signal_type == 'LONG' else kernel_accel < 0
    else:
        kernel_accelerating = True
    return has_volatility and momentum_aligned and can_move and kernel_accelerating


def calculate_market_structure(df):
    if len(df) < 200:
        return 'NEUTRAL'
    sma20 = df['close'].rolling(20).mean()
    sma50 = df['close'].rolling(50).mean()
    sma200 = df['close'].rolling(200).mean()
    bullish_structure = (df['close'].iloc[-1] > sma20.iloc[-1] > sma50.iloc[-1] > sma200.iloc[-1])
    bearish_structure = (df['close'].iloc[-1] < sma20.iloc[-1] < sma50.iloc[-1] < sma200.iloc[-1])
    if bullish_structure:
        ma_spread = (sma20.iloc[-1] - sma200.iloc[-1]) / sma200.iloc[-1] * 100
        return 'STRONG_BULLISH' if ma_spread > 0.5 else 'BULLISH'
    elif bearish_structure:
        ma_spread = (sma200.iloc[-1] - sma20.iloc[-1]) / sma200.iloc[-1] * 100
        return 'STRONG_BEARISH' if ma_spread > 0.5 else 'BEARISH'
    return 'NEUTRAL'


# === Optimized Prediction Logic ===

@njit
def process_single_prediction_pine(X, Y, i, max_lookback, neighbor_count):
    lastDistance = -1.0
    distances = np.zeros(neighbor_count)
    votes = np.zeros(neighbor_count, dtype=np.int32)
    count = 0
    size = min(max_lookback, i)
    for loop_idx in range(size):
        if loop_idx % 4 == 0:
            continue
        hist_idx = i - size + loop_idx
        if hist_idx >= i:
            continue
        d = 0.0
        for k in range(X.shape[1]):
            d += np.log(1 + np.abs(X[i, k] - X[hist_idx, k]))
        if d >= lastDistance:
            lastDistance = d
            if count < neighbor_count:
                distances[count] = d
                votes[count] = Y[hist_idx]
                count += 1
            else:
                for idx in range(neighbor_count - 1):
                    distances[idx] = distances[idx + 1]
                    votes[idx] = votes[idx + 1]
                distances[neighbor_count - 1] = d
                votes[neighbor_count - 1] = Y[hist_idx]
            lastDistance = distances[int(neighbor_count * 3 / 4)]
    return np.sum(votes[:count]), count


@jit(nopython=True)
def predict_labels(X, close, lookahead=LOOKAHEAD, neighbor_count=NEIGHBOR_COUNT, max_bars_back=MAX_BARS_BACK,
                   label_threshold=LABEL_THRESHOLD, confidence_threshold=CONFIDENCE_THRESHOLD,
                   lucky_number=LUCKY_NUMBER):
    n = len(X)
    Y = np.zeros(n, dtype=np.int32)
    for i in range(n - lookahead):
        delta = (close[i + lookahead] - close[i]) / close[i]
        if delta > label_threshold:
            Y[i] = 1
        elif delta < -label_threshold:
            Y[i] = -1
    predictions = np.zeros(n, dtype=np.int32)
    for i in range(lookahead, n):
        max_lookback = min(i, max_bars_back)
        if max_lookback < neighbor_count:
            continue
        vote_sum, vote_count = process_single_prediction_pine(X, Y, i, max_lookback, neighbor_count)
        min_consensus_ratio = 0.6
        if vote_count >= lucky_number:
            consensus_ratio = abs(vote_sum) / vote_count
            if abs(vote_sum) >= confidence_threshold and consensus_ratio >= min_consensus_ratio:
                predictions[i] = int(np.sign(vote_sum))
    return predictions


@jit(nopython=True)
def predict_labels_single(X, close, current_idx, lookahead=LOOKAHEAD, neighbor_count=NEIGHBOR_COUNT,
                          max_bars_back=MAX_BARS_BACK,
                          label_threshold=LABEL_THRESHOLD, confidence_threshold=CONFIDENCE_THRESHOLD,
                          lucky_number=LUCKY_NUMBER):
    n = len(X)
    Y = np.zeros(n, dtype=np.int32)
    for i in range(n - lookahead):
        delta = (close[i + lookahead] - close[i]) / close[i]
        if delta > label_threshold:
            Y[i] = 1
        elif delta < -label_threshold:
            Y[i] = -1
    # Predict only for current_idx
    max_lookback = min(current_idx, max_bars_back)
    if max_lookback < neighbor_count:
        return 0
    vote_sum, vote_count = process_single_prediction_pine(X, Y, current_idx, max_lookback, neighbor_count)
    min_consensus_ratio = 0.6
    if vote_count >= lucky_number:
        consensus_ratio = abs(vote_sum) / vote_count
        if abs(vote_sum) >= confidence_threshold and consensus_ratio >= min_consensus_ratio:
            return int(np.sign(vote_sum))
    return 0


# === Processing Logic (Unified for Live and Backtest) ===

def process_symbol(symbol_dict, signal_state, trader, quote_data=None, mode='live', trade_recorder=None, df=None,
                   prev_features=None, precalc_data=None):
    symbol, index = next(iter(symbol_dict.items()))
    if mode == 'live':
        if DEBUG:
            print(f"[LIVE] Fetching OHLC for {index}...")
        df = fetch_ohlc(symbol, interval=15)
        if quote_data and 'last_price' in quote_data:
            now = pd.Timestamp.now().floor("15min")
            if now not in df.index:
                ltp = float(quote_data['last_price'])
                new_row = pd.DataFrame({'open': ltp, 'high': ltp, 'low': ltp, 'close': ltp}, index=[now])
                df = pd.concat([df, new_row]).sort_index()
                if DEBUG:
                    print(f"[LIVE] Appended live candle for {index} at {now}")
    else:
        if DEBUG:
            print(f"[BACKTEST] Processing candle for {index}...")
        pass  # df is passed in backtest

    if len(df) > MAX_BARS_BACK:
        df = df.iloc[-MAX_BARS_BACK:]

    if precalc_data:
        features = precalc_data['features']
        current_signal = precalc_data['prediction']
        current_bullish = precalc_data['bullish']
        current_bearish = precalc_data['bearish']
        atr = precalc_data['atr']  # Full series, but we use [-1]
        current_atr = atr[-1] if len(atr) > 0 else 0
        if DEBUG:
            print(f"[BACKTEST] Using precalculated data for {index}")
    else:
        features = build_features_incremental(df, prev_features)
        if len(features) < LOOKAHEAD:
            if DEBUG:
                print(f"[INFO] Insufficient features for {index}, skipping...")
            return prev_features if prev_features is not None else None
        predictions = predict_labels(features.values, df['close'].values)
        current_signal = predictions[-1]
        bullish, bearish, _, _ = get_kernel_signals(df['close'])
        current_bullish = bullish.iloc[-1]
        current_bearish = bearish.iloc[-1]
        atr = calculate_atr(df['high'].values, df['low'].values, df['close'].values)
        current_atr = atr[-1] if len(atr) > 0 else 0
        if DEBUG:
            print(f"[INFO] Computed fresh data for {index}")

    current_time = df.index[-1]
    current_price = df['close'].iloc[-1]

    current_position = signal_state.get_position(index)
    bars_held = signal_state.get_bars_held(index)

    current_vol_filter = calculate_volatility_filter(df['close'], USE_VOLATILITY_FILTER)
    current_regime_filter = calculate_regime_filter(df['close'], REGIME_THRESHOLD, USE_REGIME_FILTER)

    if USE_ADX_FILTER:
        adx = calculate_adx(df['high'].values, df['low'].values, df['close'].values, period=14)
        current_adx_filter = adx[-1] > ADX_THRESHOLD
    else:
        current_adx_filter = True

    if USE_EMA_FILTER:
        ema = df['close'].ewm(span=EMA_PERIOD, adjust=False).mean()
        ema_uptrend = current_price > ema.iloc[-1]
        ema_downtrend = current_price < ema.iloc[-1]
    else:
        ema_uptrend = ema_downtrend = True

    if USE_SMA_FILTER:
        sma = df['close'].rolling(window=SMA_PERIOD).mean()
        sma_uptrend = current_price > sma.iloc[-1]
        sma_downtrend = current_price < sma.iloc[-1]
    else:
        sma_uptrend = sma_downtrend = True

    all_filters = current_vol_filter and current_regime_filter and current_adx_filter
    long_filters = all_filters and ema_uptrend and sma_uptrend
    short_filters = all_filters and ema_downtrend and sma_downtrend

    previous_signal = signal_state.get_last_signal(index)
    if all_filters and current_signal != 0:
        persisted_signal = current_signal
    else:
        persisted_signal = previous_signal

    action_taken = None
    current_atr = atr[-1]

    if current_position == 'LONG':
        signal_state.increment_bars_held(index)
        bars_held = signal_state.get_bars_held(index)
        entry_info = signal_state.get_entry_info(index)
        entry_price = entry_info.get('price', current_price)
        profit_pct = ((current_price - entry_price) / entry_price) * 100
        profit_target_hit = profit_pct >= 0.1
        strict_exit = bars_held >= STRICT_EXIT_BARS
        opposite_signal = persisted_signal == -1 and current_bearish and short_filters
        dynamic_exit = check_dynamic_exit_conditions(df, 'LONG', bars_held)
        if profit_target_hit or strict_exit or opposite_signal or dynamic_exit:
            action_taken = "SELL"
            signal_state.update_position(index, None)
            signal_state.reset_bars_held(index)
            if DEBUG:
                print(f"[TRADE] Exiting LONG for {index}: {action_taken}")
    elif current_position == 'SHORT':
        signal_state.increment_bars_held(index)
        bars_held = signal_state.get_bars_held(index)
        entry_info = signal_state.get_entry_info(index)
        entry_price = entry_info.get('price', current_price)
        profit_pct = ((entry_price - current_price) / entry_price) * 100
        profit_target_hit = profit_pct >= 0.1
        strict_exit = bars_held >= STRICT_EXIT_BARS
        opposite_signal = persisted_signal == 1 and current_bullish and long_filters
        dynamic_exit = check_dynamic_exit_conditions(df, 'SHORT', bars_held)
        if profit_target_hit or strict_exit or opposite_signal or dynamic_exit:
            action_taken = "COVER"
            signal_state.update_position(index, None)
            signal_state.reset_bars_held(index)
            if DEBUG:
                print(f"[TRADE] Exiting SHORT for {index}: {action_taken}")

    if current_position is None:
        market_structure = calculate_market_structure(df)
        long_quality = calculate_entry_quality(df, 'LONG')
        short_quality = calculate_entry_quality(df, 'SHORT')
        can_go_long = market_structure in ['BULLISH', 'STRONG_BULLISH']
        can_go_short = market_structure in ['BEARISH', 'STRONG_BEARISH']
        if persisted_signal == 1 and current_bullish and long_filters and long_quality and can_go_long:
            action_taken = "BUY"
            signal_state.update_position(index, 'LONG', current_price, current_time)
            signal_state.reset_bars_held(index)
            if DEBUG:
                print(f"[TRADE] Entering LONG for {index}: {action_taken}")
        elif persisted_signal == -1 and current_bearish and short_filters and short_quality and can_go_short:
            action_taken = "SHORT"
            signal_state.update_position(index, 'SHORT', current_price, current_time)
            signal_state.reset_bars_held(index)
            if DEBUG:
                print(f"[TRADE] Entering SHORT for {index}: {action_taken}")

    signal_state.update_signal(index, persisted_signal)

    if action_taken:
        if mode == 'live':
            subject = f"{index} SIGNAL: {action_taken}"
            log_str = f"INDEX: {index}, Time: {current_time}, Signal: {persisted_signal}, Close: {current_price:.2f}, ATR: {current_atr:.2f}, Action: {action_taken}"
            send_email(subject, log_str)
            doc_ref = db.collection("live_alerts").document(index)
            doc_ref.update({"alerts": ArrayUnion([log_str])})
            trader.execute_single_trade(timestamp=current_time, index_name=index, signal_type=action_taken)
            print(f"[LIVE TRADE] Executed {action_taken} for {index}")
        elif mode == 'backtest' and trade_recorder:
            trade_recorder.record_trade(current_time, index, action_taken, current_price)
            if DEBUG:
                print(f"[BACKTEST TRADE] Recorded {action_taken} for {index}")

    return features  # Return for incremental use


# === Optimized Backtest ===

def run_backtest_analysis():
    print("\n" + "=" * 80)
    print("RUNNING OPTIMIZED BACKTEST ANALYSIS")
    print("=" * 80 + "\n")

    indices = [
        {"13": "NIFTY"},
        {"25": "BANKNIFTY"},
        {"51": "SENSEX"},
        {"27": "FINNIFTY"},
        {"442": "MIDCPNIFTY"}
    ]

    all_results = {}
    with ThreadPoolExecutor(max_workers=len(indices)) as executor:
        future_to_index = {}
        for idx in indices:
            future = executor.submit(process_index_backtest, idx)
            future_to_index[future] = idx

        for future in as_completed(future_to_index):
            index_name, recorder = future.result()
            if recorder.positions:
                total_trades = len(recorder.positions)
                winning_trades = sum(1 for p in recorder.positions if p['profit'] > 0)
                total_profit = sum(p['profit'] for p in recorder.positions)
                win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
                all_results[index_name] = {
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
                print(f"\n{index_name} Performance:")
                print(f" Total Trades: {total_trades}")
                print(f" Win Rate: {win_rate:.2f}%")
                print(f" Total Profit: {total_profit:.2f}")

    save_backtest_results(all_results)
    return all_results


def process_index_backtest(idx):
    symbol, index_name = next(iter(idx.items()))
    backtest_state = SignalState(filepath=f"backtest_signal_state_{index_name}.json")
    trade_recorder = TradeRecorder()

    if DEBUG:
        print(f"[BACKTEST] Starting backtest for {index_name}...")

    # Preload full dataset
    input_path = f'testing_data/{index_name}_input.csv'
    test_path = f'testing_data/{index_name}_test.csv'
    full_df = pd.read_csv(input_path, parse_dates=['datetime'], index_col='datetime')
    test_df = pd.read_csv(test_path, parse_dates=['datetime'], index_col='datetime')
    full_df = pd.concat([full_df, test_df]).sort_index()

    # Precompute everything possible once
    if DEBUG:
        print(f"[BACKTEST] Precomputing features and indicators for {index_name}...")
    full_features = build_features_incremental(full_df)  # Full features
    full_atr = calculate_atr(full_df['high'].values, full_df['low'].values, full_df['close'].values)
    full_bullish, full_bearish, _, _ = get_kernel_signals(full_df['close'])

    # Simulate incremental backtest with slicing
    test_start = len(full_df) - len(test_df)
    for i in range(max(test_start, LOOKAHEAD), len(full_df)):
        window_start = max(0, i - MAX_BARS_BACK + 1)
        window_features = full_features.iloc[window_start:i + 1]
        window_close = full_df['close'].iloc[window_start:i + 1].values
        precalc_data = {
            'features': window_features,
            'prediction': predict_labels_single(window_features.values, window_close, i - window_start),
            'bullish': full_bullish.iloc[i],
            'bearish': full_bearish.iloc[i],
            'atr': full_atr[window_start:i + 1]
        }
        sim_df = full_df.iloc[window_start:i + 1]
        process_symbol(idx, backtest_state, None, mode='backtest', trade_recorder=trade_recorder, df=sim_df,
                       precalc_data=precalc_data)
        if DEBUG and (i % 100 == 0):
            print(f"[BACKTEST] Processed {i - test_start} candles for {index_name}...")

    if os.path.exists(f"backtest_signal_state_{index_name}.json"):
        os.remove(f"backtest_signal_state_{index_name}.json")

    if DEBUG:
        print(f"[BACKTEST] Completed backtest for {index_name}")
    return index_name, trade_recorder


def save_backtest_results(all_results):
    output_dir = "backtest_results"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
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
        pd.DataFrame(summary_data).to_csv(f"{output_dir}/backtest_summary_{timestamp}.csv", index=False)
    if DEBUG:
        print(f"[BACKTEST] Results saved to {output_dir}/backtest_summary_{timestamp}.csv")


# === Live Trading Functions ===

def fetch_ohlc(symbol, interval=15, months=5):
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
    return df.sort_index()


def is_market_closed():
    now = datetime.now()
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return now >= market_close


def wait_for_market_open():
    while True:
        now = datetime.now()
        if now.weekday() > 4:
            print(f"[INFO] It's {now.strftime('%A')}. Market is closed on weekends.")
            return False
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        if market_open <= now <= market_close:
            print("[INFO] Market is open! Starting strategy...")
            return True
        if now > market_close:
            print("[INFO] Market has closed for today at 3:30 PM.")
            return False
        time_to_open = (market_open - now).total_seconds()
        hours = int(time_to_open // 3600)
        minutes = int((time_to_open % 3600) // 60)
        seconds = int(time_to_open % 60)
        print(f"[INFO] Market opens at 9:15 AM. Waiting {hours}h {minutes}m {seconds}s...")
        time.sleep(30)


def run_live_trading(signal_state, trader):
    if not wait_for_market_open():
        print("[INFO] Exiting as market won't open today.")
        return
    print("[INFO] ✅ Live trading started. Will run every 15 minutes until 3:30 PM IST.")
    next_run = get_next_candle_time()
    wait_time = (next_run - datetime.now()).total_seconds()
    if wait_time > 0:
        print(f"[INFO] First run at {next_run.strftime('%H:%M:%S')}. Waiting {int(wait_time)} seconds...")
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
            print("[INFO] ⏹️ Market closed at 3:30 PM. Stopping strategy.")
            break
        print(f"\n{'=' * 60}")
        print(f"[INFO] 🕒 Running strategy at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'=' * 60}\n")

        all_symbols = [int(list(idx.keys())[0]) for idx in indices]
        try:
            quote_response = dhan_object.quote_data({"IDX_I": all_symbols})
            quotes = quote_response['data']['data']['IDX_I'] if 'data' in quote_response and 'data' in quote_response[
                'data'] and 'IDX_I' in quote_response['data']['data'] else {}
            if DEBUG:
                print("[LIVE] Fetched quotes successfully")
        except Exception as e:
            print(f"[ERROR] Error fetching quotes: {e}")
            quotes = {}

        threads = []
        for index_dict in indices:
            def process_live_index(index_dict, quotes, signal_state, trader):
                try:
                    symbol, index = next(iter(index_dict.items()))
                    process_symbol(index_dict, signal_state, trader, quotes.get(symbol), mode='live')
                    if DEBUG:
                        print(f"[LIVE] Processed {index} successfully")
                except Exception as e:
                    print(f"[ERROR] Error processing {index}: {str(e)}")

            thread = threading.Thread(target=process_live_index, args=(index_dict, quotes, signal_state, trader))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        next_run = get_next_candle_time()
        sleep_duration = (next_run - datetime.now()).total_seconds()
        if sleep_duration > 0:
            print(
                f"\n[INFO] 🕒 Next run at {next_run.strftime('%H:%M:%S')}. Sleeping for {int(sleep_duration)} seconds...\n")
            time.sleep(sleep_duration)


def get_next_candle_time(interval=15):
    now = datetime.now()
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    if now < market_open:
        return market_open + timedelta(minutes=interval)
    minutes_since_open = (now - market_open).total_seconds() / 60
    candles_passed = int(minutes_since_open // interval)
    next_candle_close = market_open + timedelta(minutes=(candles_passed + 1) * interval)
    return next_candle_close + timedelta(seconds=30)


# === Main Entry Point ===

if __name__ == '__main__':
    load_dotenv()
    CLIENT_ID = os.getenv("DHAN_CLIENT_ID")
    ACCESS_TOKEN = os.getenv("DHAN_API_KEY")
    KOTAK_ACCESS_TOKEN = os.getenv("KOTAK_ACCESS_TOKEN")

    dhan_object = dhanhq.dhanhq(CLIENT_ID, ACCESS_TOKEN)

    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == 'backtest':
            print("[INFO] Running in BACKTEST mode...")
            run_backtest_analysis()
        elif sys.argv[1] == 'test_live':
            print("[INFO] Running in TEST LIVE mode...")
            trader = KotakOptionsTrader(test_mode=True)
            firebase_admin.initialize_app(credentials.Certificate("stock-monitoring-fb.json"))
            db = firestore.client()
            signal_state = SignalState()
            run_live_trading(signal_state, trader)
        else:
            print("[INFO] Running in LIVE mode...")
            trader = KotakOptionsTrader(test_mode=False)
            status = trader.get_account_status()
            if not status['logged_in']:
                print("[ERROR] Failed to login to Kotak. Please check your access token.")
                sys.exit(1)
            firebase_admin.initialize_app(credentials.Certificate("stock-monitoring-fb.json"))
            db = firestore.client()
            signal_state = SignalState()
            run_live_trading(signal_state, trader)
    else:
        print("[INFO] Running in LIVE mode (default)...")
        trader = KotakOptionsTrader(test_mode=False)
        status = trader.get_account_status()
        if not status['logged_in']:
            print("[ERROR] Failed to login to Kotak. Please check your access token.")
            sys.exit(1)
        firebase_admin.initialize_app(credentials.Certificate("stock-monitoring-fb.json"))
        db = firestore.client()
        signal_state = SignalState()
        run_live_trading(signal_state, trader)
