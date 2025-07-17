#
# Momentum X - Integrated Trading & Backtesting Framework
#
# This script combines a robust trading framework with a high-fidelity Python
# implementation of the "Momentum X" Pine Script indicator.
#
# Original Framework: options_backtest_strat3.py by the user
# Signal Engine: MomentumX class by Gemini
#
# Date of Integration: July 17, 2025
#
import time
import time
import json
import os
import sys
from datetime import datetime, timedelta
from collections import deque

from numba import jit
from collections import deque

from firebase_admin import credentials, firestore, initialize_app
import pandas as pd
import numpy as np
import pandas_ta as ta
from dotenv import load_dotenv
from google.cloud.firestore_v1 import ArrayUnion

# It's better to import specific modules if they are known
from send_mail import send_email
import dhanhq
from trade_placement import KotakOptionsTrader

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# This is the modern way to handle the SettingWithCopyWarning if needed,
# though the script is written to avoid it by using .copy().
pd.options.mode.chained_assignment = None  # Suppress the warning globally


# ==============================================================================
# ==== MomentumX Signal Generation Engine (The "Brain") ====
# ==============================================================================

class MomentumX:
    """
    A Python implementation of the 'Momentum X' TradingView indicator.
    Fixed to exactly match Pine Script logic.
    """

    def __init__(self, df, **kwargs):
        """
        Initializes the MomentumX strategy with data and parameters.
        """
        self.df = df.copy()

        # Default parameters exactly matching Pine Script
        self.params = {
            "source": "close",
            "neighborsCount": 8,
            "maxBarsBack": 5000,
            "featureCount": 5,
            "colorCompression": 1,
            "useDynamicExits": False,
            "f1_string": "RSI", "f1_paramA": 14, "f1_paramB": 1,
            "f2_string": "WT", "f2_paramA": 10, "f2_paramB": 11,
            "f3_string": "CCI", "f3_paramA": 20, "f3_paramB": 1,
            "f4_string": "ADX", "f4_paramA": 20, "f4_paramB": 2,
            "f5_string": "RSI", "f5_paramA": 9, "f5_paramB": 1,
            "useVolatilityFilter": True,
            "useRegimeFilter": True, "regimeThreshold": -0.1,
            "useAdxFilter": False, "adxThreshold": 20,
            "useEmaFilter": False, "emaPeriod": 200,
            "useSmaFilter": False, "smaPeriod": 200,
            "useKernelFilter": True,
            "useKernelSmoothing": True,
            "h_lookback": 8,
            "r_relative_weight": 8.0,
            "x_regression_level": 25,
            "lag": 2,
        }
        self.params.update(kwargs)

    def _normalize_series(self, series, min_val=-1, max_val=1):
        """Min-Max normalization to scale a series to a given range."""
        if not isinstance(series, pd.Series):
            series = pd.Series(series, index=self.df.index)
        rolling_max = series.rolling(window=self.params['maxBarsBack'], min_periods=1).max()
        rolling_min = series.rolling(window=self.params['maxBarsBack'], min_periods=1).min()
        range_val = rolling_max - rolling_min
        range_val[range_val == 0] = 1
        normalized = min_val + (series - rolling_min) * (max_val - min_val) / range_val
        return normalized.fillna(0)

    def _wavetrend(self, src, chlen, avglen):
        """Calculates the WaveTrend Oscillator."""
        hlc3 = ta.hlc3(self.df['high'], self.df['low'], self.df['close'])
        esa = ta.ema(hlc3, length=chlen)
        d = ta.ema(abs(hlc3 - esa), length=chlen)
        ci = (hlc3 - esa) / (0.015 * d)
        tci = ta.ema(ci, length=avglen)
        return tci

    def _series_from(self, feature_string, paramA, paramB):
        """Calculates and normalizes a feature series based on user input."""
        src_series = self.df[self.params['source']]
        if feature_string == "RSI":
            raw_series = ta.rsi(src_series, length=paramA)
        elif feature_string == "WT":
            raw_series = self._wavetrend(src_series, paramA, paramB)
        elif feature_string == "CCI":
            raw_series = ta.cci(self.df['high'], self.df['low'], src_series, length=paramA)
        elif feature_string == "ADX":
            adx_df = ta.adx(self.df['high'], self.df['low'], src_series, length=paramA)
            raw_series = adx_df[f'ADX_{paramA}']
        else:
            raw_series = pd.Series(np.zeros(len(self.df)), index=self.df.index)
        return self._normalize_series(raw_series.fillna(0))

    def _rma(self, series, period):
        """Pine Script's RMA (Running Moving Average)"""
        alpha = 1.0 / period
        return series.ewm(alpha=alpha, adjust=False).mean()

    def _kernel_regression(self, series, h, r, x, is_gaussian=False, lag=0):
        """Nadaraya-Watson Kernel Regression implementation."""
        yhat = pd.Series(np.nan, index=self.df.index)
        series_np = series.to_numpy()

        for i in range(h, len(series_np)):
            weights_sum, weighted_sum = 0, 0
            for j in range(max(0, i - h + 1), i + 1):
                if is_gaussian:
                    weight = np.exp(-((i - j) ** 2) / (2 * h ** 2))
                else:
                    weight = (1 + ((i - j) ** 2) / ((r ** 2) * (h ** 2))) ** (-r)
                weights_sum += weight
                weighted_sum += series_np[j] * weight
            if weights_sum != 0:
                yhat.iloc[i] = weighted_sum / weights_sum
            else:
                yhat.iloc[i] = series_np[i]
        return yhat.bfill()

    @staticmethod
    @jit(nopython=True)
    def _process_single_prediction(X, Y, i, max_lookback, neighbor_count):
        """Optimized Lorentzian distance k-NN matching Pine Script exactly"""
        lastDistance = -1.0
        distances = np.zeros(neighbor_count)
        votes = np.zeros(neighbor_count, dtype=np.int32)
        count = 0

        size = min(max_lookback - 1, i)

        for j in range(size + 1):
            if j % 4 == 0:  # Pine Script skips every 4th bar
                continue

            hist_idx = i - size + j - 1
            if hist_idx >= i or hist_idx < 0:
                continue

            # Lorentzian distance
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
                    # Shift arrays (Pine Script behavior)
                    for idx in range(neighbor_count - 1):
                        distances[idx] = distances[idx + 1]
                        votes[idx] = votes[idx + 1]
                    distances[neighbor_count - 1] = d
                    votes[neighbor_count - 1] = Y[hist_idx]

                    # Update threshold to 75th percentile
                    lastDistance = distances[int(neighbor_count * 3 / 4)]

        return np.sum(votes[:count])

    def run(self):
        """Executes the full strategy logic."""
        self._calculate_features()
        self._calculate_training_labels()
        self._calculate_predictions()
        self._apply_filters()
        self._calculate_entries_exits()
        return self.df

    def _calculate_features(self):
        """Calculate all 5 features"""
        self.df['f1'] = self._series_from(self.params['f1_string'], self.params['f1_paramA'], self.params['f1_paramB'])
        self.df['f2'] = self._series_from(self.params['f2_string'], self.params['f2_paramA'], self.params['f2_paramB'])
        self.df['f3'] = self._series_from(self.params['f3_string'], self.params['f3_paramA'], self.params['f3_paramB'])
        self.df['f4'] = self._series_from(self.params['f4_string'], self.params['f4_paramA'], self.params['f4_paramB'])
        self.df['f5'] = self._series_from(self.params['f5_string'], self.params['f5_paramA'], self.params['f5_paramB'])

    def _calculate_training_labels(self):
        """Calculate training labels (4 bars ahead)"""
        src = self.df[self.params['source']]
        # Pine Script: src[4] < src[0] ? direction.short : src[4] > src[0] ? direction.long : direction.neutral
        self.df['y_train'] = 0  # Initialize as neutral
        for i in range(len(self.df) - 4):
            future_price = src.iloc[i + 4]
            current_price = src.iloc[i]
            if future_price < current_price:
                self.df.iloc[i, self.df.columns.get_loc('y_train')] = -1  # Short
            elif future_price > current_price:
                self.df.iloc[i, self.df.columns.get_loc('y_train')] = 1  # Long

    def _calculate_predictions(self):
        """Calculate ML predictions using optimized k-NN"""
        features = self.df[['f1', 'f2', 'f3', 'f4', 'f5']].values
        labels = self.df['y_train'].values.astype(np.int32)
        predictions = np.zeros(len(self.df), dtype=np.int32)

        for i in range(len(self.df)):
            if i < self.params['neighborsCount']:
                predictions[i] = 0
                continue

            max_lookback = min(self.params['maxBarsBack'], i)
            vote_sum = self._process_single_prediction(
                features, labels, i, max_lookback, self.params['neighborsCount']
            )
            predictions[i] = vote_sum

        self.df['prediction'] = predictions

    def _apply_filters(self):
        """Apply all filters matching Pine Script exactly"""
        # Volatility Filter
        atr1 = ta.atr(self.df['high'], self.df['low'], self.df['close'], length=1)
        atr10 = ta.atr(self.df['high'], self.df['low'], self.df['close'], length=10)
        self.df['filter_volatility'] = (atr1 > atr10) if self.params['useVolatilityFilter'] else True

        # Regime Filter
        if self.params['useRegimeFilter']:
            ohlc4 = ta.ohlc4(self.df['open'], self.df['high'], self.df['low'], self.df['close'])
            lookback = 50  # Pine Script default
            x = np.arange(lookback)
            slopes = []

            for i in range(lookback, len(ohlc4) + 1):
                y = ohlc4.iloc[i - lookback:i].values
                y_mean = np.mean(y)
                if y_mean != 0:
                    y_normalized = (y - y_mean) / y_mean
                    slope = np.polyfit(x, y_normalized, 1)[0]
                else:
                    slope = 0
                slopes.append(slope)

            slope_series = pd.Series([0] * lookback + slopes, index=self.df.index)
            self.df['filter_regime'] = slope_series > self.params['regimeThreshold']
        else:
            self.df['filter_regime'] = True

        # ADX Filter
        if self.params['useAdxFilter']:
            adx_df = ta.adx(self.df['high'], self.df['low'], self.df['close'], length=14)
            self.df['filter_adx'] = adx_df['ADX_14'] > self.params['adxThreshold']
        else:
            self.df['filter_adx'] = True

        # Combined filters
        self.df['filter_all'] = self.df['filter_volatility'] & self.df['filter_regime'] & self.df['filter_adx']

        # EMA/SMA Filters
        if self.params['useEmaFilter']:
            ema = ta.ema(self.df['close'], self.params['emaPeriod'])
            self.df['ema_uptrend'] = self.df['close'] > ema
            self.df['ema_downtrend'] = self.df['close'] < ema
        else:
            self.df['ema_uptrend'] = True
            self.df['ema_downtrend'] = True

        if self.params['useSmaFilter']:
            sma = ta.sma(self.df['close'], self.params['smaPeriod'])
            self.df['sma_uptrend'] = self.df['close'] > sma
            self.df['sma_downtrend'] = self.df['close'] < sma
        else:
            self.df['sma_uptrend'] = True
            self.df['sma_downtrend'] = True

        # Kernel calculations
        src_series = self.df[self.params['source']]
        self.df['yhat1'] = self._kernel_regression(src_series, self.params['h_lookback'],
                                                   self.params['r_relative_weight'], self.params['x_regression_level'])
        self.df['yhat2'] = self._kernel_regression(src_series, self.params['h_lookback'] - self.params['lag'],
                                                   self.params['r_relative_weight'], self.params['x_regression_level'],
                                                   is_gaussian=True)

        # Kernel filters
        if self.params['useKernelSmoothing']:
            self.df['is_bullish'] = self.df['yhat2'] >= self.df['yhat1']
            self.df['is_bearish'] = self.df['yhat2'] <= self.df['yhat1']
        else:
            # Rate-based (slope)
            yhat1_prev = self.df['yhat1'].shift(1)
            self.df['is_bullish'] = self.df['yhat1'] > yhat1_prev
            self.df['is_bearish'] = self.df['yhat1'] < yhat1_prev

        if not self.params['useKernelFilter']:
            self.df['is_bullish'] = True
            self.df['is_bearish'] = True

    def _calculate_entries_exits(self):
        """Calculate entries and exits matching Pine Script exactly"""
        # 1. Signal persistence logic (Pine Script: nz(signal[1]))
        raw_predictions = self.df['prediction'].values
        filter_all = self.df['filter_all'].values
        signal = np.zeros(len(self.df), dtype=np.int32)

        for i in range(len(self.df)):
            if i == 0:
                signal[i] = 0
            else:
                # Pine Script: prediction > 0 and filter_all ? direction.long :
                #              prediction < 0 and filter_all ? direction.short : nz(signal[1])
                if raw_predictions[i] > 0 and filter_all[i]:
                    signal[i] = 1
                elif raw_predictions[i] < 0 and filter_all[i]:
                    signal[i] = -1
                else:
                    signal[i] = signal[i - 1]  # Persist previous signal

        self.df['signal'] = signal

        # 2. Calculate bars held
        bars_held = np.zeros(len(self.df), dtype=np.int32)
        signal_changed = np.diff(signal, prepend=signal[0]) != 0

        for i in range(len(self.df)):
            if signal_changed[i]:
                bars_held[i] = 0
            else:
                bars_held[i] = bars_held[i - 1] + 1 if i > 0 else 0

        self.df['barsHeld'] = bars_held

        # 3. Entry signals (Pine Script logic)
        is_new_signal = signal_changed
        is_buy_signal = (signal == 1) & self.df['ema_uptrend'] & self.df['sma_uptrend']
        is_sell_signal = (signal == -1) & self.df['ema_downtrend'] & self.df['sma_downtrend']

        self.df['startLongTrade'] = is_new_signal & is_buy_signal & self.df['is_bullish']
        self.df['startShortTrade'] = is_new_signal & is_sell_signal & self.df['is_bearish']

        # 4. Exit signals (Pine Script logic - simplified)
        if not self.params['useDynamicExits']:
            # Strict exits only
            is_held_four_bars = bars_held == 4
            is_held_less_than_four = (bars_held > 0) & (bars_held < 4)

            # Check what signal was 4 bars ago
            signal_4_bars_ago = pd.Series(signal).shift(4).fillna(0).values
            start_long_4_bars_ago = self.df['startLongTrade'].shift(4).fillna(False)
            start_short_4_bars_ago = self.df['startShortTrade'].shift(4).fillna(False)

            # Pine Script exit logic
            self.df['endLongTrade'] = (
                    ((is_held_four_bars & (signal_4_bars_ago == 1)) |
                     (is_held_less_than_four & is_new_signal & (signal == -1) & (signal_4_bars_ago == 1))) &
                    start_long_4_bars_ago
            )

            self.df['endShortTrade'] = (
                    ((is_held_four_bars & (signal_4_bars_ago == -1)) |
                     (is_held_less_than_four & is_new_signal & (signal == 1) & (signal_4_bars_ago == -1))) &
                    start_short_4_bars_ago
            )
        else:
            # Dynamic exits would go here
            self.df['endLongTrade'] = False
            self.df['endShortTrade'] = False


# ==============================================================================
# ==== Trading Framework (The "Chassis") ====
# ==============================================================================

# === CONFIG (User's Original Framework Settings) ===
SIGNAL_STATE_FILE = "signal_state.json"
DEBUG = True
CANDLE_INTERVAL_MINUTES = 15
CANDLE_COMPLETION_BUFFER = 5
BACKTEST_SLIPPAGE = 0.02
MAX_BARS_BACK_FRAMEWORK = 5000  # Max history to keep in memory for framework


class SignalState:
    """Manages signal persistence across runs"""

    def __init__(self, filepath=SIGNAL_STATE_FILE):
        self.filepath = filepath
        self.state = self.load_state()

    def load_state(self):
        if os.path.exists(self.filepath):
            with open(self.filepath, 'r') as f: return json.load(f)
        return {}

    def save_state(self):
        with open(self.filepath, 'w') as f: json.dump(self.state, f)

    def get_position(self, symbol):
        return self.state.get(symbol, {}).get('position', None)

    def update_position(self, symbol, position, entry_price=None, entry_time=None):
        if symbol not in self.state: self.state[symbol] = {}
        self.state[symbol]['position'] = position
        if position is not None and entry_price is not None:
            self.state[symbol]['entry_info'] = {
                'price': float(entry_price),
                'time': entry_time.isoformat() if hasattr(entry_time, 'isoformat') else str(entry_time)
            }
        elif position is None:
            self.state[symbol]['entry_info'] = {}
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
        timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S') if isinstance(timestamp, pd.Timestamp) else str(
            timestamp)
        self.trades.append({'timestamp': timestamp_str, 'index': index_name, 'action': signal_type, 'price': price})
        if signal_type in ['BUY', 'SHORT']:
            self.current_position = signal_type
            self.entry_price = price
            self.entry_time = timestamp_str
        elif signal_type in ['SELL', 'COVER'] and self.current_position:
            profit = price - self.entry_price if self.current_position == 'BUY' else self.entry_price - price
            self.positions.append({
                'index': index_name, 'entry_time': self.entry_time, 'exit_time': timestamp_str,
                'entry_price': self.entry_price, 'exit_price': price, 'type': self.current_position,
                'profit': profit, 'profit_pct': (profit / self.entry_price) * 100
            })
            self.current_position = None


def save_historical_data(symbols_dict, months=60):
    """Fetch and save historical data locally"""
    data_dir = "historical_data"
    os.makedirs(data_dir, exist_ok=True)
    end = datetime.now()
    start = end - timedelta(days=30 * months)
    date_suffix = f"{months}months"
    print(f"Fetching last {months} months of data")
    for symbol, index in symbols_dict.items():
        print(f"Fetching data for {index}...")
        chunks = []
        current_start = start
        while current_start <= end:
            current_end = min(current_start + timedelta(days=75), end)
            try:
                r = dhan_object.intraday_minute_data(
                    security_id=symbol, from_date=current_start.strftime('%Y-%m-%d'),
                    to_date=current_end.strftime('%Y-%m-%d'), interval=CANDLE_INTERVAL_MINUTES,
                    exchange_segment="IDX_I", instrument_type="INDEX"
                )
                if data := r.get('data', []): chunks.append(pd.DataFrame(data))
            except Exception as e:
                print(f"  Error fetching {index}: {e}")
            current_start = current_end + timedelta(days=1)
        if not chunks:
            print(f"  No data fetched for {index}")
            continue
        df = pd.concat(chunks).reset_index(drop=True)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s').dt.tz_localize('UTC').dt.tz_convert(
            'Asia/Kolkata').dt.tz_localize(None)
        df.set_index('datetime', inplace=True)
        df = df[['open', 'high', 'low', 'close']].astype(float).sort_index()
        filename = f"{data_dir}/{index}_{date_suffix}.csv"
        df.to_csv(filename)
        print(f"  Saved {len(df)} rows to {filename}")


def fetch_ohlc(symbol, months=10):
    """Fetch OHLC data from Dhan for live mode"""
    end = datetime.now()
    start = end - timedelta(days=30 * months)
    chunks = []
    current_start = start
    while current_start <= end:
        current_end = min(current_start + timedelta(days=75), end)
        r = dhan_object.intraday_minute_data(
            security_id=symbol, from_date=current_start.strftime('%Y-%m-%d'),
            to_date=current_end.strftime('%Y-%m-%d'), interval=CANDLE_INTERVAL_MINUTES,
            exchange_segment="IDX_I", instrument_type="INDEX"
        )
        if data := r.get('data', []): chunks.append(pd.DataFrame(data))
        current_start = current_end + timedelta(days=1)
    if not chunks: return pd.DataFrame()
    df = pd.concat(chunks).reset_index(drop=True)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s').dt.tz_localize('UTC').dt.tz_convert(
        'Asia/Kolkata').dt.tz_localize(None)
    df.set_index('datetime', inplace=True)
    return df[['open', 'high', 'low', 'close']].astype(float).sort_index()


def process_symbol(symbol_dict, signal_state, trader, mode='live', trade_recorder=None, window_df=None):
    """
    Processes a single symbol - SIMPLIFIED TO MATCH PINE SCRIPT
    """
    symbol, index = next(iter(symbol_dict.items()))
    print(f"\n--- Processing Symbol: {index} ---")

    # Get data
    df = window_df if mode != 'live' else fetch_ohlc(symbol)
    if df.empty or len(df) < 200:
        print(f"⚠️ Insufficient data for {index} ({len(df)} bars). Skipping.")
        return

    print(f"  Data loaded. Shape: {df.shape}. Running MomentumX engine...")

    # Run the MomentumX engine
    strategy = MomentumX(df)
    df_signals = strategy.run()

    # Get the latest signals
    latest = df_signals.iloc[-1]
    start_long = latest['startLongTrade']
    start_short = latest['startShortTrade']
    end_long = latest['endLongTrade']
    end_short = latest['endShortTrade']

    current_time = latest.name
    current_price = latest['close']
    current_position = signal_state.get_position(index)

    # Get signal for persistence
    current_signal = int(latest['signal'])
    signal_state.update_signal(index, current_signal)

    action_taken = None

    print(f"  Current Position: {current_position}")
    print(f"  Signals: StartLong={start_long}, StartShort={start_short}, EndLong={end_long}, EndShort={end_short}")

    # Simple Pine Script logic - NO modifications!

    # Exit Logic (exactly as Pine Script)
    if current_position == 'LONG' and end_long:
        action_taken = "SELL"
        print(f"  Decision: Exit LONG position.")

    elif current_position == 'SHORT' and end_short:
        action_taken = "COVER"
        print(f"  Decision: Exit SHORT position.")

    # Entry Logic (exactly as Pine Script)
    elif current_position is None:
        if start_long:
            action_taken = "BUY"
            print("  Decision: Enter new LONG position.")

        elif start_short:
            action_taken = "SHORT"
            print("  Decision: Enter new SHORT position.")

    # Execute trade if needed
    if action_taken:
        print(f"\n{'=' * 20} TRADE ALERT {'=' * 20}")
        print(f"📈 ACTION: {action_taken} on {index} at {current_price:.2f}")
        print(f"{'=' * 53}\n")

        if mode == 'live' and trader:
            subject = f"{index} SIGNAL: {action_taken}"
            send_email(subject, f"{action_taken} {index} at {current_price}")
            db.collection("live_alerts").document(index).update(
                {"alerts": ArrayUnion([f"{action_taken} at {current_price}"])})
            trader.execute_single_trade(timestamp=current_time, index_name=index, signal_type=action_taken)

        elif mode == 'backtest' and trade_recorder:
            # Apply slippage
            slip_multiplier = (1 + BACKTEST_SLIPPAGE / 100) if action_taken in ['BUY', 'SHORT'] else (
                        1 - BACKTEST_SLIPPAGE / 100)
            execution_price = current_price * slip_multiplier
            trade_recorder.record_trade(current_time, index, action_taken, execution_price)

        # Update position state
        if action_taken in ["BUY", "SHORT"]:
            new_position = "LONG" if action_taken == "BUY" else "SHORT"
            signal_state.update_position(index, new_position, current_price, current_time)
        elif action_taken in ["SELL", "COVER"]:
            signal_state.update_position(index, None)

    else:
        print(f"  No action taken.")


def process_index_backtest(index_dict, backtest_state_lock):
    """
    Optimized backtest processing with pre-calculation
    """
    symbol, index = next(iter(index_dict.items()))
    index_state = SignalState(filepath=f"backtest_signal_state_{index}.json")
    trade_recorder = TradeRecorder()

    print(f"\n🔄 Starting optimized backtest for {index}...")

    try:
        # Load all data at once
        history_df = pd.read_csv(f'testing_data/{index}_test.csv', index_col='datetime', parse_dates=True)
        new_data_df = pd.read_csv(f'testing_data/{index}_input.csv', index_col='datetime', parse_dates=True)
        all_data = pd.concat([history_df, new_data_df])

        print(f"  Pre-calculating all signals for {index}...")

        # Run MomentumX on entire dataset once
        momentum = MomentumX(all_data)
        df_with_signals = momentum.run()

        print(f"  Signals calculated. Processing trades...")

        test_start_idx = len(history_df)

        # Now just iterate through the pre-calculated signals
        for i in range(test_start_idx, len(df_with_signals)):
            current_row = df_with_signals.iloc[i]

            # Simplified processing using pre-calculated signals
            start_long = current_row['startLongTrade']
            start_short = current_row['startShortTrade']
            end_long = current_row['endLongTrade']
            end_short = current_row['endShortTrade']

            current_time = current_row.name
            current_price = current_row['close']
            current_position = index_state.get_position(index)
            current_signal = int(current_row['signal'])

            # Update signal state
            index_state.update_signal(index, current_signal)

            action_taken = None

            # Same simple logic as live
            if current_position == 'LONG' and end_long:
                action_taken = "SELL"
            elif current_position == 'SHORT' and end_short:
                action_taken = "COVER"
            elif current_position is None:
                if start_long:
                    action_taken = "BUY"
                elif start_short:
                    action_taken = "SHORT"

            if action_taken:
                # Record trade with slippage
                slip_multiplier = (1 + BACKTEST_SLIPPAGE / 100) if action_taken in ['BUY', 'SHORT'] else (1 - BACKTEST_SLIPPAGE / 100)
                execution_price = current_price * slip_multiplier
                trade_recorder.record_trade(current_time, index, action_taken, execution_price)

                # Update position
                if action_taken in ["BUY", "SHORT"]:
                    new_position = "LONG" if action_taken == "BUY" else "SHORT"
                    index_state.update_position(index, new_position, current_price, current_time)
                else:
                    index_state.update_position(index, None)

            # Progress update
            if (i - test_start_idx) % 100 == 0:
                processed = i - test_start_idx + 1
                total = len(df_with_signals) - test_start_idx
                print(f"  {index}: Processed {processed}/{total} candles...")

    except Exception as e:
        print(f"❌ Error in backtest for {index}: {e}")
        import traceback
        traceback.print_exc()

    print(f"✅ Completed {index} backtest")

    # Cleanup
    if os.path.exists(index_state.filepath):
        os.remove(index_state.filepath)

    return index, trade_recorder


def run_backtest_analysis():
    """Run comprehensive backtest using parallel processing"""
    print("\n" + "=" * 80 + "\nRUNNING OPTIMIZED PARALLEL BACKTEST\n" + "=" * 80 + "\n")

    indices = [
        # {"13": "NIFTY"},
        # {"25": "BANKNIFTY"},
        # {"51": "SENSEX"},
        # {"27": "FINNIFTY"},
        {"442": "MIDCPNIFTY"}
    ]

    backtest_state_lock = threading.Lock()
    trade_recorders = {}

    with ThreadPoolExecutor(max_workers=len(indices)) as executor:
        # Use the optimized function
        future_to_index = {
            executor.submit(process_index_backtest, idx, backtest_state_lock): idx
            for idx in indices
        }

        for future in as_completed(future_to_index):
            try:
                index_name, recorder = future.result()
                trade_recorders[index_name] = recorder
            except Exception as exc:
                print(f"❌ Backtest task generated an exception: {exc}")

    print("\n" + "=" * 80 + "\nBACKTEST RESULTS\n" + "=" * 80)
    all_results = {}
    for index, recorder in trade_recorders.items():
        if recorder.positions:
            total_trades = len(recorder.positions)
            winning_trades = sum(1 for p in recorder.positions if p['profit'] > 0)
            total_profit = sum(p['profit'] for p in recorder.positions)
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            all_results[index] = {
                'trades': recorder.trades, 'positions': recorder.positions,
                'performance': {
                    'total_trades': total_trades, 'winning_trades': winning_trades,
                    'losing_trades': total_trades - winning_trades, 'win_rate': win_rate,
                    'total_profit': total_profit,
                    'avg_profit': total_profit / total_trades if total_trades > 0 else 0,
                    'avg_profit_pct': sum(
                        p['profit_pct'] for p in recorder.positions) / total_trades if total_trades > 0 else 0
                }
            }
            print(
                f"\n{index} Performance:\n  Total Trades: {total_trades}\n  Win Rate: {win_rate:.2f}%\n  Total Profit: {total_profit:.2f}")
    save_backtest_results(all_results)


def save_backtest_results(all_results):
    """Save backtest results to CSV"""
    output_dir = "backtest_results"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_data = []
    for index, results in all_results.items():
        perf = results['performance']
        summary_data.append({
            'Index': index, 'Total_Trades': perf['total_trades'], 'Win_Rate_%': round(perf['win_rate'], 2),
            'Total_Profit': round(perf['total_profit'], 2)
        })
    if summary_data:
        pd.DataFrame(summary_data).to_csv(f"{output_dir}/backtest_summary_{timestamp}.csv", index=False)
    for index, results in all_results.items():
        if results['positions']:
            pd.DataFrame(results['positions']).to_csv(f"{output_dir}/{index}_trades_{timestamp}.csv", index=False)
    print(f"\nResults saved to {output_dir}/")


def get_next_candle_time():
    """Calculate the next candle close time"""
    now = datetime.now()
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    if now < market_open: return market_open + timedelta(minutes=CANDLE_INTERVAL_MINUTES,
                                                         seconds=CANDLE_COMPLETION_BUFFER)
    minutes_since_open = (now - market_open).total_seconds() / 60
    candles_passed = int(minutes_since_open // CANDLE_INTERVAL_MINUTES)
    next_candle_close = market_open + timedelta(minutes=(candles_passed + 1) * CANDLE_INTERVAL_MINUTES)
    return next_candle_close + timedelta(seconds=CANDLE_COMPLETION_BUFFER)


def run_live_trading(signal_state, trader):
    """Run live trading loop"""
    print(f"✅ Live trading started. Will run every {CANDLE_INTERVAL_MINUTES} minutes until 3:30 PM IST.")
    indices = [{"13": "NIFTY"}, {"25": "BANKNIFTY"}, {"51": "SENSEX"}, {"27": "FINNIFTY"}, {"442": "MIDCPNIFTY"}]
    while True:
        now = datetime.now()
        if now.hour == 15 and now.minute >= 30:
            print("⏹️ Market closed. Stopping strategy.")
            break

        next_run = get_next_candle_time()
        wait_time = (next_run - now).total_seconds()
        if wait_time > 0:
            print(f"\n⏳ Next run at {next_run.strftime('%H:%M:%S')}. Waiting {int(wait_time)} seconds...\n")
            time.sleep(wait_time)

        print(f"\n{'=' * 60}\n🕒 Running strategy at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'=' * 60}\n")
        for index_dict in indices:
            try:
                process_symbol(index_dict, signal_state, trader, mode='live')
            except Exception as e:
                symbol, index = next(iter(index_dict.items()))
                print(f"❌ Error processing {index}: {e}")


# === Main Entry Point ===
if __name__ == '__main__':
    load_dotenv()
    CLIENT_ID = os.getenv("DHAN_CLIENT_ID")
    ACCESS_TOKEN = os.getenv("DHAN_API_KEY")
    dhan_object = dhanhq.dhanhq(CLIENT_ID, ACCESS_TOKEN)

    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == 'backtest':
            run_backtest_analysis()
        elif mode == 'save_data':
            print("Saving historical data...")
            indices = {"13": "NIFTY", "25": "BANKNIFTY", "51": "SENSEX", "27": "FINNIFTY", "442": "MIDCPNIFTY"}
            save_historical_data(indices, months=60)
        elif mode in ['test_live', 'live']:
            is_test = mode == 'test_live'
            print(f"Running in {'TEST LIVE' if is_test else 'LIVE TRADING'} mode...")
            trader = KotakOptionsTrader(test_mode=is_test)
            if not is_test and not trader.get_account_status()['logged_in']:
                print("❌ Failed to login to Kotak. Please check token.")
                sys.exit(1)
            cred = credentials.Certificate("stock-monitoring-fb.json")
            initialize_app(cred)
            db = firestore.client()
            signal_state = SignalState()
            run_live_trading(signal_state, trader)
        else:
            print(f"Unknown mode: {mode}. Use 'backtest', 'save_data', 'test_live', or 'live'.")
    else:
        print("Defaulting to LIVE TRADING mode. Use 'test_live' for paper trading.")
        # Add a safeguard for accidental live run
        confirm = input("Are you sure you want to run with REAL money? (yes/no): ")
        if confirm.lower() == 'yes':
            trader = KotakOptionsTrader(test_mode=False)
            if not trader.get_account_status()['logged_in']:
                print("❌ Failed to login to Kotak.")
                sys.exit(1)
            cred = credentials.Certificate("stock-monitoring-fb.json")
            initialize_app(cred)
            db = firestore.client()
            signal_state = SignalState()
            run_live_trading(signal_state, trader)
        else:
            print("Exiting.")
