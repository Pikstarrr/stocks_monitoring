# momentum_x_backtest.py — improved version with label thresholding, confidence filter, kernel tuning, and ATR exit logic
import math
import time
from collections import deque
from datetime import datetime, timedelta
from firebase_admin import credentials, firestore, initialize_app

import pandas as pd
import numpy as np
import os

from dotenv import load_dotenv
from google.cloud.firestore_v1 import ArrayUnion
from sklearn.metrics.pairwise import manhattan_distances

from send_mail import send_email
from strategy_features import compute_feature, rational_quadratic_kernel, gaussian_kernel

import dhanhq

# === CONFIG ===
FEATURE_CONFIG = [
    ("RSI", 14, 1),
    ("WT", 10, 11),
    ("CCI", 20, 1),
    ("ADX", 20, 2),
    ("RSI", 9, 1),
]
NEIGHBOR_COUNT = 5
STRICT_EXIT_BARS = 4
DEBUG = True
DATASET_DIR = "DataSets"
OUTPUT_DIR = "OutputData"
LOOKAHEAD = 4
LABEL_THRESHOLD = 0.002  # 0.2% move threshold
CONFIDENCE_THRESHOLD = 3

# === Feature Calculation ===
def build_features(df):
    features = []
    for kind, param_a, param_b in FEATURE_CONFIG:
        f = compute_feature(df.copy(), kind, param_a, param_b)
        f.name = f"{kind}_{param_a}_{param_b}"
        features.append(f)
    return pd.concat(features, axis=1).dropna()

# === Fast Label Prediction with Threshold and Confidence ===
def predict_labels(features, close_series):
    X = features.values
    close = close_series.loc[features.index].values
    n = len(X)
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

    distances = manhattan_distances(X, X)
    np.fill_diagonal(distances, np.inf)

    predictions = []
    for i in range(n):
        if i < LOOKAHEAD:
            predictions.append(0)
            continue
        idx = np.argsort(distances[i])[:NEIGHBOR_COUNT]
        vote = np.sum(Y[idx])
        if abs(vote) < CONFIDENCE_THRESHOLD:
            predictions.append(0)
        else:
            predictions.append(int(np.sign(vote)))

    return pd.Series(predictions, index=features.index)

# === Kernel Signal Detection (tuned params) ===
def get_kernel_signals(series, h=10, r=6.0, x=15, lag=1):
    yhat1 = rational_quadratic_kernel(series, h, r, x)
    yhat2 = gaussian_kernel(series, h - lag, x)
    bullish = (yhat1.shift(1) < yhat1)
    bearish = (yhat1.shift(1) > yhat1)
    return bullish, bearish, yhat1, yhat2

# === ATR Calculation ===
def calculate_atr(df, period=14):
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

def simulate_streaming_base_df(base_csv, stream_csv):
    base_df = pd.read_csv(base_csv, parse_dates=['datetime'])
    stream_df = pd.read_csv(stream_csv, parse_dates=['datetime'])

    base_df.set_index('datetime', inplace=True)
    stream_df.set_index('datetime', inplace=True)

    base_df = base_df.sort_index()
    stream_df = stream_df.sort_index()

    for timestamp, new_row in stream_df.iterrows():
        # Drop oldest row and append new row
        base_df = base_df.iloc[1:]  # Drop first row
        base_df.loc[timestamp] = new_row  # Add new row by timestamp

        # Sort in case timestamps are out of order
        base_df = base_df.sort_index()

        yield base_df.copy()  # Yield a copy to avoid mutation outside

def fast_simulate_streaming_base_df(base_csv, stream_csv):
    base_df = pd.read_csv(base_csv, parse_dates=['datetime'])
    stream_df = pd.read_csv(stream_csv, parse_dates=['datetime'])

    # Convert both to sorted list of rows (to avoid expensive .iloc later)
    base_records = base_df.sort_values('datetime').to_dict('records')
    stream_records = stream_df.sort_values('datetime').to_dict('records')

    # Create deque for rolling window (initial base window)
    window = deque(base_records, maxlen=len(base_records))

    for new_row in stream_records:
        window.append(new_row)  # Automatically removes oldest row
        df_out = pd.DataFrame(window)
        df_out['datetime'] = pd.to_datetime(df_out['datetime'])
        df_out.set_index('datetime', inplace=True)
        yield df_out.copy()


def fetch_ohlc(symbol, interval=15, months=5):
    end = datetime.now()
    start = end - timedelta(days=30 * months)

    chunks = []
    current_start = start
    while current_start < end:
        current_end = min(current_start + timedelta(days=74), end)
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
    now = pd.Timestamp.now().floor("15min")
    if now not in df.index or (datetime.now() - df.index[-1].to_pydatetime()).total_seconds() > 1200:
        print(f"🔄 Appending live LTP for {symbol}")
        r2 = dhan_object.quote_data({
            "IDX_I": [int(symbol)],
        })

        ltp = float(r2['data']['data']['IDX_I'][symbol]['last_price'])
        df.loc[now] = [ltp, ltp, ltp, ltp]

    return df.sort_index()

# === Backtest Logic ===
def backtest(df, predictions):
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
        atr_val = atr.iloc[i]
        log_row = {
            "Time": ts,
            "Signal": signal,
            "Close": price,
            "ATR": atr_val,
            "RK": rk.iloc[i],
            "GK": gk.iloc[i],
            "Bullish": bullish.iloc[i],
            "Bearish": bearish.iloc[i],
            "Position": position
        }

        if position == 'LONG':
            bar_hold += 1
            if bearish.iloc[i] or bar_hold >= STRICT_EXIT_BARS or price < entry_price - atr_val:
                trades.append((ts, 'SELL', price))
                log_row["Action"] = "SELL (exit)"
                position = None
                bar_hold = 0
        elif position == 'SHORT':
            bar_hold += 1
            if bullish.iloc[i] or bar_hold >= STRICT_EXIT_BARS or price > entry_price + atr_val:
                trades.append((ts, 'COVER', price))
                log_row["Action"] = "COVER (exit)"
                position = None
                bar_hold = 0

        if position is None:
            if signal == 1 and bullish.iloc[i]:
                trades.append((ts, 'BUY', price))
                log_row["Action"] = "BUY (entry)"
                position = 'LONG'
                entry_price = price
            elif signal == -1 and bearish.iloc[i]:
                trades.append((ts, 'SHORT', price))
                log_row["Action"] = "SHORT (entry)"
                position = 'SHORT'
                entry_price = price

        if DEBUG:
            log_entries.append(log_row)

    return trades, pd.DataFrame(log_entries)

# === Process All Files Year by Year ===
# os.makedirs(OUTPUT_DIR, exist_ok=True)
# summary_stats = []

# for yearly_df in fast_simulate_streaming_base_df("DataSets/history_data.csv", "DataSets/live_data.csv"):
#     if len(yearly_df) < 100:
#         print(f"Skipping: too little data")
#         continue
#
#     print(f"Running with {len(yearly_df)} rows...")
#
#     features = build_features(yearly_df)
#     if len(features) < LOOKAHEAD:
#         print(f"Not enough features to run ML on")
#         continue
#
#     predictions = predict_labels(features, yearly_df['close'])
#     trades, logs = backtest(yearly_df.loc[predictions.index], predictions)
#
#     pnl = 0
#     positions = []
#     for i in range(1, len(trades), 2):
#         t1, action1, price1 = trades[i - 1]
#         t2, action2, price2 = trades[i]
#         profit = price2 - price1 if action1 == 'BUY' else price1 - price2
#         pnl += profit
#         positions.append((t1, action1, price1, t2, action2, price2, profit))
#
#     results = pd.DataFrame(positions,
#                            columns=["EntryTime", "EntryType", "EntryPrice", "ExitTime", "ExitType", "ExitPrice", "PnL"])
#     win_rate = (results['PnL'] > 0).sum() / len(results) * 100 if len(results) > 0 else 0
#
#     summary_stats.append({
#         "Total Trades": len(results),
#         "Total PnL": round(pnl, 2),
#         "Win Rate (%)": round(win_rate, 2)
#     })
#
#     logs.to_csv(os.path.join(OUTPUT_DIR, f"debug_log.csv"), index=False)
#     results.to_csv(os.path.join(OUTPUT_DIR, f"backtest_results.csv"), index=False)
#
# # === Print Summary Table ===
# summary_df = pd.DataFrame(summary_stats).sort_values(by=["Interval", "Year"])
# print("===== STRATEGY PERFORMANCE SUMMARY =====")
# print(summary_df.to_string(index=False))
#
# # === Identify Best Performing Interval ===
# best_row = summary_df.sort_values(by="Total PnL", ascending=False).iloc[0]
# print("📈 Best Performing Interval:")
# print(f"Interval: {best_row['Interval']} | Year: {int(best_row['Year'])} | Total PnL: {best_row['Total PnL']} | Win Rate: {best_row['Win Rate (%)']}%")


def your_strategy_function(symbol):
    yearly_df = fetch_ohlc(symbol)

    if len(yearly_df) < 100:
        print(f"Skipping: too little data")
        return

    print(f"Running with {len(yearly_df)} rows...")

    features = build_features(yearly_df)
    if len(features) < LOOKAHEAD:
        print(f"Not enough features to run ML on")
        return

    predictions = predict_labels(features, yearly_df['close'])
    trades, logs = backtest(yearly_df.loc[predictions.index], predictions)

    if math.isnan(logs['Action'][logs.index[-1]]):
       print("No prediction")
    else:
        last_row = logs.iloc[-1]
        index = 'NIFTY' if '13' in symbol else 'BANKNIFTY'
        log_str = (
            f"INDEX: {index}, "
            f"Time: {last_row['Time']}, "
            f"Signal: {last_row['Signal']}, "
            f"Close: {last_row['Close']:.2f}, "
            f"ATR: {last_row['ATR']:.2f}, "
            f"RK: {last_row['RK']:.2f}, "
            f"GK: {last_row['GK']:.2f}, "
            f"Bullish: {last_row['Bullish']}, "
            f"Bearish: {last_row['Bearish']}, "
            f"Position: {last_row['Position']}, "
            f"Action: {last_row['Action']}"
        )

        subject = f"{index} SIGNAL: {last_row['Action']}"
        body = log_str
        print(f"📧 Alert: {body}")
        send_email(subject, body)

        doc_ref = db.collection("live_alerts").document(index)
        doc_ref.update({
            "alerts": ArrayUnion([log_str])
        })

    last_trade = trades[-1]
    print(f"📈 Last Trade: {last_trade}")
    last_log = logs.iloc[-1]
    print(f"📝 Logs: {last_log}")

def is_market_closed():
    # IST is UTC+5:30
    now = datetime.now()
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return now >= market_close

def run_every_15_minutes_until_close():
    print("✅ Strategy started. Will run every 15 minutes until 3:30 PM IST.")
    while True:
        # if is_market_closed():
        #     print("⏹️ Market closed at 3:30 PM. Stopping strategy.")
        #     break
        your_strategy_function("13")
        your_strategy_function("25")
        # Sleep until the next 15-minute mark
        now = datetime.now()
        next_run = (now + timedelta(minutes=15)).replace(second=0, microsecond=0)
        sleep_duration = (next_run - now).total_seconds()
        print(f"🕒 Sleeping for {int(sleep_duration)} seconds...\n")
        time.sleep(sleep_duration)


if __name__ == '__main__':
    load_dotenv()
    CLIENT_ID = os.getenv("DHAN_CLIENT_ID")
    ACCESS_TOKEN = os.getenv("DHAN_API_KEY")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dhan_object = dhanhq.dhanhq(CLIENT_ID, ACCESS_TOKEN)

    # Firebase Setup
    cred = credentials.Certificate("stock-monitoring-fb.json")
    initialize_app(cred)
    db = firestore.client()

    run_every_15_minutes_until_close()


