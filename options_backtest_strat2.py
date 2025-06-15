# momentum_x_backtest.py — improved version with label thresholding, confidence filter, kernel tuning, and ATR exit logic

import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import manhattan_distances
from strategy_features import compute_feature, rational_quadratic_kernel, gaussian_kernel

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
os.makedirs(OUTPUT_DIR, exist_ok=True)
summary_stats = []
all_files = [f for f in os.listdir(DATASET_DIR) if f.endswith("_data.csv")]

for filename in sorted(all_files):
    filepath = os.path.join(DATASET_DIR, filename)
    print(f"\n=== Processing file: {filename} ===")
    df = pd.read_csv(filepath, parse_dates=['datetime'])
    df.set_index('datetime', inplace=True)

    for year in sorted(df.index.year.unique()):
        yearly_df = df[df.index.year == year]
        if len(yearly_df) < 100:
            print(f"Skipping {filename} {year}: too little data")
            continue

        print(f"Running {filename} for year {year} with {len(yearly_df)} rows...")

        features = build_features(yearly_df)
        if len(features) < LOOKAHEAD:
            print(f"Not enough features to run ML on {filename} {year}")
            continue

        predictions = predict_labels(features, yearly_df['close'])
        trades, logs = backtest(yearly_df.loc[predictions.index], predictions)

        pnl = 0
        positions = []
        for i in range(1, len(trades), 2):
            t1, action1, price1 = trades[i - 1]
            t2, action2, price2 = trades[i]
            profit = price2 - price1 if action1 == 'BUY' else price1 - price2
            pnl += profit
            positions.append((t1, action1, price1, t2, action2, price2, profit))

        results = pd.DataFrame(positions, columns=["EntryTime", "EntryType", "EntryPrice", "ExitTime", "ExitType", "ExitPrice", "PnL"])
        win_rate = (results['PnL'] > 0).sum() / len(results) * 100 if len(results) > 0 else 0

        summary_stats.append({
            "Interval": filename.replace("_data.csv", ""),
            "Year": year,
            "Total Trades": len(results),
            "Total PnL": round(pnl, 2),
            "Win Rate (%)": round(win_rate, 2)
        })

        prefix = filename.replace("_data.csv", f"_{year}")
        logs.to_csv(os.path.join(OUTPUT_DIR, f"{prefix}_debug_log.csv"), index=False)
        results.to_csv(os.path.join(OUTPUT_DIR, f"{prefix}_backtest_results.csv"), index=False)

# === Print Summary Table ===
summary_df = pd.DataFrame(summary_stats).sort_values(by=["Interval", "Year"])
print("===== STRATEGY PERFORMANCE SUMMARY =====")
print(summary_df.to_string(index=False))

# === Identify Best Performing Interval ===
best_row = summary_df.sort_values(by="Total PnL", ascending=False).iloc[0]
print("📈 Best Performing Interval:")
print(f"Interval: {best_row['Interval']} | Year: {int(best_row['Year'])} | Total PnL: {best_row['Total PnL']} | Win Rate: {best_row['Win Rate (%)']}%")
