import pandas as pd
import os

# === CONFIG ===
INPUT_FILE = "my_data_bank.csv"  # <-- change to your actual filename

intervals = {
    '15min': '15min',
    '30min': '30min',
    '1h': '1h'
}

# === Load and Parse Data ===
df = pd.read_csv(INPUT_FILE)

# Rename and clean if needed
if 'datetime' not in df.columns:
    df.rename(columns={'date': 'datetime'}, inplace=True)

df['datetime'] = pd.to_datetime(df['datetime'], format="%d-%m-%Y %H:%M", errors='coerce')
df = df.dropna(subset=['datetime'])  # remove rows where date parsing failed
df = df.sort_values('datetime')
df.set_index('datetime', inplace=True)

# Confirm datetime index
if not isinstance(df.index, pd.DatetimeIndex):
    raise TypeError("Index is not a DatetimeIndex")

# Get all years
years = sorted(df.index.year.unique())

# === Process Each Interval & Year ===
for name, rule in intervals.items():
    print(f"\n⏱ Processing interval: {name}")

    df_resampled = df.resample(rule).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    for year in years:
        df_year = df_resampled[df_resampled.index.year == year]
        if df_year.empty:
            continue

        # Save to CSV
        df_year = df_year.reset_index()
        output_file = f"../DataSets/{name}_{year}_data.csv"
        df_year.to_csv(output_file, index=False)

        print(f"✅ Saved {output_file} with {len(df_year)} rows.")
