import pandas as pd

df = pd.read_csv("backtest_results.csv", parse_dates=['EntryTime', 'ExitTime'])

df['HoldHours'] = (df['ExitTime'] - df['EntryTime']).dt.total_seconds() / 3600
df['Month'] = df['EntryTime'].dt.to_period('M')

summary = {
    "Total Trades": len(df),
    "Win Rate (%)": round((df['PnL'] > 0).mean() * 100, 2),
    "Total PnL": round(df['PnL'].sum(), 2),
    "Avg PnL per Trade": round(df['PnL'].mean(), 2),
    "Max Profit": round(df['PnL'].max(), 2),
    "Max Loss": round(df['PnL'].min(), 2),
    "Avg Hold Duration (hrs)": round(df['HoldHours'].mean(), 2),
    "Monthly PnL Breakdown": df.groupby('Month')['PnL'].sum().to_dict()
}

print(summary)