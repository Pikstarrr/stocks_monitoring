import dhanhq
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np

load_dotenv()
CLIENT_ID = os.getenv("DHAN_CLIENT_ID")
ACCESS_TOKEN = os.getenv("DHAN_API_KEY")
dhan_object = dhanhq.dhanhq(CLIENT_ID, ACCESS_TOKEN)

security_ids = [2, 27, 25, 51, 442]

capital = 100000  # Initial capital
position = 0
entry_price = 0
stop_loss = 0
trailing_stop = 0
trades = []

def fetch_stock_values_from_dhan():
    current_price = 0.0
    try:
        current_index_data = dhan_object.intraday_minute_data(
            "2",
            "IDX_I",
            "INDEX",
            "2025-01-22",
            "2025-01-26",
            1
        )

        if current_index_data["status"] == "success":
            simulate_trading(current_index_data['data'])
        else:
            print("NO DATA")
    except Exception as e:
        print(e)
    return current_price


def calculate_indicators(df):
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df['RSI_14'] = compute_rsi(df['close'], 14)
    df['EMA_9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['MACD'], df['Signal_Line'] = compute_macd(df['close'])

def compute_rsi(series, period):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(series, short_window=12, long_window=26, signal_window=9):
    short_ema = series.ewm(span=short_window, adjust=False).mean()
    long_ema = series.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal_line

def trading_strategy(df):
    global position
    global entry_price
    global stop_loss
    global trailing_stop
    global capital

    for i in range(1, len(df)):
        # Buy condition: SMA crossover + RSI confirmation
        if df['SMA_50'][i] < df['SMA_20'][i] < df['EMA_9'][i] and df['RSI_14'][i] > 50 and df['MACD'][i] > df['Signal_Line'][i] and position == 0:
            entry_price = df['close'][i]
            stop_loss = entry_price * 0.97  # 3% stop loss
            trailing_stop = entry_price * 1.05  # 5% target
            position = capital / entry_price
            date = dhan_object.convert_to_date_time(epoch=df['timestamp'][i])
            trades.append(('BUY', date.strftime('%m/%d/%Y'), entry_price))

        # Sell condition: Target hit or stop loss hit
        elif position > 0:
            if df['close'][i] >= trailing_stop or df['close'][i] <= stop_loss:
                exit_price = df['close'][i]
                profit = (exit_price - entry_price) * position
                capital += profit
                position = 0
                trades.append(('SELL', df['timestamp'][i], exit_price))

    return trades, capital

def simulate_trading(data):
    global trades
    # Simulating live trading
    df = pd.DataFrame(data)
    calculate_indicators(df)
    trades, final_capital = trading_strategy(df)

    # Statistics
    profit_trades = len([t for t in trades if t[0] == 'SELL' and t[2] > entry_price])
    loss_trades = len(trades) // 2 - profit_trades

    print(f"Total Trades: {len(trades) // 2}")
    print(f"Profitable Trades: {profit_trades}")
    print(f"Losing Trades: {loss_trades}")
    print(f"Final Capital: {final_capital}")

# Run Script
if __name__ == "__main__":
    fetch_stock_values_from_dhan()
