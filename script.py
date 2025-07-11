# from firebase_admin import credentials, firestore, initialize_app
# import dhanhq
# import os
# from dotenv import load_dotenv
#
# from send_mail import send_email
#
# # Firebase Setup
# cred = credentials.Certificate("stock-monitoring-fb.json")
# initialize_app(cred)
# db = firestore.client()
# doc_ref = db.collection("stock_data").document("values")
#
# # CLIENT_ID = os.environ["DHAN_CLIENT_ID"]
# # ACCESS_TOKEN = os.environ["DHAN_API_KEY"]
# load_dotenv()
# CLIENT_ID = os.getenv("DHAN_CLIENT_ID")
# ACCESS_TOKEN = os.getenv("DHAN_API_KEY")
#
# security_ids = {
#     'DYCL': 10417,
#     'MPSLTD': 10578,
#     'GENESYS': 10905,
#     'INDIGO': 11195,
#     'GOKEX': 11778,
#     'ICIL': 11987,
#     'KFINTECH': 13359,
#     'AARTIPHARM': 13868,
#     'HBLENGINE': 13966,
#     'KDDL': 14908,
#     'TARIL': 15178,
#     'ECLERX': 15179,
#     'HCG': 15555,
#     'SUVENPHAR': 17945,
#     'NUVAMA': 18721,
#     'BLUEJET': 19686,
#     'INTERARCH': 24909,
#     'ORIENTTECH': 24961,
#     'TDPOWERSYS': 25178,
#     'PGEL': 25358,
#     'WAAREEENER': 25907,
#     'EIEL': 27213,
#     'KITEX': 28903,
#     'INDUSTOWER': 29135,
#     'MCX': 31181,
#     'BORORENEW': 3155,
#     'SBCL': 4656,
#     'AMIORG': 5578,
#     'PAYTM': 6705,
#     'GANECOS': 6944,
#     'WOCKPHARMA': 7506,
#     'ZENTEC': 7508,
#     'GRWRHITECH': 7982,
#     'PIXTRANS': 9087,
#     'SKIPPER': 9428,
#     'AXISCADES': 9440,
#     'UTISENSETF': 10515,
#     'NIFTYBEES': 10576,
#     'JUNIORBEES': 10939,
#     'GOLDBEES': 14428,
#     'LIQUIDCASE': 21750,
#     'CPSEETF': 2328,
#     'BANKNIFTY1': 5851,
#     'ICICIB22': 522,
#     'ONESOURCE': 29224
# }
#
#
# def calculate_current_value(stocks_current_prices):
#     total_current = 0.0
#     for key, value in stocks_current_prices:
#         total_current += value["last_price"]
#     return total_current
#
#
# def fetch_stock_values_from_dhan():
#     current_price = 0.0
#     try:
#         dhan_object = dhanhq.dhanhq(CLIENT_ID, ACCESS_TOKEN)
#         current_stock_data = dhan_object.quote_data({
#             "NSE_EQ": list(security_ids.values()),
#         })
#
#         # parse the data from
#         current_price = calculate_current_value(current_stock_data["data"]["data"]["NSE_EQ"].items())
#     except Exception as e:
#         print(e)
#     return current_price
#
# def update_value_in_db_and_user():
#     current_value = fetch_stock_values_from_dhan()
#
#     if current_value == 0:
#         return
#
#     doc = doc_ref.get()
#     current_highest_value = doc.to_dict().get("highest_value", 0.0) if doc.exists else 0.0
#
#     print("Before :")
#     print(current_value, current_highest_value)
#     if current_value > current_highest_value:
#         current_highest_value = current_value
#         doc_ref.update({"highest_value": current_highest_value})
#         doc_ref.update({"current_value": current_value})
#         doc_ref.update({"difference_percent": 0.0})
#     else:
#         difference_percent = ((current_highest_value - current_value) / current_highest_value) * 100
#         if difference_percent > 6:
#             send_email(
#                 "Stock Alert: Significant Drop",
#                 f"Current Value: {current_value}, Highest Value: {current_highest_value}",
#             )
#             current_highest_value = current_value
#             doc_ref.update({"highest_value": current_highest_value})
#             doc_ref.update({"current_value": current_highest_value})
#             doc_ref.update({"difference_percent": 0.0})
#         else:
#             doc_ref.update({"difference_percent": difference_percent})
#             doc_ref.update({"current_value": current_value})
#         print(difference_percent)
#
#     print("After : ")
#     print(current_value, current_highest_value)
#
#
# # Run Script
# if __name__ == "__main__":
#     update_value_in_db_and_user()

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from strategy_features import compute_feature, rational_quadratic_kernel, gaussian_kernel
from sklearn.metrics.pairwise import manhattan_distances

import dhanhq

# from firebase_admin import credentials, firestore, initialize_app
import os
from dotenv import load_dotenv

from send_mail import send_email
#
# # Firebase Setup
# cred = credentials.Certificate("stock-monitoring-fb.json")
# initialize_app(cred)
# db = firestore.client()
# doc_ref = db.collection("stock_data").document("values")

load_dotenv()
CLIENT_ID = os.getenv("DHAN_CLIENT_ID")
ACCESS_TOKEN = os.getenv("DHAN_API_KEY")

FEATURE_CONFIG = [
    ("RSI", 14, 1),
    ("WT", 10, 11),
    ("CCI", 20, 1),
    ("ADX", 20, 2),
    ("RSI", 9, 1),
]
NEIGHBOR_COUNT = 5
LOOKAHEAD = 4
LABEL_THRESHOLD = 0.002
CONFIDENCE_THRESHOLD = 3

# === Fetch Historical Data with Live Append ===
def fetch_ohlc(symbol, interval=15, months=6):
    end = datetime.now()
    start = end - timedelta(days=30 * months)

    dhan_object = dhanhq.dhanhq(CLIENT_ID, ACCESS_TOKEN)

    chunks = []
    current_start = start
    while current_start < end:
        current_end = min(current_start + timedelta(days=89), end)
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
    df['datetime'] = pd.to_datetime(df['timestamp'])
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

# === Feature Engineering ===
def build_features(df):
    feats = []
    for kind, a, b in FEATURE_CONFIG:
        f = compute_feature(df.copy(), kind, a, b)
        f.name = f"{kind}_{a}_{b}"
        feats.append(f)
    return pd.concat(feats, axis=1).dropna()

# === Predict Labels ===
def predict_labels(features, close):
    X = features.values
    close = close.loc[features.index].values
    Y = np.zeros(len(X))
    for i in range(LOOKAHEAD, len(X)):
        future = close[i]
        now = close[i - LOOKAHEAD]
        delta = (future - now) / now
        if delta > LABEL_THRESHOLD:
            Y[i - LOOKAHEAD] = 1
        elif delta < -LABEL_THRESHOLD:
            Y[i - LOOKAHEAD] = -1

    distances = manhattan_distances(X, X)
    np.fill_diagonal(distances, float('inf'))

    preds = []
    for i in range(len(X)):
        if i < LOOKAHEAD:
            preds.append(0)
            continue
        idx = distances[i].argsort()[:NEIGHBOR_COUNT]
        vote = np.sum(Y[idx])
        preds.append(int(np.sign(vote)) if abs(vote) >= CONFIDENCE_THRESHOLD else 0)
    return pd.Series(preds, index=features.index)

# === Kernel Confirmation ===
def confirm_signal(series, signal):
    rk = rational_quadratic_kernel(series, h=10, r=6.0, x=15)
    gk = gaussian_kernel(series, h=9, x=15)
    if signal == 1:
        return rk.iloc[-1] > rk.iloc[-2] and gk.iloc[-1] > gk.iloc[-2]
    elif signal == -1:
        return rk.iloc[-1] < rk.iloc[-2] and gk.iloc[-1] < gk.iloc[-2]
    return False

# === Run for One Symbol ===
def run_for_symbol(symbol):
    df = fetch_ohlc(symbol)
    features = build_features(df)
    if len(features) < 100:
        print(f"Not enough data for {symbol}")
        return

    preds = predict_labels(features, df['close'])
    signal = preds.iloc[-1]
    if signal == 0:
        print(f"No signal for {symbol}")
        return

    confirmed = confirm_signal(df['close'].loc[features.index], signal)
    if confirmed:
        price = df['close'].iloc[-1]
        ts = df.index[-1].strftime("%Y-%m-%d %H:%M")
        name = "NIFTY" if "NIFTY" in symbol else "BANKNIFTY"
        subject = f"{name} SIGNAL: {'BUY' if signal == 1 else 'SELL'}"
        body = f"{name} {('BUY' if signal == 1 else 'SELL')} @ {price:.2f} on {ts}"
        print(f"📧 Alert: {body}")
        send_email(subject, body)
    else:
        print(f"Signal not confirmed for {symbol}")

if __name__ == '__main__':
    run_for_symbol("13")
    run_for_symbol("25")
