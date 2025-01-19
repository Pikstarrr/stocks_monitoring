import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from firebase_admin import credentials, firestore, initialize_app
import dhanhq

from send_mail import send_email

# Firebase Setup
cred = credentials.Certificate("stock-monitoring-fb.json")
initialize_app(cred)
db = firestore.client()
doc_ref = db.collection("stock_data").document("values")

CLIENT_ID = "1105995499"
ACCESS_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzM5ODc5NjU2LCJ0b2tlbkNvbnN1bWVyVHlwZSI6IlNFTEYiLCJ3ZWJob29rVXJsIjoiIiwiZGhhbkNsaWVudElkIjoiMTEwNTk5NTQ5OSJ9.yZTzZoNO4_Is1Bm0QB8rvyduXR04vtUi-4H-D4VHMnDCgQpOTt6n0juPkvvfyLnFjUTsUK2if1SdR1rwuOCKFw"

security_ids = [
    10417, 10905, 11195, 1134, 11626, 11987, 13359, 13966, 14908, 15179,
    18721, 193, 19401, 24909, 24961, 25358, 25907, 27213, 28378, 29135,
    29284, 31181, 3672, 4656, 6125, 6445, 6705, 6944, 7374, 7506, 7508,
    7982, 8840, 9428, 10515, 10576, 10939, 2328, 5851, 8506
]

def calculate_current_value(stocks_current_prices):
    total = 0.0
    for key, value in stocks_current_prices:
        total += value["ohlc"]["high"]
    return total

def fetch_stock_values_from_dhan():
    current_price = 0.0
    try:
        dhan_object = dhanhq.dhanhq(CLIENT_ID, ACCESS_TOKEN)
        current_stock_data = dhan_object.quote_data({
            "NSE_EQ": security_ids,
        })

        # parse the data from
        current_price = calculate_current_value(current_stock_data["data"]["data"]["NSE_EQ"].items())
    except Exception as e:
        print(e)
    return  current_price

def update_value_in_db_and_user():
    current_value = fetch_stock_values_from_dhan()

    doc = doc_ref.get()
    highest_value = doc.to_dict().get("highest_value", 0.0) if doc.exists else 0.0

    if current_value > highest_value:
        highest_value = current_value
        doc_ref.set({"highestValue": highest_value})
    elif (highest_value - current_value) / highest_value > 0.05:
        send_email(
            "Stock Alert: Significant Drop",
            f"Current Value: {current_value}, Highest Value: {highest_value}",
        )
        highest_value = current_value
        doc_ref.set({"highestValue": highest_value})

# Run Script
if __name__ == "__main__":
    update_value_in_db_and_user()

