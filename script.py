from firebase_admin import credentials, firestore, initialize_app
import dhanhq
import os
from dotenv import load_dotenv

from send_mail import send_email

# Firebase Setup
cred = credentials.Certificate("stock-monitoring-fb.json")
initialize_app(cred)
db = firestore.client()
doc_ref = db.collection("stock_data").document("values")

# CLIENT_ID = os.environ["DHAN_CLIENT_ID"]
# ACCESS_TOKEN = os.environ["DHAN_API_KEY"]
load_dotenv()
CLIENT_ID = os.getenv("DHAN_CLIENT_ID")
ACCESS_TOKEN = os.getenv("DHAN_API_KEY")

security_ids = [
    10417, 10905, 11195, 1134, 11626, 11987, 13359, 13966, 14908, 15179,
    18721, 193, 19401, 24909, 24961, 25358, 25907, 27213, 28378, 29135,
    4847, 31181, 3672, 4656, 6125, 6445, 6705, 6944, 7374, 7506, 7508,
    7982, 8840, 9428, 10515, 10576, 10939, 2328, 5851, 8506
]


def calculate_current_value(stocks_current_prices):
    total_high = 0.0
    total_low = 0.0
    for key, value in stocks_current_prices:
        total_high += value["ohlc"]["high"]
        total_low += value["ohlc"]["low"]
    return total_high, total_low


def fetch_stock_values_from_dhan():
    current_high_price = 0.0
    current_low_price = 0.0
    try:
        dhan_object = dhanhq.dhanhq(CLIENT_ID, ACCESS_TOKEN)
        current_stock_data = dhan_object.quote_data({
            "NSE_EQ": security_ids,
        })

        # parse the data from
        current_high_price, current_low_price = calculate_current_value(current_stock_data["data"]["data"]["NSE_EQ"].items())
    except Exception as e:
        print(e)
    return current_high_price, current_low_price


def update_value_in_db_and_user():
    current_high_value, current_low_value = fetch_stock_values_from_dhan()

    doc = doc_ref.get()
    current_highest_value = doc.to_dict().get("highest_value", 0.0) if doc.exists else 0.0

    difference_percent = (current_high_value - current_low_value) / current_high_value
    if current_high_value > current_highest_value:
        current_highest_value = current_high_value
        doc_ref.update({"highest_value": current_highest_value})
        doc_ref.update({"difference_percent": 0.0})

    elif difference_percent > 0.04:
        send_email(
            "Stock Alert: Significant Drop",
            f"Current Value: {current_low_value}, Highest Value: {current_highest_value}",
        )
        doc_ref.update({"highest_value": current_highest_value})
        doc_ref.update({"difference_percent": difference_percent})
    else:
        doc_ref.update({"difference_percent": difference_percent})

    print(current_high_value, current_high_value, current_highest_value, difference_percent)


# Run Script
if __name__ == "__main__":
    update_value_in_db_and_user()
