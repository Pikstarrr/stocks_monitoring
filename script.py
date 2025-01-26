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

security_ids = [10417, 10905, 11195, 11626, 11987, 13359, 13966, 14908, 15178, 15179,
 18721, 193, 19401, 24909, 24961, 25358, 25907, 27213, 28378, 28903,
 29135, 31181, 3672, 4656, 4847, 6125, 6445, 6705, 6944, 7374,
 7506, 7508, 7982, 8840, 9087, 9428, 10515, 10576, 10939, 14428,
 21750, 2328, 3010, 522, 5851, 8506]


def calculate_current_value(stocks_current_prices):
    total_current = 0.0
    for key, value in stocks_current_prices:
        total_current += value["last_price"]
    return total_current


def fetch_stock_values_from_dhan():
    current_high_price = 0.0
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
    return current_price


def update_value_in_db_and_user():
    current_value = fetch_stock_values_from_dhan()

    if current_value == 0:
        return

    doc = doc_ref.get()
    current_highest_value = doc.to_dict().get("highest_value", 0.0) if doc.exists else 0.0

    print("Before :")
    print(current_value, current_highest_value)
    if current_value > current_highest_value:
        current_highest_value = current_value
        doc_ref.update({"highest_value": current_highest_value})
        doc_ref.update({"difference_percent": 0.0})
    else:
        difference_percent = ((current_highest_value - current_value) / current_highest_value) * 100
        if difference_percent > 4:
            send_email(
                "Stock Alert: Significant Drop",
                f"Current Value: {current_value}, Highest Value: {current_highest_value}",
            )
            current_highest_value = current_value
            doc_ref.update({"highest_value": current_highest_value})
            doc_ref.update({"difference_percent": 0.0})
        else:
            doc_ref.update({"difference_percent": difference_percent})
        print(difference_percent)

    print("After : ")
    print(current_value, current_highest_value)


# Run Script
if __name__ == "__main__":
    update_value_in_db_and_user()
