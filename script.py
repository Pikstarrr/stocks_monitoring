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

security_ids = {
    'DYCL': 10417,
    'MPSLTD': 10578,
    'GENESYS': 10905,
    'INDIGO': 11195,
    'GOKEX': 11778,
    'ICIL': 11987,
    'KFINTECH': 13359,
    'AARTIPHARM': 13868,
    'HBLENGINE': 13966,
    'KDDL': 14908,
    'TARIL': 15178,
    'ECLERX': 15179,
    'HCG': 15555,
    'SUVENPHAR': 17945,
    'NUVAMA': 18721,
    'BLUEJET': 19686,
    'INTERARCH': 24909,
    'ORIENTTECH': 24961,
    'TDPOWERSYS': 25178,
    'PGEL': 25358,
    'WAAREEENER': 25907,
    'EIEL': 27213,
    'KITEX': 28903,
    'INDUSTOWER': 29135,
    'MCX': 31181,
    'BORORENEW': 3155,
    'SBCL': 4656,
    'AMIORG': 5578,
    'PAYTM': 6705,
    'GANECOS': 6944,
    'WOCKPHARMA': 7506,
    'ZENTEC': 7508,
    'GRWRHITECH': 7982,
    'PIXTRANS': 9087,
    'SKIPPER': 9428,
    'AXISCADES': 9440,
    'UTISENSETF': 10515,
    'NIFTYBEES': 10576,
    'JUNIORBEES': 10939,
    'GOLDBEES': 14428,
    'LIQUIDCASE': 21750,
    'CPSEETF': 2328,
    'BANKNIFTY1': 5851,
    'MID150BEES': 8506,
    'AXISTECETF': 3010,
    'ICICIB22': 522,
    'ONESOURCE': 29224
}


def calculate_current_value(stocks_current_prices):
    total_current = 0.0
    for key, value in stocks_current_prices:
        total_current += value["last_price"]
    return total_current


def fetch_stock_values_from_dhan():
    current_price = 0.0
    try:
        dhan_object = dhanhq.dhanhq(CLIENT_ID, ACCESS_TOKEN)
        current_stock_data = dhan_object.quote_data({
            "NSE_EQ": list(security_ids.values()),
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
        if difference_percent > 5:
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
