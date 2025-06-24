import smtplib
from email.mime.text import MIMEText
import os

from dotenv import load_dotenv

# EMAIL = os.environ["SENDER_EMAIL"]
# PASSWORD = os.environ["MAIL_PASSWORD"]
# RECIPIENT = os.environ["RECEIVER_MAIL"]

load_dotenv()
EMAIL = os.getenv("SENDER_EMAIL")
PASSWORD = os.getenv("MAIL_PASSWORD")
RECIPIENT = os.getenv("RECEIVER_MAIL")

def send_email(subject, body):
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL
    msg["To"] = RECIPIENT

    msg["X-Priority"] = "1"
    msg["Importance"] = "High"
    msg["X-MSMail-Priority"] = "High"

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(EMAIL, PASSWORD)
        server.sendmail(EMAIL, RECIPIENT, msg.as_string())
