import smtplib
from email.mime.text import MIMEText

EMAIL = "prasad0kulkarni.pk@gmail.com"
PASSWORD = "xies knqi leps oaos"
RECIPIENT = "prasad0kulkarni.pk@gmail.com"

def send_email(subject, body):
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL
    msg["To"] = RECIPIENT

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(EMAIL, PASSWORD)
        server.sendmail(EMAIL, RECIPIENT, msg.as_string())
