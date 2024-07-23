import os
from twilio.rest import Client
from dotenv import load_dotenv

load_dotenv()

account_sid = os.getenv("my_account_sid")
auth_token = os.getenv("my_auth_token")

client = Client(account_sid, auth_token)

try:
    account = client.api.accounts(account_sid).fetch()
    print("Authenticated successfully!")
    print(account.friendly_name)
except Exception as e:
    print(f"Error: {e}")
