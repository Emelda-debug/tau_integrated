import requests
import json

base_url = "http://127.0.0.1:8000"

def test_ivr():
    response = requests.post(f"{base_url}/ivr")
    print("IVR Response:", response.text)

    response = requests.post(f"{base_url}/process_speech", data={'SpeechResult': 'Hi'})
    print("Process Speech Response:", response.text)

def test_send_whatsapp():
    response = requests.post(f"{base_url}/send-whatsapp", json={"phone_number": "+263773344079"})
    print("Send WhatsApp Response:", response.json())

def test_call_user():
    global webhook_url
    webhook_url = "http://your-webhook-url.com"  # Define the webhook_url here
    response = requests.post(f"{base_url}/call-user", json={"phone_number": "+263773344079"})
    print("Call User Response:", response.json())

def test_send_email():
    response = requests.post(f"{base_url}/send-email", json={"email": "test@example.com"})
    print("Send Email Response:", response.json())

if __name__ == "__main__":
    test_ivr()
    test_send_whatsapp()
    test_call_user()
    test_send_email()
