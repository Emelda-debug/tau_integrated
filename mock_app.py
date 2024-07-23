import os
import ast
import json
import pandas as pd
from scipy import spatial
from flask import Flask, request, jsonify
from twilio.twiml.voice_response import VoiceResponse
from twilio.rest import Client
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv
from twilio.twiml.messaging_response import MessagingResponse
from textblob import TextBlob
import sendgrid
from sendgrid.helpers.mail import Mail
from sendgrid import SendGridAPIClient

# Import the mock client
from mock_twilio import mock_twilio_client

load_dotenv()

app = Flask(__name__)

# SendGrid API key
sendgrid_api_key = os.getenv("SENDGRID_API_KEY")
sendgrid_client = SendGridAPIClient(sendgrid_api_key)

# Twilio account credentials
account_sid = os.getenv("my_account_sid")
auth_token = os.getenv("my_auth_token")

# Comment out real Twilio client and use mock client
# client = Client(account_sid, auth_token)
client = mock_twilio_client

# Comment out real webhook update
# phone_number_sid = os.getenv("my_phone_number_sid")
# webhook_url = "https://your-ngrok-url/ivr"
# phone_number = client.incoming_phone_numbers(phone_number_sid).fetch()
# phone_number.update(voice_url=webhook_url, voice_method='POST')

# Whatsapp credentials
whatsapp_number = "whatsapp:+14155238886"
whatsapp_account_sid = os.getenv("whatsapp_sid")
whatsapp_auth_token = os.getenv("whatsapp_auth")
# whatsapp_client = Client(whatsapp_account_sid, whatsapp_auth_token)
whatsapp_client = mock_twilio_client

# OpenAI API key
openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

embeddings_path = "DATASET/emdeddings_dataset.csv"
EMBEDDING_MODEL = "text-embedding-3-small"
GPT_MODEL = "gpt-3.5-turbo"
df = pd.read_csv(embeddings_path)
df['embedding'] = df['embedding'].apply(ast.literal_eval)

# ... (rest of your code remains unchanged)
def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100
) -> tuple[list[str], list[float]]:
    query_embedding_response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL, input=query,
    )
    query_embedding = query_embedding_response.data[0].embedding
    strings_and_relatednesses = [
        (row["text"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]

def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def query_message(
    query: str, df: pd.DataFrame, model: str, token_budget: int
) -> str:
    strings, relatednesses = strings_ranked_by_relatedness(query, df)
    
    # Log ranked strings and relatedness scores
    app.logger.info(f"Ranked strings: {strings}")
    app.logger.info(f"Relatedness scores: {relatednesses}")

    introduction = 'Use the below information from the Star International. Answer as a virtual assistant for the company. Try your best to answer all the questions using the provided information. If the answer cannot be found in the info, write "Sorry, I can not fully answer that, instead let me refer you to my colleague, who will reach out shortly, if they delay please, contact our number, 0 7 7 8 0 4 0 4 9 7 3 or visit our website (www.starinternational.co.zw) for more information."'
    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in strings:
        next_article = f'\n\nINFORMATION FOR Star International:\n"""\n{string}\n"""'
        if num_tokens(message + next_article + question, model=model) > token_budget:
            break
        else:
            message += next_article
    
    final_message = message + question
    
    # Log the final constructed message
    app.logger.info(f"Constructed message: {final_message}")

    return final_message

def ask(
    query: str, df: pd.DataFrame = df,
    model: str = GPT_MODEL, token_budget: int = 4096 - 500, print_message: bool = False,
) -> str:
    message = query_message(query, df, model=model, token_budget=token_budget)
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": "You answer questions about Star International and persuade customers to use the transporting services. Be friendly and empathetic."},
        {"role": "user", "content": message},
    ]
    response = openai_client.chat.completions.create(model=model, messages=messages, temperature=0)
    response_message = response.choices[0].message.content
    
    # Log the response from OpenAI
    app.logger.info(f"Response from OpenAI: {response_message}")
    
    return response_message

# Counter to track the number of interactions
interaction_counter = 0

# Allowed maximum number of interactions before ending the call
MAX_INTERACTIONS = 15

def load_customers():
    with open('customers.json', 'r') as file:
        return json.load(file)

customers = load_customers()

existing_customers = customers["existing_customers"]

def get_customer_status(phone_number):
    if phone_number in existing_customers:
        return "existing"
    else:
        return "new"

@app.route('/', methods=['GET'])
def home():
    return "Welcome to Tau's IVR"

def handle_conversation(speech_input, customer_status, response_type):
    global interaction_counter, greeted, asked_about_loads, asked_about_wellbeing

    # Initialize variables if not already defined
    if 'asked_about_loads' not in globals():
        asked_about_loads = False
    if 'asked_about_wellbeing' not in globals():
        asked_about_wellbeing = False
    if 'greeted' not in globals():
        greeted = False

    # Define the response object based on response type
    if response_type == 'voice':
        response = VoiceResponse()
    else:
        response = MessagingResponse()
        message = response.message()

    # Handle initial greeting
    if not greeted:
        if customer_status == "new":
            greeting_message = "Hi, this is Tau from Star International. How are you doing today?"
        else:
            greeting_message = "Hi, this is Tau from Star International. How can we assist you today?"
        if response_type == 'voice':
            response.say(greeting_message, voice='Polly.Gregory-Neural')
        else:
            message.body(greeting_message)
        greeted = True
        interaction_counter += 1
        return response

    # Sentiment analysis and follow-up
    if not asked_about_wellbeing:
        sentiment = TextBlob(speech_input).sentiment.polarity
        if sentiment < -0.1:  # More negative sentiment threshold
            follow_up_message = "I'm sorry to hear that you're not feeling well. What's wrong?"
        elif sentiment > 0.1:  # More positive sentiment threshold
            follow_up_message = "Glad to hear you're doing well! I just wanted to check in, do you have any loads for us to carry?"
        else:
            follow_up_message = "I hope you're doing okay. I just wanted to check in,do you have any loads for us to carry?"
        if response_type == 'voice':
            response.say(follow_up_message, voice='Polly.Gregory-Neural')
        else:
            message.body(follow_up_message)
        asked_about_wellbeing = True
        interaction_counter += 1
        return response

    # Handle the response to the loads question
    if asked_about_wellbeing and not asked_about_loads:
        sentiment = TextBlob(speech_input).sentiment.polarity
        if sentiment > 0.1:  # Positive sentiment
            follow_up_message = "Great! Could you please provide more information about the load?"
            asked_about_loads = True
        elif sentiment < -0.1:  # Negative sentiment
            follow_up_message = "I understand. If there's anything else we can assist with, please let us know."
            interaction_counter = 0  # Reset counter
            greeted = False  # Reset greeting flag
            asked_about_loads = False
            asked_about_wellbeing = False
        else:
            follow_up_message = "I'm sorry, I didn't quite catch that. I just wanted to check in, do you have any loads you'd like us to carry?"
        if response_type == 'voice':
            response.say(follow_up_message, voice='Polly.Gregory-Neural')
        else:
            message.body(follow_up_message)
        interaction_counter += 1
        return response

    # End the interaction if max interactions are reached
    if interaction_counter >= MAX_INTERACTIONS:
        if response_type == 'voice':
            response.say("Thank you for your time.Have a great day Goodbye! and remember whenever you need any assistance, reach out I'm here anytime. ", voice='Polly.Gregory-Neural')
        else:
            message.body("Thank you for your time.Have a great day Goodbye! and remember whenever you need any assistance, reach out I'm here anytime")
        interaction_counter = 0  # Reset counter
        greeted = False  # Reset greeting flag
        asked_about_loads = False
        asked_about_wellbeing = False
        return response

    # Handle default response if no conditions are met
    default_message = "Hello, how can I assist you today?"
    if response_type == 'voice':
        response.say(default_message, voice='Polly.Gregory-Neural')
    else:
        message.body(default_message)
    interaction_counter += 1
    return response

@app.route('/ivr', methods=['POST'])
def ivr():
    response = VoiceResponse()
    gather = response.gather(
        input='speech',
        timeout=3,
        action='/process_speech',
        method='POST'
    )
    gather.say('Hello, how can I assist you today?', voice='Polly.Gregory-Neural')
    return str(response)

@app.route('/process_speech', methods=['POST'])
def process_speech():
    global interaction_counter, asked_about_loads
    speech_input = request.values.get('SpeechResult', '')
    phone_number = request.values.get('From', '')
    customer_status = get_customer_status(phone_number)
    
    if phone_number not in customers['existing_customers'] and phone_number not in customers['new_customers']:
        response = VoiceResponse()
        response.say("Phone number not found in customers list.", voice='Polly.Gregory-Neural')
        response.hangup()
        return str(response)

    # Log the speech input
    app.logger.info(f"Incoming speech input: {speech_input}")

    response = handle_conversation(speech_input, customer_status, response_type='voice')
    return str(response)

@app.route('/whatsapp', methods=['POST'])
def handle_whatsapp():
    incoming_msg = request.values.get('Body', '').lower()
    phone_number = request.values.get('From', '')
    customer_status = get_customer_status(phone_number)

    if phone_number not in customers['existing_customers'] and phone_number not in customers['new_customers']:
        return jsonify({'error': 'Phone number not found in customers.json'}), 404

    # Log the incoming WhatsApp message
    app.logger.info(f"Incoming WhatsApp message: {incoming_msg}")

    response = handle_conversation(incoming_msg, customer_status, response_type='whatsapp')
    return str(response)

@app.route('/send-whatsapp', methods=['POST'])
def send_whatsapp():
    to = request.json.get('to')
    body = request.json.get('body')
    
    if not to or not body:
        return jsonify({'error': 'Missing "to" or "body" in request'}), 400
    
    if to not in customers['existing_customers'] and to not in customers['new_customers']:
        return jsonify({'error': 'Phone number not found in customers.json'}), 404
    
    message = whatsapp_client.messages.create(
        body=body,
        from_=whatsapp_number,
        to=to
    )
    
    return jsonify({'message_sid': message.sid}), 200

@app.route('/call-user', methods=['POST'])
def call_user():
    try:
        # Extract phone number from request
        phone_number = request.json.get('phone_number')
        
        if not phone_number:
            return jsonify({'error': 'Missing "phone_number" in request'}), 400
        
        # Check if the phone number exists in the customer lists
        if phone_number not in customers['existing_customers'] and phone_number not in customers['new_customers']:
            return jsonify({'error': 'Phone number not found in customers.json'}), 404

        # Call the customer using Twilio
        call = client.calls.create(
            #url=webhook_url,  # The URL of your IVR endpoint
           # to=phone_number,  # The phone number of the customer
          #  from_=os.getenv("TWILIO_PHONE_NUMBER")  # Your Twilio phone number
        )

        # Logging call initiation
        app.logger.info(f"Call initiated. Call SID: {call.sid}")
        return jsonify({"message": "Call initiated", "call_sid": call.sid}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/send-email', methods=['POST'])
def send_email():
    data = request.json
    to_email = data['to_email']
    subject = data['subject']
    content = data['content']
    
    status_code = send_email(to_email, subject, content)
    if status_code == 202:
        return jsonify({"message": "Email sent successfully"}), 200
    else:
        return jsonify({"message": "Failed to send email"}), 500

@app.route('/process-email', methods=['POST'])
def process_email():
    email_content = request.json.get('email_content', '')
    customer_email = request.json.get('from_email', '')
    customer_status = get_customer_status(customer_email)
    
    if customer_email not in customers['existing_customers'] and customer_email not in customers['new_customers']:
        return jsonify({'error': 'Email address not found in customers.json'}), 404

    # Log the incoming email content
    app.logger.info(f"Incoming email content: {email_content}")

    response = handle_conversation(email_content, customer_status, response_type='email')
    if 'subject' in response and 'body' in response:
        send_email(customer_email, response['subject'], response['body'])
        return jsonify({"message": "Response email sent successfully"}), 200
    else:
        return jsonify({"message": "Failed to process email response"}), 500

if __name__ == '__main__':
    app.run(port=8000, debug=True)

