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
from sendgrid.helpers.mail import Mail, Email, To, Content


load_dotenv()

app = Flask(__name__)

# SendGrid API key
sendgrid_api_key = os.getenv("SENDGRID_API_KEY")
sendgrid_client = SendGridAPIClient(sendgrid_api_key)

# Twilio account credentials
account_sid = os.getenv("my_account_sid")
auth_token = os.getenv("my_auth_token")

# Creating the Twilio client
client = Client(account_sid, auth_token)

# Updating the webhook URL for the Twilio phone
phone_number_sid = os.getenv("my_phone_number_sid")
webhook_url = "https://53ea-2605-6440-4000-e000-00-2856.ngrok-free.app/ivr"
phone_number = client.incoming_phone_numbers(phone_number_sid).fetch()
phone_number.update(voice_url=webhook_url, voice_method='POST')

# Whatsapp credentials
whatsapp_number = "whatsapp:+14155238886"
whatsapp_account_sid = os.getenv("whatsapp_sid")
whatsapp_auth_token = os.getenv("whatsapp_auth")
whatsapp_client = Client(whatsapp_account_sid, whatsapp_auth_token)

# OpenAI API key
openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

# Loading datasets
with open('customers.json') as f:
    customers = json.load(f)

with open('customer_trades.json') as f:
    customer_trades = json.load(f)

with open('customer_emails.json') as f:
    customer_emails = json.load(f)

embeddings_path = "DATASET/emdeddings_dataset.csv"
df = pd.read_csv(embeddings_path)
df['embedding'] = df['embedding'].apply(ast.literal_eval)

# Defining models and API keys
EMBEDDING_MODEL = "text-embedding-3-small"
GPT_MODEL = "gpt-3.5-turbo"

# Helper Functions
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

    introduction = 'Use the below information from Star International. Answer as a virtual assistant and marketing agent for the company. Try your best to answer all the questions using the provided information. If the answer cannot be found in the info, write "Sorry, I can not fully answer that, instead let me refer you to my colleague, who will reach out shortly, if they delay please, contact our number, 0 7 7 8 0 4 0 4 9 7 3 or visit our website (www.starinternational.co.zw) for more information."'
    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in strings:
        next_article = f'\n\nINFORMATION FOR Star International:\n"""\n{string}\n"""'
        if num_tokens(message + next_article + question, model=model) > token_budget:
            break
        else:
            message += next_article
    
    final_message = message + question
    
    # Logging the final constructed message
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
    
    # Logging the response from OpenAI
    app.logger.info(f"Response from OpenAI: {response_message}")
    
    return response_message

def get_customer_status(phone_number):
    if phone_number in customers['existing_customers']:
        return "existing"
    elif phone_number in customers["new_customers"]:
        return "new"
    else:
        return "uknown"


def send_email(to_email, subject, content):
    sg = SendGridAPIClient(api_key=os.getenv('SENDGRID_API_KEY'))
    from_email = Email('emeldam@starinternational.co.zw')
    to_email = To(to_email)
    content = Content('text/plain', content)
    mail = Mail(from_email, to_email, subject, content)
    response = sg.send(mail)
    return response

# Global Variables
interaction_counter = 0
MAX_INTERACTIONS = 15
asked_about_loads = False
asked_about_wellbeing = False
asked_about_business = False


@app.route('/', methods=['GET'])
def home():
    return "Welcome to Tau's IVR"

def handle_conversation(speech_input, customer_status, response_type, phone_number):
    global interaction_counter, greeted, asked_about_loads, asked_about_wellbeing, asked_about_business

    # Variable Initialization
    if 'asked_about_loads' not in globals():
        asked_about_loads = False
    if 'asked_about_wellbeing' not in globals():
        asked_about_wellbeing = False
    if 'greeted' not in globals():
        greeted = False
    if 'asked_about_business' not in globals():
        asked_about_business = False

    if response_type == 'voice':
        response = VoiceResponse()
    elif response_type == 'email':
        response = {}
    else:
        response = MessagingResponse()
        message = response.message()

    def handle_new_customer_conversation():
        global asked_about_business, asked_about_wellbeing, asked_about_loads
        customer_trade = customer_trade.get(phone_number, "your industry")

        if not greeted:
            greeting_message = "Hi, this is Tau from Star International. We are a transport and logistics company in Harare. How are you doing today?"
            if response_type == 'voice':
                response.say(greeting_message, voice='Polly.Gregory-Neural')
            elif response_type == 'email':
                response['subject'] = "Welcome to Star International"
                response['body'] = greeting_message
            greeted = True
            interaction_counter += 1
            return response

        if not asked_about_business:
            business_intro = f"I see you are in the {customer_trade} business. We at Star International understand how crucial reliable transport and logistics are for {customer_trade}. How is business going for you?"
            if response_type == 'voice':
                response.say(business_intro, voice='Polly.Gregory-Neural')
            elif response_type == 'email':
                response['body'] = business_intro
            asked_about_business = True
            interaction_counter += 1
            return response

        if asked_about_business and not asked_about_wellbeing:
            sentiment = TextBlob(speech_input).sentiment.polarity
            if sentiment < -0.1:
                follow_up_message = "I'm sorry to hear that you're facing challenges. If there's anything specific we can do to help, please let us know."
            elif sentiment > 0.1:
                follow_up_message = "That's great to hear! Would you like to know more about how Star International can enhance your business operations with our transport and logistics services?"
            else:
                follow_up_message = "I hope things are going okay. Would you be interested in learning how Star International can support your business with our transport and logistics services?"
            if response_type == 'voice':
                response.say(follow_up_message, voice='Polly.Gregory-Neural')
            elif response_type == 'email':
                response['body'] = follow_up_message
            asked_about_wellbeing = True
            interaction_counter += 1
            return response

        if asked_about_wellbeing and not asked_about_loads:
            sentiment = TextBlob(speech_input).sentiment.polarity
            if sentiment > 0.1:
                follow_up_message = "Fantastic! Let me share more about our services and how they can benefit your business."
                query = "why should customers work with Star International?"
                response_message = ask(query, df)
            else:
                follow_up_message = "I understand. When would be a convenient time for us to reach out again? We can discuss how our services can align with your needs."
            if response_type == 'voice':
                response.say(follow_up_message + " " + response_message, voice='Polly.Gregory-Neural')
            elif response_type == 'email':
                response['body'] = follow_up_message + " " + response_message
            asked_about_loads = True
            interaction_counter += 1
            return response

    def handle_existing_customer_conversation():
        global asked_about_loads, asked_about_wellbeing
        
        if not greeted:
            greeting_message = "Hi, this is Tau from Star International. How are you?"
            if response_type == 'voice':
                response.say(greeting_message, voice='Polly.Gregory-Neural')
            elif response_type == 'email':
                response['subject'] = "Checking In"
                response['body'] = greeting_message
            greeted = True
            interaction_counter += 1
            return response

        if not asked_about_wellbeing:
            sentiment = TextBlob(speech_input).sentiment.polarity
            if sentiment < -0.1:
                follow_up_message = "I'm sorry to hear that you're not feeling well. What's wrong?"
            elif sentiment > 0.1:
                follow_up_message = "Glad to hear you're doing well! So, I just wanted to check in with you, do you have any loads you'd like us to transport for you?"
            else:
                follow_up_message = "I hope you're doing okay. I just wanted to check in, do you have any loads you'd like for us to transport for you?"
            if response_type == 'voice':
                response.say(follow_up_message, voice='Polly.Gregory-Neural')
            elif response_type == 'email':
                response['body'] = follow_up_message
            asked_about_wellbeing = True
            interaction_counter += 1
            return response

        if asked_about_wellbeing and not asked_about_loads:
            sentiment = TextBlob(speech_input).sentiment.polarity
            if sentiment > 0.1:
                follow_up_message = "Great! Could you please provide more information about the load?"
                asked_about_loads = True
            elif sentiment < -0.1:
                follow_up_message = "I understand. Please let me know when you have any loads you need transported. If there's anything else I can assist with, please let me know. You can call, text or email. In the meantime, you can also check out our website https://www.starinternational.co.zw to see what we are up to."
                interaction_counter = 0
                greeted = False
                asked_about_loads = False
                asked_about_wellbeing = False
            else:
                follow_up_message = "I'm sorry, I didn't quite catch that. would you mind repeating?"
            if response_type == 'voice':
                response.say(follow_up_message, voice='Polly.Gregory-Neural')
            elif response_type == 'email':
                response['body'] = follow_up_message
            interaction_counter += 1
            return response
        
    if customer_status == "unknown":
        if response_type == 'voice':
            response.say("Phone number not found in customers list.", voice='Polly.Gregory-Neural')
            response.hangup()
        elif response_type == 'email':
            response['subject'] = "Customer Not Found"
            response['body'] = "Email address not found in customers list."
        return response

    if customer_status == "new":
        return handle_new_customer_conversation()

    if customer_status == "existing":
        return handle_existing_customer_conversation()

    return response

@app.route('/call-user', methods=['POST'])
def call_user():
    try:
        phone_number = request.json.get('phone_number')

        if not phone_number:
            return jsonify({'error': 'Missing "phone_number" in request'}), 400

        if phone_number not in customers['existing_customers'] and phone_number not in customers['new_customers']:
            return jsonify({'error': 'Phone number not found in customers.json'}), 404

        call = client.calls.create(
            url=webhook_url,  # The URL of your IVR endpoint
            to=phone_number,
            from_=os.getenv("TWILIO_PHONE_NUMBER")
        )

        app.logger.info(f"Call initiated. Call SID: {call.sid}")
        return jsonify({"message": "Call initiated", "call_sid": call.sid}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/ivr', methods=['POST'])
def ivr():
    response = VoiceResponse()
    gather = response.gather(
        input='speech',
        timeout=3,
        action='/process_speech',
        method='POST'
    )
    gather.say('Tau from Star International, how can I assist you today?', voice='Polly.Gregory-Neural')
    return str(response)

@app.route('/process_speech', methods=['POST'])
def process_speech():
    speech_input = request.values.get('SpeechResult', '')
    phone_number = request.values.get('From', '')

    if not phone_number:
        response = VoiceResponse()
        response.say("Missing phone number.", voice='Polly.Gregory-Neural')
        response.hangup()
        return str(response)

    customer_status = get_customer_status(phone_number)

    if customer_status == 'unknown':
        response = VoiceResponse()
        response.say("Phone number not found in customers list.", voice='Polly.Gregory-Neural')
        response.hangup()
        return str(response)

    app.logger.info(f"Incoming speech input: {speech_input}")

    response = handle_conversation(speech_input, customer_status, response_type='voice', phone_number=phone_number)
    return str(response)



@app.route('/send-whatsapp', methods=['POST'])
def send_whatsapp():
    to = request.json.get('to')
    body = request.json.get('body')
    
    if not to or not body:
        return jsonify({'error': 'Missing "to" or "body" in request'}), 400
    
    # Determine customer status
    customer_status = get_customer_status(to)
    
    if to not in customers['existing_customers'] and to not in customers['new_customers']:
        return jsonify({'error': 'Phone number not found in customers.json'}), 404
    
    # Process conversation based on customer status
    response = handle_conversation(body, customer_status, response_type='whatsapp', phone_number=to)
    
    # Send the message
    message = whatsapp_client.messages.create(
        body=response,
        from_=whatsapp_number,
        to=to
    )
    
    return jsonify({'message_sid': message.sid}), 200


@app.route('/process-whatsapp', methods=['POST'])
def process_whatsapp():
    incoming_msg = request.values.get('Body', '').strip()
    from_number = request.values.get('From', '').strip()

    if not incoming_msg or not from_number:
        return '', 400

    # Determine customer status
    customer_status = get_customer_status(from_number)

    # Process conversation based on customer status
    response = handle_conversation(incoming_msg, customer_status, response_type='whatsapp', phone_number=from_number)
    
    return str(response), 200

@app.route('/send-email', methods=['POST'])
def send_email_route():
    data = request.json
    to_email = data.get('to_email')
    subject = data.get('subject')
    content = data.get('content')

    if not to_email or not subject or not content:
        return jsonify({'error': 'Missing "to_email", "subject", or "content" in request'}), 400

    customer_status = get_customer_status(to_email)

    if customer_status == "unknown":
        return jsonify({'error': 'Email address not found in customers.json'}), 404

    try:
        response = send_email(to_email, subject, content)
        if response.status_code == 202:
            return jsonify({"message": "Email sent successfully"}), 200
        else:
            return jsonify({"message": "Failed to send email"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/process-email', methods=['POST'])
def process_email():
    email_content = request.json.get('email_content', '')
    customer_email = request.json.get('from_email', '')
    customer_status = get_customer_status(customer_email)

    if customer_email not in customers['existing_customers'] and customer_email not in customers['new_customers']:
        return jsonify({'error': 'Email address not found in customers.json'}), 404

    app.logger.info(f"Incoming email content: {email_content}")

    response = handle_conversation(email_content, customer_status, response_type='email', phone_number=customer_email)
    if 'subject' in response and 'body' in response:
        try:
            send_email(customer_email, response['subject'], response['body'])
            return jsonify({"message": "Response email sent successfully"}), 200
        except Exception as e:
            return jsonify({"message": "Failed to send email"}), 500
    else:
        return jsonify({"message": "Failed to process email response"}), 500


if __name__ == '__main__':
    app.run(port=8000, debug=True)
