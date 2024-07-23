import os
import ast
import json
import pandas as pd
from scipy import spatial
from flask import Flask, request, jsonify
from twilio.twiml.voice_response import VoiceResponse
from twilio.rest import Client as TwilioClient
from textblob import TextBlob
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Configure logging
app.logger.setLevel('INFO')

class MarketingAgent:
    def __init__(self):
        # Initialize API clients
        self.sendgrid_client = SendGridAPIClient(os.getenv("SENDGRID_API_KEY"))
        self.twilio_client = TwilioClient(
            os.getenv("my_account_sid"),
            os.getenv("my_auth_token")
        )
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Load datasets
        with open('customers.json') as f:
            self.customers = json.load(f)
        with open('customer_trades.json') as f:
            self.customer_trades = json.load(f)
        with open('customer_emails.json') as f:
            self.customer_emails = json.load(f)

        # Load embeddings dataset
        embeddings_path = "DATASET/emdeddings_dataset.csv"
        self.df = pd.read_csv(embeddings_path)
        self.df['embedding'] = self.df['embedding'].apply(ast.literal_eval)

        # Initialize global variables
        self.interaction_counter = 0
        self.MAX_INTERACTIONS = 15
        self.greeted = False
        self.asked_about_loads = False
        self.asked_about_wellbeing = False
        self.asked_about_business = False

    def strings_ranked_by_relatedness(self, query, top_n=100):
        query_embedding_response = self.openai_client.embeddings.create(
            model="text-embedding-3-small", input=query
        )
        query_embedding = query_embedding_response.data[0].embedding
        strings_and_relatednesses = [
            (row["text"], 1 - spatial.distance.cosine(query_embedding, row["embedding"]))
            for _, row in self.df.iterrows()
        ]
        strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
        strings, relatednesses = zip(*strings_and_relatednesses)
        return strings[:top_n], relatednesses[:top_n]

    def num_tokens(self, text, model="gpt-3.5-turbo"):
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))

    def query_message(self, query, model="gpt-3.5-turbo", token_budget=4096 - 500):
        strings, _ = self.strings_ranked_by_relatedness(query)
        introduction = 'Use the below information from Star International. Answer as a virtual assistant and marketing agent for the company...'
        question = f"\n\nQuestion: {query}"
        message = introduction
        for string in strings:
            next_article = f'\n\nINFORMATION FOR Star International:\n"""\n{string}\n"""'
            if self.num_tokens(message + next_article + question, model=model) > token_budget:
                break
            message += next_article
        final_message = message + question
        app.logger.info(f"Constructed message: {final_message}")
        return final_message

    def ask(self, query, model="gpt-3.5-turbo", token_budget=4096 - 500, print_message=False):
        message = self.query_message(query, model=model, token_budget=token_budget)
        if print_message:
            print(message)
        messages = [
            {"role": "system", "content": "You answer questions about Star International and persuade customers to use the transporting services. Be friendly and empathetic."},
            {"role": "user", "content": message}
        ]
        response = self.openai_client.chat.completions.create(model=model, messages=messages, temperature=0)
        response_message = response.choices[0].message.content
        app.logger.info(f"Response from OpenAI: {response_message}")
        return response_message

    def get_customer_status(self, phone_number):
        if phone_number in self.customers['existing_customers']:
            return "existing"
        elif phone_number in self.customers["new_customers"]:
            return "new"
        else:
            return "unknown"

    def send_email(self, to_email, subject, content):
        from_email = 'emeldam@starinternational.co.zw'
        mail = Mail(from_email, to_email, subject, content)
        response = self.sendgrid_client.send(mail)
        return response

    def handle_conversation(self, speech_input, customer_status, response_type, phone_number):
        response = VoiceResponse() if response_type == 'voice' else MessagingResponse() if response_type == 'sms' else {}

        if customer_status == "unknown":
            if response_type == 'voice':
                response.say("Phone number not found in customers list.", voice='Polly.Gregory-Neural')
                response.hangup()
            elif response_type == 'email':
                response['subject'] = "Customer Not Found"
                response['body'] = "Email address not found in customers list."
            return response

        if customer_status == "new":
            if not self.greeted:
                greeting_message = "Hi, this is Tau from Star International. We are a transport and logistics company in Harare. How are you doing today?"
                if response_type == 'voice':
                    response.say(greeting_message, voice='Polly.Gregory-Neural')
                elif response_type == 'email':
                    response['subject'] = "Welcome to Star International"
                    response['body'] = greeting_message
                self.greeted = True
                self.interaction_counter += 1
                return response

            if not self.asked_about_business:
                customer_trade = self.customer_trades.get(phone_number, "your industry")
                business_intro = f"I see you are in the {customer_trade} business. We at Star International understand how crucial reliable transport and logistics are for {customer_trade}. How is business going for you?"
                if response_type == 'voice':
                    response.say(business_intro, voice='Polly.Gregory-Neural')
                elif response_type == 'email':
                    response['body'] = business_intro
                self.asked_about_business = True
                self.interaction_counter += 1
                return response

            if self.asked_about_business and not self.asked_about_wellbeing:
                sentiment = TextBlob(speech_input).sentiment.polarity
                follow_up_message = "I'm sorry to hear that you're facing challenges. If there's anything specific we can do to help, please let us know." if sentiment < -0.1 else "That's great to hear! Would you like to know more about how Star International can enhance your business operations with our transport and logistics services?" if sentiment > 0.1 else "I hope things are going okay. Would you be interested in learning how Star International can support your business with our transport and logistics services?"
                if response_type == 'voice':
                    response.say(follow_up_message, voice='Polly.Gregory-Neural')
                elif response_type == 'email':
                    response['body'] = follow_up_message
                self.asked_about_wellbeing = True
                self.interaction_counter += 1
                return response

            if self.asked_about_wellbeing and not self.asked_about_loads:
                sentiment = TextBlob(speech_input).sentiment.polarity
                follow_up_message = "Fantastic! Let me share more about our services and how they can benefit your business." if sentiment > 0.1 else "I understand. When would be a convenient time for us to reach out again? We can discuss how our services can align with your needs."
                query = "why should customers work with Star International?"
                response_message = self.ask(query)
                if response_type == 'voice':
                    response.say(follow_up_message + " " + response_message, voice='Polly.Gregory-Neural')
                elif response_type == 'email':
                    response['body'] = follow_up_message + " " + response_message
                self.asked_about_loads = True
                self.interaction_counter += 1
                return response

        if customer_status == "existing":
            if not self.greeted:
                greeting_message = "Hi, this is Tau from Star International. How are you?"
                if response_type == 'voice':
                    response.say(greeting_message, voice='Polly.Gregory-Neural')
                elif response_type == 'email':
                    response['subject'] = "Checking In"
                    response['body'] = greeting_message
                self.greeted = True
                self.interaction_counter += 1
                return response

            if not self.asked_about_wellbeing:
                sentiment = TextBlob(speech_input).sentiment.polarity
                follow_up_message = "I'm sorry to hear that you're not feeling well. What's wrong?" if sentiment < -0.1 else "Glad to hear you're doing well! So, I just wanted to check in with you, do you have any loads you'd like us to transport for you?" if sentiment > 0.1 else "I hope you're doing okay. I just wanted to check in, do you have any transportation needs we can assist with?"
                if response_type == 'voice':
                    response.say(follow_up_message, voice='Polly.Gregory-Neural')
                elif response_type == 'email':
                    response['body'] = follow_up_message
                self.asked_about_wellbeing = True
                self.interaction_counter += 1
                return response

            if self.asked_about_wellbeing:
                sentiment = TextBlob(speech_input).sentiment.polarity
                if sentiment > 0.1:
                    return "Great to hear you're feeling well! If there's anything specific you need, just let us know."
                else:
                    return "Sorry to hear you're not feeling great. If there's anything we can do to assist you, please don't hesitate to reach out."

        if self.interaction_counter >= self.MAX_INTERACTIONS:
            end_message = "Thank you for your time. If you need any more assistance, please feel free to reach out to us. Have a great day!"
            if response_type == 'voice':
                response.say(end_message, voice='Polly.Gregory-Neural')
                response.hangup()
            elif response_type == 'email':
                response['subject'] = "Thank You"
                response['body'] = end_message
            return response

        return response

@app.route('/voice', methods=['POST'])
def handle_voice():
    phone_number = request.form['From']
    speech_input = request.form['SpeechInput']
    agent = MarketingAgent()
    customer_status = agent.get_customer_status(phone_number)
    response = agent.handle_conversation(speech_input, customer_status, 'voice', phone_number)
    return str(response)

@app.route('/sms', methods=['POST'])
def handle_sms():
    phone_number = request.form['From']
    sms_body = request.form['Body']
    agent = MarketingAgent()
    customer_status = agent.get_customer_status(phone_number)
    response = agent.handle_conversation(sms_body, customer_status, 'sms', phone_number)
    return str(response)

@app.route('/email', methods=['POST'])
def handle_email():
    email_address = request.form['email']
    subject = request.form['subject']
    body = request.form['body']
    response = MarketingAgent().send_email(email_address, subject, body)
    return jsonify({'status': 'email sent', 'response': response.status_code})

if __name__ == '__main__':
    app.run(debug=True)
