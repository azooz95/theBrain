
from dotenv import load_dotenv
import os 
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
load_dotenv("app.env")

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_USER_ID = os.getenv("SLACK_USER_ID")
SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = os.getenv("SMTP_PORT")
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS","")
APP_PASSWORD = os.getenv("APP_PASSWORD")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  
TRELLO_API_KEY = os.getenv("TRELLO_API_KEY")
TRELLO_TOKEN = os.getenv("TRELLO_TOKEN")
BASE_URL = os.getenv("BASE_URL")

client = WebClient(token=SLACK_BOT_TOKEN)

def send_email_notification(sender_email, sender_password ,to_email: str, subject: str, body: str) -> bool:
        """Send an email notification"""
        try:
            msg = MIMEMultipart()
            msg["From"] = sender_email
            msg["To"] = to_email
            msg["Subject"] = subject

            msg.attach(MIMEText(body, "plain"))

            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.send_message(msg)

            return f"Successfully sent email to {to_email}"

        except Exception as e:
            return f"Error sending email: {e}"


def get_user_id_by_email(email):
  """
  this function is used to get the slack user id of the person you want to send the message to and it's used in the send_slack_message function
  """
  try:
        response = client.users_lookupByEmail(email=email)
        user_id = response["user"]["id"]
        print(f"User ID for {email}: {user_id}")
        return user_id
  except SlackApiError as e:
        print(f"Error finding user by email: {e.response['error']}")
        return None


def send_slack_message(message: str, user:str = None  ,Email:str = None)->None:
  if Email:
    user=get_user_id_by_email(Email)

  try:
        response = client.chat_postMessage(
            channel=user,
            text=message
        )
        print(f"Message sent: {response['ts']}")
  except SlackApiError as e:
        print(f"Error sending message: {e.response['error']}")


send_slack_message(SLACK_USER_ID, "🚀 Task report is ready! Check the dashboard.")