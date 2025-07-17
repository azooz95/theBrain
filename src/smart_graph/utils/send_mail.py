import os
import sendgrid
from sendgrid.helpers.mail import Mail, Email, To, Content

SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")  # Store API key as environment variable.

def send_email(from_email: str, to_email: str, subject: str, message: str) -> int:
    """
    Sends an email using SendGrid API.
    
    Args:
        from_email (str): Sender email (must be verified in SendGrid)..
        to_email (str): Recipient email.
        subject (str): Email subject.
        message (str): Email body text.

    Returns:
        int: HTTP status code from SendGrid response.
    """
    sg = sendgrid.SendGridAPIClient(api_key=SENDGRID_API_KEY)
    content = Content("text/plain", message)
    mail = Mail(Email(from_email), To(to_email), subject, content)
    response = sg.client.mail.send.post(request_body=mail.get())

    print(f"Status Code: {response.status_code}")
    print(f"Headers: {response.headers}")

    return response.status_code