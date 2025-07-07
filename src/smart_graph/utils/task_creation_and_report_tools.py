import os
from dotenv import load_dotenv
from utils.trello import trello
import requests
from utils.send_notifications import send_email_notification
from pydantic import BaseModel, Field 
from typing import Optional, List 
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv("app.env")
SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = os.getenv("SMTP_PORT")
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS","")
APP_PASSWORD = os.getenv("APP_PASSWORD")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  
TRELLO_API_KEY = os.getenv("TRELLO_API_KEY")
TRELLO_TOKEN = os.getenv("TRELLO_TOKEN")
BASE_URL = os.getenv("BASE_URL")

tr=trello()



class Task(BaseModel):
    """Task model for parsing AI output"""
    task_name: str = Field(description="The name of the task")
    due_date: Optional[str] = Field(None, description="The due date of the task in YYYY-MM-DD format")
    priority: str = Field(description="The priority of the task: High, Medium, or Low")
    assigned_to: List[str] = Field(default_factory=list, description="List of people assigned to the task")
    members_email: Optional[List[str]] = Field(default_factory=list, description="List of email addresses of people assigned to the task")


class TaskList(BaseModel):
    """List of tasks for parsing AI output"""
    tasks: List[Task] = Field(description="List of extracted tasks")


def create_task(board_name: str, tasks: TaskList) -> str:
    """Create Trello cards for tasks on a board and notify members by email."""
    boards_url = f"{BASE_URL}/members/me/boards"
    board_params = {"key": TRELLO_API_KEY, "token": TRELLO_TOKEN, "fields": "name"}
    boards = {b["name"]: b["id"] for b in requests.get(boards_url, params=board_params).json()}

    if board_name not in boards:
        return f"❌ Board '{board_name}' not found. Available: {', '.join(boards.keys())}"

    board_id = boards[board_name]
    lists_url = f"{BASE_URL}/boards/{board_id}/lists"
    lists_response = requests.get(lists_url, params={"key": TRELLO_API_KEY, "token": TRELLO_TOKEN}).json()
    target_list = next((l for l in lists_response if l["name"].lower() == "to do"), lists_response[0])
    list_id = target_list["id"]

    results = []

    # Access the dictionary using tasks.tasks[0]
    for task in tasks.tasks:  
        task_name = task.task_name  # Use dot notation to access attributes
        due_date = task.due_date
        priority = task.priority
        members = task.assigned_to
        emails = task.members_email

        card_params = {
            "key": TRELLO_API_KEY,
            "token": TRELLO_TOKEN,
            "idList": list_id,
            "name": task_name,
            "desc": f"Priority: {priority}",
            "due": due_date
        }

        card_resp = requests.post(f"{BASE_URL}/cards", params=card_params)
        if card_resp.status_code != 200:
            results.append(f"❌ Failed to create: {task_name}")
            continue

        card_id = card_resp.json()["id"]
        results.append(f"✅ Created: {task_name}")

        if members:
          for member in members:
            member_resp = tr.get_trello_board_members(board_name)
            member_id = next((m["id"] for m in member_resp if m["fullName"].lower() == member.lower()), None)
            if not member_id:
                results.append(f"⚠️ Member not found: {member}")
                continue
            assign_url = f"{BASE_URL}/cards/{card_id}/idMembers"
            assign_resp = requests.post(assign_url, params={"key": TRELLO_API_KEY, "token": TRELLO_TOKEN, "value": member_id})
            if assign_resp.status_code == 200:
                results.append(f"👤 Assigned to : {member}")
        if emails:
          for email in emails:
                subject = f"Task Assigned: {task_name}"
                reciever = email
                body=f"You have been assigned to : {task_name} (Priority: {priority}, Due: {due_date})"
                results.append(send_email_notification(sender_email=EMAIL_ADDRESS, sender_password=APP_PASSWORD ,to_email= reciever, subject=subject, body=body))
    return "\n".join(results)






    