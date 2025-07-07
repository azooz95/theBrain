from dotenv import load_dotenv
from utils.trello import trello
import requests
from utils.send_notifications import send_email_notification
from pydantic import BaseModel, Field 
from typing import Optional, List , Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import PydanticOutputParser
from utils.task_creation_and_report_tools import create_task, Task,TaskList
import os 


tr=trello()

load_dotenv("app.env")
SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = os.getenv("SMTP_PORT")
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS","")
APP_PASSWORD = os.getenv("APP_PASSWORD")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  
TRELLO_API_KEY = os.getenv("TRELLO_API_KEY")
TRELLO_TOKEN = os.getenv("TRELLO_TOKEN")
BASE_URL = os.getenv("BASE_URL")

class intention(BaseModel):
    user_intention: Literal["create_task", "generate_report", "friendly conversation"] = Field(description="what the user want to do ")
    board : Optional[str] = Field(description="the name of the board that the user want you to work on ")
    AiResponse: str = Field(description="the ai response")

tr=trello()

intention_parser=PydanticOutputParser(pydantic_object=intention)
intention_format=intention_parser.get_format_instructions()

tasks_parser=PydanticOutputParser(pydantic_object=TaskList)
tasks_format=tasks_parser.get_format_instructions()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GEMINI_API_KEY)

def intent_and_board_agent(user_input: str) -> dict:
    boards = tr.get_trello_boards()
    

    prompt = f"""
    You are an AI assistant that determines the user's intent from the following messages:
    "{user_input}"
    Your job is to:
    1. Decide if the user wants to "create_task" or "generate_report" or "friendly conversation".
    2. use the intention as friendly conversation if the the board choice is not clear  or the intentions are not clear, or the task is done.
    3. Ask them to choose a board from: {boards} if not mentioned or if he mentioned a board name that is not in the available ones.
    Be conversational in your follow-up if the board isn't clear.
    the final Respond should be using this format:
    {intention_format}
    """
    response = llm.invoke(prompt)
    parsed = intention_parser.parse(response.content)
    
    
    if not parsed.board:
        parsed.user_intention = "friendly conversation"
    
    
    return parsed

def task_extractor_agent(board : str, user_input:str ) -> str:
    """Create Trello cards for tasks on a board and notify members by email."""
    prompt = "\n".join(["Extract tasks from the given text and return them in JSON format.\n"
            f"{tasks_format}\n"
            "Only return valid JSON without any additional text. If due_date isn't specified, return null for that field.\n"
            "If assigned members aren't explicitly mentioned, leave the assigned_to array empty.\n"
            "Extract all relevant task information based on the context.",
            f"text : {user_input}"])
    
    response = llm.invoke(prompt)
    
    try:
        # Parse the response into a TaskList object
        tasks = tasks_parser.parse(response.content)
        
        # Call create_task with the board name and tasks
        results = create_task(board, tasks)
        
        return "".join(results)
    except Exception as e:
        return f"Error creating tasks: {str(e)}"
    
def generate_report(board) -> str:
    """Generates a summary of all tasks on a specific Trello board."""
    report = tr.generate_task_progress_report(board)

    return report


