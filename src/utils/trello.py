import requests
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field 
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
import os
from dotenv import load_dotenv

load_dotenv("app.env")
SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = os.getenv("SMTP_PORT")
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS","")
APP_PASSWORD = os.getenv("APP_PASSWORD")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  
TRELLO_API_KEY = os.getenv("TRELLO_API_KEY")
TRELLO_TOKEN = os.getenv("TRELLO_TOKEN")
BASE_URL = os.getenv("BASE_URL")


class trello:
    def __init__(self):

        self.trello_api_key = TRELLO_API_KEY
        self.trello_token = TRELLO_TOKEN 
        self.trello_username = "mohamedbahaa45"
        self.trello_base_params = {"key": self.trello_api_key, "token": self.trello_token}



    def get_trello_boards(self) -> List[Dict]:
        """Get all Trello boards for the user"""
        url = f"https://api.trello.com/1/members/{self.trello_username}/boards"
        
        try:
            params={
                **self.trello_base_params,
                "fields":"name"
            }
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                boards=response.json()

                board_names = [board["name"] for board in boards ]
                return boards
            else:
                return f"Error fetching boards: {response.status_code}, {response.text}"
                
        except Exception as e:
            return f"Error getting boards: {e}"
    def select_board(self, board_name: str) -> str:

        """Given a board name, return the Trello board ID."""
        boards = self.get_trello_boards()

        if not boards:
            raise Exception("No Trello boards found")

    # Normalize input and compare
        normalized_input = board_name.strip().lower()
        for board in boards:
            if board['name'].strip().lower() == normalized_input:
                return board['id']

        raise ValueError(f"No board found with name '{board_name}'. Available boards: {[b['name'] for b in boards]}")
    
    def get_trello_lists(self) -> List[Dict]:
        """Get all lists in a Trello board"""
        
        url = f"https://api.trello.com/1/boards/{self.board_id}/lists"
        
        try:
            response = requests.get(url, params=self.trello_base_params)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error fetching lists: {response.status_code}, {response.text}")
                return []
        
        except Exception as e:
            print(f"Error getting lists: {e}")
            return []
    
    def get_trello_board_members(self, board_name: str) -> str:
      """Get all members of a Trello board by board name (non-interactive, for LLM tools)"""
    
    # Step 1: Fetch all boards
      boards_url = f"https://api.trello.com/1/members/{self.trello_username}/boards"
      try:
        boards_response = requests.get(boards_url, params=self.trello_base_params)
        if boards_response.status_code != 200:
            return f"Error fetching boards: {boards_response.status_code}, {boards_response.text}"

        boards = boards_response.json()
        board_map = {board["name"].lower(): board["id"] for board in boards}
        
        board_id = board_map.get(board_name.lower())
        if not board_id:
            return f"Board named '{board_name}' not found."

      except Exception as e:
        return f"Error retrieving boards: {e}"

    # Step 2: Fetch board members using the board ID
      members_url = f"https://api.trello.com/1/boards/{board_id}/members"
      try:
        response = requests.get(members_url, params=self.trello_base_params)
        if response.status_code == 200:
            members = response.json()
            if not members:
                return f"No members found in board '{board_name}'."
            return members
        else:
            return f"Error fetching members: {response.status_code}, {response.text}"

      except Exception as e:
        return f"Error getting board members: {e}"

        
        
    def get_member_full_name(self, member_id: str) -> str:
      url = f"https://api.trello.com/1/members/{member_id}"
      try:
        response = requests.get(url, params=self.trello_base_params)
        if response.status_code == 200:
            return response.json().get("fullName", "Unknown")
        else:
            return "Unknown"
      except:
        return "Unknown"

    def get_trello_member_details(self, member_id: str = None) -> Dict:
        """Get details for a specific Trello member"""
        if not member_id:
            members=self.get_trello_board_members()
            member_choise=input("====choose member====")
            member_id=members[int(member_choise)-1]["id"]


        url = f"https://api.trello.com/1/members/{member_id}"
        
        try:
            response = requests.get(url, params=self.trello_base_params)
            
            if response.status_code == 200:
                
                return response.json()
            else:
                print(f"Error fetching member details: {response.status_code}, {response.text}")
                return {}
        
        except Exception as e:
            print(f"Error getting member details: {e}")
            return {}
    def get_cards(self, list_id: str = None)-> Dict :
        """get all cards in a list"""
        if not list_id:
            lists=self.get_trello_lists()
            for i, list in enumerate(lists):
                print(f"{i+1}.{list['name']}")
            choice=input("choose list")
            list_id=lists[int(choice)-1]["id"]
        url = f"https://api.trello.com/1/lists/{list_id}/cards"

        try:
            response = requests.get(url, params=self.trello_base_params)
            if response.status_code== 200:
                print(f"Availabel tasks in { lists[int(choice)-1]['name'] }")
                for i, card in enumerate(response.json()):
                    print(f"{i+1}.{card['name']}")
                return response.json()
            else:
                print(f"Error fetching cards : {response.status_code}, {response.text}")
                return {}
        except Exception as e:
            print(f"Error getting cards: {e}")
            return {}
        
    
    def create_trello_card(self, task: Dict, list_id: str =None ) -> Dict:
        """Create a card in a Trello list for a task"""
        url = "https://api.trello.com/1/cards"
        if not list_id:
            lists=self.get_trello_lists()
            for i, list in enumerate(lists):
                print(f"{i+1}.{list['name']}")
            choice=input("choose list")
            list_id=lists[int(choice)-1]["id"]
        
        params = {
            **self.trello_base_params,
            "idList": list_id,
            "name": task["name"],
            "desc": f"Priority: {task['priority']}"
        }
        
        if task["due_date"]:
            params["due"] = task["due_date"].strftime("%Y-%m-%dT%H:%M:%S.000Z")
        
        try:
            response = requests.post(url, params=params)
            
            if response.status_code == 200:
                card = response.json()
                task["trello_card_id"] = card["id"]
                task["trello_card_url"] = card["shortUrl"]
                return card
            else:
                print(f"Error creating card: {response.status_code}, {response.text}")
                return {}
        
        except Exception as e:
            print(f"Error creating Trello card: {e}")
            return {}
    
    def assign_member_to_card(self, card_id: str, member_id: str) -> bool:
        """Assign a member to a Trello card"""
        url = f"https://api.trello.com/1/cards/{card_id}/idMembers"
        
        params = {
            **self.trello_base_params,
            "value": member_id
        }
        
        try:
            response = requests.post(url, params=params)
            
            if response.status_code == 200:
                return True
            else:
                print(f"Error assigning member: {response.status_code}, {response.text}")
                return False
        
        except Exception as e:
            print(f"Error assigning member to card: {e}")
            return False
    def generate_task_progress_report(self,board_name :str) -> None:
      """Generate a report of task progress and save it as a PDF."""
      self.board_id=self.select_board(board_name)
      # Fetch all lists in the board
      lists = self.get_trello_lists()
      if not lists:
          print("No lists found in the board.")
          return

      list_names = []
      task_counts = []
      all_tasks = {}

      for lst in lists:
        list_names.append(lst['name'])

        # Fetch cards in the current list
        url = f"https://api.trello.com/1/lists/{lst['id']}/cards"
        try:
          response = requests.get(url, params=self.trello_base_params)
          if response.status_code == 200:
            cards = response.json()
            task_counts.append(len(cards))  # Count the number of cards in the list
            all_tasks[lst['name']] = []

            for card in cards:
              task_info = {
                        "name": card['name'],
                        "due_date": card.get('due', 'No due date'),
                        "comments": card.get('desc', 'No comments'),
                        "assigned_members": [self.get_member_full_name(member_id) for member_id in card.get('idMembers', [])]
                    }
              all_tasks[lst['name']].append(task_info)
          else:
            print(f"Error fetching cards for list {lst['name']}: {response.status_code}, {response.text}")
            task_counts.append(0)

        except Exception as e:
          print(f"Error fetching cards for list {lst['name']}: {e}")
          task_counts.append(0)

    # Create a bar chart
      plt.figure(figsize=(10, 6))
      plt.barh(list_names, task_counts, color='skyblue')
      plt.xlabel('Number of Tasks')
      plt.title('Task Progress Report')
      plt.grid(axis='x')

    # Save the chart as an image
      chart_filename = 'task_progress_chart.png'
      plt.savefig(chart_filename)
      plt.close()  # Close the plot to free memory

    # Create a PDF report
      pdf_filename = 'task_progress_report.pdf'
      c = canvas.Canvas(pdf_filename, pagesize=letter)
      width, height = letter

    # Add title
      c.setFont("Helvetica-Bold", 16)
      c.drawString(1 * inch, height - 1 * inch, "Task Progress Report")

    # Add task names, due dates, comments, and assigned members
      c.setFont("Helvetica", 12)
      y_position = height - 1.5 * inch
      for list_name, tasks in all_tasks.items():
          c.drawString(1 * inch, y_position, f"{list_name}:")
          y_position -= 0.2 * inch
          for task in tasks:
              if y_position < 1 * inch:  # Check if we need to create a new page
                  c.showPage()  # Create a new page
                  c.setFont("Helvetica", 12)  # Reset font for new page
                  y_position = height - 1 * inch  # Reset y position for new page
                  c.drawString(1 * inch, y_position, "Task Progress Report")  # Re-add title
                  y_position -= 0.5 * inch  # Add space after title

              c.drawString(1.5 * inch, y_position, f"- {task['name']}")
              y_position -= 0.2 * inch
              c.drawString(2 * inch, y_position, f"  Due Date: {task['due_date']}")
              y_position -= 0.2 * inch
              c.drawString(2 * inch, y_position, f"  Comments: {task['comments']}")
              y_position -= 0.2 * inch
              c.drawString(2 * inch, y_position, f"  Assigned Members: {', '.join(task['assigned_members'])}")
              y_position -= 0.4 * inch  # Add space between tasks

    # Add the chart image to the PDF
      if y_position < 5 * inch:  # Check if we need to create a new page for the chart
          c.showPage()  # Create a new page
          c.setFont("Helvetica", 12)  # Reset font for new page
          y_position = height - 1 * inch  # Reset y position for new page
          c.drawString(1 * inch, y_position, "Task Progress Report")  # Re-add title
          y_position -= 0.5 * inch  # Add space after title

      c.drawImage(chart_filename, 1 * inch, y_position - 4 * inch, width=8 * inch, height=4 * inch)

    # Finalize the PDF
      c.save()

      print(f"Task progress report saved as {pdf_filename}.")
    # Optionally, remove the chart image after saving the PDF
      if os.path.exists(chart_filename):
          os.remove(chart_filename)

 