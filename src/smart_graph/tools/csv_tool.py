# src/smart_graph/tools/csv_tool.py

from typing import Annotated
from langchain_core.tools import tool
from src.smart_graph.agents.DataAnalysis import GeminiCSVAgent

session_agent = GeminiCSVAgent()  # Persistent session

@tool
def analyze_csv(input: str) -> str:
    """
    Interactively analyze a CSV file. The flow is:
    1. Wait for user to upload a CSV (if none exists).
    2. Display available files and ask for filename.
    3. Once file is selected, accept and answer questions.
    """

    # Step 1: Wait for at least one CSV
    files = session_agent.get_available_csv_files()
    if not files:
        session_agent.wait_for_file()
        files = session_agent.get_available_csv_files()

    # Step 2: No file selected yet
    if session_agent.selected_filename is None:
        if input.lower().endswith(".csv"):
            try:
                session_agent.select_file(input)
                return "📄 File loaded successfully. Now ask me any question about your data."
            except FileNotFoundError:
                return f"❌ File not found: {input}. Try one of these: {files}"
        else:
            return f"✅ Found CSV files: {files}. Please enter one of them to continue."

    # Step 3: A file is already selected → answer questions
    try:
        answer = session_agent.ask(input)
        return answer
    except Exception as e:
        return f"❌ Error while analyzing: {e}"
