# src/smart_graph/tools/csv_tool.py

from typing import Annotated
from langchain_core.tools import tool
from src.smart_graph.agents.DataAnalysis import GeminiCSVAgent

@tool
def analyze_csv(filename: str, question: str) -> str:
    """
    Analyze a CSV file uploaded to the /uploads folder and answer a question about it.
    """
    agent = GeminiCSVAgent()

    # Step 1: Wait until CSVs are available
    files = agent.wait_for_file()

    # Step 2: If no filename, return available
    if not filename:
        return f"✅ Found CSV files: {files}. Please provide filename to continue."

    # Step 3: Try selecting file
    try:
        agent.select_file(filename)
    except FileNotFoundError as e:
        return str(e)

    # Step 4: Ask the question
    try:
        answer = agent.ask(question)
        return answer
    except Exception as e:
        return f"❌ Error: {e}"