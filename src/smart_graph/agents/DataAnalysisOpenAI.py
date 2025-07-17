import os
import time
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent

# Load environment variables from .env
load_dotenv()

# Check for API key
if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("❌ OPENAI_API_KEY environment variable is not set.")


class PandasDataFrameAgent:
    def __init__(self, upload_folder: str = "uploads"):
        self.upload_folder = upload_folder
        self.agent = None
        self.selected_filename = None
        self._ensure_upload_folder()

    def _ensure_upload_folder(self):
        os.makedirs(self.upload_folder, exist_ok=True)

    def wait_for_file(self):
        while True:
            csv_files = self.get_available_csv_files()
            if csv_files:
                return csv_files
            time.sleep(1)

    def get_available_csv_files(self):
        return [f for f in os.listdir(self.upload_folder) if f.endswith(".csv")]

    def select_file(self, filename: str):
        file_path = os.path.join(self.upload_folder, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        self.selected_filename = filename

    def _load_dataframe(self):
        if not self.selected_filename:
            raise ValueError("No CSV file selected.")
        file_path = os.path.join(self.upload_folder, self.selected_filename)
        return pd.read_csv(file_path)

    def _create_agent(self):
        df = self._load_dataframe()
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
        )
        self.agent = create_pandas_dataframe_agent(
            llm,
            df,
            agent_type="tool-calling",
            verbose=True,
            allow_dangerous_code=True,
        )

    def ask(self, question: str):
        if self.agent is None:
            self._create_agent()
        return self.agent.invoke(question)


# 🧪 CLI runner
if __name__ == "__main__":
    agent = PandasDataFrameAgent()

    print("🕓 Waiting for a CSV file to be uploaded to /uploads ...")
    files = agent.wait_for_file()
    print("✅ Found CSV files:")
    for f in files:
        print(" -", f)

    selected = input("📄 Enter the CSV filename to analyze: ").strip()
    try:
        agent.select_file(selected)
    except FileNotFoundError:
        print("❌ File not found.")
        exit()

    print("\n🤖 Ask me anything about your data (type 'exit' to quit)\n")
    while True:
        q = input("🧠 You: ")
        if q.lower() in ["exit", "quit"]:
            print("👋 Bye!")
            break
        try:
            result = agent.ask(q)
            print("🤖 Agent:", result, "\n")
        except Exception as e:
            print("❌ Error:", e)
