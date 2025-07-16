import os
import time
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

class GeminiCSVAgent:
    def __init__(self, upload_folder="uploads"):
        self.upload_folder = upload_folder
        self.df = None
        self.model = None
        self.selected_filename = None
        self._ensure_upload_folder()

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("❌ GEMINI_API_KEY is not set.")
        genai.configure(api_key=api_key)

    def _ensure_upload_folder(self):
        os.makedirs(self.upload_folder, exist_ok=True)

    def wait_for_file(self):
        while True:
            csvs = self.get_available_csv_files()
            if csvs:
                return csvs
            time.sleep(1)

    def get_available_csv_files(self):
        return [f for f in os.listdir(self.upload_folder) if f.endswith(".csv")]

    def select_file(self, filename: str):
        path = os.path.join(self.upload_folder, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV file not found: {path}")
        self.selected_filename = filename
        self.df = pd.read_csv(path)
        self.model = genai.GenerativeModel("gemini-2.0-flash")

    def ask(self, question: str):
        if self.df is None or self.model is None:
            raise RuntimeError("No file selected yet.")

        preview = self.df.head(5).to_csv(index=False)
        schema = ", ".join(f"{col} ({dtype})" for col, dtype in zip(self.df.columns, self.df.dtypes))
        prompt = f"""
You are a data analyst. Here's a CSV dataset with these columns:
{schema}

Here are the first few rows:
{preview}
cd 
Answer the following user question using only this data:
{question}
        """.strip()

        response = self.model.generate_content(prompt)
        return response.text

""" 
# 🧪 CLI Runner
if __name__ == "__main__":
    agent = GeminiCSVAgent()

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
            print("🤖 Gemini:", result, "\n")
        except Exception as e:
            print("❌ Error:", e, "\n")
 """