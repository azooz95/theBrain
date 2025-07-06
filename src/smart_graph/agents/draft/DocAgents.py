import os
import uuid
from datetime import datetime
import tkinter as tk
from tkinter import filedialog
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pymilvus import MilvusClient
import google.generativeai as genai

from text_extractor import TextImgExtractor
from send_mail import send_email

from dotenv import load_dotenv
load_dotenv()

# Set up Gemini API globally
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_multiple_file_paths():
    root = tk.Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames()
    return file_paths


class DocumentIntelligencePipeline:
    def __init__(self):
        self._setup_gemini_api()
        self._setup_milvus()
        self.col_name = "documents_collection"
        self.text_extractor = TextImgExtractor()
        self.from_email = os.getenv("SENDER_EMAIL")

    def _setup_gemini_api(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
            max_tokens=500
        )

    def _setup_milvus(self):
        self.milvus_client = MilvusClient(uri=os.getenv("MILVUS_URI"), token=os.getenv("MILVUS_TOKEN"), db_name="default")

    def extract_text_from_image(self, image_path):
        return self.text_extractor.extract_text_from_image(image_path)

    def analyze_document(self, text):
        classification_prompt = PromptTemplate(
            input_variables=["text"],
            template="""
                  classify the provided document into *one of the predefined categories only*. 

                  Strict Instructions:
                  - Choose ONLY from the following categories: [email, invoice, report, legal, resume, article, non-sense, other]
                  - If the text is meaningless, empty, or random characters, classify it as *non-sense*
                  - If you can not determine exactly the category, classify it as *other*
                  - DO NOT invent new categories
                  - DO NOT add explanation — return only the category in lowercase

                  Examples:
                  1. "Dear Hiring Manager, I am applying for the data science position..."  
                    → email

                  2. "asdf asdf jkljlkj"  
                    → non-sense

                  Text to classify: {text}
                  """
        )

        classify_chain = LLMChain(llm=self.llm, prompt=classification_prompt, output_key="document_category")

        summary_prompt = PromptTemplate(
            input_variables=["text", "document_category"],
            template="""<s><|user|>
            Summarize the key points from this {text} lengthy document with the category {document_category}.
            Use only two sentences.<|end|> <|assistant|>"""
        )

        summary_chain = LLMChain(llm=self.llm, prompt=summary_prompt, output_key="summary")

        result = classify_chain.invoke({"text": text})
        category = result["document_category"].strip()

        summary = summary_chain.invoke({"text": text, "document_category": category})["summary"].strip()

        return {"document_category": category, "summary": summary}

    def notify_user(self, to_email, document_name, category):
        subject = "New Document Categorized"
        message = f"Document: {document_name} has been categorized as '{category}'."
        print(f"📩 Email Notification Sent: {message}")
        return send_email(self.from_email, to_email, subject, message)

    def extract_metadata(self, path):
        return os.path.basename(path), os.path.splitext(path)[1]

    def process_and_store(self, file_paths, receiver_email=None):
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        for file_path in file_paths:
            ext = os.path.splitext(file_path)[1].lower()
            is_image = ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']

            if is_image:
                text = self.extract_text_from_image(file_path)
            else:
                with open(file_path, 'rb') as f:
                    text = f.read().decode("latin-1")

            analysis = self.analyze_document(text)
            category, summary = analysis["document_category"], analysis["summary"]

            if category.lower() == "non-sense":
                print(f"⚠️  {file_path} → 'non-sense'. Skipping.")
                continue

            if category.lower() == "other":
                print(f"⚠️  {file_path} → 'other'. Please enter the correct category manually.")
                category = input("Enter correct category: ").strip().lower()

            file_name, file_format = self.extract_metadata(file_path)
            vector = genai.embed_content(model="models/text-embedding-004", content=text)["embedding"]

            self.milvus_client.insert(
                collection_name=self.col_name,
                data=[{
                    "vector": vector,
                    "category": category,
                    "summary": summary,
                    "file_name": file_name,
                    "file_format": file_format,
                    "document": text,
                    "date": datetime.now().isoformat()
                }]
            )

            print(f"✅ Stored {file_name} as (Category: {category})")

            if receiver_email:
                self.notify_user(receiver_email, file_name, category)

    def search_documents(self, question):
        embedding = genai.embed_content(model="models/text-embedding-004", content=question)["embedding"]
        result = self.milvus_client.search(
            collection_name=self.col_name,
            data=[embedding],
            limit=1,
            search_params={"metric_type": "COSINE", "params": {}},
            output_fields=["document"]
        )
        return result

    def run(self):
        print("\n===== Document Intelligence Pipeline =====")
        print("Smart OCR | Classification | Summarization | Search | Notification")
        print("==========================================")

        while True:
            print("\nWhat would you like to do?")
            print("1. Process a document (via GUI)")
            print("2. Smart search")
            print("3. Exit")

            choice = input("\nEnter your choice (1-3): ")

            if choice == "1":
                file_paths = get_multiple_file_paths()
                file_paths = [path for path in file_paths if os.path.isfile(path)]

                if not file_paths:
                    print("❌ No valid file(s) provided.")
                    continue
                email = input("Enter notification email (or press Enter to skip): ").strip()
                email = email if email else None
                self.process_and_store(file_paths, email)

            elif choice == "2":
                query = input("\nEnter your search query:\n")
                results = self.search_documents(query)
                print("\nTop Result:")
                if results and results[0]:
                    print(results[0][0]["document"])
                else:
                    print("No results found.")

            elif choice == "3":
                print("\nThank you for using the Document Intelligence Pipeline. Goodbye!")
                break

            else:
                print("\nInvalid choice. Please try again.")


if __name__ == "__main__":
    pipeline = DocumentIntelligencePipeline()
    pipeline.run()