import os
import uuid
from datetime import datetime
from pymilvus import MilvusClient
import google.generativeai as genai

from src.smart_graph.utils.text_extractor import TextImgExtractor
from src.smart_graph.utils.send_mail import send_email

from dotenv import load_dotenv
load_dotenv()

# Set up Gemini API globally
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

class DocumentIntelligencePipeline:
    def __init__(self):
        self.col_name = "documents_collection"
        self._setup_gemini_api()
        self._setup_milvus()
        self.text_extractor = TextImgExtractor()
        self.from_email = os.getenv("SENDER_EMAIL")

    def _setup_gemini_api(self):
        self.llm = genai.GenerativeModel("gemini-1.5-flash")

    def _setup_milvus(self):
        self.milvus_client = MilvusClient(
            uri=os.getenv("MILVUS_URI"),
            token=os.getenv("MILVUS_TOKEN"),
            db_name="default"
        )

        if self.col_name not in self.milvus_client.list_collections():
            self.milvus_client.create_collection(
                collection_name=self.col_name,
                dimension=768,
                metric_type="COSINE",
                schema=[
                    {"name": "vector", "type": "FLOAT_VECTOR", "dim": 768},
                    {"name": "category", "type": "VARCHAR", "max_length": 50},
                    {"name": "summary", "type": "VARCHAR", "max_length": 1024},
                    {"name": "file_name", "type": "VARCHAR", "max_length": 255},
                    {"name": "file_format", "type": "VARCHAR", "max_length": 10},
                    {"name": "document", "type": "VARCHAR", "max_length": 65535},
                    {"name": "date", "type": "VARCHAR", "max_length": 100}
                ]
            )

    def extract_text_from_image(self, image_path):
        return self.text_extractor.extract_text_from_image(image_path)

    def analyze_document(self, text):
        classification_prompt = f"""
        classify the provided document into *one of the predefined categories only*.

        Strict Instructions:
        - Choose ONLY from the following categories: [email, invoice, report, legal, resume, article, non-sense, other]
        - If the text is meaningless, empty, or random characters, classify it as *non-sense*
        - If you can not determine exactly the category, classify it as *other*
        - DO NOT invent new categories
        - DO NOT add explanation — return only the category in lowercase

        Text to classify: {text}
        """

        summary_prompt = f"""
        <s><|user|>
        Summarize the key points from this {text} lengthy document.
        Use only two sentences.<|end|> <|assistant|>
        """

        category_response = self.llm.generate_content(classification_prompt)
        category = category_response.text.strip().lower()

        summary_response = self.llm.generate_content(summary_prompt)
        summary = summary_response.text.strip()

        return {"document_category": category, "summary": summary}

    def summarize_only(self, text):
        prompt = f"""
        Summarize the following document in 2-3 sentences:

        {text}
        """
        response = self.llm.generate_content(prompt)
        return response.text.strip()

    def notify_user(self, to_email, document_name, category):
        subject = "New Document Categorized"
        message = f"Document: {document_name} has been categorized as '{category}'."
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

            if category.lower() in ["non-sense"]:
                continue

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
