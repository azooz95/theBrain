import os
from datetime import datetime
from pathlib import Path
import google.generativeai as genai
from agents.docOrganization import DocumentIntelligencePipeline
import traceback

class DocumentAgent:
    def __init__(self):
        self.pipeline = DocumentIntelligencePipeline()

    def run(self, input: dict) -> str:
        file_path = input.get("file_path")
        email = input.get("email")
        mode = input.get("mode", "analyze").lower()  # default is full analysis

        if not file_path or not Path(file_path).exists():
            return "❌ Invalid or missing 'file_path'. Please provide a valid path to a document or image."

        ext = os.path.splitext(file_path)[1].lower()
        is_image = ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']

        try:
            # Extract text
            if is_image:
                text = self.pipeline.extract_text_from_image(file_path)
            else:
                with open(file_path, 'rb') as f:
                    text = f.read().decode("latin-1")

            # 🟡 Mode: Summarize Only
            if mode == "summarize":
                summary = self.pipeline.summarize_only(text)
                return f"📝 Summary:\n{summary}"

            # 🟢 Mode: Analyze + Store
            analysis = self.pipeline.analyze_document(text)
            category = analysis.get("document_category", "unknown")
            summary = analysis.get("summary", "No summary available.")

            if category.lower() in ["non-sense", "other"]:
                return f"⚠️ Document category detected as '{category}', skipping storage.\n📝 Summary:\n{summary}"

            file_name, file_format = self.pipeline.extract_metadata(file_path)

            # Vector embedding
            embed_result = genai.embed_content(
                model="models/text-embedding-004",
                content=text
            )
            vector = embed_result.get("embedding", None)

            if not vector or not isinstance(vector, list) or not all(isinstance(x, float) for x in vector):
                raise ValueError("❌ Invalid embedding vector generated.")

            self.pipeline.milvus_client.insert(
                collection_name=self.pipeline.col_name,
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

            if email:
                self.pipeline.notify_user(email, file_name, category)

            return (
                f"✅ Document analyzed successfully.\n"
                f"📄 File: {file_name}\n"
                f"🏷️ Category: {category}\n"
                f"📝 Summary: {summary}\n"
                f"📦 Document has been stored in the vector database."
            )

        except Exception as e:
            return f"❌ Failed to analyze the document: {str(e)}\n{traceback.format_exc()}"
