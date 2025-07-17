# embedder.py
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables
load_dotenv()

# Initialize Gemini embedding model
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)
