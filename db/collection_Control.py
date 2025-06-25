# document_control.py

import os
from datetime import datetime
from dotenv import load_dotenv
from pymilvus import Collection
from langchain_milvus import Milvus
from embedder import embedding_model
import bcrypt
from pymilvus import Collection, connections

# Load environment variables
load_dotenv()

# Separate vector stores
user_store = Milvus(
    embedding_function=embedding_model,
    collection_name="user_profiles",
    connection_args={
        "uri": os.environ["MILVUS_URI"],
        "token": os.environ["MILVUS_TOKEN"]
    },
    index_params={"index_type": "FLAT", "metric_type": "L2"},
)

document_store = Milvus(
    embedding_function=embedding_model,
    collection_name="document_embeddings",
    connection_args={
        "uri": os.environ["MILVUS_URI"],
        "token": os.environ["MILVUS_TOKEN"]
    },
    index_params={"index_type": "FLAT", "metric_type": "L2"},
)

# -------- USER FUNCTIONS -------- #

def _generate_user_vector(profile_text="default user"):
    try:
        return embedding_model.embed_query(profile_text)
    except:
        return [0.1] * 768

def register_user(username, email, password, profile_desc="default user"):
    if not all([username, email, password]):
        print("❌ Username, email, and password are required.")
        return

    # Check if email already exists
    results = user_store.similarity_search(f"user profile for {email}", k=1)
    if results and results[0].metadata.get("email") == email:
        print("❌ Email already registered.")
        return

    vector = _generate_user_vector(profile_desc)
    hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode("utf-8")

    metadata = {
        "username": username,
        "email": email,
        "password_hash": hashed_pw,
        "created_at": datetime.now().isoformat()
    }

    user_store.add_texts([f"user profile for {email}"], metadatas=[metadata])
    print(f"✅ User '{username}' registered.")

def login_user(email, password):
    if not all([email, password]):
        print("❌ Email and password are required.")
        return None

    results = user_store.similarity_search(f"user profile for {email}", k=1)
    if not results:
        print("❌ User not found.")
        return None

    metadata = results[0].metadata
    stored_email = metadata.get("email")
    stored_hash = metadata.get("password_hash")

    if stored_email != email:
        print("❌ Email mismatch.")
        return None

    if not stored_hash or not bcrypt.checkpw(password.encode(), stored_hash.encode()):
        print("❌ Incorrect password.")
        return None

    print(f"✅ Login successful for {metadata.get('username')}")
    return metadata

# -------- DOCUMENT FUNCTIONS -------- #

def insert_document(text, file_name, file_format, summary, category, user_id, folder_id):
    if not all([text, file_name, file_format]):
        print("❌ Missing required document fields.")
        return

    vector = embedding_model.embed_query(text)
    metadata = {
        "file_name": file_name,
        "file_format": file_format,
        "summary": summary,
        "category": category,
        "date": datetime.now().isoformat()
    }

    document_store.add_texts([text], metadatas=[metadata])
    print(f"✅ Document '{file_name}' inserted.")

def search_by_filename(file_name):
    if not file_name:
        print("❌ Filename is required.")
        return []

    # ✅ Connect to Milvus before querying
    connections.connect(
        alias="default",
        uri=os.environ["MILVUS_URI"],
        token=os.environ["MILVUS_TOKEN"]
    )

    collection = Collection(name="document_embeddings")
    collection.load()

    results = collection.query(
        expr=f'file_name == "{file_name}"',
        output_fields=["file_name", "summary", "category", "date"]
    )

    return results

def delete_document(file_name):
    if not file_name:
        print("❌ Filename is required.")
        return

    collection = Collection(name="document_embeddings")
    collection.delete(expr=f'file_name == "{file_name}"')
    print(f"🗑️ Document '{file_name}' deleted.")