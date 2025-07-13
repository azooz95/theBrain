import os
from datetime import datetime
from dotenv import load_dotenv
from pymilvus import Collection, connections, utility
from langchain_milvus import Milvus
from embedder import embedding_model
import bcrypt
from functools import wraps

is_server = True
# Load environment variables
load_dotenv()

server_ip = "127.0.0.1"
port = 19530

milvus_uri = os.environ.get('MILVUS_URI', "https://in03-797ad2fd750bbe3.serverless.gcp-us-west1.cloud.zilliz.com")
milvus_token = os.environ.get('MILVUS_TOKEN', '6a5203dfe89e59dd3491f39a9efb2c33657a3bbe439e8c9d7e348321d8c0330305463f9fef4175bb64bc18074d15928867acd3f9')


if not is_server: 
    connections_configs= {
        'host': server_ip,
        'port': port
    }
else:
    connections_configs= {
        "uri": os.environ["MILVUS_URI"],
        "token": os.environ["MILVUS_TOKEN"]
    }

index_params = {"index_type": "FLAT", "metric_type": "L2"}

# Separate vector stores
user_store = Milvus(
    embedding_function=embedding_model,
    collection_name="user_profiles",
    connection_args=connections_configs,
    index_params=index_params,
)

document_store = Milvus(
    embedding_function=embedding_model,
    collection_name="document_embeddings",
    connection_args=connections_configs,
    index_params=index_params,
)

# -------- Decorator -------- #
def require_collections(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            connections.connect(
                alias="default",
                uri=os.environ["MILVUS_URI"],
                token=os.environ["MILVUS_TOKEN"]
            )
        except:
            return {"success": False, "message": "❌ Failed to connect to Milvus."}

        expected = {"user_profiles", "document_embeddings"}
        existing = set(utility.list_collections())

        if not expected.issubset(existing):
            return {"success": False,
                    "message": "❌ Required collections are missing."}

        return func(*args, **kwargs)

    return wrapper


# -------- USER FUNCTIONS -------- #

def _generate_user_vector(profile_text="default user"):
    try:
        return embedding_model.embed_query(profile_text)
    except:
        return [0.1] * 768

@require_collections
def register_user(username, email, password, profile_desc="default user"):
    if not all([username, email, password]):
        return {"success": False, "message": "Username, email, and password are required."}

    results = user_store.similarity_search(f"user profile for {email}", k=1)
    if results and results[0].metadata.get("email") == email:
        return {"success": False, "message": "Email already registered."}

    vector = _generate_user_vector(profile_desc)
    hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode("utf-8")

    metadata = {
        "username": username,
        "email": email,
        "password_hash": hashed_pw,
        "created_at": datetime.now().isoformat()
    }

    user_store.add_texts([f"user profile for {email}"], metadatas=[metadata])
    return {"success": True, "message": f"User '{username}' registered."}

@require_collections
def login_user(email, password):
    if not all([email, password]):
        return {"success": False, "message": "Email and password are required."}

    results = user_store.similarity_search(f"user profile for {email}", k=1)
    if not results:
        return {"success": False, "message": "User not found."}

    metadata = results[0].metadata
    stored_email = metadata.get("email")
    stored_hash = metadata.get("password_hash")

    if stored_email != email or not bcrypt.checkpw(password.encode(), stored_hash.encode()):
        return {"success": False, "message": "Incorrect email or password."}

    # Get user ID (if available)
    connections.connect(
        alias="default",
        uri=os.environ["MILVUS_URI"],
        token=os.environ["MILVUS_TOKEN"]
    )
    collection = Collection("user_profiles")
    collection.load()
    result = collection.query(
        expr=f'email == "{email}"',
        output_fields=["email"]
    )
    user_id = email  # fallback to email as identifier

    return {"success": True, "id": user_id, **metadata}

# -------- DOCUMENT FUNCTIONS -------- #

@require_collections
def insert_document(text, file_name, file_format, summary, category, user_id, folder_id):
    if not all([text, file_name, file_format, user_id, folder_id]):
        return {"success": False, "message": "Missing required document fields."}

    vector = embedding_model.embed_query(text)
    metadata = {
        "file_name": file_name,
        "file_format": file_format,
        "summary": summary,
        "category": category,
        "date": datetime.now().isoformat(),
        "user_id": str(user_id),
        "folder_id": str(folder_id)
    }

    document_store.add_texts([text], metadatas=[metadata])
    return {"success": True, "message": f"Document '{file_name}' inserted."}

@require_collections
def search_by_filename(file_name):
    if not file_name:
        return []

    connections.connect(
        alias="default",
        uri=os.environ["MILVUS_URI"],
        token=os.environ["MILVUS_TOKEN"]
    )

    collection = Collection(name="document_embeddings")
    collection.load()

    results = collection.query(
        expr=f'file_name == "{file_name}"',
        output_fields=["file_name", "summary", "category", "date", "user_id", "folder_id"]
    )

    return results

@require_collections
def delete_document(file_name):
    if not file_name:
        return {"success": False, "message": "Filename is required."}

    collection = Collection(name="document_embeddings")
    collection.delete(expr=f'file_name == "{file_name}"')
    return {"success": True, "message": f"Document '{file_name}' deleted."}
