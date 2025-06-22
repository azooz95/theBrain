# initialization.py

import os
from dotenv import load_dotenv
from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType

# Load environment variables
load_dotenv()

def init_milvus():
    connections.connect(
        alias="default",
        uri=os.environ["MILVUS_URI"],
        token=os.environ["MILVUS_TOKEN"]
    )
    print("✅ Connected to Zilliz Milvus Cloud")

def create_user_collection():
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768),
        FieldSchema(name="username", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="email", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="password_hash", dtype=DataType.VARCHAR, max_length=200),
        FieldSchema(name="created_at", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000),
    ]

    schema = CollectionSchema(fields=fields, description="User Profile Collection")

    if "user_profiles" in utility.list_collections():
        Collection("user_profiles").drop()
        print("🗑️ Dropped existing 'user_profiles' collection.")

    Collection(name="user_profiles", schema=schema)
    print("✅ User collection created.")

def create_document_collection():
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768),
        FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=200),
        FieldSchema(name="file_format", dtype=DataType.VARCHAR, max_length=20),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=10000),  # ✅ Add this
        FieldSchema(name="date", dtype=DataType.VARCHAR, max_length=100),
    ]

    schema = CollectionSchema(fields=fields, description="Document Embeddings Collection")

    if "document_embeddings" in utility.list_collections():
        Collection("document_embeddings").drop()
        print("🗑️ Dropped existing 'document_embeddings' collection.")

    Collection(name="document_embeddings", schema=schema)
    print("✅ Document collection created.")

if __name__ == "__main__":
    init_milvus()
    create_user_collection()
    create_document_collection()
