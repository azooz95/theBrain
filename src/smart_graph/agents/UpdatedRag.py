# -*- coding: utf-8 -*-
"""
Clean RAG pipeline with LangChain + Milvus + LangGraph
- Fixes missing imports and undefined llm
- Uses GoogleGenerativeAIEmbeddings (dim=768)
- Milvus Lite by default (local .db file) — switch to host/port if needed
- Removes duplicate Search class
- Handles Web loader (async) cleanly
- Adds safe fallback for prompt hub
"""

import io
import os
import asyncio
from enum import Enum
from typing import List, TypedDict, Literal

from pydantic import BaseModel, Field
from typing_extensions import Annotated

from PIL import Image

# LangChain core
from langchain import hub
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Loaders
from langchain_community.document_loaders import (
    DirectoryLoader,
    MergedDataLoader,
)
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders.pdf import UnstructuredPDFLoader
from langchain_community.document_loaders import (
    WebBaseLoader,
    UnstructuredWordDocumentLoader,
)

# Vector store (Milvus)
from langchain_milvus import Milvus

# Google GenAI
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)

# LangGraph
from langgraph.graph import START, StateGraph


# ---------------------------
# Models & Config
# ---------------------------

class Search(BaseModel):
    query: str = Field(..., description="Search query to run.")
    section: Literal["beginning", "middle", "end"] = Field(..., description="Section to query.")


class State(TypedDict):
    question: str
    query: Search
    context: List[Document]
    answer: str


class FileType(Enum):
    excel = 1
    pdf = 2
    docx = 3
    web = 4


# ---------------------------
# File-specific loaders
# ---------------------------

class Excelsheet:
    def __call__(self, path: str):
        return UnstructuredExcelLoader(path, mode="elements")


class PDF:
    def __call__(self, path: str):
        return UnstructuredPDFLoader(path, mode="elements")


class Docx:
    def __call__(self, path: str):
        return UnstructuredWordDocumentLoader(path, mode="elements")


class Web:
    async def __call__(self, paths: List[str]) -> List[Document]:
        loader = WebBaseLoader(paths)
        docs = []
        async for doc in loader.alazy_load():
            docs.append(doc)
        return docs


class FilesHandler:
    def __init__(self, file_type: FileType, file_path_or_paths):
        self.file_type = file_type
        self.file_path_or_paths = file_path_or_paths
        self._loader = self._selector(file_type)

    def _selector(self, file_type: FileType):
        mapping = {
            FileType.excel: Excelsheet,
            FileType.pdf: PDF,
            FileType.docx: Docx,
            FileType.web: Web,
        }
        if file_type not in mapping:
            raise ValueError(f"Unsupported file type: {file_type}")
        return mapping[file_type]()

    def load(self) -> List[Document]:
        """Sync loaders (excel/pdf/docx)."""
        if self.file_type == FileType.web:
            raise RuntimeError("Use `await load_web()` for web loader.")
        loader = self._loader(self.file_path_or_paths)  # returns a Loader
        return loader.load()

    async def load_web(self) -> List[Document]:
        """Async loader for web pages."""
        if self.file_type != FileType.web:
            raise RuntimeError("`load_web` is only for web loader.")
        return await self._loader(self.file_path_or_paths)


# ---------------------------
# RAG Core
# ---------------------------

class LangChainRAG:
    def __init__(self, vector_store: Milvus = None, milvus_uri: str = "milvus_demo2.db"):
        # Embeddings (Google embedding-001 has 768 dims)
        self.llm_embedding = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            # No temperature in embeddings API; leave defaults
        )

        # LLM for reasoning + structured output
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",  # or "gemini-pro"
            temperature=0,
        )

        # Vector store
        if vector_store:
            self.vector_store = vector_store
        else:
            # Milvus Lite (local file). For server: {"host":"localhost", "port":"19530"}
            self.vector_store = Milvus(
                embedding_function=self.llm_embedding,
                connection_args={"uri": milvus_uri},
                index_params={"index_type": "FLAT", "metric_type": "L2"},
                collection_name="rag_collection",
                # text_field / metadata_field are managed internally by LangChain Milvus
            )

        # Prompt from hub with fallback
        try:
            self.prompt = hub.pull("rlm/rag-prompt")
        except Exception:
            # Minimal fallback prompt
            from langchain_core.prompts import ChatPromptTemplate
            self.prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "Use the provided context to answer the question concisely and accurately."),
                    ("human", "Question:\n{question}\n\nContext:\n{context}")
                ]
            )

    # -------- Loading & Splitting --------

    def load_documents(self, dir_path: str = "./data") -> List[Document]:
        """
        Load all supported files under dir_path.
        DirectoryLoader patterns vary by LC version; the safest is to load all and filter.
        """
        loader = DirectoryLoader(dir_path, recursive=True, silent_errors=True)
        docs = loader.load()
        # Optional: filter by extensions if needed
        allowed_ext = {".xlsx", ".xls", ".pdf", ".docx", ".html", ".htm"}
        filtered = [d for d in docs if d.metadata.get("source", "").lower().endswith(tuple(allowed_ext))]
        return filtered

    @staticmethod
    def doc_splitter(docs: List[Document], chunk_size: int = 2000, chunk_overlap: int = 200) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        return splitter.split_documents(docs)

    @staticmethod
    def add_metadata_by_section(texts_splits: List[Document]) -> List[Document]:
        total = len(texts_splits)
        if total == 0:
            return texts_splits
        third = max(1, total // 3)
        for i, d in enumerate(texts_splits):
            if i < third:
                d.metadata["section"] = "beginning"
            elif i < 2 * third:
                d.metadata["section"] = "middle"
            else:
                d.metadata["section"] = "end"
        return texts_splits

    def add_embeddings(self, texts_splits: List[Document]):
        # Persist into Milvus
        self.vector_store.add_documents(texts_splits)

    # -------- Graph Steps --------

    def analyze_query(self, state: State):
        structured = self.llm.with_structured_output(Search)
        query = structured.invoke(state["question"])
        return {"query": query}

    def retrieve(self, state: State):
        q: Search = state["query"]
        # Get a wider pool then filter locally by metadata["section"]
        # (to avoid relying on Milvus scalar expr config)
        retrieved = self.vector_store.similarity_search(q.query, k=8)
        filtered = [d for d in retrieved if d.metadata.get("section") == q.section]
        # If filtering wipes out everything, fall back to unfiltered
        context_docs = filtered if filtered else retrieved
        return {"context": context_docs}

    def generate(self, state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = self.prompt.invoke(
            {"question": state["question"], "context": docs_content}
        )
        response = self.llm.invoke(messages)
        return {"answer": response.content}

    # -------- Graph Wiring --------

    def build_graph(self):
        graph = StateGraph(State)
        graph.add_node("analyze_query", self.analyze_query)
        graph.add_node("retrieve", self.retrieve)
        graph.add_node("generate", self.generate)

        graph.add_edge(START, "analyze_query")
        graph.add_edge("analyze_query", "retrieve")
        graph.add_edge("retrieve", "generate")
        return graph.compile()

    @staticmethod
    def save_graph_structure(graph, path: str = "graph.png"):
        png_bytes = graph.get_graph().draw_mermaid_png()
        img = Image.open(io.BytesIO(png_bytes))
        img.save(path)


# ---------------------------
# Main (example usage)
# ---------------------------

if __name__ == "__main__":
    # Ensure API key set
    if not os.getenv("GOOGLE_API_KEY"):
        raise EnvironmentError("Please set GOOGLE_API_KEY environment variable.")

    rag = LangChainRAG(milvus_uri="milvus_demo2.db")  # Milvus Lite local file

    # Example: load a single PDF
    pdf_loader = FilesHandler(FileType.pdf, "./data/1btcpp.pdf")
    raw_docs = pdf_loader.load()  # List[Document]

    # Tag sections, split, and persist
    raw_docs = rag.add_metadata_by_section(raw_docs)
    splits = rag.doc_splitter(raw_docs, chunk_size=2000, chunk_overlap=200)

    # (Optional) normalize metadata field types
    for d in splits:
        # Example: keep languages as a list if exists; don't overwrite with stringified metadata
        if "languages" not in d.metadata:
            d.metadata["languages"] = []

    rag.add_embeddings(splits)

    # Build & persist graph image
    graph = rag.build_graph()
    rag.save_graph_structure(graph)

    # Run a sample question
    question = "Explain this line: we expand TL to element-wise lookup table (ELUT) for low-bit LLMs in the appendix?"
    for update in graph.stream(
        {"question": question},
        stream_mode="updates",
    ):
        print(update, "\n----------------")
