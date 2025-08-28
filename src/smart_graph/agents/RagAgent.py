import os
import re
import sys
import asyncio
import logging
import hashlib
from typing import List, Optional, Dict

# Set a benign User-Agent if not set, to silence warnings
os.environ.setdefault("USER_AGENT", "UpdatedRag/1.0 (Windows; LangChain; Milvus)")

from dotenv import load_dotenv

# LangChain core
from langchain import hub
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Loaders (kept close to your original choices)
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders.pdf import UnstructuredPDFLoader
from langchain_community.document_loaders import (
    WebBaseLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredHTMLLoader,
    TextLoader,
)

# Vector store (Milvus)
from langchain_community.vectorstores import Milvus

# Google GenAI (Embeddings + Chat)
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)

# BM25 (lexical retriever)
from langchain_community.retrievers import BM25Retriever

# Logging
logger = logging.getLogger("rag_app")
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)


# Utilities

def load_supported_from_dir(dir_path: str) -> List[Document]:
    """
    Load recursively from a directory, then filter by allowed extensions.
    """
    loader = DirectoryLoader(dir_path, recursive=True, silent_errors=True)
    docs = loader.load()
    allowed_ext = {".xlsx", ".xls", ".pdf", ".docx", ".html", ".htm", ".txt", ".md"}
    filtered = [d for d in docs if str(d.metadata.get("source", "")).lower().endswith(tuple(allowed_ext))]
    logger.info(f"[dir] Loaded {len(filtered)} supported docs from directory.")
    return filtered


def chunk_documents(docs: List[Document], chunk_size: int = 1500, chunk_overlap: int = 200) -> List[Document]:
    """
    Split documents into overlapping chunks for better retrieval.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)


def tag_sections(chunks: List[Document]) -> List[Document]:
    """
    Add a simple 'section' metadata label (beginning/middle/end).
    """
    total = len(chunks)
    if total == 0:
        return chunks
    third = max(1, total // 3)
    for i, d in enumerate(chunks):
        d.metadata["section"] = "beginning" if i < third else "middle" if i < 2 * third else "end"
        d.metadata["source"] = d.metadata.get("source") or d.metadata.get("file_path") or "unknown_source"
    return chunks


def _stable_id(doc: Document) -> str:
    """
    Create a stable ID for a document chunk based on source and content.
    Avoids Python's salted hash() so results are stable across runs.
    """
    src = str(doc.metadata.get("source", ""))
    digest = hashlib.sha1(doc.page_content.encode("utf-8")).hexdigest()
    return f"{src}::{digest}"


def rr_fusion(rank_lists: List[List[Document]], k: int = 8, k_rr: int = 60) -> List[Document]:
    """
    Reciprocal Rank Fusion: combine multiple ranked lists (vector + BM25).
    """
    scores: Dict[str, float] = {}
    seen: Dict[str, Document] = {}
    for results in rank_lists:
        for rnk, doc in enumerate(results):
            key = _stable_id(doc)
            seen[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (k_rr + rnk + 1)
    fused = sorted(seen.items(), key=lambda kv: scores.get(kv[0], 0.0), reverse=True)
    return [doc for _, doc in fused][:k]


# Typed loaders

class Excelsheet:
    def __call__(self, path: str):
        return UnstructuredExcelLoader(path, mode="elements")


class PDF:
    def __call__(self, path: str):
        return UnstructuredPDFLoader(path, mode="elements")


class Docx:
    def __call__(self, path: str):
        return UnstructuredWordDocumentLoader(path, mode="elements")


class HTML:
    def __call__(self, path: str):
        return UnstructuredHTMLLoader(path)


class Web:
    async def __call__(self, paths: List[str]) -> List[Document]:
        """
        Async web loader to fetch multiple URLs; supports both new and old WebBaseLoader signatures.
        """
        docs: List[Document] = []
        try:
            try:
                loader = WebBaseLoader(web_paths=paths)  # newer signature
            except TypeError:
                loader = WebBaseLoader(paths)           # older signature
            try:
                async for doc in loader.alazy_load():
                    docs.append(doc)
            except AttributeError:
                docs.extend(loader.load())
        except Exception as e:
            logger.error(f"[web] Load error: {e}")
        logger.info(f"[web] Loaded {len(docs)} docs from {len(paths)} URLs.")
        return docs


# File type detection

class FileType:
    EXCEL = "excel"
    PDF = "pdf"
    DOCX = "docx"
    HTML = "html"
    WEB = "web"
    DIR = "dir"
    OTHER = "other"


def classify_path(p: str) -> str:
    p_low = p.lower().strip()
    if re.match(r"^https?://", p_low):
        return FileType.WEB
    if os.path.isdir(p):
        return FileType.DIR
    _, ext = os.path.splitext(p_low)
    return {
        ".xlsx": FileType.EXCEL,
        ".xls": FileType.EXCEL,
        ".pdf": FileType.PDF,
        ".docx": FileType.DOCX,
        ".html": FileType.HTML,
        ".htm": FileType.HTML,
        ".txt": FileType.OTHER,
        ".md": FileType.OTHER,
    }.get(ext, FileType.OTHER)


def load_one_path(kind: str, path: str) -> List[Document]:
    if kind == FileType.EXCEL:
        loader = Excelsheet()(path)
        docs = loader.load()
    elif kind == FileType.PDF:
        loader = PDF()(path)
        docs = loader.load()
    elif kind == FileType.DOCX:
        loader = Docx()(path)
        docs = loader.load()
    elif kind == FileType.HTML:
        loader = HTML()(path)
        docs = loader.load()
    elif kind == FileType.OTHER:
        # Try Text/MD directly; fallback to DirectoryLoader pattern
        try:
            if path.lower().endswith((".txt", ".md")):
                docs = TextLoader(path, encoding="utf-8").load()
            else:
                base = os.path.dirname(path) or "."
                pat = os.path.basename(path)
                docs = [d for d in DirectoryLoader(base, glob=pat, silent_errors=True).load()]
        except Exception:
            logger.warning(f"[load] Unsupported file type for {path}; skipping.")
            docs = []
    else:
        docs = []
    logger.info(f"[load] Loaded {len(docs)} docs from file.")
    return docs


# RAG Core — Milvus backend
class LangChainRAG:
    """
    Core RAG engine:
      - Embeddings: Google Generative AI Embeddings (embedding-001)
      - Vector store: Milvus (remote / server)
      - Optional BM25 lexical retriever
      - Answer generation: Gemini Flash
    """
    def __init__(
        self,
        collection_name: Optional[str] = None,
        hybrid: bool = True,
    ):
        load_dotenv(override=True)

        if not os.getenv("GOOGLE_API_KEY"):
            raise EnvironmentError("GOOGLE_API_KEY is required in the environment or .env")

        # Embeddings (embedding-001 has 768 dims)
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        # LLM (Gemini Flash) — used only to generate final answers
        try:
            self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
            logger.info("[llm] Using gemini-2.0-flash")
        except Exception:
            logger.warning("[llm] gemini-2.0-flash unavailable; falling back to gemini-1.5-flash")
            self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

        # Milvus connection args from .env
        self.collection_name = collection_name or os.getenv("MILVUS_COLLECTION", "rag_collection")

        secure = str(os.getenv("MILVUS_SECURE", "false")).lower() == "true"
        tls_ca = os.getenv("MILVUS_TLS_CA_CERT", "").strip() or None

        self.connection_args = {
            "host": os.getenv("MILVUS_HOST", "localhost"),
            "port": os.getenv("MILVUS_PORT", "19530"),
            # These may be optional depending on your deployment:
            "user": os.getenv("MILVUS_USER") or None,
            "password": os.getenv("MILVUS_PASSWORD") or None,
            "secure": secure,
            "tls_ca_cert": tls_ca,
        }

        # Index/search params from .env (with safe defaults)
        import json
        self.metric_type = os.getenv("MILVUS_METRIC_TYPE", "COSINE")
        index_type = os.getenv("MILVUS_INDEX_TYPE", "IVF_FLAT")

        try:
            index_param_json = os.getenv("MILVUS_INDEX_PARAM_JSON", '{"nlist":1024}')
            self.index_params = {
                "index_type": index_type,
                "metric_type": self.metric_type,
                "params": json.loads(index_param_json),
            }
        except Exception:
            logger.warning("[milvus] Invalid MILVUS_INDEX_PARAM_JSON. Falling back to {'nlist':1024}.")
            self.index_params = {"index_type": index_type, "metric_type": self.metric_type, "params": {"nlist": 1024}}

        try:
            search_param_json = os.getenv("MILVUS_SEARCH_PARAM_JSON", '{"nprobe":16}')
            self.search_params = {
                "metric_type": self.metric_type,
                "params": json.loads(search_param_json),
            }
        except Exception:
            logger.warning("[milvus] Invalid MILVUS_SEARCH_PARAM_JSON. Falling back to {'nprobe':16}.")
            self.search_params = {"metric_type": self.metric_type, "params": {"nprobe": 16}}

        # Vector store handle
        self.vector_store: Optional[Milvus] = None
        self._try_load_milvus_collection()

        self.hybrid_enabled = hybrid
        self._bm25_retriever: Optional[BM25Retriever] = None

        # Prompt template from Hub with a safe fallback
        try:
            self.prompt = hub.pull("rlm/rag-prompt")
            logger.info("[prompt] Pulled prompt from hub: rlm/rag-prompt")
        except Exception as e:
            logger.warning(f"[prompt] Hub pull failed; using fallback prompt. ({e})")
            from langchain_core.prompts import ChatPromptTemplate
            self.prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "Use the provided context to answer the question concisely and accurately."),
                    ("human", "Question:\n{question}\n\nContext:\n{context}")
                ]
            )

    def _try_load_milvus_collection(self):
        """
        Try to bind to an existing Milvus collection.
        If it doesn't exist yet, we'll create it on first ingest.
        """
        try:
            self.vector_store = Milvus.from_existing_collection(
                embedding=self.embeddings,
                collection_name=self.collection_name,
                connection_args={k: v for k, v in self.connection_args.items() if v not in (None, "", False)},
                search_params=self.search_params,
            )
            logger.info(f"[milvus] Connected to existing collection '{self.collection_name}'.")
        except Exception as e:
            logger.info(f"[milvus] No existing collection '{self.collection_name}' yet ({e}). Will create on first ingest.")
            self.vector_store = None

    def _create_or_attach_collection_with_docs(self, chunks: List[Document]):
        """
        Create the collection (if absent) and insert docs.
        """
        self.vector_store = Milvus.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            connection_args={k: v for k, v in self.connection_args.items() if v not in (None, "", False)},
            index_params=self.index_params,
            search_params=self.search_params,

        )
        logger.info(f"[milvus] Created/updated collection '{self.collection_name}' and inserted {len(chunks)} chunks.")

    def ingest_documents(self, docs: List[Document], chunk_size=1500, chunk_overlap=200) -> int:
        if not docs:
            logger.warning("[ingest] No docs to ingest.")
            return 0

        chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = tag_sections(chunks)
        for d in chunks:
            d.metadata.setdefault("languages", [])

        if self.vector_store is None:
            # First time: create collection and index by inserting documents
            self._create_or_attach_collection_with_docs(chunks)
        else:
            # Collection exists: just add docs
            self.vector_store.add_documents(chunks)
            logger.info(f"[ingest] Inserted {len(chunks)} chunks into Milvus collection '{self.collection_name}'.")

        if self.hybrid_enabled:
            self._bm25_retriever = BM25Retriever.from_documents(chunks)
            self._bm25_retriever.k = 8
            logger.info("[ingest] BM25 retriever built/updated.")
        return len(chunks)

    def retrieve(self, query: str, k: int = 8) -> List[Document]:
        if self.vector_store is None:
            logger.warning("[retrieve] Vector store empty. Ingest first.")
            return []
        vec_docs = self.vector_store.similarity_search(query, k=k)
        if self.hybrid_enabled and self._bm25_retriever is not None:
            bm25_docs = self._bm25_retriever.invoke(query)
            fused = rr_fusion([vec_docs, bm25_docs], k=k, k_rr=60)
            logger.info(f"[retrieve] vec={len(vec_docs)}, bm25={len(bm25_docs)}, fused={len(fused)}")
            return fused
        logger.info(f"[retrieve] vec_only={len(vec_docs)}")
        return vec_docs

    def generate(self, question: str, docs: List[Document]) -> str:
        if not docs:
            return (
                "No relevant context found from the ingested materials. "
                "Please ingest files/URLs first or try a clearer question."
            )
        docs_content = "\n\n".join(doc.page_content for doc in docs)
        messages = self.prompt.invoke({"question": question, "context": docs_content})
        response = self.llm.invoke(messages)
        return response.content

    def answer(self, question: str, k: int = 8) -> str:
        ctx = self.retrieve(question, k=k)
        return self.generate(question, ctx)


# Ingestion orchestration

async def ingest_inputs(user_input: str, rag: LangChainRAG) -> int:
    """
    Accept single or comma-separated inputs. Each item can be:
      - directory path
      - file path (pdf, docx, xlsx, html, txt, md)
      - http/https URL
    """
    raw_items = [s.strip() for s in user_input.split(",") if s.strip()]
    if not raw_items:
        logger.error("No path or URL provided.")
        return 0

    web_urls: List[str] = []
    docs_acc: List[Document] = []

    for item in raw_items:
        kind = classify_path(item)
        if kind == FileType.WEB:
            web_urls.append(item)
        elif kind == FileType.DIR:
            docs_acc.extend(load_supported_from_dir(item))
        else:
            docs_acc.extend(load_one_path(kind, item))

    if web_urls:
        web_loader = Web()
        web_docs = await web_loader(web_urls)
        docs_acc.extend(web_docs)

    if not docs_acc:
        logger.warning("[ingest] No documents collected from inputs.")
        return 0

    n_chunks = rag.ingest_documents(docs_acc)
    return n_chunks


# CLI (commented out to keep your original pattern)
"""
def main():
    load_dotenv(override=True)

    if not os.getenv("GOOGLE_API_KEY"):
        print("ERROR: set GOOGLE_API_KEY in .env or environment and rerun.")
        sys.exit(1)

    # Initialize RAG engine (Milvus)
    rag = LangChainRAG(
        collection_name=os.getenv("MILVUS_COLLECTION", "rag_collection"),
        hybrid=True,
    )

    print("Enter a path/URL to ingest (you can provide multiple, comma-separated).")
    print("Examples:")
    print("  ./data")
    print("  ./data/report.pdf, https://example.com/page")

    # 1) Collect inputs for ingestion
    try:
        inputs_line = input("Paths/URLs: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nExiting.")
        return

    if not inputs_line:
        print("No paths/URLs entered. Exiting.")
        return

    # 2) Ingest
    try:
        n_chunks = asyncio.run(ingest_inputs(inputs_line, rag))
        print(f"Ingested {n_chunks} chunks.")
        if n_chunks == 0:
            print("Nothing ingested. Please check your inputs and try again.")
            return
    except Exception as e:
        logger.error(f"[ingest] Failed: {e}")
        print("Ingestion failed. See logs for details.")
        return

    # 3) Q&A loop
    print("\nNow ask your questions. Type 'exit' to quit.")
    while True:
        try:
            q = input("\nQ: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit", "q"):
            print("Bye.")
            break

        try:
            answer_text = rag.answer(q, k=8)
            print("ANSWER\n")
            print(answer_text or "No answer generated.")
        except Exception as e:
            logger.error(f"[answer] Failed: {e}")
            print("An error occurred while generating an answer. Check logs and try again.")
            continue


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
"""
