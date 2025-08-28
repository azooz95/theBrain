from __future__ import annotations

import os
import re
import asyncio
from pathlib import Path
from typing import Optional, List, Dict

from langchain_core.tools import tool
from langchain_core.messages import ToolMessage

from src.smart_graph.agents.RagAgent import LangChainRAG, ingest_inputs  # noqa: E402

from dotenv import load_dotenv
load_dotenv()

UPLOAD_DIR = os.getenv("UPLOAD_DIR", r"C:\Users\Omneya\theBrain\uploads")

ALLOWED_EXTS = {
    ".pdf", ".docx", ".txt", ".md", ".xlsx", ".xls", ".html", ".htm"
}

SESSION: Dict[str, object] = {
    "waiting_for_upload": False,   # Expecting user to upload/ingest a file
    "ready": False,                # Documents ingested and ready for Q&A
    "last_ingested_path": None,    # Path of the last file ingested
    "last_ingested_mtime": 0.0,    # mtime of the last file ingested
}

# Singleton RAG instance
_RAG: Optional[LangChainRAG] = None


def _rag() -> LangChainRAG:
    """Create or return a singleton RAG engine."""
    global _RAG
    if _RAG is None:
        _RAG = LangChainRAG(
            collection_name=os.getenv("MILVUS_COLLECTION", "rag_collection"),
            hybrid=True,
        )
    return _RAG


# ==== Friendly prompts ====

ASK_FOR_UPLOAD = (
    "Please upload your file (PDF/DOCX/TXT/MD/Excel/HTML) or send an ingest command like:\n"
    "ingest: ./data/myfile.pdf\n\n"
    "Tip: You can also just say **use latest upload** after uploading.\n"
    "After ingestion completes, I’ll reply: 'Start to ask any question.'"
)
READY_TO_ASK = "Start to ask any question."
NO_INDEX_HINT = "I don’t see any ingested documents yet.\n" + ASK_FOR_UPLOAD
INGEST_EMPTY = (
    "⚠️ No sources to ingest. "
)


# ==== Heuristics (detect 'ask about a file') ====

FILE_QA_PATTERNS = [
    r"\bi want to ask\b.*\bquestion(s)?\b.*\bfile\b",
    r"\bquestion(s)?\b.*\babout\b.*\bfile\b",
    r"\bchat\b.*\bfile\b",
    r"\bq&a\b.*\bfile\b",
    r"\bask\b.*\bfile\b",
]


def _seems_file_qa(text: str) -> bool:
    t = (text or "").lower()
    return any(re.search(p, t) for p in FILE_QA_PATTERNS)


# ==== Uploads folder helpers ====

def _latest_upload_file() -> Optional[Path]:
    """Return the most recently modified allowed file in UPLOAD_DIR (recursively)."""
    try:
        base = Path(UPLOAD_DIR)
    except Exception:
        return None
    if not base.exists() or not base.is_dir():
        return None
    files = [
        f for f in base.rglob("*")
        if f.is_file() and f.suffix.lower() in ALLOWED_EXTS
    ]
    if not files:
        return None
    return max(files, key=lambda f: f.stat().st_mtime)


def _should_ingest(path: Path) -> bool:
    """Avoid re-ingesting the exact same file (same mtime)."""
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        return False
    last_mtime = float(SESSION.get("last_ingested_mtime") or 0.0)
    return mtime > last_mtime


async def _ingest_paths_async(items: List[str]) -> str:
    """
    Bridge to async ingest_inputs(user_input: str, rag: LangChainRAG).
    We accept multiple paths, join them into a single comma-separated string (as RagAgent expects).
    """
    payload = ", ".join(items)
    n_chunks = await ingest_inputs(payload, _rag())
    if n_chunks <= 0:
        return "⚠️ Nothing ingested. Please check the file."
    return f"✅ Ingested {n_chunks} chunks."


def _ingest_sync(items: List[str]) -> str:
    # Run the async bridge in a fresh event loop
    return asyncio.run(_ingest_paths_async(items))


def _ingest_latest_upload_or_hint() -> str:
    """
    If there is a fresh file in the uploads folder, ingest it and mark session ready.
    Otherwise, explain what to do next.
    """
    latest = _latest_upload_file()
    if latest is None:
        return "I don’t see any files in the uploads folder yet. Please upload a file."

    if not _should_ingest(latest):
        # Already ingested this exact file (same mtime)
        return "Latest file is already ingested. " + READY_TO_ASK

    summary = _ingest_sync([str(latest)])

    # Mark as ready + remember file
    try:
        mtime = latest.stat().st_mtime
    except FileNotFoundError:
        mtime = 0.0
    SESSION["ready"] = True
    SESSION["waiting_for_upload"] = False
    SESSION["last_ingested_path"] = str(latest)
    SESSION["last_ingested_mtime"] = mtime

    return summary.strip() + "\n\n" + READY_TO_ASK


# ==== The RAG Tool (friendly interaction) ====

@tool
def rag_agent(input: str) -> ToolMessage:
    """
    Friendly RAG agent with four behaviors:
      1) If user says they want to ask about a file → (try auto-ingest newest upload) or ask to upload.
      2) If user sends 'use latest upload' → ingest most recent file in uploads folder.
      3) If user sends 'ingest: <paths,urls>' → ingest those explicitly.
      4) If ingestion exists → answer questions against the current index.
    """
    try:
        text = (input or "").strip()
        low = text.lower()

        # No input → reflect current readiness
        if not text:
            return ToolMessage(
                content=READY_TO_ASK if SESSION.get("ready") else NO_INDEX_HINT,
                name="rag_agent",
                tool_call_id="tool_call_rag_agent",
            )

        # 1) Friendly entry: "I want to ask some questions about a file"
        if _seems_file_qa(text):
            SESSION["waiting_for_upload"] = True
            if SESSION.get("ready"):
                return ToolMessage(
                    content="You can use the existing documents. " + READY_TO_ASK,
                    name="rag_agent",
                    tool_call_id="tool_call_rag_agent",
                )
            # Try to auto-use the latest upload immediately if present
            auto = _ingest_latest_upload_or_hint()
            if auto.endswith(READY_TO_ASK) and auto.startswith("✅"):
                return ToolMessage(content=auto, name="rag_agent", tool_call_id="tool_call_rag_agent")
            # Otherwise ask the user to upload
            return ToolMessage(
                content=ASK_FOR_UPLOAD,
                name="rag_agent",
                tool_call_id="tool_call_rag_agent",
            )

        # 1.5) Manual shortcut: user explicitly says "use latest upload"
        if low == "use latest upload":
            msg = _ingest_latest_upload_or_hint()
            return ToolMessage(content=msg, name="rag_agent", tool_call_id="tool_call_rag_agent")

        # 2) Explicit ingestion flow (paths/URLs)
        if low.startswith("ingest:"):
            payload = text.split(":", 1)[1].strip()
            items: List[str] = [s.strip() for s in payload.split(",") if s.strip()]
            if not items:
                return ToolMessage(
                    content=INGEST_EMPTY,
                    name="rag_agent",
                    tool_call_id="tool_call_rag_agent",
                )

            # Use the async ingest bridge
            summary = _ingest_sync(items)
            SESSION["ready"] = True
            SESSION["waiting_for_upload"] = False

            final_msg = (summary.strip() + "\n\n" + READY_TO_ASK) if summary else ("✅ Ingested.\n\n" + READY_TO_ASK)
            return ToolMessage(
                content=final_msg,
                name="rag_agent",
                tool_call_id="tool_call_rag_agent",
            )

        # 3) Q&A request without prior ingestion → opportunistically ingest newest upload
        if not SESSION.get("ready"):
            auto = _ingest_latest_upload_or_hint()
            if auto.endswith(READY_TO_ASK) and ("✅ Ingested" in auto):
                # We just ingested; user can now ask
                return ToolMessage(content=auto, name="rag_agent", tool_call_id="tool_call_rag_agent")
            # Still no docs available
            SESSION["waiting_for_upload"] = True
            return ToolMessage(
                content=NO_INDEX_HINT,
                name="rag_agent",
                tool_call_id="tool_call_rag_agent",
            )

        # 4) Q&A with an existing index
        answer = _rag().answer(text, k=8)
        return ToolMessage(
            content=answer or "No answer generated.",
            name="rag_agent",
            tool_call_id="tool_call_rag_agent",
        )

    except Exception as e:
        return ToolMessage(
            content=f"❌ RAG error: {e}",
            name="rag_agent",
            tool_call_id="tool_call_rag_agent",
        )
