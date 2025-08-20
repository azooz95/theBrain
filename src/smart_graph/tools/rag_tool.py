from __future__ import annotations
import re
from typing import Optional, List, Dict

from langchain_core.tools import tool
from langchain_core.messages import ToolMessage

from src.smart_graph.agents.RagAgent import LangChainRAG, ingest_inputs  # noqa: E402

# Simple in-process session store (flags)
SESSION: Dict[str, bool] = {
    "waiting_for_upload": False,  # Expecting user to upload/ingest a file
    "ready": False,               # Documents ingested and ready for Q&A
}

# Singleton RAG instance
_RAG: Optional[LangChainRAG] = None

def _rag() -> LangChainRAG:
    global _RAG
    if _RAG is None:
        _RAG = LangChainRAG(
            collection_name="rag_collection",
            persist_dir=".rag_store",
        )
    return _RAG

# Friendly prompts
ASK_FOR_UPLOAD = (
    "Please upload your file (PDF/DOCX/TXT) or send an ingest command like:\n"
    "ingest: ./data/myfile.pdf\n\n"
    "After ingestion completes, I’ll reply: 'Start to ask any question.'"
)
READY_TO_ASK = "Start to ask any question."
NO_INDEX_HINT = "I don’t see any ingested documents yet.\n" + ASK_FOR_UPLOAD
INGEST_EMPTY = (
    "⚠️ No sources to ingest. Example:\n"
    "ingest: ./data, ./docs/policy.pdf, https://example.com/help"
)

# Patterns that detect when the user *wants to ask about a file*
FILE_QA_PATTERNS = [
    r"\bi want to ask\b.*\bquestion(s)?\b.*\bfile\b",
    r"\bquestion(s)?\b.*\babout\b.*\bfile\b",
    r"\bchat\b.*\bfile\b",
    r"\bq&a\b.*\bfile\b",
    r"\bask\b.*\bfile\b",
]

def _seems_file_qa(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in FILE_QA_PATTERNS)

# The RAG tool (friendly interaction)

@tool
def rag_agent(input: str) -> ToolMessage:
    """
    Friendly RAG agent with three modes:
      1) If user says they want to ask about a file -> ask for upload.
      2) If user uses 'ingest:' -> build index and confirm with 'Start to ask any question'.
      3) If user asks a question without ingestion -> guide them to upload first.
      4) If ingestion exists -> answer normally.
    """
    try:
        text = (input or "").strip()
        if not text:
            return ToolMessage(
                content=READY_TO_ASK if SESSION["ready"] else NO_INDEX_HINT,
                name="rag_agent",
                tool_call_id="tool_call_rag_agent",
            )

        # 1) Friendly entry: "I want to ask some questions about a file"
        if _seems_file_qa(text):
            SESSION["waiting_for_upload"] = True
            if SESSION["ready"]:
                return ToolMessage(
                    content="You can use the existing documents. " + READY_TO_ASK,
                    name="rag_agent",
                    tool_call_id="tool_call_rag_agent",
                )
            return ToolMessage(
                content=ASK_FOR_UPLOAD,
                name="rag_agent",
                tool_call_id="tool_call_rag_agent",
            )

        # 2) Ingestion flow
        if text.lower().startswith("ingest:"):
            payload = text.split(":", 1)[1].strip()
            items: List[str] = [s.strip() for s in payload.split(",") if s.strip()]
            if not items:
                return ToolMessage(
                    content=INGEST_EMPTY,
                    name="rag_agent",
                    tool_call_id="tool_call_rag_agent",
                )

            summary = ingest_inputs(items)
            SESSION["ready"] = True
            SESSION["waiting_for_upload"] = False

            final_msg = (summary.strip() + "\n\n" + READY_TO_ASK) if summary else ("✅ Ingested.\n\n" + READY_TO_ASK)
            return ToolMessage(
                content=final_msg,
                name="rag_agent",
                tool_call_id="tool_call_rag_agent",
            )

        # 3) Q&A without ingestion
        if not SESSION["ready"]:
            SESSION["waiting_for_upload"] = True
            return ToolMessage(
                content=NO_INDEX_HINT,
                name="rag_agent",
                tool_call_id="tool_call_rag_agent",
            )

        # 4) Q&A with ingestion ready
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
