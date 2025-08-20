from __future__ import annotations
import os
import asyncio
from typing import Optional
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from src.smart_graph.agents.RagAgent import LangChainRAG, ingest_inputs  

# Singleton RAG engine shared across tool calls (same process)
_RAG_SINGLETON: Optional[LangChainRAG] = None

def _get_rag() -> LangChainRAG:
    global _RAG_SINGLETON
    if _RAG_SINGLETON is None:
        _RAG_SINGLETON = LangChainRAG(
            collection_name="rag_collection",
            hybrid=True,
            faiss_dir=os.getenv("FAISS_DIR", "./faiss_index"),
        )
    return _RAG_SINGLETON

def _run_async(coro):
    """Run a coroutine whether or not an event loop is already running."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        # In web frameworks, prefer creating a background task; here we keep it simple.
        return asyncio.run(coro)
    return asyncio.run(coro)

@tool
def rag_agent(input: str) -> ToolMessage:
    """
    RAG agent.
    Usage:
      - 'ingest: ./data, https://example.com/page'  -> ingest docs/URLs into FAISS
      - 'What is ... ?'                              -> answer using current index (hybrid vec+BM25)

    Returns a ToolMessage with plain-text content.
    """
    rag = _get_rag()
    text = (input or "").strip()

    # Ingest mode (explicit)
    if text.lower().startswith("ingest:"):
        payload = text[len("ingest:"):].strip(" ,")
        try:
            n = _run_async(ingest_inputs(payload, rag))
            return ToolMessage(
                content=f"✅ Ingested {n} chunks." if n > 0 else "⚠️ Please check your paths/URLs.",
                name="rag_agent",
                tool_call_id="tool_call_rag_agent",
            )
        except Exception as e:
            return ToolMessage(
                content=f"❌ RAG ingest failed: {e}",
                name="rag_agent",
                tool_call_id="tool_call_rag_agent",
            )

    # Query mode (default)
    try:
        # If no index exists yet, guide user to ingest first
        if getattr(rag, "vector_store", None) is None:
            return ToolMessage(
                content="ℹ️ No index found. Please run: `ingest: <path or URL, comma-separated>`",
                name="rag_agent",
                tool_call_id="tool_call_rag_agent",
            )
        answer = rag.answer(text, k=8)
        return ToolMessage(
            content=answer or "No answer generated.",
            name="rag_agent",
            tool_call_id="tool_call_rag_agent",
        )
    except Exception as e:
        return ToolMessage(
            content=f"❌ RAG query failed: {e}",
            name="rag_agent",
            tool_call_id="tool_call_rag_agent",
        )
