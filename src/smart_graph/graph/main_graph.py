import os
import re
from typing import Optional, List

from dotenv import load_dotenv
import google.generativeai as genai

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import RunnableConfig

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    SystemMessage,
    trim_messages,
)

# --- Project imports (kept as in your original) ---
from src.smart_graph.utils.state import AgentState
from src.smart_graph.tools.contract_tool import create_contract
from src.smart_graph.tools.meeting_tool import schedule_meeting
from src.smart_graph.tools.task_tool import create_or_report_task
from src.smart_graph.tools.data_tool import analyze_data
from src.smart_graph.tools.sql_tool import query_database
from src.smart_graph.tools.rag_tool import rag_agent  # <-- NEW

# 0) Environment & Model
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

gen_config = genai.types.GenerationConfig(
    temperature=0.3,
    top_p=0.9,
    top_k=40,
    max_output_tokens=256,
)

model = genai.GenerativeModel("gemini-2.0-flash")

# 1) Trimming / Token counting
def _word_token_counter(text: str) -> int:
    return len(str(text).split())

TRIM_CONFIG = {
    "strategy": "last",
    "token_counter": _word_token_counter,
    "max_tokens": 100,  # as requested
    "start_on": "human",
    "end_on": ("human", "tool"),
    "include_system": True,
    "allow_partial": False,
}

# 2) Simple in-process profile memory
PROFILE_STORE: dict[str, dict] = {}

def extract_and_store_name(text: str, thread_id: str) -> None:
    """Extract 'my name is X' or 'I'm X' and store in a tiny in-process memory."""
    if not text:
        return
    m = re.search(r"\bmy\s+name\s+is\s+([A-Za-z][\w'\-]+)", text, re.I)
    if not m:
        m = re.search(r"\bI\s*'?m\s+([A-Za-z][\w'\-]+)", text, re.I)
    if m:
        PROFILE_STORE.setdefault(thread_id, {})["name"] = m.group(1).strip()

def profile_system_hint(thread_id: str) -> Optional[str]:
    prof = PROFILE_STORE.get(thread_id, {})
    hints: List[str] = []
    if "name" in prof:
        hints.append(f"User's name is {prof['name']}. Address them by name when appropriate.")
    return " ".join(hints) if hints else None

# 3) Convert LC messages -> Gemini contents
def messages_to_gemini_contents(msgs: List[BaseMessage]) -> list:
    """Map LangChain messages to Gemini's expected format. Tool messages are ignored."""
    contents = []
    for m in msgs:
        if isinstance(m, SystemMessage):
            contents.append({"role": "user", "parts": [{"text": f"[SYSTEM] {m.content}"}]})
        elif isinstance(m, HumanMessage):
            contents.append({"role": "user", "parts": [{"text": m.content}]} )
        elif isinstance(m, AIMessage):
            contents.append({"role": "model", "parts": [{"text": m.content}]} )
    return contents

# 4) Tools registry
tools = [
    create_contract,
    schedule_meeting,
    create_or_report_task,
    analyze_data,
    query_database,
    rag_agent,  # <-- NEW
]
tool_node = ToolNode(tools=tools)

# 5) Agent logic (routing + general reply)
INTENT_CLASSIFIER_PROMPT = """You are an assistant in a multi-agent system. 
Classify the latest user message into exactly one of:
- create_contract
- schedule_meeting
- create_or_report_task
- analyze_data
- query_database
- general_query

Return only the label.
Latest user message:
"""

# Friendly RAG triggers without changing overall routing style
FILE_QA_PATTERNS = [
    r"\bi want to ask\b.*\bquestion(s)?\b.*\bfile\b",
    r"\bquestion(s)?\b.*\babout\b.*\bfile\b",
    r"\bchat\b.*\bfile\b",
    r"\bq&a\b.*\bfile\b",
    r"\bask\b.*\bfile\b",
]
def _looks_like_file_qa(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in FILE_QA_PATTERNS)

def agent_logic(state: AgentState, config: Optional[RunnableConfig] = None) -> AgentState:
    messages = state["messages"]
    user_input = messages[-1].content.strip() if messages else ""
    current_agent = state.get("current_agent")

    # Stable thread_id if provided
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = (config.get("configurable") or {}).get("thread_id")
    thread_id = thread_id or "default_thread"

    # Capture simple profile info
    extract_and_store_name(user_input, thread_id)

    # Trim history
    try:
        trimmed_history = trim_messages(
            messages,
            strategy=TRIM_CONFIG["strategy"],
            token_counter=TRIM_CONFIG["token_counter"],
            max_tokens=TRIM_CONFIG["max_tokens"],
            start_on=TRIM_CONFIG["start_on"],
            end_on=TRIM_CONFIG["end_on"],
            include_system=TRIM_CONFIG["include_system"],
            allow_partial=TRIM_CONFIG["allow_partial"],
        )
    except Exception:
        trimmed_history = messages[-6:]

    # Add system hint if available
    hint = profile_system_hint(thread_id)
    if hint:
        trimmed_history = [SystemMessage(content=hint)] + trimmed_history

    # Friendly RAG routing
    if user_input.lower().startswith("ingest:") or _looks_like_file_qa(user_input):
        return {
            "messages": messages + [
                AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "rag_agent",
                        "args": {"input": user_input},
                        "id": "tool_call_rag_agent",
                    }],
                )
            ],
            "current_agent": "rag_agent",
        }

    # Standard intent classification
    try:
        intent = model.generate_content(
            f"{INTENT_CLASSIFIER_PROMPT}{user_input}",
            generation_config=gen_config,
        ).text.strip()
    except Exception:
        intent = "general_query"

    # Route to a specific tool if matched
    if intent in {"create_contract", "schedule_meeting", "create_or_report_task", "analyze_data", "query_database"}:
        if intent == "create_contract" and current_agent != "create_contract":
            # Keep your existing contract-session reset behavior
            try:
                from src.smart_graph.tools import contract_tool
                from src.smart_graph.agents import Create_Contract
                contract_tool.SESSION_CACHE["default_user"] = {}
                Create_Contract.CONTRACT_SESSION["default_user"] = {}
                print("[INFO] Reset contract session due to agent switch.")
            except Exception as e:
                print(f"[WARNING] Contract session reset failed: {e}")

        return {
            "messages": messages + [
                AIMessage(
                    content="",
                    tool_calls=[{
                        "name": intent,
                        "args": {"input": user_input},
                        "id": f"tool_call_{intent}",
                    }],
                )
            ],
            "current_agent": intent,
        }

    # General reply path with trimmed history
    try:
        gemini_contents = messages_to_gemini_contents(trimmed_history)
        reply = model.generate_content(
            gemini_contents,
            generation_config=gen_config
        ).text.strip()
        return {
            "messages": messages + [AIMessage(content=reply)],
            "current_agent": None,
        }
    except Exception as e:
        print(f"[ERROR] General reply failed: {e}")
        return {
            "messages": messages + [AIMessage(content="Sorry, I faced an internal error. Please try again.")],
            "current_agent": None,
        }

# 6) Respond with tool output (old routing style: do NOT keep tool sticky)
def respond_with_tool(state: AgentState) -> AgentState:
    tool_messages = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    if tool_messages:
        return {
            "messages": state["messages"] + [AIMessage(content=tool_messages[-1].content)],
            "current_agent": None,  # old strategy: release back to classifier
        }
    return {
        "messages": state["messages"] + [AIMessage(content="⚠️ Tool returned nothing.")],
        "current_agent": None,
    }

# 7) Control flow edges
def should_continue(state: AgentState) -> str:
    """If the last AI message has tool_calls, route to the tool node; else end."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tool"
    return "end"

# 8) Build & Compile the graph
checkpointer = InMemorySaver()

graph = StateGraph(AgentState)
graph.add_node("llm_call", agent_logic)
graph.add_node("tool", tool_node)
graph.add_node("respond", respond_with_tool)

graph.set_entry_point("llm_call")
graph.add_conditional_edges("llm_call", should_continue, {
    "tool": "tool",
    "end": END,
})
graph.add_edge("tool", "respond")
graph.add_edge("respond", END)

app = graph.compile(checkpointer=checkpointer)
