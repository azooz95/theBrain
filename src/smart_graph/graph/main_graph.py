import os
import re
from typing import Optional, List, Set, Dict

from dotenv import load_dotenv

# LangGraph / LangChain
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

# Gemini (LangChain wrapper) for tool binding
from langchain_google_genai import ChatGoogleGenerativeAI

# ===== Project imports (your tools must be LangChain Tools) =====
from src.smart_graph.utils.state import AgentState
from src.smart_graph.tools.contract_tool import create_contract
from src.smart_graph.tools.meeting_tool import schedule_meeting
from src.smart_graph.tools.task_tool import create_or_report_task
from src.smart_graph.tools.data_tool import analyze_data
from src.smart_graph.tools.sql_tool import query_database
from src.smart_graph.tools.rag_tool import rag_agent  # RAG integration

# ===================== Environment =====================
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Router LLM (tool-calling)
LLM_ROUTER = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.2,               # low temperature for stable routing
    top_p=0.9,
    convert_system_message_to_human=True,
    api_key=GOOGLE_API_KEY,
)

# General LLM (non tool-calling) for normal replies (greetings/small-talk/general)
LLM_GENERAL = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.6,               # a bit more creative for user-facing text
    top_p=0.9,
    convert_system_message_to_human=True,
    api_key=GOOGLE_API_KEY,
)

# ===================== Tools Registry =====================
TOOLS_ORDER: List[str] = [
    "create_contract",
    "schedule_meeting",
    "create_or_report_task",
    "analyze_data",
    "query_database",
    "rag_agent",
]
TOOLS_MAP = {
    "create_contract": create_contract,
    "schedule_meeting": schedule_meeting,
    "create_or_report_task": create_or_report_task,
    "analyze_data": analyze_data,
    "query_database": query_database,
    "rag_agent": rag_agent,
}
TOOL_NAMES: Set[str] = set(TOOLS_ORDER)
tool_node = ToolNode(tools=list(TOOLS_MAP.values()))

# Bind tools so Gemini returns structured tool calls
LLM_ROUTER_BOUND = LLM_ROUTER.bind_tools(list(TOOLS_MAP.values()))

# ===================== Trimming / Token counting =====================
def _word_token_counter(text: str) -> int:
    return len(str(text).split())

TRIM_CONFIG: Dict[str, object] = {
    "strategy": "last",
    "token_counter": _word_token_counter,
    "max_tokens": 100,
    "start_on": "human",
    "end_on": ("human", "tool"),
    "include_system": True,
    "allow_partial": False,
}

# ===================== Simple Profile Memory =====================
PROFILE_STORE: dict[str, dict] = {}

def extract_and_store_name(text: str, thread_id: str) -> None:
    """Extract 'my name is X' / 'I'm X' / 'اسمي X' and store it."""
    if not text:
        return
    m = re.search(r"\bmy\s+name\s+is\s+([A-Za-z][\w'\-]+)", text, re.I)
    if not m:
        m = re.search(r"\bI\s*'?m\s+([A-Za-z][\w'\-]+)", text, re.I)
    if not m:
        m = re.search(r"(?:اسمي|انا)\s+([^\s]+)", text, re.I)
    if m:
        PROFILE_STORE.setdefault(thread_id, {})["name"] = m.group(1).strip()

def profile_system_hint(thread_id: str) -> Optional[str]:
    """Return a short system hint derived from memory."""
    prof = PROFILE_STORE.get(thread_id, {})
    hints: List[str] = []
    if "name" in prof:
        hints.append(f"User's name is {prof['name']}. Address them by name when appropriate.")
    return " ".join(hints) if hints else None

# ===================== Greetings / Breakers / Switch =====================
BREAK_WORDS = {
    "cancel", "stop", "exit", "new", "new topic", "switch", "change agent",
    "hi", "hello", "hey", "thanks", "thank you", "ok", "okay"
}
GREETING_REGEX = re.compile(
    r"^\s*(hi|hello|hey|yo|morning|good\s+morning|evening|good\s+evening|"
    r"thanks|thank\s+you|thx|ok|okay|cool|great|nice)\s*[!.]*\s*$",
    re.I
)
def _is_greeting(text: str) -> bool:
    return bool(GREETING_REGEX.match(text or ""))

# Explicit switching helpers (aliases and commands)
AGENT_ALIASES = {
    "contract": "create_contract", "contracts": "create_contract",
    "meeting": "schedule_meeting",
    "task": "create_or_report_task", "tasks": "create_or_report_task",
    "data": "analyze_data",
    "database": "query_database", "db": "query_database", "sql": "query_database", "databases": "query_database",
    "rag": "rag_agent", "files": "rag_agent",
}
SWITCH_REGEX = re.compile(
    r"(?:^|\b)(?:switch|route|go|use|start|run|open)\s*(?:to|with|the)?\s*(@?\w+|/agent\s+\w+|agent:\s*\w+)\b",
    re.I
)
HANDLE_REGEXES = [
    re.compile(r"^/agent\s+(\w+)$", re.I),
    re.compile(r"^agent:\s*(\w+)$", re.I),
    re.compile(r"^@(\w+)$", re.I),
]
def _resolve_agent_name(name: str) -> Optional[str]:
    n = name.strip().lower().lstrip("@")
    if n in TOOLS_ORDER:
        return n
    if n in AGENT_ALIASES:
        return AGENT_ALIASES[n]
    return None
def _mentions_other_agent(text: str, current_agent: Optional[str]) -> Optional[str]:
    """Detect explicit/implicit requests to switch to another agent."""
    t = text.strip()

    # 1) explicit handles
    for rx in HANDLE_REGEXES:
        m = rx.search(t)
        if m:
            target = _resolve_agent_name(m.group(1))
            if target and target != current_agent:
                return target

    # 2) "switch/use/start ... <name>"
    m = SWITCH_REGEX.search(t)
    if m:
        raw = m.group(1)
        raw = re.sub(r"^/agent\s+", "", raw, flags=re.I)
        raw = re.sub(r"^agent:\s*", "", raw, flags=re.I)
        target = _resolve_agent_name(raw)
        if target and target != current_agent:
            return target

    # 3) plain mentions (tool/alias words)
    lt = t.lower()
    for name in TOOLS_ORDER:
        if name != current_agent and re.search(rf"\b{name}\b", lt):
            return name
    for alias, canonical in AGENT_ALIASES.items():
        if canonical != current_agent and re.search(rf"\b{alias}\b", lt):
            return canonical

    # 4) last-resort: database keywords
    if re.search(r"\b(data\s*base|database|databases|db|sql)\b", lt):
        return "query_database"

    return None

# ===================== RAG Heuristics + Safeguard =====================
FILE_QA_PATTERNS = [
    r"\bi want to ask\b.*\bquestion(s)?\b.*\bfile\b",
    r"\bquestion(s)?\b.*\babout\b.*\bfile\b",
    r"\bchat\b.*\bfile\b",
    r"\bq&a\b.*\bfile\b",
    r"\bask\b.*\bfile\b",
    r"\b(سؤال|اسئلة)\b.*\b(ملف|مستند)\b",
]
def _looks_like_file_qa(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in FILE_QA_PATTERNS)

# ===================== Reset support for create_contract =====================
RESET_CONTRACT_REGEXES = [
    re.compile(r"\b(reset|restart|start\s*over)\s+(contract|create\s*contract)\b", re.I),
    re.compile(r"^/reset\s+contract\b", re.I),
]
def _wants_reset_contract(text: str) -> bool:
    for rx in RESET_CONTRACT_REGEXES:
        if rx.search(text or ""):
            return True
    return False
def _reset_create_contract_session():
    """Force-clear any caches/sessions for create_contract before starting fresh."""
    try:
        from src.smart_graph.tools import contract_tool
        from src.smart_graph.agents import Create_Contract
        try:
            contract_tool.SESSION_CACHE["default_user"] = {}
        except Exception:
            pass
        try:
            contract_tool.PLACEHOLDER_CACHE.clear()
        except Exception:
            pass
        try:
            Create_Contract.CONTRACT_SESSION["default_user"] = {}
        except Exception:
            pass
        print("[INFO] create_contract session reset.")
    except Exception as e:
        print(f"[WARNING] create_contract session reset failed: {e}")

# ===================== Tool completion detector =====================
DONE_PATTERNS = [
    r"\bcontract generated\b",
    r"\breport generated\b",
    r"\bsaved as\b",
    r"\bgenerated successfully\b",
    r"\bcompleted\b",
    r"\bdone\b",
    r"\bdownload\b",
]
def _tool_output_is_done(text: str) -> bool:
    t = (text or "").lower()
    return any(re.search(p, t) for p in DONE_PATTERNS)

# ===================== Helpers =====================
def _llm_reply(messages_for_llm: List[BaseMessage]) -> str:
    """Generate a natural language reply using the general LLM (no tool calling)."""
    try:
        out = LLM_GENERAL.invoke(messages_for_llm)
        content = out.content if isinstance(out.content, str) else (out.content or "")
        return content.strip() or "How can I help you next?"
    except Exception as e:
        print(f"[WARN] _llm_reply error: {e}")
        return "How can I help you next?"

# ===================== Agent Logic =====================
def agent_logic(state: AgentState, config: Optional[RunnableConfig] = None) -> AgentState:
    messages = state["messages"]
    user_input = messages[-1].content.strip() if messages else ""
    current_agent = state.get("current_agent")

    # Stable thread_id
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = (config.get("configurable") or {}).get("thread_id")
    thread_id = thread_id or "default_thread"

    # Capture simple profile info
    extract_and_store_name(user_input, thread_id)

    # 0) Greeting short-circuit → generate reply with LLM (no routing)
    if _is_greeting(user_input):
        # Build a minimal trimmed history to keep the tone contextual
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
            trimmed_history = messages[-4:]
        hint = profile_system_hint(thread_id)
        if hint:
            trimmed_history = [SystemMessage(content=hint)] + trimmed_history
        reply = _llm_reply(trimmed_history + [HumanMessage(content=user_input)])
        return {"messages": state["messages"] + [AIMessage(content=reply)], "current_agent": None}

    # Build trimmed history (used for both router and general reply paths)
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
    hint = profile_system_hint(thread_id)
    if hint:
        trimmed_history = [SystemMessage(content=hint)] + trimmed_history

    # 1) Explicit reset for contracts
    if _wants_reset_contract(user_input):
        _reset_create_contract_session()
        return {
            "messages": messages + [AIMessage(content="", tool_calls=[{
                "name": "create_contract", "args": {"input": "single"}, "id": "tool_call_create_contract"
            }])],
            "current_agent": "create_contract",
        }

    # 2) Break words clear stickiness
    if any(w in user_input.lower() for w in BREAK_WORDS):
        current_agent = None

    # 3) Explicit switch wins immediately
    named = _mentions_other_agent(user_input, current_agent)
    if named:
        if named == "create_contract":
            _reset_create_contract_session()
        return {
            "messages": messages + [AIMessage(content="", tool_calls=[{
                "name": named, "args": {"input": user_input}, "id": f"tool_call_{named}"
            }])],
            "current_agent": named,
        }

    # 4) Sticky routing (continue with current agent)
    if current_agent in TOOLS_MAP:
        return {
            "messages": messages + [AIMessage(content="", tool_calls=[{
                "name": current_agent, "args": {"input": user_input}, "id": f"tool_call_{current_agent}"
            }])],
            "current_agent": current_agent,
        }

    # 5) RAG fast-path (heuristics)
    if user_input.lower().startswith("ingest:") or _looks_like_file_qa(user_input):
        return {
            "messages": messages + [AIMessage(content="", tool_calls=[{
                "name": "rag_agent", "args": {"input": user_input}, "id": "tool_call_rag_agent"
            }])],
            "current_agent": "rag_agent",
        }

    # 6) LLM Router via bind_tools (Gemini-2.0-flash)
    router_input: List[BaseMessage] = trimmed_history + [HumanMessage(content=user_input)]
    router_out: AIMessage = LLM_ROUTER_BOUND.invoke(router_input)

    tool_calls = getattr(router_out, "tool_calls", None)
    if tool_calls:
        tc = tool_calls[0]
        chosen = tc.get("name")
        args = tc.get("args", {}) or {"input": user_input}

        # RAG safety: only allow if user clearly intends doc Q&A / ingest
        if chosen == "rag_agent" and not (
            user_input.lower().startswith("ingest:") or _looks_like_file_qa(user_input)
        ):
            chosen = None

        # Reset when switching into create_contract
        if chosen == "create_contract":
            _reset_create_contract_session()

        if chosen in TOOLS_MAP:
            return {
                "messages": messages + [AIMessage(content="", tool_calls=[{
                    "name": chosen, "args": args, "id": f"tool_call_{chosen}"
                }])],
                "current_agent": chosen,
            }

    # 7) No (or filtered) tool call → produce a normal LLM answer
    reply = _llm_reply(router_input)
    return {"messages": messages + [AIMessage(content=reply)], "current_agent": None}

# ===================== Respond with tool output =====================
def respond_with_tool(state: AgentState) -> AgentState:
    tool_messages = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    if tool_messages:
        last_tool = tool_messages[-1]
        content = last_tool.content

        # Auto-unstick after clear completion messages
        if _tool_output_is_done(content):
            return {"messages": state["messages"] + [AIMessage(content=content)], "current_agent": None}

        return {"messages": state["messages"] + [AIMessage(content=content)],
                "current_agent": state.get("current_agent")}

    return {"messages": state["messages"] + [AIMessage(content="⚠️ Tool returned nothing.")],
            "current_agent": state.get("current_agent")}

# ===================== Control Flow =====================
def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tool"
    return "end"

# ===================== Graph Build =====================
checkpointer = InMemorySaver()

graph = StateGraph(AgentState)
graph.add_node("llm_call", agent_logic)
graph.add_node("tool", tool_node)
graph.add_node("respond", respond_with_tool)

graph.set_entry_point("llm_call")
graph.add_conditional_edges("llm_call", should_continue, {"tool": "tool", "end": END})
graph.add_edge("tool", "respond")
graph.add_edge("respond", END)

app = graph.compile(checkpointer=checkpointer)
