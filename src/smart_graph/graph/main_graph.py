import os
import re
from typing import Optional, List, Set, Dict

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

# ===== Project imports =====
from src.smart_graph.utils.state import AgentState
from src.smart_graph.tools.contract_tool import create_contract
from src.smart_graph.tools.meeting_tool import schedule_meeting
from src.smart_graph.tools.task_tool import create_or_report_task
from src.smart_graph.tools.data_tool import analyze_data
from src.smart_graph.tools.sql_tool import query_database
from src.smart_graph.tools.rag_tool import rag_agent  # RAG integration

# ====== Environment & Model ======
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

gen_config = genai.types.GenerationConfig(
    temperature=0.3,
    top_p=0.9,
    top_k=40,
    max_output_tokens=256,
)

model = genai.GenerativeModel("gemini-2.0-flash")

# ====== Trimming / Token counting ======
def _word_token_counter(text: str) -> int:
    return len(str(text).split())

TRIM_CONFIG: Dict[str, object] = {
    "strategy": "last",
    "token_counter": _word_token_counter,
    "max_tokens": 100,  # per preference
    "start_on": "human",
    "end_on": ("human", "tool"),
    "include_system": True,
    "allow_partial": False,
}

# ====== Simple in-process profile memory ======
PROFILE_STORE: dict[str, dict] = {}

def extract_and_store_name(text: str, thread_id: str) -> None:
    """Extract 'my name is X' or 'I'm X' and store it."""
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

# ====== Helper: convert LC messages -> Gemini contents ======
def messages_to_gemini_contents(msgs: List[BaseMessage]) -> list:
    contents = []
    for m in msgs:
        if isinstance(m, SystemMessage):
            contents.append({"role": "user", "parts": [{"text": f"[SYSTEM] {m.content}"}]})
        elif isinstance(m, HumanMessage):
            contents.append({"role": "user", "parts": [{"text": m.content}]})
        elif isinstance(m, AIMessage):
            contents.append({"role": "model", "parts": [{"text": m.content}]})
        # ToolMessage intentionally ignored for summarization input
    return contents

# ====== Tools registry (includes rag_agent) ======
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
tool_node = ToolNode(tools=list(TOOLS_MAP.values()))
TOOL_NAMES: Set[str] = set(TOOLS_ORDER)

# ====== Friendly RAG triggers (heuristics) ======
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

# ====== Intent classifier prompt (rag_agent included) ======
INTENT_CLASSIFIER_PROMPT = """
You are a smart AI assistant in a multi-agent system. Classify the latest user message into exactly one of:
- create_contract
- schedule_meeting
- create_or_report_task
- analyze_data
- query_database
- rag_agent
- general_query

Rules:
- If the message continues the current task, return the same agent name.
- Return only the label.

Latest user message:
""".strip()

# ====== Sticky routing: breakers, explicit switching, & completion detection ======
BREAK_WORDS = {
    "cancel", "stop", "exit", "new", "new topic", "switch", "change agent",
    # greetings/small-talk should end sticky mode
    "hi", "hello", "hey", "thanks", "thank you", "ok", "okay"
}

# aliases for quick switching
AGENT_ALIASES = {
    "contract": "create_contract",
    "contracts": "create_contract",
    "meeting": "schedule_meeting",
    "task": "create_or_report_task",
    "tasks": "create_or_report_task",
    "data": "analyze_data",
    "sql": "query_database",
    "rag": "rag_agent",
    "files": "rag_agent",
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

# ====== RESET commands for create_contract ======
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
    """
    Force-clear all known caches/states for the create_contract agent.
    """
    try:
        from src.smart_graph.tools import contract_tool
        from src.smart_graph.agents import Create_Contract
        # Clear tool-level caches
        try:
            contract_tool.SESSION_CACHE["default_user"] = {}
        except Exception:
            pass
        try:
            contract_tool.PLACEHOLDER_CACHE.clear()
        except Exception:
            pass
        # Clear agent-level session if used
        try:
            Create_Contract.CONTRACT_SESSION["default_user"] = {}
        except Exception:
            pass
        print("[INFO] create_contract session reset.")
    except Exception as e:
        print(f"[WARNING] create_contract session reset failed: {e}")

def _resolve_agent_name(name: str) -> Optional[str]:
    n = name.strip().lower()
    if n in TOOLS_ORDER:
        return n
    n = n.lstrip("@")
    if n in AGENT_ALIASES:
        return AGENT_ALIASES[n]
    return None

def _mentions_other_agent(text: str, current_agent: Optional[str]) -> Optional[str]:
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

    # 3) plain tool name mention
    lt = t.lower()
    for name in TOOLS_ORDER:
        if name != current_agent and name in lt:
            return name

    # 4) alias mention
    for alias, canonical in AGENT_ALIASES.items():
        if canonical != current_agent and re.search(rf"\b{alias}\b", lt):
            return canonical

    return None

# tool “done” detector: auto-unstick after completion responses
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

# --- Greetings / small-talk detector ---
GREETING_REGEX = re.compile(
    r"^\s*(hi|hello|hey|yo|morning|good\s+morning|evening|good\s+evening|"
    r"thanks|thank\s+you|thx|ok|okay|cool|great|nice)\s*[!.]*\s*$",
    re.I
)
def _is_greeting(text: str) -> bool:
    return bool(GREETING_REGEX.match(text or ""))

# ====== Core agent logic ======
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

    # ===== Small-talk / greeting short-circuit =====
    if _is_greeting(user_input):
        # clear stickiness and reply normally without routing/classifier
        return {
            "messages": state["messages"] + [AIMessage(content="Hi! How can I help you next?")],
            "current_agent": None,
        }

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

    # ===== RESET command (explicit) =====
    if _wants_reset_contract(user_input):
        _reset_create_contract_session()
        return {
            "messages": messages + [
                AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "create_contract",
                        "args": {"input": "single"},  # or empty; tool will ask to choose mode
                        "id": "tool_call_create_contract",
                    }],
                )
            ],
            "current_agent": "create_contract",
        }

    # ===== Order of decision =====
    # 1) Break words clear stickiness
    if any(w in user_input.lower() for w in BREAK_WORDS):
        current_agent = None

    # 2) Explicit switch wins immediately
    named = _mentions_other_agent(user_input, current_agent)
    if named:
        # If switching into create_contract, reset BEFORE routing
        if named == "create_contract":
            _reset_create_contract_session()

        current_agent = named
        return {
            "messages": messages + [
                AIMessage(
                    content="",
                    tool_calls=[{
                        "name": current_agent,
                        "args": {"input": user_input},
                        "id": f"tool_call_{current_agent}",
                    }],
                )
            ],
            "current_agent": current_agent,
        }

    # 3) Sticky routing (continue with current agent)
    if current_agent in TOOLS_NAMES if (TOOLS_NAMES := set(TOOLS_ORDER)) else TOOLS_ORDER:
        return {
            "messages": messages + [
                AIMessage(
                    content="",
                    tool_calls=[{
                        "name": current_agent,
                        "args": {"input": user_input},
                        "id": f"tool_call_{current_agent}",
                    }],
                )
            ],
            "current_agent": current_agent,
        }

    # 4) RAG fast-path (heuristic)
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

    # 5) Standard intent classification (with rag_agent allowed)
    try:
        intent = model.generate_content(
            f"{INTENT_CLASSIFIER_PROMPT}\n{user_input}",
            generation_config=gen_config,
        ).text.strip().lower()
    except Exception:
        intent = "general_query"

    # 5a) SAFEGUARD: only allow rag_agent if message looks like RAG
    if intent == "rag_agent" and not (
        user_input.lower().startswith("ingest:") or _looks_like_file_qa(user_input)
    ):
        intent = "general_query"

    # 5b) If classifier says general but we had a previous agent (and no switch), keep it
    if (
        intent == "general_query"
        and state.get("current_agent") in TOOLS_MAP  # safe check
        and not named
        and not _is_greeting(user_input)
        and len(user_input.split()) > 2
    ):
        intent = state["current_agent"]

    # 6) Route to tool if matched
    if intent in TOOLS_MAP:
        # Reset on switch into create_contract (classifier path)
        if intent == "create_contract" and state.get("current_agent") != "create_contract":
            _reset_create_contract_session()

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

    # 7) General reply path with trimmed history
    try:
        gemini_contents = messages_to_gemini_contents(trimmed_history)
        reply = model.generate_content(gemini_contents, generation_config=gen_config).text.strip()
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

# ===== Respond with tool output (auto-unstick on “done”) =====
def respond_with_tool(state: AgentState) -> AgentState:
    tool_messages = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    if tool_messages:
        last_tool = tool_messages[-1]
        content = last_tool.content

        if _tool_output_is_done(content):
            # unstick after successful completion
            return {
                "messages": state["messages"] + [AIMessage(content=content)],
                "current_agent": None,
            }

        # keep sticky otherwise
        return {
            "messages": state["messages"] + [AIMessage(content=content)],
            "current_agent": state.get("current_agent"),
        }

    return {
        "messages": state["messages"] + [AIMessage(content="⚠️ Tool returned nothing.")],
        "current_agent": state.get("current_agent"),
    }

# ===== Control flow edges =====
def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tool"
    return "end"

# ===== Build & Compile graph =====
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
