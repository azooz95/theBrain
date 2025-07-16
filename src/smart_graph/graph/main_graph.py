
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import AIMessage, ToolMessage
from src.smart_graph.utils.state import AgentState
import google.generativeai as genai

# --- Load environment and configure Gemini ---
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash")

# --- Register Tools ---
from src.smart_graph.tools.contract_tool import create_contract
from src.smart_graph.tools.meeting_tool import schedule_meeting
from src.smart_graph.tools.task_tool import create_or_report_task
from src.smart_graph.tools.csv_tool import analyze_csv 

tools = [
    create_contract,
    schedule_meeting,
    create_or_report_task,
    analyze_csv #
]

tool_node = ToolNode(tools=tools)
checkpointer = InMemorySaver()

# --- Agent Logic with Intent Detection ---
def agent_logic(state: AgentState) -> AgentState:
    messages = state["messages"]
    user_input = messages[-1].content.strip()
    current_agent = state.get("current_agent")

    if current_agent:
        return {
            "messages": messages + [
                AIMessage(content="", tool_calls=[{
                    "name": current_agent,
                    "args": {"input": user_input},
                    "id": f"tool_call_{current_agent}"
                }])
            ],
            "current_agent": current_agent
        }

    # Intent classification prompt
    context = "\n".join(
        f"{m.type.upper()}: {m.content.strip()}"
        for m in messages[-5:] if hasattr(m, "content")
    )

    intent_prompt = f"""
You are a multi-agent assistant. Classify the latest user message into one of:
- create_contract
- schedule_meeting
- create_or_report_task
- analyze_csv
- general_query

Context:
{context}
""".strip()

    try:
        intent = model.generate_content(intent_prompt).text.strip().lower()
        print(f"[DEBUG] Detected intent: {intent}")
    except Exception:
        return {
            "messages": messages + [AIMessage(content="❌ Intent detection failed.")],
            "current_agent": None
        }

    if intent in ["create_contract", "schedule_meeting", "create_or_report_task", "analyze_csv"]:
        return {
            "messages": messages + [
                AIMessage(content="", tool_calls=[{
                    "name": intent,
                    "args": {"input": user_input},
                    "id": f"tool_call_{intent}"
                }])
            ],
            "current_agent": intent
        }

    # Fallback: free-form Gemini reply
    try:
        reply = model.generate_content(user_input).text.strip()
        return {
            "messages": messages + [AIMessage(content=reply)],
            "current_agent": None
        }
    except:
        return {
            "messages": messages + [AIMessage(content="⚠️ Gemini failed to respond.")],
            "current_agent": None
        }

# --- Tool Response Handling ---
def respond_with_tool(state: AgentState) -> AgentState:
    tool_messages = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    if tool_messages:
        return {
            "messages": state["messages"] + [AIMessage(content=tool_messages[-1].content)],
            "current_agent": None
        }

    # ✅ Fallback: check last AIMessage with tool_calls (i.e., tool returned but no ToolMessage)
    last_ai_tool_call = next(
        (m for m in reversed(state["messages"])
         if isinstance(m, AIMessage) and getattr(m, "tool_calls", None)),
        None
    )
    if last_ai_tool_call:
        return {
            "messages": state["messages"] + [AIMessage(content="⚠️ Tool executed but no ToolMessage was received.")],
            "current_agent": None
        }

    return {
        "messages": state["messages"] + [AIMessage(content="⚠️ No response from tool.")],
        "current_agent": None
    }


# --- Routing Logic ---
def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tool"
    return "end"

# --- Build the LangGraph Agent ---
builder = StateGraph(AgentState)
builder.add_node("llm_call", agent_logic)
builder.add_node("tool", tool_node)
builder.add_node("respond", respond_with_tool)

builder.set_entry_point("llm_call")
builder.add_conditional_edges("llm_call", should_continue, {
    "tool": "tool",
    "end": END
})
builder.add_edge("tool", "respond")
builder.add_edge("respond", END)

# --- Compile and Expose the Agent ---
agent = builder.compile(checkpointer=checkpointer)
app = agent
