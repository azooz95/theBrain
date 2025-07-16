import os
import google.generativeai as genai
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from typing import Literal
from src.smart_graph.tools.contract_tool import create_contract
from src.smart_graph.tools.meeting_tool import schedule_meeting
from src.smart_graph.tools.task_tool import create_or_report_task
from src.smart_graph.utils.state import AgentState  

from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()

# --- Step 1: Load environment and configure Gemini ---
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash")

# --- Step 2: Register tools ---
tools = [create_contract, schedule_meeting, create_or_report_task]
tool_node = ToolNode(tools=tools)

# --- Step 3: Agent logic with multi-turn memory ---
def agent_logic(state: AgentState) -> AgentState:
    messages = state["messages"]
    user_input = messages[-1].content.strip()
    current_agent = state.get("current_agent")

    if current_agent:
        # 🧠 If an agent is active, continue conversation with it
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

    # Otherwise: detect user intent
    context = "\n".join(
        f"{m.type.upper()}: {m.content.strip()}"
        for m in messages[-5:] if hasattr(m, "content")
    )

    intent_prompt = f"""
You are a smart AI assistant in a multi-agent system. Classify the **latest user message** into one of:
- create_contract
- schedule_meeting
- create_or_report_task
- general_query

Context:
{context}
"""

    try:
        intent = model.generate_content(intent_prompt).text.strip().lower()
        print(f"[DEBUG] Detected intent: {intent}")
    except Exception:
        return {
            "messages": messages + [AIMessage(content="❌ Failed to detect intent.")],
            "current_agent": None
        }

    if intent in ["create_contract", "schedule_meeting", "create_or_report_task"]:
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

    # Otherwise, fallback to generic reply
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

# --- Step 4: Respond with tool output ---
def respond_with_tool(state: AgentState) -> AgentState:
    tool_messages = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    if tool_messages:
        return {
            "messages": state["messages"] + [AIMessage(content=tool_messages[-1].content)],
            "current_agent": None  # ✅ Reset agent after tool completes
        }
    return {
        "messages": state["messages"] + [AIMessage(content="⚠️ Tool returned nothing.")],
        "current_agent": None
    }

# --- Step 5: Should we continue? ---
def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tool"
    return "end"

# --- Step 6: Build the graph ---
agent_builder = StateGraph(AgentState)
agent_builder.add_node("llm_call", agent_logic)
agent_builder.add_node("tool", tool_node)
agent_builder.add_node("respond", respond_with_tool)

agent_builder.set_entry_point("llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue, {
    "tool": "tool",
    "end": END
})
agent_builder.add_edge("tool", "respond")
agent_builder.add_edge("respond", END)

# --- Step 7: Compile the graph ---
agent = agent_builder.compile(checkpointer=checkpointer)
app = agent
