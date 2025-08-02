# main_graph.py

import os
import google.generativeai as genai
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from typing import Literal

# Import tools
from src.smart_graph.tools.contract_tool import create_contract
from src.smart_graph.tools.meeting_tool import schedule_meeting
from src.smart_graph.tools.task_tool import create_or_report_task
from src.smart_graph.tools.data_tool import analyze_data
from src.smart_graph.tools.sql_tool import query_database  

from src.smart_graph.utils.state import AgentState
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()

# --- Step 1: Load environment and configure Gemini ---
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash")

# --- Step 2: Register tools ---
tools = [
    create_contract,
    schedule_meeting,
    create_or_report_task,
    analyze_data,
    query_database  
]
tool_node = ToolNode(tools=tools)

# --- Step 3: Agent logic with smart switching ---
def agent_logic(state: AgentState) -> AgentState:
    messages = state["messages"]
    user_input = messages[-1].content.strip()
    current_agent = state.get("current_agent")

    # Detect intent from message
    context = "\n".join(
        f"{m.type.upper()}: {m.content.strip()}"
        for m in messages[-5:] if hasattr(m, "content")
    )

    intent_prompt = f"""
You are a smart AI assistant in a multi-agent system. Classify the **latest user message** into one of:
- create_contract
- schedule_meeting
- create_or_report_task
- analyze_data
- query_database
- general_query

If the message continues the current task, return the same agent name.
If the message starts a new topic, return the new agent name.

Context:
{context}
""".strip()

    try:
        intent = model.generate_content(intent_prompt).text.strip().lower()
        print(f"[DEBUG] Detected intent: {intent}")
    except Exception:
        return {
            "messages": messages + [AIMessage(content="❌ Failed to detect intent.")],
            "current_agent": None
        }
    if intent in [
        "create_contract",
        "schedule_meeting",
        "create_or_report_task",
        "analyze_data",
        "query_database"
    ]:
       
       # Reset contract session if switching back to contract agent
        if intent == "create_contract" and current_agent != "create_contract":
            try:
                from src.smart_graph.tools import contract_tool
                from src.smart_graph.agents import Create_Contract

                contract_tool.SESSION_CACHE["default_user"] = {}
                Create_Contract.CONTRACT_SESSION["default_user"] = {}
                print("[INFO] Reset contract session due to agent switch.")
            except Exception as e:
                print(f"[WARNING] Failed to reset contract session: {e}")

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


    # Otherwise: reply directly
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
            "current_agent": None
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
