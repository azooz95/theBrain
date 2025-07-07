import os
import google.generativeai as genai
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import MessagesState
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# Import all tools (agents)
from src.smart_graph.tools.contract_tool import create_contract
from src.smart_graph.tools.meeting_tool import schedule_meeting
from src.smart_graph.tools.task_reminder_tool import remind_tasks  # ⬅️ NEW tool

# Step 1 – Load environment and configure Gemini
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash")

# Step 2 – Register tools
tools = [create_contract, schedule_meeting, remind_tasks]  # ⬅️ Include the new tool
tool_node = ToolNode(tools=tools)

# Step 3 – Core agent logic (intent detection and routing)
def agent_logic(state: MessagesState) -> MessagesState:
    messages = state["messages"]
    user_input = messages[-1].content.strip()

    # Extract recent conversation context
    context = "\n".join(
        f"{m.type.upper()}: {m.content.strip()}"
        for m in messages[-5:] if hasattr(m, "content")
    )

    # Gemini prompt to detect intent
    intent_prompt = f"""
You are a smart AI assistant in a multi-agent system. Your task is to classify the user's intent
based on the **latest user message** while considering recent context.

The system supports:
1. create_contract → for generating or uploading contracts (single, multiple, placeholders).
2. schedule_meeting → for scheduling meetings (time, date, participants).
3. remind_tasks → for sending reminders for tasks due soon.
4. general_query → greetings, questions, anything else.

Context:
{context}

Classify ONLY the **latest user message** into one of the following:
- create_contract
- schedule_meeting
- remind_tasks
- general_query
"""

    try:
        intent = model.generate_content(intent_prompt).text.strip().lower()
        print(f"[DEBUG] Detected intent: {intent}")
    except Exception:
        return {
            "messages": messages + [AIMessage(content="❌ Failed to detect intent.")]
        }

    # Route based on intent
    if intent == "create_contract":
        return {
            "messages": messages + [
                AIMessage(content="", tool_calls=[{
                    "name": "create_contract",
                    "args": {"input": user_input},
                    "id": "tool_call_contract"
                }])
            ]
        }

    elif intent == "schedule_meeting":
        return {
            "messages": messages + [
                AIMessage(content="", tool_calls=[{
                    "name": "schedule_meeting",
                    "args": {"input": user_input},
                    "id": "tool_call_meeting"
                }])
            ]
        }

    elif intent == "remind_tasks":
        return {
            "messages": messages + [
                AIMessage(content="", tool_calls=[{
                    "name": "remind_tasks",
                    "args": {"input": user_input},
                    "id": "tool_call_reminders"
                }])
            ]
        }

    # Fallback – general Gemini response
    try:
        reply = model.generate_content(user_input).text.strip()
        return {"messages": messages + [AIMessage(content=reply)]}
    except:
        return {"messages": messages + [AIMessage(content="⚠️ Gemini failed to respond.")]}

# Step 4 – Handle tool responses
def respond_with_tool(state: MessagesState) -> MessagesState:
    tool_messages = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    if tool_messages:
        return {
            "messages": state["messages"] + [AIMessage(content=tool_messages[-1].content)]
        }
    return {
        "messages": state["messages"] + [AIMessage(content="⚠️ Tool did not return anything.")]
    }

# Step 5 – Should the graph continue?
def should_continue(state: MessagesState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tool"
    return "end"

# Step 6 – Build the LangGraph
agent_builder = StateGraph(MessagesState)
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

# Step 7 – Compile the graph
agent = agent_builder.compile()
app = agent
