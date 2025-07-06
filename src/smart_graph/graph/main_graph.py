import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage, ToolMessage
from src.smart_graph.utils.state import AgentState
from src.smart_graph.tools.contract_tool import create_contract
from src.smart_graph.tools.meeting_tool import schedule_meeting

from langgraph.checkpoint.memory import InMemorySaver

# Initialize the saver
memory_saver = InMemorySaver()

# Load .env for Gemini key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.0-flash")

# Register tool functions
tools = [create_contract, schedule_meeting]
tool_node = ToolNode(tools=tools)


# --- INTENT DETECTION FUNCTION ---
def call_model(state: AgentState) -> AgentState:
    messages = state["messages"]
    user_input = messages[-1].content.strip()

    # --- Gemini Prompt for intent classification ---
    intent_prompt = f"""
    Classify this user request into ONE of the following intents:
    - schedule_meeting
    - create_contract
    - general_query

    If the input looks like a JSON or includes "contract" or "template", assume create_contract.
    If it includes scheduling, dates, times, or "meeting", assume schedule_meeting.

    Now classify ONLY the **latest user message** into ONE of these intents:
    - schedule_meeting
    - create_contract
    - general_query

    Instructions:
    - If the message includes JSON or contract-related terms (contract, template, mode, answers), classify as `create_contract`.
    - If the user recently mentioned contract and now says "single", "multiple", or gives short answers — it’s still `create_contract`.
    - If the message includes scheduling, time, participants, or dates — it's `schedule_meeting`.
    - Otherwise, classify as `general_query`.

    Only return one word (no explanation): `create_contract`, `schedule_meeting`, or `general_query`.
    User message: "{user_input}"
    """

    try:
        intent = model.generate_content(intent_prompt).text.strip().lower()
    except Exception:
        return {"messages": messages + [
            AIMessage(content="❌ Sorry, I couldn’t understand your request due to a model error. Please try again.")]}

    if "schedule_meeting" in intent:
        return {
            "messages": messages + [
                AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "schedule_meeting",
                        "args": {"input": user_input},
                        "id": "tool_call_schedule_meeting"
                    }]
                )
            ]
        }

    elif "create_contract" in intent:  
        return {
            "messages": messages + [
                AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "create_contract",
                        "args": {"input": user_input},
                        "id": "tool_call_create_contract"
                    }]
                )
            ]
        }

    # Default to chat completion
    try:
        response = model.generate_content(user_input)
        return {"messages": messages + [AIMessage(content=response.text)]}
    except Exception:
        return {"messages": messages + [AIMessage(content="⚠️ Something went wrong while generating a response.")]}


# --- CONDITION TO TRIGGER TOOL EXECUTION ---
def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "continue"
    return "end"


# --- TOOL RESPONSE HANDLER ---
def respond_with_tool_output(state: AgentState) -> AgentState:
    messages = state["messages"]
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]

    if tool_messages:
        last_output = tool_messages[-1].content
        return {"messages": messages + [AIMessage(content=last_output)]}

    return {"messages": messages + [AIMessage(content="⚠️ Tool didn’t return anything.")]}

# --- Step 4: Graph definition ---
workflow = StateGraph(AgentState)  # ✅ تم إصلاح الخطأ هنا بتمرير AgentState
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_node("respond", respond_with_tool_output)

workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {
    "continue": "tools",
    "end": END
})
workflow.add_edge("tools", "respond")
workflow.add_edge("respond", END)

# --- COMPILE APP ---
app = workflow.compile(checkpointer=memory_saver)
