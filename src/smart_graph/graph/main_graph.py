import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from src.smart_graph.utils.state import AgentState
from src.smart_graph.tools.contract_tool import create_contract
from src.smart_graph.tools.meeting_tool import schedule_meeting

# Load environment and configure Gemini
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash")

# Register tools
tools = [create_contract, schedule_meeting]
tool_node = ToolNode(tools=tools)

# Track recent context
RECENT_CONTEXT = []

# --- Step 1: Intent Detection Agent ---
def call_model(state: AgentState) -> AgentState:
    messages = state["messages"]
    user_input = messages[-1].content.strip()

    # Use last 5 message turns as dialogue history
    dialogue = "\n".join([
        f"{m.type.upper()}: {m.content.strip()}"
        for m in messages[-5:] if hasattr(m, "content") and m.content.strip()
    ])
    recent_context = dialogue.lower()
    RECENT_CONTEXT.append(recent_context)
    RECENT_CONTEXT[:] = RECENT_CONTEXT[-5:]  # Keep it short

    # Extra logic: if recent messages already discussed contracts, override classification
    contract_keywords = ["contract", "template", "docx", "placeholder", "single", "multiple", "once", "batch", "upload"]
    if any(kw in " ".join(RECENT_CONTEXT).lower() for kw in contract_keywords):
        if user_input.lower() in ["single", "just one", "once", "one", "1", "multiple", "many", "batch"]:
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

    # Prompt Gemini to infer intent
    intent_prompt = f"""
You are an AI assistant that helps route user requests to tools in a multi-agent system.

Your job is to classify the **latest user message** based on the ongoing conversation.

The system supports 3 tools:
1. `create_contract` → Used when the user wants to generate a contract (single or multiple), upload a template, fill placeholders, etc.
2. `schedule_meeting` → Used when the user wants to schedule a meeting, specify time, participants, etc.
3. `general_query` → Everything else: general questions, greetings, or small talk.

Below is the recent conversation:

{dialogue}

Now classify the LAST user message ONLY into one of:
- create_contract
- schedule_meeting
- general_query

Respond with just the tool name, nothing else.
"""
    try:
        intent = model.generate_content(intent_prompt).text.strip().lower()
        print(f"[DEBUG] Intent classified as: {intent}")
    except Exception:
        return {"messages": messages + [
            AIMessage(content="❌ Intent detection failed. Please try again.")
        ]}

    # Route based on detected intent
    if intent == "schedule_meeting":
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

    elif intent == "create_contract":
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

    # Fallback to general LLM response
    try:
        response = model.generate_content(user_input)
        return {"messages": messages + [AIMessage(content=response.text)]}
    except Exception:
        return {"messages": messages + [AIMessage(content="⚠️ Something went wrong while generating a response.")]}

# --- Step 2: Decide if tools should run ---
def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "continue"
    return "end"

# --- Step 3: Handle tool output ---
def respond_with_tool_output(state: AgentState) -> AgentState:
    messages = state["messages"]
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]

    if tool_messages:
        last_output = tool_messages[-1].content
        return {"messages": messages + [AIMessage(content=last_output)]}

    return {"messages": messages + [AIMessage(content="⚠️ Tool didn’t return anything.")]}

# --- Step 4: Graph definition ---
workflow = StateGraph(AgentState)
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

# --- Step 5: Compile app ---
app = workflow.compile()