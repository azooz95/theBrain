import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
from langgraph.graph import Graph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage, ToolMessage
from utils.state import AgentState
from tools.contract_tool import create_contract
from tools.meeting_tool import schedule_meeting
from tools.document_tool import analyze_document

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.0-flash")

# Register tools
tools = [create_contract, schedule_meeting, analyze_document]
tool_node = ToolNode(tools=tools)

# --- INTENT DETECTION FUNCTION ---
def call_model(state: AgentState) -> AgentState:
    messages = state["messages"]
    user_input = messages[-1].content.strip()

    # If input is JSON (fallback for contract mode)
    if user_input.startswith("{") and user_input.endswith("}"):
        try:
            parsed = json.loads(user_input)
            if "mode" in parsed and "template_path" in parsed:
                return {
                    "messages": messages + [
                        AIMessage(
                            content="",
                            tool_calls=[{
                                "name": "create_contract",
                                "args": {"input": parsed},
                                "id": "tool_call_create_contract"
                            }]
                        )
                    ]
                }
        except Exception:
            pass

    intent_prompt = (
        "Classify this user request into ONE of the following intents:\n"
        "- schedule_meeting\n"
        "- create_contract\n"
        "- analyze_document\n"
        "- general_query\n\n"
        "If the input looks like a JSON or includes 'contract' or 'template', assume create_contract.\n"
        "If it includes scheduling, dates, times, or words like 'meeting', 'calendar', 'book', assume schedule_meeting.\n"
        "If it includes anything related to analyzing, processing, summarizing, scanning, classifying, reading,\n"
        "or extracting information from a file, image, or document — assume analyze_document.\n\n"
        f"Input: \"{user_input}\""
    )

    try:
        intent_response = model.generate_content(intent_prompt)
        intent = intent_response.text.strip().lower() if intent_response.text else ""
    except Exception:
        return {
            "messages": messages + [
                AIMessage(content="❌ Sorry, I couldn’t understand your request due to a model error. Please try again.")
            ]
        }

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

    elif "analyze_document" in intent:
        file_path = state.get("file_path_for_analysis")
        if file_path:
            return {
                "messages": messages + [
                    AIMessage(
                        content="",
                        tool_calls=[{
                            "name": "analyze_document",
                            "args": {"input": {"file_path": file_path}},
                            "id": "tool_call_analyze_document"
                        }]
                    )
                ]
            }
        else:
            return {
                "messages": messages + [
                    AIMessage(content="❌ Invalid or missing 'file_path'. Please upload a document first.")
                ]
            }

    # Fallback to general Gemini response
    try:
        response = model.generate_content(user_input)
        return {
            "messages": messages + [AIMessage(content=response.text or "⚠️ No response generated.")]
        }
    except Exception:
        return {
            "messages": messages + [AIMessage(content="⚠️ Something went wrong while generating a response.")]
        }

# --- CONDITIONAL FLOW CONTROL ---
def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "continue"
    if state.get("file_path_for_analysis") and "analyze" in last.content.lower():
        return "continue"
    return "end"

# --- HANDLE TOOL OUTPUT ---
def respond_with_tool_output(state: AgentState) -> AgentState:
    messages = state["messages"]
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    if tool_messages:
        return {
            "messages": messages + [AIMessage(content=tool_messages[-1].content)]
        }
    return {
        "messages": messages + [AIMessage(content="⚠️ Tool didn’t return anything.")]
    }

# --- BUILD AND COMPILE GRAPH ---
workflow = Graph()
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

app = workflow.compile()
