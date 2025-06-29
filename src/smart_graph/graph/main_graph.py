import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
from langgraph.graph import Graph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage, ToolMessage
from src.smart_graph.utils.state import AgentState
from src.smart_graph.tools.contract_tool import create_contract, PLACEHOLDER_CACHE
from src.smart_graph.tools.meeting_tool import schedule_meeting

# Load environment and configure Gemini
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash")

# Register tools
tools = [create_contract, schedule_meeting]
tool_node = ToolNode(tools=tools)

# --- Step 1: Intent Detection Agent ---
def call_model(state: AgentState) -> AgentState:
    messages = state["messages"]
    user_input = messages[-1].content.strip()

    # Attempt to handle comma-separated contract answers directly
    if "," in user_input and len(user_input.split(",")) >= 5:
        try:
            uploads = os.listdir("uploads")
            latest_template = sorted(
                [f for f in uploads if f.endswith(".docx")],
                key=lambda x: os.path.getctime(os.path.join("uploads", x)),
                reverse=True
            )[0]
            template_path = os.path.join("uploads", latest_template)

            if template_path in PLACEHOLDER_CACHE:
                fields = PLACEHOLDER_CACHE[template_path]
            else:
                from src.smart_graph.agents.testContract import ContractGenerator
                generator = ContractGenerator(template_path=template_path)
                fields = generator.extract_placeholders()

            values = [v.strip() for v in user_input.split(",")]

            if len(fields) != len(values):
                return {"messages": messages + [
                    AIMessage(content=f"⚠️ Your response does not match the expected number of fields. Expected {len(fields)}, got {len(values)}.")
                ]}

            parsed_answers = {k: v for k, v in zip(fields, values)}
            parsed = {
                "mode": "single",
                "template_path": template_path,
                "answers": parsed_answers
            }

            print("✅ Auto-parsed answers:", parsed_answers)

            return {
                "messages": messages + [
                    AIMessage(
                        content="",
                        tool_calls=[{
                            "name": "create_contract",
                            "args": parsed,
                            "id": "tool_call_create_contract"
                        }]
                    )
                ]
            }

        except Exception as e:
            print("❌ Parsing error:", str(e))
            return {"messages": messages + [AIMessage(content="❌ Failed to process your contract answers.")]}

    # --- Build last 5 messages context ---
    previous_context = "\n".join([
        f"{m.type.upper()}: {m.content.strip()}" for m in messages[-5:]
        if hasattr(m, "content") and m.content.strip()
    ])

    intent_prompt = f"""
You are a smart intent classifier for a multi-agent assistant.

Below is the recent conversation:
{previous_context}

Now classify ONLY the **latest user message** into ONE of these intents:
- schedule_meeting
- create_contract
- general_query

Instructions:
- If the message includes JSON or contract-related terms (contract, template, mode, answers), classify as `create_contract`.
- If the user recently mentioned contract and now says \"single\", \"multiple\", or gives short answers — it’s still `create_contract`.
- If the message includes scheduling, time, participants, or dates — it's `schedule_meeting`.
- Otherwise, classify as `general_query`.

Only return one word (no explanation): `create_contract`, `schedule_meeting`, or `general_query`.
User message: \"{user_input}\"
"""

    try:
        intent = model.generate_content(intent_prompt).text.strip().lower()
    except Exception:
        return {"messages": messages + [
            AIMessage(content="❌ Sorry, I couldn’t understand your request due to a model error. Please try again.")
        ]}

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
        try:
            parsed = json.loads(user_input)
        except Exception:
            parsed = {"raw_input": user_input}

        return {
            "messages": messages + [
                AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "create_contract",
                        "args": parsed,
                        "id": "tool_call_create_contract"
                    }]
                )
            ]
        }

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

# --- Step 5: Compile app ---
app = workflow.compile()
