#demo
import gradio as gr
import os
from graph.main_graph import app
from langchain_core.messages import HumanMessage, AIMessage
from agents.testContract import ContractGenerator

# Session state
session = {
    "chat": [],
    "contract_active": False,
    "contract_gen": None,
    "placeholders": [],
    "answers": {},
    "question_index": 0,
    "docx_path": "",
    "pdf_path": "",
    "mode": "single",
    "uploaded_file": None
}

def reset_session():
    session.update({
        "chat": [],
        "contract_active": False,
        "contract_gen": None,
        "placeholders": [],
        "answers": {},
        "question_index": 0,
        "docx_path": "",
        "pdf_path": "",
        "mode": "single",
        "uploaded_file": None
    })
    return [], "", None, None

def postprocess_chat(chat_list):
    return [[None, f"👤 {msg}"] if role == "user" else [f"🤖 {msg}", None] for msg, role in chat_list]

def chat_logic(user_input, uploaded_file):
    user_input = user_input.strip()
    session["chat"].append((user_input, "user"))

    # Use remembered uploaded file if current input is None
    effective_file = uploaded_file or session.get("uploaded_file")

    # Step-by-step contract session
    if session["contract_active"]:
        try:
            idx = int(session["question_index"])
        except Exception:
            session["chat"].append(("❌ Internal error. Try restarting the session.", "bot"))
            session["contract_active"] = False
            return postprocess_chat(session["chat"]), "", None, None

        if idx >= len(session["placeholders"]):
            session["chat"].append(("❌ Placeholder index error.", "bot"))
            session["contract_active"] = False
            return postprocess_chat(session["chat"]), "", None, None

        field = session["placeholders"][idx]
        session["answers"][field] = user_input
        session["question_index"] += 1

        if session["question_index"] >= len(session["placeholders"]):
            docx = session["contract_gen"].fill_placeholders(session["answers"])
            pdf = session["contract_gen"].export_responses_pdf()
            session["docx_path"] = docx
            session["pdf_path"] = pdf
            session["contract_active"] = False
            session["chat"].append(("✅ Contract complete! Download below.", "bot"))
            return postprocess_chat(session["chat"]), "", docx, pdf

        next_field = session["placeholders"][session["question_index"]]
        question = session["contract_gen"].generate_questions().get(next_field, f"Provide `{next_field}`:")
        session["chat"].append((question, "bot"))
        return postprocess_chat(session["chat"]), "", None, None

    # Trigger contract mode
    contract_keywords = ["create contract", "contract from", "generate agreement", "nda", "start contract"]
    if any(kw in user_input.lower() for kw in contract_keywords):
        session["contract_active"] = True
        if effective_file:
            session["uploaded_file"] = effective_file
            try:
                gen = ContractGenerator(template_path=effective_file.name)
                session["contract_gen"] = gen
                session["placeholders"] = gen.extract_placeholders()

                if not session["placeholders"]:
                    session["contract_active"] = False
                    session["chat"].append(("❌ No placeholders found in the DOCX file.", "bot"))
                    return postprocess_chat(session["chat"]), "", None, None

                session["answers"] = {}
                session["question_index"] = 0
                field = session["placeholders"][0]
                question = gen.generate_questions().get(field, f"Provide `{field}`:")
                session["chat"].append((question, "bot"))
            except Exception as e:
                session["chat"].append((f"❌ Failed to load template: {e}", "bot"))
        else:
            session["chat"].append(("📂 Please upload a DOCX file below to begin the contract process.", "bot"))
        return postprocess_chat(session["chat"]), "", None, None

    # Default: forward to LangGraph agent
    try:
        result = app.invoke({"messages": [HumanMessage(content=user_input)]})
        for m in result["messages"]:
            if isinstance(m, AIMessage):
                session["chat"].append((m.content, "bot"))
    except Exception as e:
        session["chat"].append((f"❌ LangGraph error: {str(e)}", "bot"))

    return postprocess_chat(session["chat"]), "", None, None

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🤖 Smart Multi-Agent Assistant — Contract & Meeting")

    chatbot = gr.Chatbot(label="Conversation", height=600)
    msg = gr.Textbox(label="Message")
    file_upload = gr.File(label="Upload DOCX Template", file_types=[".docx"])
    clear = gr.Button("Clear Session")
    docx_file = gr.File(label="Download DOCX", visible=False)
    pdf_file = gr.File(label="Download PDF", visible=False)

    msg.submit(chat_logic, inputs=[msg, file_upload], outputs=[chatbot, msg, docx_file, pdf_file])
    clear.click(reset_session, outputs=[chatbot, msg, docx_file, pdf_file])

if __name__ == "__main__":
    demo.launch()
