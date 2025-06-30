import gradio as gr
import os
import shutil
from src.smart_graph.graph.main_graph import app
from langchain_core.messages import HumanMessage, AIMessage

# Session state
chat_history = []

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def reset():
    chat_history.clear()
    # Clean old uploaded files
    for f in os.listdir(UPLOAD_DIR):
        if f.endswith(".docx"):
            os.remove(os.path.join(UPLOAD_DIR, f))
    return [], "", None


def handle_chat(user_input, uploaded_file):
    user_input = user_input.strip()
    display = []

    # Handle file upload
    if uploaded_file:
        filename = os.path.basename(uploaded_file.name)
        saved_path = os.path.join(UPLOAD_DIR, filename)
        shutil.copy(uploaded_file.name, saved_path)
        chat_history.append((f"📁 Uploaded: {filename}", "bot"))
        display.append(["🤖", f"📁 File saved as: `{saved_path}`"])
        return display + [[None, user_input]], "", None

    # Send user message to LangGraph
    chat_history.append((user_input, "user"))
    try:
        result = app.invoke({"messages": [HumanMessage(content=user_input)]})
        for msg in result["messages"]:
            if isinstance(msg, AIMessage):
                chat_history.append((msg.content, "bot"))
    except Exception as e:
        chat_history.append((f"❌ Error: {str(e)}", "bot"))

    # Format for Gradio
    return [[f"🤖 {m}" if role == "bot" else None, f"👤 {m}" if role == "user" else None] for m, role in chat_history], "", None

# Build Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🤖 Smart Contract & Meeting Agent")

    chatbot = gr.Chatbot(label="Chat", height=600)
    msg = gr.Textbox(label="Type your message here")
    file_upload = gr.File(label="Upload DOCX Template", file_types=[".docx"])
    clear_btn = gr.Button("🔄 Reset")

    msg.submit(handle_chat, [msg, file_upload], [chatbot, msg, file_upload])
    clear_btn.click(reset, outputs=[chatbot, msg, file_upload])

if __name__ == "__main__":
    demo.launch()
