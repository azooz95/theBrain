import os
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
import google.generativeai as genai
from dotenv import load_dotenv

# Load API Key for Gemini
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# In-memory session tracking
SESSION_CACHE = {}

@tool
def analyze_data(input: str) -> ToolMessage:
    """
    Interactively analyzes uploaded CSV/XLSX files using Gemini + Pandas.
    The tool supports: listing files, selecting by name or number, answering user questions, and gracefully ending the conversation.
    """
    user_id = "default_user"
    session = SESSION_CACHE.get(user_id, {})
    agent = session.get("agent") or GeminiCSVAgent()

    input_clean = input.strip().lower()

    # Step 0: Friendly exit detection
    if session.get("file_loaded"):
        try:
            intent_prompt = f"""Determine if this message means the user wants to end the conversation:
\"{input_clean}\"
Reply with only \"yes\" or \"no\"."""
            intent = gemini_model.generate_content(intent_prompt).text.strip().lower()
            if intent.startswith("yes"):
                SESSION_CACHE.pop(user_id, None)
                return ToolMessage(
                    content="👋 Conversation ended! Feel free to upload another file later 😊",
                    name="analyze_data",
                    tool_call_id="tool_call_analyze_data"
                )
        except Exception:
            pass  # fallback

    try:
        files = agent.get_available_files()
        if not files:
            return ToolMessage(
                content="⏳ No files found. Please upload a file.",
                name="analyze_data",
                tool_call_id="tool_call_analyze_data"
            )

        # Step 1 & 2: Select file if not loaded yet
        if not session.get("file_loaded"):
            selected_file = None
            if input_clean.isdigit():
                index = int(input_clean) - 1
                if 0 <= index < len(files):
                    selected_file = files[index]
            elif input_clean in [f.lower() for f in files]:
                selected_file = next((f for f in files if f.lower() == input_clean), None)

            if selected_file:
                try:
                    agent.select_file(selected_file)
                    session["file_loaded"] = True
                    session["filename"] = selected_file
                    session["agent"] = agent
                    SESSION_CACHE[user_id] = session
                    return ToolMessage(
                        content=f"📂 `{selected_file}` loaded successfully! Ask me anything about this file. 😊",
                        name="analyze_data",
                        tool_call_id="tool_call_analyze_data"
                    )
                except Exception as e:
                    return ToolMessage(
                        content=f"❌ Failed to load file: {str(e)}",
                        name="analyze_data",
                        tool_call_id="tool_call_analyze_data"
                    )

            # Prompt user to select valid file
            numbered_list = "\n".join(f"{i+1}. {f}" for i, f in enumerate(files))
            session["agent"] = agent
            SESSION_CACHE[user_id] = session
            return ToolMessage(
                content=f"✅ Found data files:\n{numbered_list}\n\n📄 Please tell me the file name or number you'd like to analyze.",
                name="analyze_data",
                tool_call_id="tool_call_analyze_data"
            )

        # Step 3: Answer questions about loaded file
        if session.get("file_loaded"):
            result = agent.ask(input)
            return ToolMessage(
                content=f"🤖 {result}\n\n💬 Any thing else ?.",
                name="analyze_data",
                tool_call_id="tool_call_analyze_data"
            )

    except Exception as e:
        return ToolMessage(
            content=f"❌ Unexpected error: {str(e)}",
            name="analyze_data",
            tool_call_id="tool_call_analyze_data"
        )
