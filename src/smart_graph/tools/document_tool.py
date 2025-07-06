from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from agents.testDocument import DocumentAgent
import json

@tool
def analyze_document(input: dict | str) -> ToolMessage:
    """
    Analyze a document: extract text, classify, summarize, and store.
    """
    print("🛠️ Raw input received by analyze_document:", input)

    # Parse input if it's a JSON string
    if isinstance(input, str):
        if not input.strip():
            raise ValueError("❌ Empty input string. No data provided.")
        try:
            input = json.loads(input)
        except json.JSONDecodeError as e:
            raise ValueError(f"❌ Could not parse input JSON: {e}")

    if not isinstance(input, dict):
        raise ValueError("❌ Input must be a dictionary.")

    file_path = input.get("file_path")
    if not file_path:
        return ToolMessage(
            content="❌ Invalid or missing 'file_path'. Please upload a document first.",
            name="analyze_document",
            tool_call_id="tool_call_analyze_document"
        )

    try:
        agent = DocumentAgent()
        result = agent.run(input)  # should return plain text only
        if not isinstance(result, str):
            result = str(result)
        return ToolMessage(
            content=result,
            name="analyze_document",
            tool_call_id="tool_call_analyze_document"
        )
    except Exception as e:
        return ToolMessage(
            content=f"❌ Failed to analyze the document: {e}",
            name="analyze_document",
            tool_call_id="tool_call_analyze_document"
        )
