# task_tool.py
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from src.smart_graph.agents.Task_Creation import (
    intent_and_board_agent,
    task_extractor_agent,
    generate_report
)

@tool
def create_or_report_task(input: str) -> ToolMessage:
    """
    Create tasks on Trello or generate progress reports based on user input.
    """
    try:
        intent = intent_and_board_agent(input)

        if intent.user_intention == "create_task" and intent.board:
            result = task_extractor_agent(intent.board, input)
        elif intent.user_intention == "generate_report" and intent.board:
            result = generate_report(intent.board)
        else:
            result = intent.AiResponse

        return ToolMessage(
            content=result,
            name="create_or_report_task",
            tool_call_id="tool_call_create_or_report_task"
        )

    except Exception as e:
        return ToolMessage(
            content=f"❌ Error processing task: {str(e)}",
            name="create_or_report_task",
            tool_call_id="tool_call_create_or_report_task"
        )
