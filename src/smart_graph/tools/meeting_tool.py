from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from agents.testSmartScheduale import MeetingSchedulingAgent

@tool
def schedule_meeting(input: str) -> ToolMessage:
    """
    Schedules a meeting using user input (stateless).
    """
    agent = MeetingSchedulingAgent()
    result = agent.run(input=input)
    return ToolMessage(
        content=result,
        name="schedule_meeting",
        tool_call_id="tool_call_schedule_meeting"
    )
