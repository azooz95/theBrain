

from typing import Optional
from src.smart_graph.agents.Smart_Scheduale import MeetingSchedulingAgent



def schedule_meeting(input: str) -> str:
    """
    Stateless entry to schedule a meeting from free text.
    Always returns a short, friendly one-liner.
    """
    try:
        agent = MeetingSchedulingAgent()
        return agent.run(input=input)
    except Exception:
        
        return "Something went wrong—please try again."
