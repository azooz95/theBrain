from typing import TypedDict, Annotated, Sequence, Union, Literal, Optional
import operator
from langchain_core.messages import HumanMessage, AIMessage

class AgentState(TypedDict):
    messages: Annotated[Sequence[Union[HumanMessage, AIMessage]], operator.add]
    current_agent: Optional[Literal["create_contract", "schedule_meeting", "create_or_report_task"]]
