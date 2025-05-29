from typing import TypedDict, Annotated, Sequence, Union
import operator
from langchain_core.messages import HumanMessage, AIMessage

class AgentState(TypedDict):
    messages: Annotated[Sequence[Union[HumanMessage, AIMessage]], operator.add]