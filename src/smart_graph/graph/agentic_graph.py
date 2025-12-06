from langchain_community.tools.office365.events_search import O365SearchEvents
from langchain_community.tools.office365.messages_search import O365SearchEmails
from langchain_community.tools.office365.send_event import O365SendEvent
from langchain_community.tools.office365.send_message import O365SendMessage
from langchain_community.tools.office365.create_draft_message import (
    O365CreateDraftMessage,
)
from langchain_core.messages.ai import AIMessage

from langchain_core.messages.system import SystemMessage
from langchain_core.messages.tool import ToolMessage
from langchain_community.agent_toolkits.office365.toolkit import O365Toolkit
import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI

from typing_extensions import TypedDict, Any, Optional
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.state import CompiledStateGraph

from O365 import Account

# from IPython.display import Image, display

from typing_extensions import Literal

from langgraph.checkpoint.memory import InMemorySaver

from config.config import file_paths

from src.smart_graph.agents.Smart_Scheduale import GoogleCalendarToolkit, GoogleCreateMeet, O365CreateTeamsMeeting

from src.authentications.microsoft_authenticator import microsoft_run_local_server

# import tool langgraph
memory = InMemorySaver()

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


O365SearchEmails.model_rebuild()
O365SearchEvents.model_rebuild()
O365SendEvent.model_rebuild()
O365SendMessage.model_rebuild()
O365CreateDraftMessage.model_rebuild()
O365CreateTeamsMeeting.model_rebuild()
GoogleCreateMeet.model_rebuild()
O365Toolkit.model_rebuild()

class Tools():

    def __init__(self, google_token: str = 'default', o365_flow: str = 'default', o365_token: str = 'default') -> None:
        self.google_token_path = file_paths.google_token_path + f'/{google_token}.json'
        self.o365_flow_path = file_paths.microsoft_flow_dir + f'/{o365_flow}.json'
        self.o365_token_path = file_paths.microsoft_token_dir + f'/{o365_token}.txt'
        self.microsoft_dir = file_paths.microsoft_token_dir

        self.account : Optional[Account] = microsoft_run_local_server(port=5000, user=o365_token)

        self.tools = []
    
    @property
    def get_existed_tools(self) -> list[Any]:
        return self.tools
    
    @property
    def google_token(self) -> str:
        return self.self.google_token_path
    
    @property
    def o365_token_flow(self) -> str:
        return self.o365_token_path, self.o365_flow_path
    
    @property
    def microsoft_account(self) -> Optional[Account]:
        return self.account
    
    @microsoft_account.setter
    def set_microsoft_account(self, user_email: str): 
        self.account = microsoft_run_local_server(port=5000, user=user_email)
    
    @google_token.setter
    def set_google_token(self, user_email: str):
        renamed_path = file_paths.google_token_path + f'/{user_email}.json'
        if not os.path.exists(renamed_path):
            self.google_token_path = renamed_path

    def append_tool(self, tool: Any) -> None:
        if tool: 
            self.tools.append(tool)

    def get_tools(self) -> list[Any]:
        toolkit = O365Toolkit(account=self.account)
        microsoft_tools = toolkit.get_tools()
        tools = microsoft_tools + [GoogleCalendarToolkit(path_token=self.google_token_path), 
                         GoogleCreateMeet(path_token=self.google_token_path), 
                         O365CreateTeamsMeeting()] + self.tools
        return tools

class AgentState(MessagesState):
    pass 

class TokensTracker():
    
    def __init__(self, model_type: str = 'google_flash_2.0'):
        self.promot_tokens: Optional[int] = None
        self.output_tokens: Optional[int] = None
        self.total_promot_tokens: Optional[int] = 0
        self.total_output_tokens: Optional[int] = 0
        self.input_cost_per_token: float = (0.1 * 3.75) / 1e06
        self.output_cost_per_token: float = (0.4 * 3.75) / 1e06
        self.total_cost: float = 0.0

    def record_tokens(self, event: AIMessage) -> None:
        self.promot_tokens = event.usage_metadata['input_tokens']
        self.output_tokens = event.usage_metadata['output_tokens']

        self.total_promot_tokens += self.promot_tokens
        self.total_output_tokens += self.output_tokens

    def report(self) -> dict[str, Optional[int]]:
        self.input_cost = self.total_promot_tokens * self.input_cost_per_token
        self.output_cost = self.total_output_tokens * self.output_cost_per_token
        self.total_cost = self.input_cost + self.output_cost
        return {
            "total_prompt_tokens": format(self.total_promot_tokens,'.2f'),
            "total_output_tokens": format(self.total_output_tokens,'.2f'),
            "input_cost": format(self.input_cost,'.5f'),
            "output_cost": format(self.output_cost,'.5f'),
            "total_cost": format(self.total_cost,'.5f')
        }


class ParsingAgentState():

    def __init__(self):
        self.promot_tokens: Optional[int] = None
        self.output_tokens: Optional[int] = None
        self.total_promot_tokens: Optional[int] = None
        self.total_output_tokens: Optional[int] = None

    def __call__(self, graph_event: Any) -> str:
    
        msgs = None
        if isinstance(graph_event, dict):
            if "messages" in graph_event:
                msgs = graph_event["messages"]
            elif "values" in graph_event and isinstance(graph_event["values"], dict) and "messages" in graph_event["values"]:
                msgs = graph_event["values"]["messages"]
            elif "value" in graph_event and isinstance(graph_event["value"], dict) and "messages" in graph_event["value"]:
                msgs = graph_event["value"]["messages"]
        
        last = msgs[-1]
        if hasattr(last, "content"):
            content = last.content
        elif isinstance(last, dict) and "content" in last:
            content = last["content"]
        else:
            content = str(last)

        return content

        

class AgenticGraph():
    def __init__(self, tools: list[Any], thread_id: str) -> None:
        self.tools = tools
        self.tools_by_name = {tool.name: tool for tool in tools}
        self.thread_id = thread_id
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.9,
            api_key=GOOGLE_API_KEY,
        )

        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.memory = InMemorySaver()

    @property
    def get_config(self) -> dict:
        return {"configurable": {"thread_id": self.thread_id}}
    
    def llm_call(self, state: AgentState):
        """LLM decides whether to call a tool or not"""

        return {
            "messages": [
                self.llm_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant tasked with performing actions based on a set of inputs." \
                        "check if the input has attachment or file path to use in the tools if needed, if it is none ignore it."
                    )
                ]
                + state["messages"]
            )
        ]
    }

    def tool_node(self, state: dict):
        """Performs the tool call"""

        result = []
        for tool_call in state["messages"][-1].tool_calls:
            tool = self.tools_by_name[tool_call["name"]]
            observation = tool.invoke(tool_call["args"])
            print("tool invoke returned:", type(observation), observation)

            if isinstance(observation, list):
                observation = "\n".join(str(item) for item in observation)

            result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
        return {"messages": result}

    def should_continue(self, state: AgentState) -> Literal["Action", END]:
        """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

        messages = state["messages"]
        last_message = messages[-1]
        # If the LLM makes a tool call, then perform an action
        if last_message.tool_calls:
            return "Action"
        # Otherwise, we stop (reply to the user)
        return END

    def build_agent(self) -> CompiledStateGraph:
        # Build workflow
        agent_builder = StateGraph(AgentState)

        # Add nodes
        agent_builder.add_node("llm_call", self.llm_call)
        agent_builder.add_node("environment", self.tool_node)

        # Add edges to connect nodes
        agent_builder.add_edge(START, "llm_call")
        agent_builder.add_conditional_edges(
            "llm_call",
            self.should_continue,
            {
                # Name returned by should_continue : Name of next node to visit
                "Action": "environment",
                END: END,
            },
        )
        agent_builder.add_edge("environment", "llm_call")

        # Compile the agent
        agent = agent_builder.compile(checkpointer=self.memory)

        return agent


    @staticmethod
    def run(agent: CompiledStateGraph, 
            config: dict[str, str], 
            inputs: str, 
            attachment: Optional[str] = None, 
            tracker: Optional[TokensTracker] = None) -> str: 

        if attachment: 
            inputs = inputs + f" attached file: {attachment}"

        event = agent.invoke(
            {"messages": [{"role": "user", "content": inputs}]},
            config=config,
        )
        if tracker: 
            tracker.record_tokens(event["messages"][-1])

        print("Raw event:", event.keys(), event['messages'][-1].usage_metadata, )
        # print('Input tokens:', event['messag'].prompt_token_count)
        # print('Output tokens:', event.candidates_token_count)

        return ParsingAgentState()(event) 

if __name__ == "__main__":
    tools_instance = Tools(google_token='default', o365_flow='default', o365_token='default')
    tools_list = tools_instance.get_tools()
    agent_graph = AgenticGraph(tools=tools_list, thread_id='1')
    agent = agent_graph.build_agent()
    tracker = TokensTracker()

    while True:
        inputs = input("User: ")
        attachment_path = "None"
        parsing_instance = AgenticGraph.run(agent=agent, 
                                            config=agent_graph.get_config, 
                                            inputs=inputs, 
                                            attachment=attachment_path,
                                            tracker=tracker)
        print("Parsing outputs:", parsing_instance, tracker.report())