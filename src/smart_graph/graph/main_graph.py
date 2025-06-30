import hashlib
import os
import json
from typing import Dict, List, Optional, TypedDict
from datetime import datetime
import google.generativeai as genai
from dotenv import load_dotenv
from langgraph.graph import Graph
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from smart_graph.agents import testContract, fu
# --- Configuration ---
load_dotenv()

class AgentConfig(BaseModel):
    model_name: str = "gemini-1.5-pro"
    max_history: int = 10
    checkpoint_interval: int = 3  # Save state every N interactions
    enable_memory: bool = True

# --- Memory-Enhanced Agent ---
class AIAgent:
    def __init__(self, tools: List):
        self.config = AgentConfig()
        self.workflow = self._build_workflow(tools)
        self.memory = MemorySaver()
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def process(self, user_input: str, thread_id: Optional[str] = None) -> str:
        """Process input with automatic state checkpointing"""
        if not thread_id:
            thread_id = f"thread_{hashlib.sha256(user_input.encode()).hexdigest()[:8]}"

        # Initialize or load state
        state = self._get_initial_state(thread_id, user_input)

        # Run through workflow
        for step in self.workflow.stream(
            state,
            {"configurable": {"thread_id": thread_id}},
            stream_mode="values"
        ):
            state = step

        # Get final response
        return state["messages"][-1].content

    def _build_workflow(self, tools: List) -> Graph:
        """Construct workflow with memory checkpointing"""
        workflow = Graph()

        # Define nodes
        workflow.add_node("receive_input", self._receive_input)
        workflow.add_node("analyze_intent", self._analyze_intent)
        workflow.add_node("generate_response", self._generate_response)
        workflow.add_node("execute_tools", ToolNode(tools))
        workflow.add_node("format_output", self._format_output)

        # Define edges
        workflow.set_entry_point("receive_input")
        workflow.add_edge("receive_input", "analyze_intent")
        workflow.add_edge("format_output", "receive_input")  # Conversation loop

        # Conditional tool execution
        workflow.add_conditional_edges(
            "analyze_intent",
            self._should_use_tools,
            {
                "use_tools": "execute_tools",
                "direct_response": "generate_response"
            }
        )
        workflow.add_edge("execute_tools", "generate_response")

        # Add memory checkpointing
        workflow.add_node("checkpoint", self._save_checkpoint)
        workflow.add_edge("generate_response", "checkpoint")
        workflow.add_edge("checkpoint", "format_output")

        return workflow.compile(
            checkpointer=MemorySaver(),
            interrupt_before=["execute_tools"],
            interrupt_after=["generate_response"]
        )

    def _get_initial_state(self, thread_id: str, user_input: str) -> Dict:
        """Get or initialize state with memory"""
        # Try to load existing state
        if self.config.enable_memory:
            try:
                state = self.memory.get({"configurable": {"thread_id": thread_id}})
                if state:
                    return self._update_state(state, user_input)
            except Exception:
                pass

        # Initialize new state
        return {
            "messages": [
                SystemMessage(content="You are a helpful AI assistant."),
                HumanMessage(content=user_input)
            ],
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "session_id": self.session_id,
                "interaction_count": 0
            }
        }

    def _update_state(self, state: Dict, new_input: str) -> Dict:
        """Update existing state with new input"""
        return {
            "messages": state["messages"] + [HumanMessage(content=new_input)],
            "metadata": {
                **state["metadata"],
                "updated_at": datetime.now().isoformat(),
                "interaction_count": state["metadata"]["interaction_count"] + 1
            }
        }

    def _save_checkpoint(self, state: Dict) -> Dict:
        """Save state to memory at key points"""
        if self.config.enable_memory:
            # Only checkpoint every N interactions
            if state["metadata"]["interaction_count"] % self.config.checkpoint_interval == 0:
                self.memory.put(state)
        return state

    # --- Processing Methods ---
    def _receive_input(self, state: Dict) -> Dict:
        """Prepare input for processing"""
        return state

    def _analyze_intent(self, state: Dict) -> Dict:
        """Analyze user intent with conversation context"""
        messages = state["messages"]
        user_input = messages[-1].content

        prompt = f"""
        Analyze this conversation context and determine intent:
        
        Conversation History:
        {"".join(m.content for m in messages[:-1])}
        
        Current Input:
        {user_input}
        
        Should we use tools or respond directly?
        """
        
        try:
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            model = genai.GenerativeModel(self.config.model_name)
            response = model.generate_content(prompt)
            
            return {
                **state,
                "intent": json.loads(response.text),
                "messages": messages
            }
        except Exception as e:
            print(f"Intent analysis error: {e}")
            return {
                **state,
                "intent": {"action": "direct_response"},
                "messages": messages
            }

    def _should_use_tools(self, state: Dict) -> str:
        """Determine next step based on intent"""
        return state.get("intent", {}).get("action", "direct_response")

    def _generate_response(self, state: Dict) -> Dict:
        """Generate appropriate response"""
        messages = state["messages"]
        
        if state["intent"]["action"] == "direct_response":
            try:
                response = self._call_llm(messages[-1].content)
                return {
                    **state,
                    "messages": messages + [AIMessage(content=response)]
                }
            except Exception as e:
                print(f"Response generation error: {e}")
                return {
                    **state,
                    "messages": messages + [
                        AIMessage(content="Sorry, I encountered an error processing your request.")
                    ]
                }
        
        # Tool responses are handled by the ToolNode
        return state

    def _format_output(self, state: Dict) -> Dict:
        """Prepare final output"""
        return state

    def _call_llm(self, prompt: str) -> str:
        """Wrapper for LLM calls"""
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel(self.config.model_name)
        response = model.generate_content(prompt)
        return response.text
    

if __name__ == "__main__":
    agent = AIAgent(tools=[create_contract, schedule_meeting])

    # Start new conversation thread
    response1 = agent.process("Book a meeting for tomorrow at 2pm", thread_id="project_x")

    # Later continue the same thread
    response2 = agent.process("What meetings do I have scheduled?", thread_id="project_x")

    # The agent maintains context between interactions