from langchain_core.messages import HumanMessage, AIMessage
from graph.main_graph import app

def run_assistant():
    print("👋 Hello! I'm your smart assistant. Ask me anything (type 'exit' to quit).")

    while True:
        user_input = input("\n🗣️ You: ").strip()
        if user_input.lower() == "exit":
            print("\n👋 Goodbye!")
            break

        inputs = {
            "messages": [HumanMessage(content=user_input)],
            "memory": []
        }
        for output in app.stream(inputs, config={"configurable": {"thread_id": "user-123"}}):
            for _, val in output.items():
                if isinstance(val, dict) and "messages" in val:
                    messages = val["messages"]
                    if isinstance(messages, list) and messages:
                        print(f"\n🤖 {messages[-1].content}\n")

if __name__ == "__main__":
    run_assistant()