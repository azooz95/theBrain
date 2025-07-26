
import os
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import initialize_agent, AgentType
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()
DEFAULT_DB_PATH = os.getenv("DATABASE_BASE_URL")

if not DEFAULT_DB_PATH:
    raise ValueError("❌ DATABASE_BASE_URL is not set in the .env file.")

def create_sql_agent():
    db = SQLDatabase.from_uri(DEFAULT_DB_PATH, sample_rows_in_table_info=False)

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.0)

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    agent = initialize_agent(
        tools=toolkit.get_tools(),
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
        handle_parsing_errors=True,
    )
    return agent

def format_friendly_response(query: str, answer: str) -> str:
    """Formats a human-friendly message based on the query and result."""
    query_lower = query.lower()

    # Try to infer intent from query
    if "average" in query_lower or "mean" in query_lower:
        return f"📊 The average value you asked about is approximately {round(float(answer), 2)}."
    elif "how many" in query_lower or "count" in query_lower:
        return f"🔢 There are {answer} records matching your request."
    elif "most" in query_lower or "top" in query_lower:
        return f"🏆 The top result is: {answer}"
    else:
        return f"📄 Here’s the answer: {answer}"

def sql_agent_node(state: dict) -> dict:
    query = state.get("user_input")
    if not query:
        return {"error": "Missing user_input in state."}

    agent = create_sql_agent()
    result = agent.invoke(query)
    return {"sql_result": result}

if __name__ == "__main__":
    try:
        print(f"✅ Connected to database at: {DEFAULT_DB_PATH}")
        agent = create_sql_agent()

        while True:
            query = input("\n🔎 Enter your natural language query : ").strip()
            if query.lower() in ("exit", "quit"):
                print("👋 Exiting. Goodbye!")
                break

            try:
                result = agent.invoke(query)

                # Extract output from LangChain's result
                answer = result["output"] if isinstance(result, dict) and "output" in result else result

                # Convert answer (e.g., [(43.9,)] or 43.9) into plain text
                if isinstance(answer, list) and len(answer) > 0 and isinstance(answer[0], tuple):
                    answer = answer[0][0]

                message = format_friendly_response(query, answer)
                print(f"\n{message}")

            except Exception as e:
                print(f"❌ Error running query: {e}")

    except Exception as err:
        print(f"❌ Failed to initialize agent: {err}")
