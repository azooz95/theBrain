
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from src.smart_graph.agents.SQL_Agent import create_sql_agent,  format_friendly_response


@tool
def query_database(input: str) -> str:
    """Query the database using natural language and return a human-friendly message."""
    try:
        agent = create_sql_agent()
        result = agent.invoke(input)

        answer = result["output"] if isinstance(result, dict) and "output" in result else result

        if isinstance(answer, list):
            if len(answer) > 0 and isinstance(answer[0], tuple):
                if len(answer[0]) == 1:
                    answer = answer[0][0]
                else:
                    answer = ", ".join(str(v) for v in answer[0])
            else:
                answer = str(answer)

        message = format_friendly_response(input, str(answer))
        return message

    except Exception as e:
        return f"❌ An error occurred while processing your query: {e}"