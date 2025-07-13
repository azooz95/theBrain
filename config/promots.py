
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import JsonOutputParser

FORMAT_PROMOT = lambda veriables, promot_template: promot_template.format(**veriables)

class GraphPrompts:

    @staticmethod
    def get_intent_classification_prompt_template(input: str, classification: list, format: JsonOutputParser) -> str:

        promot_template = PromptTemplate.from_template("""
            Classify this user request into ONE of the following intents:
            {classification}

            If the input looks like a JSON or includes "contract" or "template", assume create_contract.
            If it includes scheduling, dates, times, or "meeting", assume schedule_meeting.

            Input: "{input}"
            Format: "{fomrat}:
            """
        )
        return FORMAT_PROMOT({"input": input, "classification":classification}, promot_template)

# 
class RunPromots: 
    def __init__(self, outputs=None):

        pass
    @staticmethod
    def run(llm, promote_template, inputs) -> str:
        chain = (
            RunnableLambda(promote_template)
            | llm
            | JsonOutputParser()
        )

        return chain.invoke(**inputs)
    

if __name__ == "__main__":
    import os
    import google.generativeai as genai
    from dotenv import load_dotenv

    from langchain_google_genai import ChatGoogleGenerativeAI

    load_dotenv()
    llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            # other params...
)
    inputs = {
        "input": "Schedule a meeting with John and Jane tomorrow at 3 PM for 1 hour",
        "classification": ["schedule_meeting", "create_contract", "general_query"]
    }
    print(RunPromots.run(llm, GraphPrompts.get_intent_classification_prompt_template, inputs))
    

