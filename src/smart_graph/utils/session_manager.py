import os
from langchain_core.messages import AIMessage
from src.smart_graph.agents.Create_Contract import ContractGenerator

class SessionManager:
    def __init__(self):
        self.reset_session()

    def reset_session(self):
        self.session = {
            "chat": [],
            "contract_active": False,
            "contract_gen": None,
            "placeholders": [],
            "answers": {},
            "question_index": 0,
            "docx_path": "",
            "pdf_path": "",
            "mode": "single",
            "uploaded_file": None
        }

    def process_contract_step(self, user_input):
        """ Handles the structured contract Q&A process. """
        try:
            idx = int(self.session["question_index"])
        except Exception:
            self.session["contract_active"] = False
            return AIMessage(content="❌ Internal error. Try restarting the session.")

        if idx >= len(self.session["placeholders"]):
            self.session["contract_active"] = False
            return AIMessage(content="❌ Placeholder index error.")

        field = self.session["placeholders"][idx]
        self.session["answers"][field] = user_input
        self.session["question_index"] += 1

        if self.session["question_index"] >= len(self.session["placeholders"]):
            docx = self.session["contract_gen"].fill_placeholders(self.session["answers"])
            pdf = self.session["contract_gen"].export_responses_pdf()
            self.session["docx_path"] = docx
            self.session["pdf_path"] = pdf
            self.session["contract_active"] = False
            return AIMessage(content="✅ Contract complete! Download below.")

        next_field = self.session["placeholders"][self.session["question_index"]]
        question = self.session["contract_gen"].generate_questions().get(next_field, f"Provide `{next_field}`:")
        return AIMessage(content=question)

    def start_contract(self, template_path):
        """ Initializes contract processing from the detected intent. """
        try:
            self.session["contract_active"] = True
            self.session["contract_gen"] = ContractGenerator(template_path=template_path)
            self.session["placeholders"] = self.session["contract_gen"].extract_placeholders()

            if not self.session["placeholders"]:
                self.session["contract_active"] = False
                return AIMessage(content="❌ No placeholders found in the template file.")

            field = self.session["placeholders"][0]
            question = self.session["contract_gen"].generate_questions().get(field, f"Provide `{field}`:")
            return AIMessage(content=question)

        except Exception as e:
            return AIMessage(content=f"❌ Failed to load template: {e}")