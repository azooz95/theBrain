import os
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from src.smart_graph.agents.testContract import ContractGenerator

# Temporary cache for placeholders/questions (in production use a DB or session store)
PLACEHOLDER_CACHE = {}

@tool
def create_contract(
    mode: str = "",
    template_path: str = "",
    answers: dict = None,
    excel_path: str = ""
) -> ToolMessage:
    """
    Create a contract using a DOCX template. Defaults to 'single' mode if not specified.
    """

    # ✅ Default to single mode
    mode = (mode or "single").lower().strip()

    if mode not in ["single", "multiple"]:
        return ToolMessage(
            content="❓ Invalid mode. Please use 'single' or 'multiple'.",
            name="create_contract",
            tool_call_id="tool_call_create_contract"
        )

    # ✅ Auto-pick latest uploaded .docx if path not given
    if not template_path:
        try:
            uploads = sorted(
                [f for f in os.listdir("uploads") if f.endswith(".docx")],
                key=lambda x: os.path.getctime(os.path.join("uploads", x)),
                reverse=True
            )
            if uploads:
                template_path = os.path.join("uploads", uploads[0])
            else:
                raise FileNotFoundError
        except:
            return ToolMessage(
                content="📄 I need a `.docx` contract template to proceed. Please upload it.",
                name="create_contract",
                tool_call_id="tool_call_create_contract"
            )

    if not os.path.exists(template_path):
        return ToolMessage(
            content="❌ The provided `template_path` does not exist. Please double-check the file path.",
            name="create_contract",
            tool_call_id="tool_call_create_contract"
        )

    generator = ContractGenerator(template_path=template_path)

    if mode == "single":
        if not answers:
            placeholders = generator.extract_placeholders()
            questions = generator.generate_questions()

            friendly_questions = list(questions.values())
            ordered_fields = list(questions.keys())

            # Cache fields for next interaction
            PLACEHOLDER_CACHE[template_path] = ordered_fields

            prompt = "📝 Please answer the following questions **in one message**, separated by commas:\n\n"
            for i, q in enumerate(friendly_questions, 1):
                prompt += f"{i}. {q}\n"
            prompt += (
                "\n📌 Example: `2025-06-24, John Doe, Jane Smith, Business Plan, 2 years`\n"
                "⏳ Waiting for your response..."
            )

            return ToolMessage(
                content=prompt,
                name="create_contract",
                tool_call_id="tool_call_create_contract"
            )

        generator.extract_placeholders()
        generator.fill_placeholders(answers)
        generator.export_responses_pdf()

        return ToolMessage(
            content=f"✅ Contract created!\n📄 DOCX: `{generator.docx_output}`\n📑 PDF: `{generator.pdf_output}`",
            name="create_contract",
            tool_call_id="tool_call_create_contract"
        )

    elif mode == "multiple":
        if not excel_path:
            generator.extract_placeholders()
            excel_template = generator.prepare_excel_template()

            return ToolMessage(
                content=(
                    f"📄 Please fill this Excel template with contract data:\n"
                    f"`{excel_template}`\n\n"
                    "Then upload it and send:\n"
                    "```json\n"
                    "{ \"template_path\": \"...\", \"excel_path\": \"...\" }\n"
                    "```"
                ),
                name="create_contract",
                tool_call_id="tool_call_create_contract"
            )

        files = generator.generate_bulk_contracts(excel_path)

        return ToolMessage(
            content=(
                f"✅ All {len(files)} contracts were generated successfully.\n" +
                "\n".join([f"📄 {f}" for f in files])
            ),
            name="create_contract",
            tool_call_id="tool_call_create_contract"
        )