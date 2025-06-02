import os
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from src.smart_graph.agents.testContract import ContractGenerator

@tool
def create_contract(input: dict) -> ToolMessage:
    """
    Create a single or multiple contract from uploaded template and structured input.

    Expected `input`:
    {
        "mode": "single" | "multiple",
        "template_path": "path/to/template.docx",
        "answers": {placeholder: answer, ...},  # optional, for 'single'
        "excel_path": "path/to/excel.xlsx"       # optional, for 'multiple'
    }
    """

    mode = input.get("mode", "").lower()
    template_path = input.get("template_path")

    if not mode or mode not in ["single", "multiple"]:
        return ToolMessage(
            content="❓ Please specify whether you want to create a `single` or `multiple` contracts.",
            name="create_contract",
            tool_call_id="tool_call_create_contract"
        )

    if not template_path or not os.path.exists(template_path):
        return ToolMessage(
            content="📂 Please upload a valid `.docx` template and provide its path as `template_path`.",
            name="create_contract",
            tool_call_id="tool_call_create_contract"
        )

    generator = ContractGenerator(template_path=template_path)

    if mode == "single":
        answers = input.get("answers")

        if not answers:
            # Step 1: Extract placeholders and questions
            placeholders = generator.extract_placeholders()
            questions = generator.generate_questions()

            return ToolMessage(
                content="📝 I need your answers to the following fields:\n" +
                        "\n".join([f"- {ph}: {q}" for ph, q in questions.items()]) +
                        "\n\nPlease respond with a JSON like:\n```json\n{\"answers\": {\"PLACEHOLDER1\": \"value1\", ...}}\n```",
                name="create_contract",
                tool_call_id="tool_call_create_contract"
            )

        # Step 2: Fill contract
        generator.extract_placeholders()
        generator.fill_placeholders(answers)
        generator.export_responses_pdf()

        return ToolMessage(
            content=(
                f"✅ Contract created!\n\n"
                f"📄 DOCX: `{generator.docx_output}`\n"
                f"📑 PDF: `{generator.pdf_output}`"
            ),
            name="create_contract",
            tool_call_id="tool_call_create_contract"
        )

    elif mode == "multiple":
        excel_path = input.get("excel_path")

        if not excel_path:
            # Step 1: extract and send Excel template
            generator.extract_placeholders()
            excel_template = generator.prepare_excel_template()

            return ToolMessage(
                content=(
                    f"📄 Please fill this Excel template with contract data:\n"
                    f"`{excel_template}`\n\n"
                    "Then re-upload it and provide the path as `excel_path`."
                ),
                name="create_contract",
                tool_call_id="tool_call_create_contract"
            )

        # Step 2: process Excel and generate contracts
        files = generator.generate_bulk_contracts(excel_path)

        return ToolMessage(
            content=(
                f"✅ All {len(files)} contracts have been generated successfully.\n" +
                "\n".join([f"📄 {f}" for f in files])
            ),
            name="create_contract",
            tool_call_id="tool_call_create_contract"
        )
