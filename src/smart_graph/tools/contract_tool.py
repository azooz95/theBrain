import os
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from src.smart_graph.agents.testContract import ContractGenerator

# In-memory session and field caches
PLACEHOLDER_CACHE = {}
SESSION_CACHE = {}

def infer_mode_from_text(text: str) -> str:
    """
    Try to infer contract mode (single or multiple) from user message.
    """
    text = text.lower()
    single_keywords = ["one", "single", "once", "just one", "only one"]
    multiple_keywords = ["many", "multiple", "batch", "several", "more than one"]

    for word in single_keywords:
        if word in text:
            return "single"
    for word in multiple_keywords:
        if word in text:
            return "multiple"
    return ""

@tool
def create_contract(input: str = "") -> ToolMessage:
    """
    Smart contract creation agent. Interactively handles single or multiple contract generation.
    """
    user_id = "default_user"  # You can make dynamic later
    session = SESSION_CACHE.get(user_id, {})

    # STEP 1 – Decide mode
    if "mode" not in session:
        guessed_mode = infer_mode_from_text(input)
        if guessed_mode:
            session["mode"] = guessed_mode
            SESSION_CACHE[user_id] = session
        else:
            return ToolMessage(
                content=(
                    "🛠️ Would you like to create a **single** contract or **multiple** contracts?\n"
                    "Say something like: `just one`, `batch`, `multiple`, `only one`, etc."
                ),
                name="create_contract",
                tool_call_id="tool_call_create_contract"
            )

    # STEP 2 – Check template availability
    if "template_path" not in session:
        try:
            uploads_dir = os.path.abspath("uploads")
            os.makedirs(uploads_dir, exist_ok=True)

            upload_files = sorted(
                [f for f in os.listdir(uploads_dir) if f.endswith(".docx")],
                key=lambda x: os.path.getctime(os.path.join(uploads_dir, x)),
                reverse=True
            )

            if upload_files:
                template_path = os.path.join(uploads_dir, upload_files[0])
                session["template_path"] = template_path
                SESSION_CACHE[user_id] = session
            else:
                return ToolMessage(
                    content="📂 No `.docx` template found. Please upload a DOCX file to the `uploads/` folder and try again.",
                    name="create_contract",
                    tool_call_id="tool_call_create_contract"
                )
        except Exception as e:
            return ToolMessage(
                content=f"❌ Error checking uploaded files: {str(e)}",
                name="create_contract",
                tool_call_id="tool_call_create_contract"
            )

    # STEP 3 – Proceed based on mode
    mode = session["mode"]
    template_path = session["template_path"]
    generator = ContractGenerator(template_path=template_path)

    # ---- SINGLE MODE ----
    if mode == "single":
        if not session.get("answers_collected"):
            if "awaiting_fields" in session and input:
                try:
                    values = [v.strip() for v in input.split(",")]
                    fields = session["awaiting_fields"]

                    if len(values) != len(fields):
                        raise ValueError(f"Expected {len(fields)} answers, but got {len(values)}.")

                    data = dict(zip(fields, values))
                    session["answers_collected"] = True
                    session["answers"] = data
                    SESSION_CACHE[user_id] = session

                    output_path = generator.fill_placeholders(data)
                    return ToolMessage(
                        content=f"✅ Contract generated successfully!\n\n📄 Output file: `{output_path}`",
                        name="create_contract",
                        tool_call_id="tool_call_create_contract"
                    )

                except Exception as e:
                    return ToolMessage(
                        content=f"⚠️ Error processing your answers: {str(e)}\n\nPlease re-enter your answers in correct format.",
                        name="create_contract",
                        tool_call_id="tool_call_create_contract"
                    )

            else:
                placeholders = generator.extract_placeholders()
                questions = generator.generate_questions()
                fields = list(questions.keys())
                PLACEHOLDER_CACHE[template_path] = fields
                session["awaiting_fields"] = fields
                SESSION_CACHE[user_id] = session

                question_list = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions.values())])
                prompt = (
                    "📝 Please answer the following questions in **one message**, separated by commas:\n\n"
                    f"{question_list}\n\n"
                    "📌 Example: `2025-06-24, John Doe, Jane Smith, Business Plan, 2 years`\n"
                    "⏳ Waiting for your response..."
                )

                return ToolMessage(
                    content=prompt,
                    name="create_contract",
                    tool_call_id="tool_call_create_contract"
                )

        return ToolMessage(
            content="⚠️ You already submitted your answers. Restart the process if you'd like to change them.",
            name="create_contract",
            tool_call_id="tool_call_create_contract"
        )

    # ---- MULTIPLE MODE ----
    elif mode == "multiple":
        uploads_dir = os.path.abspath("uploads")

        if not session.get("excel_template_generated"):
            generator.extract_placeholders()
            excel_template = generator.prepare_excel_template()
            session["excel_template_generated"] = True
            SESSION_CACHE[user_id] = session

            return ToolMessage(
                content=(
                    f"📊 Please fill out this Excel template for batch contract generation:\n"
                    f"`{excel_template}`\n\n"
                    "Once you're done, place the filled Excel file in the `uploads/` folder and type anything to proceed."
                ),
                name="create_contract",
                tool_call_id="tool_call_create_contract"
            )

        # Step 2 – Detect latest filled Excel (excluding templates) and generate
        try:
            filled_files = sorted(
                [f for f in os.listdir(uploads_dir)
                 if f.endswith(".xlsx") and not f.endswith("_template.xlsx")],
                key=lambda x: os.path.getctime(os.path.join(uploads_dir, x)),
                reverse=True
            )

            if not filled_files:
                return ToolMessage(
                    content="⚠️ No filled Excel file found in `uploads/`. Please upload the completed template first.",
                    name="create_contract",
                    tool_call_id="tool_call_create_contract"
                )

            filled_excel_path = os.path.join(uploads_dir, filled_files[0])
            output_paths = generator.generate_bulk_contracts(filled_excel_path)

            return ToolMessage(
                content=(
                    f"✅ Contracts generated from Excel file `{filled_files[0]}`:\n\n" +
                    "\n".join([f"📄 {path}" for path in output_paths])
                ),
                name="create_contract",
                tool_call_id="tool_call_create_contract"
            )

        except Exception as e:
            return ToolMessage(
                content=f"❌ Failed to process Excel file: {str(e)}",
                name="create_contract",
                tool_call_id="tool_call_create_contract"
            )

    return ToolMessage(
        content="❌ Something went wrong in the contract creation flow. Please restart.",
        name="create_contract",
        tool_call_id="tool_call_create_contract"
    )
