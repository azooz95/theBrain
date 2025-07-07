
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from src.smart_graph.utils.trello import trello
from datetime import datetime, timedelta

@tool
def remind_tasks(input: str = "") -> ToolMessage:
    """
    Sends reminders for tasks that are due within the next 24 hours.
    """
    tr = trello()
    try:
        boards = tr.get_trello_boards()
        board_names = [b["name"] for b in boards]

        reminders = []
        for board in board_names:
            board_id = tr.select_board(board)
            tr.board_id = board_id
            lists = tr.get_trello_lists()
            for lst in lists:
                cards = tr.get_cards(lst['id'])
                for card in cards:
                    due_date = card.get("due")
                    name = card.get("name")
                    members = card.get("idMembers", [])

                    if due_date:
                        due = datetime.strptime(due_date[:10], "%Y-%m-%d")
                        if due <= datetime.now() + timedelta(days=1):
                            assigned_names = [tr.get_member_full_name(m) for m in members]
                            reminders.append(f"🔔 **{name}** (Due: {due.strftime('%Y-%m-%d')}) → Assigned to: {', '.join(assigned_names)}")

        if reminders:
            return ToolMessage(
                content="📌 Upcoming Task Reminders:\n\n" + "\n".join(reminders),
                name="remind_tasks",
                tool_call_id="tool_call_reminders"
            )
        else:
            return ToolMessage(
                content="✅ No tasks due within the next 24 hours.",
                name="remind_tasks",
                tool_call_id="tool_call_reminders"
            )
    except Exception as e:
        return ToolMessage(
            content=f"❌ Error generating reminders: {str(e)}",
            name="remind_tasks",
            tool_call_id="tool_call_reminders"
        )
