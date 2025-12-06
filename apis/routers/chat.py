import os
import shutil
import re
from uuid import uuid4

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    UploadFile,
    File,
    Form,
    WebSocket,
    WebSocketDisconnect,
    FastAPI,
)
from fastapi.responses import FileResponse
from langchain_core.messages import HumanMessage
from starlette.websockets import WebSocketState

from apis.token_generator import get_current_user, verify_token
# from src.smart_graph.graph.main_graph import app as graph_app
from src.smart_graph.graph.agentic_graph import Tools, AgenticGraph, TokensTracker
# from db.sql import utils

app = FastAPI()
router = APIRouter()

# Uploads directory
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

USER_THREADS = {}
AGENTS = {}

TRACKER = TokensTracker()

def get_user_thread(user_id: str) -> str:
    """Retrieve existing thread or create new one."""
    if user_id not in USER_THREADS:
        USER_THREADS[user_id] = str(uuid4())
    return USER_THREADS[user_id]



def _inject_download_link(text: str):
    """
    Finds 'Report generated successfully: <file>.pdf' and replaces <file>.pdf
    with a clickable download link. Also returns a plain download URL.
    """
    if not text:
        return text, None

    # Match something like: Report generated successfully: task_progress_report.pdf
    m = re.search(r"(Report generated successfully:\s*)([A-Za-z0-9_\-\.]+\.pdf)", text)
    if not m:
        return text, None

    prefix = m.group(1)
    filename = os.path.basename(m.group(2))  # prevent path traversal
    download_path = f"/download/{filename}"
    # HTML anchor with download attribute to trigger Save dialog
    anchor = f"<a href='{download_path}' download>{filename}</a>"
    new_text = text.replace(m.group(0), f"{prefix}{anchor}")
    return new_text, download_path

@router.post("/chat")
async def chat_endpoint(
    message: str = Form(...),
    attachment: UploadFile = File(None),
    user_info: str = Depends(get_current_user)
):
    if not message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    file_path = None
    if attachment is not None and attachment.filename:
        file_ext = os.path.splitext(attachment.filename)[1]
        saved_filename = f"{uuid4().hex}{file_ext}"
        # file_path = os.path.join(UPLOAD_DIR, saved_filename)

        # with open(file_path, "wb") as f:
        #     shutil.copyfileobj(attachment.file, f)

    user_thread = get_user_thread(user_info["email"])

    tools_instance = Tools(google_token=user_info["email"], 
                           o365_flow=user_info["email"], 
                           o365_token=user_info["email"])
    
    tools_list = tools_instance.get_tools()
    
    # if user_info not in AGENTS:
    if user_info["email"] not in AGENTS:
        agent_graph = AgenticGraph(tools=tools_list, thread_id=user_thread)
        agent = agent_graph.build_agent()
        AGENTS[user_info["email"]] = (agent_graph, agent)
    else: 
        agent_graph, agent = AGENTS[user_info["email"]]
        
    parsing_instance = agent_graph.run(agent=agent, 
                                    config=agent_graph.get_config, 
                                    inputs=message, 
                                    attachment=None,
                                    tracker=TRACKER)
        
    return {
        "response": parsing_instance,
        "user_id": user_info,
        "uploaded_file": file_path,
    }

# REST API endpoint (POST /chat)
# @router.post("/old_chat")
# async def chat_endpoint(
#     message: str = Form(...),
#     attachment: UploadFile = File(None),
#     user_info: str = Depends(get_current_user)
# ):
#     if not message.strip():
#         raise HTTPException(status_code=400, detail="Message cannot be empty")
    
#     saved_filename = None
#     if attachment is not None and attachment.filename:
#         file_ext = os.path.splitext(attachment.filename)[1]
#         saved_filename = f"{uuid4().hex}{file_ext}"
#         file_path = os.path.join(UPLOAD_DIR, saved_filename)

#         with open(file_path, "wb") as f:
#             shutil.copyfileobj(attachment.file, f)

#         print(f"📁 ed file saved to: {file_path}")

#     attachments = {}
#     if attachment is not None:
#         attachments = {
#             'file_name': attachment.filename,
#             "file_path": file_path
#         }

#     user_thread = get_user_thread(user_info["email"])
#     user_input = message + " attached file: " + (f"{attachments}" if attachment else "No attachment")
#     inputs = {"messages": [HumanMessage(content=user_input)]}
#     config = {"configurable": {"thread_id": user_thread}}
#     last_response = None
#     for output in graph_app.stream(inputs, config):
#         for _, val in output.items():
#             if isinstance(val, dict) and "messages" in val:
#                 messages = val["messages"]
#                 if isinstance(messages, list) and messages:
#                     response = messages[-1].content
#                     if response and response != last_response:
#                         last_response = response

#                         # Inject a clickable download link and expose raw URL too
#                         rendered_response, download_url = _inject_download_link(response)

#                         return {
#                             "response": rendered_response or response,
#                             "user_id": user_info,
#                             "uploaded_file": saved_filename,
#                             **({"download_url": download_url} if download_url else {}),
#                         }

#     return {
#         "response": f"Received message: {message}",
#         "user_id": user_info,
#         "uploaded_file": saved_filename,
#     }


# Download endpoint to serve generated PDFs/DOCs
@router.get("/download/{filename}")
async def download_file(filename: str):
    safe_name = os.path.basename(filename)
    # Common search locations; add your report/contract directory if different
    candidates = [
        os.path.join(os.getcwd(), safe_name),
        os.path.join(UPLOAD_DIR, safe_name),
        os.path.join("reports", safe_name),
        os.path.join("/mnt/data", safe_name),
        os.path.join("contracts", safe_name),
        os.path.join("generated", safe_name),
    ]
    for path in candidates:
        if os.path.exists(path):
            # filename= sets Content-Disposition: attachment; filename="..."
            # Let the browser open a Save dialog.
            # Infer media type by extension—default to octet-stream if unknown.
            ext = os.path.splitext(safe_name)[1].lower()
            mt = "application/octet-stream"
            if ext == ".pdf":
                mt = "application/pdf"
            elif ext == ".docx":
                mt = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            elif ext == ".xlsx":
                mt = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

            return FileResponse(path=path, filename=safe_name, media_type=mt)

    raise HTTPException(status_code=404, detail="File not found")


# WebSocket endpoint (/ws/chat) – unchanged logic
@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    token = websocket.headers.get("authorization")

    if not token or not token.startswith("Bearer "):
        await websocket.close(code=1008)
        return

    token_value = token.replace("Bearer ", "")
    try:
        user = verify_token(token_value)
    except HTTPException as e:
        await websocket.send_text(f"Invalid token: {e.detail}")
        await websocket.close(code=1008)
        return
    except Exception as e:
        await websocket.send_text(f"Error verifying token: {str(e)}")
        await websocket.close(code=1008)
        return

    await websocket.accept()
    await websocket.send_text(f"Authenticated as {user}")

    try:
        while True:
            data = await websocket.receive_text()
            if not data.strip():
                await websocket.send_text("Empty message")
                continue

            inputs = {"messages": [HumanMessage(content=data)]}
            last_response = None
            thread_id = {"configurable": {"thread_id": "1"}}

            for output in graph_app.stream(inputs, thread_id):
                for _, val in output.items():
                    if isinstance(val, dict) and "messages" in val:
                        messages = val["messages"]
                        if isinstance(messages, list) and messages:
                            response = messages[-1].content
                            if response and response != last_response:
                                await websocket.send_text(response)
                                last_response = response
    except WebSocketDisconnect:
        print("🔌 WebSocket disconnected")
        await websocket.close(code=1001)
    except Exception as e:
        print(f"⚠️ Error: {e}")
        if websocket.client_state != WebSocketState.DISCONNECTED:
            await websocket.send_text(f"Error: {str(e)}")
            await websocket.close()


app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
