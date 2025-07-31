import os
import shutil
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
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, ConfigDict
from starlette.websockets import WebSocketState

from apis.token_generator import get_current_user, verify_token
from src.smart_graph.graph.main_graph import app as graph_app

app = FastAPI()
router = APIRouter()

# Uploads directory
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# REST API endpoint (POST /chat)
@router.post("/chat")
async def chat_endpoint(
    message: str = Form(...),
    attachment: UploadFile = File(None),
    user_info: str = Depends(get_current_user)
):
    if not message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    saved_filename = None
    if attachment.filename:
        file_ext = os.path.splitext(attachment.filename)[1]
        saved_filename = f"{uuid4().hex}{file_ext}"
        file_path = os.path.join(UPLOAD_DIR, saved_filename)

        with open(file_path, "wb") as f:
            shutil.copyfileobj(attachment.file, f)

        print(f"📁 Uploaded file saved to: {file_path}")

    user_input = message
    inputs = {"messages": [HumanMessage(content=user_input)]}
    config = {"configurable": {"thread_id": "1"}}

    last_response = None
    for output in graph_app.stream(inputs, config):
        for _, val in output.items():
            if isinstance(val, dict) and "messages" in val:
                messages = val["messages"]
                if isinstance(messages, list) and messages:
                    response = messages[-1].content
                    if response and response != last_response:
                        last_response = response
                        return {
                            "response": response,
                            "user_id": user_info,
                            "uploaded_file": saved_filename,
                        }

    return {
        "response": f"Received message: {message}",
        "user_id": user_info,
        "uploaded_file": saved_filename,
    }


# WebSocket endpoint (/ws/chat)
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
