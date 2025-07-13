# fast router

from langchain_core.messages import HumanMessage, AIMessage
from fastapi import APIRouter, Depends, HTTPException, FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ConfigDict
from apis.token_generator import get_current_user, verify_token
from starlette.websockets import WebSocketState
from src.smart_graph.graph.main_graph import app as graph_app

app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    model_config = ConfigDict(extra='forbid')

router = APIRouter()

# REST API endpoint
@router.post("/chat")
async def chat_endpoint(request: ChatRequest, user_info: str = Depends(get_current_user)):
    if not request.message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    user_input = request.message
    inputs = {"messages": [HumanMessage(content=user_input)]}

    last_response = None
    for output in graph_app.stream(inputs):
        for _, val in output.items():
            if isinstance(val, dict) and "messages" in val:
                messages = val["messages"]
                if isinstance(messages, list) and messages:
                    response = messages[-1].content
                    if response and response != last_response:
                        last_response = response
                        return {"response": response, "user_id": user_info}

    return {"response": f"Received message: {request.message}, user_id: {user_info}"}

# WebSocket endpoint
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

            for output in graph_app.stream(inputs):
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

# Mount the router and run
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
