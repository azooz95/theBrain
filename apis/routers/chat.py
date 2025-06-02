
# fast router 

from langchain_core.messages import HumanMessage, AIMessage

from fastapi import APIRouter, Depends, HTTPException, FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from apis.token_generator import get_current_user, check_token
from starlette.websockets import WebSocketState

from src.smart_graph.graph.main_graph import app as graph_app

app = FastAPI()
class ChatRequest(BaseModel):
    message: str


router = APIRouter()

@router.post("/chat")
async def chat_endpoint(request: ChatRequest, user_id: str = Depends(get_current_user)):

    if not request.message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    

    user_input = request.message

    inputs = {"messages": [HumanMessage(content=user_input)]}
    for output in graph_app.stream(inputs):
        for _, val in output.items():
            if isinstance(val, dict) and "messages" in val:
                messages = val["messages"]
                if isinstance(messages, list) and messages:
                    print(f"\n🤖 {messages[-1].content}\n")
                    return {"response": messages[-1].content, "user_id": user_id}

    return {"response": f"Received message: {request.message}, user_id: {user_id}"}


@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    # Read token from the headers


    # Validate the token
    token = websocket.headers.get("authorization")

    if not token or not token.startswith("Bearer "):
        await websocket.close(code=1008)
        return

    token_value = token.replace("Bearer ", "")

    try: 
        user = check_token(token_value)

    except HTTPException as e:
        await websocket.send_text(f"❌ Invalid token: {e.detail}")
        await websocket.close(code=1008)
        print(f"⚠️ Invalid token: {e.detail}")
        return
    
    except Exception as e:
        await websocket.send_text(f"❌ Error checking token: {str(e)}")
        await websocket.close(code=1008)
        print(f"⚠️ Error checking token: {e}")
        return

    try:
        await websocket.accept()
        await websocket.send_text(f"✅ Authenticated as {user['username']}")

        while True:
            data = await websocket.receive_text()

            if not data.strip():
                await websocket.send_text("❌ Empty message")
                continue

            inputs = {"messages": [HumanMessage(content=data)]}
            for output in graph_app.stream(inputs):
                for _, val in output.items():
                    if isinstance(val, dict) and "messages" in val:
                        messages = val["messages"]
                        if isinstance(messages, list) and messages:
                            response = messages[-1].content
                            print(f"\n🤖 {response}\n")
                            await websocket.send_text(response)

    except WebSocketDisconnect:
        print("🔌 WebSocket disconnected")
        await websocket.send_text("🔌 WebSocket disconnected")
        await websocket.close(code=1008)

    except Exception as e:
        print(f"⚠️ Error: {e}")
        await websocket.send_text(f"❌ Error: {str(e)}")
        if websocket.client_state != WebSocketState.DISCONNECTED:
            await websocket.close()

if __name__ == "__main__":

    app.include_router(router)
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)