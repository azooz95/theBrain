
# fast router 

from langchain_core.messages import HumanMessage, AIMessage

from fastapi import APIRouter, Depends, HTTPException, FastAPI
from pydantic import BaseModel
from apis.token_generator import get_current_user

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


if __name__ == "__main__":

    app.include_router(router)
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)