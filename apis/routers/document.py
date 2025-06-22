from fastapi import HTTPException, status
from apis.token_generator import get_current_user, verify_token
from fastapi import APIRouter, Depends
from pydantic import BaseModel,ConfigDict

class DocumentEmbedding(BaseModel):
    urls: list[str]

    model_config = ConfigDict(extra='forbid')

router = APIRouter()

@router.get("/document")
def file_handler():
    return {"message": "File handler endpoint is working!"}

@router.post("/document_register")
def embedding_file_handler(request: DocumentEmbedding, user_info: str = Depends(get_current_user)):
    return {'message': 'created successfully', "status":status.HTTP_201_CREATED} 

