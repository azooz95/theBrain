

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from apis.routers.chat import router as chat_router

app = FastAPI()

# CORS configuration

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Include the chat router
app.include_router(chat_router, prefix="/api", tags=["chat"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000) # to run muli process add workers=4
