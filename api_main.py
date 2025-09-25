from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from backend import gemini_chat_wrapper  # Import wrapper function from backend.py

# Initialize FastAPI app
app = FastAPI(title="SafeMeds RAG Chatbot API", version="1.0")

# ======================
#   Data Models
# ======================

# Input schema for chat request
class ChatRequest(BaseModel):
    user_id: str
    query: str

# Output schema for chat response
class ChatResponse(BaseModel):
    answer: str
    citations: List[str] = []

# Schema for storing chat history (role + message)
class ChatHistoryItem(BaseModel):
    role: str 
    message: str

# In-memory storage for chat history per user
chat_histories: Dict[str, List[ChatHistoryItem]] = {}

# ======================
#   API Endpoints
# ======================

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Chat endpoint:
    - Stores user query in history
    - Calls backend Gemini wrapper for answer
    - Stores assistant response in history
    - Returns answer with citations
    """
    # Get old history or create a new one
    history = chat_histories.setdefault(request.user_id, [])
    history.append(ChatHistoryItem(role="user", message=request.query))

    # Call Gemini wrapper + database search
    answer = gemini_chat_wrapper(request.query, history=[h.dict() for h in history])

    # Placeholder for sources (can extend if backend provides citations)
    citations = ["[SafeMeds DB]"]

    # Save assistant reply in history
    history.append(ChatHistoryItem(role="assistant", message=answer))

    return ChatResponse(answer=answer, citations=citations)


@app.get("/chat/history/{user_id}", response_model=List[ChatHistoryItem])
def get_history(user_id: str):
    """
    Get chat history for a specific user_id
    """
    return chat_histories.get(user_id, [])
