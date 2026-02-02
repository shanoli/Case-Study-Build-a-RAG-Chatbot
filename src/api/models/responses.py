from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class ChatResponse(BaseModel):
    """
    Chat response model
    """
    session_id: str
    category: str
    reply: str
    confidence: float
    source: str
    context_used: bool
    language: str
    retrieved_chunks: int
    timestamp: str

class ErrorResponse(BaseModel):
    """
    Error response model
    """
    detail: str
