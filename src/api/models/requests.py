from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, Dict, Any

class ChatRequest(BaseModel):
    """
    Chat request model
    """
    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(None, description="Session ID for continuing conversation")
    user_id: Optional[str] = Field(None, description="Unique user identifier (e.g., mail_id)")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")

    # Allow extra fields (fixes 15ae64130e69b2222cd817eeb9a01c5fd9b5d6a2 issue)
    model_config = ConfigDict(extra='ignore')
