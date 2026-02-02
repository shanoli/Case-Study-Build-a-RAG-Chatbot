from fastapi import APIRouter, HTTPException, Query, Depends
from src.core.security import get_current_user_id
from typing import Optional, List, Dict, Any

from src.services.session_manager import SessionManager
from src.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()
from src.services.session_manager import session_manager

@router.get("/", tags=["Sessions"])
async def get_sessions(current_user_id: str = Depends(get_current_user_id)):
    """List all active sessions for the current user"""
    try:
        sessions = await session_manager.list_active_sessions(user_id=current_user_id)
        return {
            "status": "success",
            "count": len(sessions),
            "sessions": sessions
        }
    except Exception as e:
        logger.error("list_sessions_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{session_id}/history", tags=["Sessions"])
async def get_session_history(session_id: str, current_user_id: str = Depends(get_current_user_id)):
    """Get conversation history for a specific session"""
    try:
        session = await session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
            
        # Verify ownership
        if session.get("user_id") != current_user_id:
            raise HTTPException(status_code=403, detail="Not authorized to access this session")
            
        return {
            "status": "success",
            "session_id": session_id,
            "user_id": session.get("user_id"),
            "conversation_history": session["conversation_history"],
            "message_count": session["message_count"]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_history_failed", session_id=session_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
