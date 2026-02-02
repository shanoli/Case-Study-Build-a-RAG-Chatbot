"""
Session Manager - Task 3
Manages conversation sessions with in-memory or Redis storage
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import uuid
import asyncio
from collections import OrderedDict

from src.core.config import settings
from src.core.logging import get_logger
from src.core.exceptions import SessionException

logger = get_logger(__name__)


class InMemorySessionStore:
    """
    In-memory session store with TTL and LRU eviction
    
    Features:
    - Automatic session expiration
    - LRU cache for memory efficiency
    - Thread-safe operations
    """
    
    def __init__(self, max_sessions: int = 1000):
        """Initialize in-memory store"""
        self.sessions: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.max_sessions = max_sessions
        self._lock = asyncio.Lock()
        logger.info("in_memory_session_store_initialized", max_sessions=max_sessions)
    
    async def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data"""
        async with self._lock:
            if session_id in self.sessions:
                session = self.sessions[session_id]
                
                # Check if expired
                if datetime.fromisoformat(session["expires_at"]) < datetime.now():
                    del self.sessions[session_id]
                    logger.debug("session_expired", session_id=session_id)
                    return None
                
                # Move to end (LRU)
                self.sessions.move_to_end(session_id)
                return session
            
            return None
    
    async def set(self, session_id: str, data: Dict[str, Any], ttl: int) -> None:
        """Set session data with TTL"""
        async with self._lock:
            # Add expiration
            data["expires_at"] = (
                datetime.now() + timedelta(seconds=ttl)
            ).isoformat()
            
            self.sessions[session_id] = data
            self.sessions.move_to_end(session_id)
            
            # Evict oldest if over limit
            while len(self.sessions) > self.max_sessions:
                oldest_id = next(iter(self.sessions))
                del self.sessions[oldest_id]
                logger.debug("session_evicted", session_id=oldest_id)
    
    async def delete(self, session_id: str) -> bool:
        """Delete session"""
        async with self._lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                return True
            return False
    
    async def list_active(self) -> List[str]:
        """List all active session IDs"""
        async with self._lock:
            now = datetime.now()
            active = []
            
            for session_id, session in list(self.sessions.items()):
                if datetime.fromisoformat(session["expires_at"]) >= now:
                    active.append(session_id)
                else:
                    del self.sessions[session_id]
            
            return active
    
    async def cleanup_expired(self) -> int:
        """Remove expired sessions"""
        async with self._lock:
            now = datetime.now()
            expired = [
                sid for sid, session in self.sessions.items()
                if datetime.fromisoformat(session["expires_at"]) < now
            ]
            
            for sid in expired:
                del self.sessions[sid]
            
            if expired:
                logger.info("expired_sessions_cleaned", count=len(expired))
            
            return len(expired)


# Global session store instance for shared state across all Manager instances
_global_store = InMemorySessionStore()

class SessionManager:
    """
    Manages conversation sessions
    
    Features:
    - Create and retrieve sessions
    - Conversation history management
    - Session metadata tracking
    - Automatic expiration
    """
    
    def __init__(self):
        """Initialize session manager using the global store"""
        # Use a truly shared global store
        self.store = _global_store
        self.ttl = settings.SESSION_TTL_SECONDS
        
        logger.info(
            "session_manager_initialized",
            ttl_seconds=self.ttl,
            store="in_memory"
        )
    
    def _create_session_id(self) -> str:
        """Generate unique session ID"""
        return f"session_{uuid.uuid4().hex[:16]}"
    
    async def create_session(
        self,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create new session
        
        Args:
            metadata: Optional session metadata
        
        Returns:
            New session ID
        """
        session_id = self._create_session_id()
        
        session_data = {
            "session_id": session_id,
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "conversation_history": [],
            "metadata": metadata or {},
            "message_count": 0
        }
        
        await self.store.set(session_id, session_data, self.ttl)
        
        logger.info(
            "session_created",
            session_id=session_id,
            user_id=user_id,
            metadata=metadata
        )
        
        return session_id
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session data
        
        Args:
            session_id: Session ID
        
        Returns:
            Session data or None if not found
        """
        session = await self.store.get(session_id)
        
        if session:
            logger.debug("session_retrieved", session_id=session_id)
        else:
            logger.debug("session_not_found", session_id=session_id)
        
        return session
    
    async def update_session(
        self,
        session_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """
        Update session data
        
        Args:
            session_id: Session ID
            updates: Data to update
        
        Returns:
            Success status
        """
        session = await self.get_session(session_id)
        
        if not session:
            logger.warning("session_update_failed_not_found", session_id=session_id)
            return False
        
        # Update fields
        session.update(updates)
        session["updated_at"] = datetime.now().isoformat()
        
        # Save back with refreshed TTL
        await self.store.set(session_id, session, self.ttl)
        
        logger.debug("session_updated", session_id=session_id)
        return True
    
    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add message to conversation history
        
        Args:
            session_id: Session ID
            role: Message role (user/assistant)
            content: Message content
            metadata: Optional message metadata
        
        Returns:
            Success status
        """
        session = await self.get_session(session_id)
        
        if not session:
            return False
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        session["conversation_history"].append(message)
        session["message_count"] = len(session["conversation_history"])
        
        # Trim history if exceeds limit
        max_history = settings.MAX_CONVERSATION_HISTORY
        if len(session["conversation_history"]) > max_history:
            session["conversation_history"] = session["conversation_history"][-max_history:]
            logger.debug(
                "conversation_history_trimmed",
                session_id=session_id,
                limit=max_history
            )
        
        session["updated_at"] = datetime.now().isoformat()
        
        # Save with refreshed TTL
        await self.store.set(session_id, session, self.ttl)
        
        logger.debug(
            "message_added",
            session_id=session_id,
            role=role,
            total_messages=session["message_count"]
        )
        
        return True
    
    async def get_conversation_history(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history
        
        Args:
            session_id: Session ID
            limit: Optional limit on number of messages
        
        Returns:
            Conversation history
        """
        session = await self.get_session(session_id)
        
        if not session:
            return []
        
        history = session["conversation_history"]
        
        if limit:
            history = history[-limit:]
        
        return history
    
    async def delete_session(self, session_id: str) -> bool:
        """
        Delete session
        
        Args:
            session_id: Session ID
        
        Returns:
            Success status
        """
        success = await self.store.delete(session_id)
        
        if success:
            logger.info("session_deleted", session_id=session_id)
        else:
            logger.warning("session_delete_failed", session_id=session_id)
        
        return success
    
    async def list_active_sessions(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all active sessions
        
        Args:
            user_id: Optional user ID to filter by
            
        Returns:
            List of active session summaries
        """
        session_ids = await self.store.list_active()
        
        sessions = []
        for session_id in session_ids:
            session = await self.get_session(session_id)
            if session:
                # Filter by user_id if provided
                if user_id and session.get("user_id") != user_id:
                    continue
                    
                sessions.append({
                    "session_id": session_id,
                    "user_id": session.get("user_id"),
                    "created_at": session["created_at"],
                    "updated_at": session["updated_at"],
                    "message_count": session["message_count"],
                    "metadata": session.get("metadata", {})
                })
        
        logger.info("active_sessions_listed", count=len(sessions))
        return sessions
    
    async def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions
        
        Returns:
            Number of sessions cleaned up
        """
        count = await self.store.cleanup_expired()
        return count


# Global session manager instance
session_manager = SessionManager()