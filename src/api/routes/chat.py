"""
Chat Routes
Main conversational interface with LangGraph workflow
"""
from fastapi import APIRouter, HTTPException
from datetime import datetime

from src.api.models.requests import ChatRequest
from src.api.models.responses import ChatResponse, ErrorResponse
from src.services.session_manager import SessionManager
from src.services.langgraph_workflow import ConversationalWorkflow
from src.core.logging import get_logger
from src.utils.prompts import GREETING_RESPONSES, detect_language
from src.core.security import get_current_user_id
from fastapi import APIRouter, HTTPException, Depends

logger = get_logger(__name__)
router = APIRouter()

# Initialize services
from src.services.session_manager import session_manager
workflow = ConversationalWorkflow()


@router.post("/chat", response_model=ChatResponse, responses={500: {"model": ErrorResponse}})
async def chat(request: ChatRequest, current_user_id: str = Depends(get_current_user_id)):
    """
    Main chat endpoint
    
    Process user message through LangGraph workflow:
    1. Create/retrieve session
    2. Execute LangGraph workflow (classify → route → respond)
    3. Update session with conversation history
    4. Return response
    
    **New session greeting:**
    If no session_id is provided, a new session is created and a greeting is returned.
    
    **Existing session:**
    If session_id is provided, the conversation continues with context.
    """
    start_time = datetime.now()
    
    try:
        # Handle session
        if request.session_id:
            # Existing session
            session = await session_manager.get_session(request.session_id)
            if not session:
                logger.warning("session_not_found", session_id=request.session_id)
                raise HTTPException(
                    status_code=404,
                    detail=f"Session {request.session_id} not found"
                )
            session_id = request.session_id
            is_new_session = False
        else:
            # New session - create and send greeting
            session_id = await session_manager.create_session(
                user_id=current_user_id,
                metadata=request.metadata
            )
            is_new_session = True
            logger.info("new_session_created", session_id=session_id)
        
        # Get conversation history
        conversation_history = await session_manager.get_conversation_history(
            session_id,
            limit=10  # Last 10 messages for context
        )
        
        # If new session, return greeting
        if is_new_session:
            language = detect_language(request.message)
            greeting = GREETING_RESPONSES.get(language, GREETING_RESPONSES["en"])
            
            # Add user message and greeting to history
            await session_manager.add_message(
                session_id,
                role="user",
                content=request.message
            )
            await session_manager.add_message(
                session_id,
                role="assistant",
                content=greeting,
                metadata={"type": "greeting", "is_first_message": True}
            )
            
            response = ChatResponse(
                session_id=session_id,
                category="greeting",
                reply=greeting,
                confidence=1.0,
                source="greeting",
                context_used=False,
                language=language,
                retrieved_chunks=0,
                timestamp=datetime.now().isoformat()
            )
            
            logger.info(
                "greeting_sent",
                session_id=session_id,
                language=language
            )
            
            return response
        
        # Execute LangGraph workflow
        logger.info(
            "chat_request_received",
            session_id=session_id,
            message_length=len(request.message)
        )
        
        workflow_result = await workflow.execute(
            message=request.message,
            session_id=session_id,
            conversation_history=conversation_history
        )
        
        # Add messages to session history
        await session_manager.add_message(
            session_id,
            role="user",
            content=request.message
        )
        await session_manager.add_message(
            session_id,
            role="assistant",
            content=workflow_result["reply"],
            metadata={
                "category": workflow_result["category"],
                "source": workflow_result["source"],
                "confidence": workflow_result["confidence"]
            }
        )
        
        # Build response
        response = ChatResponse(
            session_id=workflow_result["session_id"],
            category=workflow_result["category"],
            reply=workflow_result["reply"],
            confidence=workflow_result["confidence"],
            source=workflow_result["source"],
            context_used=workflow_result["context_used"],
            language=workflow_result.get("language", "en"),
            retrieved_chunks=workflow_result.get("metadata", {}).get("retrieved_chunks", 0),
            timestamp=workflow_result["timestamp"]
        )
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        logger.info(
            "chat_request_completed",
            session_id=session_id,
            category=workflow_result["category"],
            processing_time_ms=int(processing_time)
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("chat_request_failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Chat processing failed: {str(e)}"
        )