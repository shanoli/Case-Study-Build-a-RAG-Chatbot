"""
Knowledge Base Routes
Endpoints for managing the knowledge base and testing RAG retrieval
"""
from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime

from src.services.knowledge_base import KnowledgeBaseService
from src.services.rag_engine import RAGEngine
from src.core.logging import get_logger
from src.core.config import settings
from src.core.exceptions import KnowledgeBaseException, RetrievalException

logger = get_logger(__name__)
router = APIRouter()

# Initialize services (lazy loading)
_kb_service = None
_rag_engine = None
_initialized = False


def get_kb_service():
    """Get or create KB service"""
    global _kb_service
    if _kb_service is None:
        _kb_service = KnowledgeBaseService()
    return _kb_service


def get_rag_engine():
    """Get or create RAG engine"""
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine


async def _ensure_initialized():
    """Ensure knowledge base is initialized"""
    global _initialized
    if not _initialized:
        try:
            kb = get_kb_service()
            await kb.initialize()
            _initialized = True
            logger.info("knowledge_base_initialized")
        except Exception as e:
            logger.error("knowledge_base_initialization_failed", error=str(e))
            # Don't raise - let endpoints handle errors gracefully


# Initialize on module load (not using @router.on_event which is deprecated)
async def _init_kb():
    """Initialize knowledge base on startup"""
    try:
        await _ensure_initialized()
        
        # Check if knowledge base is empty and load default data
        try:
            kb = get_kb_service()
            status = await kb.get_status()
            if status["total_chunks"] == 0:
                logger.info("knowledge_base_empty_loading_defaults")
                try:
                    await load_product_info()
                except Exception as e:
                    logger.warning("failed_to_load_product_info", error=str(e))
        except Exception as e:
            logger.warning("failed_to_check_kb_status", error=str(e))
    except Exception as e:
        logger.warning("knowledge_base_startup_warning", error=str(e))


# Try to initialize async on import
try:
    import asyncio
    # This will be called when the module is imported
    # The actual initialization happens when endpoints are first accessed
    pass
except Exception:
    pass


@router.get("/status", tags=["Knowledge Base"])
async def get_knowledge_status():
    """
    Get knowledge base status
    
    Returns statistics about the loaded knowledge base:
    - Total chunks stored
    - Embedding dimensions
    - Storage size estimate
    - Vector search readiness
    """
    try:
        await _ensure_initialized()
        kb = get_kb_service()
        
        try:
            status = await kb.get_status()
        except Exception as e:
            logger.warning("status_retrieval_error", error=str(e))
            status = {
                "total_chunks": 0,
                "total_documents": 0,
                "embedding_model": settings.EMBEDDING_MODEL,
                "embedding_dimensions": 0,
                "last_updated": datetime.now().isoformat(),
                "storage_size_mb": 0.0,
                "index_ready": False,
                "vector_search_enabled": True,
                "similarity_metric": "cosine",
                "error": str(e)
            }
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "knowledge_base": status
        }
    except Exception as e:
        logger.error("status_retrieval_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to check KB status: {str(e)}")


@router.post("/load-default", tags=["Knowledge Base"])
async def load_product_info():
    """
    Load default product information from data/product_info.txt
    
    This endpoint:
    1. Reads product_info.txt from data folder
    2. Chunks the text using RecursiveCharacterTextSplitter
    3. Generates embeddings using text-embedding-004
    4. Stores in ChromaDB for vector search
    
    **Response includes**:
    - Number of chunks created
    - Embeddings generated
    - Processing time
    - Vector search readiness
    """
    try:
        await _ensure_initialized()
        
        # Load product info file
        product_file = Path("data/product_info.txt")
        if not product_file.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Product info file not found: {product_file}"
            )
        
        logger.info("loading_product_info", file=str(product_file))
        
        with open(product_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        if not content.strip():
            raise HTTPException(
                status_code=400,
                detail="Product info file is empty"
            )
        
        # Upload to knowledge base
        kb = get_kb_service()
        result = await kb.upload_knowledge_base(
            content=content,
            replace_existing=True,
            source="product_info.txt"
        )
        
        # Refresh RAG engine index
        rag = get_rag_engine()
        await rag.refresh_index()
        
        logger.info("product_info_loaded_successfully", **result)
        
        return {
            "status": "success",
            "message": "Product information loaded successfully",
            "timestamp": datetime.now().isoformat(),
            "upload_result": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("product_info_loading_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to load product info: {str(e)}")


@router.post("/upload", tags=["Knowledge Base"])
async def upload_knowledge_file(file: UploadFile = File(...)):
    """
    Upload custom knowledge base file
    
    Accepts text files and processes them into the vector database.
    
    **Supported formats**: .txt, .md
    
    **Parameters**:
    - file: Text file to upload
    
    **Response includes**:
    - Chunks created
    - Embeddings generated
    - Processing statistics
    """
    try:
        await _ensure_initialized()
        
        # Check file type
        if not file.filename.endswith(('.txt', '.md')):
            raise HTTPException(
                status_code=400,
                detail="Only .txt and .md files are supported"
            )
        
        # Read file content
        content = await file.read()
        content_str = content.decode('utf-8')
        
        if not content_str.strip():
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty"
            )
        
        logger.info("uploading_knowledge_file", filename=file.filename)
        
        # Upload to knowledge base
        kb = get_kb_service()
        result = await kb.upload_knowledge_base(
            content=content_str,
            replace_existing=False,
            source=file.filename
        )
        
        # Refresh RAG engine index
        rag = get_rag_engine()
        await rag.refresh_index()
        
        logger.info("knowledge_file_uploaded", filename=file.filename, **result)
        
        return {
            "status": "success",
            "message": f"File '{file.filename}' uploaded successfully",
            "timestamp": datetime.now().isoformat(),
            "upload_result": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("file_upload_failed", filename=file.filename, error=str(e))
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")


@router.post("/query", tags=["Knowledge Base"])
async def test_retrieval(query: str):
    """
    Test vector search and retrieval
    
    This endpoint tests the RAG retrieval component without generation:
    1. Embeds the query
    2. Performs cosine similarity search
    3. Returns relevant chunks with similarity scores
    
    **Parameters**:
    - query: Search query
    
    **Response includes**:
    - Retrieved chunks with similarity scores
    - Number of results
    - Average relevance score
    - Retrieval time
    - Vector search metadata
    
    **Example queries**:
    - "Tell me about SmartWatch Pro X price"
    - "What's the warranty on Wireless Earbuds?"
    - "How do I return a product?"
    """
    try:
        await _ensure_initialized()
        
        if not query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        logger.info("retrieval_test_started", query=query)
        
        # Test retrieval
        rag = get_rag_engine()
        retrieval_result = await rag.retrieve(query, top_k=5)
        
        logger.info(
            "retrieval_test_completed",
            query=query,
            chunks_found=retrieval_result["total_chunks"]
        )
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "retrieval": retrieval_result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("retrieval_test_failed", query=query, error=str(e))
        raise HTTPException(status_code=500, detail=f"Retrieval test failed: {str(e)}")


@router.post("/rag-query", tags=["Knowledge Base"])
async def test_rag_complete(query: str):
    """
    Test complete RAG pipeline (retrieval + generation)
    
    This endpoint tests the full RAG workflow:
    1. Embeds the query
    2. Performs vector search
    3. Generates answer using Gemini with context
    
    **Parameters**:
    - query: User question
    
    **Response includes**:
    - Generated answer
    - Retrieved context chunks
    - Relevance scores
    - Token usage
    - Generation time
    - Context used indication
    
    **Example queries**:
    - "What are the features of SmartWatch Pro X?"
    - "What's the return policy?"
    - "Tell me about the Power Bank Ultra"
    """
    try:
        await _ensure_initialized()
        
        if not query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        logger.info("rag_test_started", query=query)
        
        # Test complete RAG
        rag = get_rag_engine()
        rag_result = await rag.rag_query(query, top_k=5)
        
        logger.info(
            "rag_test_completed",
            query=query,
            chunks_used=rag_result["retrieved_chunks"]
        )
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "rag_result": rag_result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("rag_test_failed", query=query, error=str(e))
        raise HTTPException(status_code=500, detail=f"RAG test failed: {str(e)}")


@router.post("/workflow-query", tags=["Knowledge Base"])
async def test_langgraph_workflow(query: str, session_id: Optional[str] = None):
    """
    Test complete LangGraph workflow
    
    This endpoint tests the full workflow including:
    1. Intent classification
    2. Conditional routing
    3. RAG responder or escalation
    
    **Parameters**:
    - query: User message
    - session_id: Optional session identifier
    
    **Response includes**:
    - Classification result
    - Routing decision
    - Generated response
    - Confidence score
    - Source (RAG, greeting, escalation)
    - Execution metadata
    
    **Example queries**:
    - Product queries: "Tell me about SmartWatch Pro X"
    - Return queries: "What's the return policy?"
    - Greeting: "Hello!"
    - Unclear: "asdfghjkl"
    """
    try:
        await _ensure_initialized()
        
        if not query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        from src.services.langgraph_workflow import ConversationalWorkflow
        
        logger.info("workflow_test_started", query=query, session_id=session_id)
        
        # Import here to avoid circular imports
        workflow = ConversationalWorkflow()
        
        # Execute workflow
        workflow_result = await workflow.execute(
            message=query,
            session_id=session_id or "test-session",
            conversation_history=None
        )
        
        logger.info(
            "workflow_test_completed",
            query=query,
            category=workflow_result["category"],
            source=workflow_result["source"]
        )
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "workflow_result": workflow_result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("workflow_test_failed", query=query, error=str(e))
        raise HTTPException(status_code=500, detail=f"Workflow test failed: {str(e)}")


@router.delete("/reset", tags=["Knowledge Base"])
async def reset_knowledge_base():
    """
    Clear and reset knowledge base
    
    WARNING: This will delete all stored chunks and embeddings.
    
    **Response**:
    - Confirmation of reset
    - Previous chunk count
    """
    try:
        await _ensure_initialized()
        
        kb = get_kb_service()
        status_before = await kb.get_status()
        
        logger.info("resetting_knowledge_base", chunks_before=status_before["total_chunks"])
        
        # Clear knowledge base
        await kb.clear()
        
        # Refresh RAG engine index (will be empty)
        rag = get_rag_engine()
        await rag.refresh_index()
        
        logger.info("knowledge_base_reset_completed")
        
        return {
            "status": "success",
            "message": "Knowledge base reset successfully",
            "chunks_cleared": status_before["total_chunks"],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error("knowledge_base_reset_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")
