"""
Main FastAPI Application
Entry point for the RAG Chatbot API
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio

from src.core.config import settings
from src.core.logging import get_logger
from src.api.routes import chat, knowledge, sessions, health, debug, auth

logger = get_logger(__name__)


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager
    
    Handles startup and shutdown events
    """
    # Startup
    logger.info(
        "application_starting",
        host=settings.API_HOST,
        port=settings.API_PORT,
        debug=settings.DEBUG_MODE
    )
    
    # Initialize knowledge base on startup
    try:
        from src.services.knowledge_base import KnowledgeBaseService
        from pathlib import Path
        
        kb_service = KnowledgeBaseService()
        await kb_service.initialize()
        
        # Check if knowledge base is empty
        status = await kb_service.get_status()
        if status["total_chunks"] == 0:
            logger.info("knowledge_base_empty_loading_defaults")
            
            # Load default product info
            product_file = Path("data/product_info.txt")
            if product_file.exists():
                with open(product_file, "r", encoding="utf-8") as f:
                    content = f.read()
                
                result = await kb_service.upload_knowledge_base(
                    content=content,
                    replace_existing=True,
                    source="product_info.txt"
                )
                
                logger.info(
                    "default_product_info_loaded",
                    chunks=result["chunks_created"],
                    embeddings=result["embeddings_generated"]
                )
            else:
                logger.warning("product_info_file_not_found", path=str(product_file))
        else:
            logger.info("knowledge_base_already_initialized", chunks=status["total_chunks"])
    
    except Exception as e:
        logger.error("knowledge_base_initialization_failed_on_startup", error=str(e))
    
    yield
    
    # Shutdown
    logger.info("application_shutting_down")


# Create FastAPI app
app = FastAPI(
    title="RAG Chatbot API",
    description="Conversational RAG chatbot with LangGraph orchestration",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(chat.router, tags=["Chat"])
app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(knowledge.router, prefix="/knowledge", tags=["Knowledge Base"])
app.include_router(sessions.router, prefix="/sessions", tags=["Sessions"])
app.include_router(debug.router, prefix="/debug", tags=["Debug"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "RAG Chatbot API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/ui")
async def ui():
    """Serve the Chatbot UI"""
    from fastapi.responses import FileResponse
    from pathlib import Path
    
    ui_path = Path("src/chatbot_ui.html")
    if not ui_path.exists():
        return {"error": "UI file not found"}
        
    return FileResponse(ui_path)


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG_MODE,
        log_level=settings.LOG_LEVEL.lower()
    )