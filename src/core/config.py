"""
Core configuration and settings
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import List, Optional
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Google AI
    GOOGLE_API_KEY: str = Field(validation_alias="GEMINI_API_KEY")
    GEMINI_MODEL: str = "gemini-1.5-pro-latest"
    EMBEDDING_MODEL: str = "models/text-embedding-004"
    MAX_OUTPUT_TOKENS: int = 1024
    TEMPERATURE: float = 0.7
    
    # ChromaDB
    CHROMA_PERSIST_DIRECTORY: str = "./data/chroma_db"
    CHROMA_TELEMETRY_ENABLED: bool = False
    ANONYMIZED_TELEMETRY: bool = False
    
    # Session Management
    SESSION_TTL_SECONDS: int = 1800
    MAX_CONVERSATION_HISTORY: int = 50
    
    # Text Processing
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    RELEVANCE_THRESHOLD: float = 0.4
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG_MODE: bool = True
    LOG_LEVEL: str = "INFO"
    
    # CORS
    CORS_ORIGINS: List[str] = ["*"]
    
    # Redis (Optional)
    REDIS_URL: str = "redis://localhost:6379"
    USE_REDIS: bool = False
    
    # Serper
    SERPER_API_KEY: Optional[str] = None
    
    # JWT Authentication
    JWT_SECRET: str = Field(default="secret", validation_alias="JWT_SECRET")
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore"
    )


# Global settings instance
settings = Settings()
