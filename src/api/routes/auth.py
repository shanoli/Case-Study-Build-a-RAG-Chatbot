"""
Authentication Routes
"""
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, EmailStr
from src.core.security import create_access_token
from src.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()

class LoginRequest(BaseModel):
    email: EmailStr

@router.post("/login")
async def login(request: LoginRequest):
    """
    Simple login endpoint that returns a JWT for an email
    """
    logger.info("login_attempt", email=request.email)
    
    # In a real app, you'd verify password here
    # For this assessment, we'll assume email exists and is valid
    
    access_token = create_access_token(data={"sub": request.email})
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "email": request.email
    }
