"""
FastAPI dependencies for dependency injection and security
"""

from typing import Optional, Dict, Any
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import time
import redis
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.core.security import verify_api_key, get_current_user, RateLimiter
from app.services.orchestrator import PDFProcessingOrchestrator

# Security schemes
security = HTTPBearer(auto_error=False)

# Global orchestrator instance (will be set in main.py)
_orchestrator: Optional[PDFProcessingOrchestrator] = None

def set_orchestrator(orchestrator: PDFProcessingOrchestrator):
    """Set the global orchestrator instance"""
    global _orchestrator
    _orchestrator = orchestrator

def get_orchestrator() -> PDFProcessingOrchestrator:
    """Get the orchestrator instance"""
    if _orchestrator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="System not initialized. Please wait for startup to complete."
        )
    return _orchestrator

def get_redis_client() -> redis.Redis:
    """Get Redis client for caching"""
    try:
        client = redis.from_url(settings.redis_url)
        client.ping()  # Test connection
        return client
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Redis connection failed: {str(e)}"
        )

def verify_api_key_dependency(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Dict[str, Any]:
    """Verify API key if authentication is enabled"""
    
    # Skip authentication in development
    if settings.debug and not settings.api_key:
        return {"user_id": "development_user", "permissions": ["all"]}
    
    if not credentials:
        if settings.api_key:  # API key required
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key required"
            )
        else:
            return {"user_id": "anonymous", "permissions": ["read"]}
    
    # Verify the API key
    if not verify_api_key(credentials.credentials):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    return {"user_id": "api_user", "permissions": ["all"]}

def get_current_user_info(
    user_data: Dict[str, Any] = Depends(verify_api_key_dependency)
) -> Dict[str, Any]:
    """Get current user information"""
    return user_data

def rate_limit_dependency(request: Request) -> None:
    """Rate limiting dependency"""
    limiter = RateLimiter(
        requests_per_minute=settings.rate_limit_per_minute,
        burst_limit=settings.rate_limit_burst
    )
    
    client_ip = request.client.host
    
    if not limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later."
        )

def validate_file_upload(request: Request) -> None:
    """Validate file upload requirements"""
    content_length = request.headers.get("content-length")
    
    if content_length:
        content_length = int(content_length)
        max_size = settings.max_file_size_mb * 1024 * 1024  # Convert to bytes
        
        if content_length > max_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Maximum size: {settings.max_file_size_mb}MB"
            )

def get_processing_context(
    user_data: Dict[str, Any] = Depends(get_current_user_info),
    orchestrator: PDFProcessingOrchestrator = Depends(get_orchestrator)
) -> Dict[str, Any]:
    """Get processing context with user and system info"""
    return {
        "user_id": user_data["user_id"],
        "permissions": user_data["permissions"],
        "orchestrator": orchestrator,
        "timestamp": time.time()
    }

# Dependency for database session
def get_database_session(db: Session = Depends(get_db)) -> Session:
    """Get database session"""
    return db

# Cache dependency
def get_cache_client(redis_client: redis.Redis = Depends(get_redis_client)) -> redis.Redis:
    """Get cache client"""
    return redis_client

# System health dependency
def check_system_health(
    orchestrator: PDFProcessingOrchestrator = Depends(get_orchestrator)
) -> Dict[str, Any]:
    """Check system health before processing"""
    status = orchestrator.get_system_status()
    
    # Check if system is overloaded
    resource_usage = status.get("resource_usage", {})
    if resource_usage.get("memory_percent", 0) > 90:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="System overloaded - high memory usage"
        )
    
    if resource_usage.get("cpu_percent", 0) > 95:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="System overloaded - high CPU usage"
        )
    
    return status

# Query validation dependency
def validate_query_request(request: Request) -> None:
    """Validate query request parameters"""
    # Add custom query validation logic here
    pass

# Document access dependency
def check_document_access(
    doc_id: str,
    user_data: Dict[str, Any] = Depends(get_current_user_info),
    orchestrator: PDFProcessingOrchestrator = Depends(get_orchestrator)
) -> Dict[str, Any]:
    """Check if user has access to specific document"""
    
    if doc_id not in orchestrator.processed_documents:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    # Add access control logic here if needed
    # For now, all authenticated users can access all documents
    
    return orchestrator.processed_documents[doc_id]
