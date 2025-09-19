"""
Custom middleware for the FastAPI application
"""

import time
import logging
import uuid
from typing import Callable
from fastapi import Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
from starlette.middleware.base import RequestResponseEndpoint
from prometheus_client import Counter, Histogram, Gauge
import psutil

from app.core.config import settings

# Prometheus metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ACTIVE_REQUESTS = Gauge('http_requests_active', 'Active HTTP requests')
SYSTEM_MEMORY = Gauge('system_memory_usage_bytes', 'System memory usage')
SYSTEM_CPU = Gauge('system_cpu_usage_percent', 'System CPU usage')

logger = logging.getLogger(__name__)

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all HTTP requests with detailed information"""
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Log request
        start_time = time.time()
        client_ip = request.client.host
        user_agent = request.headers.get("user-agent", "Unknown")
        
        logger.info(
            f"Request started: {request_id} {request.method} {request.url} "
            f"from {client_ip} ({user_agent})"
        )
        
        # Process request
        try:
            response = await call_next(request)
            
            # Log response
            duration = time.time() - start_time
            logger.info(
                f"Request completed: {request_id} {response.status_code} "
                f"in {duration:.3f}s"
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"Request failed: {request_id} after {duration:.3f}s - {str(e)}"
            )
            raise

class MetricsMiddleware(BaseHTTPMiddleware):
    """Collect Prometheus metrics for monitoring"""
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Record active request
        ACTIVE_REQUESTS.inc()
        
        start_time = time.time()
        
        try:
            response = await call_next(request)
            
            # Record metrics
            duration = time.time() - start_time
            REQUEST_DURATION.observe(duration)
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=self._get_endpoint_path(request),
                status=response.status_code
            ).inc()
            
            return response
            
        except Exception as e:
            # Record failed request
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=self._get_endpoint_path(request),
                status=500
            ).inc()
            raise
        finally:
            ACTIVE_REQUESTS.dec()
    
    def _get_endpoint_path(self, request: Request) -> str:
        """Extract endpoint path for metrics"""
        path = request.url.path
        
        # Normalize paths with IDs
        if "/documents/" in path and path.count("/") > 3:
            return "/api/v1/documents/{doc_id}"
        elif "/query/" in path:
            return "/api/v1/query"
        
        return path

class SystemMonitoringMiddleware(BaseHTTPMiddleware):
    """Monitor system resources and update metrics"""
    
    def __init__(self, app, update_interval: int = 30):
        super().__init__(app)
        self.update_interval = update_interval
        self.last_update = 0
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Update system metrics periodically
        current_time = time.time()
        if current_time - self.last_update > self.update_interval:
            self._update_system_metrics()
            self.last_update = current_time
        
        return await call_next(request)
    
    def _update_system_metrics(self):
        """Update system resource metrics"""
        try:
            # Memory usage
            memory = psutil.virtual_memory()
            SYSTEM_MEMORY.set(memory.used)
            
            # CPU usage
            cpu_percent = psutil.cpu_percent()
            SYSTEM_CPU.set(cpu_percent)
            
        except Exception as e:
            logger.warning(f"Failed to update system metrics: {e}")

class SecurityMiddleware(BaseHTTPMiddleware):
    """Add security headers and basic protection"""
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        if not settings.debug:
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        return response

class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Global error handling and standardization"""
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        try:
            return await call_next(request)
            
        except Exception as e:
            request_id = getattr(request.state, 'request_id', 'unknown')
            
            # Log the error
            logger.error(
                f"Unhandled exception in request {request_id}: {str(e)}",
                exc_info=True
            )
            
            # Return standardized error response
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "request_id": request_id,
                    "message": "An unexpected error occurred. Please try again later."
                }
            )

class CacheControlMiddleware(BaseHTTPMiddleware):
    """Add appropriate cache control headers"""
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        
        # Set cache headers based on endpoint
        path = request.url.path
        
        if path.startswith("/static/"):
            # Static files - cache for 1 day
            response.headers["Cache-Control"] = "public, max-age=86400"
        elif path in ["/health", "/metrics"]:
            # Health/metrics - no cache
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        elif path.startswith("/api/v1/query"):
            # Query results - cache for 5 minutes
            response.headers["Cache-Control"] = "private, max-age=300"
        else:
            # Default - cache for 1 minute
            response.headers["Cache-Control"] = "private, max-age=60"
        
        return response

class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Limit request body size to prevent abuse"""
    
    def __init__(self, app, max_size: int = None):
        super().__init__(app)
        self.max_size = max_size or (settings.max_file_size_mb * 1024 * 1024)
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Check content length
        content_length = request.headers.get("content-length")
        
        if content_length:
            content_length = int(content_length)
            if content_length > self.max_size:
                from fastapi.responses import JSONResponse
                return JSONResponse(
                    status_code=413,
                    content={
                        "error": "Request entity too large",
                        "max_size_mb": self.max_size // (1024 * 1024),
                        "received_size_mb": content_length // (1024 * 1024)
                    }
                )
        
        return await call_next(request)

class ProcessingTimeoutMiddleware(BaseHTTPMiddleware):
    """Add timeout for long-running processing requests"""
    
    def __init__(self, app, timeout_seconds: int = 300):
        super().__init__(app)
        self.timeout_seconds = timeout_seconds
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        import asyncio
        
        # Only apply timeout to processing endpoints
        if not (request.url.path.startswith("/api/v1/query") or 
                request.url.path.startswith("/api/v1/ingest")):
            return await call_next(request)
        
        try:
            return await asyncio.wait_for(
                call_next(request), 
                timeout=self.timeout_seconds
            )
        except asyncio.TimeoutError:
            request_id = getattr(request.state, 'request_id', 'unknown')
            logger.warning(f"Request {request_id} timed out after {self.timeout_seconds}s")
            
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=408,
                content={
                    "error": "Request timeout",
                    "message": f"Request exceeded {self.timeout_seconds} second timeout",
                    "request_id": request_id
                }
            )

class DatabaseConnectionMiddleware(BaseHTTPMiddleware):
    """Ensure database connections are properly managed"""
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Database connection will be handled by FastAPI dependencies
        # This middleware can add connection health checks if needed
        return await call_next(request)
