"""
Main application entry point
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from app.core.config import settings
from app.core.database import init_db
from app.core.logging import setup_logging
from app.api.routes import router as api_router
from app.services.orchestrator import PDFProcessingOrchestrator
from app.models.requests import QueryRequest, IngestionRequest
from app.core.exceptions import setup_exception_handlers

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Global orchestrator instance
orchestrator: Optional[PDFProcessingOrchestrator] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global orchestrator
    
    # Startup
    logger.info("Starting Enterprise Multi-Agent PDF Processing System")
    
    # Initialize database
    await init_db()
    
    # Initialize orchestrator
    orchestrator = PDFProcessingOrchestrator(settings.processing_config)
    await orchestrator.initialize()
    
    logger.info("System initialization completed")
    
    yield
    
    # Shutdown
    logger.info("Shutting down system")
    if orchestrator:
        await orchestrator.cleanup()

# Create FastAPI application
app = FastAPI(
    title="Enterprise Multi-Agent PDF Processing System",
    description="Advanced AI system for processing 1000+ PDFs with relationship mapping",
    version="2.0.0",
    lifespan=lifespan
)

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup exception handlers
setup_exception_handlers(app)

# Include API routes
app.include_router(api_router, prefix="/api/v1")

# Health check endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "system": "enterprise-pdf-processor",
        "version": "2.0.0"
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Main processing endpoints
@app.post("/ingest")
async def ingest_documents(request: IngestionRequest, background_tasks: BackgroundTasks):
    """Ingest large batch of PDFs"""
    if not orchestrator:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    # Start ingestion in background
    background_tasks.add_task(
        orchestrator.ingest_documents, 
        request.pdf_paths
    )
    
    return {
        "status": "ingestion_started",
        "documents_queued": len(request.pdf_paths),
        "estimated_time": len(request.pdf_paths) * 2
    }

@app.post("/query")
async def process_query(request: QueryRequest):
    """Process query with full multi-agent analysis"""
    if not orchestrator:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        result = await orchestrator.process_query(
            query=request.query,
            user_id=request.user_id,
            processing_mode=request.processing_mode,
            max_documents=request.max_documents,
            include_relationships=request.include_relationships
        )
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
        
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/query")
async def websocket_query_processing(websocket: WebSocket):
    """Stream query processing results in real-time"""
    await websocket.accept()
    
    if not orchestrator:
        await websocket.send_json({"error": "System not initialized"})
        return
    
    try:
        while True:
            # Receive query from client
            data = await websocket.receive_json()
            query = data.get("query", "")
            user_id = data.get("user_id", "default")
            
            if not query:
                await websocket.send_json({"error": "Query required"})
                continue
            
            # Process query with streaming updates
            async for update in orchestrator.process_query_streaming(query, user_id, websocket):
                await websocket.send_json(update)
                
    except Exception as e:
        await websocket.send_json({"status": "error", "error": str(e)})
    finally:
        await websocket.close()

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """Serve enterprise dashboard"""
    with open("app/static/index.html", "r") as f:
        return HTMLResponse(content=f.read())

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers if not settings.debug else 1,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
