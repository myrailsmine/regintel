"""
API routes for the enterprise PDF processing system
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends, File, UploadFile
from fastapi.responses import JSONResponse

from app.models.requests import (
    QueryRequest, IngestionRequest, QueryResponse, 
    IngestionResponse, SystemStatus, DocumentAnalysis,
    ConfigurationRequest, ErrorResponse
)
from app.core.config import settings
from app.services.orchestrator import PDFProcessingOrchestrator

router = APIRouter()

# Dependency to get orchestrator instance
def get_orchestrator() -> PDFProcessingOrchestrator:
    # This would be injected from the main app
    return getattr(router, 'orchestrator', None)

@router.get("/status", response_model=SystemStatus)
async def get_system_status(orchestrator: PDFProcessingOrchestrator = Depends(get_orchestrator)):
    """Get comprehensive system status"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    status = orchestrator.get_system_status()
    return SystemStatus(**status)

@router.post("/ingest", response_model=IngestionResponse)
async def ingest_documents(
    request: IngestionRequest,
    orchestrator: PDFProcessingOrchestrator = Depends(get_orchestrator)
):
    """Ingest documents for processing"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        result = await orchestrator.ingest_documents(request.pdf_paths)
        return IngestionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    orchestrator: PDFProcessingOrchestrator = Depends(get_orchestrator)
):
    """Process a query against the document collection"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        result = await orchestrator.process_query(
            query=request.query,
            user_id=request.user_id,
            processing_mode=request.processing_mode.value,
            max_documents=request.max_documents,
            include_relationships=request.include_relationships
        )
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return QueryResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    orchestrator: PDFProcessingOrchestrator = Depends(get_orchestrator)
):
    """Upload PDF files"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    uploaded_files = []
    
    for file in files:
        if not file.filename.endswith('.pdf'):
            continue
        
        # Save uploaded file
        file_path = f"{settings.upload_dir}/{file.filename}"
        try:
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            
            uploaded_files.append({
                "filename": file.filename,
                "size": len(content),
                "path": file_path
            })
        except Exception as e:
            continue
    
    return {
        "message": f"Uploaded {len(uploaded_files)} files",
        "files": uploaded_files
    }

@router.get("/documents")
async def list_documents(
    orchestrator: PDFProcessingOrchestrator = Depends(get_orchestrator)
):
    """List processed documents"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    documents = []
    for doc_id, doc_meta in orchestrator.processed_documents.items():
        documents.append({
            "doc_id": doc_id,
            "filename": doc_meta.filename,
            "page_count": doc_meta.page_count,
            "entities_count": len(doc_meta.extracted_entities),
            "relationships_count": len(doc_meta.relationships),
            "quality_score": doc_meta.quality_score,
            "processing_timestamp": doc_meta.processing_timestamp
        })
    
    return {"documents": documents, "total": len(documents)}

@router.get("/documents/{doc_id}", response_model=DocumentAnalysis)
async def get_document_analysis(
    doc_id: str,
    orchestrator: PDFProcessingOrchestrator = Depends(get_orchestrator)
):
    """Get detailed analysis of specific document"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    if doc_id not in orchestrator.processed_documents:
        raise HTTPException(status_code=404, detail="Document not found")
    
    doc = orchestrator.processed_documents[doc_id]
    
    return DocumentAnalysis(
        doc_id=doc_id,
        filename=doc.filename,
        metadata={
            "file_size": doc.file_size,
            "page_count": doc.page_count,
            "author": doc.author,
            "title": doc.title,
            "creation_date": doc.creation_date,
            "processing_timestamp": doc.processing_timestamp,
            "content_hash": doc.content_hash
        },
        entity_analysis={
            "total_entities": len(doc.extracted_entities),
            "entity_types": list(set(e["label"] for e in doc.extracted_entities)),
            "high_confidence_entities": [e for e in doc.extracted_entities if e["confidence"] > 0.8]
        },
        relationship_analysis={
            "total_relationships": len(doc.relationships),
            "relationship_types": list(set(r["relationship_type"] for r in doc.relationships))
        },
        quality_metrics={
            "quality_score": doc.quality_score,
            "completeness": min(1.0, len(doc.extracted_entities) / max(1, doc.page_count))
        }
    )

@router.delete("/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    orchestrator: PDFProcessingOrchestrator = Depends(get_orchestrator)
):
    """Delete a document from the system"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    if doc_id not in orchestrator.processed_documents:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Remove from processed documents
    del orchestrator.processed_documents[doc_id]
    
    # Note: In a production system, you'd also remove from vector indices
    # This would require rebuilding indices or implementing incremental updates
    
    return {"message": f"Document {doc_id} deleted successfully"}

@router.post("/configure")
async def configure_system(
    request: ConfigurationRequest,
    orchestrator: PDFProcessingOrchestrator = Depends(get_orchestrator)
):
    """Update system configuration"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    # Update configuration
    config = orchestrator.config
    
    if request.extraction_strategy:
        config.extraction_strategy = request.extraction_strategy
    if request.chunk_size:
        config.chunk_size = request.chunk_size
    if request.chunk_overlap:
        config.chunk_overlap = request.chunk_overlap
    if request.enable_validation is not None:
        config.enable_validation = request.enable_validation
    
    return {
        "message": "Configuration updated successfully",
        "current_config": {
            "extraction_strategy": config.extraction_strategy.value,
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "enable_validation": getattr(config, 'enable_validation', True)
        }
    }

@router.get("/search")
async def search_content(
    q: str,
    limit: int = 10,
    orchestrator: PDFProcessingOrchestrator = Depends(get_orchestrator)
):
    """Search content across documents"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query parameter 'q' is required")
    
    try:
        # Perform search using vector store
        search_results = await orchestrator.vector_store.multi_level_search(q, top_k=limit)
        
        # Format results
        formatted_results = []
        for chunk in search_results.get("chunks", [])[:limit]:
            metadata = chunk.get("metadata", {})
            formatted_results.append({
                "content": metadata.get("chunk_text", ""),
                "document": metadata.get("doc_id", "Unknown"),
                "page": metadata.get("page_number", "Unknown"),
                "similarity": chunk.get("similarity", 0),
                "chunk_id": metadata.get("chunk_id", "")
            })
        
        return {
            "query": q,
            "results": formatted_results,
            "total_found": len(search_results.get("chunks", [])),
            "search_stats": {
                "documents": len(search_results.get("documents", [])),
                "pages": len(search_results.get("pages", [])),
                "chunks": len(search_results.get("chunks", [])),
                "entities": len(search_results.get("entities", []))
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}

@router.get("/metrics")
async def get_metrics(
    orchestrator: PDFProcessingOrchestrator = Depends(get_orchestrator)
):
    """Get system metrics for monitoring"""
    if not orchestrator:
        return {"error": "System not initialized"}
    
    stats = orchestrator.get_system_status()
    
    return {
        "metrics": {
            "documents_processed": stats["system_info"]["documents_indexed"],
            "total_queries": stats["processing_stats"]["total_queries"],
            "average_processing_time": stats["processing_stats"]["average_processing_time"],
            "uptime_seconds": stats["system_info"]["uptime_seconds"],
            "memory_usage_percent": stats["resource_usage"]["memory_percent"],
            "cpu_usage_percent": stats["resource_usage"]["cpu_percent"]
        },
        "vector_indices": stats["system_info"]["vector_store_stats"],
        "agent_performance": stats["agent_stats"]
    }
