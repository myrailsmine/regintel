"""
Pydantic models for API requests and responses
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum

class ProcessingMode(str, Enum):
    FAST = "fast"
    BALANCED = "balanced"
    COMPREHENSIVE = "comprehensive"
    REAL_TIME = "real_time"

class QueryRequest(BaseModel):
    """Request model for query processing"""
    query: str = Field(..., min_length=1, max_length=1000, description="Query to process")
    user_id: str = Field(default="default", description="User identifier")
    processing_mode: ProcessingMode = Field(default=ProcessingMode.BALANCED, description="Processing mode")
    max_documents: Optional[int] = Field(default=None, gt=0, le=1000, description="Maximum documents to analyze")
    include_relationships: bool = Field(default=True, description="Include cross-document relationships")
    stream_response: bool = Field(default=False, description="Stream response updates")
    custom_prompt: Optional[str] = Field(default=None, description="Custom prompt template")
    
    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty or only whitespace')
        return v.strip()

class IngestionRequest(BaseModel):
    """Request model for document ingestion"""
    pdf_paths: List[str] = Field(..., min_items=1, description="List of PDF file paths")
    batch_size: int = Field(default=100, gt=0, le=500, description="Batch size for processing")
    processing_mode: ProcessingMode = Field(default=ProcessingMode.COMPREHENSIVE, description="Processing mode")
    force_reprocess: bool = Field(default=False, description="Force reprocessing of existing documents")
    
    @validator('pdf_paths')
    def validate_pdf_paths(cls, v):
        if not v:
            raise ValueError('At least one PDF path must be provided')
        
        for path in v:
            if not path.endswith('.pdf'):
                raise ValueError(f'Invalid file type: {path}. Only PDF files are allowed.')
        
        return v

class QueryResponse(BaseModel):
    """Response model for query processing"""
    query_id: str
    query: str
    response: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    processing_time: float
    
    # Supporting evidence
    supporting_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    cross_references: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Document coverage
    document_coverage: Dict[str, int] = Field(default_factory=dict)
    
    # Processing metadata
    processing_metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Sources
    sources: List[str] = Field(default_factory=list)

class IngestionResponse(BaseModel):
    """Response model for document ingestion"""
    status: str
    documents_processed: int
    processing_time: float
    failed_documents: List[str] = Field(default_factory=list)
    
    # Index statistics
    index_stats: Dict[str, int] = Field(default_factory=dict)
    
    # Processing summary
    processing_summary: Dict[str, Any] = Field(default_factory=dict)

class SystemStatus(BaseModel):
    """System status response"""
    status: str
    system_info: Dict[str, Any]
    processing_stats: Dict[str, Any]
    resource_usage: Dict[str, Any]
    index_stats: Dict[str, Any]

class DocumentAnalysis(BaseModel):
    """Document analysis response"""
    doc_id: str
    filename: str
    metadata: Dict[str, Any]
    entity_analysis: Dict[str, Any]
    relationship_analysis: Dict[str, Any]
    quality_metrics: Dict[str, Any]

class WebSocketMessage(BaseModel):
    """WebSocket message model"""
    type: str  # status, progress, result, error
    data: Dict[str, Any]
    timestamp: str

class ConfigurationRequest(BaseModel):
    """Request to update system configuration"""
    extraction_strategy: Optional[str] = Field(default=None)
    response_format: Optional[str] = Field(default=None)
    chunk_size: Optional[int] = Field(default=None, gt=100, le=2000)
    chunk_overlap: Optional[int] = Field(default=None, ge=0, le=500)
    top_k_retrieval: Optional[int] = Field(default=None, gt=0, le=100)
    enable_validation: Optional[bool] = Field(default=None)
    enable_memory: Optional[bool] = Field(default=None)
    confidence_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    custom_prompt_template: Optional[str] = Field(default=None)

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    error_code: str
    details: Optional[Dict[str, Any]] = None
    timestamp: str
