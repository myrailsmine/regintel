"""
Configuration management for the enterprise PDF processing system
"""

import os
from typing import List, Optional
from enum import Enum
from pydantic_settings import BaseSettings
from pydantic import Field

class ProcessingMode(Enum):
    FAST = "fast"
    BALANCED = "balanced"
    COMPREHENSIVE = "comprehensive"
    REAL_TIME = "real_time"

class ExtractionStrategy(Enum):
    PYMUPDF = "pymupdf"
    PDFPLUMBER = "pdfplumber"
    HYBRID = "hybrid"
    OCR = "ocr"

class ResponseFormat(Enum):
    STRUCTURED = "structured"
    NARRATIVE = "narrative"
    TECHNICAL = "technical"
    SUMMARY = "summary"
    CUSTOM = "custom"

class ProcessingConfig:
    """Processing configuration for the PDF system"""
    
    def __init__(self):
        self.mode = ProcessingMode.BALANCED
        self.extraction_strategy = ExtractionStrategy.HYBRID
        self.response_format = ResponseFormat.STRUCTURED
        
        # Processing parameters
        self.max_workers = os.cpu_count() * 2
        self.chunk_size = 512
        self.chunk_overlap = 64
        self.batch_size = 100
        
        # Vector indexing
        self.embedding_model = "all-MiniLM-L6-v2"
        self.vector_dimensions = 384
        self.index_type = "IVF"
        self.nlist = 1000
        
        # Query processing
        self.query_expansion = True
        self.semantic_search_k = 50
        self.rerank_top_k = 20
        self.max_context_length = 8192
        
        # Caching and performance
        self.enable_redis_cache = True
        self.cache_ttl = 3600
        self.enable_incremental_updates = True
        
        # Response generation
        self.response_streaming = True
        self.include_confidence_scores = True
        self.include_source_mapping = True
        
        # Performance optimization
        self.use_gpu = False
        self.memory_limit_gb = 8.0
        self.enable_compression = True

class Settings(BaseSettings):
    """Application settings"""
    
    # Application
    app_name: str = "Enterprise PDF Processor"
    app_version: str = "2.0.0"
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Server
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    workers: int = Field(default=4, env="WORKERS")
    
    # Database
    database_url: str = Field(env="DATABASE_URL")
    redis_url: str = Field(env="REDIS_URL")
    
    # Security
    secret_key: str = Field(env="SECRET_KEY")
    api_key: Optional[str] = Field(default=None, env="API_KEY")
    jwt_expire_minutes: int = Field(default=30, env="JWT_EXPIRE_MINUTES")
    
    # CORS
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    
    # File handling
    max_file_size_mb: int = Field(default=100, env="MAX_FILE_SIZE_MB")
    allowed_file_types: List[str] = Field(default=[".pdf"], env="ALLOWED_FILE_TYPES")
    upload_dir: str = Field(default="./data/uploads", env="UPLOAD_DIR")
    
    # ML Models
    embedding_model: str = Field(default="all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    spacy_model: str = Field(default="en_core_web_lg", env="SPACY_MODEL")
    model_cache_dir: str = Field(default="./models", env="MODEL_CACHE_DIR")
    
    # Processing
    max_concurrent_jobs: int = Field(default=10, env="MAX_CONCURRENT_JOBS")
    processing_timeout: int = Field(default=300, env="PROCESSING_TIMEOUT")
    
    # Rate Limiting
    rate_limit_per_minute: int = Field(default=100, env="RATE_LIMIT_PER_MINUTE")
    rate_limit_burst: int = Field(default=200, env="RATE_LIMIT_BURST")
    
    # Monitoring
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    sentry_dsn: Optional[str] = Field(default=None, env="SENTRY_DSN")
    
    # External APIs (Optional)
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    
    @property
    def processing_config(self) -> ProcessingConfig:
        """Get processing configuration"""
        config = ProcessingConfig()
        config.embedding_model = self.embedding_model
        config.use_gpu = os.getenv("CUDA_VISIBLE_DEVICES") is not None
        return config
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()

# Ensure directories exist
os.makedirs(settings.upload_dir, exist_ok=True)
os.makedirs(settings.model_cache_dir, exist_ok=True)
os.makedirs("./data/vectors", exist_ok=True)
os.makedirs("./data/processed", exist_ok=True)
