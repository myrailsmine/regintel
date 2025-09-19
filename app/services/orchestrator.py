"""
Main orchestration system that coordinates all components
"""

import asyncio
import time
import logging
import hashlib
from typing import Dict, List, Any, Optional, AsyncGenerator
from datetime import datetime

from app.core.config import settings, ProcessingConfig
from app.services.document_processor import DocumentProcessor, DocumentMetadata
from app.services.vector_store import EnterpriseVectorStore
from app.services.agents import QueryPlannerAgent, DocumentRetrievalAgent, ReasoningAgent

logger = logging.getLogger(__name__)

class PDFProcessingOrchestrator:
    """Main orchestrator for enterprise-scale PDF processing"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.document_processor = DocumentProcessor(config)
        self.vector_store = EnterpriseVectorStore(config)
        
        # Initialize agents
        self.query_planner = QueryPlannerAgent(config)
        self.document_retrieval_agent = DocumentRetrievalAgent(config)
        self.reasoning_agent = ReasoningAgent(config)
        
        # Processing state
        self.processed_documents = {}
        self.processing_stats = {
            "total_documents": 0,
            "total_queries": 0,
            "average_processing_time": 0.0,
            "cache_hit_rate": 0.0,
            "system_uptime": time.time()
        }
        
        # Background tasks
        self.background_tasks = set()
    
    async def initialize(self):
        """Initialize the orchestrator system"""
        logger.info("Initializing PDF Processing Orchestrator")
        
        # Try to load existing indices
        indices_loaded = await self.vector_store.load_indices()
        if indices_loaded:
            logger.info("Existing vector indices loaded successfully")
        else:
            logger.info("No existing indices found, will build on first ingestion")
        
        logger.info("Orchestrator initialization completed")
    
    async def ingest_documents(self, pdf_paths: List[str]) -> Dict[str, Any]:
        """Ingest large batch of PDFs with distributed processing"""
        
        ingestion_start = time.time()
        logger.info(f"Starting ingestion of {len(pdf_paths)} documents")
        
        try:
            # Validate file paths
            valid_paths = []
            for path in pdf_paths:
                if self._validate_pdf_path(path):
                    valid_paths.append(path)
                else:
                    logger.warning(f"Invalid PDF path: {path}")
            
            if not valid_paths:
                return {
                    "status": "error",
                    "error": "No valid PDF paths provided",
                    "documents_processed": 0
                }
            
            # Process documents in batches
            batch_size = self.config.batch_size
            all_processed_docs = []
            failed_documents = []
            
            for i in range(0, len(valid_paths), batch_size):
                batch = valid_paths[i:i+batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(valid_paths)-1)//batch_size + 1}")
                
                try:
                    batch_docs = await self.document_processor.process_document_batch(batch)
                    all_processed_docs.extend(batch_docs)
                except Exception as e:
                    logger.error(f"Batch processing failed: {e}")
                    failed_documents.extend(batch)
            
            # Build hierarchical vector indices
            if all_processed_docs:
                await self.vector_store.build_hierarchical_index(all_processed_docs)
                
                # Store processed documents
                for doc in all_processed_docs:
                    self.processed_documents[doc.doc_id] = doc
            
            # Update stats
            ingestion_time = time.time() - ingestion_start
            self.processing_stats["total_documents"] = len(all_processed_docs)
            
            logger.info(f"Completed ingestion in {ingestion_time:.2f}s: "
                       f"{len(all_processed_docs)} docs processed")
            
            return {
                "status": "completed",
                "documents_processed": len(all_processed_docs),
                "failed_documents": failed_documents,
                "processing_time": ingestion_time,
                "index_stats": self.vector_store.get_statistics()
            }
            
        except Exception as e:
            logger.error(f"Document ingestion failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "documents_processed": 0,
                "processing_time": time.time() - ingestion_start
            }
    
    async def process_query(self, query: str, user_id: str = "default", 
                          processing_mode: str = "balanced", 
                          max_documents: Optional[int] = None,
                          include_relationships: bool = True) -> Dict[str, Any]:
        """Process complex query across entire document collection"""
        
        query_start = time.time()
        query_id = hashlib.md5(f"{query}{user_id}{time.time()}".encode()).hexdigest()[:8]
        
        logger.info(f"Processing query {query_id}: {query[:100]}...")
        
        try:
            # Step 1: Create intelligent execution plan
            planning_task = {
                "query": query,
                "available_docs": list(self.processed_documents.values()),
                "user_preferences": {
                    "processing_mode": processing_mode,
                    "max_documents": max_documents,
                    "include_relationships": include_relationships
                }
            }
            
            planning_result = await self.query_planner.process(planning_task)
            if planning_result.get("status") == "error":
                raise Exception(f"Query planning failed: {planning_result.get('error')}")
            
            execution_plan = planning_result["execution_plan"]
            query_analysis = planning_result["query_analysis"]
            
            # Step 2: Multi-level hierarchical search
            search_results = await self.vector_store.multi_level_search(
                query, 
                top_k=execution_plan.get("search_parameters", {}).get("top_k", 20)
            )
            
            # Step 3: Document retrieval and ranking
            retrieval_task = {
                "query": query,
                "search_results": search_results,
                "execution_plan": execution_plan,
                "query_analysis": query_analysis
            }
            
            retrieval_result = await self.document_retrieval_agent.process(retrieval_task)
            if retrieval_result.get("status") == "error":
                raise Exception(f"Document retrieval failed: {retrieval_result.get('error')}")
            
            # Step 4: Generate intelligent response
            reasoning_task = {
                "query": query,
                "context_windows": retrieval_result["context_windows"],
                "query_analysis": query_analysis,
                "execution_plan": execution_plan
            }
            
            reasoning_result = await self.reasoning_agent.process(reasoning_task)
            if reasoning_result.get("status") == "error":
                raise Exception(f"Reasoning failed: {reasoning_result.get('error')}")
            
            # Step 5: Compile final response
            processing_time = time.time() - query_start
            
            final_response = {
                "query_id": query_id,
                "query": query,
                "response": reasoning_result["response"],
                "confidence_score": reasoning_result["confidence_score"],
                "processing_time": processing_time,
                
                # Supporting evidence
                "supporting_evidence": self._compile_supporting_evidence(retrieval_result),
                "cross_references": search_results.get("graph_paths", []),
                
                # Document coverage
                "document_coverage": {
                    "documents_analyzed": len(search_results.get("documents", [])),
                    "pages_analyzed": len(search_results.get("pages", [])),
                    "chunks_analyzed": len(search_results.get("chunks", [])),
                    "entities_found": len(search_results.get("entities", []))
                },
                
                # Processing metadata
                "processing_metadata": {
                    "execution_plan": execution_plan,
                    "query_analysis": query_analysis,
                    "agents_used": execution_plan["agents_pipeline"],
                    "search_strategy": execution_plan["strategy"],
                    "retrieval_stats": retrieval_result.get("retrieval_stats", {}),
                    "reasoning_chain": reasoning_result.get("reasoning_chain", [])
                },
                
                # Sources
                "sources": self._compile_sources(retrieval_result)
            }
            
            # Update processing stats
            self.processing_stats["total_queries"] += 1
            self._update_average_processing_time(processing_time)
            
            logger.info(f"Completed query {query_id} in {processing_time:.2f}s")
            
            return final_response
            
        except Exception as e:
            processing_time = time.time() - query_start
            logger.error(f"Query processing failed for {query_id}: {e}")
            return {
                "query_id": query_id,
                "error": str(e),
                "processing_time": processing_time,
                "status": "error"
            }
    
    async def process_query_streaming(self, query: str, user_id: str, websocket) -> AsyncGenerator[Dict[str, Any], None]:
        """Process query with real-time streaming updates"""
        
        query_id = hashlib.md5(f"{query}{user_id}{time.time()}".encode()).hexdigest()[:8]
        
        try:
            # Step 1: Query Planning
            yield {
                "status": "planning",
                "message": "Analyzing query and creating execution plan...",
                "query_id": query_id,
                "progress": 10
            }
            
            planning_task = {
                "query": query,
                "available_docs": list(self.processed_documents.values())
            }
            
            planning_result = await self.query_planner.process(planning_task)
            execution_plan = planning_result["execution_plan"]
            
            yield {
                "status": "planning_complete",
                "message": f"Plan created: {execution_plan['strategy']} with {len(execution_plan['agents_pipeline'])} agents",
                "execution_plan": execution_plan,
                "progress": 25
            }
            
            # Step 2: Document Search
            yield {
                "status": "searching",
                "message": f"Searching across {len(self.processed_documents)} documents...",
                "progress": 40
            }
            
            search_results = await self.vector_store.multi_level_search(query, top_k=50)
            
            yield {
                "status": "search_complete",
                "message": f"Found {len(search_results.get('chunks', []))} relevant chunks",
                "search_stats": {
                    "documents": len(search_results.get("documents", [])),
                    "pages": len(search_results.get("pages", [])),
                    "chunks": len(search_results.get("chunks", [])),
                    "entities": len(search_results.get("entities", []))
                },
                "progress": 60
            }
            
            # Step 3: Content Retrieval
            yield {
                "status": "retrieving",
                "message": "Retrieving and ranking relevant content...",
                "progress": 70
            }
            
            retrieval_task = {
                "query": query,
                "search_results": search_results,
                "execution_plan": execution_plan,
                "query_analysis": planning_result["query_analysis"]
            }
            
            retrieval_result = await self.document_retrieval_agent.process(retrieval_task)
            
            # Step 4: Response Generation
            yield {
                "status": "reasoning",
                "message": "Generating intelligent response...",
                "progress": 85
            }
            
            reasoning_task = {
                "query": query,
                "context_windows": retrieval_result["context_windows"],
                "query_analysis": planning_result["query_analysis"],
                "execution_plan": execution_plan
            }
            
            reasoning_result = await self.reasoning_agent.process(reasoning_task)
            
            # Step 5: Final Result
            yield {
                "status": "completed",
                "message": "Processing completed successfully!",
                "result": {
                    "query_id": query_id,
                    "response": reasoning_result["response"],
                    "confidence_score": reasoning_result["confidence_score"],
                    "supporting_evidence": self._compile_supporting_evidence(retrieval_result),
                    "document_coverage": {
                        "documents_analyzed": len(search_results.get("documents", [])),
                        "pages_analyzed": len(search_results.get("pages", [])),
                        "chunks_analyzed": len(search_results.get("chunks", []))
                    },
                    "sources": self._compile_sources(retrieval_result)
                },
                "progress": 100
            }
            
        except Exception as e:
            yield {
                "status": "error",
                "error": str(e),
                "query_id": query_id
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "status": "operational",
            "system_info": {
                "documents_indexed": len(self.processed_documents),
                "uptime_seconds": time.time() - self.processing_stats["system_uptime"],
                "vector_store_stats": self.vector_store.get_statistics()
            },
            "processing_stats": self.processing_stats.copy(),
            "resource_usage": self._get_resource_usage(),
            "agent_stats": {
                "query_planner": self.query_planner.processing_stats,
                "document_retrieval": self.document_retrieval_agent.processing_stats,
                "reasoning": self.reasoning_agent.processing_stats
            }
        }
    
    async def cleanup(self):
        """Cleanup system resources"""
        logger.info("Cleaning up orchestrator resources")
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        logger.info("Orchestrator cleanup completed")
    
    def _validate_pdf_path(self, path: str) -> bool:
        """Validate PDF file path"""
        import os
        return (
            os.path.exists(path) and
            os.path.isfile(path) and
            path.lower().endswith('.pdf') and
            os.path.getsize(path) > 0
        )
    
    def _compile_supporting_evidence(self, retrieval_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Compile supporting evidence from retrieval results"""
        evidence = []
        
        context_windows = retrieval_result.get("context_windows", [])
        for window in context_windows[:5]:  # Top 5 pieces of evidence
            for source in window.get("sources", []):
                evidence.append({
                    "source": source.get("doc_id", "Unknown"),
                    "page": source.get("page_number", "Unknown"),
                    "content": source.get("chunk_text", source.get("summary", ""))[:200] + "...",
                    "relevance": source.get("enhanced_similarity", source.get("similarity", 0))
                })
        
        return evidence
    
    def _compile_sources(self, retrieval_result: Dict[str, Any]) -> List[str]:
        """Compile source references"""
        sources = set()
        
        context_windows = retrieval_result.get("context_windows", [])
        for window in context_windows:
            for source in window.get("sources", []):
                doc_id = source.get("doc_id", "Unknown")
                page_num = source.get("page_number", "")
                if page_num:
                    sources.add(f"{doc_id}:Page {page_num}")
                else:
                    sources.add(doc_id)
        
        return sorted(list(sources))
    
    def _update_average_processing_time(self, processing_time: float):
        """Update average processing time"""
        total_queries = self.processing_stats["total_queries"]
        if total_queries == 1:
            self.processing_stats["average_processing_time"] = processing_time
        else:
            current_avg = self.processing_stats["average_processing_time"]
            new_avg = ((current_avg * (total_queries - 1)) + processing_time) / total_queries
            self.processing_stats["average_processing_time"] = new_avg
    
    def _get_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage"""
        try:
            import psutil
            return {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent,
                "memory_used_gb": psutil.virtual_memory().used / (1024**3)
            }
        except ImportError:
            return {
                "cpu_percent": 0,
                "memory_percent": 0,
                "disk_percent": 0,
                "memory_used_gb": 0
            }
