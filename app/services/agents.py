"""
Multi-agent processing system for sophisticated PDF analysis
"""

import asyncio
import json
import time
import hashlib
from typing import Dict, List, Any, Optional, AsyncGenerator
from abc import ABC, abstractmethod
import logging

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.core.config import ProcessingConfig
from app.services.vector_store import EnterpriseVectorStore
from app.services.document_processor import DocumentProcessor

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """Abstract base class for processing agents"""
    
    def __init__(self, name: str, role: str, config: ProcessingConfig):
        self.name = name
        self.role = role
        self.config = config
        self.processing_stats = {
            "tasks_processed": 0,
            "total_processing_time": 0.0,
            "average_processing_time": 0.0,
            "success_rate": 1.0
        }
    
    @abstractmethod
    async def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process the given task"""
        pass
    
    def update_stats(self, processing_time: float, success: bool):
        """Update agent processing statistics"""
        self.processing_stats["tasks_processed"] += 1
        self.processing_stats["total_processing_time"] += processing_time
        
        # Update average
        self.processing_stats["average_processing_time"] = (
            self.processing_stats["total_processing_time"] / 
            self.processing_stats["tasks_processed"]
        )
        
        # Update success rate
        if not success:
            current_success_count = self.processing_stats["success_rate"] * (self.processing_stats["tasks_processed"] - 1)
            self.processing_stats["success_rate"] = current_success_count / self.processing_stats["tasks_processed"]

class QueryPlannerAgent(BaseAgent):
    """Intelligent query planning and decomposition"""
    
    def __init__(self, config: ProcessingConfig):
        super().__init__("QueryPlanner", "Plan and coordinate query execution", config)
    
    async def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Create intelligent execution plan based on query complexity"""
        start_time = time.time()
        
        try:
            query = task.get("query", "")
            available_docs = task.get("available_docs", [])
            
            # Analyze query
            query_analysis = await self._analyze_query(query)
            
            # Create execution plan
            execution_plan = await self._create_execution_plan(query_analysis, available_docs)
            
            # Estimate resources needed
            resource_estimation = self._estimate_resources(execution_plan, available_docs)
            
            result = {
                "query_id": hashlib.md5(query.encode()).hexdigest()[:8],
                "original_query": query,
                "query_analysis": query_analysis,
                "execution_plan": execution_plan,
                "resource_estimation": resource_estimation,
                "status": "completed"
            }
            
            processing_time = time.time() - start_time
            self.update_stats(processing_time, True)
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.update_stats(processing_time, False)
            logger.error(f"Query planning failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Deep query analysis to determine processing approach"""
        
        query_lower = query.lower()
        
        # Classify complexity
        complexity_indicators = {
            "simple": ["what", "who", "when", "where", "define", "show"],
            "complex": ["analyze", "compare", "evaluate", "relationship", "impact", "trend", "explain"],
            "multi_domain": ["across", "between", "correlate", "comprehensive", "holistic", "integrate", "all"]
        }
        
        complexity = "simple"
        for level, indicators in complexity_indicators.items():
            if any(indicator in query_lower for indicator in indicators):
                complexity = level
        
        # Identify domain focus
        domain_indicators = {
            "financial": ["revenue", "profit", "cost", "financial", "budget", "investment", "girr", "capital", "risk"],
            "legal": ["contract", "agreement", "clause", "compliance", "regulation", "policy", "legal"],
            "technical": ["system", "architecture", "implementation", "technical", "specification", "calculate"],
            "operational": ["process", "procedure", "workflow", "operation", "management"]
        }
        
        detected_domains = []
        for domain, indicators in domain_indicators.items():
            if any(indicator in query_lower for indicator in indicators):
                detected_domains.append(domain)
        
        # Identify required relationship types
        relationship_types = []
        if any(word in query_lower for word in ["relationship", "connection", "link", "relate"]):
            relationship_types.append("entity_relationships")
        if any(word in query_lower for word in ["across", "between", "multiple", "all"]):
            relationship_types.append("cross_document")
        if any(word in query_lower for word in ["timeline", "sequence", "chronological", "over time"]):
            relationship_types.append("temporal")
        
        return {
            "complexity": complexity,
            "domains": detected_domains or ["general"],
            "relationship_types": relationship_types,
            "requires_cross_document": any(word in query_lower for word in ["across", "all", "comprehensive"]),
            "requires_temporal_analysis": any(word in query_lower for word in ["trend", "change", "over time"]),
            "requires_quantitative": any(word in query_lower for word in ["how much", "calculate", "amount", "percentage"]),
            "query_intent": self._classify_intent(query_lower),
            "key_terms": self._extract_key_terms(query)
        }
    
    def _classify_intent(self, query_lower: str) -> str:
        """Classify query intent"""
        if any(word in query_lower for word in ["explain", "describe", "what is"]):
            return "explanation"
        elif any(word in query_lower for word in ["compare", "difference", "vs", "versus"]):
            return "comparison"
        elif any(word in query_lower for word in ["calculate", "compute", "formula"]):
            return "calculation"
        elif any(word in query_lower for word in ["find", "search", "show", "list"]):
            return "retrieval"
        elif any(word in query_lower for word in ["analyze", "assessment", "evaluation"]):
            return "analysis"
        else:
            return "general"
    
    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract key terms from query"""
        # Simple keyword extraction - could be enhanced with NLP
        import re
        words = re.findall(r'\b[A-Za-z]{3,}\b', query)
        # Filter out common words
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'man', 'she', 'use', 'way', 'her', 'oil', 'sit', 'set', 'run', 'eat', 'far', 'sea', 'eye'}
        return [word.lower() for word in words if word.lower() not in stop_words]
    
    async def _create_execution_plan(self, query_analysis: Dict[str, Any], available_docs: List) -> Dict[str, Any]:
        """Create execution plan based on analysis"""
        
        if query_analysis["complexity"] == "simple":
            agents_pipeline = ["DocumentRetrievalAgent", "ReasoningAgent"]
            strategy = "direct_retrieval"
        elif query_analysis["complexity"] == "complex":
            agents_pipeline = ["DocumentRetrievalAgent", "EntityAgent", "ReasoningAgent", "ValidationAgent"]
            strategy = "enhanced_analysis"
        else:  # multi_domain
            agents_pipeline = ["DocumentRetrievalAgent", "EntityAgent", "RelationshipAgent", "CrossReferenceAgent", "SynthesisAgent", "ValidationAgent"]
            strategy = "comprehensive_analysis"
        
        # Add domain-specific agents
        if "financial" in query_analysis["domains"]:
            agents_pipeline.insert(-1, "FinancialAnalysisAgent")
        if "legal" in query_analysis["domains"]:
            agents_pipeline.insert(-1, "LegalAnalysisAgent")
        
        return {
            "strategy": strategy,
            "agents_pipeline": agents_pipeline,
            "parallel_processing": len(available_docs) > 50,
            "estimated_processing_time": self._estimate_processing_time(query_analysis, available_docs),
            "sub_queries": await self._decompose_query(query_analysis),
            "search_parameters": {
                "top_k": 50 if query_analysis["complexity"] == "multi_domain" else 20,
                "similarity_threshold": 0.3,
                "enable_reranking": query_analysis["complexity"] != "simple"
            }
        }
    
    def _estimate_resources(self, execution_plan: Dict[str, Any], available_docs: List) -> Dict[str, Any]:
        """Estimate computational resources needed"""
        doc_count = len(available_docs)
        agent_count = len(execution_plan["agents_pipeline"])
        
        return {
            "estimated_memory_gb": min(8.0, doc_count * 0.01 + agent_count * 0.5),
            "estimated_cpu_cores": min(self.config.max_workers, agent_count + 2),
            "estimated_processing_time_seconds": execution_plan["estimated_processing_time"],
            "priority": "high" if execution_plan["strategy"] == "comprehensive_analysis" else "normal"
        }
    
    def _estimate_processing_time(self, query_analysis: Dict[str, Any], available_docs: List) -> float:
        """Estimate processing time"""
        base_time = 2.0  # Base processing time
        
        # Add time based on complexity
        if query_analysis["complexity"] == "complex":
            base_time += 2.0
        elif query_analysis["complexity"] == "multi_domain":
            base_time += 4.0
        
        # Add time based on document count
        doc_time = min(len(available_docs) * 0.1, 10.0)
        
        # Add time for cross-document analysis
        if query_analysis["requires_cross_document"]:
            base_time += 3.0
        
        return base_time + doc_time
    
    async def _decompose_query(self, query_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Break complex queries into sub-queries"""
        if query_analysis["complexity"] == "simple":
            return [{"sub_query": query_analysis["query"], "priority": 1, "type": "direct"}]
        
        sub_queries = []
        
        # Domain-specific decomposition
        for domain in query_analysis["domains"]:
            sub_queries.append({
                "sub_query": f"Extract {domain} information",
                "priority": 1,
                "type": "domain_extraction",
                "domain": domain
            })
        
        # Relationship extraction
        if query_analysis["relationship_types"]:
            sub_queries.append({
                "sub_query": "Extract entity relationships",
                "priority": 2,
                "type": "relationship_extraction"
            })
        
        # Synthesis
        sub_queries.append({
            "sub_query": "Synthesize findings",
            "priority": 3,
            "type": "synthesis"
        })
        
        return sub_queries

class DocumentRetrievalAgent(BaseAgent):
    """Advanced document retrieval and ranking"""
    
    def __init__(self, config: ProcessingConfig):
        super().__init__("DocumentRetrieval", "Retrieve and rank relevant content", config)
    
    async def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve and rerank most relevant document sections"""
        start_time = time.time()
        
        try:
            query = task.get("query", "")
            search_results = task.get("search_results", {})
            execution_plan = task.get("execution_plan", {})
            
            # Perform multi-level retrieval
            retrieval_results = await self._multi_level_retrieval(query, search_results, execution_plan)
            
            # Rerank results
            reranked_results = await self._rerank_results(query, retrieval_results)
            
            # Build context windows
            context_windows = self._build_context_windows(reranked_results)
            
            result = {
                "retrieval_results": retrieval_results,
                "reranked_results": reranked_results,
                "context_windows": context_windows,
                "retrieval_stats": {
                    "total_chunks_analyzed": sum(len(results) for results in search_results.values()),
                    "documents_covered": len(set(r.get("metadata", {}).get("doc_id") for r in reranked_results)),
                    "average_relevance": np.mean([r.get("similarity", 0) for r in reranked_results[:10]]) if reranked_results else 0
                },
                "status": "completed"
            }
            
            processing_time = time.time() - start_time
            self.update_stats(processing_time, True)
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.update_stats(processing_time, False)
            logger.error(f"Document retrieval failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _multi_level_retrieval(self, query: str, search_results: Dict[str, Any], execution_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Perform retrieval across multiple levels"""
        top_k = execution_plan.get("search_parameters", {}).get("top_k", 20)
        
        # Combine results from different levels
        combined_results = []
        
        # Prioritize chunk-level results (most granular)
        chunks = search_results.get("chunks", [])
        combined_results.extend(chunks[:top_k//2])
        
        # Add page-level results
        pages = search_results.get("pages", [])
        combined_results.extend(pages[:top_k//4])
        
        # Add document-level results
        documents = search_results.get("documents", [])
        combined_results.extend(documents[:top_k//4])
        
        # Add entity results if available
        entities = search_results.get("entities", [])
        combined_results.extend(entities[:top_k//4])
        
        return {
            "combined_results": combined_results[:top_k],
            "level_breakdown": {
                "chunks": len(chunks),
                "pages": len(pages),
                "documents": len(documents),
                "entities": len(entities)
            }
        }
    
    async def _rerank_results(self, query: str, retrieval_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rerank results using advanced scoring"""
        results = retrieval_results.get("combined_results", [])
        
        if not results:
            return []
        
        # Calculate enhanced similarity scores
        for result in results:
            # Original similarity score
            base_similarity = result.get("similarity", 0.0)
            
            # Add level-based boost
            level_boost = {
                "chunk": 1.0,
                "page": 0.9,
                "document": 0.8,
                "entity": 0.7
            }.get(result.get("level", "chunk"), 1.0)
            
            # Add freshness boost (if metadata available)
            freshness_boost = 1.0  # Could be enhanced with document date analysis
            
            # Calculate final score
            final_score = base_similarity * level_boost * freshness_boost
            result["enhanced_similarity"] = final_score
        
        # Sort by enhanced similarity
        reranked = sorted(results, key=lambda x: x.get("enhanced_similarity", 0), reverse=True)
        
        return reranked
    
    def _build_context_windows(self, reranked_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build context windows for LLM processing"""
        context_windows = []
        current_window = {"content": "", "sources": [], "token_count": 0}
        max_tokens = self.config.max_context_length - 1000  # Reserve for prompt
        
        for result in reranked_results:
            metadata = result.get("metadata", {})
            content = metadata.get("full_text") or metadata.get("chunk_text") or metadata.get("summary", "")
            
            # Estimate token count (rough approximation)
            estimated_tokens = len(content.split()) * 1.3
            
            if current_window["token_count"] + estimated_tokens > max_tokens:
                if current_window["content"]:
                    context_windows.append(current_window.copy())
                current_window = {"content": content, "sources": [metadata], "token_count": estimated_tokens}
            else:
                current_window["content"] += "\n\n" + content
                current_window["sources"].append(metadata)
                current_window["token_count"] += estimated_tokens
        
        # Add final window
        if current_window["content"]:
            context_windows.append(current_window)
        
        return context_windows

class ReasoningAgent(BaseAgent):
    """Advanced reasoning and response generation"""
    
    def __init__(self, config: ProcessingConfig):
        super().__init__("Reasoning", "Generate intelligent responses", config)
    
    async def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Generate sophisticated response based on retrieved content"""
        start_time = time.time()
        
        try:
            query = task.get("query", "")
            context_windows = task.get("context_windows", [])
            query_analysis = task.get("query_analysis", {})
            
            # Generate response based on query type
            if query_analysis.get("query_intent") == "explanation":
                response = await self._generate_explanation_response(query, context_windows, query_analysis)
            elif query_analysis.get("query_intent") == "calculation":
                response = await self._generate_calculation_response(query, context_windows, query_analysis)
            elif query_analysis.get("query_intent") == "comparison":
                response = await self._generate_comparison_response(query, context_windows, query_analysis)
            else:
                response = await self._generate_general_response(query, context_windows, query_analysis)
            
            # Add confidence scoring
            confidence_score = self._calculate_confidence(response, context_windows)
            
            result = {
                "response": response,
                "confidence_score": confidence_score,
                "response_type": query_analysis.get("query_intent", "general"),
                "sources_used": len(context_windows),
                "reasoning_chain": await self._generate_reasoning_chain(query, context_windows),
                "status": "completed"
            }
            
            processing_time = time.time() - start_time
            self.update_stats(processing_time, True)
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.update_stats(processing_time, False)
            logger.error(f"Reasoning failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _generate_explanation_response(self, query: str, context_windows: List[Dict], query_analysis: Dict) -> str:
        """Generate detailed explanation response"""
        
        # Combine relevant content
        combined_content = ""
        key_sources = []
        
        for window in context_windows[:3]:  # Use top 3 context windows
            combined_content += window["content"] + "\n\n"
            key_sources.extend(window["sources"][:2])  # Top 2 sources per window
        
        # Generate structured explanation
        domains = query_analysis.get("domains", ["general"])
        primary_domain = domains[0] if domains else "general"
        
        if primary_domain == "financial":
            response = self._generate_financial_explanation(query, combined_content, key_sources)
        elif primary_domain == "legal":
            response = self._generate_legal_explanation(query, combined_content, key_sources)
        elif primary_domain == "technical":
            response = self._generate_technical_explanation(query, combined_content, key_sources)
        else:
            response = self._generate_general_explanation(query, combined_content, key_sources)
        
        return response
    
    def _generate_financial_explanation(self, query: str, content: str, sources: List[Dict]) -> str:
        """Generate financial domain explanation"""
        
        # Extract key financial concepts from content
        key_concepts = self._extract_financial_concepts(content)
        
        response = f"# Financial Analysis: {query}\n\n"
        response += "## Executive Summary\n"
        response += f"Based on analysis of {len(sources)} documents, here are the key findings:\n\n"
        
        if key_concepts:
            response += "## Key Financial Concepts\n"
            for concept in key_concepts[:5]:
                response += f"• **{concept['term']}**: {concept['description']}\n"
            response += "\n"
        
        response += "## Detailed Analysis\n"
        response += f"The analysis of your query '{query}' reveals several important aspects:\n\n"
        
        # Add content-based analysis
        if "calculation" in query.lower() or "formula" in query.lower():
            response += self._extract_calculation_methodology(content)
        
        response += "\n## Supporting Evidence\n"
        for i, source in enumerate(sources[:3], 1):
            doc_name = source.get("doc_id", f"Document {i}")
            page_num = source.get("page_number", "Unknown")
            response += f"{i}. **{doc_name}** (Page {page_num})\n"
        
        return response
    
    def _generate_technical_explanation(self, query: str, content: str, sources: List[Dict]) -> str:
        """Generate technical domain explanation"""
        
        response = f"# Technical Analysis: {query}\n\n"
        response += "## Overview\n"
        response += f"This analysis covers the technical aspects of your query based on {len(sources)} source documents.\n\n"
        
        # Extract technical procedures/formulas
        if "calculate" in query.lower() or "formula" in query.lower():
            response += "## Calculation Methodology\n"
            methodology = self._extract_calculation_methodology(content)
            response += methodology + "\n\n"
        
        response += "## Implementation Details\n"
        # Add implementation guidance based on content
        response += "Based on the documentation, here are the key implementation steps:\n\n"
        
        steps = self._extract_procedural_steps(content)
        for i, step in enumerate(steps, 1):
            response += f"{i}. {step}\n"
        
        response += "\n## Technical References\n"
        for source in sources[:3]:
            doc_name = source.get("doc_id", "Unknown Document")
            response += f"• {doc_name}\n"
        
        return response
    
    def _generate_general_explanation(self, query: str, content: str, sources: List[Dict]) -> str:
        """Generate general explanation"""
        
        response = f"# Analysis: {query}\n\n"
        response += "## Summary\n"
        response += f"Based on analysis of the available documents, here's what I found regarding your query:\n\n"
        
        # Extract key points from content
        key_points = self._extract_key_points(content, query)
        
        if key_points:
            response += "## Key Findings\n"
            for i, point in enumerate(key_points[:5], 1):
                response += f"{i}. {point}\n"
            response += "\n"
        
        response += "## Detailed Information\n"
        response += "The documents provide the following relevant information:\n\n"
        
        # Add summarized content
        summary = self._summarize_content(content, 300)  # 300 word summary
        response += summary + "\n\n"
        
        response += "## Sources\n"
        for source in sources[:3]:
            doc_name = source.get("doc_id", "Unknown Document")
            page_num = source.get("page_number", "")
            page_info = f" (Page {page_num})" if page_num else ""
            response += f"• {doc_name}{page_info}\n"
        
        return response
