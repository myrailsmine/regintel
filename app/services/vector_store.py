"""
Enterprise vector store with hierarchical indexing
"""

import os
import json
import pickle
import asyncio
from typing import Dict, List, Any, Optional, Tuple
import logging

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import networkx as nx
import redis

from app.core.config import settings, ProcessingConfig
from app.services.document_processor import DocumentMetadata

logger = logging.getLogger(__name__)

class EnterpriseVectorStore:
    """Multi-level hierarchical vector indexing system"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.embedder = SentenceTransformer(config.embedding_model)
        
        # Multi-level indices
        self.document_index = None      # Document-level embeddings
        self.page_index = None          # Page-level embeddings  
        self.chunk_index = None         # Chunk-level embeddings
        self.entity_index = None        # Entity-level embeddings
        
        # Metadata stores
        self.doc_metadata = {}
        self.page_metadata = {}
        self.chunk_metadata = {}
        self.entity_metadata = {}
        
        # Knowledge graph
        self.knowledge_graph = nx.Graph()
        
        # Redis client for caching
        self.redis_client = None
        if config.enable_redis_cache:
            try:
                import redis
                self.redis_client = redis.from_url(settings.redis_url)
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}")
        
        # Initialize storage directories
        self.vector_dir = "./data/vectors"
        os.makedirs(self.vector_dir, exist_ok=True)
    
    async def build_hierarchical_index(self, processed_docs: List[DocumentMetadata]) -> None:
        """Build multi-level hierarchical vector indices"""
        
        logger.info(f"Building hierarchical index for {len(processed_docs)} documents")
        
        if not processed_docs:
            logger.warning("No documents provided for indexing")
            return
        
        # Prepare embeddings at each level
        doc_embeddings = []
        page_embeddings = []
        chunk_embeddings = []
        entity_embeddings = []
        
        for doc_meta in processed_docs:
            try:
                # Load document content
                content_data = await self._load_document_content(doc_meta.file_path)
                
                if not content_data:
                    continue
                
                # Document-level embedding
                doc_text = self._create_document_summary(content_data)
                if doc_text:
                    doc_embedding = self.embedder.encode([doc_text])[0]
                    doc_embeddings.append(doc_embedding)
                    self.doc_metadata[doc_meta.doc_id] = {
                        "doc_id": doc_meta.doc_id,
                        "filename": doc_meta.filename,
                        "summary": doc_text[:500],
                        "page_count": doc_meta.page_count,
                        "entities": doc_meta.extracted_entities
                    }
                
                # Page-level embeddings
                for i, text_chunk in enumerate(content_data.get("text_chunks", [])):
                    if not text_chunk.strip():
                        continue
                    
                    page_embedding = self.embedder.encode([text_chunk])[0]
                    page_embeddings.append(page_embedding)
                    
                    page_number = content_data["metadata"][i]["page"]
                    page_id = f"{doc_meta.doc_id}_page_{page_number}"
                    self.page_metadata[page_id] = {
                        "page_id": page_id,
                        "doc_id": doc_meta.doc_id,
                        "page_number": page_number,
                        "summary": text_chunk[:300],
                        "char_count": len(text_chunk)
                    }
                    
                    # Chunk-level embeddings (more granular)
                    chunks = self._create_chunks(text_chunk)
                    for chunk_idx, chunk in enumerate(chunks):
                        if not chunk.strip():
                            continue
                            
                        chunk_embedding = self.embedder.encode([chunk])[0]
                        chunk_embeddings.append(chunk_embedding)
                        
                        chunk_id = f"{page_id}_chunk_{chunk_idx}"
                        self.chunk_metadata[chunk_id] = {
                            "chunk_id": chunk_id,
                            "page_id": page_id,
                            "doc_id": doc_meta.doc_id,
                            "page_number": page_number,
                            "chunk_text": chunk[:200],
                            "full_text": chunk,
                            "position": chunk_idx,
                            "length": len(chunk)
                        }
                
                # Entity-level embeddings
                for entity in doc_meta.extracted_entities:
                    entity_text = f"{entity['text']} {entity.get('context', '')}"
                    if entity_text.strip():
                        entity_embedding = self.embedder.encode([entity_text])[0]
                        entity_embeddings.append(entity_embedding)
                        
                        self.entity_metadata[entity['entity_id']] = {
                            "entity_id": entity['entity_id'],
                            "doc_id": doc_meta.doc_id,
                            "text": entity['text'],
                            "label": entity['label'],
                            "confidence": entity['confidence'],
                            "page_number": entity['page_number']
                        }
                
                # Build knowledge graph
                self._update_knowledge_graph(doc_meta)
                
            except Exception as e:
                logger.error(f"Failed to process document {doc_meta.filename}: {e}")
                continue
        
        # Create FAISS indices
        if doc_embeddings:
            self.document_index = self._create_faiss_index(np.array(doc_embeddings))
            logger.info(f"Built document index with {len(doc_embeddings)} documents")
            
        if page_embeddings:
            self.page_index = self._create_faiss_index(np.array(page_embeddings))
            logger.info(f"Built page index with {len(page_embeddings)} pages")
            
        if chunk_embeddings:
            self.chunk_index = self._create_faiss_index(np.array(chunk_embeddings))
            logger.info(f"Built chunk index with {len(chunk_embeddings)} chunks")
            
        if entity_embeddings:
            self.entity_index = self._create_faiss_index(np.array(entity_embeddings))
            logger.info(f"Built entity index with {len(entity_embeddings)} entities")
        
        # Save indices to disk
        await self._save_indices()
        
        logger.info("Hierarchical index building completed")
    
    def _create_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Create optimized FAISS index"""
        dimension = embeddings.shape[1]
        
        if self.config.index_type == "HNSW":
            index = faiss.IndexHNSWFlat(dimension, 32)
            index.hnsw.efConstruction = 200
            index.hnsw.efSearch = 50
        elif self.config.index_type == "IVF" and embeddings.shape[0] > 1000:
            quantizer = faiss.IndexFlatL2(dimension)
            nlist = min(self.config.nlist, embeddings.shape[0] // 10)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            index.train(embeddings.astype('float32'))
        else:
            index = faiss.IndexFlatL2(dimension)
        
        index.add(embeddings.astype('float32'))
        return index
    
    async def multi_level_search(self, query: str, top_k: int = 50) -> Dict[str, List[Dict]]:
        """Perform hierarchical search across all levels"""
        
        query_embedding = self.embedder.encode([query]).astype('float32')
        results = {}
        
        # Document-level search
        if self.document_index and len(self.doc_metadata) > 0:
            doc_distances, doc_indices = self.document_index.search(query_embedding, k=min(top_k//2, 20))
            results["documents"] = [
                {
                    "metadata": list(self.doc_metadata.values())[idx],
                    "similarity": 1 / (1 + dist),
                    "level": "document"
                }
                for idx, dist in zip(doc_indices[0], doc_distances[0])
                if idx < len(self.doc_metadata) and dist < 10.0  # Filter very distant results
            ]
        
        # Page-level search
        if self.page_index and len(self.page_metadata) > 0:
            page_distances, page_indices = self.page_index.search(query_embedding, k=min(top_k, 30))
            results["pages"] = [
                {
                    "metadata": list(self.page_metadata.values())[idx],
                    "similarity": 1 / (1 + dist),
                    "level": "page"
                }
                for idx, dist in zip(page_indices[0], page_distances[0])
                if idx < len(self.page_metadata) and dist < 10.0
            ]
        
        # Chunk-level search (most granular)
        if self.chunk_index and len(self.chunk_metadata) > 0:
            chunk_distances, chunk_indices = self.chunk_index.search(query_embedding, k=top_k)
            results["chunks"] = [
                {
                    "metadata": list(self.chunk_metadata.values())[idx],
                    "similarity": 1 / (1 + dist),
                    "level": "chunk"
                }
                for idx, dist in zip(chunk_indices[0], chunk_distances[0])
                if idx < len(self.chunk_metadata) and dist < 10.0
            ]
        
        # Entity-level search
        if self.entity_index and len(self.entity_metadata) > 0:
            entity_distances, entity_indices = self.entity_index.search(query_embedding, k=min(top_k//2, 20))
            results["entities"] = [
                {
                    "metadata": list(self.entity_metadata.values())[idx],
                    "similarity": 1 / (1 + dist),
                    "level": "entity"
                }
                for idx, dist in zip(entity_indices[0], entity_distances[0])
                if idx < len(self.entity_metadata) and dist < 10.0
            ]
        
        # Knowledge graph traversal
        graph_paths = self._find_knowledge_graph_paths(query)
        if graph_paths:
            results["graph_paths"] = graph_paths
        
        return results
    
    async def _load_document_content(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Load document content from processed data or extract on-the-fly"""
        # Try to load from cache first
        cache_key = f"content:{hashlib.md5(file_path.encode()).hexdigest()}"
        
        if self.redis_client:
            try:
                cached_content = self.redis_client.get(cache_key)
                if cached_content:
                    return json.loads(cached_content)
            except Exception:
                pass
        
        # Load from file system cache
        cache_file = os.path.join(self.vector_dir, f"{os.path.basename(file_path)}.content.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cached content: {e}")
        
        # Extract content on-the-fly
        try:
            from app.services.document_processor import HybridExtractor
            extractor = HybridExtractor()
            content_data = extractor.extract(file_path)
            
            # Cache the result
            if self.redis_client:
                try:
                    self.redis_client.setex(cache_key, 3600, json.dumps(content_data))
                except Exception:
                    pass
            
            # Save to file system cache
            try:
                with open(cache_file, 'w') as f:
                    json.dump(content_data, f)
            except Exception:
                pass
            
            return content_data
            
        except Exception as e:
            logger.error(f"Failed to extract content from {file_path}: {e}")
            return None
    
    def _create_document_summary(self, content_data: Dict[str, Any]) -> str:
        """Create a summary of the entire document"""
        text_chunks = content_data.get("text_chunks", [])
        if not text_chunks:
            return ""
        
        # Combine first few pages for document-level summary
        summary_text = ""
        char_limit = 1000
        
        for chunk in text_chunks[:5]:  # First 5 pages
            if len(summary_text) + len(chunk) > char_limit:
                remaining = char_limit - len(summary_text)
                summary_text += chunk[:remaining]
                break
            summary_text += chunk + " "
        
        return summary_text.strip()
    
    def _create_chunks(self, text: str) -> List[str]:
        """Create overlapping text chunks"""
        chunk_size = self.config.chunk_size
        chunk_overlap = self.config.chunk_overlap
        
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at word boundary
            if end < len(text):
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - chunk_overlap
            if start >= len(text):
                break
        
        return chunks
    
    def _update_knowledge_graph(self, doc_meta: DocumentMetadata):
        """Build knowledge graph from entities and relationships"""
        # Add entity nodes
        for entity in doc_meta.extracted_entities:
            self.knowledge_graph.add_node(
                entity['entity_id'],
                text=entity['text'],
                label=entity['label'],
                doc_id=doc_meta.doc_id,
                page_number=entity['page_number']
            )
        
        # Add relationship edges
        for relationship in doc_meta.relationships:
            source = relationship['source_entity']
            target = relationship['target_entity']
            
            if source in self.knowledge_graph and target in self.knowledge_graph:
                self.knowledge_graph.add_edge(
                    source, target,
                    relationship_type=relationship['relationship_type'],
                    confidence=relationship['confidence']
                )
    
    def _find_knowledge_graph_paths(self, query: str) -> List[Dict]:
        """Find relevant paths in knowledge graph"""
        if not self.knowledge_graph.nodes:
            return []
        
        # Simple keyword matching for now
        query_words = set(query.lower().split())
        relevant_nodes = []
        
        for node_id, node_data in self.knowledge_graph.nodes(data=True):
            node_text = node_data.get('text', '').lower()
            if any(word in node_text for word in query_words):
                relevant_nodes.append(node_id)
        
        # Find paths between relevant nodes
        paths = []
        for i, node1 in enumerate(relevant_nodes):
            for node2 in relevant_nodes[i+1:]:
                try:
                    if nx.has_path(self.knowledge_graph, node1, node2):
                        path = nx.shortest_path(self.knowledge_graph, node1, node2)
                        if len(path) <= 4:  # Limit path length
                            paths.append({
                                "path": path,
                                "length": len(path),
                                "nodes": [self.knowledge_graph.nodes[node] for node in path]
                            })
                except nx.NetworkXNoPath:
                    continue
        
        return paths[:10]  # Limit to top 10 paths
    
    async def _save_indices(self):
        """Save FAISS indices to disk"""
        try:
            if self.document_index:
                faiss.write_index(self.document_index, os.path.join(self.vector_dir, "document_index.faiss"))
            if self.page_index:
                faiss.write_index(self.page_index, os.path.join(self.vector_dir, "page_index.faiss"))
            if self.chunk_index:
                faiss.write_index(self.chunk_index, os.path.join(self.vector_dir, "chunk_index.faiss"))
            if self.entity_index:
                faiss.write_index(self.entity_index, os.path.join(self.vector_dir, "entity_index.faiss"))
            
            # Save metadata
            with open(os.path.join(self.vector_dir, "metadata.pkl"), 'wb') as f:
                pickle.dump({
                    "doc_metadata": self.doc_metadata,
                    "page_metadata": self.page_metadata,
                    "chunk_metadata": self.chunk_metadata,
                    "entity_metadata": self.entity_metadata
                }, f)
            
            # Save knowledge graph
            nx.write_gpickle(self.knowledge_graph, os.path.join(self.vector_dir, "knowledge_graph.pkl"))
            
            logger.info("Indices saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save indices: {e}")
    
    async def load_indices(self):
        """Load FAISS indices from disk"""
        try:
            # Load indices
            doc_index_path = os.path.join(self.vector_dir, "document_index.faiss")
            if os.path.exists(doc_index_path):
                self.document_index = faiss.read_index(doc_index_path)
            
            page_index_path = os.path.join(self.vector_dir, "page_index.faiss")
            if os.path.exists(page_index_path):
                self.page_index = faiss.read_index(page_index_path)
            
            chunk_index_path = os.path.join(self.vector_dir, "chunk_index.faiss")
            if os.path.exists(chunk_index_path):
                self.chunk_index = faiss.read_index(chunk_index_path)
            
            entity_index_path = os.path.join(self.vector_dir, "entity_index.faiss")
            if os.path.exists(entity_index_path):
                self.entity_index = faiss.read_index(entity_index_path)
            
            # Load metadata
            metadata_path = os.path.join(self.vector_dir, "metadata.pkl")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                    self.doc_metadata = metadata.get("doc_metadata", {})
                    self.page_metadata = metadata.get("page_metadata", {})
                    self.chunk_metadata = metadata.get("chunk_metadata", {})
                    self.entity_metadata = metadata.get("entity_metadata", {})
            
            # Load knowledge graph
            graph_path = os.path.join(self.vector_dir, "knowledge_graph.pkl")
            if os.path.exists(graph_path):
                self.knowledge_graph = nx.read_gpickle(graph_path)
            
            logger.info("Indices loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load indices: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        return {
            "document_count": len(self.doc_metadata),
            "page_count": len(self.page_metadata),
            "chunk_count": len(self.chunk_metadata),
            "entity_count": len(self.entity_metadata),
            "graph_nodes": self.knowledge_graph.number_of_nodes(),
            "graph_edges": self.knowledge_graph.number_of_edges(),
            "indices_built": {
                "documents": self.document_index is not None,
                "pages": self.page_index is not None,
                "chunks": self.chunk_index is not None,
                "entities": self.entity_index is not None
            }
        }
