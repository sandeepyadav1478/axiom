"""
Hybrid Retrieval System combining Vector Search + Graph Traversal.

Architecture:
1. Vector Search: ChromaDB semantic similarity
2. Graph Enhancement: Neo4j relationship context
3. Reranking: Relevance scoring and filtering
4. DSPy Integration: Optimized retrieval chains
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

import numpy as np

from ...database.vector_store import get_vector_store, VectorStoreType
from .document_processor import DocumentChunk
from .embedding_service import EmbeddingService
from .graph_enhancer import GraphEnhancer, GraphContext

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """A single retrieval result with context."""
    
    chunk: DocumentChunk
    score: float
    rank: int
    
    # Context
    graph_context: Optional[GraphContext] = None
    rerank_score: Optional[float] = None
    
    # Metadata
    retrieval_method: str = "hybrid"  # vector, graph, or hybrid
    retrieved_at: datetime = field(default_factory=datetime.now)


@dataclass
class RetrievalContext:
    """Complete retrieval context for a query."""
    
    query: str
    results: List[RetrievalResult]
    
    # Statistics
    total_retrieved: int
    vector_results: int
    graph_enhanced: int
    reranked: bool
    
    # Performance
    retrieval_time_ms: float
    embedding_time_ms: float
    graph_time_ms: float
    
    # Query metadata
    query_embedding: Optional[List[float]] = None
    extracted_entities: Dict[str, List[str]] = field(default_factory=dict)
    
    retrieved_at: datetime = field(default_factory=datetime.now)


class HybridRetriever:
    """
    Hybrid retrieval combining vector search and graph traversal.
    
    Pipeline:
    1. Query embedding generation
    2. Vector similarity search (ChromaDB)
    3. Graph context enhancement (Neo4j)
    4. Reranking and filtering
    5. Return top-k results
    """
    
    def __init__(
        self,
        collection_name: str = "ma_documents",
        embedding_service: Optional[EmbeddingService] = None,
        graph_enhancer: Optional[GraphEnhancer] = None,
        vector_store_type: VectorStoreType = VectorStoreType.CHROMA,
        top_k: int = 10,
        similarity_threshold: float = 0.7,
        enable_reranking: bool = True,
        graph_weight: float = 0.3
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            collection_name: Vector store collection name
            embedding_service: Embedding service instance
            graph_enhancer: Graph enhancer instance
            vector_store_type: Type of vector store
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score
            enable_reranking: Enable result reranking
            graph_weight: Weight for graph-based scoring (0-1)
        """
        self.collection_name = collection_name
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.enable_reranking = enable_reranking
        self.graph_weight = graph_weight
        
        # Initialize components
        self.embedding_service = embedding_service or EmbeddingService()
        
        self.vector_store = get_vector_store(store_type=vector_store_type)
        
        # Ensure collection exists
        try:
            self.vector_store.create_collection(
                name=collection_name,
                dimension=self.embedding_service.dimension
            )
        except Exception as e:
            logger.info(f"Collection {collection_name} may already exist: {e}")
        
        # Graph enhancer (optional)
        self.graph_enhancer = graph_enhancer
        if graph_enhancer is None:
            try:
                self.graph_enhancer = GraphEnhancer()
            except Exception as e:
                logger.warning(f"Graph enhancer not available: {e}")
        
        logger.info(f"Initialized HybridRetriever with collection '{collection_name}'")
    
    def index_chunks(
        self,
        chunks: List[DocumentChunk],
        batch_size: int = 100
    ) -> int:
        """
        Index document chunks in vector store.
        
        Args:
            chunks: List of document chunks
            batch_size: Batch size for indexing
            
        Returns:
            Number of chunks indexed
        """
        if not chunks:
            return 0
        
        logger.info(f"Indexing {len(chunks)} chunks...")
        
        indexed_count = 0
        
        for batch_start in range(0, len(chunks), batch_size):
            batch_end = min(batch_start + batch_size, len(chunks))
            batch = chunks[batch_start:batch_end]
            
            # Generate embeddings if not present
            texts_to_embed = []
            chunks_to_embed = []
            
            for chunk in batch:
                if chunk.embedding is None:
                    texts_to_embed.append(chunk.text)
                    chunks_to_embed.append(chunk)
            
            if texts_to_embed:
                embeddings = self.embedding_service.embed_texts(texts_to_embed)
                for chunk, embedding in zip(chunks_to_embed, embeddings):
                    chunk.embedding = embedding
                    chunk.embedding_model = self.embedding_service.model
            
            # Prepare data for vector store
            vectors = [chunk.embedding for chunk in batch]
            ids = [chunk.chunk_id for chunk in batch]
            metadata = [
                {
                    "document_id": chunk.document_id,
                    "document_name": chunk.document_name,
                    "chunk_index": chunk.chunk_index,
                    "page_number": chunk.page_number,
                    "content": chunk.text[:1000],  # Store preview
                    "companies": ",".join(chunk.companies_mentioned[:5]),
                    **chunk.metadata
                }
                for chunk in batch
            ]
            
            # Upsert to vector store
            try:
                self.vector_store.upsert(
                    collection_name=self.collection_name,
                    vectors=vectors,
                    ids=ids,
                    metadata=metadata
                )
                indexed_count += len(batch)
                logger.info(f"Indexed batch {batch_start}-{batch_end} ({indexed_count} total)")
                
            except Exception as e:
                logger.error(f"Failed to index batch: {e}")
        
        logger.info(f"Successfully indexed {indexed_count} chunks")
        return indexed_count
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        enable_graph: bool = True
    ) -> RetrievalContext:
        """
        Retrieve relevant documents for query.
        
        Args:
            query: Search query
            top_k: Number of results (overrides default)
            filters: Metadata filters
            enable_graph: Enable graph enhancement
            
        Returns:
            Retrieval context with results
        """
        import time
        
        start_time = time.time()
        k = top_k or self.top_k
        
        # Generate query embedding
        embed_start = time.time()
        query_embedding = self.embedding_service.embed_text(query)
        embedding_time = (time.time() - embed_start) * 1000
        
        # Vector search
        vector_results = self.vector_store.search(
            collection_name=self.collection_name,
            query_vector=np.array(query_embedding),
            top_k=k * 2,  # Retrieve more for reranking
            filter=filters
        )
        
        logger.info(f"Retrieved {len(vector_results)} vector results")
        
        # Convert to retrieval results
        results = []
        for i, result in enumerate(vector_results):
            if result['score'] < self.similarity_threshold:
                continue
            
            # Reconstruct chunk from metadata
            chunk = self._reconstruct_chunk_from_metadata(result)
            
            retrieval_result = RetrievalResult(
                chunk=chunk,
                score=result['score'],
                rank=i,
                retrieval_method="vector"
            )
            
            results.append(retrieval_result)
        
        # Graph enhancement
        graph_time = 0.0
        if enable_graph and self.graph_enhancer and results:
            graph_start = time.time()
            results = self._enhance_with_graph(results)
            graph_time = (time.time() - graph_start) * 1000
        
        # Reranking
        if self.enable_reranking and results:
            results = self._rerank_results(results, query)
        
        # Take top-k after reranking
        results = results[:k]
        
        retrieval_time = (time.time() - start_time) * 1000
        
        # Create context
        context = RetrievalContext(
            query=query,
            results=results,
            total_retrieved=len(results),
            vector_results=len(vector_results),
            graph_enhanced=sum(1 for r in results if r.graph_context is not None),
            reranked=self.enable_reranking,
            retrieval_time_ms=retrieval_time,
            embedding_time_ms=embedding_time,
            graph_time_ms=graph_time,
            query_embedding=query_embedding
        )
        
        logger.info(
            f"Retrieved {len(results)} results in {retrieval_time:.1f}ms "
            f"(embed: {embedding_time:.1f}ms, graph: {graph_time:.1f}ms)"
        )
        
        return context
    
    def _enhance_with_graph(
        self,
        results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Enhance results with graph context."""
        
        for result in results:
            chunk = result.chunk
            
            # Get graph context for companies in chunk
            if chunk.companies_mentioned:
                graph_context = self.graph_enhancer.enhance_with_graph_context(
                    text=chunk.text,
                    companies=chunk.companies_mentioned,
                    metadata=chunk.metadata
                )
                
                result.graph_context = graph_context
                
                # Boost score based on graph relationships
                if graph_context.relationship_count > 0:
                    graph_boost = min(0.2, graph_context.relationship_count * 0.02)
                    result.score = min(1.0, result.score + graph_boost)
        
        return results
    
    def _rerank_results(
        self,
        results: List[RetrievalResult],
        query: str
    ) -> List[RetrievalResult]:
        """Rerank results using hybrid scoring."""
        
        for result in results:
            # Combine vector score with graph score
            vector_score = result.score
            
            graph_score = 0.0
            if result.graph_context and result.graph_context.relationship_count > 0:
                # Normalize graph score
                graph_score = min(1.0, result.graph_context.relationship_count / 20.0)
            
            # Weighted combination
            combined_score = (
                (1 - self.graph_weight) * vector_score +
                self.graph_weight * graph_score
            )
            
            result.rerank_score = combined_score
        
        # Sort by rerank score
        results.sort(key=lambda r: r.rerank_score, reverse=True)
        
        # Update ranks
        for i, result in enumerate(results):
            result.rank = i
        
        return results
    
    def _reconstruct_chunk_from_metadata(
        self,
        result: Dict[str, Any]
    ) -> DocumentChunk:
        """Reconstruct document chunk from vector store metadata."""
        
        metadata = result.get('metadata', {})
        
        # Extract companies
        companies_str = metadata.get('companies', '')
        companies = [c.strip() for c in companies_str.split(',') if c.strip()]
        
        chunk = DocumentChunk(
            chunk_id=result['id'],
            text=metadata.get('content', ''),
            document_id=metadata.get('document_id', ''),
            document_name=metadata.get('document_name', ''),
            chunk_index=metadata.get('chunk_index', 0),
            page_number=metadata.get('page_number'),
            companies_mentioned=companies,
            metadata=metadata
        )
        
        return chunk
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        
        collections = self.vector_store.list_collections()
        
        return {
            "collection_name": self.collection_name,
            "vector_store_type": type(self.vector_store).__name__,
            "total_collections": len(collections),
            "embedding_model": self.embedding_service.model,
            "embedding_dimension": self.embedding_service.dimension,
            "cached_embeddings": self.embedding_service.get_cache_size(),
            "graph_enabled": self.graph_enhancer is not None,
            "reranking_enabled": self.enable_reranking,
            "similarity_threshold": self.similarity_threshold,
            "top_k": self.top_k
        }


__all__ = ["HybridRetriever", "RetrievalContext", "RetrievalResult"]