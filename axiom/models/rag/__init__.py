"""
Production RAG System for M&A Intelligence.

Architecture:
- Document Ingestion: PDF/DOCX/TXT parsing with intelligent chunking
- Vector Store: ChromaDB for semantic search
- Graph Enhancement: Neo4j for relationship context
- Retrieval: DSPy-optimized hybrid retrieval (vector + graph)
- Generation: Claude API for high-quality responses
- Orchestration: Complete pipeline with monitoring
"""

from .document_processor import DocumentProcessor, DocumentChunk
from .embedding_service import EmbeddingService
from .hybrid_retriever import HybridRetriever, RetrievalContext
from .rag_pipeline import RAGPipeline, RAGConfig, RAGResponse
from .graph_enhancer import GraphEnhancer

__all__ = [
    "DocumentProcessor",
    "DocumentChunk",
    "EmbeddingService",
    "HybridRetriever",
    "RetrievalContext",
    "RAGPipeline",
    "RAGConfig",
    "RAGResponse",
    "GraphEnhancer",
]