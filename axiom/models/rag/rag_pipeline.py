"""
Production RAG Pipeline for M&A Intelligence.

Complete pipeline:
1. Document ingestion → chunking → embedding → storage
2. Query → retrieval (vector + graph) → reranking
3. Context augmentation → Claude generation
4. Response with sources and confidence

Features:
- DSPy-optimized retrieval chains
- Claude API integration
- Multi-step reasoning
- Source attribution
- Confidence scoring
"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

import dspy
from anthropic import Anthropic

from ...config.settings import settings
from .document_processor import DocumentProcessor, DocumentChunk
from .embedding_service import EmbeddingService
from .hybrid_retriever import HybridRetriever, RetrievalContext
from .graph_enhancer import GraphEnhancer

logger = logging.getLogger(__name__)


@dataclass
class RAGConfig:
    """Configuration for RAG pipeline."""
    
    # Document processing
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Retrieval
    top_k: int = 10
    similarity_threshold: float = 0.7
    enable_graph: bool = True
    enable_reranking: bool = True
    
    # Generation
    model: str = "claude-3-5-sonnet-20241022"
    temperature: float = 0.1
    max_tokens: int = 4000
    
    # Features
    enable_multi_hop: bool = True
    enable_citations: bool = True
    confidence_threshold: float = 0.6


@dataclass
class RAGResponse:
    """Response from RAG pipeline."""
    
    query: str
    answer: str
    
    # Context
    retrieval_context: RetrievalContext
    sources: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    confidence: float = 0.0
    reasoning_steps: List[str] = field(default_factory=list)
    
    # Performance
    total_time_ms: float = 0.0
    retrieval_time_ms: float = 0.0
    generation_time_ms: float = 0.0
    
    # Model info
    model_used: str = ""
    tokens_used: int = 0
    
    generated_at: datetime = field(default_factory=datetime.now)


class RAGSignature(dspy.Signature):
    """DSPy signature for RAG generation."""
    
    query = dspy.InputField(desc="User query about M&A intelligence")
    context = dspy.InputField(desc="Retrieved context from documents and knowledge graph")
    answer = dspy.OutputField(desc="Detailed answer with sources and reasoning")


class RAGModule(dspy.Module):
    """DSPy module for RAG generation."""
    
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(RAGSignature)
    
    def forward(self, query: str, context: str):
        """Generate answer with chain-of-thought reasoning."""
        return self.generate(query=query, context=context)


class RAGPipeline:
    """
    Production RAG pipeline for M&A intelligence.
    
    Architecture:
    - Document Processing: PDF/DOCX → chunks → embeddings
    - Hybrid Retrieval: Vector search + graph enhancement
    - DSPy Optimization: Optimized retrieval chains
    - Claude Generation: High-quality responses with reasoning
    """
    
    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        collection_name: str = "ma_documents"
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            config: Pipeline configuration
            collection_name: Vector store collection name
        """
        self.config = config or RAGConfig()
        self.collection_name = collection_name
        
        # Initialize components
        self.document_processor = DocumentProcessor(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        
        self.embedding_service = EmbeddingService()
        
        self.retriever = HybridRetriever(
            collection_name=collection_name,
            embedding_service=self.embedding_service,
            graph_enhancer=GraphEnhancer() if self.config.enable_graph else None,
            top_k=self.config.top_k,
            similarity_threshold=self.config.similarity_threshold,
            enable_reranking=self.config.enable_reranking
        )
        
        # Initialize Claude
        api_key = settings.claude_api_key
        if not api_key or api_key in ["sk-placeholder", "test_key"]:
            raise ValueError("Valid Claude API key required for generation")
        
        self.claude = Anthropic(api_key=api_key)
        
        # DSPy configuration (optional - requires valid LM)
        self.dspy_enabled = False
        try:
            # Configure DSPy with Claude
            lm = dspy.Claude(
                model=self.config.model,
                api_key=api_key
            )
            dspy.settings.configure(lm=lm)
            self.rag_module = RAGModule()
            self.dspy_enabled = True
            logger.info("DSPy optimization enabled")
        except Exception as e:
            logger.warning(f"DSPy not available: {e}")
            self.rag_module = None
        
        logger.info(f"Initialized RAG pipeline with collection '{collection_name}'")
    
    def ingest_documents(
        self,
        file_paths: List[str],
        metadata: Optional[Dict[str, Any]] = None,
        batch_size: int = 100
    ) -> int:
        """
        Ingest documents into RAG system.
        
        Args:
            file_paths: List of document file paths
            metadata: Additional metadata
            batch_size: Batch size for indexing
            
        Returns:
            Total number of chunks indexed
        """
        logger.info(f"Ingesting {len(file_paths)} documents...")
        
        all_chunks = []
        
        for file_path in file_paths:
            try:
                chunks = self.document_processor.process_file(
                    file_path=file_path,
                    metadata=metadata
                )
                all_chunks.extend(chunks)
                logger.info(f"Processed {file_path}: {len(chunks)} chunks")
                
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
        
        # Index chunks
        if all_chunks:
            indexed = self.retriever.index_chunks(all_chunks, batch_size=batch_size)
            logger.info(f"Ingested {len(file_paths)} documents ({indexed} chunks)")
            return indexed
        
        return 0
    
    def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        enable_graph: Optional[bool] = None,
        temperature: Optional[float] = None
    ) -> RAGResponse:
        """
        Query the RAG system.
        
        Args:
            query: User query
            top_k: Number of results to retrieve
            enable_graph: Enable graph enhancement
            temperature: Generation temperature
            
        Returns:
            RAG response with answer and sources
        """
        start_time = time.time()
        
        # Retrieve context
        retrieval_start = time.time()
        retrieval_context = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            enable_graph=enable_graph if enable_graph is not None else self.config.enable_graph
        )
        retrieval_time = (time.time() - retrieval_start) * 1000
        
        # Generate response
        generation_start = time.time()
        
        if self.dspy_enabled and self.rag_module:
            answer, tokens, reasoning = self._generate_with_dspy(query, retrieval_context)
        else:
            answer, tokens, reasoning = self._generate_with_claude(
                query,
                retrieval_context,
                temperature or self.config.temperature
            )
        
        generation_time = (time.time() - generation_start) * 1000
        
        # Extract sources
        sources = self._extract_sources(retrieval_context)
        
        # Calculate confidence
        confidence = self._calculate_confidence(retrieval_context, answer)
        
        total_time = (time.time() - start_time) * 1000
        
        response = RAGResponse(
            query=query,
            answer=answer,
            retrieval_context=retrieval_context,
            sources=sources,
            confidence=confidence,
            reasoning_steps=reasoning,
            total_time_ms=total_time,
            retrieval_time_ms=retrieval_time,
            generation_time_ms=generation_time,
            model_used=self.config.model,
            tokens_used=tokens
        )
        
        logger.info(
            f"Query completed in {total_time:.1f}ms "
            f"(retrieval: {retrieval_time:.1f}ms, generation: {generation_time:.1f}ms)"
        )
        
        return response
    
    def _generate_with_dspy(
        self,
        query: str,
        context: RetrievalContext
    ) -> tuple[str, int, List[str]]:
        """Generate answer using DSPy module."""
        
        # Format context
        context_str = self._format_context_for_generation(context)
        
        # Generate with DSPy
        try:
            result = self.rag_module(query=query, context=context_str)
            answer = result.answer
            
            # Extract reasoning steps
            reasoning = []
            if hasattr(result, 'reasoning'):
                reasoning = [result.reasoning]
            
            # Estimate tokens (rough approximation)
            tokens = len(answer.split()) * 1.3
            
            return answer, int(tokens), reasoning
            
        except Exception as e:
            logger.error(f"DSPy generation failed: {e}")
            # Fallback to Claude
            return self._generate_with_claude(query, context, self.config.temperature)
    
    def _generate_with_claude(
        self,
        query: str,
        context: RetrievalContext,
        temperature: float
    ) -> tuple[str, int, List[str]]:
        """Generate answer using Claude API."""
        
        # Format context
        context_str = self._format_context_for_generation(context)
        
        # Create prompt
        system_prompt = """You are an expert M&A analyst with deep knowledge of mergers, acquisitions, and corporate transactions. 

Your task is to provide detailed, accurate answers to questions about M&A deals, market trends, and strategic analysis. Use the provided context from documents and knowledge graphs to support your answers.

Guidelines:
1. Base answers on the provided context
2. Cite specific sources when making claims
3. Acknowledge limitations if information is insufficient
4. Provide reasoning for your conclusions
5. Use professional, precise language
6. Include relevant financial figures and dates when available"""
        
        user_prompt = f"""Question: {query}

Context from M&A documents and knowledge graph:
{context_str}

Please provide a comprehensive answer with:
1. Direct answer to the question
2. Supporting evidence from the context
3. Relevant insights from related deals or relationships
4. Any caveats or limitations in the available information

Answer:"""
        
        try:
            response = self.claude.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            answer = response.content[0].text
            tokens = response.usage.input_tokens + response.usage.output_tokens
            
            # Extract reasoning steps (simplified)
            reasoning = ["Generated answer using Claude with retrieved context"]
            
            return answer, tokens, reasoning
            
        except Exception as e:
            logger.error(f"Claude generation failed: {e}")
            raise
    
    def _format_context_for_generation(self, context: RetrievalContext) -> str:
        """Format retrieval context for LLM generation."""
        
        formatted_parts = []
        
        for i, result in enumerate(context.results, 1):
            chunk = result.chunk
            
            # Document info
            doc_info = f"[Source {i}] {chunk.document_name}"
            if chunk.page_number is not None:
                doc_info += f" (Page {chunk.page_number})"
            
            # Content
            content = chunk.text
            
            # Companies mentioned
            if chunk.companies_mentioned:
                companies = ", ".join(chunk.companies_mentioned[:3])
                doc_info += f" - Companies: {companies}"
            
            # Graph context
            if result.graph_context and result.graph_context.relationship_count > 0:
                doc_info += f" [+{result.graph_context.relationship_count} graph relationships]"
            
            formatted_parts.append(f"{doc_info}\n{content}\n")
        
        return "\n---\n".join(formatted_parts)
    
    def _extract_sources(self, context: RetrievalContext) -> List[Dict[str, Any]]:
        """Extract source information from retrieval context."""
        
        sources = []
        
        for result in context.results:
            chunk = result.chunk
            
            source = {
                "document_id": chunk.document_id,
                "document_name": chunk.document_name,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "relevance_score": result.score,
                "companies": chunk.companies_mentioned[:3],
                "preview": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text
            }
            
            # Add graph context if available
            if result.graph_context:
                source["graph_relationships"] = result.graph_context.relationship_count
            
            sources.append(source)
        
        return sources
    
    def _calculate_confidence(
        self,
        context: RetrievalContext,
        answer: str
    ) -> float:
        """Calculate confidence score for the answer."""
        
        # Factors for confidence:
        # 1. Retrieval scores
        # 2. Number of sources
        # 3. Graph relationships
        # 4. Answer length and specificity
        
        if not context.results:
            return 0.0
        
        # Average retrieval score
        avg_score = sum(r.score for r in context.results) / len(context.results)
        
        # Source diversity score (more sources = higher confidence)
        source_score = min(1.0, len(context.results) / 10.0)
        
        # Graph enhancement score
        graph_score = 0.0
        if context.graph_enhanced > 0:
            graph_score = min(1.0, context.graph_enhanced / len(context.results))
        
        # Answer specificity (simple heuristic)
        specificity_score = min(1.0, len(answer.split()) / 200.0)
        
        # Weighted combination
        confidence = (
            avg_score * 0.4 +
            source_score * 0.2 +
            graph_score * 0.2 +
            specificity_score * 0.2
        )
        
        return round(confidence, 2)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        
        retriever_stats = self.retriever.get_statistics()
        embedding_stats = self.embedding_service.get_statistics()
        
        return {
            "pipeline": {
                "collection_name": self.collection_name,
                "dspy_enabled": self.dspy_enabled,
                "generation_model": self.config.model,
                "top_k": self.config.top_k,
            },
            "retriever": retriever_stats,
            "embedding": embedding_stats,
            "config": {
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
                "similarity_threshold": self.config.similarity_threshold,
                "enable_graph": self.config.enable_graph,
                "enable_reranking": self.config.enable_reranking,
            }
        }


__all__ = ["RAGPipeline", "RAGConfig", "RAGResponse"]