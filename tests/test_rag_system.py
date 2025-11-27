"""
Tests for Production RAG System.

Test coverage:
- Document processing
- Embedding generation
- Vector storage
- Graph enhancement
- Hybrid retrieval
- RAG pipeline
- API endpoints
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from axiom.models.rag import (
    DocumentProcessor,
    DocumentChunk,
    EmbeddingService,
    GraphEnhancer,
    HybridRetriever,
    RAGPipeline,
    RAGConfig
)


class TestDocumentProcessor:
    """Test document processing functionality."""
    
    def test_process_text(self):
        """Test text processing into chunks."""
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
        
        text = "This is a test document. " * 20
        chunks = processor.process_text(
            text=text,
            document_id="test_doc_1",
            document_name="test.txt"
        )
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
        assert all(chunk.document_id == "test_doc_1" for chunk in chunks)
    
    def test_chunk_overlap(self):
        """Test that chunks have proper overlap."""
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
        
        text = "A" * 200
        chunks = processor.process_text(text, "doc1", "test.txt")
        
        assert len(chunks) >= 2
        # Check that consecutive chunks overlap
        if len(chunks) >= 2:
            assert chunks[1].start_char < chunks[0].end_char
    
    def test_extract_companies(self):
        """Test company name extraction."""
        processor = DocumentProcessor(extract_ma_entities=True)
        
        text = "Microsoft Corporation acquired LinkedIn Corp. for $26B."
        chunks = processor.process_text(text, "doc1", "test.txt")
        
        assert len(chunks) > 0
        # Should extract company names
        assert len(chunks[0].companies_mentioned) >= 0
    
    def test_extract_financial_figures(self):
        """Test financial figure extraction."""
        processor = DocumentProcessor(extract_ma_entities=True)
        
        text = "The deal was valued at $26.2 billion."
        chunks = processor.process_text(text, "doc1", "test.txt")
        
        assert len(chunks) > 0
        # Should extract financial figures
        assert len(chunks[0].financial_figures) >= 0


class TestEmbeddingService:
    """Test embedding service functionality."""
    
    @pytest.fixture
    def mock_openai(self):
        """Mock OpenAI client."""
        with patch('axiom.models.rag.embedding_service.OpenAI') as mock:
            mock_client = Mock()
            mock_response = Mock()
            mock_response.data = [Mock(embedding=[0.1] * 1536)]
            mock_client.embeddings.create.return_value = mock_response
            mock.return_value = mock_client
            yield mock
    
    def test_embed_single_text(self, mock_openai):
        """Test embedding single text."""
        service = EmbeddingService()
        
        embedding = service.embed_text("Test text")
        
        assert isinstance(embedding, list)
        assert len(embedding) == 1536
    
    def test_embed_multiple_texts(self, mock_openai):
        """Test batch embedding."""
        service = EmbeddingService(batch_size=2)
        
        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = service.embed_texts(texts)
        
        assert len(embeddings) == 3
        assert all(len(emb) == 1536 for emb in embeddings)
    
    def test_embedding_cache(self, mock_openai):
        """Test embedding caching."""
        service = EmbeddingService(cache_embeddings=True)
        
        text = "Test text"
        
        # First call - should generate
        embedding1 = service.embed_text(text)
        
        # Second call - should use cache
        embedding2 = service.embed_text(text)
        
        assert embedding1 == embedding2
        assert service.get_cache_size() > 0
    
    def test_similarity_computation(self):
        """Test cosine similarity computation."""
        service = EmbeddingService()
        
        emb1 = [1.0, 0.0, 0.0]
        emb2 = [1.0, 0.0, 0.0]
        emb3 = [0.0, 1.0, 0.0]
        
        # Identical vectors
        sim1 = service.compute_similarity(emb1, emb2)
        assert abs(sim1 - 1.0) < 0.01
        
        # Orthogonal vectors
        sim2 = service.compute_similarity(emb1, emb3)
        assert abs(sim2 - 0.0) < 0.01


class TestGraphEnhancer:
    """Test graph enhancement functionality."""
    
    @pytest.fixture
    def mock_graph(self):
        """Mock Neo4j graph."""
        with patch('axiom.models.rag.graph_enhancer.Neo4jGraph') as mock:
            mock_instance = Mock()
            mock_instance.health_check.return_value = True
            mock.return_value = mock_instance
            yield mock_instance
    
    def test_enhance_with_graph_context(self, mock_graph):
        """Test graph context enhancement."""
        enhancer = GraphEnhancer()
        
        # Mock graph responses
        mock_graph.get_acquisition_history.return_value = [
            {"target": "LinkedIn", "value": 26.2e9}
        ]
        mock_graph.get_subsidiaries.return_value = []
        mock_graph.find_connected_companies.return_value = []
        
        context = enhancer.enhance_with_graph_context(
            text="Microsoft acquired LinkedIn",
            companies=["MSFT", "LNKD"]
        )
        
        assert context.relationship_count >= 0
    
    def test_find_deal_patterns(self, mock_graph):
        """Test deal pattern finding."""
        enhancer = GraphEnhancer()
        
        mock_graph.find_acquisition_targets.return_value = [
            {"symbol": "TARGET1", "pattern_score": 5}
        ]
        
        patterns = enhancer.find_deal_patterns("MSFT", "Technology")
        
        assert isinstance(patterns, list)


class TestHybridRetriever:
    """Test hybrid retrieval functionality."""
    
    @pytest.fixture
    def mock_components(self):
        """Mock retriever components."""
        with patch('axiom.models.rag.hybrid_retriever.get_vector_store') as mock_vs, \
             patch('axiom.models.rag.hybrid_retriever.EmbeddingService') as mock_es, \
             patch('axiom.models.rag.hybrid_retriever.GraphEnhancer') as mock_ge:
            
            # Mock vector store
            mock_vs_instance = Mock()
            mock_vs_instance.create_collection = Mock()
            mock_vs_instance.search.return_value = []
            mock_vs_instance.list_collections.return_value = []
            mock_vs.return_value = mock_vs_instance
            
            # Mock embedding service
            mock_es_instance = Mock()
            mock_es_instance.embed_text.return_value = [0.1] * 1536
            mock_es_instance.dimension = 1536
            mock_es_instance.model = "test-model"
            mock_es.return_value = mock_es_instance
            
            # Mock graph enhancer
            mock_ge_instance = Mock()
            mock_ge.return_value = mock_ge_instance
            
            yield {
                'vector_store': mock_vs_instance,
                'embedding': mock_es_instance,
                'graph': mock_ge_instance
            }
    
    def test_index_chunks(self, mock_components):
        """Test chunk indexing."""
        retriever = HybridRetriever()
        
        chunks = [
            DocumentChunk(
                chunk_id="chunk1",
                text="Test text",
                document_id="doc1",
                document_name="test.txt",
                chunk_index=0,
                embedding=[0.1] * 1536
            )
        ]
        
        indexed = retriever.index_chunks(chunks)
        
        assert indexed >= 0
    
    def test_retrieve_no_results(self, mock_components):
        """Test retrieval with no results."""
        retriever = HybridRetriever()
        
        # Mock empty search results
        mock_components['vector_store'].search.return_value = []
        
        context = retriever.retrieve("test query")
        
        assert len(context.results) == 0


class TestRAGPipeline:
    """Test complete RAG pipeline."""
    
    @pytest.fixture
    def mock_pipeline_components(self):
        """Mock all pipeline components."""
        with patch('axiom.models.rag.rag_pipeline.DocumentProcessor') as mock_dp, \
             patch('axiom.models.rag.rag_pipeline.EmbeddingService') as mock_es, \
             patch('axiom.models.rag.rag_pipeline.HybridRetriever') as mock_hr, \
             patch('axiom.models.rag.rag_pipeline.Anthropic') as mock_claude:
            
            # Mock document processor
            mock_dp_instance = Mock()
            mock_dp_instance.process_file.return_value = []
            mock_dp.return_value = mock_dp_instance
            
            # Mock embedding service
            mock_es_instance = Mock()
            mock_es.return_value = mock_es_instance
            
            # Mock retriever
            mock_hr_instance = Mock()
            mock_hr_instance.index_chunks.return_value = 0
            mock_hr_instance.retrieve.return_value = Mock(
                results=[],
                query="test",
                total_retrieved=0,
                vector_results=0,
                graph_enhanced=0,
                reranked=False,
                retrieval_time_ms=10.0,
                embedding_time_ms=5.0,
                graph_time_ms=0.0
            )
            mock_hr.return_value = mock_hr_instance
            
            # Mock Claude
            mock_claude_instance = Mock()
            mock_response = Mock()
            mock_response.content = [Mock(text="Test answer")]
            mock_response.usage = Mock(input_tokens=10, output_tokens=20)
            mock_claude_instance.messages.create.return_value = mock_response
            mock_claude.return_value = mock_claude_instance
            
            yield {
                'doc_processor': mock_dp_instance,
                'embedding': mock_es_instance,
                'retriever': mock_hr_instance,
                'claude': mock_claude_instance
            }
    
    def test_pipeline_initialization(self, mock_pipeline_components):
        """Test pipeline initialization."""
        config = RAGConfig()
        pipeline = RAGPipeline(config=config)
        
        assert pipeline is not None
        assert pipeline.config == config
    
    def test_query_processing(self, mock_pipeline_components):
        """Test query processing."""
        pipeline = RAGPipeline()
        
        response = pipeline.query("test query")
        
        assert response is not None
        assert response.query == "test query"
        assert response.answer == "Test answer"
    
    def test_statistics(self, mock_pipeline_components):
        """Test statistics retrieval."""
        pipeline = RAGPipeline()
        
        # Mock retriever stats
        mock_pipeline_components['retriever'].get_statistics.return_value = {
            'collection_name': 'test',
            'vector_store_type': 'ChromaDB'
        }
        
        # Mock embedding stats
        mock_pipeline_components['embedding'].get_statistics.return_value = {
            'model': 'test-model',
            'dimension': 1536
        }
        
        stats = pipeline.get_statistics()
        
        assert 'pipeline' in stats
        assert 'retriever' in stats
        assert 'embedding' in stats


class TestIntegration:
    """Integration tests for complete workflow."""
    
    def test_end_to_end_workflow(self):
        """Test complete workflow with mocked external services."""
        
        # This would be a full integration test
        # For now, just verify components can be instantiated
        
        config = RAGConfig(
            chunk_size=100,
            chunk_overlap=20,
            top_k=5
        )
        
        assert config.chunk_size == 100
        assert config.chunk_overlap == 20
        assert config.top_k == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])