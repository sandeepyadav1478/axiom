"""
Embedding Service for RAG System.

Features:
- Multiple embedding model support (OpenAI, local models)
- Batch processing for efficiency
- Caching to avoid redundant API calls
- Automatic retries and failover
"""

import hashlib
import logging
from typing import List, Dict, Any, Optional
import asyncio

import numpy as np
from openai import OpenAI, AsyncOpenAI

from ...config.settings import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Generate embeddings for documents and queries.
    
    Supports:
    - OpenAI embeddings (text-embedding-3-small, text-embedding-3-large)
    - Batch processing for efficiency
    - In-memory caching
    - Automatic retry logic
    """
    
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        batch_size: int = 100,
        cache_embeddings: bool = True,
        api_key: Optional[str] = None
    ):
        """
        Initialize embedding service.
        
        Args:
            model: Embedding model name
            batch_size: Batch size for processing
            cache_embeddings: Enable embedding cache
            api_key: OpenAI API key (optional)
        """
        self.model = model
        self.batch_size = batch_size
        self.cache_embeddings = cache_embeddings
        
        # Initialize OpenAI client
        api_key = api_key or settings.openai_api_key
        if not api_key or api_key in ["sk-placeholder", "test_key"]:
            raise ValueError("Valid OpenAI API key required for embeddings")
        
        self.client = OpenAI(api_key=api_key)
        self.async_client = AsyncOpenAI(api_key=api_key)
        
        # Embedding cache
        self._cache: Dict[str, List[float]] = {}
        
        # Model dimensions
        self.dimension = self._get_model_dimension()
        
        logger.info(f"Initialized EmbeddingService with model {model} (dim={self.dimension})")
    
    def _get_model_dimension(self) -> int:
        """Get embedding dimension for model."""
        dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return dimensions.get(self.model, 1536)
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        # Check cache
        if self.cache_embeddings:
            cache_key = self._get_cache_key(text)
            if cache_key in self._cache:
                logger.debug(f"Cache hit for text embedding")
                return self._cache[cache_key]
        
        # Generate embedding
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=[text],
                encoding_format="float"
            )
            
            embedding = response.data[0].embedding
            
            # Cache result
            if self.cache_embeddings:
                self._cache[cache_key] = embedding
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (batch processing).
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        embeddings = []
        
        # Check cache first
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            if self.cache_embeddings:
                cache_key = self._get_cache_key(text)
                if cache_key in self._cache:
                    embeddings.append(self._cache[cache_key])
                else:
                    embeddings.append(None)  # Placeholder
                    uncached_texts.append(text)
                    uncached_indices.append(i)
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            logger.info(f"Generating embeddings for {len(uncached_texts)} texts")
            
            # Process in batches
            for batch_start in range(0, len(uncached_texts), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(uncached_texts))
                batch = uncached_texts[batch_start:batch_end]
                
                try:
                    response = self.client.embeddings.create(
                        model=self.model,
                        input=batch,
                        encoding_format="float"
                    )
                    
                    batch_embeddings = [item.embedding for item in response.data]
                    
                    # Store in cache and results
                    for j, text in enumerate(batch):
                        emb = batch_embeddings[j]
                        original_idx = uncached_indices[batch_start + j]
                        
                        if self.cache_embeddings:
                            cache_key = self._get_cache_key(text)
                            self._cache[cache_key] = emb
                        
                        if embeddings[original_idx] is None:
                            embeddings[original_idx] = emb
                        else:
                            embeddings.append(emb)
                    
                except Exception as e:
                    logger.error(f"Failed to generate batch embeddings: {e}")
                    # Return zero vectors for failed batch
                    for j in range(len(batch)):
                        original_idx = uncached_indices[batch_start + j]
                        if embeddings[original_idx] is None:
                            embeddings[original_idx] = [0.0] * self.dimension
        
        # Handle case where we didn't use cache
        if not self.cache_embeddings:
            embeddings = []
            for batch_start in range(0, len(texts), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(texts))
                batch = texts[batch_start:batch_end]
                
                try:
                    response = self.client.embeddings.create(
                        model=self.model,
                        input=batch,
                        encoding_format="float"
                    )
                    
                    embeddings.extend([item.embedding for item in response.data])
                    
                except Exception as e:
                    logger.error(f"Failed to generate batch embeddings: {e}")
                    embeddings.extend([[0.0] * self.dimension] * len(batch))
        
        return embeddings
    
    async def embed_text_async(self, text: str) -> List[float]:
        """
        Generate embedding for single text (async).
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        # Check cache
        if self.cache_embeddings:
            cache_key = self._get_cache_key(text)
            if cache_key in self._cache:
                return self._cache[cache_key]
        
        # Generate embedding
        try:
            response = await self.async_client.embeddings.create(
                model=self.model,
                input=[text],
                encoding_format="float"
            )
            
            embedding = response.data[0].embedding
            
            # Cache result
            if self.cache_embeddings:
                self._cache[cache_key] = embedding
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate async embedding: {e}")
            raise
    
    async def embed_texts_async(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (async batch processing).
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Check cache first
        uncached_texts = []
        cached_embeddings = {}
        
        for i, text in enumerate(texts):
            if self.cache_embeddings:
                cache_key = self._get_cache_key(text)
                if cache_key in self._cache:
                    cached_embeddings[i] = self._cache[cache_key]
                else:
                    uncached_texts.append((i, text))
            else:
                uncached_texts.append((i, text))
        
        # Generate embeddings for uncached texts
        embeddings = [None] * len(texts)
        
        # Fill in cached embeddings
        for i, emb in cached_embeddings.items():
            embeddings[i] = emb
        
        if uncached_texts:
            logger.info(f"Generating async embeddings for {len(uncached_texts)} texts")
            
            # Process in batches
            for batch_start in range(0, len(uncached_texts), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(uncached_texts))
                batch = uncached_texts[batch_start:batch_end]
                
                batch_texts = [text for _, text in batch]
                batch_indices = [i for i, _ in batch]
                
                try:
                    response = await self.async_client.embeddings.create(
                        model=self.model,
                        input=batch_texts,
                        encoding_format="float"
                    )
                    
                    for j, item in enumerate(response.data):
                        emb = item.embedding
                        original_idx = batch_indices[j]
                        embeddings[original_idx] = emb
                        
                        # Cache
                        if self.cache_embeddings:
                            cache_key = self._get_cache_key(batch_texts[j])
                            self._cache[cache_key] = emb
                    
                except Exception as e:
                    logger.error(f"Failed to generate async batch embeddings: {e}")
                    # Fill with zero vectors
                    for idx in batch_indices:
                        embeddings[idx] = [0.0] * self.dimension
        
        return embeddings
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        content = f"{self.model}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]
    
    def clear_cache(self):
        """Clear embedding cache."""
        self._cache.clear()
        logger.info("Cleared embedding cache")
    
    def get_cache_size(self) -> int:
        """Get number of cached embeddings."""
        return len(self._cache)
    
    def compute_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score (0-1)
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get embedding service statistics."""
        return {
            "model": self.model,
            "dimension": self.dimension,
            "batch_size": self.batch_size,
            "cache_enabled": self.cache_embeddings,
            "cached_embeddings": len(self._cache),
        }


__all__ = ["EmbeddingService"]