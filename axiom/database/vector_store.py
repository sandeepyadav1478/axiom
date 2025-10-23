"""
Vector Database Integration for Semantic Search and RAG.

Supports:
- Pinecone for production deployments
- Weaviate for self-hosted solutions
- ChromaDB for local development
- Company embeddings for M&A target search
- Document embeddings for SEC filings and research
"""

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
import hashlib
import json

import numpy as np

from ..config.settings import settings

logger = logging.getLogger(__name__)


class VectorStoreType(Enum):
    """Supported vector database types."""
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"
    CHROMA = "chroma"


class VectorStore(ABC):
    """
    Abstract base class for vector database operations.
    
    Provides unified interface for:
    - Storing embeddings
    - Semantic search
    - Similarity queries
    - RAG support
    """
    
    @abstractmethod
    def create_collection(
        self,
        name: str,
        dimension: int,
        metadata_config: Optional[Dict[str, Any]] = None
    ):
        """Create a new collection/index."""
        pass
    
    @abstractmethod
    def upsert(
        self,
        collection_name: str,
        vectors: List[np.ndarray],
        ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ):
        """Insert or update vectors."""
        pass
    
    @abstractmethod
    def search(
        self,
        collection_name: str,
        query_vector: np.ndarray,
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        pass
    
    @abstractmethod
    def delete(
        self,
        collection_name: str,
        ids: List[str]
    ):
        """Delete vectors by IDs."""
        pass
    
    @abstractmethod
    def get(
        self,
        collection_name: str,
        ids: List[str]
    ) -> List[Dict[str, Any]]:
        """Get vectors by IDs."""
        pass
    
    @abstractmethod
    def list_collections(self) -> List[str]:
        """List all collections."""
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        """Check vector store health."""
        pass


class PineconeVectorStore(VectorStore):
    """
    Pinecone vector database integration.
    
    Production-grade vector database for:
    - High-performance semantic search
    - Scalable embeddings storage
    - Low-latency queries
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        environment: Optional[str] = None
    ):
        """
        Initialize Pinecone client.
        
        Args:
            api_key: Pinecone API key
            environment: Pinecone environment (e.g., 'us-east1-gcp')
        """
        try:
            import pinecone
            self.pinecone = pinecone
        except ImportError:
            raise ImportError(
                "Pinecone not installed. Install with: pip install pinecone-client"
            )
        
        self.api_key = api_key or getattr(settings, 'pinecone_api_key', None)
        self.environment = environment or getattr(settings, 'pinecone_environment', 'us-east1-gcp')
        
        if not self.api_key:
            raise ValueError("Pinecone API key not provided")
        
        # Initialize Pinecone
        self.pinecone.init(api_key=self.api_key, environment=self.environment)
        logger.info(f"Initialized Pinecone in environment: {self.environment}")
    
    def create_collection(
        self,
        name: str,
        dimension: int,
        metadata_config: Optional[Dict[str, Any]] = None
    ):
        """Create Pinecone index."""
        if name in self.pinecone.list_indexes():
            logger.info(f"Index '{name}' already exists")
            return
        
        self.pinecone.create_index(
            name=name,
            dimension=dimension,
            metric='cosine',
            metadata_config=metadata_config or {}
        )
        logger.info(f"Created Pinecone index: {name} (dimension={dimension})")
    
    def upsert(
        self,
        collection_name: str,
        vectors: List[np.ndarray],
        ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ):
        """Upsert vectors to Pinecone."""
        index = self.pinecone.Index(collection_name)
        
        # Prepare data for upsert
        vectors_list = [vec.tolist() if isinstance(vec, np.ndarray) else vec for vec in vectors]
        
        if metadata:
            data = list(zip(ids, vectors_list, metadata))
        else:
            data = list(zip(ids, vectors_list))
        
        # Batch upsert (Pinecone recommends batches of 100)
        batch_size = 100
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            index.upsert(vectors=batch)
        
        logger.info(f"Upserted {len(ids)} vectors to {collection_name}")
    
    def search(
        self,
        collection_name: str,
        query_vector: np.ndarray,
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search Pinecone index."""
        index = self.pinecone.Index(collection_name)
        
        query_list = query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector
        
        results = index.query(
            vector=query_list,
            top_k=top_k,
            include_metadata=True,
            filter=filter
        )
        
        return [
            {
                'id': match.id,
                'score': match.score,
                'metadata': match.metadata
            }
            for match in results.matches
        ]
    
    def delete(self, collection_name: str, ids: List[str]):
        """Delete vectors from Pinecone."""
        index = self.pinecone.Index(collection_name)
        index.delete(ids=ids)
        logger.info(f"Deleted {len(ids)} vectors from {collection_name}")
    
    def get(self, collection_name: str, ids: List[str]) -> List[Dict[str, Any]]:
        """Fetch vectors from Pinecone."""
        index = self.pinecone.Index(collection_name)
        result = index.fetch(ids=ids)
        
        return [
            {
                'id': id,
                'vector': result.vectors[id].values,
                'metadata': result.vectors[id].metadata
            }
            for id in ids if id in result.vectors
        ]
    
    def list_collections(self) -> List[str]:
        """List Pinecone indexes."""
        return self.pinecone.list_indexes()
    
    def health_check(self) -> bool:
        """Check Pinecone health."""
        try:
            self.pinecone.list_indexes()
            return True
        except Exception as e:
            logger.error(f"Pinecone health check failed: {e}")
            return False


class WeaviateVectorStore(VectorStore):
    """
    Weaviate vector database integration.
    
    Self-hosted vector database for:
    - On-premise deployments
    - Data sovereignty requirements
    - GraphQL-based queries
    """
    
    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize Weaviate client.
        
        Args:
            url: Weaviate instance URL
            api_key: Weaviate API key (if authentication enabled)
        """
        try:
            import weaviate
            self.weaviate = weaviate
        except ImportError:
            raise ImportError(
                "Weaviate not installed. Install with: pip install weaviate-client"
            )
        
        self.url = url or getattr(settings, 'weaviate_url', 'http://localhost:8080')
        self.api_key = api_key or getattr(settings, 'weaviate_api_key', None)
        
        # Initialize client
        if self.api_key:
            auth_config = weaviate.AuthApiKey(api_key=self.api_key)
            self.client = weaviate.Client(url=self.url, auth_client_secret=auth_config)
        else:
            self.client = weaviate.Client(url=self.url)
        
        logger.info(f"Initialized Weaviate at: {self.url}")
    
    def create_collection(
        self,
        name: str,
        dimension: int,
        metadata_config: Optional[Dict[str, Any]] = None
    ):
        """Create Weaviate class."""
        class_obj = {
            "class": name,
            "vectorizer": "none",  # We provide our own vectors
            "properties": [
                {
                    "name": "content",
                    "dataType": ["text"]
                },
                {
                    "name": "metadata",
                    "dataType": ["text"]
                }
            ]
        }
        
        # Add custom properties from metadata_config
        if metadata_config:
            for key, dtype in metadata_config.items():
                class_obj["properties"].append({
                    "name": key,
                    "dataType": [dtype]
                })
        
        try:
            self.client.schema.create_class(class_obj)
            logger.info(f"Created Weaviate class: {name}")
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.info(f"Class '{name}' already exists")
            else:
                raise
    
    def upsert(
        self,
        collection_name: str,
        vectors: List[np.ndarray],
        ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ):
        """Upsert vectors to Weaviate."""
        with self.client.batch as batch:
            batch.batch_size = 100
            
            for i, (id, vector) in enumerate(zip(ids, vectors)):
                vector_list = vector.tolist() if isinstance(vector, np.ndarray) else vector
                
                properties = {
                    "content": metadata[i].get("content", "") if metadata else "",
                    "metadata": json.dumps(metadata[i]) if metadata else "{}"
                }
                
                batch.add_data_object(
                    data_object=properties,
                    class_name=collection_name,
                    uuid=id,
                    vector=vector_list
                )
        
        logger.info(f"Upserted {len(ids)} vectors to {collection_name}")
    
    def search(
        self,
        collection_name: str,
        query_vector: np.ndarray,
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search Weaviate."""
        query_list = query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector
        
        query = (
            self.client.query
            .get(collection_name, ["content", "metadata"])
            .with_near_vector({"vector": query_list})
            .with_limit(top_k)
            .with_additional(["id", "distance"])
        )
        
        # Apply filters if provided
        if filter:
            where_filter = self._build_where_filter(filter)
            query = query.with_where(where_filter)
        
        result = query.do()
        
        if "data" not in result or "Get" not in result["data"]:
            return []
        
        objects = result["data"]["Get"].get(collection_name, [])
        
        return [
            {
                'id': obj["_additional"]["id"],
                'score': 1 - obj["_additional"]["distance"],  # Convert distance to similarity
                'metadata': json.loads(obj.get("metadata", "{}"))
            }
            for obj in objects
        ]
    
    def _build_where_filter(self, filter: Dict[str, Any]) -> Dict[str, Any]:
        """Build Weaviate WHERE filter."""
        # Simple implementation - can be extended
        return {
            "operator": "And",
            "operands": [
                {
                    "path": [key],
                    "operator": "Equal",
                    "valueText": value
                }
                for key, value in filter.items()
            ]
        }
    
    def delete(self, collection_name: str, ids: List[str]):
        """Delete vectors from Weaviate."""
        for id in ids:
            self.client.data_object.delete(uuid=id, class_name=collection_name)
        logger.info(f"Deleted {len(ids)} vectors from {collection_name}")
    
    def get(self, collection_name: str, ids: List[str]) -> List[Dict[str, Any]]:
        """Get vectors from Weaviate."""
        results = []
        for id in ids:
            try:
                obj = self.client.data_object.get_by_id(
                    uuid=id,
                    class_name=collection_name,
                    with_vector=True
                )
                results.append({
                    'id': id,
                    'vector': obj.get("vector", []),
                    'metadata': json.loads(obj.get("properties", {}).get("metadata", "{}"))
                })
            except Exception as e:
                logger.warning(f"Failed to get object {id}: {e}")
        
        return results
    
    def list_collections(self) -> List[str]:
        """List Weaviate classes."""
        schema = self.client.schema.get()
        return [cls["class"] for cls in schema.get("classes", [])]
    
    def health_check(self) -> bool:
        """Check Weaviate health."""
        try:
            return self.client.is_ready()
        except Exception as e:
            logger.error(f"Weaviate health check failed: {e}")
            return False


class ChromaVectorStore(VectorStore):
    """
    ChromaDB vector database integration.
    
    Lightweight vector database for:
    - Local development
    - Testing
    - Small-scale deployments
    """
    
    def __init__(self, persist_directory: Optional[str] = None):
        """
        Initialize ChromaDB client.
        
        Args:
            persist_directory: Directory to persist data
        """
        try:
            import chromadb
            self.chromadb = chromadb
        except ImportError:
            raise ImportError(
                "ChromaDB not installed. Install with: pip install chromadb"
            )
        
        self.persist_directory = persist_directory or getattr(
            settings, 'chroma_persist_directory', './data/chroma'
        )
        
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        logger.info(f"Initialized ChromaDB at: {self.persist_directory}")
    
    def create_collection(
        self,
        name: str,
        dimension: int,
        metadata_config: Optional[Dict[str, Any]] = None
    ):
        """Create ChromaDB collection."""
        try:
            self.client.create_collection(
                name=name,
                metadata=metadata_config or {}
            )
            logger.info(f"Created ChromaDB collection: {name}")
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.info(f"Collection '{name}' already exists")
            else:
                raise
    
    def upsert(
        self,
        collection_name: str,
        vectors: List[np.ndarray],
        ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ):
        """Upsert to ChromaDB."""
        collection = self.client.get_collection(name=collection_name)
        
        vectors_list = [
            vec.tolist() if isinstance(vec, np.ndarray) else vec
            for vec in vectors
        ]
        
        # ChromaDB requires documents
        documents = [
            meta.get("content", "") if meta else ""
            for meta in (metadata or [{} for _ in ids])
        ]
        
        collection.upsert(
            ids=ids,
            embeddings=vectors_list,
            documents=documents,
            metadatas=metadata
        )
        
        logger.info(f"Upserted {len(ids)} vectors to {collection_name}")
    
    def search(
        self,
        collection_name: str,
        query_vector: np.ndarray,
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search ChromaDB."""
        collection = self.client.get_collection(name=collection_name)
        
        query_list = [query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector]
        
        results = collection.query(
            query_embeddings=query_list,
            n_results=top_k,
            where=filter
        )
        
        if not results['ids'] or not results['ids'][0]:
            return []
        
        return [
            {
                'id': results['ids'][0][i],
                'score': 1 - results['distances'][0][i],  # Convert distance to similarity
                'metadata': results['metadatas'][0][i] if results['metadatas'] else {}
            }
            for i in range(len(results['ids'][0]))
        ]
    
    def delete(self, collection_name: str, ids: List[str]):
        """Delete from ChromaDB."""
        collection = self.client.get_collection(name=collection_name)
        collection.delete(ids=ids)
        logger.info(f"Deleted {len(ids)} vectors from {collection_name}")
    
    def get(self, collection_name: str, ids: List[str]) -> List[Dict[str, Any]]:
        """Get from ChromaDB."""
        collection = self.client.get_collection(name=collection_name)
        results = collection.get(ids=ids, include=['embeddings', 'metadatas'])
        
        return [
            {
                'id': results['ids'][i],
                'vector': results['embeddings'][i] if results['embeddings'] else [],
                'metadata': results['metadatas'][i] if results['metadatas'] else {}
            }
            for i in range(len(results['ids']))
        ]
    
    def list_collections(self) -> List[str]:
        """List ChromaDB collections."""
        collections = self.client.list_collections()
        return [col.name for col in collections]
    
    def health_check(self) -> bool:
        """Check ChromaDB health."""
        try:
            self.client.heartbeat()
            return True
        except Exception as e:
            logger.error(f"ChromaDB health check failed: {e}")
            return False


def get_vector_store(
    store_type: Optional[VectorStoreType] = None,
    **kwargs
) -> VectorStore:
    """
    Get vector store instance.
    
    Args:
        store_type: Type of vector store
        **kwargs: Additional arguments for vector store
        
    Returns:
        VectorStore instance
    """
    if store_type is None:
        # Determine from settings
        store_type_str = getattr(settings, 'vector_store_type', 'chroma').lower()
        store_type = VectorStoreType(store_type_str)
    
    if store_type == VectorStoreType.PINECONE:
        return PineconeVectorStore(**kwargs)
    elif store_type == VectorStoreType.WEAVIATE:
        return WeaviateVectorStore(**kwargs)
    elif store_type == VectorStoreType.CHROMA:
        return ChromaVectorStore(**kwargs)
    else:
        raise ValueError(f"Unsupported vector store type: {store_type}")


# Export
__all__ = [
    "VectorStore",
    "VectorStoreType",
    "PineconeVectorStore",
    "WeaviateVectorStore",
    "ChromaVectorStore",
    "get_vector_store",
]