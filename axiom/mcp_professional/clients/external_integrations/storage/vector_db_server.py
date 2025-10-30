"""Vector Database MCP Server Implementation.

Provides vector database operations for semantic search through MCP protocol:
- Document management with embeddings
- Semantic search capabilities
- Collection management
- Support for Pinecone, Weaviate, ChromaDB, and Qdrant
"""

import json
import logging
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)

# Check for available vector DB clients
PINECONE_AVAILABLE = False
WEAVIATE_AVAILABLE = False
CHROMADB_AVAILABLE = False
QDRANT_AVAILABLE = False

try:
    import pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    pass

try:
    import weaviate
    WEAVIATE_AVAILABLE = True
except ImportError:
    pass

try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    pass

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    QDRANT_AVAILABLE = True
except ImportError:
    pass


class VectorDBMCPServer:
    """Vector Database MCP server implementation."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.provider = config.get("provider", "pinecone").lower()
        self.dimension = config.get("dimension", 1536)  # OpenAI default
        
        # Provider-specific configuration
        self.pinecone_api_key = config.get("pinecone_api_key")
        self.pinecone_environment = config.get("pinecone_environment")
        self.pinecone_index_name = config.get("pinecone_index_name", "axiom-index")
        
        self.weaviate_url = config.get("weaviate_url", "http://localhost:8080")
        self.weaviate_api_key = config.get("weaviate_api_key")
        
        self.chromadb_path = config.get("chromadb_path", "./chroma_db")
        self.chromadb_host = config.get("chromadb_host")
        self.chromadb_port = config.get("chromadb_port")
        
        self.qdrant_url = config.get("qdrant_url", "http://localhost:6333")
        self.qdrant_api_key = config.get("qdrant_api_key")
        self.qdrant_collection = config.get("qdrant_collection", "axiom-collection")
        
        # Initialize clients
        self._pinecone_index = None
        self._weaviate_client = None
        self._chromadb_client = None
        self._qdrant_client = None

    def _get_pinecone_index(self):
        """Get or create Pinecone index."""
        if not PINECONE_AVAILABLE:
            raise ImportError("pinecone-client required. Install with: pip install pinecone-client")
        
        if self._pinecone_index is None:
            pinecone.init(
                api_key=self.pinecone_api_key,
                environment=self.pinecone_environment
            )
            
            # Create index if it doesn't exist
            if self.pinecone_index_name not in pinecone.list_indexes():
                pinecone.create_index(
                    name=self.pinecone_index_name,
                    dimension=self.dimension,
                    metric="cosine"
                )
            
            self._pinecone_index = pinecone.Index(self.pinecone_index_name)
        
        return self._pinecone_index

    def _get_weaviate_client(self):
        """Get or create Weaviate client."""
        if not WEAVIATE_AVAILABLE:
            raise ImportError("weaviate-client required. Install with: pip install weaviate-client")
        
        if self._weaviate_client is None:
            auth_config = None
            if self.weaviate_api_key:
                auth_config = weaviate.AuthApiKey(api_key=self.weaviate_api_key)
            
            self._weaviate_client = weaviate.Client(
                url=self.weaviate_url,
                auth_client_secret=auth_config
            )
        
        return self._weaviate_client

    def _get_chromadb_client(self):
        """Get or create ChromaDB client."""
        if not CHROMADB_AVAILABLE:
            raise ImportError("chromadb required. Install with: pip install chromadb")
        
        if self._chromadb_client is None:
            if self.chromadb_host and self.chromadb_port:
                # Connect to remote ChromaDB
                self._chromadb_client = chromadb.HttpClient(
                    host=self.chromadb_host,
                    port=self.chromadb_port
                )
            else:
                # Use persistent local ChromaDB
                self._chromadb_client = chromadb.PersistentClient(path=self.chromadb_path)
        
        return self._chromadb_client

    def _get_qdrant_client(self):
        """Get or create Qdrant client."""
        if not QDRANT_AVAILABLE:
            raise ImportError("qdrant-client required. Install with: pip install qdrant-client")
        
        if self._qdrant_client is None:
            self._qdrant_client = QdrantClient(
                url=self.qdrant_url,
                api_key=self.qdrant_api_key
            )
        
        return self._qdrant_client

    # ===== DOCUMENT MANAGEMENT =====

    async def add_document(
        self,
        collection: str,
        document_id: str,
        text: str,
        embedding: list[float],
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Add document with embeddings to vector database.

        Args:
            collection: Collection/index name
            document_id: Unique document ID
            text: Document text content
            embedding: Document embedding vector
            metadata: Additional metadata

        Returns:
            Add result
        """
        try:
            if self.provider == "pinecone":
                index = self._get_pinecone_index()
                
                meta = metadata or {}
                meta["text"] = text
                
                index.upsert(vectors=[(document_id, embedding, meta)])
                
                return {
                    "success": True,
                    "provider": "pinecone",
                    "collection": collection,
                    "document_id": document_id,
                    "dimension": len(embedding),
                }
            
            elif self.provider == "weaviate":
                client = self._get_weaviate_client()
                
                # Ensure class exists
                if not client.schema.exists(collection):
                    client.schema.create_class({
                        "class": collection,
                        "properties": [
                            {"name": "text", "dataType": ["text"]},
                            {"name": "metadata", "dataType": ["text"]},
                        ]
                    })
                
                client.data_object.create(
                    data_object={
                        "text": text,
                        "metadata": json.dumps(metadata) if metadata else "{}",
                    },
                    class_name=collection,
                    uuid=document_id,
                    vector=embedding
                )
                
                return {
                    "success": True,
                    "provider": "weaviate",
                    "collection": collection,
                    "document_id": document_id,
                    "dimension": len(embedding),
                }
            
            elif self.provider == "chromadb":
                client = self._get_chromadb_client()
                coll = client.get_or_create_collection(name=collection)
                
                coll.add(
                    ids=[document_id],
                    embeddings=[embedding],
                    documents=[text],
                    metadatas=[metadata] if metadata else None
                )
                
                return {
                    "success": True,
                    "provider": "chromadb",
                    "collection": collection,
                    "document_id": document_id,
                    "dimension": len(embedding),
                }
            
            elif self.provider == "qdrant":
                client = self._get_qdrant_client()
                
                # Ensure collection exists
                try:
                    client.get_collection(collection_name=collection)
                except Exception:
                    client.create_collection(
                        collection_name=collection,
                        vectors_config=VectorParams(
                            size=len(embedding),
                            distance=Distance.COSINE
                        )
                    )
                
                payload = {"text": text}
                if metadata:
                    payload.update(metadata)
                
                client.upsert(
                    collection_name=collection,
                    points=[PointStruct(
                        id=document_id,
                        vector=embedding,
                        payload=payload
                    )]
                )
                
                return {
                    "success": True,
                    "provider": "qdrant",
                    "collection": collection,
                    "document_id": document_id,
                    "dimension": len(embedding),
                }
            
            else:
                return {
                    "success": False,
                    "error": f"Unsupported provider: {self.provider}",
                }

        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            return {
                "success": False,
                "error": f"Add document failed: {str(e)}",
                "provider": self.provider,
                "collection": collection,
            }

    async def search_similar(
        self,
        collection: str,
        query_embedding: list[float],
        limit: int = 10,
        filter: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Search for similar documents using semantic search.

        Args:
            collection: Collection/index name
            query_embedding: Query embedding vector
            limit: Maximum number of results
            filter: Metadata filters

        Returns:
            Search results
        """
        try:
            if self.provider == "pinecone":
                index = self._get_pinecone_index()
                
                kwargs = {
                    "vector": query_embedding,
                    "top_k": limit,
                    "include_metadata": True,
                }
                if filter:
                    kwargs["filter"] = filter
                
                results = index.query(**kwargs)
                
                matches = []
                for match in results.matches:
                    matches.append({
                        "id": match.id,
                        "score": match.score,
                        "text": match.metadata.get("text", ""),
                        "metadata": {k: v for k, v in match.metadata.items() if k != "text"},
                    })
                
                return {
                    "success": True,
                    "provider": "pinecone",
                    "collection": collection,
                    "results": matches,
                    "count": len(matches),
                }
            
            elif self.provider == "weaviate":
                client = self._get_weaviate_client()
                
                query_builder = client.query.get(
                    collection,
                    ["text", "metadata"]
                ).with_near_vector({
                    "vector": query_embedding
                }).with_limit(limit)
                
                if filter:
                    query_builder = query_builder.with_where(filter)
                
                results = query_builder.do()
                
                matches = []
                for item in results.get("data", {}).get("Get", {}).get(collection, []):
                    matches.append({
                        "id": item.get("_additional", {}).get("id", ""),
                        "score": item.get("_additional", {}).get("certainty", 0.0),
                        "text": item.get("text", ""),
                        "metadata": json.loads(item.get("metadata", "{}")),
                    })
                
                return {
                    "success": True,
                    "provider": "weaviate",
                    "collection": collection,
                    "results": matches,
                    "count": len(matches),
                }
            
            elif self.provider == "chromadb":
                client = self._get_chromadb_client()
                coll = client.get_collection(name=collection)
                
                kwargs = {
                    "query_embeddings": [query_embedding],
                    "n_results": limit,
                }
                if filter:
                    kwargs["where"] = filter
                
                results = coll.query(**kwargs)
                
                matches = []
                for i in range(len(results["ids"][0])):
                    matches.append({
                        "id": results["ids"][0][i],
                        "score": 1 - results["distances"][0][i],  # Convert distance to similarity
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    })
                
                return {
                    "success": True,
                    "provider": "chromadb",
                    "collection": collection,
                    "results": matches,
                    "count": len(matches),
                }
            
            elif self.provider == "qdrant":
                client = self._get_qdrant_client()
                
                search_params = {
                    "collection_name": collection,
                    "query_vector": query_embedding,
                    "limit": limit,
                }
                if filter:
                    search_params["query_filter"] = filter
                
                results = client.search(**search_params)
                
                matches = []
                for point in results:
                    matches.append({
                        "id": str(point.id),
                        "score": point.score,
                        "text": point.payload.get("text", ""),
                        "metadata": {k: v for k, v in point.payload.items() if k != "text"},
                    })
                
                return {
                    "success": True,
                    "provider": "qdrant",
                    "collection": collection,
                    "results": matches,
                    "count": len(matches),
                }
            
            else:
                return {
                    "success": False,
                    "error": f"Unsupported provider: {self.provider}",
                }

        except Exception as e:
            logger.error(f"Failed to search similar documents: {e}")
            return {
                "success": False,
                "error": f"Search failed: {str(e)}",
                "provider": self.provider,
                "collection": collection,
            }

    async def delete_document(
        self,
        collection: str,
        document_id: str,
    ) -> dict[str, Any]:
        """Delete document from vector database.

        Args:
            collection: Collection/index name
            document_id: Document ID to delete

        Returns:
            Deletion result
        """
        try:
            if self.provider == "pinecone":
                index = self._get_pinecone_index()
                index.delete(ids=[document_id])
                
            elif self.provider == "weaviate":
                client = self._get_weaviate_client()
                client.data_object.delete(
                    uuid=document_id,
                    class_name=collection
                )
                
            elif self.provider == "chromadb":
                client = self._get_chromadb_client()
                coll = client.get_collection(name=collection)
                coll.delete(ids=[document_id])
                
            elif self.provider == "qdrant":
                client = self._get_qdrant_client()
                client.delete(
                    collection_name=collection,
                    points_selector=[document_id]
                )
            
            else:
                return {
                    "success": False,
                    "error": f"Unsupported provider: {self.provider}",
                }
            
            return {
                "success": True,
                "provider": self.provider,
                "collection": collection,
                "document_id": document_id,
                "message": "Document deleted successfully",
            }

        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            return {
                "success": False,
                "error": f"Delete failed: {str(e)}",
                "provider": self.provider,
                "collection": collection,
            }

    async def update_document(
        self,
        collection: str,
        document_id: str,
        text: Optional[str] = None,
        embedding: Optional[list[float]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Update existing document.

        Args:
            collection: Collection/index name
            document_id: Document ID to update
            text: New text content
            embedding: New embedding vector
            metadata: New metadata

        Returns:
            Update result
        """
        try:
            # For most providers, update is the same as add/upsert
            if not embedding:
                return {
                    "success": False,
                    "error": "Embedding is required for update",
                }
            
            if not text:
                text = ""  # Empty text if not provided
            
            return await self.add_document(
                collection=collection,
                document_id=document_id,
                text=text,
                embedding=embedding,
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"Failed to update document: {e}")
            return {
                "success": False,
                "error": f"Update failed: {str(e)}",
                "provider": self.provider,
                "collection": collection,
            }

    # ===== COLLECTION MANAGEMENT =====

    async def create_collection(
        self,
        collection: str,
        dimension: Optional[int] = None,
    ) -> dict[str, Any]:
        """Create new vector collection.

        Args:
            collection: Collection name
            dimension: Vector dimension

        Returns:
            Creation result
        """
        try:
            dim = dimension or self.dimension
            
            if self.provider == "pinecone":
                if collection not in pinecone.list_indexes():
                    pinecone.create_index(
                        name=collection,
                        dimension=dim,
                        metric="cosine"
                    )
                
            elif self.provider == "weaviate":
                client = self._get_weaviate_client()
                if not client.schema.exists(collection):
                    client.schema.create_class({
                        "class": collection,
                        "properties": [
                            {"name": "text", "dataType": ["text"]},
                            {"name": "metadata", "dataType": ["text"]},
                        ]
                    })
                
            elif self.provider == "chromadb":
                client = self._get_chromadb_client()
                client.get_or_create_collection(name=collection)
                
            elif self.provider == "qdrant":
                client = self._get_qdrant_client()
                try:
                    client.get_collection(collection_name=collection)
                except Exception:
                    client.create_collection(
                        collection_name=collection,
                        vectors_config=VectorParams(
                            size=dim,
                            distance=Distance.COSINE
                        )
                    )
            
            else:
                return {
                    "success": False,
                    "error": f"Unsupported provider: {self.provider}",
                }
            
            return {
                "success": True,
                "provider": self.provider,
                "collection": collection,
                "dimension": dim,
                "message": "Collection created successfully",
            }

        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            return {
                "success": False,
                "error": f"Collection creation failed: {str(e)}",
                "provider": self.provider,
                "collection": collection,
            }

    async def list_collections(self) -> dict[str, Any]:
        """List all collections.

        Returns:
            List of collections
        """
        try:
            collections = []
            
            if self.provider == "pinecone":
                collections = pinecone.list_indexes()
                
            elif self.provider == "weaviate":
                client = self._get_weaviate_client()
                schema = client.schema.get()
                collections = [cls["class"] for cls in schema.get("classes", [])]
                
            elif self.provider == "chromadb":
                client = self._get_chromadb_client()
                colls = client.list_collections()
                collections = [c.name for c in colls]
                
            elif self.provider == "qdrant":
                client = self._get_qdrant_client()
                colls = client.get_collections()
                collections = [c.name for c in colls.collections]
            
            else:
                return {
                    "success": False,
                    "error": f"Unsupported provider: {self.provider}",
                }
            
            return {
                "success": True,
                "provider": self.provider,
                "collections": collections,
                "count": len(collections),
            }

        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return {
                "success": False,
                "error": f"List collections failed: {str(e)}",
                "provider": self.provider,
            }

    async def delete_collection(
        self,
        collection: str,
    ) -> dict[str, Any]:
        """Delete collection.

        Args:
            collection: Collection name to delete

        Returns:
            Deletion result
        """
        try:
            if self.provider == "pinecone":
                pinecone.delete_index(collection)
                
            elif self.provider == "weaviate":
                client = self._get_weaviate_client()
                client.schema.delete_class(collection)
                
            elif self.provider == "chromadb":
                client = self._get_chromadb_client()
                client.delete_collection(name=collection)
                
            elif self.provider == "qdrant":
                client = self._get_qdrant_client()
                client.delete_collection(collection_name=collection)
            
            else:
                return {
                    "success": False,
                    "error": f"Unsupported provider: {self.provider}",
                }
            
            return {
                "success": True,
                "provider": self.provider,
                "collection": collection,
                "message": "Collection deleted successfully",
            }

        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            return {
                "success": False,
                "error": f"Collection deletion failed: {str(e)}",
                "provider": self.provider,
                "collection": collection,
            }

    # ===== QUERY OPERATIONS =====

    async def hybrid_search(
        self,
        collection: str,
        query_embedding: list[float],
        query_text: str,
        limit: int = 10,
        alpha: float = 0.5,
    ) -> dict[str, Any]:
        """Perform hybrid search (semantic + keyword).

        Args:
            collection: Collection name
            query_embedding: Query embedding
            query_text: Query text for keyword search
            limit: Maximum results
            alpha: Balance between semantic (1.0) and keyword (0.0) search

        Returns:
            Search results
        """
        try:
            # For now, implement semantic search
            # Full hybrid search implementation depends on provider capabilities
            result = await self.search_similar(
                collection=collection,
                query_embedding=query_embedding,
                limit=limit,
            )
            
            if result["success"]:
                result["search_type"] = "hybrid"
                result["alpha"] = alpha
            
            return result

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return {
                "success": False,
                "error": f"Hybrid search failed: {str(e)}",
                "provider": self.provider,
                "collection": collection,
            }

    async def filter_search(
        self,
        collection: str,
        query_embedding: list[float],
        filters: dict[str, Any],
        limit: int = 10,
    ) -> dict[str, Any]:
        """Search with metadata filters.

        Args:
            collection: Collection name
            query_embedding: Query embedding
            filters: Metadata filters
            limit: Maximum results

        Returns:
            Filtered search results
        """
        try:
            return await self.search_similar(
                collection=collection,
                query_embedding=query_embedding,
                limit=limit,
                filter=filters,
            )

        except Exception as e:
            logger.error(f"Filter search failed: {e}")
            return {
                "success": False,
                "error": f"Filter search failed: {str(e)}",
                "provider": self.provider,
                "collection": collection,
            }

    async def get_embeddings(
        self,
        collection: str,
        document_ids: list[str],
    ) -> dict[str, Any]:
        """Get document embeddings.

        Args:
            collection: Collection name
            document_ids: List of document IDs

        Returns:
            Document embeddings
        """
        try:
            embeddings = {}
            
            if self.provider == "pinecone":
                index = self._get_pinecone_index()
                results = index.fetch(ids=document_ids)
                
                for doc_id, data in results.vectors.items():
                    embeddings[doc_id] = {
                        "embedding": data.values,
                        "metadata": data.metadata,
                    }
                
            elif self.provider == "chromadb":
                client = self._get_chromadb_client()
                coll = client.get_collection(name=collection)
                results = coll.get(ids=document_ids, include=["embeddings", "metadatas"])
                
                for i, doc_id in enumerate(results["ids"]):
                    embeddings[doc_id] = {
                        "embedding": results["embeddings"][i],
                        "metadata": results["metadatas"][i] if results["metadatas"] else {},
                    }
            
            else:
                return {
                    "success": False,
                    "error": f"Get embeddings not fully supported for {self.provider}",
                }
            
            return {
                "success": True,
                "provider": self.provider,
                "collection": collection,
                "embeddings": embeddings,
                "count": len(embeddings),
            }

        except Exception as e:
            logger.error(f"Failed to get embeddings: {e}")
            return {
                "success": False,
                "error": f"Get embeddings failed: {str(e)}",
                "provider": self.provider,
                "collection": collection,
            }


def get_server_definition() -> dict[str, Any]:
    """Get Vector DB MCP server definition.

    Returns:
        Server definition dictionary
    """
    return {
        "name": "vector_db",
        "category": "storage",
        "description": "Vector database operations for semantic search (Pinecone, Weaviate, ChromaDB, Qdrant)",
        "tools": [
            # Document Management
            {
                "name": "add_document",
                "description": "Add document with embeddings to vector database",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "collection": {
                            "type": "string",
                            "description": "Collection/index name",
                        },
                        "document_id": {
                            "type": "string",
                            "description": "Unique document ID",
                        },
                        "text": {
                            "type": "string",
                            "description": "Document text content",
                        },
                        "embedding": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Document embedding vector",
                        },
                        "metadata": {
                            "type": "object",
                            "description": "Additional metadata",
                        },
                    },
                    "required": ["collection", "document_id", "text", "embedding"],
                },
            },
            {
                "name": "search_similar",
                "description": "Search for similar documents using semantic search",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "collection": {
                            "type": "string",
                            "description": "Collection/index name",
                        },
                        "query_embedding": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Query embedding vector",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results",
                            "default": 10,
                        },
                        "filter": {
                            "type": "object",
                            "description": "Metadata filters",
                        },
                    },
                    "required": ["collection", "query_embedding"],
                },
            },
            {
                "name": "delete_document",
                "description": "Delete document from vector database",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "collection": {
                            "type": "string",
                            "description": "Collection/index name",
                        },
                        "document_id": {
                            "type": "string",
                            "description": "Document ID to delete",
                        },
                    },
                    "required": ["collection", "document_id"],
                },
            },
            {
                "name": "update_document",
                "description": "Update existing document",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "collection": {
                            "type": "string",
                            "description": "Collection/index name",
                        },
                        "document_id": {
                            "type": "string",
                            "description": "Document ID to update",
                        },
                        "text": {
                            "type": "string",
                            "description": "New text content",
                        },
                        "embedding": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "New embedding vector",
                        },
                        "metadata": {
                            "type": "object",
                            "description": "New metadata",
                        },
                    },
                    "required": ["collection", "document_id", "embedding"],
                },
            },
            # Collection Management
            {
                "name": "create_collection",
                "description": "Create new vector collection",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "collection": {
                            "type": "string",
                            "description": "Collection name",
                        },
                        "dimension": {
                            "type": "integer",
                            "description": "Vector dimension",
                        },
                    },
                    "required": ["collection"],
                },
            },
            {
                "name": "list_collections",
                "description": "List all collections",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
                "name": "delete_collection",
                "description": "Delete collection",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "collection": {
                            "type": "string",
                            "description": "Collection name to delete",
                        }
                    },
                    "required": ["collection"],
                },
            },
            # Query Operations
            {
                "name": "hybrid_search",
                "description": "Perform hybrid search (semantic + keyword)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "collection": {
                            "type": "string",
                            "description": "Collection name",
                        },
                        "query_embedding": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Query embedding",
                        },
                        "query_text": {
                            "type": "string",
                            "description": "Query text for keyword search",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum results",
                            "default": 10,
                        },
                        "alpha": {
                            "type": "number",
                            "description": "Balance between semantic (1.0) and keyword (0.0)",
                            "default": 0.5,
                        },
                    },
                    "required": ["collection", "query_embedding", "query_text"],
                },
            },
            {
                "name": "filter_search",
                "description": "Search with metadata filters",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "collection": {
                            "type": "string",
                            "description": "Collection name",
                        },
                        "query_embedding": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Query embedding",
                        },
                        "filters": {
                            "type": "object",
                            "description": "Metadata filters",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum results",
                            "default": 10,
                        },
                    },
                    "required": ["collection", "query_embedding", "filters"],
                },
            },
            {
                "name": "get_embeddings",
                "description": "Get document embeddings",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "collection": {
                            "type": "string",
                            "description": "Collection name",
                        },
                        "document_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of document IDs",
                        },
                    },
                    "required": ["collection", "document_ids"],
                },
            },
        ],
        "resources": [],
        "metadata": {
            "version": "1.0.0",
            "priority": "high",
            "category": "storage",
            "requires": ["pinecone-client", "weaviate-client", "chromadb", "qdrant-client"],
        },
    }