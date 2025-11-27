"""
FastAPI Service for RAG System.

Endpoints:
- POST /ingest - Ingest documents
- POST /query - Query the RAG system
- GET /health - Health check
- GET /stats - System statistics
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from .rag_pipeline import RAGPipeline, RAGConfig, RAGResponse
from .document_processor import DocumentChunk

logger = logging.getLogger(__name__)


# Request/Response models
class QueryRequest(BaseModel):
    """Query request model."""
    query: str = Field(..., description="Query text")
    top_k: Optional[int] = Field(None, description="Number of results")
    enable_graph: Optional[bool] = Field(None, description="Enable graph enhancement")
    temperature: Optional[float] = Field(None, description="Generation temperature")


class QueryResponse(BaseModel):
    """Query response model."""
    query: str
    answer: str
    confidence: float
    sources: List[Dict[str, Any]]
    reasoning_steps: List[str]
    performance: Dict[str, float]
    metadata: Dict[str, Any]


class IngestRequest(BaseModel):
    """Ingest request model."""
    document_urls: Optional[List[str]] = Field(None, description="Document URLs to ingest")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class IngestResponse(BaseModel):
    """Ingest response model."""
    success: bool
    chunks_indexed: int
    documents_processed: int
    message: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime
    components: Dict[str, bool]
    version: str


class StatsResponse(BaseModel):
    """Statistics response."""
    pipeline: Dict[str, Any]
    retriever: Dict[str, Any]
    embedding: Dict[str, Any]
    uptime_seconds: float


# Create FastAPI app
app = FastAPI(
    title="RAG Intelligence API",
    description="Production RAG system for M&A intelligence with vector search and graph enhancement",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline: Optional[RAGPipeline] = None
start_time: datetime = datetime.now()


@app.on_event("startup")
async def startup_event():
    """Initialize RAG pipeline on startup."""
    global pipeline
    
    try:
        config = RAGConfig(
            top_k=10,
            enable_graph=True,
            enable_reranking=True,
            enable_citations=True
        )
        
        pipeline = RAGPipeline(config=config)
        logger.info("RAG pipeline initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down RAG API service")


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Query the RAG system.
    
    Args:
        request: Query request with parameters
        
    Returns:
        Query response with answer and sources
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    try:
        # Execute query
        response = pipeline.query(
            query=request.query,
            top_k=request.top_k,
            enable_graph=request.enable_graph,
            temperature=request.temperature
        )
        
        # Format response
        return QueryResponse(
            query=response.query,
            answer=response.answer,
            confidence=response.confidence,
            sources=response.sources,
            reasoning_steps=response.reasoning_steps,
            performance={
                "total_ms": response.total_time_ms,
                "retrieval_ms": response.retrieval_time_ms,
                "generation_ms": response.generation_time_ms
            },
            metadata={
                "model": response.model_used,
                "tokens": response.tokens_used,
                "sources_count": len(response.sources),
                "graph_enhanced": response.retrieval_context.graph_enhanced
            }
        )
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(request: IngestRequest):
    """
    Ingest documents into RAG system.
    
    Args:
        request: Ingest request with document paths/URLs
        
    Returns:
        Ingest response with statistics
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    try:
        # For now, only support local file paths
        # In production, add S3/URL download support
        if not request.document_urls:
            raise HTTPException(status_code=400, detail="No documents provided")
        
        chunks_indexed = pipeline.ingest_documents(
            file_paths=request.document_urls,
            metadata=request.metadata
        )
        
        return IngestResponse(
            success=True,
            chunks_indexed=chunks_indexed,
            documents_processed=len(request.document_urls),
            message=f"Successfully ingested {len(request.document_urls)} documents"
        )
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/upload")
async def upload_and_ingest(
    files: List[UploadFile] = File(...),
    metadata: Optional[str] = None
):
    """
    Upload and ingest documents.
    
    Args:
        files: List of files to upload
        metadata: Optional JSON metadata
        
    Returns:
        Ingest response
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    try:
        import tempfile
        import os
        import json
        
        # Parse metadata
        meta_dict = json.loads(metadata) if metadata else {}
        
        # Save files temporarily and ingest
        temp_paths = []
        for file in files:
            # Create temp file
            suffix = os.path.splitext(file.filename)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                content = await file.read()
                tmp.write(content)
                temp_paths.append(tmp.name)
        
        # Ingest
        chunks_indexed = pipeline.ingest_documents(
            file_paths=temp_paths,
            metadata=meta_dict
        )
        
        # Cleanup temp files
        for path in temp_paths:
            try:
                os.unlink(path)
            except:
                pass
        
        return IngestResponse(
            success=True,
            chunks_indexed=chunks_indexed,
            documents_processed=len(files),
            message=f"Successfully uploaded and ingested {len(files)} documents"
        )
        
    except Exception as e:
        logger.error(f"Upload and ingest failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Health status of all components
    """
    components = {
        "pipeline": pipeline is not None,
        "embedding_service": False,
        "vector_store": False,
        "graph_db": False,
        "claude_api": False
    }
    
    if pipeline:
        try:
            # Check embedding service
            components["embedding_service"] = pipeline.embedding_service is not None
            
            # Check vector store
            components["vector_store"] = pipeline.retriever.vector_store.health_check()
            
            # Check graph DB
            if pipeline.retriever.graph_enhancer:
                components["graph_db"] = pipeline.retriever.graph_enhancer.graph.health_check()
            
            # Check Claude API (simple test)
            components["claude_api"] = pipeline.claude is not None
            
        except Exception as e:
            logger.warning(f"Health check component failure: {e}")
    
    all_healthy = all(components.values())
    
    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        timestamp=datetime.now(),
        components=components,
        version="1.0.0"
    )


@app.get("/stats", response_model=StatsResponse)
async def get_statistics():
    """
    Get system statistics.
    
    Returns:
        Statistics from all components
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    try:
        stats = pipeline.get_statistics()
        
        uptime = (datetime.now() - start_time).total_seconds()
        
        return StatsResponse(
            pipeline=stats.get("pipeline", {}),
            retriever=stats.get("retriever", {}),
            embedding=stats.get("embedding", {}),
            uptime_seconds=uptime
        )
        
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "RAG Intelligence API",
        "version": "1.0.0",
        "description": "Production RAG system for M&A intelligence",
        "endpoints": {
            "query": "POST /query",
            "ingest": "POST /ingest",
            "upload": "POST /ingest/upload",
            "health": "GET /health",
            "stats": "GET /stats"
        }
    }


def start_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False
):
    """
    Start the API server.
    
    Args:
        host: Host to bind to
        port: Port to listen on
        reload: Enable auto-reload
    """
    uvicorn.run(
        "axiom.models.rag.api_service:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    start_server()


__all__ = ["app", "start_server"]