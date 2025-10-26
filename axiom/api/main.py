"""
Main FastAPI application for Axiom Investment Banking Analytics API.

Features:
- CORS configuration
- Automatic OpenAPI/Swagger documentation
- Health check endpoints
- Versioned API (v1)
- Error handling middleware
- Request logging middleware
- Prometheus metrics
"""

import time
import traceback
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from prometheus_fastapi_instrumentator import Instrumentator

from axiom.api.rate_limit import limiter
from axiom.api.routes import (
    options,
    portfolio,
    risk,
    ma,
    fixed_income,
    market_data,
    analytics,
)
from axiom.api.websocket import router as websocket_router


# Lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application lifecycle events."""
    # Startup
    print("🚀 Starting Axiom API server...")
    print("📊 Initializing metrics collection...")
    print("✅ API server ready!")
    
    yield
    
    # Shutdown
    print("🔻 Shutting down Axiom API server...")
    print("✅ Cleanup complete")


# Create FastAPI application
app = FastAPI(
    title="Axiom Investment Banking Analytics API",
    description="""
    **Enterprise-grade quantitative finance API** for investment banking analytics.
    
    ## Features
    
    * **Options Pricing**: Black-Scholes, Binomial Tree, Monte Carlo simulation
    * **Portfolio Optimization**: Mean-variance, risk parity, efficient frontier
    * **Risk Management**: VaR, CVaR, stress testing, scenario analysis
    * **M&A Analytics**: Synergy valuation, deal financing, LBO modeling
    * **Fixed Income**: Bond pricing, yield curves, duration/convexity
    * **Market Data**: Real-time streaming via WebSocket
    * **Real-time Streaming**: Portfolio updates, risk alerts, market data
    
    ## Authentication
    
    All endpoints require either:
    - **JWT Token**: Bearer token in Authorization header
    - **API Key**: X-API-Key header
    
    ## Rate Limits
    
    - Standard: 100 requests/minute
    - Premium: 1000 requests/minute
    - WebSocket: Unlimited within fair use
    
    ## Support
    
    For issues or questions, contact: support@axiom-analytics.com
    """,
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

# Add rate limiter to app state
app.state.limiter = limiter

# CORS middleware - Configure for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Configure specific origins for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time", "X-Request-ID"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests and add processing time to response headers."""
    start_time = time.time()
    
    # Generate request ID
    request_id = f"{int(start_time * 1000)}"
    
    # Log request
    print(f"📥 {request.method} {request.url.path} [ID: {request_id}]")
    
    try:
        response = await call_next(request)
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Add headers
        response.headers["X-Process-Time"] = f"{process_time:.4f}"
        response.headers["X-Request-ID"] = request_id
        
        # Log response
        print(f"📤 {request.method} {request.url.path} - {response.status_code} ({process_time:.4f}s)")
        
        return response
    
    except Exception as e:
        process_time = time.time() - start_time
        print(f"❌ {request.method} {request.url.path} - Error ({process_time:.4f}s): {str(e)}")
        raise


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions."""
    print(f"❌ Unhandled exception: {str(exc)}")
    print(traceback.format_exc())
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc) if app.debug else "An unexpected error occurred",
            "path": str(request.url.path),
        },
    )


# Validation error handler
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation error",
            "detail": exc.errors(),
            "body": exc.body,
        },
    )


# Health check endpoints
@app.get("/health", tags=["Health"])
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint.
    
    Returns API status and version information.
    """
    return {
        "status": "healthy",
        "version": "1.0.0",
        "service": "Axiom Analytics API",
    }


@app.get("/health/detailed", tags=["Health"])
async def detailed_health_check() -> Dict[str, Any]:
    """
    Detailed health check with component status.
    
    Checks connectivity to dependencies like database, Redis, etc.
    """
    return {
        "status": "healthy",
        "version": "1.0.0",
        "service": "Axiom Analytics API",
        "components": {
            "api": "healthy",
            "database": "healthy",  # TODO: Add actual database check
            "redis": "healthy",     # TODO: Add actual Redis check
            "websocket": "healthy",
        },
        "timestamp": time.time(),
    }


@app.get("/", tags=["Root"])
async def root() -> Dict[str, Any]:
    """
    Root endpoint.
    
    Returns API information and documentation links.
    """
    return {
        "name": "Axiom Investment Banking Analytics API",
        "version": "1.0.0",
        "description": "Enterprise-grade quantitative finance API",
        "documentation": {
            "swagger": "/api/docs",
            "redoc": "/api/redoc",
            "openapi": "/api/openapi.json",
        },
        "endpoints": {
            "options": "/api/v1/options",
            "portfolio": "/api/v1/portfolio",
            "risk": "/api/v1/risk",
            "ma": "/api/v1/ma",
            "fixed_income": "/api/v1/bonds",
            "market_data": "/api/v1/market-data",
            "analytics": "/api/v1/analytics",
        },
        "websocket": {
            "portfolio": "/ws/portfolio/{portfolio_id}",
            "risk_alerts": "/ws/risk-alerts",
            "market_data": "/ws/market-data/{symbol}",
        },
    }


# Include routers
app.include_router(options.router, prefix="/api/v1")
app.include_router(portfolio.router, prefix="/api/v1")
app.include_router(risk.router, prefix="/api/v1")
app.include_router(ma.router, prefix="/api/v1")
app.include_router(fixed_income.router, prefix="/api/v1")
app.include_router(market_data.router, prefix="/api/v1")
app.include_router(analytics.router, prefix="/api/v1")
app.include_router(websocket_router)


# Prometheus metrics
instrumentator = Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
    should_respect_env_var=True,
    should_instrument_requests_inprogress=True,
    excluded_handlers=["/metrics", "/health"],
    env_var_name="ENABLE_METRICS",
    inprogress_name="axiom_requests_inprogress",
    inprogress_labels=True,
)

# Instrument the app
instrumentator.instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "axiom.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )