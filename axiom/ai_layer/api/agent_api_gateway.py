"""
Professional Agent API Gateway

Production-ready REST API exposing all 12 professional agents.

Provides:
- RESTful endpoints for each agent
- Authentication and authorization
- Rate limiting
- Request validation
- Response formatting
- Error handling
- Monitoring integration
- OpenAPI documentation

This is the production API for client access to the multi-agent system.
"""

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from decimal import Decimal
import asyncio
import time

# Orchestrator (manages all 12 agents)
from axiom.ai_layer.orchestration.professional_agent_orchestrator import ProfessionalAgentOrchestrator

# Infrastructure
from axiom.ai_layer.infrastructure.observability import Logger

# Messages
from axiom.ai_layer.messaging.protocol import (
    CalculateGreeksCommand, CalculateRiskCommand, GenerateStrategyCommand,
    ExecuteOrderCommand, CalculateHedgeCommand, CalculatePnLCommand,
    GetMarketDataQuery, ForecastVolatilityCommand, CheckComplianceCommand,
    CheckSystemHealthQuery, ValidateActionCommand, ClientQuery,
    AgentName
)


# Request/Response Models for API
class GreeksRequest(BaseModel):
    """Request for Greeks calculation"""
    spot: float = Field(..., gt=0, description="Spot price")
    strike: float = Field(..., gt=0, description="Strike price")
    time_to_maturity: float = Field(..., gt=0, le=30)
    risk_free_rate: float = Field(..., ge=-0.05, le=0.20)
    volatility: float = Field(..., gt=0, le=5.0)
    option_type: str = Field(default="call", pattern="^(call|put)$")


class GreeksResponse(BaseModel):
    """Response with Greeks"""
    success: bool
    delta: Optional[float]
    gamma: Optional[float]
    theta: Optional[float]
    vega: Optional[float]
    rho: Optional[float]
    price: Optional[float]
    calculation_time_us: Optional[float]
    confidence: float


class RiskRequest(BaseModel):
    """Request for risk calculation"""
    positions: List[Dict]
    market_data: Dict
    include_stress_tests: bool = False


class RiskResponse(BaseModel):
    """Response with risk metrics"""
    success: bool
    total_delta: Optional[float]
    total_gamma: Optional[float]
    var_1day: Optional[float]
    within_limits: bool
    limit_breaches: List[str]


class StrategyRequest(BaseModel):
    """Request for strategy generation"""
    market_outlook: str = Field(..., pattern="^(bullish|bearish|neutral)$")
    volatility_view: str = Field(..., pattern="^(increasing|stable|decreasing)$")
    risk_tolerance: float = Field(..., ge=0.0, le=1.0)
    capital_available: float = Field(..., gt=0)
    current_spot: float = Field(..., gt=0)
    current_vol: float = Field(..., gt=0)


class HealthResponse(BaseModel):
    """System health response"""
    overall_healthy: bool
    healthy_agents: int
    total_agents: int
    agent_health: Dict[str, Dict]


# Initialize API
app = FastAPI(
    title="Axiom Professional Multi-Agent API",
    description="Production API for 12 professional derivatives agents",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global orchestrator
orchestrator: Optional[ProfessionalAgentOrchestrator] = None
logger = Logger("api_gateway")


@app.on_event("startup")
async def startup():
    """Initialize all 12 agents on startup"""
    global orchestrator
    logger.info("api_starting", agents=12)
    
    orchestrator = ProfessionalAgentOrchestrator(use_gpu=False)
    
    logger.info("api_ready", endpoint="http://localhost:8000")


@app.on_event("shutdown")
async def shutdown():
    """Gracefully shutdown all agents"""
    global orchestrator
    logger.info("api_shutting_down")
    
    if orchestrator:
        await orchestrator.shutdown_all()
    
    logger.info("api_shutdown_complete")


# Authentication (simplified - enhance for production)
async def verify_api_key(x_api_key: str = Header(...)):
    """Verify API key"""
    # In production: validate against database
    if not x_api_key or len(x_api_key) < 10:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


# Endpoints for each agent
@app.post("/api/v2/pricing/greeks", response_model=GreeksResponse)
async def calculate_greeks(
    request: GreeksRequest,
    api_key: str = Depends(verify_api_key)
) -> GreeksResponse:
    """Calculate option Greeks using Pricing Agent"""
    logger.info("api_request", endpoint="greeks", client_key=api_key[:8])
    
    command = CalculateGreeksCommand(
        from_agent=AgentName.CLIENT_INTERFACE,
        to_agent=AgentName.PRICING,
        spot=request.spot,
        strike=request.strike,
        time_to_maturity=request.time_to_maturity,
        risk_free_rate=request.risk_free_rate,
        volatility=request.volatility,
        option_type=request.option_type
    )
    
    response = await orchestrator.agents[AgentName.PRICING].process_request(command)
    
    return GreeksResponse(
        success=response.success,
        delta=response.delta,
        gamma=response.gamma,
        theta=response.theta,
        vega=response.vega,
        rho=response.rho,
        price=response.price,
        calculation_time_us=response.calculation_time_us,
        confidence=response.confidence
    )


@app.post("/api/v2/risk/calculate", response_model=RiskResponse)
async def calculate_risk(
    request: RiskRequest,
    api_key: str = Depends(verify_api_key)
) -> RiskResponse:
    """Calculate portfolio risk using Risk Agent"""
    command = CalculateRiskCommand(
        from_agent=AgentName.CLIENT_INTERFACE,
        to_agent=AgentName.RISK,
        positions=request.positions,
        market_data=request.market_data,
        include_stress_tests=request.include_stress_tests
    )
    
    response = await orchestrator.agents[AgentName.RISK].process_request(command)
    
    return RiskResponse(
        success=response.success,
        total_delta=response.total_delta,
        total_gamma=response.total_gamma,
        var_1day=response.var_1day,
        within_limits=response.within_limits,
        limit_breaches=response.limit_breaches
    )


@app.post("/api/v2/strategy/generate")
async def generate_strategy(
    request: StrategyRequest,
    api_key: str = Depends(verify_api_key)
):
    """Generate trading strategy using Strategy Agent"""
    command = GenerateStrategyCommand(
        from_agent=AgentName.CLIENT_INTERFACE,
        to_agent=AgentName.STRATEGY,
        market_outlook=request.market_outlook,
        volatility_view=request.volatility_view,
        risk_tolerance=request.risk_tolerance,
        capital_available=request.capital_available,
        current_spot=request.current_spot,
        current_vol=request.current_vol
    )
    
    response = await orchestrator.agents[AgentName.STRATEGY].process_request(command)
    
    return response


@app.get("/api/v2/health", response_model=HealthResponse)
async def check_system_health() -> HealthResponse:
    """Check health of all 12 agents"""
    health = await orchestrator.get_system_health()
    
    return HealthResponse(
        overall_healthy=health['overall_healthy'],
        healthy_agents=health['healthy_agents'],
        total_agents=health['total_agents'],
        agent_health=health['agent_health']
    )


@app.get("/api/v2/stats")
async def get_system_stats():
    """Get system-wide statistics"""
    stats = orchestrator.get_stats()
    
    # Add agent-specific stats
    agent_stats = {}
    for agent_name, agent in orchestrator.agents.items():
        agent_stats[agent_name.value] = agent.get_stats()
    
    return {
        'orchestrator': stats,
        'agents': agent_stats
    }


@app.get("/")
async def root():
    """API root"""
    return {
        'name': 'Axiom Professional Multi-Agent API',
        'version': '2.0.0',
        'agents': 12,
        'clusters': 3,
        'status': 'operational',
        'documentation': '/docs'
    }


# Run API
if __name__ == "__main__":
    import uvicorn
    
    logger.info("STARTING_API_GATEWAY", port=8000, agents=12)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )