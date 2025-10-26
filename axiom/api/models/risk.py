"""
Pydantic models for Risk API endpoints.
"""

from typing import List, Dict, Optional
from enum import Enum
from pydantic import BaseModel, Field


class VaRMethod(str, Enum):
    """VaR calculation methods."""
    PARAMETRIC = "parametric"
    HISTORICAL = "historical"
    MONTE_CARLO = "monte_carlo"


class VaRRequest(BaseModel):
    """Request for Value at Risk calculation."""
    
    returns: List[float] = Field(..., min_items=30, description="Historical returns")
    confidence_level: float = Field(0.95, ge=0.9, le=0.99, description="Confidence level")
    method: VaRMethod = Field(VaRMethod.PARAMETRIC, description="VaR method")
    time_horizon: int = Field(1, ge=1, le=30, description="Time horizon in days")
    portfolio_value: float = Field(..., gt=0, description="Current portfolio value")


class VaRResponse(BaseModel):
    """Response for VaR calculation."""
    
    var: float = Field(..., description="Value at Risk")
    cvar: float = Field(..., description="Conditional VaR (Expected Shortfall)")
    confidence_level: float
    method: str
    execution_time_ms: float


class StressTestRequest(BaseModel):
    """Request for stress testing."""
    
    portfolio_returns: List[float] = Field(..., min_items=30)
    scenarios: Dict[str, Dict[str, float]] = Field(..., description="Stress scenarios")


class StressTestResponse(BaseModel):
    """Response for stress testing."""
    
    scenario_results: Dict[str, float]
    worst_case: Dict[str, float]
    best_case: Dict[str, float]
    execution_time_ms: float


class RiskMetricsResponse(BaseModel):
    """Portfolio risk metrics."""
    
    volatility: float
    var_95: float
    cvar_95: float
    max_drawdown: float
    beta: Optional[float] = None
    execution_time_ms: float