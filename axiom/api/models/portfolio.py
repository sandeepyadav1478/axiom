"""
Pydantic models for Portfolio API endpoints.

Models for:
- Portfolio optimization
- Efficient frontier
- Risk metrics
- Rebalancing
- Performance attribution
"""

from typing import List, Dict, Optional
from enum import Enum
from pydantic import BaseModel, Field, validator


class OptimizationMethod(str, Enum):
    """Portfolio optimization methods."""
    MEAN_VARIANCE = "mean_variance"
    MIN_VARIANCE = "min_variance"
    MAX_SHARPE = "max_sharpe"
    RISK_PARITY = "risk_parity"
    MAX_RETURN = "max_return"


class RiskMeasure(str, Enum):
    """Risk measurement methods."""
    VOLATILITY = "volatility"
    VAR = "var"
    CVAR = "cvar"
    MAX_DRAWDOWN = "max_drawdown"


# Asset class
class Asset(BaseModel):
    """Asset representation."""
    
    symbol: str = Field(..., description="Asset symbol/ticker")
    name: Optional[str] = Field(None, description="Asset name")
    weight: Optional[float] = Field(None, ge=0, le=1, description="Portfolio weight")
    price: Optional[float] = Field(None, gt=0, description="Current price")
    quantity: Optional[int] = Field(None, ge=0, description="Number of shares")


# Portfolio Optimization
class PortfolioOptimizationRequest(BaseModel):
    """Request for portfolio optimization."""
    
    assets: List[str] = Field(..., min_items=2, max_items=100, description="Asset symbols")
    expected_returns: List[float] = Field(..., description="Expected returns for each asset")
    covariance_matrix: List[List[float]] = Field(..., description="Covariance matrix")
    method: OptimizationMethod = Field(..., description="Optimization method")
    target_return: Optional[float] = Field(None, description="Target return (for some methods)")
    risk_free_rate: Optional[float] = Field(0.02, ge=0, le=1, description="Risk-free rate")
    constraints: Optional[Dict[str, float]] = Field(
        None,
        description="Constraints: min_weight, max_weight, etc."
    )
    
    @validator('covariance_matrix')
    def validate_covariance_matrix(cls, v, values):
        """Validate covariance matrix dimensions."""
        if 'assets' in values:
            n = len(values['assets'])
            if len(v) != n or any(len(row) != n for row in v):
                raise ValueError(f"Covariance matrix must be {n}x{n}")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "assets": ["AAPL", "GOOGL", "MSFT"],
                "expected_returns": [0.12, 0.15, 0.10],
                "covariance_matrix": [
                    [0.04, 0.01, 0.02],
                    [0.01, 0.05, 0.015],
                    [0.02, 0.015, 0.03]
                ],
                "method": "max_sharpe",
                "risk_free_rate": 0.02
            }
        }


class PortfolioOptimizationResponse(BaseModel):
    """Response for portfolio optimization."""
    
    weights: Dict[str, float] = Field(..., description="Optimal weights for each asset")
    expected_return: float = Field(..., description="Expected portfolio return")
    volatility: float = Field(..., description="Portfolio volatility/risk")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    method_used: str = Field(..., description="Optimization method used")
    converged: bool = Field(..., description="Whether optimization converged")
    execution_time_ms: float
    
    class Config:
        schema_extra = {
            "example": {
                "weights": {
                    "AAPL": 0.30,
                    "GOOGL": 0.45,
                    "MSFT": 0.25
                },
                "expected_return": 0.1325,
                "volatility": 0.1823,
                "sharpe_ratio": 0.617,
                "method_used": "max_sharpe",
                "converged": True,
                "execution_time_ms": 45.3
            }
        }


# Efficient Frontier
class EfficientFrontierRequest(BaseModel):
    """Request for efficient frontier calculation."""
    
    assets: List[str] = Field(..., min_items=2, max_items=100)
    expected_returns: List[float]
    covariance_matrix: List[List[float]]
    risk_free_rate: Optional[float] = Field(0.02, ge=0, le=1)
    num_portfolios: Optional[int] = Field(100, ge=10, le=1000, description="Number of portfolios")


class FrontierPoint(BaseModel):
    """Single point on efficient frontier."""
    
    return_value: float = Field(..., alias="return")
    risk: float
    sharpe_ratio: float
    weights: Dict[str, float]
    
    class Config:
        populate_by_name = True


class EfficientFrontierResponse(BaseModel):
    """Response for efficient frontier."""
    
    frontier: List[FrontierPoint] = Field(..., description="Points on efficient frontier")
    min_variance_portfolio: FrontierPoint = Field(..., description="Minimum variance portfolio")
    max_sharpe_portfolio: FrontierPoint = Field(..., description="Maximum Sharpe ratio portfolio")
    execution_time_ms: float


# Portfolio Metrics
class PortfolioMetricsRequest(BaseModel):
    """Request for portfolio metrics."""
    
    assets: List[Asset] = Field(..., min_items=1)
    returns_history: Optional[List[List[float]]] = Field(None, description="Historical returns")
    benchmark_returns: Optional[List[float]] = Field(None, description="Benchmark returns")


class PortfolioMetricsResponse(BaseModel):
    """Response with portfolio metrics."""
    
    total_value: float = Field(..., description="Total portfolio value")
    total_return: float = Field(..., description="Total return")
    annualized_return: float = Field(..., description="Annualized return")
    volatility: float = Field(..., description="Annualized volatility")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    sortino_ratio: float = Field(..., description="Sortino ratio")
    max_drawdown: float = Field(..., description="Maximum drawdown")
    var_95: Optional[float] = Field(None, description="95% VaR")
    cvar_95: Optional[float] = Field(None, description="95% CVaR")
    beta: Optional[float] = Field(None, description="Beta vs benchmark")
    alpha: Optional[float] = Field(None, description="Alpha vs benchmark")
    tracking_error: Optional[float] = Field(None, description="Tracking error")
    information_ratio: Optional[float] = Field(None, description="Information ratio")
    execution_time_ms: float


# Rebalancing
class RebalancingRequest(BaseModel):
    """Request for portfolio rebalancing."""
    
    current_holdings: Dict[str, float] = Field(..., description="Current holdings (symbol: quantity)")
    target_weights: Dict[str, float] = Field(..., description="Target weights")
    current_prices: Dict[str, float] = Field(..., description="Current prices")
    transaction_cost: Optional[float] = Field(0.001, ge=0, le=0.1, description="Transaction cost rate")
    min_trade_amount: Optional[float] = Field(0.0, ge=0, description="Minimum trade amount")


class Trade(BaseModel):
    """Trade instruction."""
    
    symbol: str
    action: str = Field(..., description="'buy' or 'sell'")
    quantity: float
    price: float
    value: float
    cost: float


class RebalancingResponse(BaseModel):
    """Response for rebalancing."""
    
    trades: List[Trade] = Field(..., description="Rebalancing trades")
    total_cost: float = Field(..., description="Total transaction costs")
    current_weights: Dict[str, float]
    target_weights: Dict[str, float]
    new_weights: Dict[str, float] = Field(..., description="Weights after rebalancing")
    tracking_error: float = Field(..., description="Distance from target")
    execution_time_ms: float


# Performance Attribution
class PerformanceAttributionRequest(BaseModel):
    """Request for performance attribution."""
    
    portfolio_returns: List[float] = Field(..., min_items=10)
    benchmark_returns: List[float] = Field(..., min_items=10)
    sector_weights: Optional[Dict[str, float]] = Field(None)
    sector_returns: Optional[Dict[str, List[float]]] = Field(None)


class PerformanceAttributionResponse(BaseModel):
    """Response for performance attribution."""
    
    total_return: float
    benchmark_return: float
    excess_return: float = Field(..., description="Portfolio return - benchmark return")
    allocation_effect: Optional[float] = Field(None, description="Return from allocation decisions")
    selection_effect: Optional[float] = Field(None, description="Return from security selection")
    interaction_effect: Optional[float] = Field(None, description="Interaction between allocation and selection")
    execution_time_ms: float


# Batch Operations
class BatchOptimizationRequest(BaseModel):
    """Request for batch portfolio optimization."""
    
    portfolios: List[PortfolioOptimizationRequest] = Field(..., min_items=1, max_items=50)


class BatchOptimizationResponse(BaseModel):
    """Response for batch optimization."""
    
    results: List[PortfolioOptimizationResponse]
    total_execution_time_ms: float
    successful: int
    failed: int