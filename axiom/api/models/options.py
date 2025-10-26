"""
Pydantic models for Options API endpoints.

Models for:
- Option pricing (Black-Scholes, Binomial, Monte Carlo)
- Greeks calculation
- Implied volatility
- Option chain analysis
"""

from typing import Optional, Dict, List
from enum import Enum
from pydantic import BaseModel, Field, validator


class OptionType(str, Enum):
    """Option type enumeration."""
    CALL = "call"
    PUT = "put"


class PricingModel(str, Enum):
    """Pricing model enumeration."""
    BLACK_SCHOLES = "black_scholes"
    BINOMIAL = "binomial"
    MONTE_CARLO = "monte_carlo"


# Option Pricing Models
class OptionPriceRequest(BaseModel):
    """Request model for option pricing."""
    
    spot_price: float = Field(..., gt=0, description="Current price of underlying asset")
    strike: float = Field(..., gt=0, description="Strike price of the option")
    time_to_expiry: float = Field(..., gt=0, le=30, description="Time to expiration in years")
    risk_free_rate: float = Field(..., ge=0, le=1, description="Risk-free interest rate (decimal)")
    volatility: float = Field(..., gt=0, le=5, description="Volatility (decimal)")
    option_type: OptionType = Field(..., description="Option type: 'call' or 'put'")
    dividend_yield: Optional[float] = Field(0.0, ge=0, le=1, description="Dividend yield (decimal)")
    model: Optional[PricingModel] = Field(
        PricingModel.BLACK_SCHOLES,
        description="Pricing model to use"
    )
    
    # Model-specific parameters
    steps: Optional[int] = Field(100, ge=10, le=1000, description="Steps for binomial/MC (if applicable)")
    simulations: Optional[int] = Field(10000, ge=1000, le=1000000, description="Simulations for Monte Carlo")
    
    class Config:
        schema_extra = {
            "example": {
                "spot_price": 100.0,
                "strike": 100.0,
                "time_to_expiry": 1.0,
                "risk_free_rate": 0.05,
                "volatility": 0.25,
                "option_type": "call",
                "dividend_yield": 0.0,
                "model": "black_scholes"
            }
        }


class Greeks(BaseModel):
    """Option Greeks."""
    
    delta: float = Field(..., description="Rate of change of option price w.r.t. underlying")
    gamma: float = Field(..., description="Rate of change of delta w.r.t. underlying")
    theta: float = Field(..., description="Rate of change of option price w.r.t. time")
    vega: float = Field(..., description="Rate of change of option price w.r.t. volatility")
    rho: float = Field(..., description="Rate of change of option price w.r.t. interest rate")


class OptionPriceResponse(BaseModel):
    """Response model for option pricing."""
    
    price: float = Field(..., description="Calculated option price")
    greeks: Greeks = Field(..., description="Option Greeks")
    intrinsic_value: float = Field(..., description="Intrinsic value of the option")
    time_value: float = Field(..., description="Time value of the option")
    model_used: str = Field(..., description="Pricing model used")
    execution_time_ms: float = Field(..., description="Execution time in milliseconds")
    
    class Config:
        schema_extra = {
            "example": {
                "price": 10.45,
                "greeks": {
                    "delta": 0.54,
                    "gamma": 0.019,
                    "theta": -0.023,
                    "vega": 0.38,
                    "rho": 0.47
                },
                "intrinsic_value": 0.0,
                "time_value": 10.45,
                "model_used": "black_scholes",
                "execution_time_ms": 2.3
            }
        }


# Greeks Calculation
class OptionGreeksRequest(BaseModel):
    """Request model for Greeks calculation."""
    
    spot_price: float = Field(..., gt=0)
    strike: float = Field(..., gt=0)
    time_to_expiry: float = Field(..., gt=0, le=30)
    risk_free_rate: float = Field(..., ge=0, le=1)
    volatility: float = Field(..., gt=0, le=5)
    option_type: OptionType
    dividend_yield: Optional[float] = Field(0.0, ge=0, le=1)


class OptionGreeksResponse(BaseModel):
    """Response model for Greeks calculation."""
    
    greeks: Greeks
    spot_price: float
    execution_time_ms: float


# Implied Volatility
class ImpliedVolatilityRequest(BaseModel):
    """Request model for implied volatility calculation."""
    
    spot_price: float = Field(..., gt=0, description="Current price of underlying")
    strike: float = Field(..., gt=0, description="Strike price")
    time_to_expiry: float = Field(..., gt=0, le=30, description="Time to expiration")
    risk_free_rate: float = Field(..., ge=0, le=1, description="Risk-free rate")
    option_price: float = Field(..., gt=0, description="Market price of option")
    option_type: OptionType = Field(..., description="Option type")
    dividend_yield: Optional[float] = Field(0.0, ge=0, le=1)
    
    class Config:
        schema_extra = {
            "example": {
                "spot_price": 100.0,
                "strike": 100.0,
                "time_to_expiry": 1.0,
                "risk_free_rate": 0.05,
                "option_price": 10.45,
                "option_type": "call",
                "dividend_yield": 0.0
            }
        }


class ImpliedVolatilityResponse(BaseModel):
    """Response model for implied volatility."""
    
    implied_volatility: float = Field(..., description="Calculated implied volatility")
    iterations: int = Field(..., description="Number of iterations to converge")
    converged: bool = Field(..., description="Whether calculation converged")
    execution_time_ms: float


# Option Chain Analysis
class OptionChainRequest(BaseModel):
    """Request for option chain analysis."""
    
    spot_price: float = Field(..., gt=0)
    strikes: List[float] = Field(..., min_items=1, max_items=50, description="Strike prices")
    time_to_expiry: float = Field(..., gt=0, le=30)
    risk_free_rate: float = Field(..., ge=0, le=1)
    volatility: float = Field(..., gt=0, le=5)
    dividend_yield: Optional[float] = Field(0.0, ge=0, le=1)


class OptionChainEntry(BaseModel):
    """Single entry in option chain."""
    
    strike: float
    call_price: float
    put_price: float
    call_delta: float
    put_delta: float
    call_implied_vol: Optional[float] = None
    put_implied_vol: Optional[float] = None


class OptionChainResponse(BaseModel):
    """Response for option chain analysis."""
    
    spot_price: float
    chain: List[OptionChainEntry]
    atm_volatility: float = Field(..., description="At-the-money volatility")
    put_call_parity_check: bool = Field(..., description="Whether put-call parity holds")
    execution_time_ms: float


# Batch Pricing
class BatchOptionPriceRequest(BaseModel):
    """Request for batch option pricing."""
    
    options: List[OptionPriceRequest] = Field(..., min_items=1, max_items=100)


class BatchOptionPriceResponse(BaseModel):
    """Response for batch option pricing."""
    
    results: List[OptionPriceResponse]
    total_execution_time_ms: float
    successful: int
    failed: int


# Option Strategy Models
class OptionLeg(BaseModel):
    """Single leg of an option strategy."""
    
    option_type: OptionType
    strike: float = Field(..., gt=0)
    quantity: int = Field(..., description="Number of contracts (positive for long, negative for short)")
    premium: float = Field(..., gt=0, description="Premium paid/received per contract")


class OptionStrategyRequest(BaseModel):
    """Request for option strategy analysis."""
    
    spot_price: float = Field(..., gt=0)
    legs: List[OptionLeg] = Field(..., min_items=1, max_items=10)
    price_range: Optional[tuple[float, float]] = Field(None, description="Price range for P&L analysis")


class OptionStrategyResponse(BaseModel):
    """Response for option strategy analysis."""
    
    strategy_name: str = Field(..., description="Detected strategy name")
    max_profit: Optional[float] = Field(None, description="Maximum profit (None if unlimited)")
    max_loss: Optional[float] = Field(None, description="Maximum loss (None if unlimited)")
    breakeven_points: List[float] = Field(..., description="Breakeven prices")
    net_premium: float = Field(..., description="Net premium paid/received")
    profit_loss_data: Dict[float, float] = Field(..., description="P&L at different spot prices")
    execution_time_ms: float


# Error Response
class OptionErrorResponse(BaseModel):
    """Error response for options API."""
    
    error: str
    detail: str
    code: Optional[str] = None