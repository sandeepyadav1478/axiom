"""
Pydantic models for Fixed Income API endpoints.
"""

from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from datetime import date


class BondPriceRequest(BaseModel):
    """Request for bond pricing."""
    
    face_value: float = Field(1000.0, gt=0, description="Face/par value")
    coupon_rate: float = Field(..., ge=0, le=1, description="Annual coupon rate")
    years_to_maturity: float = Field(..., gt=0, le=30, description="Years to maturity")
    yield_to_maturity: float = Field(..., ge=0, le=1, description="Yield to maturity")
    frequency: int = Field(2, ge=1, le=12, description="Coupon payments per year")


class BondPriceResponse(BaseModel):
    """Response for bond pricing."""
    
    price: float = Field(..., description="Bond price")
    clean_price: float = Field(..., description="Price without accrued interest")
    accrued_interest: float
    duration: float = Field(..., description="Macaulay duration")
    modified_duration: float
    convexity: float
    execution_time_ms: float


class YieldCurveRequest(BaseModel):
    """Request for yield curve construction."""
    
    maturities: List[float] = Field(..., min_items=3, description="Maturities in years")
    yields: List[float] = Field(..., min_items=3, description="Yields for each maturity")
    interpolation_method: Optional[str] = Field("cubic_spline", description="Interpolation method")


class YieldCurveResponse(BaseModel):
    """Response for yield curve."""
    
    curve_points: Dict[float, float] = Field(..., description="Maturity -> Yield mapping")
    forward_rates: Dict[float, float] = Field(..., description="Forward rates")
    execution_time_ms: float


class BondYTMRequest(BaseModel):
    """Request for yield to maturity calculation."""
    
    price: float = Field(..., gt=0)
    face_value: float = Field(1000.0, gt=0)
    coupon_rate: float = Field(..., ge=0, le=1)
    years_to_maturity: float = Field(..., gt=0, le=30)
    frequency: int = Field(2, ge=1, le=12)


class BondYTMResponse(BaseModel):
    """Response for YTM calculation."""
    
    ytm: float = Field(..., description="Yield to maturity")
    current_yield: float
    execution_time_ms: float