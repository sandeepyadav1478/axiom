"""
Pydantic models for M&A API endpoints.
"""

from typing import List, Dict, Optional
from pydantic import BaseModel, Field


class SynergyValuationRequest(BaseModel):
    """Request for synergy valuation."""
    
    revenue_synergies: float = Field(..., description="Annual revenue synergies")
    cost_synergies: float = Field(..., description="Annual cost synergies")
    implementation_cost: float = Field(..., gt=0, description="One-time implementation cost")
    time_to_realize: float = Field(..., gt=0, le=10, description="Years to realize synergies")
    discount_rate: float = Field(..., ge=0, le=1, description="Discount rate")
    tax_rate: float = Field(0.25, ge=0, le=1, description="Corporate tax rate")


class SynergyValuationResponse(BaseModel):
    """Response for synergy valuation."""
    
    synergy_npv: float = Field(..., description="NPV of synergies")
    pv_revenue_synergies: float
    pv_cost_synergies: float
    net_synergy_value: float = Field(..., description="NPV minus implementation cost")
    execution_time_ms: float


class DealFinancingRequest(BaseModel):
    """Request for deal financing optimization."""
    
    deal_value: float = Field(..., gt=0, description="Total deal value")
    cash_available: float = Field(..., ge=0, description="Available cash")
    debt_capacity: float = Field(..., ge=0, description="Maximum debt capacity")
    stock_price: float = Field(..., gt=0, description="Current stock price")
    cost_of_debt: float = Field(..., ge=0, le=1, description="Cost of debt")
    cost_of_equity: float = Field(..., ge=0, le=1, description="Cost of equity")


class DealFinancingResponse(BaseModel):
    """Response for deal financing."""
    
    optimal_mix: Dict[str, float] = Field(..., description="Cash/Debt/Stock percentages")
    wacc: float = Field(..., description="Weighted average cost of capital")
    total_cost: float
    execution_time_ms: float


class LBOModelRequest(BaseModel):
    """Request for LBO modeling."""
    
    purchase_price: float = Field(..., gt=0)
    equity_contribution: float = Field(..., gt=0)
    debt_amount: float = Field(..., gt=0)
    interest_rate: float = Field(..., ge=0, le=1)
    ebitda: float = Field(..., gt=0)
    revenue_growth: float = Field(..., ge=-0.5, le=2.0)
    exit_multiple: float = Field(..., gt=0)
    holding_period: int = Field(..., ge=3, le=10, description="Years")


class LBOModelResponse(BaseModel):
    """Response for LBO modeling."""
    
    irr: float = Field(..., description="Internal rate of return")
    moic: float = Field(..., description="Multiple on invested capital")
    exit_value: float
    total_return: float
    execution_time_ms: float


class MergerArbitrageRequest(BaseModel):
    """Request for merger arbitrage analysis."""
    
    target_price: float = Field(..., gt=0)
    offer_price: float = Field(..., gt=0)
    deal_close_probability: float = Field(..., ge=0, le=1)
    days_to_close: int = Field(..., ge=1, le=730)
    risk_free_rate: float = Field(0.02, ge=0, le=1)


class MergerArbitrageResponse(BaseModel):
    """Response for merger arbitrage."""
    
    spread: float = Field(..., description="Offer price - current price")
    spread_percentage: float
    annualized_return: float
    risk_adjusted_return: float
    execution_time_ms: float