"""
M&A Analytics API endpoints.
"""

import time
from fastapi import APIRouter, Depends, HTTPException, status

from axiom.api.auth import get_current_user, User
from axiom.api.rate_limit import limiter, standard_rate_limit
from axiom.api.models.ma import (
    SynergyValuationRequest,
    SynergyValuationResponse,
    DealFinancingRequest,
    DealFinancingResponse,
    LBOModelRequest,
    LBOModelResponse,
    MergerArbitrageRequest,
    MergerArbitrageResponse,
)

router = APIRouter(prefix="/ma", tags=["M&A Analytics"])


@router.post("/synergy-valuation", response_model=SynergyValuationResponse)
@standard_rate_limit
async def value_synergies(
    request: SynergyValuationRequest,
    current_user: User = Depends(get_current_user),
):
    """Value deal synergies using DCF approach."""
    start_time = time.time()
    
    try:
        # Calculate PV of synergies
        periods = int(request.time_to_realize)
        discount_factor = 1 / (1 + request.discount_rate)
        
        pv_revenue = sum(
            request.revenue_synergies * (discount_factor ** t)
            for t in range(1, periods + 1)
        )
        
        pv_cost = sum(
            request.cost_synergies * (discount_factor ** t)
            for t in range(1, periods + 1)
        )
        
        synergy_npv = (pv_revenue + pv_cost) * (1 - request.tax_rate)
        net_synergy = synergy_npv - request.implementation_cost
        
        execution_time = (time.time() - start_time) * 1000
        
        return SynergyValuationResponse(
            synergy_npv=synergy_npv,
            pv_revenue_synergies=pv_revenue * (1 - request.tax_rate),
            pv_cost_synergies=pv_cost * (1 - request.tax_rate),
            net_synergy_value=net_synergy,
            execution_time_ms=execution_time,
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error valuing synergies: {str(e)}",
        )


@router.post("/deal-financing", response_model=DealFinancingResponse)
@standard_rate_limit
async def optimize_deal_financing(
    request: DealFinancingRequest,
    current_user: User = Depends(get_current_user),
):
    """Optimize deal financing mix."""
    start_time = time.time()
    
    try:
        # Simple optimization
        cash_pct = min(request.cash_available / request.deal_value, 0.3)
        debt_pct = min(request.debt_capacity / request.deal_value, 0.5)
        stock_pct = max(0, 1.0 - cash_pct - debt_pct)
        
        wacc = (
            debt_pct * request.cost_of_debt * 0.75 +  # Tax shield
            (cash_pct + stock_pct) * request.cost_of_equity
        )
        
        execution_time = (time.time() - start_time) * 1000
        
        return DealFinancingResponse(
            optimal_mix={
                "cash": cash_pct,
                "debt": debt_pct,
                "stock": stock_pct,
            },
            wacc=wacc,
            total_cost=request.deal_value * wacc,
            execution_time_ms=execution_time,
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error optimizing financing: {str(e)}",
        )


@router.post("/lbo-model", response_model=LBOModelResponse)
@standard_rate_limit
async def model_lbo(
    request: LBOModelRequest,
    current_user: User = Depends(get_current_user),
):
    """Model LBO returns."""
    start_time = time.time()
    
    try:
        # Calculate exit value
        future_ebitda = request.ebitda * ((1 + request.revenue_growth) ** request.holding_period)
        exit_value = future_ebitda * request.exit_multiple
        
        # Calculate returns
        total_return = exit_value - request.purchase_price
        moic = exit_value / request.equity_contribution
        irr = ((exit_value / request.equity_contribution) ** (1 / request.holding_period)) - 1
        
        execution_time = (time.time() - start_time) * 1000
        
        return LBOModelResponse(
            irr=irr,
            moic=moic,
            exit_value=exit_value,
            total_return=total_return,
            execution_time_ms=execution_time,
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error modeling LBO: {str(e)}",
        )


@router.post("/merger-arbitrage", response_model=MergerArbitrageResponse)
@standard_rate_limit
async def analyze_merger_arbitrage(
    request: MergerArbitrageRequest,
    current_user: User = Depends(get_current_user),
):
    """Analyze merger arbitrage opportunity."""
    start_time = time.time()
    
    try:
        spread = request.offer_price - request.target_price
        spread_pct = spread / request.target_price
        
        # Annualized return
        years_to_close = request.days_to_close / 365
        annualized_return = (spread_pct / years_to_close)
        
        # Risk-adjusted
        risk_adjusted = annualized_return * request.deal_close_probability
        
        execution_time = (time.time() - start_time) * 1000
        
        return MergerArbitrageResponse(
            spread=spread,
            spread_percentage=spread_pct,
            annualized_return=annualized_return,
            risk_adjusted_return=risk_adjusted,
            execution_time_ms=execution_time,
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing merger arbitrage: {str(e)}",
        )