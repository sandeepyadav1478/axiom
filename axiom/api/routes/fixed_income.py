"""
Fixed Income API endpoints.
"""

import time
from fastapi import APIRouter, Depends, HTTPException, status

from axiom.api.auth import get_current_user, User
from axiom.api.rate_limit import limiter, standard_rate_limit
from axiom.api.models.fixed_income import (
    BondPriceRequest,
    BondPriceResponse,
    YieldCurveRequest,
    YieldCurveResponse,
    BondYTMRequest,
    BondYTMResponse,
)

router = APIRouter(prefix="/bonds", tags=["Fixed Income"])


@router.post("/price", response_model=BondPriceResponse)
@standard_rate_limit
async def calculate_bond_price(
    request: BondPriceRequest,
    current_user: User = Depends(get_current_user),
):
    """Calculate bond price and metrics."""
    start_time = time.time()
    
    try:
        # Calculate bond price
        periods = int(request.years_to_maturity * request.frequency)
        coupon_payment = request.face_value * request.coupon_rate / request.frequency
        discount_rate = request.yield_to_maturity / request.frequency
        
        # PV of coupons
        pv_coupons = sum(
            coupon_payment / ((1 + discount_rate) ** t)
            for t in range(1, periods + 1)
        )
        
        # PV of face value
        pv_face = request.face_value / ((1 + discount_rate) ** periods)
        
        price = pv_coupons + pv_face
        
        # Calculate duration (simplified Macaulay duration)
        duration = sum(
            t * coupon_payment / ((1 + discount_rate) ** t)
            for t in range(1, periods + 1)
        ) / price
        duration += periods * pv_face / price
        duration = duration / request.frequency
        
        modified_duration = duration / (1 + request.yield_to_maturity / request.frequency)
        
        # Convexity (simplified)
        convexity = sum(
            t * (t + 1) * coupon_payment / ((1 + discount_rate) ** t)
            for t in range(1, periods + 1)
        ) / (price * (1 + discount_rate) ** 2)
        
        execution_time = (time.time() - start_time) * 1000
        
        return BondPriceResponse(
            price=price,
            clean_price=price,
            accrued_interest=0.0,
            duration=duration,
            modified_duration=modified_duration,
            convexity=convexity,
            execution_time_ms=execution_time,
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error calculating bond price: {str(e)}",
        )


@router.post("/ytm", response_model=BondYTMResponse)
@standard_rate_limit
async def calculate_ytm(
    request: BondYTMRequest,
    current_user: User = Depends(get_current_user),
):
    """Calculate yield to maturity."""
    start_time = time.time()
    
    try:
        # Simple approximation
        annual_coupon = request.face_value * request.coupon_rate
        years = request.years_to_maturity
        
        ytm = (annual_coupon + (request.face_value - request.price) / years) / ((request.face_value + request.price) / 2)
        current_yield = annual_coupon / request.price
        
        execution_time = (time.time() - start_time) * 1000
        
        return BondYTMResponse(
            ytm=ytm,
            current_yield=current_yield,
            execution_time_ms=execution_time,
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error calculating YTM: {str(e)}",
        )


@router.post("/yield-curve", response_model=YieldCurveResponse)
@standard_rate_limit
async def construct_yield_curve(
    request: YieldCurveRequest,
    current_user: User = Depends(get_current_user),
):
    """Construct yield curve from market data."""
    start_time = time.time()
    
    try:
        # Create curve points
        curve_points = {m: y for m, y in zip(request.maturities, request.yields)}
        
        # Calculate forward rates
        forward_rates = {}
        for i in range(len(request.maturities) - 1):
            t1, t2 = request.maturities[i], request.maturities[i + 1]
            r1, r2 = request.yields[i], request.yields[i + 1]
            forward = ((1 + r2) ** t2) / ((1 + r1) ** t1)
            forward_rates[t2] = forward ** (1 / (t2 - t1)) - 1
        
        execution_time = (time.time() - start_time) * 1000
        
        return YieldCurveResponse(
            curve_points=curve_points,
            forward_rates=forward_rates,
            execution_time_ms=execution_time,
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error constructing yield curve: {str(e)}",
        )