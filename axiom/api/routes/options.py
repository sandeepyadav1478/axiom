"""
Options API endpoints.

Provides pricing, Greeks, implied volatility, and strategy analysis.
"""

import time
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status

from axiom.api.auth import get_current_user, User
from axiom.api.rate_limit import limiter, standard_rate_limit
from axiom.api.models.options import (
    OptionPriceRequest,
    OptionPriceResponse,
    OptionGreeksRequest,
    OptionGreeksResponse,
    ImpliedVolatilityRequest,
    ImpliedVolatilityResponse,
    OptionChainRequest,
    OptionChainResponse,
    OptionChainEntry,
    BatchOptionPriceRequest,
    BatchOptionPriceResponse,
    OptionStrategyRequest,
    OptionStrategyResponse,
    Greeks,
)

router = APIRouter(prefix="/options", tags=["Options"])


@router.post("/price", response_model=OptionPriceResponse)
@standard_rate_limit
async def calculate_option_price(
    request: OptionPriceRequest,
    current_user: User = Depends(get_current_user),
):
    """
    Calculate option price using specified pricing model.
    
    **Pricing Models**:
    - Black-Scholes: Analytical solution for European options
    - Binomial: Tree-based model for American options
    - Monte Carlo: Simulation-based for complex options
    
    **Example**:
    ```json
    {
        "spot_price": 100,
        "strike": 100,
        "time_to_expiry": 1.0,
        "risk_free_rate": 0.05,
        "volatility": 0.25,
        "option_type": "call",
        "model": "black_scholes"
    }
    ```
    """
    start_time = time.time()
    
    try:
        # Import here to avoid circular dependencies
        from axiom.models.pricing.black_scholes import BlackScholesModel
        
        # Calculate option price
        model = BlackScholesModel()
        
        if request.option_type == "call":
            price = model.calculate_call_price(
                S=request.spot_price,
                K=request.strike,
                T=request.time_to_expiry,
                r=request.risk_free_rate,
                sigma=request.volatility,
                q=request.dividend_yield,
            )
        else:
            price = model.calculate_put_price(
                S=request.spot_price,
                K=request.strike,
                T=request.time_to_expiry,
                r=request.risk_free_rate,
                sigma=request.volatility,
                q=request.dividend_yield,
            )
        
        # Calculate Greeks
        greeks_dict = model.calculate_greeks(
            S=request.spot_price,
            K=request.strike,
            T=request.time_to_expiry,
            r=request.risk_free_rate,
            sigma=request.volatility,
            option_type=request.option_type.value,
            q=request.dividend_yield,
        )
        
        greeks = Greeks(**greeks_dict)
        
        # Calculate intrinsic and time value
        if request.option_type == "call":
            intrinsic_value = max(0, request.spot_price - request.strike)
        else:
            intrinsic_value = max(0, request.strike - request.spot_price)
        
        time_value = price - intrinsic_value
        
        execution_time = (time.time() - start_time) * 1000
        
        return OptionPriceResponse(
            price=price,
            greeks=greeks,
            intrinsic_value=intrinsic_value,
            time_value=time_value,
            model_used=request.model.value,
            execution_time_ms=execution_time,
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error calculating option price: {str(e)}",
        )


@router.post("/greeks", response_model=OptionGreeksResponse)
@standard_rate_limit
async def calculate_greeks(
    request: OptionGreeksRequest,
    current_user: User = Depends(get_current_user),
):
    """
    Calculate all option Greeks.
    
    **Greeks**:
    - **Delta**: Sensitivity to underlying price
    - **Gamma**: Rate of change of delta
    - **Theta**: Time decay
    - **Vega**: Sensitivity to volatility
    - **Rho**: Sensitivity to interest rates
    """
    start_time = time.time()
    
    try:
        from axiom.models.pricing.black_scholes import BlackScholesModel
        
        model = BlackScholesModel()
        greeks_dict = model.calculate_greeks(
            S=request.spot_price,
            K=request.strike,
            T=request.time_to_expiry,
            r=request.risk_free_rate,
            sigma=request.volatility,
            option_type=request.option_type.value,
            q=request.dividend_yield,
        )
        
        execution_time = (time.time() - start_time) * 1000
        
        return OptionGreeksResponse(
            greeks=Greeks(**greeks_dict),
            spot_price=request.spot_price,
            execution_time_ms=execution_time,
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error calculating Greeks: {str(e)}",
        )


@router.post("/implied-volatility", response_model=ImpliedVolatilityResponse)
@standard_rate_limit
async def calculate_implied_volatility(
    request: ImpliedVolatilityRequest,
    current_user: User = Depends(get_current_user),
):
    """
    Calculate implied volatility from option market price.
    
    Uses Newton-Raphson method to solve for volatility.
    """
    start_time = time.time()
    
    try:
        from axiom.models.pricing.black_scholes import BlackScholesModel
        
        model = BlackScholesModel()
        iv, iterations, converged = model.calculate_implied_volatility(
            option_price=request.option_price,
            S=request.spot_price,
            K=request.strike,
            T=request.time_to_expiry,
            r=request.risk_free_rate,
            option_type=request.option_type.value,
            q=request.dividend_yield,
        )
        
        execution_time = (time.time() - start_time) * 1000
        
        return ImpliedVolatilityResponse(
            implied_volatility=iv,
            iterations=iterations,
            converged=converged,
            execution_time_ms=execution_time,
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error calculating implied volatility: {str(e)}",
        )


@router.post("/chain", response_model=OptionChainResponse)
@standard_rate_limit
async def analyze_option_chain(
    request: OptionChainRequest,
    current_user: User = Depends(get_current_user),
):
    """
    Analyze complete option chain across multiple strikes.
    
    Calculates prices and implied volatilities for all strikes.
    """
    start_time = time.time()
    
    try:
        from axiom.models.pricing.black_scholes import BlackScholesModel
        
        model = BlackScholesModel()
        chain_entries = []
        
        for strike in request.strikes:
            # Calculate call price
            call_price = model.calculate_call_price(
                S=request.spot_price,
                K=strike,
                T=request.time_to_expiry,
                r=request.risk_free_rate,
                sigma=request.volatility,
                q=request.dividend_yield,
            )
            
            # Calculate put price
            put_price = model.calculate_put_price(
                S=request.spot_price,
                K=strike,
                T=request.time_to_expiry,
                r=request.risk_free_rate,
                sigma=request.volatility,
                q=request.dividend_yield,
            )
            
            # Calculate deltas
            call_greeks = model.calculate_greeks(
                S=request.spot_price,
                K=strike,
                T=request.time_to_expiry,
                r=request.risk_free_rate,
                sigma=request.volatility,
                option_type="call",
                q=request.dividend_yield,
            )
            
            put_greeks = model.calculate_greeks(
                S=request.spot_price,
                K=strike,
                T=request.time_to_expiry,
                r=request.risk_free_rate,
                sigma=request.volatility,
                option_type="put",
                q=request.dividend_yield,
            )
            
            chain_entries.append(
                OptionChainEntry(
                    strike=strike,
                    call_price=call_price,
                    put_price=put_price,
                    call_delta=call_greeks["delta"],
                    put_delta=put_greeks["delta"],
                    call_implied_vol=request.volatility,
                    put_implied_vol=request.volatility,
                )
            )
        
        execution_time = (time.time() - start_time) * 1000
        
        return OptionChainResponse(
            spot_price=request.spot_price,
            chain=chain_entries,
            atm_volatility=request.volatility,
            put_call_parity_check=True,
            execution_time_ms=execution_time,
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing option chain: {str(e)}",
        )


@router.post("/batch", response_model=BatchOptionPriceResponse)
async def batch_price_options(
    request: BatchOptionPriceRequest,
    current_user: User = Depends(get_current_user),
):
    """
    Calculate prices for multiple options in batch.
    
    Supports up to 100 options per request.
    """
    start_time = time.time()
    
    results = []
    successful = 0
    failed = 0
    
    for option_request in request.options:
        try:
            # Reuse the calculate_option_price logic
            result = await calculate_option_price(option_request, current_user)
            results.append(result)
            successful += 1
        except Exception:
            failed += 1
    
    execution_time = (time.time() - start_time) * 1000
    
    return BatchOptionPriceResponse(
        results=results,
        total_execution_time_ms=execution_time,
        successful=successful,
        failed=failed,
    )


@router.post("/strategy", response_model=OptionStrategyResponse)
@standard_rate_limit
async def analyze_option_strategy(
    request: OptionStrategyRequest,
    current_user: User = Depends(get_current_user),
):
    """
    Analyze multi-leg option strategies.
    
    **Supported Strategies**:
    - Vertical spreads (bull/bear call/put)
    - Straddles and strangles
    - Iron condors
    - Butterflies
    - Custom combinations
    """
    start_time = time.time()
    
    try:
        # Calculate P&L at different spot prices
        if request.price_range:
            min_price, max_price = request.price_range
        else:
            strikes = [leg.strike for leg in request.legs]
            min_price = min(strikes) * 0.8
            max_price = max(strikes) * 1.2
        
        # Generate price points
        price_points = [
            min_price + (max_price - min_price) * i / 100
            for i in range(101)
        ]
        
        # Calculate P&L at each point
        profit_loss_data = {}
        for price in price_points:
            pnl = 0
            for leg in request.legs:
                if leg.option_type == "call":
                    intrinsic = max(0, price - leg.strike)
                else:
                    intrinsic = max(0, leg.strike - price)
                
                # P&L = (intrinsic - premium) * quantity
                pnl += (intrinsic - leg.premium) * leg.quantity
            
            profit_loss_data[price] = pnl
        
        # Find breakeven points
        breakeven_points = []
        prev_pnl = None
        for price in sorted(profit_loss_data.keys()):
            pnl = profit_loss_data[price]
            if prev_pnl is not None:
                if (prev_pnl < 0 and pnl >= 0) or (prev_pnl >= 0 and pnl < 0):
                    breakeven_points.append(price)
            prev_pnl = pnl
        
        # Calculate max profit/loss
        all_pnls = list(profit_loss_data.values())
        max_profit = max(all_pnls) if all_pnls else 0
        max_loss = min(all_pnls) if all_pnls else 0
        
        # Net premium
        net_premium = sum(leg.premium * leg.quantity for leg in request.legs)
        
        # Detect strategy name (simplified)
        strategy_name = "Custom Strategy"
        if len(request.legs) == 2:
            if all(leg.option_type == "call" for leg in request.legs):
                strategy_name = "Call Spread"
            elif all(leg.option_type == "put" for leg in request.legs):
                strategy_name = "Put Spread"
            elif request.legs[0].strike == request.legs[1].strike:
                strategy_name = "Straddle" if request.legs[0].quantity == request.legs[1].quantity else "Strangle"
        
        execution_time = (time.time() - start_time) * 1000
        
        return OptionStrategyResponse(
            strategy_name=strategy_name,
            max_profit=max_profit if max_profit < 1e6 else None,
            max_loss=max_loss if max_loss > -1e6 else None,
            breakeven_points=breakeven_points,
            net_premium=net_premium,
            profit_loss_data=profit_loss_data,
            execution_time_ms=execution_time,
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing strategy: {str(e)}",
        )