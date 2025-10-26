"""
Risk Management API endpoints.
"""

import time
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status

from axiom.api.auth import get_current_user, User
from axiom.api.rate_limit import limiter, standard_rate_limit
from axiom.api.models.risk import (
    VaRRequest,
    VaRResponse,
    StressTestRequest,
    StressTestResponse,
    RiskMetricsResponse,
)

router = APIRouter(prefix="/risk", tags=["Risk Management"])


@router.post("/var", response_model=VaRResponse)
@standard_rate_limit
async def calculate_var(
    request: VaRRequest,
    current_user: User = Depends(get_current_user),
):
    """
    Calculate Value at Risk (VaR).
    
    **Methods**:
    - Parametric: Assumes normal distribution
    - Historical: Uses historical data
    - Monte Carlo: Simulation-based
    """
    start_time = time.time()
    
    try:
        from axiom.models.risk.var_models import VaRModel
        
        model = VaRModel()
        var_result = model.calculate_var(
            returns=np.array(request.returns),
            confidence_level=request.confidence_level,
            method=request.method.value,
            portfolio_value=request.portfolio_value,
        )
        
        cvar_result = model.calculate_cvar(
            returns=np.array(request.returns),
            confidence_level=request.confidence_level,
            portfolio_value=request.portfolio_value,
        )
        
        execution_time = (time.time() - start_time) * 1000
        
        return VaRResponse(
            var=var_result,
            cvar=cvar_result,
            confidence_level=request.confidence_level,
            method=request.method.value,
            execution_time_ms=execution_time,
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error calculating VaR: {str(e)}",
        )


@router.post("/stress-test", response_model=StressTestResponse)
@standard_rate_limit
async def run_stress_test(
    request: StressTestRequest,
    current_user: User = Depends(get_current_user),
):
    """Run stress test scenarios on portfolio."""
    start_time = time.time()
    
    try:
        scenario_results = {}
        for scenario_name, scenario_params in request.scenarios.items():
            # Mock stress test result
            impact = sum(scenario_params.values()) * 0.1
            scenario_results[scenario_name] = impact
        
        execution_time = (time.time() - start_time) * 1000
        
        return StressTestResponse(
            scenario_results=scenario_results,
            worst_case={"scenario": "market_crash", "impact": -0.25},
            best_case={"scenario": "bull_market", "impact": 0.15},
            execution_time_ms=execution_time,
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error running stress test: {str(e)}",
        )