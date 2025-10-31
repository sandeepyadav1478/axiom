#!/usr/bin/env python3
"""
Quick verification script for the test fixes
"""

import numpy as np

# Test 1: EWMA span calculation
print("Testing EWMA span calculation...")
from axiom.models.time_series.ewma import EWMAModel

model = EWMAModel(span=20)
print(f"  span parameter: 20")
print(f"  calculated span: {model.span}")
print(f"  decay factor: {model.decay_factor}")
expected_decay = 2.0 / 21  # 2 / (span + 1)
print(f"  expected decay: {expected_decay}")
assert abs(model.span - 20) < 0.1, f"Span mismatch: {model.span} != 20"
print("  ✓ EWMA span test PASSED\n")

# Test 2: ModelFactory with custom config
print("Testing ModelFactory with custom config...")
from axiom.models.base.factory import ModelFactory, ModelType
from axiom.config.model_config import VaRConfig

custom_config = VaRConfig(default_confidence_level=0.99)
model = ModelFactory.create(ModelType.PARAMETRIC_VAR, config=custom_config)
# Verify it has the config
assert hasattr(model, 'var_config'), "Model should have var_config"
assert model.var_config.default_confidence_level == 0.99, "Config not applied correctly"
print("  ✓ ModelFactory config test PASSED\n")

# Test 3: VaR time scaling
print("Testing VaR time scaling...")
from axiom.models.risk.var_models import ParametricVaR

portfolio_value = 1_000_000
np.random.seed(42)
returns = np.random.normal(0.001, 0.02, 252)

var_1d = ParametricVaR.calculate(portfolio_value, returns, 0.95, 1)
var_10d = ParametricVaR.calculate(portfolio_value, returns, 0.95, 10)

scaling_factor = var_10d.var_amount / var_1d.var_amount
print(f"  1-day VaR: ${var_1d.var_amount:,.2f}")
print(f"  10-day VaR: ${var_10d.var_amount:,.2f}")
print(f"  Scaling factor: {scaling_factor:.3f}")
print(f"  sqrt(10) = {np.sqrt(10):.3f}")
assert 2.5 < scaling_factor < 4.0, f"Scaling factor {scaling_factor} outside expected range"
print("  ✓ VaR time scaling test PASSED\n")

# Test 4: Data structures
print("Testing data structures...")
from axiom.models.portfolio.optimization import (
    OptimizationResult,
    PortfolioMetrics,
    EfficientFrontier
)
from axiom.models.portfolio.allocation import (
    AllocationResult,
    AssetClass
)

assert hasattr(OptimizationResult, 'weights'), "OptimizationResult missing weights"
assert hasattr(OptimizationResult, 'metrics'), "OptimizationResult missing metrics"
assert hasattr(OptimizationResult, 'to_dict'), "OptimizationResult missing to_dict"

assert hasattr(PortfolioMetrics, 'sharpe_ratio'), "PortfolioMetrics missing sharpe_ratio"
assert hasattr(PortfolioMetrics, 'to_dict'), "PortfolioMetrics missing to_dict"

assert hasattr(AllocationResult, 'weights'), "AllocationResult missing weights"
assert hasattr(AllocationResult, 'strategy'), "AllocationResult missing strategy"

print("  ✓ Data structures test PASSED\n")

# Test 5: Options ITM/OTM
print("Testing options ITM/OTM...")
from axiom.models.options.black_scholes import calculate_call_price

spot = 100
rate = 0.05
vol = 0.25
time_to_expiry = 1.0

# Deep ITM call
itm_call = calculate_call_price(spot, 80, time_to_expiry, rate, vol)
intrinsic = spot - 80 * np.exp(-rate * time_to_expiry)
print(f"  ITM call price: ${itm_call:.4f}")
print(f"  Intrinsic value: ${intrinsic:.4f}")
assert itm_call > intrinsic, "ITM call should exceed intrinsic value"

# Deep OTM call
otm_call = calculate_call_price(spot, 150, time_to_expiry, rate, vol)
print(f"  OTM call price: ${otm_call:.4f}")
assert otm_call < 1.0, "Deep OTM call should have small value"
print("  ✓ Options ITM/OTM test PASSED\n")

print("=" * 60)
print("ALL VERIFICATION TESTS PASSED!")
print("=" * 60)