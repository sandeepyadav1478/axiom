"""
Axiom Derivatives Module

Ultra-fast, AI-powered derivatives analytics platform
specializing in real-time options pricing, Greeks calculation,
and market making.

Target Performance: <100 microseconds for Greeks calculation
"""

from axiom.derivatives.ultra_fast_greeks import UltraFastGreeksEngine
from axiom.derivatives.exotic_pricer import ExoticOptionsPricer
from axiom.derivatives.volatility_surface import RealTimeVolatilitySurface

__all__ = [
    'UltraFastGreeksEngine',
    'ExoticOptionsPricer',
    'RealTimeVolatilitySurface'
]