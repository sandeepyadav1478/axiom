"""
External Libraries Integration Module

This module provides adapters and wrappers for production-grade external libraries
to enhance Axiom's quantitative finance capabilities.

Available Integrations:
- QuantLib: Comprehensive fixed income and derivatives pricing
- PyPortfolioOpt: Modern portfolio theory and optimization
- TA-Lib: Technical analysis indicators (150+ indicators)
- pandas-ta: Technical analysis in pandas (130+ indicators)
- statsmodels: Statistical models and time series analysis
- arch: ARCH/GARCH volatility models

Usage:
    from axiom.integrations.external_libs import (
        QuantLibBondPricer,
        PyPortfolioOptAdapter,
        TALibIndicators,
        PandasTAIntegration,
        ArchGARCH,
        LibraryConfig,
        get_library_availability
    )
    
    # Check library availability
    availability = get_library_availability()
    print(f"QuantLib available: {availability['QuantLib']}")
    
    # Use QuantLib for bond pricing
    if availability['QuantLib']:
        pricer = QuantLibBondPricer()
        bond = BondSpecification(...)
        result = pricer.price_bond(bond, ...)
    
    # Use PyPortfolioOpt for optimization
    if availability['PyPortfolioOpt']:
        optimizer = PyPortfolioOptAdapter()
        result = optimizer.optimize_portfolio(prices_df)
"""

# Configuration
from .config import (
    LibraryConfig,
    LibraryAvailability,
    get_library_availability,
    get_config,
    set_config,
    reset_config,
    log_library_status
)

# QuantLib wrapper
from .quantlib_wrapper import (
    QuantLibBondPricer,
    BondSpecification,
    BondPrice,
    YieldCurvePoint,
    DayCountConvention,
    BusinessDayConvention,
    Frequency,
    check_quantlib_availability
)

# PyPortfolioOpt adapter
from .pypfopt_adapter import (
    PyPortfolioOptAdapter,
    OptimizationResult,
    OptimizationObjective,
    RiskModel,
    ExpectedReturnModel,
    BlackLittermanInputs,
    check_pypfopt_availability
)

# TA-Lib indicators
from .talib_indicators import (
    TALibIndicators,
    MAType,
    IndicatorResult,
    check_talib_availability,
    get_available_indicators
)

# pandas-ta integration
from .pandas_ta_integration import (
    PandasTAIntegration,
    StrategyResult,
    check_pandas_ta_availability,
    get_pandas_ta_version
)

# arch GARCH models
from .arch_garch import (
    ArchGARCH,
    GARCHResult,
    VolatilityForecast,
    VolatilityModel,
    Distribution,
    check_arch_availability,
    estimate_simple_garch
)

__all__ = [
    # Configuration
    'LibraryConfig',
    'LibraryAvailability',
    'get_library_availability',
    'get_config',
    'set_config',
    'reset_config',
    'log_library_status',
    
    # QuantLib
    'QuantLibBondPricer',
    'BondSpecification',
    'BondPrice',
    'YieldCurvePoint',
    'DayCountConvention',
    'BusinessDayConvention',
    'Frequency',
    'check_quantlib_availability',
    
    # PyPortfolioOpt
    'PyPortfolioOptAdapter',
    'OptimizationResult',
    'OptimizationObjective',
    'RiskModel',
    'ExpectedReturnModel',
    'BlackLittermanInputs',
    'check_pypfopt_availability',
    
    # TA-Lib
    'TALibIndicators',
    'MAType',
    'IndicatorResult',
    'check_talib_availability',
    'get_available_indicators',
    
    # pandas-ta
    'PandasTAIntegration',
    'StrategyResult',
    'check_pandas_ta_availability',
    'get_pandas_ta_version',
    
    # arch GARCH
    'ArchGARCH',
    'GARCHResult',
    'VolatilityForecast',
    'VolatilityModel',
    'Distribution',
    'check_arch_availability',
    'estimate_simple_garch',
]

# Version information
__version__ = '1.0.0'

# Log library availability on import
try:
    log_library_status()
except Exception:
    pass  # Silently fail if logging not configured