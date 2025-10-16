"""
Axiom Financial Data Integrations Package

Cost-effective financial data providers for M&A analytics including:
- OpenBB Terminal (100% FREE, open source)
- SEC Edgar API (100% FREE, US government data)
- Alpha Vantage (FREE tier + $49/month premium)
- Financial Modeling Prep (FREE tier + $15/month premium)
- IEX Cloud (FREE tier + $9/month premium)
- Yahoo Finance (100% FREE)
- Bloomberg Terminal (reference - expensive)
- FactSet Professional (reference - expensive)
"""

from .base_financial_provider import (
    BaseFinancialProvider,
    FinancialDataResponse,
    FinancialProviderError,
)
from .openbb_provider import OpenBBProvider
from .sec_edgar_provider import SECEdgarProvider, FinancialModelingPrepProvider, IEXCloudProvider
from .alpha_vantage_provider import AlphaVantageProvider, PolygonProvider, YahooFinanceProvider
from .bloomberg_provider import BloombergProvider
from .factset_provider import FactSetProvider

__all__ = [
    # Base Classes
    "BaseFinancialProvider",
    "FinancialDataResponse", 
    "FinancialProviderError",
    
    # FREE Data Providers (Recommended)
    "OpenBBProvider",           # 100% FREE, comprehensive
    "SECEdgarProvider",         # 100% FREE, government data (highest reliability)
    "YahooFinanceProvider",     # 100% FREE, excellent coverage
    
    # Affordable Premium Providers
    "AlphaVantageProvider",     # FREE tier or $49/month
    "FinancialModelingPrepProvider",  # FREE tier or $15/month
    "IEXCloudProvider",         # FREE tier or $9/month
    "PolygonProvider",          # FREE tier or $25/month
    
    # Professional Platforms (Expensive - for reference)
    "BloombergProvider",        # $24K/year (too expensive)
    "FactSetProvider",          # $15K/year (too expensive)
]

# Recommended cost-effective setup for M&A analytics
RECOMMENDED_FREE_SETUP = {
    "primary": "OpenBB",        # 100% FREE, comprehensive
    "government": "SEC Edgar",  # 100% FREE, highest reliability
    "market_data": "Yahoo Finance",  # 100% FREE, excellent coverage
    "backup": "Alpha Vantage",  # FREE tier (500 calls/day)
    "total_cost": "$0/month",
    "data_quality": "Professional grade",
    "m&a_capability": "Complete M&A analysis"
}

AFFORDABLE_UPGRADE_OPTIONS = {
    "alpha_vantage_premium": "$49/month unlimited",
    "financial_modeling_prep": "$15/month for DCF models", 
    "iex_cloud_premium": "$9/month for enhanced data",
    "polygon_starter": "$25/month for comprehensive data",
    "total_max_cost": "$98/month (still 97% cheaper than Bloomberg)"
}

EXPENSIVE_PLATFORMS_COMPARISON = {
    "bloomberg_terminal": "$24,000/year ($2,000/month)",
    "factset_professional": "$15,000/year ($1,250/month)",
    "s&p_capital_iq": "$12,000/year ($1,000/month)",
    "total_professional": "$51,000/year ($4,250/month)",
    "cost_savings_vs_free": "100% savings using free providers",
    "cost_savings_vs_affordable": "97% savings using affordable providers"
}