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

from .alpha_vantage_provider import (
    AlphaVantageProvider,
    PolygonProvider,
)
from .base_financial_provider import (
    BaseFinancialProvider,
    FinancialDataResponse,
    FinancialProviderError,
)
from .bloomberg_provider import BloombergProvider
from .factset_provider import FactSetProvider
from .openbb_provider import OpenBBProvider
from .sec_edgar_provider import SECEdgarProvider

# New enhanced financial data sources (Phase 2)
from .yahoo_finance_provider import YahooFinanceProvider
from .finnhub_provider import FinnhubProvider
from .iex_cloud_provider import IEXCloudProvider
from .fmp_provider import FMPProvider

__all__ = [
    # Base Classes
    "BaseFinancialProvider",
    "FinancialDataResponse",
    "FinancialProviderError",

    # FREE Data Providers (Recommended)
    "OpenBBProvider",           # 100% FREE, comprehensive
    "SECEdgarProvider",         # 100% FREE, government data (highest reliability)
    "YahooFinanceProvider",     # 100% FREE, unlimited calls with yfinance

    # Affordable Premium Providers (Phase 2 Enhanced)
    "AlphaVantageProvider",     # FREE tier or $49/month
    "FinnhubProvider",          # FREE tier (60 calls/min) or $7.99/month
    "IEXCloudProvider",         # FREE tier (500K credits/month) or $9/month
    "FMPProvider",              # FREE tier (250 calls/day) or $14/month
    "PolygonProvider",          # FREE tier or $25/month

    # Professional Platforms (Expensive - for reference)
    "BloombergProvider",        # $24K/year (too expensive)
    "FactSetProvider",          # $15K/year (too expensive)
]

# Phase 2 Enhanced: Recommended cost-effective setup for M&A analytics
RECOMMENDED_FREE_SETUP = {
    "primary": "Yahoo Finance",     # 100% FREE, unlimited calls, excellent library support
    "comprehensive": "OpenBB",      # 100% FREE, professional-grade platform
    "government": "SEC Edgar",      # 100% FREE, highest reliability
    "real_time": "Finnhub",        # FREE tier 60 calls/min
    "backup": "Alpha Vantage",     # FREE tier 500 calls/day
    "total_cost": "$0/month",
    "data_quality": "Professional grade",
    "m&a_capability": "Complete M&A analysis with enhanced coverage"
}

ENHANCED_AFFORDABLE_OPTIONS = {
    "yahoo_finance": "$0/month - 100% FREE unlimited (BEST VALUE)",
    "finnhub_premium": "$7.99/month - Most affordable premium",
    "iex_cloud_start": "$9/month - 5M credits",
    "fmp_starter": "$14/month - 10K calls + DCF models",
    "alpha_vantage_premium": "$49/month - Unlimited calls",
    "polygon_starter": "$25/month - Comprehensive data",
    "total_max_cost": "$105/month (still 96% cheaper than Bloomberg)"
}

EXPENSIVE_PLATFORMS_COMPARISON = {
    "bloomberg_terminal": "$24,000/year ($2,000/month)",
    "factset_professional": "$15,000/year ($1,250/month)",
    "s&p_capital_iq": "$12,000/year ($1,000/month)",
    "total_professional": "$51,000/year ($4,250/month)",
    "cost_savings_vs_free": "100% savings using free providers",
    "cost_savings_vs_affordable": "97% savings using affordable providers"
}
