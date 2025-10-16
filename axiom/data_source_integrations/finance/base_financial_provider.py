"""
Base Financial Data Provider Abstract Class

Abstract base class for all financial data provider integrations supporting
Bloomberg Terminal, FactSet, S&P Capital IQ, Refinitiv, and other professional sources.
"""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class FinancialDataResponse(BaseModel):
    """Standardized financial data response from all providers."""

    data_type: str  # "market_data", "fundamental", "comparable", "transaction"
    provider: str
    symbol_or_entity: str
    data_payload: dict[str, Any]
    metadata: dict[str, Any]
    timestamp: str
    confidence: float | None = None


class BaseFinancialProvider(ABC):
    """
    Abstract base class for all financial data provider integrations.

    Supports: Bloomberg Terminal, FactSet, S&P Capital IQ, Refinitiv, PitchBook, etc.

    Design Principles:
    - Standardized: Same interface regardless of provider
    - Professional: Investment banking grade data quality
    - Flexible: Easy to add new financial data sources
    - Cost-Effective: Optimized API usage and caching
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        subscription_level: str = "professional",
        **kwargs,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.subscription_level = subscription_level
        self.provider_name = self.__class__.__name__.replace("Provider", "")
        self.config = kwargs

    @abstractmethod
    def get_company_fundamentals(
        self,
        company_identifier: str,
        metrics: list[str] = None,
        **kwargs,
    ) -> FinancialDataResponse:
        """
        Get fundamental financial data for company analysis.

        Args:
            company_identifier: Ticker symbol, CUSIP, or company name
            metrics: Specific financial metrics to retrieve
            
        Returns:
            Standardized financial data response
        """
        pass

    @abstractmethod
    def get_comparable_companies(
        self,
        target_company: str,
        industry_sector: str = None,
        size_criteria: dict = None,
        **kwargs,
    ) -> FinancialDataResponse:
        """Get comparable public companies for analysis."""
        pass

    @abstractmethod
    def get_transaction_comparables(
        self,
        target_industry: str,
        deal_size_range: tuple[float, float] = None,
        time_period: str = "2_years",
        **kwargs,
    ) -> FinancialDataResponse:
        """Get comparable M&A transactions."""
        pass

    @abstractmethod
    def get_market_data(
        self,
        symbols: list[str],
        data_fields: list[str] = None,
        **kwargs,
    ) -> FinancialDataResponse:
        """Get real-time or historical market data."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available and properly configured."""
        pass

    def get_provider_info(self) -> dict[str, Any]:
        """Get provider information and capabilities."""
        return {
            "name": self.provider_name,
            "subscription_level": self.subscription_level,
            "base_url": self.base_url,
            "available": self.is_available(),
            "capabilities": self.get_capabilities(),
            "config": {k: v for k, v in self.config.items() if "key" not in k.lower()},
        }

    @abstractmethod
    def get_capabilities(self) -> dict[str, bool]:
        """Get provider-specific capabilities."""
        pass

    def estimate_query_cost(self, query_type: str, query_count: int = 1) -> float:
        """Estimate cost for API queries."""
        
        # Default cost estimates (override in specific providers)
        cost_map = {
            "fundamental": 2.00,
            "market_data": 1.50, 
            "comparable": 3.00,
            "transaction": 4.00,
            "screening": 5.00
        }
        
        return cost_map.get(query_type, 2.50) * query_count


class FinancialProviderError(Exception):
    """Custom exception for financial data provider errors."""

    def __init__(self, provider: str, message: str, original_error: Exception = None):
        self.provider = provider
        self.original_error = original_error
        super().__init__(f"[{provider}] {message}")