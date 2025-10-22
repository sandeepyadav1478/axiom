"""
Unified Financial Data Aggregator for M&A Analytics

Aggregates data from multiple financial providers (Tavily, FMP, Finnhub, Alpha Vantage)
with intelligent fallback mechanisms, data quality scoring, and consensus building.
"""

import os
from typing import Any, Dict, List, Optional
from datetime import datetime
import asyncio

from axiom.core.logging.axiom_logger import AxiomLogger
from .base_financial_provider import (
    BaseFinancialProvider,
    FinancialDataResponse,
    FinancialProviderError,
)
from .alpha_vantage_provider import AlphaVantageProvider, PolygonProvider
from .fmp_provider import FMPProvider
from .finnhub_provider import FinnhubProvider
from .yahoo_finance_provider import YahooFinanceProvider
from .iex_cloud_provider import IEXCloudProvider
from .openbb_provider import OpenBBProvider
from .sec_edgar_provider import SECEdgarProvider

logger = AxiomLogger("financial_data_aggregator")


class FinancialDataAggregator:
    """
    Unified financial data aggregator combining multiple providers.
    
    Features:
    - Multi-provider data aggregation with consensus building
    - Intelligent fallback when providers fail
    - Data quality scoring and confidence assessment
    - Cost optimization across providers
    - API key rotation support
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize aggregator with available financial data providers."""
        
        self.config = config or {}
        self.providers: Dict[str, BaseFinancialProvider] = {}
        self.provider_priority = []  # Ordered list of provider names by preference
        
        logger.info("Initializing Financial Data Aggregator")
        
        # Initialize providers from environment variables
        self._initialize_providers()
        
        logger.info(f"Initialized {len(self.providers)} financial data providers",
                   providers=list(self.providers.keys()))
    
    def _initialize_providers(self) -> None:
        """Initialize all available financial data providers from environment."""
        
        # Initialize Yahoo Finance (100% FREE, unlimited - BEST VALUE)
        try:
            self.providers["yahoo_finance"] = YahooFinanceProvider()
            self.provider_priority.append("yahoo_finance")
            logger.info("Yahoo Finance provider initialized (FREE, unlimited)")
        except Exception as e:
            logger.error(f"Failed to initialize Yahoo Finance: {e}")
        
        # Initialize OpenBB (100% FREE, comprehensive)
        try:
            self.providers["openbb"] = OpenBBProvider()
            self.provider_priority.append("openbb")
            logger.info("OpenBB provider initialized (FREE, comprehensive)")
        except Exception as e:
            logger.error(f"Failed to initialize OpenBB: {e}")
        
        # Initialize SEC Edgar (100% FREE, government data, highest reliability)
        user_agent = os.getenv("SEC_EDGAR_USER_AGENT")
        if user_agent:
            try:
                self.providers["sec_edgar"] = SECEdgarProvider(user_agent=user_agent)
                self.provider_priority.append("sec_edgar")
                logger.info("SEC Edgar provider initialized (FREE, official government data)")
            except Exception as e:
                logger.error(f"Failed to initialize SEC Edgar: {e}")
        
        # Initialize Alpha Vantage (FREE tier: 500 calls/day)
        av_keys = os.getenv("ALPHA_VANTAGE_API_KEY", "").split(",")
        if av_keys and av_keys[0]:
            try:
                self.providers["alpha_vantage"] = AlphaVantageProvider(
                    api_key=av_keys[0].strip(),
                    subscription_level="free"
                )
                self.provider_priority.append("alpha_vantage")
                logger.info("Alpha Vantage provider initialized",
                           keys_available=len(av_keys))
            except Exception as e:
                logger.error(f"Failed to initialize Alpha Vantage: {e}")
        
        # Initialize Polygon.io (FREE tier: 5 calls/minute)
        polygon_key = os.getenv("POLYGON_API_KEY")
        if polygon_key:
            try:
                self.providers["polygon"] = PolygonProvider(
                    api_key=polygon_key,
                    subscription_level="free"
                )
                self.provider_priority.append("polygon")
                logger.info("Polygon.io provider initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Polygon: {e}")
        
        # Initialize FMP (FREE tier: 250 calls/day)
        fmp_key = os.getenv("FMP_API_KEY") or os.getenv("FINANCIAL_MODELING_PREP_API_KEY")
        if fmp_key:
            try:
                self.providers["fmp"] = FMPProvider(
                    api_key=fmp_key,
                    subscription_level="free"
                )
                self.provider_priority.append("fmp")
                logger.info("FMP provider initialized")
            except Exception as e:
                logger.error(f"Failed to initialize FMP: {e}")
        
        # Initialize Finnhub (FREE tier: 60 calls/minute)
        finnhub_key = os.getenv("FINNHUB_API_KEY")
        if finnhub_key:
            try:
                self.providers["finnhub"] = FinnhubProvider(
                    api_key=finnhub_key,
                    subscription_level="free"
                )
                self.provider_priority.append("finnhub")
                logger.info("Finnhub provider initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Finnhub: {e}")
        
        # Initialize IEX Cloud (FREE tier: 500K credits/month)
        iex_key = os.getenv("IEX_CLOUD_API_KEY") or os.getenv("IEX_API_KEY")
        if iex_key:
            try:
                self.providers["iex_cloud"] = IEXCloudProvider(
                    api_key=iex_key,
                    subscription_level="free"
                )
                self.provider_priority.append("iex_cloud")
                logger.info("IEX Cloud provider initialized")
            except Exception as e:
                logger.error(f"Failed to initialize IEX Cloud: {e}")
        
        if not self.providers:
            logger.warning("No financial data providers available - check API keys in .env")
        else:
            logger.info(f"Financial Data Aggregator ready with {len(self.providers)} providers",
                       providers=self.provider_priority)
    
    async def get_company_fundamentals(
        self,
        company_identifier: str,
        metrics: Optional[List[str]] = None,
        use_consensus: bool = True,
        **kwargs
    ) -> FinancialDataResponse:
        """
        Get company fundamentals with multi-provider consensus.
        
        Args:
            company_identifier: Stock ticker or company name
            metrics: Specific metrics to retrieve
            use_consensus: If True, aggregate data from multiple providers
            **kwargs: Additional provider-specific parameters
        
        Returns:
            Aggregated financial data response with enhanced confidence
        """
        
        logger.info(f"Fetching fundamentals for {company_identifier}",
                   use_consensus=use_consensus, providers=len(self.providers))
        
        if not self.providers:
            raise FinancialProviderError(
                "Aggregator",
                "No financial data providers available"
            )
        
        if use_consensus and len(self.providers) > 1:
            return await self._get_consensus_fundamentals(
                company_identifier, metrics, **kwargs
            )
        else:
            return await self._get_single_provider_fundamentals(
                company_identifier, metrics, **kwargs
            )
    
    async def _get_consensus_fundamentals(
        self,
        company_identifier: str,
        metrics: Optional[List[str]],
        **kwargs
    ) -> FinancialDataResponse:
        """Get fundamentals from multiple providers and build consensus."""
        
        # Query all providers in parallel
        tasks = []
        for provider_name in self.provider_priority:
            if provider_name in self.providers:
                provider = self.providers[provider_name]
                task = self._safe_provider_call(
                    provider.get_company_fundamentals,
                    company_identifier,
                    metrics,
                    **kwargs
                )
                tasks.append((provider_name, task))
        
        # Gather results
        results = []
        for provider_name, task in tasks:
            try:
                result = await task
                if result:
                    results.append((provider_name, result))
                    logger.info(f"Got fundamentals from {provider_name}",
                               confidence=result.confidence)
            except Exception as e:
                logger.warning(f"{provider_name} failed: {e}")
        
        if not results:
            raise FinancialProviderError(
                "Aggregator",
                f"All providers failed for {company_identifier}"
            )
        
        # Build consensus response
        return self._build_consensus_response(
            results, "fundamental", company_identifier
        )
    
    async def _get_single_provider_fundamentals(
        self,
        company_identifier: str,
        metrics: Optional[List[str]],
        **kwargs
    ) -> FinancialDataResponse:
        """Get fundamentals from single provider with fallback."""
        
        last_error = None
        
        for provider_name in self.provider_priority:
            if provider_name not in self.providers:
                continue
            
            provider = self.providers[provider_name]
            
            try:
                logger.info(f"Trying {provider_name} for fundamentals")
                result = await self._safe_provider_call(
                    provider.get_company_fundamentals,
                    company_identifier,
                    metrics,
                    **kwargs
                )
                
                if result:
                    logger.info(f"Successfully got fundamentals from {provider_name}")
                    return result
                    
            except Exception as e:
                logger.warning(f"{provider_name} failed: {e}")
                last_error = e
                continue
        
        raise FinancialProviderError(
            "Aggregator",
            f"All providers failed for {company_identifier}: {last_error}"
        )
    
    async def get_comparable_companies(
        self,
        target_company: str,
        industry_sector: Optional[str] = None,
        size_criteria: Optional[Dict] = None,
        use_consensus: bool = True,
        **kwargs
    ) -> FinancialDataResponse:
        """
        Get comparable companies with multi-provider aggregation.
        
        Returns aggregated list of comparable companies from multiple sources.
        """
        
        logger.info(f"Finding comparables for {target_company}",
                   sector=industry_sector, use_consensus=use_consensus)
        
        if not self.providers:
            raise FinancialProviderError(
                "Aggregator",
                "No financial data providers available"
            )
        
        # Query all available providers
        tasks = []
        for provider_name in self.provider_priority:
            if provider_name in self.providers:
                provider = self.providers[provider_name]
                task = self._safe_provider_call(
                    provider.get_comparable_companies,
                    target_company,
                    industry_sector,
                    size_criteria,
                    **kwargs
                )
                tasks.append((provider_name, task))
        
        # Gather results
        results = []
        for provider_name, task in tasks:
            try:
                result = await task
                if result:
                    results.append((provider_name, result))
                    logger.info(f"Got comparables from {provider_name}")
            except Exception as e:
                logger.warning(f"{provider_name} comparables failed: {e}")
        
        if not results:
            raise FinancialProviderError(
                "Aggregator",
                f"All providers failed for comparable analysis"
            )
        
        # Aggregate comparable companies
        return self._aggregate_comparables(results, target_company)
    
    async def get_market_data(
        self,
        symbols: List[str],
        data_fields: Optional[List[str]] = None,
        **kwargs
    ) -> FinancialDataResponse:
        """
        Get real-time market data with provider fallback.
        
        Uses fastest available provider with fallback to alternatives.
        """
        
        logger.info(f"Fetching market data for {len(symbols)} symbols")
        
        if not self.providers:
            raise FinancialProviderError(
                "Aggregator",
                "No financial data providers available"
            )
        
        # Prefer free unlimited providers, then paid with free tiers
        preferred_order = ["yahoo_finance", "finnhub", "polygon", "iex_cloud", "alpha_vantage", "fmp"]
        
        last_error = None
        for provider_name in preferred_order:
            if provider_name not in self.providers:
                continue
            
            provider = self.providers[provider_name]
            
            try:
                logger.info(f"Trying {provider_name} for market data")
                result = await self._safe_provider_call(
                    provider.get_market_data,
                    symbols,
                    data_fields,
                    **kwargs
                )
                
                if result:
                    logger.info(f"Got market data from {provider_name}")
                    return result
                    
            except Exception as e:
                logger.warning(f"{provider_name} market data failed: {e}")
                last_error = e
                continue
        
        raise FinancialProviderError(
            "Aggregator",
            f"All providers failed for market data: {last_error}"
        )
    
    async def _safe_provider_call(self, method, *args, **kwargs):
        """Safely call provider method with timeout and error handling."""
        try:
            # Provider methods are synchronous, so run in thread pool
            result = await asyncio.wait_for(
                asyncio.to_thread(method, *args, **kwargs),
                timeout=30.0  # 30 second timeout per provider
            )
            return result
        except asyncio.TimeoutError:
            logger.warning(f"Provider call timed out: {method.__name__}")
            return None
        except Exception as e:
            logger.error(f"Provider call failed: {method.__name__}", error=str(e))
            raise
    
    def _build_consensus_response(
        self,
        results: List[tuple],
        data_type: str,
        entity: str
    ) -> FinancialDataResponse:
        """Build consensus response from multiple provider results."""
        
        if not results:
            raise ValueError("No results to build consensus from")
        
        # Use the result with highest confidence as base
        results.sort(key=lambda x: x[1].confidence or 0, reverse=True)
        best_provider, best_result = results[0]
        
        # Aggregate data payloads
        aggregated_payload = best_result.data_payload.copy()
        aggregated_payload["consensus_data"] = {
            "provider_count": len(results),
            "providers_used": [r[0] for r in results],
            "individual_results": {}
        }
        
        # Add data from other providers
        for provider_name, result in results:
            aggregated_payload["consensus_data"]["individual_results"][provider_name] = {
                "confidence": result.confidence,
                "timestamp": result.timestamp
            }
        
        # Calculate consensus confidence
        confidences = [r[1].confidence or 0.5 for r in results]
        consensus_confidence = sum(confidences) / len(confidences)
        
        # Boost confidence with multiple sources
        source_boost = min(0.15, len(results) * 0.05)
        final_confidence = min(1.0, consensus_confidence + source_boost)
        
        return FinancialDataResponse(
            data_type=data_type,
            provider=f"Aggregator ({len(results)} sources)",
            symbol_or_entity=entity,
            data_payload=aggregated_payload,
            metadata={
                "aggregation_method": "consensus",
                "provider_count": len(results),
                "providers": [r[0] for r in results],
                "confidence_boost": source_boost,
                "primary_source": best_provider
            },
            timestamp=datetime.now().isoformat(),
            confidence=final_confidence
        )
    
    def _aggregate_comparables(
        self,
        results: List[tuple],
        target_company: str
    ) -> FinancialDataResponse:
        """Aggregate comparable companies from multiple providers."""
        
        all_comparables = []
        seen_companies = set()
        
        for provider_name, result in results:
            comparables = result.data_payload.get("comparables", [])
            
            for comp in comparables:
                comp_name = comp.get("name", "").lower().strip()
                comp_symbol = comp.get("symbol", "").upper().strip()
                
                # Use symbol as unique identifier, fallback to name
                unique_id = comp_symbol if comp_symbol else comp_name
                
                if unique_id and unique_id not in seen_companies:
                    # Add provider info
                    comp["source_provider"] = provider_name
                    all_comparables.append(comp)
                    seen_companies.add(unique_id)
        
        # Sort by similarity/fit score if available
        all_comparables.sort(
            key=lambda x: x.get("similarity_score", 0) or x.get("sector_match", 0),
            reverse=True
        )
        
        return FinancialDataResponse(
            data_type="comparable",
            provider=f"Aggregator ({len(results)} sources)",
            symbol_or_entity=target_company,
            data_payload={
                "target_company": target_company,
                "comparable_count": len(all_comparables),
                "comparables": all_comparables[:20],  # Top 20
                "sources": [r[0] for r in results],
                "aggregation_method": "multi_source_deduplication"
            },
            metadata={
                "provider_count": len(results),
                "total_comparables_found": len(all_comparables),
                "deduplication": "by_symbol_and_name"
            },
            timestamp=datetime.now().isoformat(),
            confidence=0.85  # High confidence with multiple sources
        )
    
    def get_available_providers(self) -> List[str]:
        """Get list of initialized and available providers."""
        return list(self.providers.keys())
    
    def get_provider_info(self) -> Dict[str, Dict]:
        """Get detailed information about all providers."""
        info = {}
        for name, provider in self.providers.items():
            info[name] = provider.get_provider_info()
        return info
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health status of all providers."""
        health = {}
        
        for name, provider in self.providers.items():
            try:
                is_available = await asyncio.wait_for(
                    asyncio.to_thread(provider.is_available),
                    timeout=5.0
                )
                health[name] = is_available
            except Exception:
                health[name] = False
        
        return health


# Singleton instance for easy access
_aggregator_instance: Optional[FinancialDataAggregator] = None


def get_financial_aggregator() -> FinancialDataAggregator:
    """Get or create singleton financial data aggregator instance."""
    global _aggregator_instance
    
    if _aggregator_instance is None:
        _aggregator_instance = FinancialDataAggregator()
    
    return _aggregator_instance