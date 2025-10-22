"""Investment Banking DSPy Multi-Query Module - M&A and Financial Analysis Query Expansion."""

import dspy

from axiom.integrations.ai_providers import provider_factory
from axiom.config.settings import settings


class InvestmentBankingQueryExpansion(dspy.Signature):
    """Expand investment banking queries into focused financial analysis sub-queries."""

    original_query = dspy.InputField(
        desc="The original investment banking or M&A query"
    )
    analysis_type = dspy.InputField(
        desc="Type of analysis: ma_due_diligence, ma_valuation, market_analysis, risk_assessment, or financial_analysis"
    )
    company_context = dspy.InputField(desc="Company or sector context if available")
    expanded_queries = dspy.OutputField(
        desc="4-6 focused investment banking sub-queries covering financial metrics, strategic analysis, market intelligence, and risk factors"
    )


class MASpecificExpansion(dspy.Signature):
    """Expand M&A queries into deal-specific analysis dimensions."""

    ma_query = dspy.InputField(desc="M&A transaction or analysis query")
    target_company = dspy.InputField(desc="Target company name or sector")
    deal_stage = dspy.InputField(
        desc="Deal stage: early_evaluation, due_diligence, valuation, or post_merger"
    )
    ma_focused_queries = dspy.OutputField(
        desc="5-7 M&A-specific queries covering strategic rationale, financial due diligence, valuation methodologies, synergy analysis, integration planning, and regulatory considerations"
    )


class FinancialMetricsExpansion(dspy.Signature):
    """Expand financial analysis queries into comprehensive metrics coverage."""

    financial_query = dspy.InputField(
        desc="Financial analysis or company performance query"
    )
    metrics_scope = dspy.InputField(
        desc="Metrics scope: profitability, liquidity, leverage, efficiency, or valuation"
    )
    time_horizon = dspy.InputField(
        desc="Analysis time horizon: current, historical_trends, or forward_looking"
    )
    metrics_queries = dspy.OutputField(
        desc="4-5 financial metrics queries covering ratios, trends, peer comparisons, and industry benchmarks"
    )


class InvestmentBankingMultiQueryModule(dspy.Module):
    """Enhanced multi-query module for investment banking and M&A analysis."""

    def __init__(self):
        super().__init__()
        self.ib_expand = dspy.ChainOfThought(InvestmentBankingQueryExpansion)
        self.ma_expand = dspy.ChainOfThought(MASpecificExpansion)
        self.financial_expand = dspy.ChainOfThought(FinancialMetricsExpansion)

    def forward(
        self,
        query: str,
        analysis_type: str = "financial_analysis",
        company_context: str = "",
        **kwargs,
    ) -> list[str]:
        """Generate investment banking focused queries from input query."""

        try:
            # Determine query type and route accordingly
            if self._is_ma_query(query):
                return self._expand_ma_query(query, company_context, kwargs)
            elif self._is_financial_metrics_query(query):
                return self._expand_financial_query(query, kwargs)
            else:
                return self._expand_general_ib_query(
                    query, analysis_type, company_context
                )

        except Exception as e:
            print(f"Investment banking multi-query error: {e}")
            return self._fallback_queries(query)

    def _expand_ma_query(
        self, query: str, company_context: str, kwargs: dict
    ) -> list[str]:
        """Expand M&A-specific queries."""

        result = self.ma_expand(
            ma_query=query,
            target_company=company_context
            or kwargs.get("target_company", "target company"),
            deal_stage=kwargs.get("deal_stage", "due_diligence"),
        )

        if hasattr(result, "ma_focused_queries"):
            return self._parse_queries(result.ma_focused_queries)

        return self._fallback_ma_queries(query)

    def _expand_financial_query(self, query: str, kwargs: dict) -> list[str]:
        """Expand financial metrics queries."""

        result = self.financial_expand(
            financial_query=query,
            metrics_scope=kwargs.get("metrics_scope", "profitability"),
            time_horizon=kwargs.get("time_horizon", "current"),
        )

        if hasattr(result, "metrics_queries"):
            return self._parse_queries(result.metrics_queries)

        return self._fallback_financial_queries(query)

    def _expand_general_ib_query(
        self, query: str, analysis_type: str, company_context: str
    ) -> list[str]:
        """Expand general investment banking queries."""

        result = self.ib_expand(
            original_query=query,
            analysis_type=analysis_type,
            company_context=company_context,
        )

        if hasattr(result, "expanded_queries"):
            return self._parse_queries(result.expanded_queries)

        return self._fallback_queries(query)

    def _parse_queries(self, raw_queries: str) -> list[str]:
        """Parse and clean expanded queries."""

        queries = []

        # Split by common delimiters
        raw_splits = raw_queries.replace("\n", "|").replace(";", "|").split("|")

        for q in raw_splits:
            # Clean up formatting
            q = q.strip("- •*0123456789. ()[]{}")
            q = q.strip()

            if len(q) > 15 and len(q) < 200:  # Reasonable query length
                queries.append(q)

        return queries[:6]  # Limit to 6 queries

    def _is_ma_query(self, query: str) -> bool:
        """Check if query is M&A focused."""
        ma_terms = [
            "m&a",
            "merger",
            "acquisition",
            "acquire",
            "deal",
            "transaction",
            "combine",
        ]
        return any(term in query.lower() for term in ma_terms)

    def _is_financial_metrics_query(self, query: str) -> bool:
        """Check if query is financial metrics focused."""
        metrics_terms = [
            "ratio",
            "ebitda",
            "revenue",
            "profit",
            "cash flow",
            "debt",
            "equity",
            "roe",
            "roa",
        ]
        return any(term in query.lower() for term in metrics_terms)

    def _fallback_ma_queries(self, query: str) -> list[str]:
        """Fallback M&A queries if DSPy fails."""
        base_company = self._extract_company_name(query)

        return [
            f"{base_company} financial due diligence revenue profitability debt analysis",
            f"{base_company} strategic fit market position competitive advantages",
            f"{base_company} M&A valuation DCF comparable transactions multiples",
            f"{base_company} merger synergies cost savings revenue enhancement",
            f"{base_company} acquisition risks regulatory compliance integration challenges",
        ]

    def _fallback_financial_queries(self, query: str) -> list[str]:
        """Fallback financial queries if DSPy fails."""
        company = self._extract_company_name(query)

        return [
            f"{company} financial performance revenue growth profitability trends",
            f"{company} financial ratios liquidity leverage efficiency metrics",
            f"{company} cash flow analysis operating investing financing activities",
            f"{company} peer comparison industry benchmarks valuation multiples",
        ]

    def _fallback_queries(self, query: str) -> list[str]:
        """General fallback queries."""
        return [
            f"{query} financial analysis",
            f"{query} market analysis industry trends",
            f"{query} competitive positioning strategic analysis",
            f"{query} investment analysis valuation metrics",
        ]

    def _extract_company_name(self, query: str) -> str:
        """Simple company name extraction."""
        words = query.split()

        # Look for capitalized words (potential company names)
        for word in words:
            if word[0].isupper() and len(word) > 2 and word.isalpha():
                return word

        return "target company"


class SectorMultiQueryModule(dspy.Module):
    """Multi-query expansion for sector and industry analysis."""

    def __init__(self):
        super().__init__()
        self.sector_expand = dspy.ChainOfThought(SectorAnalysisExpansion)

    def forward(self, sector: str, analysis_focus: str = "comprehensive") -> list[str]:
        """Generate sector-specific analysis queries."""

        try:
            result = self.sector_expand(sector=sector, analysis_focus=analysis_focus)

            if hasattr(result, "sector_queries"):
                return self._parse_queries(result.sector_queries)

            return self._fallback_sector_queries(sector)

        except Exception as e:
            print(f"Sector multi-query error: {e}")
            return self._fallback_sector_queries(sector)

    def _parse_queries(self, raw_queries: str) -> list[str]:
        """Parse sector analysis queries."""
        queries = []
        raw_splits = raw_queries.replace("\n", "|").split("|")

        for q in raw_splits:
            q = q.strip("- •*0123456789. ")
            if len(q) > 15:
                queries.append(q)

        return queries[:5]

    def _fallback_sector_queries(self, sector: str) -> list[str]:
        """Fallback sector queries."""
        return [
            f"{sector} industry market size growth trends outlook",
            f"{sector} sector M&A activity consolidation transactions",
            f"{sector} industry valuation multiples trading comparables",
            f"{sector} competitive landscape key players market share",
            f"{sector} regulatory environment compliance requirements changes",
        ]


class SectorAnalysisExpansion(dspy.Signature):
    """Expand sector analysis into comprehensive industry intelligence."""

    sector = dspy.InputField(desc="Industry sector or market segment")
    analysis_focus = dspy.InputField(
        desc="Analysis focus: market_trends, ma_activity, competitive_landscape, or regulatory_environment"
    )
    sector_queries = dspy.OutputField(
        desc="5 sector-specific queries covering market dynamics, competitive positioning, M&A trends, and regulatory factors"
    )


def setup_dspy_with_provider():
    """Configure DSPy with available AI provider."""

    try:
        # Try to get an available provider
        providers = provider_factory.get_available_providers()

        if not providers:
            raise Exception("No AI providers available for DSPy")

        # Use first available provider
        provider_name = providers[0]
        config = settings.get_provider_config(provider_name)

        # Configure DSPy
        dspy.configure(
            lm=dspy.OpenAI(
                model=config["model_name"],
                api_base=config["base_url"],
                api_key=config["api_key"],
                max_tokens=1500,
                temperature=0.1,  # Conservative for financial analysis
            )
        )

        print(f"DSPy configured with {provider_name} provider")

    except Exception as e:
        print(f"DSPy setup error: {e}")
        # Fallback configuration
        dspy.configure(
            lm=dspy.OpenAI(
                model=settings.openai_model_name,
                api_base=settings.openai_base_url,
                api_key=settings.openai_api_key or "placeholder",
                max_tokens=1500,
            )
        )


# Legacy aliases
MultiQueryModule = InvestmentBankingMultiQueryModule
setup_dspy_client = setup_dspy_with_provider
