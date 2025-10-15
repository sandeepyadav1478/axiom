"""
M&A Valuation & Deal Structure Workflow

Comprehensive valuation analysis using multiple methodologies:
- DCF (Discounted Cash Flow) Analysis
- Comparable Company Analysis
- Precedent Transaction Analysis
- Synergy Analysis & Quantification
- Deal Structure Optimization
"""

import asyncio
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from axiom.ai_client_integrations import AIMessage, get_layer_provider
from axiom.config.ai_layer_config import AnalysisLayer
from axiom.config.schemas import Citation, Evidence
from axiom.tools.firecrawl_client import FirecrawlClient
from axiom.tools.tavily_client import TavilyClient
from axiom.tracing.langsmith_tracer import trace_node
from axiom.utils.error_handling import FinancialDataError


class DCFAnalysis(BaseModel):
    """DCF (Discounted Cash Flow) Analysis Results."""

    # Financial Projections
    base_case_value: float | None = Field(
        None, description="Base case enterprise value"
    )
    bull_case_value: float | None = Field(
        None, description="Bull case enterprise value"
    )
    bear_case_value: float | None = Field(
        None, description="Bear case enterprise value"
    )

    # DCF Assumptions
    discount_rate_wacc: float | None = Field(
        None, description="Weighted Average Cost of Capital"
    )
    terminal_growth_rate: float = Field(
        default=0.025, description="Terminal growth rate"
    )
    forecast_period: int = Field(default=5, description="Forecast period (years)")

    # Key Metrics
    projected_revenues: list[float] = Field(
        default=[], description="5-year revenue projections"
    )
    projected_ebitda: list[float] = Field(
        default=[], description="5-year EBITDA projections"
    )
    projected_fcf: list[float] = Field(default=[], description="5-year FCF projections")

    # Sensitivity Analysis
    wacc_sensitivity: dict[str, float] = Field(
        default={}, description="Value sensitivity to WACC changes"
    )
    growth_sensitivity: dict[str, float] = Field(
        default={}, description="Value sensitivity to growth changes"
    )

    # Analysis Quality
    projection_confidence: float = Field(
        default=0.0, description="Confidence in projections (0-1)"
    )
    methodology_notes: list[str] = Field(
        default=[], description="DCF methodology notes"
    )


class ComparableAnalysis(BaseModel):
    """Comparable Company Analysis Results."""

    # Trading Multiples
    ev_revenue_multiple: float | None = Field(
        None, description="EV/Revenue trading multiple"
    )
    ev_ebitda_multiple: float | None = Field(
        None, description="EV/EBITDA trading multiple"
    )
    pe_multiple: float | None = Field(None, description="P/E trading multiple")

    # Valuation Range
    comp_low_value: float | None = Field(
        None, description="Low-end comparable valuation"
    )
    comp_median_value: float | None = Field(
        None, description="Median comparable valuation"
    )
    comp_high_value: float | None = Field(
        None, description="High-end comparable valuation"
    )

    # Comparable Companies
    selected_comps: list[str] = Field(
        default=[], description="Selected comparable companies"
    )
    comp_count: int = Field(default=0, description="Number of comparables used")

    # Analysis Quality
    comparability_score: float = Field(
        default=0.0, description="Quality of comparables (0-1)"
    )
    multiple_reliability: str = Field(
        default="medium", description="Multiple reliability assessment"
    )


class PrecedentAnalysis(BaseModel):
    """Precedent Transaction Analysis Results."""

    # Transaction Multiples
    precedent_ev_revenue: float | None = Field(
        None, description="Precedent EV/Revenue multiple"
    )
    precedent_ev_ebitda: float | None = Field(
        None, description="Precedent EV/EBITDA multiple"
    )

    # Valuation Range
    precedent_low_value: float | None = Field(
        None, description="Low-end precedent valuation"
    )
    precedent_median_value: float | None = Field(
        None, description="Median precedent valuation"
    )
    precedent_high_value: float | None = Field(
        None, description="High-end precedent valuation"
    )

    # Transaction Details
    relevant_transactions: list[dict] = Field(
        default=[], description="Relevant precedent transactions"
    )
    transaction_count: int = Field(
        default=0, description="Number of precedent transactions"
    )
    avg_premium_paid: float | None = Field(
        None, description="Average acquisition premium"
    )

    # Analysis Quality
    relevance_score: float = Field(
        default=0.0, description="Precedent relevance score (0-1)"
    )
    recency_factor: float = Field(
        default=0.0, description="Transaction recency factor (0-1)"
    )


class SynergyAnalysis(BaseModel):
    """Synergy Analysis & Quantification Results."""

    # Revenue Synergies
    revenue_synergies: float = Field(
        default=0.0, description="Total revenue synergies ($)"
    )
    cross_selling_potential: float = Field(
        default=0.0, description="Cross-selling synergies ($)"
    )
    market_expansion_value: float = Field(
        default=0.0, description="Market expansion value ($)"
    )

    # Cost Synergies
    cost_synergies: float = Field(default=0.0, description="Total cost synergies ($)")
    headcount_savings: float = Field(
        default=0.0, description="Headcount reduction savings ($)"
    )
    overhead_elimination: float = Field(
        default=0.0, description="Overhead elimination ($)"
    )
    procurement_savings: float = Field(
        default=0.0, description="Procurement synergies ($)"
    )

    # Integration Costs
    one_time_costs: float = Field(
        default=0.0, description="One-time integration costs ($)"
    )
    systems_integration: float = Field(
        default=0.0, description="Systems integration costs ($)"
    )
    severance_costs: float = Field(
        default=0.0, description="Severance and restructuring ($)"
    )

    # Synergy Metrics
    total_synergies: float = Field(
        default=0.0, description="Total synergies (revenue + cost)"
    )
    net_synergies: float = Field(
        default=0.0, description="Net synergies (total - integration costs)"
    )
    synergy_multiple: float = Field(
        default=0.0, description="Synergies as % of deal value"
    )

    # Realization Profile
    year1_realization: float = Field(
        default=0.25, description="Year 1 synergy realization %"
    )
    year2_realization: float = Field(
        default=0.65, description="Year 2 synergy realization %"
    )
    year3_realization: float = Field(
        default=0.90, description="Year 3 synergy realization %"
    )

    # Risk Assessment
    synergy_risk_level: str = Field(
        default="medium", description="Synergy realization risk"
    )
    probability_of_achievement: float = Field(
        default=0.7, description="Probability of synergy achievement"
    )


class ValuationSummary(BaseModel):
    """Comprehensive Valuation Summary."""

    target_company: str = Field(..., description="Target company name")
    valuation_date: datetime = Field(default_factory=datetime.now)

    # Valuation Results by Method
    dcf_analysis: DCFAnalysis = Field(..., description="DCF analysis results")
    comparable_analysis: ComparableAnalysis = Field(
        ..., description="Comparable company analysis"
    )
    precedent_analysis: PrecedentAnalysis = Field(
        ..., description="Precedent transaction analysis"
    )
    synergy_analysis: SynergyAnalysis = Field(..., description="Synergy analysis")

    # Valuation Range & Recommendation
    valuation_low: float | None = Field(None, description="Low-end valuation estimate")
    valuation_base: float | None = Field(
        None, description="Base case valuation estimate"
    )
    valuation_high: float | None = Field(
        None, description="High-end valuation estimate"
    )
    recommended_offer_price: float | None = Field(
        None, description="Recommended offer price"
    )

    # Deal Structure
    cash_percentage: float = Field(
        default=0.7, description="Percentage cash consideration"
    )
    stock_percentage: float = Field(
        default=0.3, description="Percentage stock consideration"
    )
    earnout_amount: float = Field(default=0.0, description="Earnout consideration ($)")
    deal_premium: float | None = Field(None, description="Acquisition premium %")

    # Analysis Summary
    valuation_confidence: float = Field(
        default=0.0, description="Overall valuation confidence (0-1)"
    )
    key_value_drivers: list[str] = Field(
        default=[], description="Primary value drivers"
    )
    valuation_risks: list[str] = Field(default=[], description="Key valuation risks")

    # Supporting Data
    evidence: list[Evidence] = Field(default=[], description="Supporting evidence")
    citations: list[Citation] = Field(default=[], description="Source citations")

    # Metadata
    analysis_duration: float = Field(default=0.0, description="Total analysis time")
    methodology_weights: dict[str, float] = Field(
        default={"dcf": 0.4, "comparables": 0.35, "precedents": 0.25},
        description="Valuation methodology weights",
    )


class MAValuationWorkflow:
    """M&A Valuation and Deal Structure Workflow."""

    def __init__(self):
        self.tavily_client = TavilyClient()
        self.firecrawl_client = FirecrawlClient()

    @trace_node("ma_valuation_comprehensive")
    async def execute_comprehensive_valuation(
        self, target_company: str, target_metrics: dict[str, Any] = None
    ) -> ValuationSummary:
        """Execute comprehensive M&A valuation using multiple methodologies."""

        start_time = datetime.now()
        print(f"💰 Starting Comprehensive Valuation for {target_company}")

        try:
            # Execute valuation methodologies in parallel
            dcf_task = self.execute_dcf_analysis(target_company, target_metrics)
            comp_task = self.execute_comparable_analysis(target_company, target_metrics)
            precedent_task = self.execute_precedent_analysis(
                target_company, target_metrics
            )
            synergy_task = self.execute_synergy_analysis(target_company, target_metrics)

            # Wait for all analyses
            dcf_result, comp_result, precedent_result, synergy_result = (
                await asyncio.gather(
                    dcf_task,
                    comp_task,
                    precedent_task,
                    synergy_task,
                    return_exceptions=True,
                )
            )

            # Handle exceptions
            if isinstance(dcf_result, Exception):
                print(f"⚠️  DCF analysis failed: {str(dcf_result)}")
                dcf_result = DCFAnalysis()

            if isinstance(comp_result, Exception):
                print(f"⚠️  Comparable analysis failed: {str(comp_result)}")
                comp_result = ComparableAnalysis()

            if isinstance(precedent_result, Exception):
                print(f"⚠️  Precedent analysis failed: {str(precedent_result)}")
                precedent_result = PrecedentAnalysis()

            if isinstance(synergy_result, Exception):
                print(f"⚠️  Synergy analysis failed: {str(synergy_result)}")
                synergy_result = SynergyAnalysis()

            # Create valuation summary
            summary = ValuationSummary(
                target_company=target_company,
                dcf_analysis=dcf_result,
                comparable_analysis=comp_result,
                precedent_analysis=precedent_result,
                synergy_analysis=synergy_result,
            )

            # Calculate weighted valuation range
            summary = self._calculate_valuation_range(summary)

            # Generate deal structure recommendation
            summary = await self._optimize_deal_structure(summary)

            # Synthesize final assessment
            summary = await self._synthesize_valuation_summary(summary)

            execution_time = (datetime.now() - start_time).total_seconds()
            summary.analysis_duration = execution_time

            print(f"✅ Comprehensive Valuation completed in {execution_time:.1f}s")
            print(
                f"💎 Valuation Range: ${summary.valuation_low/1e9:.1f}B - ${summary.valuation_high/1e9:.1f}B"
            )

            return summary

        except Exception as e:
            raise FinancialDataError(
                f"Comprehensive valuation failed for {target_company}: {str(e)}",
                context={"target": target_company, "metrics": target_metrics},
            )

    @trace_node("ma_dcf_analysis")
    async def execute_dcf_analysis(
        self, target_company: str, target_metrics: dict[str, Any] = None
    ) -> DCFAnalysis:
        """Execute DCF (Discounted Cash Flow) valuation analysis."""

        print(f"📊 Building DCF Model for {target_company}")
        result = DCFAnalysis()

        try:
            # Step 1: Gather financial projections data
            projection_data = await self._gather_projection_data(target_company)

            # Step 2: Build financial model using AI
            financial_model = await self._build_dcf_model(
                target_company, projection_data
            )

            # Update DCF results
            result.base_case_value = financial_model.get("base_case_value")
            result.bull_case_value = financial_model.get("bull_case_value")
            result.bear_case_value = financial_model.get("bear_case_value")
            result.discount_rate_wacc = financial_model.get("wacc")
            result.terminal_growth_rate = financial_model.get("terminal_growth", 0.025)
            result.projected_revenues = financial_model.get("revenues", [])
            result.projected_ebitda = financial_model.get("ebitda", [])
            result.projected_fcf = financial_model.get("fcf", [])

            # Step 3: Sensitivity analysis
            result = await self._perform_dcf_sensitivity_analysis(
                result, target_company
            )

            result.projection_confidence = self._assess_projection_confidence(
                projection_data
            )

            return result

        except Exception as e:
            print(f"⚠️  DCF analysis failed: {str(e)}")
            result.methodology_notes.append(f"DCF analysis incomplete: {str(e)}")
            result.projection_confidence = 0.3
            return result

    @trace_node("ma_comparable_analysis")
    async def execute_comparable_analysis(
        self, target_company: str, target_metrics: dict[str, Any] = None
    ) -> ComparableAnalysis:
        """Execute Comparable Company Analysis."""

        print(f"🏢 Finding Comparable Companies for {target_company}")
        result = ComparableAnalysis()

        try:
            # Step 1: Identify comparable companies
            comparable_companies = await self._identify_comparable_companies(
                target_company
            )
            result.selected_comps = [comp["name"] for comp in comparable_companies]
            result.comp_count = len(comparable_companies)

            # Step 2: Gather comparable trading data
            trading_data = await self._gather_comparable_trading_data(
                comparable_companies
            )

            # Step 3: Calculate trading multiples
            multiples = self._calculate_trading_multiples(trading_data)
            result.ev_revenue_multiple = multiples.get("ev_revenue_median")
            result.ev_ebitda_multiple = multiples.get("ev_ebitda_median")
            result.pe_multiple = multiples.get("pe_median")

            # Step 4: Apply multiples to target
            if target_metrics:
                target_revenue = target_metrics.get("revenue", 0)
                target_ebitda = target_metrics.get("ebitda", 0)

                if target_revenue and result.ev_revenue_multiple:
                    revenue_based_value = target_revenue * result.ev_revenue_multiple
                    result.comp_median_value = revenue_based_value
                    result.comp_low_value = revenue_based_value * 0.85
                    result.comp_high_value = revenue_based_value * 1.15

                elif target_ebitda and result.ev_ebitda_multiple:
                    ebitda_based_value = target_ebitda * result.ev_ebitda_multiple
                    result.comp_median_value = ebitda_based_value
                    result.comp_low_value = ebitda_based_value * 0.85
                    result.comp_high_value = ebitda_based_value * 1.15

            result.comparability_score = self._assess_comparability_quality(
                comparable_companies, target_company
            )
            result.multiple_reliability = (
                "high" if result.comparability_score > 0.7 else "medium"
            )

            return result

        except Exception as e:
            print(f"⚠️  Comparable analysis failed: {str(e)}")
            result.multiple_reliability = "low"
            result.comparability_score = 0.3
            return result

    @trace_node("ma_precedent_analysis")
    async def execute_precedent_analysis(
        self, target_company: str, target_metrics: dict[str, Any] = None
    ) -> PrecedentAnalysis:
        """Execute Precedent Transaction Analysis."""

        print(f"📋 Analyzing Precedent Transactions for {target_company}")
        result = PrecedentAnalysis()

        try:
            # Step 1: Identify relevant transactions
            precedent_transactions = await self._identify_precedent_transactions(
                target_company
            )
            result.relevant_transactions = precedent_transactions
            result.transaction_count = len(precedent_transactions)

            # Step 2: Calculate transaction multiples
            if precedent_transactions:
                multiples = self._calculate_transaction_multiples(
                    precedent_transactions
                )
                result.precedent_ev_revenue = multiples.get("ev_revenue_median")
                result.precedent_ev_ebitda = multiples.get("ev_ebitda_median")
                result.avg_premium_paid = multiples.get("avg_premium")

                # Apply to target valuation
                if target_metrics and result.precedent_ev_revenue:
                    target_revenue = target_metrics.get("revenue", 0)
                    if target_revenue:
                        precedent_value = target_revenue * result.precedent_ev_revenue
                        result.precedent_median_value = precedent_value
                        result.precedent_low_value = precedent_value * 0.80
                        result.precedent_high_value = precedent_value * 1.20

            result.relevance_score = self._assess_precedent_relevance(
                precedent_transactions, target_company
            )
            result.recency_factor = self._calculate_recency_factor(
                precedent_transactions
            )

            return result

        except Exception as e:
            print(f"⚠️  Precedent analysis failed: {str(e)}")
            result.relevance_score = 0.3
            result.recency_factor = 0.5
            return result

    @trace_node("ma_synergy_analysis")
    async def execute_synergy_analysis(
        self, target_company: str, target_metrics: dict[str, Any] = None
    ) -> SynergyAnalysis:
        """Execute Synergy Analysis and Quantification."""

        print(f"🤝 Quantifying Synergies for {target_company}")
        result = SynergyAnalysis()

        try:
            # Step 1: Identify revenue synergies
            revenue_synergies = await self._identify_revenue_synergies(
                target_company, target_metrics
            )
            result.revenue_synergies = revenue_synergies.get("total", 0)
            result.cross_selling_potential = revenue_synergies.get("cross_selling", 0)
            result.market_expansion_value = revenue_synergies.get("market_expansion", 0)

            # Step 2: Identify cost synergies
            cost_synergies = await self._identify_cost_synergies(
                target_company, target_metrics
            )
            result.cost_synergies = cost_synergies.get("total", 0)
            result.headcount_savings = cost_synergies.get("headcount", 0)
            result.overhead_elimination = cost_synergies.get("overhead", 0)
            result.procurement_savings = cost_synergies.get("procurement", 0)

            # Step 3: Estimate integration costs
            integration_costs = await self._estimate_integration_costs(
                target_company, target_metrics
            )
            result.one_time_costs = integration_costs.get("total", 0)
            result.systems_integration = integration_costs.get("systems", 0)
            result.severance_costs = integration_costs.get("severance", 0)

            # Step 4: Calculate synergy metrics
            result.total_synergies = result.revenue_synergies + result.cost_synergies
            result.net_synergies = result.total_synergies - result.one_time_costs

            # Synergy realization risk assessment
            result = await self._assess_synergy_risks(result, target_company)

            return result

        except Exception as e:
            print(f"⚠️  Synergy analysis failed: {str(e)}")
            result.synergy_risk_level = "high"
            result.probability_of_achievement = 0.4
            return result

    def _calculate_valuation_range(self, summary: ValuationSummary) -> ValuationSummary:
        """Calculate weighted valuation range across methodologies."""

        valuations = []
        weights = summary.methodology_weights

        # DCF valuation
        if summary.dcf_analysis.base_case_value:
            valuations.append(
                {
                    "value": summary.dcf_analysis.base_case_value,
                    "weight": weights.get("dcf", 0.4),
                    "method": "dcf",
                }
            )

        # Comparable valuation
        if summary.comparable_analysis.comp_median_value:
            valuations.append(
                {
                    "value": summary.comparable_analysis.comp_median_value,
                    "weight": weights.get("comparables", 0.35),
                    "method": "comparable",
                }
            )

        # Precedent valuation
        if summary.precedent_analysis.precedent_median_value:
            valuations.append(
                {
                    "value": summary.precedent_analysis.precedent_median_value,
                    "weight": weights.get("precedents", 0.25),
                    "method": "precedent",
                }
            )

        if valuations:
            # Calculate weighted average
            weighted_value = sum(v["value"] * v["weight"] for v in valuations)
            total_weight = sum(v["weight"] for v in valuations)

            if total_weight > 0:
                base_value = weighted_value / total_weight
                summary.valuation_base = base_value
                summary.valuation_low = base_value * 0.85  # 15% discount
                summary.valuation_high = base_value * 1.20  # 20% premium

                # Recommended offer price (typically 10-15% premium to current value)
                summary.recommended_offer_price = base_value * 1.125  # 12.5% premium

        return summary

    async def _build_dcf_model(
        self, company: str, projection_data: dict
    ) -> dict[str, Any]:
        """Build DCF model using AI-powered financial modeling."""

        provider = get_layer_provider(AnalysisLayer.MA_VALUATION)
        if not provider:
            return {"base_case_value": 1000000000}  # Default $1B

        # Compile financial data for modeling
        evidence_content = "\n\n".join(
            [e.content for e in projection_data.get("evidence", [])[:8]]
        )

        messages = [
            AIMessage(
                role="system",
                content="""You are an investment banking analyst building DCF models for M&A valuation.
                Build 5-year financial projections and calculate enterprise value using:
                1. Revenue growth assumptions based on historical trends
                2. EBITDA margin projections
                3. Working capital and capex assumptions
                4. Free cash flow calculations
                5. WACC estimation and terminal value

                Provide base, bull, and bear case scenarios.""",
            ),
            AIMessage(
                role="user",
                content=f"""Build DCF model for {company}:

FINANCIAL DATA:
{evidence_content}

Required DCF Model Components:
1. 5-year revenue projections (base/bull/bear cases)
2. EBITDA margin assumptions and projections
3. WACC calculation (cost of equity + cost of debt)
4. Terminal growth rate (typically 2-3% for mature companies)
5. Free cash flow projections
6. Enterprise value calculation

Provide:
- Base Case Enterprise Value
- Bull Case Enterprise Value (upside scenario)
- Bear Case Enterprise Value (downside scenario)
- Key assumptions and methodology notes

Use conservative assumptions appropriate for M&A analysis.""",
            ),
        ]

        try:
            response = await provider.generate_response_async(
                messages, max_tokens=2000, temperature=0.05
            )
            return self._parse_dcf_model(response.content)
        except Exception as e:
            print(f"⚠️  DCF modeling failed: {str(e)}")
            return {
                "base_case_value": 1000000000,
                "methodology_notes": [f"DCF modeling error: {str(e)}"],
            }

    async def _identify_comparable_companies(self, target_company: str) -> list[dict]:
        """Identify relevant comparable public companies."""

        comparable_queries = [
            f"{target_company} competitors public companies trading multiples",
            f"{target_company} industry peers comparable companies analysis",
            f"{target_company} sector leaders public comparable companies",
        ]

        comparable_companies = []

        for query in comparable_queries:
            search_results = await self.tavily_client.search(
                query=query,
                search_type="company",
                max_results=10,
                include_domains=[
                    "bloomberg.com",
                    "reuters.com",
                    "sec.gov",
                    "finance.yahoo.com",
                ],
            )

            if search_results and search_results.get("results"):
                comps = self._extract_comparable_companies(search_results["results"])
                comparable_companies.extend(comps)

        # Remove duplicates and limit to top 8-12 comparables
        unique_comps = {}
        for comp in comparable_companies:
            name = comp.get("name", "").lower()
            if name and name not in unique_comps:
                unique_comps[name] = comp

        return list(unique_comps.values())[:12]

    async def _identify_revenue_synergies(
        self, target_company: str, target_metrics: dict = None
    ) -> dict[str, float]:
        """Identify and quantify revenue synergies."""

        provider = get_layer_provider(AnalysisLayer.MA_VALUATION)
        if not provider:
            return {"total": 0, "cross_selling": 0, "market_expansion": 0}

        messages = [
            AIMessage(
                role="system",
                content="""You are an M&A analyst quantifying revenue synergies.
                Identify realistic revenue synergy opportunities:
                1. Cross-selling to existing customer base
                2. Geographic/market expansion
                3. Product portfolio enhancement
                4. Channel partnerships and distribution

                Provide conservative estimates with clear rationale.""",
            ),
            AIMessage(
                role="user",
                content=f"""Quantify revenue synergies for {target_company} acquisition:

TARGET METRICS:
Revenue: ${target_metrics.get('revenue', 0):,.0f}
Customer Base: {target_metrics.get('customers', 'Unknown')}
Geographic Presence: {target_metrics.get('geography', 'Unknown')}

Estimate revenue synergies:
1. Cross-selling synergies ($): Revenue from selling acquirer products to target customers
2. Market expansion synergies ($): Revenue from geographic/market expansion
3. Total revenue synergies ($): Sum of all revenue synergies
4. Synergy realization timeline: Years to achieve full synergies
5. Probability of achievement: Conservative estimate (0-100%)

Use conservative assumptions (10-20% of target revenue typically).""",
            ),
        ]

        try:
            response = await provider.generate_response_async(
                messages, max_tokens=1200, temperature=0.1
            )
            return self._parse_revenue_synergies(response.content)
        except Exception:
            return {"total": 0, "cross_selling": 0, "market_expansion": 0}

    async def _optimize_deal_structure(
        self, summary: ValuationSummary
    ) -> ValuationSummary:
        """Optimize deal structure based on valuation analysis."""

        # Simple deal structure optimization
        total_synergies = summary.synergy_analysis.net_synergies
        base_valuation = summary.valuation_base or 1000000000

        # Adjust deal structure based on synergies and valuation confidence
        if total_synergies > base_valuation * 0.15:  # High synergies (>15% of value)
            summary.stock_percentage = 0.4  # Higher stock component for synergy sharing
            summary.cash_percentage = 0.6
            summary.earnout_amount = (
                total_synergies * 0.3
            )  # 30% of synergies as earnout
        elif total_synergies > base_valuation * 0.05:  # Moderate synergies (5-15%)
            summary.stock_percentage = 0.3
            summary.cash_percentage = 0.7
            summary.earnout_amount = total_synergies * 0.2
        else:  # Low synergies (<5%)
            summary.stock_percentage = 0.2
            summary.cash_percentage = 0.8
            summary.earnout_amount = 0

        # Calculate deal premium
        if summary.recommended_offer_price and summary.valuation_base:
            summary.deal_premium = (
                summary.recommended_offer_price / summary.valuation_base - 1
            ) * 100

        return summary

    # Helper parsing methods
    def _parse_dcf_model(self, content: str) -> dict[str, Any]:
        """Parse DCF model results from AI response."""

        import re

        model_data = {}

        # Extract enterprise values
        value_patterns = [
            (
                r"base case enterprise value:?\s*\$?([0-9,.]+)\s*(billion|million|b|m)",
                "base_case_value",
            ),
            (
                r"bull case enterprise value:?\s*\$?([0-9,.]+)\s*(billion|million|b|m)",
                "bull_case_value",
            ),
            (
                r"bear case enterprise value:?\s*\$?([0-9,.]+)\s*(billion|million|b|m)",
                "bear_case_value",
            ),
        ]

        for pattern, key in value_patterns:
            match = re.search(pattern, content.lower())
            if match:
                amount = float(match.group(1).replace(",", ""))
                unit = match.group(2).lower()
                multiplier = 1_000_000_000 if unit in ["billion", "b"] else 1_000_000
                model_data[key] = amount * multiplier

        # Extract WACC
        wacc_match = re.search(r"wacc:?\s*([0-9.]+)%?", content.lower())
        if wacc_match:
            model_data["wacc"] = float(wacc_match.group(1)) / 100

        # Extract terminal growth
        terminal_match = re.search(r"terminal growth:?\s*([0-9.]+)%?", content.lower())
        if terminal_match:
            model_data["terminal_growth"] = float(terminal_match.group(1)) / 100

        return model_data

    def _parse_revenue_synergies(self, content: str) -> dict[str, float]:
        """Parse revenue synergies from AI analysis."""

        import re

        synergies = {"total": 0, "cross_selling": 0, "market_expansion": 0}

        # Extract synergy amounts
        patterns = [
            (
                r"cross.selling synergies:?\s*\$?([0-9,.]+)\s*(million|billion)",
                "cross_selling",
            ),
            (
                r"market expansion synergies:?\s*\$?([0-9,.]+)\s*(million|billion)",
                "market_expansion",
            ),
            (r"total revenue synergies:?\s*\$?([0-9,.]+)\s*(million|billion)", "total"),
        ]

        for pattern, key in patterns:
            match = re.search(pattern, content.lower())
            if match:
                amount = float(match.group(1).replace(",", ""))
                unit = match.group(2).lower()
                multiplier = 1_000_000_000 if unit == "billion" else 1_000_000
                synergies[key] = amount * multiplier

        # If total not provided, calculate from components
        if synergies["total"] == 0 and (
            synergies["cross_selling"] or synergies["market_expansion"]
        ):
            synergies["total"] = (
                synergies["cross_selling"] + synergies["market_expansion"]
            )

        return synergies

    # Additional helper methods (simplified implementations)
    async def _gather_projection_data(self, company: str) -> dict:
        """Gather data for financial projections."""
        return {"evidence": [], "historical_data": {}}

    def _extract_comparable_companies(self, search_results: list) -> list[dict]:
        """Extract comparable companies from search results."""
        return [{"name": "Sample Comp 1"}, {"name": "Sample Comp 2"}]  # Placeholder

    async def _gather_comparable_trading_data(self, companies: list) -> dict:
        """Gather trading data for comparable companies."""
        return {"trading_multiples": {}}  # Placeholder

    def _calculate_trading_multiples(self, trading_data: dict) -> dict:
        """Calculate trading multiples from comparable data."""
        return {
            "ev_revenue_median": 3.5,
            "ev_ebitda_median": 12.0,
            "pe_median": 18.0,
        }  # Sample multiples

    def _assess_comparability_quality(self, comps: list, target: str) -> float:
        """Assess quality of comparable company selection."""
        return min(0.8, len(comps) / 10)  # Higher score with more comparables

    async def _identify_precedent_transactions(self, company: str) -> list[dict]:
        """Identify relevant precedent M&A transactions."""
        return [
            {
                "target": "Sample Target",
                "acquirer": "Sample Acquirer",
                "value": 2000000000,
            }
        ]  # Placeholder

    def _calculate_transaction_multiples(self, transactions: list) -> dict:
        """Calculate multiples from precedent transactions."""
        return {
            "ev_revenue_median": 4.2,
            "ev_ebitda_median": 14.5,
            "avg_premium": 0.25,
        }  # Sample

    def _assess_precedent_relevance(self, transactions: list, target: str) -> float:
        """Assess relevance of precedent transactions."""
        return min(0.8, len(transactions) / 8)

    def _calculate_recency_factor(self, transactions: list) -> float:
        """Calculate recency factor for precedent transactions."""
        return 0.8  # Default good recency

    async def _identify_cost_synergies(
        self, company: str, metrics: dict = None
    ) -> dict[str, float]:
        """Identify cost synergies (placeholder)."""
        return {
            "total": 50000000,
            "headcount": 30000000,
            "overhead": 15000000,
            "procurement": 5000000,
        }

    async def _estimate_integration_costs(
        self, company: str, metrics: dict = None
    ) -> dict[str, float]:
        """Estimate one-time integration costs (placeholder)."""
        return {"total": 25000000, "systems": 15000000, "severance": 10000000}

    async def _assess_synergy_risks(
        self, result: SynergyAnalysis, company: str
    ) -> SynergyAnalysis:
        """Assess synergy realization risks (placeholder)."""
        result.synergy_risk_level = "medium"
        result.probability_of_achievement = 0.75
        return result

    async def _perform_dcf_sensitivity_analysis(
        self, result: DCFAnalysis, company: str
    ) -> DCFAnalysis:
        """Perform DCF sensitivity analysis (placeholder)."""
        result.wacc_sensitivity = {
            "8%": 1200000000,
            "10%": 1000000000,
            "12%": 850000000,
        }
        result.growth_sensitivity = {
            "2%": 950000000,
            "2.5%": 1000000000,
            "3%": 1080000000,
        }
        return result

    def _assess_projection_confidence(self, projection_data: dict) -> float:
        """Assess confidence in financial projections."""
        evidence_count = len(projection_data.get("evidence", []))
        return min(0.9, evidence_count / 10)  # Scale with available evidence

    async def _synthesize_valuation_summary(
        self, summary: ValuationSummary
    ) -> ValuationSummary:
        """Synthesize final valuation summary and insights."""

        # Key value drivers
        summary.key_value_drivers = [
            "Revenue growth and market expansion",
            "Margin improvement through synergies",
            "Market position and competitive moat",
        ]

        # Valuation risks
        summary.valuation_risks = [
            "Integration execution risk",
            "Synergy realization uncertainty",
            "Market conditions and multiple compression",
        ]

        # Calculate overall confidence
        method_confidences = [
            summary.dcf_analysis.projection_confidence,
            summary.comparable_analysis.comparability_score,
            summary.precedent_analysis.relevance_score,
        ]
        summary.valuation_confidence = sum(method_confidences) / len(method_confidences)

        return summary


# Convenience functions
async def run_dcf_valuation(
    target_company: str, target_metrics: dict = None
) -> DCFAnalysis:
    """Run DCF valuation analysis."""
    workflow = MAValuationWorkflow()
    return await workflow.execute_dcf_analysis(target_company, target_metrics)


async def run_comprehensive_valuation(
    target_company: str, target_metrics: dict = None
) -> ValuationSummary:
    """Run comprehensive M&A valuation analysis."""
    workflow = MAValuationWorkflow()
    return await workflow.execute_comprehensive_valuation(
        target_company, target_metrics
    )
