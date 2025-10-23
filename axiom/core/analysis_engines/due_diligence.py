"""
M&A Due Diligence Workflow Modules

Comprehensive due diligence analysis covering Financial, Commercial,
Legal, and Operational aspects of M&A transactions.
"""

import asyncio
import re
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from axiom.integrations.ai_providers import AIMessage, get_layer_provider
from axiom.config.ai_layer_config import AnalysisLayer
from axiom.config.schemas import Citation, Evidence
from axiom.integrations.search_tools.firecrawl_client import FirecrawlClient
from axiom.integrations.search_tools.tavily_client import TavilyClient
from axiom.tracing.langsmith_tracer import trace_node
from axiom.core.validation.error_handling import FinancialDataError
from axiom.core.logging.axiom_logger import ma_dd_logger


class FinancialDDResult(BaseModel):
    """Financial Due Diligence Analysis Result."""

    # Revenue Analysis
    revenue_quality_score: float = Field(
        default=0.0, description="Revenue quality score (0-1)"
    )
    revenue_growth_trend: str = Field(
        default="stable", description="Revenue growth trend"
    )
    revenue_concentration_risk: str = Field(
        default="medium", description="Customer concentration risk"
    )
    recurring_revenue_pct: float | None = Field(
        None, description="Recurring revenue percentage"
    )

    # Profitability Analysis
    ebitda_quality_score: float = Field(
        default=0.0, description="EBITDA quality score (0-1)"
    )
    margin_sustainability: str = Field(
        default="stable", description="Margin sustainability assessment"
    )
    cost_structure_efficiency: float = Field(
        default=0.0, description="Cost efficiency score (0-1)"
    )

    # Financial Position
    balance_sheet_strength: float = Field(
        default=0.0, description="Balance sheet strength (0-1)"
    )
    liquidity_position: str = Field(
        default="adequate", description="Liquidity assessment"
    )
    debt_profile_risk: str = Field(
        default="medium", description="Debt profile risk level"
    )
    working_capital_efficiency: float | None = Field(
        None, description="Working capital efficiency"
    )

    # Cash Flow Analysis
    fcf_conversion_rate: float | None = Field(
        None, description="Free cash flow conversion rate"
    )
    cash_generation_stability: str = Field(
        default="stable", description="Cash generation stability"
    )
    capex_intensity: float | None = Field(
        None, description="Capital expenditure intensity"
    )

    # Key Findings
    critical_financial_issues: list[str] = Field(
        default=[], description="Critical financial issues identified"
    )
    financial_strengths: list[str] = Field(
        default=[], description="Key financial strengths"
    )
    valuation_considerations: list[str] = Field(
        default=[], description="Key valuation factors"
    )

    # Supporting Evidence
    evidence: list[Evidence] = Field(default=[], description="Supporting evidence")
    citations: list[Citation] = Field(default=[], description="Source citations")

    # Metadata
    analysis_confidence: float = Field(
        default=0.0, description="Analysis confidence (0-1)"
    )
    timestamp: datetime = Field(default_factory=datetime.now)


class CommercialDDResult(BaseModel):
    """Commercial Due Diligence Analysis Result."""

    # Market Analysis
    market_size_growth: str = Field(
        default="unknown", description="Market size and growth assessment"
    )
    market_position_strength: float = Field(
        default=0.0, description="Market position strength (0-1)"
    )
    competitive_differentiation: str = Field(
        default="moderate", description="Competitive differentiation"
    )

    # Customer Analysis
    customer_diversification: float = Field(
        default=0.0, description="Customer diversification score (0-1)"
    )
    customer_loyalty_strength: str = Field(
        default="medium", description="Customer loyalty assessment"
    )
    pricing_power: str = Field(
        default="limited", description="Pricing power assessment"
    )

    # Product/Service Portfolio
    product_portfolio_strength: float = Field(
        default=0.0, description="Product portfolio strength (0-1)"
    )
    innovation_capability: str = Field(
        default="moderate", description="Innovation capability"
    )
    technology_moat: str = Field(
        default="limited", description="Technology competitive moat"
    )

    # Commercial Insights
    growth_drivers: list[str] = Field(default=[], description="Key growth drivers")
    commercial_risks: list[str] = Field(
        default=[], description="Commercial risks identified"
    )
    market_opportunities: list[str] = Field(
        default=[], description="Market opportunities"
    )

    # Supporting Evidence
    evidence: list[Evidence] = Field(default=[], description="Supporting evidence")
    analysis_confidence: float = Field(
        default=0.0, description="Analysis confidence (0-1)"
    )
    timestamp: datetime = Field(default_factory=datetime.now)


class OperationalDDResult(BaseModel):
    """Operational Due Diligence Analysis Result."""

    # Management & Organization
    management_quality: float = Field(
        default=0.0, description="Management team quality (0-1)"
    )
    organizational_capability: str = Field(
        default="adequate", description="Organizational capability"
    )
    key_person_risk: str = Field(
        default="medium", description="Key person dependency risk"
    )

    # Operations & Efficiency
    operational_efficiency: float = Field(
        default=0.0, description="Operational efficiency score (0-1)"
    )
    technology_systems: str = Field(
        default="adequate", description="Technology systems assessment"
    )
    process_maturity: str = Field(
        default="developing", description="Process maturity level"
    )

    # Supply Chain & Operations
    supply_chain_resilience: str = Field(
        default="moderate", description="Supply chain resilience"
    )
    vendor_concentration_risk: str = Field(
        default="medium", description="Vendor concentration risk"
    )
    operational_scalability: str = Field(
        default="limited", description="Operational scalability"
    )

    # Human Capital
    talent_retention_risk: str = Field(
        default="medium", description="Talent retention risk"
    )
    skill_gap_assessment: list[str] = Field(
        default=[], description="Critical skill gaps"
    )
    cultural_integration_risk: str = Field(
        default="medium", description="Cultural integration risk"
    )

    # Key Findings
    operational_strengths: list[str] = Field(
        default=[], description="Operational strengths"
    )
    integration_challenges: list[str] = Field(
        default=[], description="Integration challenges"
    )
    improvement_opportunities: list[str] = Field(
        default=[], description="Operational improvements"
    )

    # Supporting Evidence
    evidence: list[Evidence] = Field(default=[], description="Supporting evidence")
    analysis_confidence: float = Field(
        default=0.0, description="Analysis confidence (0-1)"
    )
    timestamp: datetime = Field(default_factory=datetime.now)


class ComprehensiveDDResult(BaseModel):
    """Comprehensive Due Diligence Analysis Result."""

    target_company: str = Field(..., description="Target company name")
    analysis_date: datetime = Field(default_factory=datetime.now)

    # DD Module Results
    financial_dd: FinancialDDResult = Field(
        ..., description="Financial due diligence results"
    )
    commercial_dd: CommercialDDResult = Field(
        ..., description="Commercial due diligence results"
    )
    operational_dd: OperationalDDResult = Field(
        ..., description="Operational due diligence results"
    )

    # Overall Assessment
    overall_risk_rating: str = Field(
        default="medium", description="Overall risk assessment"
    )
    investment_recommendation: str = Field(
        default="hold", description="Investment recommendation"
    )
    key_deal_breakers: list[str] = Field(
        default=[], description="Potential deal breakers"
    )
    value_creation_opportunities: list[str] = Field(
        default=[], description="Value creation opportunities"
    )

    # Executive Summary
    executive_summary: str = Field(
        default="", description="Executive summary of findings"
    )
    next_steps: list[str] = Field(default=[], description="Recommended next steps")

    # Metadata
    total_analysis_time: float = Field(
        default=0.0, description="Total analysis time (seconds)"
    )
    overall_confidence: float = Field(
        default=0.0, description="Overall analysis confidence (0-1)"
    )


class MADueDiligenceWorkflow:
    """Comprehensive M&A Due Diligence Workflow."""

    def __init__(self):
        self.tavily_client = TavilyClient()
        self.firecrawl_client = FirecrawlClient()

    @trace_node("ma_due_diligence_comprehensive")
    async def execute_comprehensive_dd(
        self, target_company: str, analysis_scope: str = "full"
    ) -> ComprehensiveDDResult:
        """Execute comprehensive due diligence analysis."""

        start_time = datetime.now()
        ma_dd_logger.info(f"Starting Comprehensive Due Diligence for {target_company}")

        try:
            # Execute all DD modules in parallel for efficiency
            financial_dd_task = self.execute_financial_dd(target_company)
            commercial_dd_task = self.execute_commercial_dd(target_company)
            operational_dd_task = self.execute_operational_dd(target_company)

            # Wait for all DD analyses to complete
            financial_dd, commercial_dd, operational_dd = await asyncio.gather(
                financial_dd_task,
                commercial_dd_task,
                operational_dd_task,
                return_exceptions=True,
            )

            # Handle any exceptions
            if isinstance(financial_dd, Exception):
                ma_dd_logger.warning(f"Financial DD failed: {str(financial_dd)}")
                financial_dd = FinancialDDResult()

            if isinstance(commercial_dd, Exception):
                ma_dd_logger.warning(f"Commercial DD failed: {str(commercial_dd)}")
                commercial_dd = CommercialDDResult()

            if isinstance(operational_dd, Exception):
                ma_dd_logger.warning(f"Operational DD failed: {str(operational_dd)}")
                operational_dd = OperationalDDResult()

            # Create comprehensive result
            result = ComprehensiveDDResult(
                target_company=target_company,
                financial_dd=financial_dd,
                commercial_dd=commercial_dd,
                operational_dd=operational_dd,
            )

            # Generate overall assessment
            result = await self._synthesize_overall_assessment(result)

            execution_time = (datetime.now() - start_time).total_seconds()
            result.total_analysis_time = execution_time

            ma_dd_logger.info(f"Comprehensive DD completed in {execution_time:.1f}s",
                            overall_risk=result.overall_risk_rating,
                            recommendation=result.investment_recommendation)

            return result

        except Exception as e:
            raise FinancialDataError(
                f"Comprehensive due diligence failed for {target_company}: {str(e)}",
                context={"target": target_company, "scope": analysis_scope},
            )

    @trace_node("ma_financial_dd")
    async def execute_financial_dd(self, target_company: str) -> FinancialDDResult:
        """Execute Financial Due Diligence analysis."""

        ma_dd_logger.info(f"Analyzing Financial DD for {target_company}")
        result = FinancialDDResult()

        try:
            # Step 1: Gather financial information
            financial_data = await self._gather_financial_information(target_company)

            # Step 2: Analyze revenue quality
            revenue_analysis = await self._analyze_revenue_quality(
                target_company, financial_data
            )
            result.revenue_quality_score = revenue_analysis["quality_score"]
            result.revenue_growth_trend = revenue_analysis["growth_trend"]
            result.revenue_concentration_risk = revenue_analysis["concentration_risk"]
            result.recurring_revenue_pct = revenue_analysis.get("recurring_pct")

            # Step 3: Assess profitability
            profitability_analysis = await self._analyze_profitability(
                target_company, financial_data
            )
            result.ebitda_quality_score = profitability_analysis["ebitda_quality"]
            result.margin_sustainability = profitability_analysis["margin_trend"]
            result.cost_structure_efficiency = profitability_analysis["cost_efficiency"]

            # Step 4: Evaluate financial position
            balance_sheet_analysis = await self._analyze_balance_sheet(
                target_company, financial_data
            )
            result.balance_sheet_strength = balance_sheet_analysis["strength_score"]
            result.liquidity_position = balance_sheet_analysis["liquidity"]
            result.debt_profile_risk = balance_sheet_analysis["debt_risk"]

            # Step 5: Cash flow analysis
            cashflow_analysis = await self._analyze_cash_flows(
                target_company, financial_data
            )
            result.fcf_conversion_rate = cashflow_analysis.get("fcf_conversion")
            result.cash_generation_stability = cashflow_analysis["stability"]
            result.capex_intensity = cashflow_analysis.get("capex_intensity")

            # Step 6: Synthesize findings
            result = await self._synthesize_financial_findings(result, target_company)

            # Add evidence
            result.evidence.extend(financial_data.get("evidence", []))
            result.analysis_confidence = min(
                0.8, len(result.evidence) / 10
            )  # Scale with evidence

            return result

        except Exception as e:
            ma_dd_logger.error(f"Financial DD analysis failed: {str(e)}")
            result.critical_financial_issues.append(f"Analysis incomplete: {str(e)}")
            result.analysis_confidence = 0.3
            return result

    @trace_node("ma_commercial_dd")
    async def execute_commercial_dd(self, target_company: str) -> CommercialDDResult:
        """Execute Commercial Due Diligence analysis."""

        ma_dd_logger.info(f"Analyzing Commercial DD for {target_company}")
        result = CommercialDDResult()

        try:
            # Step 1: Market analysis
            market_data = await self._analyze_market_dynamics(target_company)
            result.market_size_growth = market_data["market_assessment"]
            result.market_position_strength = market_data["position_strength"]
            result.competitive_differentiation = market_data["differentiation"]

            # Step 2: Customer analysis
            customer_data = await self._analyze_customer_base(target_company)
            result.customer_diversification = customer_data["diversification_score"]
            result.customer_loyalty_strength = customer_data["loyalty"]
            result.pricing_power = customer_data["pricing_power"]

            # Step 3: Product/service portfolio
            product_data = await self._analyze_product_portfolio(target_company)
            result.product_portfolio_strength = product_data["portfolio_strength"]
            result.innovation_capability = product_data["innovation"]
            result.technology_moat = product_data["tech_moat"]

            # Step 4: Synthesize commercial insights
            result = await self._synthesize_commercial_findings(result, target_company)

            result.analysis_confidence = 0.75  # Default confidence
            return result

        except Exception as e:
            ma_dd_logger.error(f"Commercial DD analysis failed: {str(e)}")
            result.commercial_risks.append(f"Analysis incomplete: {str(e)}")
            result.analysis_confidence = 0.3
            return result

    @trace_node("ma_operational_dd")
    async def execute_operational_dd(self, target_company: str) -> OperationalDDResult:
        """Execute Operational Due Diligence analysis."""

        ma_dd_logger.info(f"Analyzing Operational DD for {target_company}")
        result = OperationalDDResult()

        try:
            # Step 1: Management assessment
            mgmt_data = await self._analyze_management_team(target_company)
            result.management_quality = mgmt_data["quality_score"]
            result.key_person_risk = mgmt_data["key_person_risk"]
            result.organizational_capability = mgmt_data["org_capability"]

            # Step 2: Operational efficiency
            ops_data = await self._analyze_operational_efficiency(target_company)
            result.operational_efficiency = ops_data["efficiency_score"]
            result.technology_systems = ops_data["tech_systems"]
            result.process_maturity = ops_data["process_maturity"]

            # Step 3: Supply chain and vendor analysis
            supply_data = await self._analyze_supply_chain(target_company)
            result.supply_chain_resilience = supply_data["resilience"]
            result.vendor_concentration_risk = supply_data["vendor_risk"]
            result.operational_scalability = supply_data["scalability"]

            # Step 4: Human capital analysis
            hr_data = await self._analyze_human_capital(target_company)
            result.talent_retention_risk = hr_data["retention_risk"]
            result.skill_gap_assessment = hr_data["skill_gaps"]
            result.cultural_integration_risk = hr_data["culture_risk"]

            # Step 5: Synthesize operational findings
            result = await self._synthesize_operational_findings(result, target_company)

            result.analysis_confidence = 0.7  # Default confidence
            return result

        except Exception as e:
            ma_dd_logger.error(f"Operational DD analysis failed: {str(e)}")
            result.integration_challenges.append(f"Analysis incomplete: {str(e)}")
            result.analysis_confidence = 0.3
            return result

    # Financial DD Helper Methods
    async def _gather_financial_information(self, company: str) -> dict[str, Any]:
        """Gather comprehensive financial information using financial providers."""

        financial_data = {"evidence": [], "fundamentals": {}}

        try:
            # Get fundamental data from financial providers first
            logger.info("Gathering financial information from providers",
                       company=company)
            
            fundamentals = await self.financial_aggregator.get_company_fundamentals(
                company_identifier=company,
                use_consensus=True
            )
            
            if fundamentals and fundamentals.data_payload:
                payload = fundamentals.data_payload
                financial_data["fundamentals"] = payload
                
                # Create evidence from comprehensive financial data
                evidence_content = f"""Financial Analysis for {company}:
Revenue: ${payload.get('annual_revenue', 0):,.0f}
EBITDA: ${payload.get('ebitda', 0):,.0f}
EBITDA Margin: {payload.get('ebitda_margin', 0):.1%}
Market Cap: ${payload.get('market_cap', 0):,.0f}
Growth Rate: {payload.get('revenue_growth', 0):.1%}

Profitability Ratios: {payload.get('profitability_ratios', {})}
Liquidity Ratios: {payload.get('liquidity_ratios', {})}
Leverage Ratios: {payload.get('leverage_ratios', {})}
"""
                
                evidence = Evidence(
                    content=evidence_content,
                    source=f"Financial Providers ({fundamentals.provider})",
                    relevance_score=fundamentals.confidence or 0.90,
                    evidence_type="financial_data",
                    source_url="",
                    timestamp=datetime.now()
                )
                financial_data["evidence"].append(evidence)
                
                ma_dd_logger.info("Retrieved comprehensive financial data",
                                company=company, provider=fundamentals.provider,
                                confidence=fundamentals.confidence)
        
        except Exception as e:
            ma_dd_logger.warning("Could not get financial data from providers",
                                company=company, error=str(e))
        
        # Supplement with web search for additional context
        financial_queries = [
            f"{company} financial statements annual report 10-K",
            f"{company} quarterly earnings Q4 Q3 Q2 Q1 results",
            f"{company} revenue EBITDA profit margins financial metrics",
        ]

        for query in financial_queries[:2]:  # Limit queries if we have provider data
            search_results = await self.tavily_client.search(
                query=query,
                search_type="financial",
                max_results=5,
                include_domains=[
                    "sec.gov",
                    "investor.",
                    "ir.",
                    "bloomberg.com",
                    "reuters.com",
                ],
            )

            if search_results and search_results.get("results"):
                for result in search_results["results"][:2]:
                    evidence = Evidence(
                        content=result.get("content", result.get("snippet", "")),
                        source=result.get("title", "Financial Data"),
                        relevance_score=result.get("score", 0.7),
                        evidence_type="financial_analysis",
                        source_url=result.get("url", ""),
                        timestamp=datetime.now(),
                    )
                    financial_data["evidence"].append(evidence)

        return financial_data

    async def _analyze_revenue_quality(
        self, company: str, financial_data: dict
    ) -> dict[str, Any]:
        """Analyze revenue quality and sustainability."""

        provider = get_layer_provider(AnalysisLayer.MA_DUE_DILIGENCE)
        if not provider:
            return {
                "quality_score": 0.5,
                "growth_trend": "unknown",
                "concentration_risk": "medium",
            }

        # Compile financial evidence content
        evidence_content = "\n\n".join(
            [e.content for e in financial_data.get("evidence", [])[:5]]
        )

        messages = [
            AIMessage(
                role="system",
                content="""You are a senior investment banking analyst performing financial due diligence.
                Analyze revenue quality focusing on:
                1. Revenue sustainability and recurring nature
                2. Customer concentration and diversification
                3. Revenue growth trends and drivers
                4. Seasonal or cyclical patterns
                5. Revenue recognition quality

                Provide numerical scores (0.0-1.0) and clear assessments.""",
            ),
            AIMessage(
                role="user",
                content=f"""Analyze revenue quality for {company}:

FINANCIAL DATA:
{evidence_content}

Provide analysis covering:
1. Revenue Quality Score (0.0-1.0): Overall revenue sustainability
2. Growth Trend: accelerating/stable/declining
3. Concentration Risk: low/medium/high customer concentration
4. Recurring Revenue %: Estimate percentage of recurring revenue
5. Key Revenue Drivers: Primary sources of revenue growth

Focus on revenue sustainability and predictability for M&A valuation.""",
            ),
        ]

        try:
            response = await provider.generate_response_async(
                messages, max_tokens=1000, temperature=0.05
            )
            return self._parse_revenue_analysis(response.content)
        except Exception as e:
            ma_dd_logger.error(f"Revenue analysis failed: {str(e)}")
            return {
                "quality_score": 0.5,
                "growth_trend": "unknown",
                "concentration_risk": "medium",
            }

    async def _analyze_profitability(
        self, company: str, financial_data: dict
    ) -> dict[str, Any]:
        """Analyze profitability metrics and sustainability."""

        provider = get_layer_provider(AnalysisLayer.MA_DUE_DILIGENCE)
        if not provider:
            return {
                "ebitda_quality": 0.5,
                "margin_trend": "stable",
                "cost_efficiency": 0.5,
            }

        evidence_content = "\n\n".join(
            [e.content for e in financial_data.get("evidence", [])[:5]]
        )

        messages = [
            AIMessage(
                role="system",
                content="""You are analyzing profitability for M&A due diligence.
                Focus on:
                1. EBITDA quality and sustainability
                2. Margin trends and benchmarking
                3. Cost structure analysis
                4. Operating leverage assessment
                5. Profitability drivers and risks""",
            ),
            AIMessage(
                role="user",
                content=f"""Analyze profitability for {company}:

FINANCIAL DATA:
{evidence_content}

Provide:
1. EBITDA Quality Score (0.0-1.0): Sustainability of EBITDA generation
2. Margin Trend: improving/stable/declining
3. Cost Efficiency Score (0.0-1.0): Cost structure effectiveness
4. Key Margin Drivers
5. Profitability Risks""",
            ),
        ]

        try:
            response = await provider.generate_response_async(
                messages, max_tokens=1000, temperature=0.05
            )
            return self._parse_profitability_analysis(response.content)
        except Exception:
            return {
                "ebitda_quality": 0.5,
                "margin_trend": "stable",
                "cost_efficiency": 0.5,
            }

    async def _synthesize_overall_assessment(
        self, result: ComprehensiveDDResult
    ) -> ComprehensiveDDResult:
        """Synthesize overall DD assessment and recommendation."""

        provider = get_layer_provider(AnalysisLayer.OBSERVER)
        if not provider:
            result.investment_recommendation = "further_analysis_required"
            return result

        # Create synthesis prompt
        messages = [
            AIMessage(
                role="system",
                content="""You are an M&A investment committee member synthesizing due diligence findings.
                Provide clear investment recommendation: proceed/caution/stop
                Focus on deal breakers, value creation, and overall risk-return profile.""",
            ),
            AIMessage(
                role="user",
                content=f"""Synthesize due diligence findings for {result.target_company}:

FINANCIAL DD SUMMARY:
- Revenue Quality: {result.financial_dd.revenue_quality_score:.2f}
- EBITDA Quality: {result.financial_dd.ebitda_quality_score:.2f}
- Balance Sheet: {result.financial_dd.balance_sheet_strength:.2f}
- Critical Issues: {result.financial_dd.critical_financial_issues}

COMMERCIAL DD SUMMARY:
- Market Position: {result.commercial_dd.market_position_strength:.2f}
- Customer Base: {result.commercial_dd.customer_diversification:.2f}
- Growth Drivers: {result.commercial_dd.growth_drivers}
- Commercial Risks: {result.commercial_dd.commercial_risks}

OPERATIONAL DD SUMMARY:
- Management Quality: {result.operational_dd.management_quality:.2f}
- Operational Efficiency: {result.operational_dd.operational_efficiency:.2f}
- Integration Challenges: {result.operational_dd.integration_challenges}

Provide:
1. Overall Risk Rating: low/medium/high
2. Investment Recommendation: proceed/caution/stop
3. Key Deal Breakers (if any)
4. Value Creation Opportunities
5. Executive Summary (2-3 sentences)
6. Recommended Next Steps""",
            ),
        ]

        try:
            response = await provider.generate_response_async(
                messages, max_tokens=1500, temperature=0.03
            )
            self._parse_overall_assessment(response.content, result)
        except Exception as e:
            ma_dd_logger.error(f"Overall assessment synthesis failed: {str(e)}")
            result.investment_recommendation = "further_analysis_required"

        # Calculate overall confidence
        confidences = [
            result.financial_dd.analysis_confidence,
            result.commercial_dd.analysis_confidence,
            result.operational_dd.analysis_confidence,
        ]
        result.overall_confidence = sum(confidences) / len(confidences)

        return result

    def _parse_revenue_analysis(self, content: str) -> dict[str, Any]:
        """Parse revenue analysis from AI response."""

        # Extract revenue quality score
        quality_match = re.search(
            r"revenue quality score:?\s*([0-9.]+)", content.lower()
        )
        quality_score = float(quality_match.group(1)) if quality_match else 0.5

        # Extract growth trend
        trend = "stable"
        if "accelerating" in content.lower() or "growing" in content.lower():
            trend = "accelerating"
        elif "declining" in content.lower() or "decreasing" in content.lower():
            trend = "declining"

        # Extract concentration risk
        concentration = "medium"
        if "high concentration" in content.lower() or "concentrated" in content.lower():
            concentration = "high"
        elif "diversified" in content.lower() or "low concentration" in content.lower():
            concentration = "low"

        # Extract recurring revenue percentage
        recurring_match = re.search(
            r"recurring revenue:?\s*([0-9.]+)%", content.lower()
        )
        recurring_pct = (
            float(recurring_match.group(1)) / 100 if recurring_match else None
        )

        return {
            "quality_score": min(max(quality_score, 0.0), 1.0),
            "growth_trend": trend,
            "concentration_risk": concentration,
            "recurring_pct": recurring_pct,
        }

    def _parse_profitability_analysis(self, content: str) -> dict[str, Any]:
        """Parse profitability analysis from AI response."""

        # Extract EBITDA quality score
        ebitda_match = re.search(r"ebitda quality score:?\s*([0-9.]+)", content.lower())
        ebitda_quality = float(ebitda_match.group(1)) if ebitda_match else 0.5

        # Extract margin trend
        margin_trend = "stable"
        if "improving" in content.lower() or "increasing" in content.lower():
            margin_trend = "improving"
        elif "declining" in content.lower() or "pressured" in content.lower():
            margin_trend = "declining"

        # Extract cost efficiency
        efficiency_match = re.search(
            r"cost efficiency score:?\s*([0-9.]+)", content.lower()
        )
        cost_efficiency = float(efficiency_match.group(1)) if efficiency_match else 0.5

        return {
            "ebitda_quality": min(max(ebitda_quality, 0.0), 1.0),
            "margin_trend": margin_trend,
            "cost_efficiency": min(max(cost_efficiency, 0.0), 1.0),
        }

    def _parse_overall_assessment(
        self, content: str, result: ComprehensiveDDResult
    ) -> None:
        """Parse overall assessment from AI synthesis."""

        content_lower = content.lower()

        # Extract risk rating
        if "high risk" in content_lower:
            result.overall_risk_rating = "high"
        elif "low risk" in content_lower:
            result.overall_risk_rating = "low"
        else:
            result.overall_risk_rating = "medium"

        # Extract recommendation
        if "proceed" in content_lower and "stop" not in content_lower:
            result.investment_recommendation = "proceed"
        elif "stop" in content_lower or "avoid" in content_lower:
            result.investment_recommendation = "stop"
        else:
            result.investment_recommendation = "caution"

        # Extract executive summary (simplified)
        sentences = content.split(".")[:3]
        result.executive_summary = ". ".join(sentences).strip()

        # Extract deal breakers and opportunities (simplified)
        if "deal breaker" in content_lower:
            result.key_deal_breakers = ["Critical issues identified in analysis"]

        if "opportunity" in content_lower or "synergy" in content_lower:
            result.value_creation_opportunities = [
                "Value creation opportunities identified"
            ]

    # Placeholder methods for detailed analysis (to be implemented)
    async def _analyze_balance_sheet(self, company: str, data: dict) -> dict:
        """Placeholder for balance sheet analysis."""
        return {"strength_score": 0.6, "liquidity": "adequate", "debt_risk": "medium"}

    async def _analyze_cash_flows(self, company: str, data: dict) -> dict:
        """Placeholder for cash flow analysis."""
        return {"fcf_conversion": 0.8, "stability": "stable", "capex_intensity": 0.15}

    async def _analyze_market_dynamics(self, company: str) -> dict:
        """Placeholder for market dynamics analysis."""
        return {
            "market_assessment": "growing",
            "position_strength": 0.7,
            "differentiation": "moderate",
        }

    async def _analyze_customer_base(self, company: str) -> dict:
        """Placeholder for customer analysis."""
        return {
            "diversification_score": 0.6,
            "loyalty": "medium",
            "pricing_power": "limited",
        }

    async def _analyze_product_portfolio(self, company: str) -> dict:
        """Placeholder for product portfolio analysis."""
        return {
            "portfolio_strength": 0.65,
            "innovation": "moderate",
            "tech_moat": "limited",
        }

    async def _analyze_management_team(self, company: str) -> dict:
        """Placeholder for management analysis."""
        return {
            "quality_score": 0.7,
            "key_person_risk": "medium",
            "org_capability": "adequate",
        }

    async def _analyze_operational_efficiency(self, company: str) -> dict:
        """Placeholder for operational efficiency analysis."""
        return {
            "efficiency_score": 0.6,
            "tech_systems": "adequate",
            "process_maturity": "developing",
        }

    async def _analyze_supply_chain(self, company: str) -> dict:
        """Placeholder for supply chain analysis."""
        return {
            "resilience": "moderate",
            "vendor_risk": "medium",
            "scalability": "limited",
        }

    async def _analyze_human_capital(self, company: str) -> dict:
        """Placeholder for human capital analysis."""
        return {
            "retention_risk": "medium",
            "skill_gaps": ["Digital transformation"],
            "culture_risk": "medium",
        }

    async def _synthesize_financial_findings(
        self, result: FinancialDDResult, company: str
    ) -> FinancialDDResult:
        """Synthesize financial DD findings."""

        # Generate insights based on scores
        if result.revenue_quality_score > 0.7:
            result.financial_strengths.append("High-quality, sustainable revenue base")
        elif result.revenue_quality_score < 0.4:
            result.critical_financial_issues.append(
                "Revenue quality concerns identified"
            )

        if result.ebitda_quality_score > 0.7:
            result.financial_strengths.append("Strong profitability and margin profile")
        elif result.ebitda_quality_score < 0.4:
            result.critical_financial_issues.append(
                "Profitability sustainability concerns"
            )

        if result.balance_sheet_strength > 0.7:
            result.financial_strengths.append(
                "Strong balance sheet and financial position"
            )
        elif result.balance_sheet_strength < 0.4:
            result.critical_financial_issues.append("Balance sheet weakness identified")

        # Valuation considerations
        result.valuation_considerations = [
            "Revenue quality impact on multiple valuation",
            "Working capital adjustments required",
            "Debt refinancing considerations",
        ]

        return result

    async def _synthesize_commercial_findings(
        self, result: CommercialDDResult, company: str
    ) -> CommercialDDResult:
        """Synthesize commercial DD findings."""

        if result.market_position_strength > 0.7:
            result.growth_drivers.append("Strong market position enables growth")

        if result.customer_diversification < 0.4:
            result.commercial_risks.append("Customer concentration risk")

        result.market_opportunities = [
            "Market expansion potential",
            "Product portfolio enhancement",
            "Customer cross-selling opportunities",
        ]

        return result

    async def _synthesize_operational_findings(
        self, result: OperationalDDResult, company: str
    ) -> OperationalDDResult:
        """Synthesize operational DD findings."""

        if result.management_quality > 0.7:
            result.operational_strengths.append("Strong management team")
        elif result.management_quality < 0.4:
            result.integration_challenges.append("Management team concerns")

        if result.operational_efficiency > 0.7:
            result.operational_strengths.append("Efficient operations")
        else:
            result.improvement_opportunities.append(
                "Operational efficiency enhancement"
            )

        return result


# Convenience functions for workflow execution
async def run_financial_dd(target_company: str) -> FinancialDDResult:
    """Run Financial Due Diligence workflow."""
    workflow = MADueDiligenceWorkflow()
    return await workflow.execute_financial_dd(target_company)


async def run_comprehensive_dd(
    target_company: str, scope: str = "full"
) -> ComprehensiveDDResult:
    """Run comprehensive due diligence workflow."""
    workflow = MADueDiligenceWorkflow()
    return await workflow.execute_comprehensive_dd(target_company, scope)
