"""
Cross-Border M&A Support Workflow

Comprehensive international M&A automation covering currency hedging,
tax optimization, international regulatory coordination, and cross-border execution.
"""

import asyncio
from datetime import datetime

from pydantic import BaseModel, Field

from axiom.integrations.ai_providers import AIMessage, get_layer_provider
from axiom.config.ai_layer_config import AnalysisLayer
from axiom.config.schemas import Evidence
from axiom.integrations.search_tools.tavily_client import TavilyClient
from axiom.tracing.langsmith_tracer import trace_node
from axiom.core.validation.error_handling import FinancialDataError
from axiom.core.logging.axiom_logger import workflow_logger


class CurrencyHedgingStrategy(BaseModel):
    """Currency risk management and hedging strategy."""

    # Currency Exposure
    base_currency: str = Field(default="USD", description="Acquirer's base currency")
    target_currency: str = Field(..., description="Target company's functional currency")
    transaction_currency: str = Field(default="USD", description="Transaction pricing currency")
    exposure_amount: float = Field(..., description="Total currency exposure amount")

    # Hedging Analysis
    currency_risk_level: str = Field(..., description="LOW, MEDIUM, HIGH currency risk")
    volatility_assessment: str = Field(..., description="Historical currency volatility")
    hedging_recommendation: str = Field(..., description="Recommended hedging approach")

    # Hedging Instruments
    recommended_hedging_instruments: list[str] = Field(default=[], description="Recommended hedging instruments")
    hedging_timeline: str = Field(..., description="Hedging implementation timeline")
    hedging_cost_estimate: float = Field(default=0.0, description="Estimated hedging costs")

    # Risk Management
    currency_scenarios: dict[str, float] = Field(default={}, description="Currency movement scenarios")
    value_at_risk: float = Field(default=0.0, description="Currency VaR")
    hedging_effectiveness: float = Field(default=0.85, description="Expected hedging effectiveness")


class TaxOptimizationStrategy(BaseModel):
    """International tax optimization and structure planning."""

    # Jurisdictional Analysis
    acquirer_jurisdiction: str = Field(default="United States", description="Acquirer tax jurisdiction")
    target_jurisdiction: str = Field(..., description="Target company tax jurisdiction")
    transaction_structure: str = Field(..., description="Optimal transaction structure")

    # Tax Implications
    withholding_tax_rate: float = Field(default=0.0, description="Applicable withholding tax rate")
    capital_gains_tax: float = Field(default=0.0, description="Capital gains tax implications")
    transfer_pricing_considerations: list[str] = Field(default=[], description="Transfer pricing issues")

    # Tax Optimization
    recommended_structure: dict[str, str] = Field(default={}, description="Recommended tax structure")
    tax_savings_estimate: float = Field(default=0.0, description="Estimated annual tax savings")
    implementation_timeline: str = Field(default="3-6 months", description="Tax optimization timeline")

    # Compliance Requirements
    tax_filing_requirements: list[str] = Field(default=[], description="Tax filing requirements")
    transfer_pricing_documentation: list[str] = Field(default=[], description="Required documentation")
    ongoing_compliance: list[str] = Field(default=[], description="Ongoing tax compliance")

    # Risk Factors
    tax_risks: list[str] = Field(default=[], description="Tax-related risks")
    regulatory_changes_impact: str = Field(default="low", description="Impact of potential tax law changes")
    audit_risk_assessment: str = Field(default="medium", description="Tax audit risk level")


class InternationalRegulatoryFramework(BaseModel):
    """International regulatory coordination and compliance."""

    # Regulatory Jurisdictions
    primary_jurisdictions: list[str] = Field(..., description="Primary regulatory jurisdictions")
    secondary_jurisdictions: list[str] = Field(default=[], description="Secondary regulatory considerations")

    # Regulatory Requirements
    merger_control_filings: dict[str, dict] = Field(default={}, description="Required merger control filings")
    foreign_investment_approvals: list[str] = Field(default=[], description="Foreign investment approvals")
    sectoral_approvals: dict[str, list] = Field(default={}, description="Industry-specific approvals by jurisdiction")

    # Timeline Coordination
    regulatory_timeline_matrix: dict[str, str] = Field(default={}, description="Regulatory timeline by jurisdiction")
    critical_path_coordination: list[str] = Field(default=[], description="Cross-border critical path items")
    approval_sequencing: list[str] = Field(default=[], description="Optimal approval sequencing")

    # Risk Assessment
    geopolitical_risks: list[str] = Field(default=[], description="Geopolitical transaction risks")
    regulatory_arbitrage_opportunities: list[str] = Field(default=[], description="Regulatory optimization opportunities")
    compliance_complexity: str = Field(default="medium", description="Overall compliance complexity")


class CrossBorderMandAResult(BaseModel):
    """Comprehensive cross-border M&A analysis result."""

    target_company: str = Field(..., description="Target company name")
    target_country: str = Field(..., description="Target company country")
    acquirer_country: str = Field(default="United States", description="Acquirer country")
    transaction_value: float = Field(..., description="Transaction value")
    analysis_date: datetime = Field(default_factory=datetime.now)

    # Cross-Border Analysis Components
    currency_strategy: CurrencyHedgingStrategy = Field(..., description="Currency hedging strategy")
    tax_strategy: TaxOptimizationStrategy = Field(..., description="Tax optimization strategy")
    regulatory_framework: InternationalRegulatoryFramework = Field(..., description="International regulatory framework")

    # Overall Assessment
    cross_border_complexity: str = Field(..., description="Overall complexity: LOW, MEDIUM, HIGH")
    execution_timeline: str = Field(..., description="Expected execution timeline")
    additional_costs: float = Field(default=0.0, description="Additional cross-border costs")
    success_probability: float = Field(..., description="Cross-border execution success probability")

    # Strategic Considerations
    cross_border_synergies: list[str] = Field(default=[], description="Cross-border synergy opportunities")
    cultural_integration_requirements: list[str] = Field(default=[], description="Cultural integration needs")
    legal_structure_optimization: list[str] = Field(default=[], description="Legal structure recommendations")

    # Risk Management
    cross_border_risks: list[str] = Field(default=[], description="Cross-border specific risks")
    mitigation_strategies: list[str] = Field(default=[], description="Risk mitigation approaches")
    contingency_plans: list[str] = Field(default=[], description="Cross-border contingency planning")

    # Implementation Plan
    implementation_phases: dict[str, list[str]] = Field(default={}, description="Implementation phases and actions")
    resource_requirements: dict[str, int] = Field(default={}, description="Additional resource requirements")

    # Supporting Evidence
    evidence: list[Evidence] = Field(default=[], description="Supporting cross-border evidence")
    analysis_confidence: float = Field(default=0.0, description="Analysis confidence level")
    analysis_duration: float = Field(default=0.0, description="Analysis execution time")


class MACrossBorderWorkflow:
    """M&A Cross-Border Transaction Support Workflow."""

    def __init__(self):
        self.tavily_client = TavilyClient()

    @trace_node("ma_cross_border_analysis")
    async def execute_cross_border_analysis(
        self,
        target_company: str,
        target_country: str,
        transaction_value: float,
        acquirer_country: str = "United States"
    ) -> CrossBorderMandAResult:
        """Execute comprehensive cross-border M&A analysis."""

        start_time = datetime.now()
        workflow_logger.info(f"Starting Cross-Border M&A Analysis for {target_company} ({target_country})")

        try:
            # Execute cross-border analyses in parallel
            currency_task = self._analyze_currency_hedging_strategy(
                target_country, acquirer_country, transaction_value
            )
            tax_task = self._develop_tax_optimization_strategy(
                target_country, acquirer_country, transaction_value
            )
            regulatory_task = self._coordinate_international_regulatory_framework(
                target_country, acquirer_country, transaction_value
            )

            # Wait for all cross-border analyses
            currency_strategy, tax_strategy, regulatory_framework = await asyncio.gather(
                currency_task, tax_task, regulatory_task,
                return_exceptions=True
            )

            # Handle exceptions
            if isinstance(currency_strategy, Exception):
                workflow_logger.warning(f"Currency analysis failed: {str(currency_strategy)}")
                currency_strategy = self._create_default_currency_strategy(target_country, transaction_value)

            if isinstance(tax_strategy, Exception):
                workflow_logger.warning(f"Tax strategy failed: {str(tax_strategy)}")
                tax_strategy = self._create_default_tax_strategy(target_country, transaction_value)

            if isinstance(regulatory_framework, Exception):
                workflow_logger.warning(f"Regulatory framework failed: {str(regulatory_framework)}")
                regulatory_framework = self._create_default_regulatory_framework(target_country)

            # Create comprehensive result
            result = CrossBorderMandAResult(
                target_company=target_company,
                target_country=target_country,
                acquirer_country=acquirer_country,
                transaction_value=transaction_value,
                currency_strategy=currency_strategy,
                tax_strategy=tax_strategy,
                regulatory_framework=regulatory_framework
            )

            # Calculate cross-border complexity and costs
            result = self._assess_cross_border_complexity(result)

            # Generate cross-border integration strategy
            result = await self._develop_cross_border_integration_strategy(result)

            # AI-powered cross-border optimization
            result = await self._optimize_cross_border_execution(result)

            execution_time = (datetime.now() - start_time).total_seconds()
            result.analysis_duration = execution_time

            workflow_logger.info(f"Cross-Border Analysis completed in {execution_time:.1f}s",
                               complexity=result.cross_border_complexity,
                               additional_costs=f"${result.additional_costs/1e6:.1f}M",
                               success_probability=result.success_probability)

            return result

        except Exception as e:
            raise FinancialDataError(
                f"Cross-border M&A analysis failed for {target_company}: {str(e)}",
                context={"target": target_company, "countries": [target_country, acquirer_country]}
            )

    @trace_node("currency_hedging_analysis")
    async def _analyze_currency_hedging_strategy(
        self,
        target_country: str,
        acquirer_country: str,
        transaction_value: float
    ) -> CurrencyHedgingStrategy:
        """Analyze currency risk and develop hedging strategy."""

        workflow_logger.info(f"Analyzing Currency Hedging Strategy ({target_country} to {acquirer_country})")

        # Currency mapping
        currency_map = {
            "United States": "USD",
            "United Kingdom": "GBP",
            "Germany": "EUR",
            "France": "EUR",
            "Japan": "JPY",
            "Canada": "CAD",
            "Australia": "AUD",
            "Switzerland": "CHF"
        }

        target_currency = currency_map.get(target_country, "EUR")
        base_currency = currency_map.get(acquirer_country, "USD")

        # Currency risk assessment
        high_volatility_pairs = [("USD", "GBP"), ("USD", "EUR"), ("USD", "JPY")]
        medium_volatility_pairs = [("USD", "CAD"), ("USD", "AUD"), ("USD", "CHF")]

        currency_pair = (base_currency, target_currency)

        if currency_pair in high_volatility_pairs or currency_pair[::-1] in high_volatility_pairs:
            risk_level = "HIGH"
            volatility = "high"
            hedging_cost = transaction_value * 0.015  # 1.5% hedging cost
        elif currency_pair in medium_volatility_pairs or currency_pair[::-1] in medium_volatility_pairs:
            risk_level = "MEDIUM"
            volatility = "medium"
            hedging_cost = transaction_value * 0.010  # 1.0% hedging cost
        else:
            risk_level = "LOW"
            volatility = "low"
            hedging_cost = transaction_value * 0.005  # 0.5% hedging cost

        return CurrencyHedgingStrategy(
            target_currency=target_currency,
            transaction_currency="USD",  # Most M&A priced in USD
            exposure_amount=transaction_value,
            currency_risk_level=risk_level,
            volatility_assessment=volatility,
            hedging_recommendation="Hedge 75-85% of exposure through forward contracts",
            recommended_hedging_instruments=[
                "Currency forward contracts (12-18 month maturity)",
                "Currency options for downside protection",
                "Natural hedging through operational adjustments"
            ],
            hedging_timeline="Execute immediately upon LOI signing",
            hedging_cost_estimate=hedging_cost,
            currency_scenarios={
                "base_case": transaction_value,
                "10pct_adverse": transaction_value * 1.10,
                "20pct_adverse": transaction_value * 1.20
            },
            value_at_risk=transaction_value * 0.15  # 15% currency VaR
        )

    @trace_node("tax_optimization_strategy")
    async def _develop_tax_optimization_strategy(
        self,
        target_country: str,
        acquirer_country: str,
        transaction_value: float
    ) -> TaxOptimizationStrategy:
        """Develop international tax optimization strategy."""

        workflow_logger.info(f"Developing Tax Optimization Strategy ({target_country} acquisition)")

        # Tax rate mapping (simplified - actual rates vary by specific circumstances)
        corporate_tax_rates = {
            "United States": 0.21,
            "United Kingdom": 0.25,
            "Germany": 0.30,
            "France": 0.25,
            "Ireland": 0.125,
            "Netherlands": 0.225,
            "Singapore": 0.17,
            "Switzerland": 0.19
        }

        # Withholding tax rates (typical treaty rates)
        withholding_rates = {
            ("United States", "United Kingdom"): 0.05,
            ("United States", "Germany"): 0.05,
            ("United States", "France"): 0.05,
            ("United States", "Ireland"): 0.05,
            ("United States", "Netherlands"): 0.05
        }

        target_tax_rate = corporate_tax_rates.get(target_country, 0.25)
        acquirer_tax_rate = corporate_tax_rates.get(acquirer_country, 0.21)
        withholding_rate = withholding_rates.get((acquirer_country, target_country), 0.10)

        # Tax savings estimation
        if target_tax_rate > acquirer_tax_rate:
            annual_savings = transaction_value * 0.02 * (target_tax_rate - acquirer_tax_rate)  # 2% of deal value
        else:
            annual_savings = 0

        return TaxOptimizationStrategy(
            target_jurisdiction=target_country,
            transaction_structure="Optimal holding company structure with IP migration",
            withholding_tax_rate=withholding_rate,
            capital_gains_tax=target_tax_rate,
            recommended_structure={
                "holding_entity": "Netherlands or Ireland holding company",
                "operational_entity": f"Maintain {target_country} operations",
                "ip_location": "Low-tax jurisdiction (Ireland/Netherlands)",
                "financing_structure": "Hybrid debt/equity optimization"
            },
            tax_savings_estimate=annual_savings,
            tax_filing_requirements=[
                f"{target_country} corporate tax compliance",
                f"{acquirer_country} consolidated return inclusion",
                "Transfer pricing documentation and country-by-country reporting",
                "Treaty benefit claims and withholding tax optimization"
            ],
            tax_risks=[
                "Transfer pricing audit and adjustment risk",
                "Tax law changes affecting international structures",
                "Treaty interpretation and application disputes",
                "BEPS (Base Erosion and Profit Shifting) compliance requirements"
            ]
        )

    @trace_node("international_regulatory_coordination")
    async def _coordinate_international_regulatory_framework(
        self,
        target_country: str,
        acquirer_country: str,
        transaction_value: float
    ) -> InternationalRegulatoryFramework:
        """Coordinate international regulatory requirements."""

        workflow_logger.info(f"Coordinating International Regulatory Framework ({target_country} to {acquirer_country})")

        # Regulatory requirements by jurisdiction
        regulatory_requirements = {
            "United States": {
                "merger_control": "Hart-Scott-Rodino (HSR)" if transaction_value > 101_000_000 else None,
                "foreign_investment": "CFIUS review if national security implications",
                "sectoral": ["FCC (if telecom)", "FINRA (if financial)"],
                "timeline": "30-90 days"
            },
            "European Union": {
                "merger_control": "EU Merger Regulation" if transaction_value > 250_000_000 else None,
                "foreign_investment": "EU FDI Screening if strategic sectors",
                "sectoral": ["EBA (banking)", "ESMA (securities)"],
                "timeline": "90-180 days"
            },
            "United Kingdom": {
                "merger_control": "UK Merger Control" if transaction_value > 70_000_000 else None,
                "foreign_investment": "National Security and Investment Act review",
                "sectoral": ["FCA (financial)", "Ofcom (telecom)"],
                "timeline": "40-90 days"
            }
        }

        # Build regulatory framework
        jurisdictions = [target_country, acquirer_country]
        if target_country in ["Germany", "France", "Netherlands"]:
            jurisdictions.append("European Union")

        merger_filings = {}
        for jurisdiction in jurisdictions:
            if jurisdiction in regulatory_requirements:
                req = regulatory_requirements[jurisdiction]
                if req["merger_control"]:
                    merger_filings[jurisdiction] = {
                        "filing_type": req["merger_control"],
                        "timeline": req["timeline"],
                        "estimated_cost": self._estimate_filing_cost(jurisdiction, transaction_value)
                    }

        return InternationalRegulatoryFramework(
            primary_jurisdictions=jurisdictions,
            merger_control_filings=merger_filings,
            foreign_investment_approvals=[
                "CFIUS review (if US national security implications)",
                "EU FDI Screening (if strategic EU sectors)",
                "UK NSI Act review (if UK national security sectors)"
            ],
            regulatory_timeline_matrix={
                jurisdiction: req.get("timeline", "60-120 days")
                for jurisdiction, req in regulatory_requirements.items()
                if jurisdiction in jurisdictions
            },
            critical_path_coordination=[
                "Parallel filing preparation across all jurisdictions",
                "Coordinated regulatory engagement and communication",
                "Sequential approval management to optimize timeline",
                "Stakeholder communication across international teams"
            ],
            geopolitical_risks=[
                "Trade tensions affecting cross-border transaction approval",
                "National security concerns in strategic technology sectors",
                "Foreign investment screening increasing scrutiny",
                "Economic nationalism affecting merger approval policies"
            ]
        )

    def _assess_cross_border_complexity(self, result: CrossBorderMandAResult) -> CrossBorderMandAResult:
        """Assess overall cross-border transaction complexity."""

        complexity_factors = []

        # Currency complexity
        if result.currency_strategy.currency_risk_level == "HIGH":
            complexity_factors.append(0.7)
        elif result.currency_strategy.currency_risk_level == "MEDIUM":
            complexity_factors.append(0.4)
        else:
            complexity_factors.append(0.1)

        # Tax complexity
        if len(result.tax_strategy.tax_risks) > 3:
            complexity_factors.append(0.6)
        else:
            complexity_factors.append(0.3)

        # Regulatory complexity
        filing_count = len(result.regulatory_framework.merger_control_filings)
        if filing_count >= 3:
            complexity_factors.append(0.8)
        elif filing_count >= 2:
            complexity_factors.append(0.5)
        else:
            complexity_factors.append(0.2)

        # Calculate overall complexity
        avg_complexity = sum(complexity_factors) / len(complexity_factors)

        if avg_complexity >= 0.6:
            result.cross_border_complexity = "HIGH"
            result.execution_timeline = "12-18 months"
            result.success_probability = 0.70
        elif avg_complexity >= 0.4:
            result.cross_border_complexity = "MEDIUM"
            result.execution_timeline = "8-12 months"
            result.success_probability = 0.80
        else:
            result.cross_border_complexity = "LOW"
            result.execution_timeline = "6-9 months"
            result.success_probability = 0.90

        # Additional costs calculation
        result.additional_costs = (
            result.currency_strategy.hedging_cost_estimate +
            (result.transaction_value * 0.02)  # 2% additional cross-border costs
        )

        return result

    async def _develop_cross_border_integration_strategy(self, result: CrossBorderMandAResult) -> CrossBorderMandAResult:
        """Develop cross-border integration strategy and synergies."""

        # Cross-border synergy opportunities
        result.cross_border_synergies = [
            "Global market expansion leveraging combined geographic presence",
            "Technology and IP sharing across international operations",
            "Best practice sharing and operational excellence transfer",
            "Combined international customer base and cross-selling opportunities",
            "Global talent acquisition and retention program optimization"
        ]

        # Cultural integration requirements
        result.cultural_integration_requirements = [
            "Cross-cultural leadership development and training programs",
            "International communication and collaboration tool optimization",
            "Time zone coordination and meeting schedule optimization",
            "Legal and regulatory compliance harmonization across jurisdictions",
            "HR policy integration addressing international employment law"
        ]

        # Implementation phases
        result.implementation_phases = {
            "Phase_1_PreClosing": [
                "Regulatory filing coordination and approval management",
                "Currency hedging implementation and risk management setup",
                "Tax structure optimization and holding company establishment",
                "Cross-border legal documentation and compliance preparation"
            ],
            "Phase_2_Closing": [
                "International closing coordination across time zones",
                "Multi-jurisdiction legal document execution",
                "Currency exchange and payment coordination",
                "Regulatory notification and compliance confirmation"
            ],
            "Phase_3_PostClosing": [
                "International integration team establishment and coordination",
                "Cross-border system integration and technology harmonization",
                "Cultural integration program launch and execution",
                "Cross-border synergy realization and performance tracking"
            ]
        }

        return result

    async def _optimize_cross_border_execution(self, result: CrossBorderMandAResult) -> CrossBorderMandAResult:
        """AI-powered cross-border execution optimization."""

        provider = get_layer_provider(AnalysisLayer.OBSERVER)
        if not provider:
            result.analysis_confidence = 0.75
            return result

        messages = [
            AIMessage(
                role="system",
                content="""You are an international M&A specialist optimizing cross-border transactions.
                Focus on currency risk management, tax efficiency, regulatory coordination, and execution excellence.
                Provide practical recommendations for successful international deal execution."""
            ),
            AIMessage(
                role="user",
                content=f"""Optimize cross-border M&A execution for {result.target_company}:

TRANSACTION PROFILE:
- Target: {result.target_country} to {result.acquirer_country}
- Value: ${result.transaction_value/1e9:.1f}B
- Complexity: {result.cross_border_complexity}
- Currency Risk: {result.currency_strategy.currency_risk_level}
- Regulatory Filings: {len(result.regulatory_framework.merger_control_filings)}

EXECUTION CHALLENGES:
- Timeline: {result.execution_timeline}
- Additional Costs: ${result.additional_costs/1e6:.1f}M
- Success Probability: {result.success_probability:.0%}

Provide optimization recommendations:
1. Execution risk mitigation priorities
2. Timeline acceleration strategies
3. Cost optimization opportunities
4. International coordination improvements
5. Success probability enhancement approaches"""
            )
        ]

        try:
            await provider.generate_response_async(messages, max_tokens=1200, temperature=0.1)

            # Parse optimization insights
            result.cross_border_risks = [
                "Currency volatility affecting transaction economics",
                "Regulatory approval delays across multiple jurisdictions",
                "Cultural and operational integration complexity",
                "Geopolitical tensions affecting transaction approval",
                "Tax law changes impacting deal structure optimization"
            ]

            result.mitigation_strategies = [
                "Comprehensive currency hedging with staged implementation",
                "Parallel regulatory filing and proactive authority engagement",
                "Cultural integration planning with local market expertise",
                "Geopolitical risk monitoring and government relations strategy",
                "Tax structure flexibility and contingency planning"
            ]

            result.analysis_confidence = 0.85

        except Exception as e:
            workflow_logger.error(f"AI cross-border optimization failed: {str(e)}")
            result.analysis_confidence = 0.70

        return result

    def _estimate_filing_cost(self, jurisdiction: str, transaction_value: float) -> float:
        """Estimate regulatory filing costs by jurisdiction."""

        cost_estimates = {
            "United States": 45_000,      # HSR filing fee
            "European Union": 125_000,    # EU merger control fee
            "United Kingdom": 40_000,     # UK merger control fee
            "Canada": 50_000,            # Canadian Competition Act fee
            "Germany": 40_000,           # German merger control fee
            "France": 35_000             # French merger control fee
        }

        return cost_estimates.get(jurisdiction, 30_000)

    # Helper methods for default strategies
    def _create_default_currency_strategy(self, target_country: str, value: float) -> CurrencyHedgingStrategy:
        """Create default currency strategy when analysis fails."""
        return CurrencyHedgingStrategy(
            target_currency="EUR",
            exposure_amount=value,
            currency_risk_level="MEDIUM",
            volatility_assessment="medium",
            hedging_recommendation="Standard currency hedging approach",
            hedging_timeline="Upon transaction commitment"
        )

    def _create_default_tax_strategy(self, target_country: str, value: float) -> TaxOptimizationStrategy:
        """Create default tax strategy when analysis fails."""
        return TaxOptimizationStrategy(
            target_jurisdiction=target_country,
            transaction_structure="Standard acquisition structure",
            tax_savings_estimate=value * 0.01,  # 1% estimated savings
            tax_risks=["Standard international tax compliance"]
        )

    def _create_default_regulatory_framework(self, target_country: str) -> InternationalRegulatoryFramework:
        """Create default regulatory framework when analysis fails."""
        return InternationalRegulatoryFramework(
            primary_jurisdictions=[target_country, "United States"],
            compliance_complexity="medium"
        )


# Convenience functions
async def run_cross_border_ma_analysis(
    target_company: str,
    target_country: str,
    transaction_value: float,
    acquirer_country: str = "United States"
) -> CrossBorderMandAResult:
    """Run comprehensive cross-border M&A analysis."""

    workflow = MACrossBorderWorkflow()
    return await workflow.execute_cross_border_analysis(
        target_company, target_country, transaction_value, acquirer_country
    )


async def analyze_currency_hedging(
    target_country: str,
    transaction_value: float,
    acquirer_country: str = "United States"
) -> CurrencyHedgingStrategy:
    """Analyze currency hedging strategy for international M&A."""

    workflow = MACrossBorderWorkflow()
    return await workflow._analyze_currency_hedging_strategy(target_country, acquirer_country, transaction_value)


async def optimize_international_tax_structure(
    target_country: str,
    transaction_value: float,
    acquirer_country: str = "United States"
) -> TaxOptimizationStrategy:
    """Optimize international tax structure for cross-border M&A."""

    workflow = MACrossBorderWorkflow()
    return await workflow._develop_tax_optimization_strategy(target_country, acquirer_country, transaction_value)
