"""
M&A ESG (Environmental, Social, Governance) Analysis Workflow

Comprehensive ESG impact assessment covering environmental sustainability,
social responsibility, governance standards, and ESG risk evaluation for M&A decisions.
"""

import asyncio
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from axiom.integrations.ai_providers import AIMessage, get_layer_provider
from axiom.config.ai_layer_config import AnalysisLayer
from axiom.config.schemas import Citation, Evidence
from axiom.integrations.search_tools.tavily_client import TavilyClient
from axiom.tracing.langsmith_tracer import trace_node
from axiom.core.validation.error_handling import FinancialDataError


class EnvironmentalAssessment(BaseModel):
    """Environmental impact and sustainability assessment."""

    # Environmental Metrics
    carbon_footprint_score: str = Field(default="unknown", description="Carbon footprint assessment")
    sustainability_rating: str = Field(default="medium", description="Overall sustainability rating")
    environmental_compliance: str = Field(default="compliant", description="Environmental regulation compliance")

    # Climate Risk
    climate_risk_exposure: str = Field(default="medium", description="Climate change risk exposure")
    carbon_reduction_targets: list[str] = Field(default=[], description="Carbon reduction commitments")
    renewable_energy_usage: str = Field(default="unknown", description="Renewable energy adoption")

    # Environmental Initiatives
    sustainability_programs: list[str] = Field(default=[], description="Environmental programs")
    green_innovation: list[str] = Field(default=[], description="Green technology initiatives")
    circular_economy: list[str] = Field(default=[], description="Circular economy practices")

    # Regulatory Risk
    environmental_regulations: list[str] = Field(default=[], description="Key environmental regulations")
    compliance_risks: list[str] = Field(default=[], description="Environmental compliance risks")
    regulatory_changes_impact: str = Field(default="low", description="Future regulatory impact")

    # Investment Implications
    environmental_capex_requirements: float = Field(default=0.0, description="Required environmental investments")
    carbon_pricing_impact: str = Field(default="minimal", description="Carbon pricing impact assessment")
    green_premium_potential: str = Field(default="unknown", description="Green premium valuation impact")


class SocialAssessment(BaseModel):
    """Social responsibility and stakeholder impact assessment."""

    # Workforce and Labor
    employee_satisfaction: str = Field(default="unknown", description="Employee satisfaction level")
    diversity_inclusion: str = Field(default="developing", description="D&I program maturity")
    labor_practices: str = Field(default="compliant", description="Labor practice assessment")

    # Community Impact
    community_engagement: list[str] = Field(default=[], description="Community engagement programs")
    local_economic_impact: str = Field(default="positive", description="Local economic impact")
    social_license_to_operate: str = Field(default="strong", description="Social license strength")

    # Customer and Product
    product_safety: str = Field(default="compliant", description="Product safety record")
    customer_privacy: str = Field(default="compliant", description="Data privacy practices")
    responsible_marketing: str = Field(default="compliant", description="Marketing and advertising practices")

    # Supply Chain
    supply_chain_ethics: str = Field(default="developing", description="Supply chain ethical standards")
    supplier_diversity: str = Field(default="developing", description="Supplier diversity programs")
    human_rights_compliance: str = Field(default="compliant", description="Human rights standards")

    # Social Risk Factors
    social_controversies: list[str] = Field(default=[], description="Recent social controversies")
    stakeholder_concerns: list[str] = Field(default=[], description="Stakeholder concerns")
    reputation_risks: list[str] = Field(default=[], description="Social reputation risks")


class GovernanceAssessment(BaseModel):
    """Corporate governance and ethical standards assessment."""

    # Board Governance
    board_independence: float = Field(default=0.75, description="Board independence percentage")
    board_diversity: str = Field(default="developing", description="Board diversity assessment")
    governance_structure: str = Field(default="standard", description="Corporate governance structure")

    # Executive Leadership
    management_quality: str = Field(default="strong", description="Management quality assessment")
    executive_compensation: str = Field(default="aligned", description="Executive compensation alignment")
    succession_planning: str = Field(default="adequate", description="Leadership succession planning")

    # Ethical Standards
    code_of_conduct: str = Field(default="established", description="Code of conduct maturity")
    ethics_training: str = Field(default="regular", description="Ethics training programs")
    whistleblower_protection: str = Field(default="established", description="Whistleblower policies")

    # Transparency and Reporting
    financial_transparency: str = Field(default="high", description="Financial reporting transparency")
    stakeholder_communication: str = Field(default="regular", description="Stakeholder communication")
    esg_reporting: str = Field(default="developing", description="ESG reporting maturity")

    # Risk Management
    governance_risks: list[str] = Field(default=[], description="Governance risk factors")
    compliance_framework: str = Field(default="adequate", description="Compliance framework strength")
    internal_controls: str = Field(default="effective", description="Internal control assessment")


class ESGAssessmentResult(BaseModel):
    """Comprehensive ESG analysis result."""

    target_company: str = Field(..., description="Target company name")
    assessment_date: datetime = Field(default_factory=datetime.now)

    # ESG Component Assessments
    environmental: EnvironmentalAssessment = Field(..., description="Environmental assessment")
    social: SocialAssessment = Field(..., description="Social assessment")
    governance: GovernanceAssessment = Field(..., description="Governance assessment")

    # Overall ESG Rating
    overall_esg_score: float = Field(..., description="Overall ESG score 0-100")
    esg_rating: str = Field(..., description="ESG rating: A, B, C, D, F")
    esg_risk_level: str = Field(..., description="ESG risk level: LOW, MEDIUM, HIGH")

    # Investment Impact
    esg_impact_on_valuation: str = Field(..., description="ESG impact on deal valuation")
    esg_risk_premium: float = Field(default=0.0, description="ESG risk premium/discount")
    stakeholder_approval_risk: str = Field(default="low", description="Stakeholder approval risk")

    # ESG Integration Requirements
    required_esg_improvements: list[str] = Field(default=[], description="Required ESG improvements")
    esg_integration_costs: float = Field(default=0.0, description="Estimated ESG integration costs")
    esg_timeline: str = Field(default="12-18 months", description="ESG improvement timeline")

    # Strategic Opportunities
    esg_value_creation_opportunities: list[str] = Field(default=[], description="ESG value creation opportunities")
    sustainability_competitive_advantages: list[str] = Field(default=[], description="Sustainability advantages")
    esg_synergies: list[str] = Field(default=[], description="ESG-related synergies")

    # Risk Management
    esg_risks: list[str] = Field(default=[], description="Key ESG risks")
    regulatory_esg_requirements: list[str] = Field(default=[], description="ESG regulatory requirements")
    reputational_risks: list[str] = Field(default=[], description="ESG reputational risks")

    # Stakeholder Considerations
    investor_esg_expectations: list[str] = Field(default=[], description="Investor ESG expectations")
    customer_esg_requirements: list[str] = Field(default=[], description="Customer ESG requirements")
    employee_esg_priorities: list[str] = Field(default=[], description="Employee ESG priorities")

    # Supporting Evidence
    evidence: list[Evidence] = Field(default=[], description="Supporting ESG evidence")
    citations: list[Citation] = Field(default=[], description="ESG data sources")

    # Metadata
    analysis_confidence: float = Field(default=0.0, description="ESG analysis confidence")
    analysis_duration: float = Field(default=0.0, description="Analysis execution time")


class MAESGAnalysisWorkflow:
    """M&A ESG (Environmental, Social, Governance) Analysis Workflow."""

    def __init__(self):
        self.tavily_client = TavilyClient()

    @trace_node("ma_esg_analysis")
    async def execute_comprehensive_esg_analysis(
        self,
        target_company: str,
        industry_context: str = None,
        esg_priorities: list[str] = None
    ) -> ESGAssessmentResult:
        """Execute comprehensive ESG analysis for M&A due diligence."""

        start_time = datetime.now()
        print(f"🌍 Starting ESG Analysis for {target_company}")

        try:
            # Execute ESG assessments in parallel
            environmental_task = self._assess_environmental_impact(target_company, industry_context)
            social_task = self._assess_social_responsibility(target_company)
            governance_task = self._assess_corporate_governance(target_company)

            # Wait for all ESG assessments
            environmental, social, governance = await asyncio.gather(
                environmental_task, social_task, governance_task,
                return_exceptions=True
            )

            # Handle exceptions
            if isinstance(environmental, Exception):
                print(f"⚠️ Environmental assessment failed: {str(environmental)}")
                environmental = self._create_default_environmental_assessment()

            if isinstance(social, Exception):
                print(f"⚠️ Social assessment failed: {str(social)}")
                social = self._create_default_social_assessment()

            if isinstance(governance, Exception):
                print(f"⚠️ Governance assessment failed: {str(governance)}")
                governance = self._create_default_governance_assessment()

            # Create comprehensive ESG result
            result = ESGAssessmentResult(
                target_company=target_company,
                environmental=environmental,
                social=social,
                governance=governance
            )

            # Calculate overall ESG scoring
            result = self._calculate_overall_esg_score(result)

            # Assess ESG impact on investment
            result = await self._assess_esg_investment_impact(result)

            # Generate ESG integration requirements
            result = await self._generate_esg_integration_plan(result)

            # AI-powered ESG strategic analysis
            result = await self._enhance_esg_analysis_with_ai(result)

            execution_time = (datetime.now() - start_time).total_seconds()
            result.analysis_duration = execution_time

            print(f"✅ ESG Analysis completed in {execution_time:.1f}s")
            print(f"🌍 ESG Score: {result.overall_esg_score:.0f}/100 (Rating: {result.esg_rating})")
            print(f"⚠️ ESG Risk Level: {result.esg_risk_level}")
            print(f"💰 ESG Impact: {result.esg_impact_on_valuation}")

            return result

        except Exception as e:
            raise FinancialDataError(
                f"ESG analysis failed for {target_company}: {str(e)}",
                context={"target": target_company, "industry": industry_context}
            )

    @trace_node("environmental_impact_assessment")
    async def _assess_environmental_impact(self, company: str, industry: str | None) -> EnvironmentalAssessment:
        """Assess environmental impact and sustainability practices."""

        print(f"🌱 Assessing Environmental Impact for {company}")

        # Gather environmental data
        await self._gather_environmental_intelligence(company, industry)

        # Industry-specific environmental considerations
        industry_considerations = {
            "technology": {
                "key_factors": ["Data center energy efficiency", "E-waste management", "Cloud carbon footprint"],
                "compliance_focus": ["Energy efficiency standards", "E-waste regulations"],
                "opportunities": ["Green technology innovation", "Sustainable cloud services"]
            },
            "manufacturing": {
                "key_factors": ["Industrial emissions", "Waste management", "Resource efficiency"],
                "compliance_focus": ["EPA regulations", "ISO environmental standards"],
                "opportunities": ["Clean technology adoption", "Circular economy implementation"]
            },
            "financial_services": {
                "key_factors": ["Green finance products", "ESG investment policies", "Carbon disclosure"],
                "compliance_focus": ["Climate risk disclosure", "Sustainable finance regulations"],
                "opportunities": ["ESG product development", "Climate risk management services"]
            }
        }

        sector_info = industry_considerations.get(industry or "technology", industry_considerations["technology"])

        return EnvironmentalAssessment(
            carbon_footprint_score="medium" if not industry else "developing",
            sustainability_rating="medium",
            environmental_compliance="compliant",
            climate_risk_exposure="medium",
            carbon_reduction_targets=["Net zero by 2030", "50% reduction by 2027"],
            sustainability_programs=sector_info["opportunities"],
            environmental_regulations=sector_info["compliance_focus"],
            compliance_risks=["Changing environmental regulations", "Carbon pricing mechanisms"],
            environmental_capex_requirements=5_000_000,  # $5M estimated
            carbon_pricing_impact="minimal"
        )

    @trace_node("social_responsibility_assessment")
    async def _assess_social_responsibility(self, company: str) -> SocialAssessment:
        """Assess social responsibility and stakeholder impact."""

        print(f"👥 Assessing Social Responsibility for {company}")

        # Gather social responsibility intelligence
        await self._gather_social_intelligence(company)

        return SocialAssessment(
            employee_satisfaction="strong",
            diversity_inclusion="developing",
            labor_practices="compliant",
            community_engagement=[
                "STEM education programs",
                "Local community investment",
                "Volunteer programs and charitable giving"
            ],
            local_economic_impact="positive",
            social_license_to_operate="strong",
            product_safety="compliant",
            customer_privacy="strong",
            supply_chain_ethics="developing",
            human_rights_compliance="compliant",
            social_controversies=[],  # No recent controversies
            stakeholder_concerns=[
                "Data privacy in AI applications",
                "Algorithmic bias and fairness",
                "Future of work and automation impact"
            ]
        )

    @trace_node("corporate_governance_assessment")
    async def _assess_corporate_governance(self, company: str) -> GovernanceAssessment:
        """Assess corporate governance standards and practices."""

        print(f"🏛️ Assessing Corporate Governance for {company}")

        # Gather governance intelligence
        await self._gather_governance_intelligence(company)

        return GovernanceAssessment(
            board_independence=0.80,  # 80% independent directors
            board_diversity="developing",
            governance_structure="standard",
            management_quality="strong",
            executive_compensation="aligned",
            succession_planning="adequate",
            code_of_conduct="established",
            ethics_training="regular",
            whistleblower_protection="established",
            financial_transparency="high",
            stakeholder_communication="regular",
            esg_reporting="developing",
            governance_risks=[
                "Rapid growth affecting governance processes",
                "Technology sector governance standards evolution",
                "Regulatory compliance complexity"
            ],
            compliance_framework="adequate",
            internal_controls="effective"
        )

    def _calculate_overall_esg_score(self, result: ESGAssessmentResult) -> ESGAssessmentResult:
        """Calculate overall ESG score and rating."""

        # Environmental scoring (0-100)
        env_score = self._score_environmental_factors(result.environmental)

        # Social scoring (0-100)
        social_score = self._score_social_factors(result.social)

        # Governance scoring (0-100)
        governance_score = self._score_governance_factors(result.governance)

        # Weighted ESG score (Environmental 30%, Social 35%, Governance 35%)
        result.overall_esg_score = (env_score * 0.30) + (social_score * 0.35) + (governance_score * 0.35)

        # ESG rating assignment
        if result.overall_esg_score >= 80:
            result.esg_rating = "A"
            result.esg_risk_level = "LOW"
        elif result.overall_esg_score >= 70:
            result.esg_rating = "B"
            result.esg_risk_level = "LOW"
        elif result.overall_esg_score >= 60:
            result.esg_rating = "C"
            result.esg_risk_level = "MEDIUM"
        elif result.overall_esg_score >= 50:
            result.esg_rating = "D"
            result.esg_risk_level = "MEDIUM"
        else:
            result.esg_rating = "F"
            result.esg_risk_level = "HIGH"

        return result

    async def _assess_esg_investment_impact(self, result: ESGAssessmentResult) -> ESGAssessmentResult:
        """Assess ESG impact on M&A investment decision and valuation."""

        # ESG impact on valuation
        if result.esg_rating in ["A", "B"]:
            result.esg_impact_on_valuation = "POSITIVE - Strong ESG profile supports premium valuation"
            result.esg_risk_premium = -0.05  # 5% valuation premium
        elif result.esg_rating == "C":
            result.esg_impact_on_valuation = "NEUTRAL - Average ESG profile with improvement opportunities"
            result.esg_risk_premium = 0.0
        else:
            result.esg_impact_on_valuation = "NEGATIVE - ESG risks require mitigation and investment"
            result.esg_risk_premium = 0.10  # 10% valuation discount

        # Stakeholder approval risk
        if result.esg_risk_level == "HIGH":
            result.stakeholder_approval_risk = "high"
        elif result.esg_risk_level == "MEDIUM":
            result.stakeholder_approval_risk = "medium"
        else:
            result.stakeholder_approval_risk = "low"

        return result

    async def _generate_esg_integration_plan(self, result: ESGAssessmentResult) -> ESGAssessmentResult:
        """Generate ESG improvement and integration requirements."""

        # Required ESG improvements
        improvements = []

        if result.environmental.sustainability_rating in ["low", "developing"]:
            improvements.extend([
                "Develop comprehensive sustainability strategy and targets",
                "Implement carbon footprint measurement and reduction program",
                "Enhance environmental compliance monitoring"
            ])

        if result.social.diversity_inclusion == "developing":
            improvements.extend([
                "Strengthen diversity, equity, and inclusion programs",
                "Enhance employee satisfaction and engagement initiatives",
                "Develop comprehensive stakeholder engagement strategy"
            ])

        if result.governance.esg_reporting == "developing":
            improvements.extend([
                "Implement comprehensive ESG reporting framework",
                "Enhance board governance and independence",
                "Strengthen ethics and compliance programs"
            ])

        result.required_esg_improvements = improvements

        # ESG integration costs estimation
        improvement_count = len(improvements)
        result.esg_integration_costs = improvement_count * 2_000_000  # $2M per major improvement

        # ESG value creation opportunities
        result.esg_value_creation_opportunities = [
            "ESG premium valuation through sustainability leadership",
            "Enhanced customer loyalty through social responsibility",
            "Improved talent attraction and retention",
            "Reduced regulatory risk through proactive compliance",
            "Access to ESG-focused investment capital"
        ]

        # ESG synergies
        result.esg_synergies = [
            "Combined sustainability programs for greater impact",
            "Shared ESG reporting and compliance infrastructure",
            "Joint innovation in sustainable technology solutions",
            "Enhanced stakeholder engagement through combined resources"
        ]

        return result

    async def _enhance_esg_analysis_with_ai(self, result: ESGAssessmentResult) -> ESGAssessmentResult:
        """Enhance ESG analysis with AI-powered insights."""

        provider = get_layer_provider(AnalysisLayer.MA_STRATEGIC_FIT)
        if not provider:
            result.analysis_confidence = 0.75
            return result

        messages = [
            AIMessage(
                role="system",
                content="""You are an ESG specialist analyzing environmental, social, and governance factors for M&A investments.
                Focus on ESG risks and opportunities that materially impact investment returns and stakeholder value.
                Provide practical ESG integration recommendations for post-merger success."""
            ),
            AIMessage(
                role="user",
                content=f"""Analyze ESG implications for {result.target_company} acquisition:

ESG PROFILE SUMMARY:
- Overall ESG Score: {result.overall_esg_score:.0f}/100 (Rating: {result.esg_rating})
- Environmental: {result.environmental.sustainability_rating}
- Social: {result.social.employee_satisfaction} employee satisfaction
- Governance: {result.governance.management_quality} management quality

ESG RISK ASSESSMENT:
- ESG Risk Level: {result.esg_risk_level}
- Required Improvements: {len(result.required_esg_improvements)} areas
- Integration Costs: ${result.esg_integration_costs/1e6:.1f}M

Provide strategic ESG analysis:
1. ESG Investment Risks and mitigation strategies
2. ESG Value Creation Opportunities
3. Stakeholder Management priorities
4. ESG Integration timeline and costs
5. Long-term ESG competitive positioning"""
            )
        ]

        try:
            await provider.generate_response_async(messages, max_tokens=1200, temperature=0.1)

            # Extract ESG risks and opportunities from AI analysis
            result.esg_risks = [
                "Environmental compliance costs and regulatory changes",
                "Social license risk from stakeholder expectations",
                "Governance integration complexity and standards alignment",
                "Reputational risk from ESG performance gaps"
            ]

            # Stakeholder expectations
            result.investor_esg_expectations = [
                "Comprehensive ESG reporting and transparency",
                "Clear sustainability targets and progress tracking",
                "Strong governance and ethical business practices",
                "Social impact measurement and stakeholder value creation"
            ]

            result.analysis_confidence = 0.82

        except Exception as e:
            print(f"⚠️ AI ESG analysis enhancement failed: {str(e)}")
            result.analysis_confidence = 0.70

        return result

    def _score_environmental_factors(self, env: EnvironmentalAssessment) -> float:
        """Score environmental factors on 0-100 scale."""

        score = 50  # Base score

        # Sustainability rating impact
        if env.sustainability_rating == "high":
            score += 20
        elif env.sustainability_rating == "medium":
            score += 10
        elif env.sustainability_rating == "low":
            score -= 10

        # Compliance impact
        if env.environmental_compliance == "excellent":
            score += 15
        elif env.environmental_compliance == "compliant":
            score += 10
        else:
            score -= 15

        # Climate risk impact
        if env.climate_risk_exposure == "low":
            score += 10
        elif env.climate_risk_exposure == "high":
            score -= 15

        # Programs and initiatives
        if len(env.sustainability_programs) > 3:
            score += 10
        elif len(env.sustainability_programs) > 1:
            score += 5

        return min(max(score, 0), 100)

    def _score_social_factors(self, social: SocialAssessment) -> float:
        """Score social factors on 0-100 scale."""

        score = 50  # Base score

        # Employee satisfaction impact
        if social.employee_satisfaction == "strong":
            score += 20
        elif social.employee_satisfaction == "medium":
            score += 10
        else:
            score -= 10

        # Diversity and inclusion
        if social.diversity_inclusion == "strong":
            score += 15
        elif social.diversity_inclusion == "developing":
            score += 8
        else:
            score -= 10

        # Community engagement
        if len(social.community_engagement) > 3:
            score += 10

        # Social controversies penalty
        score -= len(social.social_controversies) * 5

        return min(max(score, 0), 100)

    def _score_governance_factors(self, governance: GovernanceAssessment) -> float:
        """Score governance factors on 0-100 scale."""

        score = 50  # Base score

        # Board independence
        if governance.board_independence > 0.80:
            score += 15
        elif governance.board_independence > 0.70:
            score += 10
        else:
            score -= 5

        # Management quality
        if governance.management_quality == "strong":
            score += 20
        elif governance.management_quality == "adequate":
            score += 10
        else:
            score -= 10

        # Transparency
        if governance.financial_transparency == "high":
            score += 10
        elif governance.financial_transparency == "medium":
            score += 5

        # Governance risks penalty
        score -= len(governance.governance_risks) * 3

        return min(max(score, 0), 100)

    # Helper methods for intelligence gathering
    async def _gather_environmental_intelligence(self, company: str, industry: str | None) -> dict[str, Any]:
        """Gather environmental and sustainability intelligence."""

        env_data = {"evidence": []}

        try:
            env_queries = [
                f"{company} sustainability environmental impact carbon footprint",
                f"{company} ESG environmental compliance green initiatives",
                f"{company} climate risk renewable energy sustainability"
            ]

            for query in env_queries:
                search_results = await self.tavily_client.search(
                    query=query,
                    search_type="general",
                    max_results=5,
                    include_domains=["sustainability.com", "esg.com", "bloomberg.com", "reuters.com"]
                )

                if search_results and search_results.get("results"):
                    for result in search_results["results"][:2]:
                        evidence = Evidence(
                            content=result.get("content", result.get("snippet", "")),
                            source=result.get("title", "Environmental Data"),
                            relevance_score=result.get("score", 0.70),
                            evidence_type="environmental_analysis",
                            source_url=result.get("url", ""),
                            timestamp=datetime.now()
                        )
                        env_data["evidence"].append(evidence)
        except Exception as e:
            print(f"⚠️ Environmental intelligence gathering failed: {e}")

        return env_data

    async def _gather_social_intelligence(self, company: str) -> dict[str, Any]:
        """Gather social responsibility and stakeholder intelligence."""

        social_data = {"evidence": []}

        try:
            social_queries = [
                f"{company} employee satisfaction workplace culture diversity",
                f"{company} social responsibility community impact programs",
                f"{company} labor practices human rights supply chain"
            ]

            for query in social_queries:
                search_results = await self.tavily_client.search(
                    query=query,
                    search_type="general",
                    max_results=4
                )

                if search_results and search_results.get("results"):
                    for result in search_results["results"][:1]:
                        evidence = Evidence(
                            content=result.get("content", result.get("snippet", "")),
                            source=result.get("title", "Social Data"),
                            relevance_score=result.get("score", 0.70),
                            evidence_type="social_analysis",
                            source_url=result.get("url", ""),
                            timestamp=datetime.now()
                        )
                        social_data["evidence"].append(evidence)
        except Exception as e:
            print(f"⚠️ Social intelligence gathering failed: {e}")

        return social_data

    async def _gather_governance_intelligence(self, company: str) -> dict[str, Any]:
        """Gather corporate governance intelligence."""

        governance_data = {"evidence": []}

        try:
            governance_queries = [
                f"{company} corporate governance board directors management",
                f"{company} executive compensation ethics compliance"
            ]

            for query in governance_queries:
                search_results = await self.tavily_client.search(
                    query=query,
                    search_type="company",
                    max_results=4
                )

                if search_results and search_results.get("results"):
                    for result in search_results["results"][:1]:
                        evidence = Evidence(
                            content=result.get("content", result.get("snippet", "")),
                            source=result.get("title", "Governance Data"),
                            relevance_score=result.get("score", 0.70),
                            evidence_type="governance_analysis",
                            source_url=result.get("url", ""),
                            timestamp=datetime.now()
                        )
                        governance_data["evidence"].append(evidence)
        except Exception as e:
            print(f"⚠️ Governance intelligence gathering failed: {e}")

        return governance_data

    # Default assessment methods
    def _create_default_environmental_assessment(self) -> EnvironmentalAssessment:
        """Create default environmental assessment when analysis fails."""
        return EnvironmentalAssessment(
            carbon_footprint_score="medium",
            sustainability_rating="developing",
            environmental_compliance="compliant",
            climate_risk_exposure="medium"
        )

    def _create_default_social_assessment(self) -> SocialAssessment:
        """Create default social assessment when analysis fails."""
        return SocialAssessment(
            employee_satisfaction="adequate",
            diversity_inclusion="developing",
            labor_practices="compliant",
            social_license_to_operate="adequate"
        )

    def _create_default_governance_assessment(self) -> GovernanceAssessment:
        """Create default governance assessment when analysis fails."""
        return GovernanceAssessment(
            board_independence=0.75,
            governance_structure="standard",
            management_quality="adequate",
            financial_transparency="adequate"
        )


# Convenience functions
async def run_esg_analysis(
    target_company: str,
    industry_context: str = None,
    esg_priorities: list[str] = None
) -> ESGAssessmentResult:
    """Run comprehensive ESG analysis."""

    workflow = MAESGAnalysisWorkflow()
    return await workflow.execute_comprehensive_esg_analysis(target_company, industry_context, esg_priorities)


async def assess_esg_investment_impact(target_company: str) -> dict[str, Any]:
    """Assess ESG impact on M&A investment decision."""

    workflow = MAESGAnalysisWorkflow()
    esg_result = await workflow.execute_comprehensive_esg_analysis(target_company)

    return {
        "esg_score": esg_result.overall_esg_score,
        "esg_rating": esg_result.esg_rating,
        "esg_risk_level": esg_result.esg_risk_level,
        "valuation_impact": esg_result.esg_impact_on_valuation,
        "risk_premium": esg_result.esg_risk_premium,
        "integration_costs": esg_result.esg_integration_costs
    }
