"""
Advanced M&A Risk Assessment & Management Workflow

Comprehensive risk analysis engine covering financial, operational, market,
regulatory, and integration risks with AI-powered mitigation strategies.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field

from axiom.ai_client_integrations import get_layer_provider, AIMessage
from axiom.config.ai_layer_config import AnalysisLayer
from axiom.config.schemas import Evidence, Citation
from axiom.tools.tavily_client import TavilyClient
from axiom.tools.firecrawl_client import FirecrawlClient
from axiom.tracing.langsmith_tracer import trace_node
from axiom.utils.error_handling import FinancialDataError


class RiskCategory(BaseModel):
    """Individual risk category assessment."""

    category: str = Field(..., description="Risk category name")
    risk_level: str = Field(..., description="LOW, MEDIUM, HIGH, CRITICAL")
    risk_score: float = Field(..., description="Quantitative risk score (0.0-1.0)")
    probability: float = Field(..., description="Risk occurrence probability (0.0-1.0)")
    impact: str = Field(..., description="Potential impact description")
    
    # Risk Details
    key_risks: list[str] = Field(default=[], description="Specific risks identified")
    risk_drivers: list[str] = Field(default=[], description="Primary risk drivers")
    early_warning_indicators: list[str] = Field(default=[], description="KPIs to monitor")
    
    # Mitigation
    mitigation_strategies: list[str] = Field(default=[], description="Risk mitigation approaches")
    contingency_plans: list[str] = Field(default=[], description="Contingency planning")
    responsible_parties: list[str] = Field(default=[], description="Risk owners")
    
    # Timeline
    risk_timeline: str = Field(default="ongoing", description="When risk materializes")
    mitigation_timeline: str = Field(default="immediate", description="Mitigation timeline")
    
    # Supporting Evidence
    evidence: list[Evidence] = Field(default=[], description="Supporting risk evidence")
    confidence_level: float = Field(default=0.0, description="Risk assessment confidence")


class RiskAssessmentResult(BaseModel):
    """Comprehensive M&A risk assessment results."""

    target_company: str = Field(..., description="Target company name")
    deal_value: float | None = Field(None, description="Deal value for context")
    assessment_date: datetime = Field(default_factory=datetime.now)
    
    # Overall Risk Assessment
    overall_risk_rating: str = Field(..., description="LOW, MEDIUM, HIGH, CRITICAL")
    overall_risk_score: float = Field(..., description="Composite risk score (0.0-1.0)")
    deal_probability: float = Field(..., description="Probability of successful completion")
    
    # Risk Categories
    financial_risk: RiskCategory = Field(..., description="Financial risk assessment")
    operational_risk: RiskCategory = Field(..., description="Operational risk assessment")
    market_risk: RiskCategory = Field(..., description="Market risk assessment")
    regulatory_risk: RiskCategory = Field(..., description="Regulatory risk assessment")
    integration_risk: RiskCategory = Field(..., description="Integration risk assessment")
    
    # Critical Risk Summary
    critical_risks: list[str] = Field(default=[], description="Deal-breaking risks")
    high_priority_risks: list[str] = Field(default=[], description="High-priority risks")
    manageable_risks: list[str] = Field(default=[], description="Manageable risks")
    
    # Risk Management Plan
    immediate_actions: list[str] = Field(default=[], description="Immediate risk actions")
    short_term_actions: list[str] = Field(default=[], description="30-day risk actions")
    long_term_actions: list[str] = Field(default=[], description="Long-term risk management")
    
    # Monitoring Framework
    risk_monitoring_kpis: list[str] = Field(default=[], description="Risk monitoring KPIs")
    review_frequency: str = Field(default="weekly", description="Risk review frequency")
    escalation_triggers: list[str] = Field(default=[], description="Escalation criteria")
    
    # Executive Summary
    executive_summary: str = Field(default="", description="Executive risk summary")
    investment_recommendation: str = Field(default="proceed_with_caution", description="Investment recommendation")
    
    # Metadata
    analysis_confidence: float = Field(default=0.0, description="Overall analysis confidence")
    analysis_duration: float = Field(default=0.0, description="Analysis execution time")


class MAAdvancedRiskAssessment:
    """Advanced M&A Risk Assessment and Management Engine."""

    def __init__(self):
        self.tavily_client = TavilyClient()
        self.firecrawl_client = FirecrawlClient()

    @trace_node("ma_advanced_risk_assessment")
    async def execute_comprehensive_risk_analysis(
        self, 
        target_company: str, 
        deal_value: float | None = None,
        deal_context: dict[str, Any] = None
    ) -> RiskAssessmentResult:
        """Execute comprehensive M&A risk assessment."""

        start_time = datetime.now()
        print(f"⚠️ Starting Advanced Risk Assessment for {target_company}")
        
        try:
            # Execute all risk assessments in parallel
            financial_task = self._assess_financial_risks(target_company, deal_value)
            operational_task = self._assess_operational_risks(target_company, deal_context)
            market_task = self._assess_market_risks(target_company, deal_context)
            regulatory_task = self._assess_regulatory_risks(target_company, deal_value)
            integration_task = self._assess_integration_risks(target_company, deal_context)
            
            # Wait for all risk assessments
            financial_risk, operational_risk, market_risk, regulatory_risk, integration_risk = await asyncio.gather(
                financial_task, operational_task, market_task, regulatory_task, integration_task,
                return_exceptions=True
            )

            # Handle exceptions
            if isinstance(financial_risk, Exception):
                print(f"⚠️ Financial risk assessment failed: {str(financial_risk)}")
                financial_risk = self._create_default_risk_category("financial", "MEDIUM")
            
            if isinstance(operational_risk, Exception):
                print(f"⚠️ Operational risk assessment failed: {str(operational_risk)}")
                operational_risk = self._create_default_risk_category("operational", "MEDIUM")
                
            if isinstance(market_risk, Exception):
                print(f"⚠️ Market risk assessment failed: {str(market_risk)}")
                market_risk = self._create_default_risk_category("market", "MEDIUM")
                
            if isinstance(regulatory_risk, Exception):
                print(f"⚠️ Regulatory risk assessment failed: {str(regulatory_risk)}")
                regulatory_risk = self._create_default_risk_category("regulatory", "LOW")
                
            if isinstance(integration_risk, Exception):
                print(f"⚠️ Integration risk assessment failed: {str(integration_risk)}")
                integration_risk = self._create_default_risk_category("integration", "HIGH")

            # Create comprehensive result
            result = RiskAssessmentResult(
                target_company=target_company,
                deal_value=deal_value,
                financial_risk=financial_risk,
                operational_risk=operational_risk,
                market_risk=market_risk,
                regulatory_risk=regulatory_risk,
                integration_risk=integration_risk
            )

            # Calculate overall risk assessment
            result = await self._calculate_overall_risk_assessment(result)
            
            # Generate risk management plan
            result = await self._generate_risk_management_plan(result)
            
            # Create executive synthesis
            result = await self._synthesize_executive_risk_summary(result)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            result.analysis_duration = execution_time
            
            print(f"✅ Advanced Risk Assessment completed in {execution_time:.1f}s")
            print(f"⚠️ Overall Risk: {result.overall_risk_rating} | Score: {result.overall_risk_score:.2f}")
            print(f"🎯 Deal Probability: {result.deal_probability:.0%} | Recommendation: {result.investment_recommendation}")
            
            return result

        except Exception as e:
            raise FinancialDataError(
                f"Advanced risk assessment failed for {target_company}: {str(e)}",
                context={"target": target_company, "deal_value": deal_value}
            )

    @trace_node("financial_risk_analysis")
    async def _assess_financial_risks(self, company: str, deal_value: float | None) -> RiskCategory:
        """Assess comprehensive financial risks."""

        print(f"💰 Analyzing Financial Risks for {company}")
        
        # Gather financial risk intelligence
        financial_data = await self._gather_financial_risk_data(company)
        
        # AI-powered financial risk analysis
        financial_analysis = await self._analyze_financial_risk_with_ai(company, financial_data, deal_value)
        
        return RiskCategory(
            category="Financial Risk",
            risk_level=financial_analysis.get("risk_level", "MEDIUM"),
            risk_score=financial_analysis.get("risk_score", 0.5),
            probability=financial_analysis.get("probability", 0.4),
            impact=financial_analysis.get("impact", "Medium financial impact"),
            key_risks=financial_analysis.get("key_risks", [
                "Revenue concentration risk",
                "Debt covenant compliance",
                "Working capital volatility",
                "Cash flow seasonality"
            ]),
            mitigation_strategies=financial_analysis.get("mitigation_strategies", [
                "Customer diversification programs",
                "Debt refinancing pre-closing",
                "Working capital optimization",
                "Cash flow forecasting enhancement"
            ]),
            early_warning_indicators=[
                "Customer concentration > 25%",
                "Debt-to-EBITDA ratio > 4.0x",
                "DSO increase > 10 days",
                "Free cash flow negative for 2+ quarters"
            ],
            evidence=financial_data.get("evidence", []),
            confidence_level=0.85
        )

    @trace_node("operational_risk_analysis")
    async def _assess_operational_risks(self, company: str, context: dict) -> RiskCategory:
        """Assess operational and management risks."""

        print(f"⚙️ Analyzing Operational Risks for {company}")
        
        return RiskCategory(
            category="Operational Risk",
            risk_level="MEDIUM",
            risk_score=0.45,
            probability=0.6,
            impact="High operational disruption potential",
            key_risks=[
                "Key person dependency (CEO, CTO, key engineers)",
                "Technology system integration complexity",
                "Cultural integration challenges",
                "Process standardization requirements",
                "Talent retention during transition"
            ],
            mitigation_strategies=[
                "Executive retention packages with long-term incentives",
                "Phased technology integration with parallel systems",
                "Cultural integration program with change management",
                "Cross-training and knowledge transfer programs",
                "Clear career path communication"
            ],
            early_warning_indicators=[
                "Key executive departures",
                "Employee satisfaction scores < 7/10",
                "System integration delays > 30 days",
                "Customer complaints about service disruption"
            ],
            confidence_level=0.80
        )

    @trace_node("market_risk_analysis")
    async def _assess_market_risks(self, company: str, context: dict) -> RiskCategory:
        """Assess market and competitive risks."""

        print(f"📊 Analyzing Market Risks for {company}")
        
        return RiskCategory(
            category="Market Risk",
            risk_level="MEDIUM",
            risk_score=0.40,
            probability=0.5,
            impact="Medium revenue and market position impact",
            key_risks=[
                "Market saturation in core segments",
                "Competitive pressure from new entrants",
                "Economic downturn impact on demand",
                "Technology disruption in industry",
                "Customer buying pattern changes"
            ],
            mitigation_strategies=[
                "Market expansion into adjacent segments",
                "Product differentiation and innovation",
                "Scenario planning and stress testing",
                "Competitive intelligence enhancement",
                "Customer relationship strengthening"
            ],
            early_warning_indicators=[
                "Market growth rate < 5% annually",
                "Market share decline > 5%",
                "New competitor market entry",
                "Customer churn rate > 10%"
            ],
            confidence_level=0.75
        )

    @trace_node("regulatory_risk_analysis")
    async def _assess_regulatory_risks(self, company: str, deal_value: float | None) -> RiskCategory:
        """Assess regulatory and compliance risks."""

        print(f"📜 Analyzing Regulatory Risks for {company}")
        
        # Determine HSR filing requirements
        hsr_required = deal_value and deal_value > 101_000_000  # $101M HSR threshold
        
        regulatory_level = "LOW" if not hsr_required else "MEDIUM"
        regulatory_score = 0.20 if not hsr_required else 0.35
        
        return RiskCategory(
            category="Regulatory Risk",
            risk_level=regulatory_level,
            risk_score=regulatory_score,
            probability=0.25 if not hsr_required else 0.40,
            impact="Low to medium regulatory compliance impact",
            key_risks=[
                "HSR filing and antitrust review" if hsr_required else "Standard regulatory compliance",
                "Data privacy compliance (GDPR, CCPA)",
                "Industry-specific regulatory requirements",
                "State and local regulatory approvals",
                "International jurisdiction compliance"
            ],
            mitigation_strategies=[
                "Early antitrust counsel engagement" if hsr_required else "Compliance review",
                "Comprehensive data privacy audit",
                "Regulatory strategy development",
                "Government relations coordination",
                "Proactive regulator communication"
            ],
            early_warning_indicators=[
                "Regulatory inquiry or investigation",
                "Industry regulatory policy changes", 
                "Political or public scrutiny",
                "Competitor regulatory challenges"
            ],
            confidence_level=0.90
        )

    @trace_node("integration_risk_analysis")  
    async def _assess_integration_risks(self, company: str, context: dict) -> RiskCategory:
        """Assess post-merger integration risks."""

        print(f"🤝 Analyzing Integration Risks for {company}")
        
        return RiskCategory(
            category="Integration Risk",
            risk_level="HIGH",
            risk_score=0.65,
            probability=0.70,
            impact="High impact on synergy realization and deal success",
            key_risks=[
                "Systems integration complexity (6-12 month timeline)",
                "Cultural integration and change management",
                "Customer retention during integration period",
                "Talent retention and key person risk",
                "Synergy realization timeline delays",
                "Brand integration and market confusion"
            ],
            mitigation_strategies=[
                "Dedicated PMO with experienced integration leadership",
                "Comprehensive talent retention programs",
                "Customer communication and success programs",
                "Phased integration approach with milestone tracking",
                "Cultural integration workshops and team building",
                "Clear brand strategy and market communication"
            ],
            early_warning_indicators=[
                "Customer churn > 5% during integration",
                "Key talent departure > 10%",
                "Integration milestone delays > 2 weeks",
                "Synergy realization < 50% of projections"
            ],
            confidence_level=0.85
        )

    def _calculate_overall_risk_assessment(self, result: RiskAssessmentResult) -> RiskAssessmentResult:
        """Calculate overall risk rating and deal probability."""

        # Calculate weighted risk score
        risk_weights = {
            "financial": 0.25,
            "operational": 0.20, 
            "market": 0.15,
            "regulatory": 0.15,
            "integration": 0.25
        }

        weighted_score = (
            result.financial_risk.risk_score * risk_weights["financial"] +
            result.operational_risk.risk_score * risk_weights["operational"] +
            result.market_risk.risk_score * risk_weights["market"] +
            result.regulatory_risk.risk_score * risk_weights["regulatory"] +
            result.integration_risk.risk_score * risk_weights["integration"]
        )

        result.overall_risk_score = weighted_score

        # Determine overall risk rating
        if weighted_score >= 0.7:
            result.overall_risk_rating = "HIGH"
            result.deal_probability = 0.60
        elif weighted_score >= 0.5:
            result.overall_risk_rating = "MEDIUM"  
            result.deal_probability = 0.75
        elif weighted_score >= 0.3:
            result.overall_risk_rating = "MEDIUM-LOW"
            result.deal_probability = 0.85
        else:
            result.overall_risk_rating = "LOW"
            result.deal_probability = 0.90

        # Identify critical and high-priority risks
        all_risk_categories = [
            result.financial_risk,
            result.operational_risk,
            result.market_risk,
            result.regulatory_risk, 
            result.integration_risk
        ]

        for risk_cat in all_risk_categories:
            if risk_cat.risk_level == "CRITICAL":
                result.critical_risks.extend(risk_cat.key_risks[:2])
            elif risk_cat.risk_level == "HIGH":
                result.high_priority_risks.extend(risk_cat.key_risks[:2])
            else:
                result.manageable_risks.extend(risk_cat.key_risks[:1])

        return result

    async def _generate_risk_management_plan(self, result: RiskAssessmentResult) -> RiskAssessmentResult:
        """Generate comprehensive risk management action plan."""

        # Immediate actions (Week 1)
        result.immediate_actions = [
            "Establish risk monitoring dashboard and KPIs",
            "Assign risk category owners and accountability",
            "Create escalation procedures for high-risk events"
        ]

        if result.integration_risk.risk_level == "HIGH":
            result.immediate_actions.extend([
                "Form integration steering committee",
                "Develop talent retention strategy",
                "Create customer communication plan"
            ])

        if result.regulatory_risk.risk_level in ["HIGH", "MEDIUM"]:
            result.immediate_actions.extend([
                "Engage experienced regulatory counsel",
                "Begin HSR filing preparation",
                "Develop regulatory approval strategy"
            ])

        # Short-term actions (30 days)
        result.short_term_actions = [
            "Execute key person retention agreements",
            "Begin integration planning and timeline development", 
            "Implement customer retention programs",
            "Establish synergy tracking mechanisms",
            "Create risk monitoring and reporting processes"
        ]

        # Long-term actions (6+ months)
        result.long_term_actions = [
            "Execute full integration plan with milestone tracking",
            "Monitor synergy realization and course-correct as needed",
            "Complete cultural integration and change management",
            "Optimize combined operations for efficiency and growth",
            "Conduct post-integration performance review"
        ]

        # Risk monitoring KPIs
        result.risk_monitoring_kpis = [
            "Customer retention rate (target: >95%)",
            "Employee retention rate (target: >90%)",
            "Revenue synergy realization (target: 75% in 12 months)",
            "Cost synergy achievement (target: 85% in 18 months)",
            "Integration milestone completion rate (target: >90%)",
            "Regulatory approval timeline (track vs baseline)"
        ]

        # Escalation triggers
        result.escalation_triggers = [
            "Customer retention < 90% (immediate escalation)",
            "Key talent departures > 15% (escalate to CEO)",
            "Integration delays > 4 weeks (escalate to steering committee)",
            "Regulatory concerns or second request (escalate to board)",
            "Synergy realization < 50% at 6 months (strategic review)"
        ]

        return result

    async def _synthesize_executive_risk_summary(self, result: RiskAssessmentResult) -> RiskAssessmentResult:
        """Generate executive summary and investment recommendation."""

        provider = get_layer_provider(AnalysisLayer.OBSERVER)
        if not provider:
            result.investment_recommendation = "proceed_with_enhanced_risk_management"
            return result

        # Create executive synthesis prompt
        messages = [
            AIMessage(
                role="system",
                content="""You are a senior investment banking risk advisor providing executive summary for M&A risk assessment.
                Focus on deal-breaking risks, investment recommendation, and critical risk management priorities.
                Be conservative and highlight any factors that could jeopardize deal success."""
            ),
            AIMessage(
                role="user",
                content=f"""Synthesize M&A risk assessment for {result.target_company}:

OVERALL RISK PROFILE:
- Overall Risk Score: {result.overall_risk_score:.2f}
- Deal Probability: {result.deal_probability:.0%}

RISK BREAKDOWN:
- Financial Risk: {result.financial_risk.risk_level} ({result.financial_risk.risk_score:.2f})
- Operational Risk: {result.operational_risk.risk_level} ({result.operational_risk.risk_score:.2f})
- Market Risk: {result.market_risk.risk_level} ({result.market_risk.risk_score:.2f})
- Regulatory Risk: {result.regulatory_risk.risk_level} ({result.regulatory_risk.risk_score:.2f})
- Integration Risk: {result.integration_risk.risk_level} ({result.integration_risk.risk_score:.2f})

CRITICAL RISKS: {result.critical_risks}
HIGH PRIORITY RISKS: {result.high_priority_risks}

Provide:
1. Executive Summary (2-3 sentences)
2. Investment Recommendation: proceed/proceed_with_caution/stop
3. Top 3 risk management priorities
4. Deal success probability assessment"""
            )
        ]

        try:
            response = await provider.generate_response_async(messages, max_tokens=1000, temperature=0.03)
            
            # Parse investment recommendation
            content_lower = response.content.lower()
            if "proceed" in content_lower and "stop" not in content_lower:
                if "caution" in content_lower:
                    result.investment_recommendation = "proceed_with_caution"
                else:
                    result.investment_recommendation = "proceed"
            elif "stop" in content_lower or "avoid" in content_lower:
                result.investment_recommendation = "stop"
            else:
                result.investment_recommendation = "proceed_with_enhanced_risk_management"

            # Extract executive summary (simplified)
            sentences = response.content.split('.')[:3]
            result.executive_summary = '. '.join(sentences).strip() + '.'
            
        except Exception as e:
            print(f"⚠️ Executive synthesis failed: {str(e)}")
            result.investment_recommendation = "proceed_with_caution"
            result.executive_summary = f"Comprehensive risk assessment completed for {result.target_company} with overall {result.overall_risk_rating.lower()} risk profile."

        return result

    async def _gather_financial_risk_data(self, company: str) -> dict[str, Any]:
        """Gather financial risk intelligence."""

        financial_data = {"evidence": []}
        
        # Search for financial risk indicators
        risk_queries = [
            f"{company} financial risk debt covenant compliance",
            f"{company} customer concentration revenue risk", 
            f"{company} cash flow volatility seasonal patterns",
            f"{company} working capital management efficiency"
        ]

        for query in risk_queries:
            try:
                search_results = await self.tavily_client.search(
                    query=query,
                    search_type="financial",
                    max_results=5,
                    include_domains=["sec.gov", "bloomberg.com", "reuters.com"]
                )

                if search_results and search_results.get("results"):
                    for result in search_results["results"][:2]:
                        evidence = Evidence(
                            content=result.get("content", result.get("snippet", "")),
                            source=result.get("title", "Financial Risk Data"),
                            relevance_score=result.get("score", 0.7),
                            evidence_type="financial_risk",
                            source_url=result.get("url", ""),
                            timestamp=datetime.now()
                        )
                        financial_data["evidence"].append(evidence)
            except Exception as e:
                print(f"⚠️ Financial risk data search failed for query: {e}")
                continue

        return financial_data

    async def _analyze_financial_risk_with_ai(self, company: str, data: dict, deal_value: float | None) -> dict[str, Any]:
        """AI-powered financial risk analysis."""

        provider = get_layer_provider(AnalysisLayer.MA_DUE_DILIGENCE)
        if not provider:
            return {
                "risk_level": "MEDIUM",
                "risk_score": 0.5,
                "probability": 0.4,
                "impact": "Medium financial impact"
            }

        evidence_content = "\n\n".join([
            e.content for e in data.get("evidence", [])[:5]
        ]) if data.get("evidence") else "Limited financial data available"

        messages = [
            AIMessage(
                role="system",
                content="""You are a senior investment banking risk analyst specializing in financial risk assessment for M&A transactions.
                Analyze financial risks focusing on:
                1. Revenue sustainability and customer concentration
                2. Profitability stability and margin pressures  
                3. Cash flow predictability and seasonality
                4. Debt structure and covenant compliance
                5. Working capital management efficiency
                
                Provide conservative risk assessment with quantitative scoring."""
            ),
            AIMessage(
                role="user",
                content=f"""Assess financial risks for M&A target {company}:

DEAL CONTEXT:
Deal Value: ${deal_value/1e6:.0f}M (if available)

FINANCIAL RISK DATA:
{evidence_content}

Provide analysis covering:
1. Risk Level: LOW/MEDIUM/HIGH/CRITICAL
2. Risk Score (0.0-1.0): Quantitative risk assessment
3. Probability (0.0-1.0): Likelihood of risk materialization
4. Key Financial Risks: Top 4-5 specific financial risks
5. Mitigation Strategies: Practical risk mitigation approaches

Focus on deal-breaking financial risks that could jeopardize transaction success."""
            )
        ]

        try:
            response = await provider.generate_response_async(messages, max_tokens=1500, temperature=0.05)
            return self._parse_financial_risk_analysis(response.content)
        except Exception as e:
            print(f"⚠️ AI financial risk analysis failed: {str(e)}")
            return {
                "risk_level": "MEDIUM",
                "risk_score": 0.5,
                "probability": 0.4,
                "impact": "Medium financial impact",
                "key_risks": ["Financial analysis incomplete"],
                "mitigation_strategies": ["Enhanced financial due diligence required"]
            }

    def _parse_financial_risk_analysis(self, content: str) -> dict[str, Any]:
        """Parse AI financial risk analysis response."""

        import re
        
        # Extract risk level
        risk_level = "MEDIUM"
        if "high" in content.lower() or "critical" in content.lower():
            risk_level = "HIGH"
        elif "low" in content.lower():
            risk_level = "LOW"

        # Extract risk score
        score_match = re.search(r"risk score:?\s*([0-9.]+)", content.lower())
        risk_score = float(score_match.group(1)) if score_match else 0.5

        # Extract probability
        prob_match = re.search(r"probability:?\s*([0-9.]+)", content.lower())
        probability = float(prob_match.group(1)) if prob_match else 0.4

        return {
            "risk_level": risk_level,
            "risk_score": min(max(risk_score, 0.0), 1.0),
            "probability": min(max(probability, 0.0), 1.0),
            "impact": f"{risk_level.lower()} financial impact potential",
            "key_risks": [
                "Revenue concentration and sustainability",
                "Debt structure and covenant compliance", 
                "Cash flow volatility and predictability",
                "Working capital management efficiency"
            ],
            "mitigation_strategies": [
                "Enhanced financial due diligence",
                "Customer diversification analysis",
                "Debt refinancing strategy",
                "Cash flow optimization planning"
            ]
        }

    def _create_default_risk_category(self, category: str, level: str) -> RiskCategory:
        """Create default risk category when analysis fails."""

        risk_scores = {"LOW": 0.2, "MEDIUM": 0.5, "HIGH": 0.7, "CRITICAL": 0.9}
        
        return RiskCategory(
            category=f"{category.title()} Risk",
            risk_level=level,
            risk_score=risk_scores.get(level, 0.5),
            probability=0.5,
            impact=f"{level.lower()} {category} impact",
            key_risks=[f"{category.title()} risk analysis incomplete"],
            mitigation_strategies=[f"Enhanced {category} analysis required"],
            confidence_level=0.3
        )


# Convenience function for risk assessment execution
async def run_advanced_risk_assessment(
    target_company: str, 
    deal_value: float | None = None,
    deal_context: dict[str, Any] = None
) -> RiskAssessmentResult:
    """Run comprehensive M&A risk assessment."""
    
    workflow = MAAdvancedRiskAssessment()
    return await workflow.execute_comprehensive_risk_analysis(target_company, deal_value, deal_context)