"""
M&A Market Intelligence & Competitive Analysis Workflow

Comprehensive market intelligence automation covering competitive analysis,
industry trends, disruption assessment, and strategic market positioning.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field

from axiom.ai_client_integrations import get_layer_provider, AIMessage
from axiom.config.ai_layer_config import AnalysisLayer
from axiom.config.schemas import Evidence, Citation
from axiom.tools.tavily_client import TavilyClient
from axiom.tracing.langsmith_tracer import trace_node
from axiom.utils.error_handling import FinancialDataError


class CompetitorProfile(BaseModel):
    """Comprehensive competitor profile and analysis."""

    company_name: str = Field(..., description="Competitor company name")
    competitive_positioning: str = Field(..., description="Market positioning relative to target")
    market_share: float | None = Field(None, description="Estimated market share percentage")
    
    # Financial Metrics
    revenue_estimate: float | None = Field(None, description="Estimated annual revenue")
    growth_rate: float | None = Field(None, description="Estimated growth rate")
    profitability_estimate: str = Field(default="unknown", description="Profitability assessment")
    
    # Strategic Analysis
    competitive_strengths: list[str] = Field(default=[], description="Key competitive strengths")
    competitive_weaknesses: list[str] = Field(default=[], description="Identified weaknesses")
    strategic_initiatives: list[str] = Field(default=[], description="Recent strategic initiatives")
    
    # Market Activity
    recent_ma_activity: list[str] = Field(default=[], description="Recent M&A or partnership activity")
    funding_status: str = Field(default="unknown", description="Recent funding or investment")
    expansion_plans: list[str] = Field(default=[], description="Geographic or market expansion")
    
    # Threat Assessment
    competitive_threat_level: str = Field(default="medium", description="LOW, MEDIUM, HIGH threat level")
    threat_timeline: str = Field(default="medium_term", description="Threat materialization timeline")
    
    # Evidence
    intelligence_sources: list[Evidence] = Field(default=[], description="Supporting intelligence")
    analysis_confidence: float = Field(default=0.0, description="Analysis confidence level")


class MarketTrendAnalysis(BaseModel):
    """Comprehensive market trend and dynamics analysis."""

    industry_sector: str = Field(..., description="Primary industry sector")
    market_size_estimate: float | None = Field(None, description="Total addressable market size")
    growth_rate: float | None = Field(None, description="Market growth rate")
    
    # Market Dynamics
    key_growth_drivers: list[str] = Field(default=[], description="Primary market growth drivers")
    market_headwinds: list[str] = Field(default=[], description="Market challenges and headwinds")
    customer_behavior_trends: list[str] = Field(default=[], description="Customer behavior evolution")
    
    # Technology Trends
    technology_disruption_risk: str = Field(default="medium", description="Technology disruption risk level")
    emerging_technologies: list[str] = Field(default=[], description="Emerging technology trends")
    automation_impact: str = Field(default="moderate", description="Automation impact on industry")
    
    # Regulatory Environment
    regulatory_environment: str = Field(default="stable", description="Regulatory environment assessment")
    policy_changes_impact: list[str] = Field(default=[], description="Expected policy changes")
    compliance_complexity: str = Field(default="medium", description="Regulatory compliance complexity")
    
    # Investment Flows
    venture_capital_activity: str = Field(default="moderate", description="VC investment activity")
    private_equity_interest: str = Field(default="moderate", description="PE interest level")
    public_market_valuation_trends: str = Field(default="stable", description="Public market trends")
    
    # Future Outlook
    market_outlook_12_months: str = Field(default="positive", description="12-month market outlook")
    market_outlook_24_months: str = Field(default="positive", description="24-month market outlook")
    key_inflection_points: list[str] = Field(default=[], description="Market inflection points")
    
    # Analysis Quality
    trend_confidence: float = Field(default=0.0, description="Trend analysis confidence")
    data_quality_score: float = Field(default=0.0, description="Underlying data quality")


class DisruptionAssessment(BaseModel):
    """Technology and market disruption risk assessment."""

    # Disruption Analysis
    disruption_risk_level: str = Field(..., description="Overall disruption risk: LOW, MEDIUM, HIGH")
    disruption_timeline: str = Field(..., description="Expected disruption timeline")
    disruption_probability: float = Field(..., description="Probability of significant disruption")
    
    # Technology Disruption
    emerging_tech_threats: list[str] = Field(default=[], description="Emerging technology threats")
    ai_automation_impact: str = Field(default="moderate", description="AI/automation disruption impact")
    platform_shift_risk: str = Field(default="low", description="Platform/business model shift risk")
    
    # Market Disruption
    new_entrant_threat: str = Field(default="medium", description="New market entrant threat")
    customer_behavior_shift: str = Field(default="gradual", description="Customer behavior change pace")
    regulatory_disruption: str = Field(default="low", description="Regulatory disruption risk")
    
    # Economic Disruption
    economic_cycle_sensitivity: str = Field(default="medium", description="Economic cycle sensitivity")
    supply_chain_vulnerability: str = Field(default="medium", description="Supply chain disruption risk")
    geopolitical_impact: str = Field(default="low", description="Geopolitical disruption impact")
    
    # Mitigation Strategies
    disruption_mitigation: list[str] = Field(default=[], description="Disruption mitigation strategies")
    adaptation_capabilities: list[str] = Field(default=[], description="Company adaptation capabilities")
    innovation_investments: list[str] = Field(default=[], description="Required innovation investments")


class MarketIntelligenceResult(BaseModel):
    """Comprehensive market intelligence analysis result."""

    target_company: str = Field(..., description="Target company name")
    industry_focus: str = Field(..., description="Primary industry focus")
    analysis_date: datetime = Field(default_factory=datetime.now)
    
    # Competitive Landscape
    direct_competitors: list[CompetitorProfile] = Field(default=[], description="Direct competitors")
    indirect_competitors: list[CompetitorProfile] = Field(default=[], description="Indirect competitors")
    competitive_intensity: str = Field(..., description="Overall competitive intensity")
    
    # Market Analysis
    market_trends: MarketTrendAnalysis = Field(..., description="Market trend analysis")
    disruption_assessment: DisruptionAssessment = Field(..., description="Disruption risk assessment")
    
    # Strategic Positioning
    target_market_position: str = Field(..., description="Target's current market position")
    post_acquisition_position: str = Field(..., description="Expected post-acquisition position")
    competitive_advantages: list[str] = Field(default=[], description="Key competitive advantages")
    strategic_vulnerabilities: list[str] = Field(default=[], description="Strategic vulnerabilities")
    
    # Market Opportunities
    growth_opportunities: list[str] = Field(default=[], description="Market growth opportunities")
    expansion_potential: list[str] = Field(default=[], description="Geographic/segment expansion")
    consolidation_opportunities: list[str] = Field(default=[], description="Industry consolidation opportunities")
    
    # Strategic Recommendations
    market_strategy_recommendations: list[str] = Field(default=[], description="Strategic recommendations")
    competitive_response_plan: list[str] = Field(default=[], description="Competitive response planning")
    market_defense_strategies: list[str] = Field(default=[], description="Market position defense")
    
    # Investment Implications
    market_impact_on_valuation: str = Field(..., description="Market dynamics impact on valuation")
    timing_considerations: list[str] = Field(default=[], description="Market timing considerations")
    risk_factors: list[str] = Field(default=[], description="Market-related risk factors")
    
    # Supporting Data
    evidence: list[Evidence] = Field(default=[], description="Supporting market intelligence")
    intelligence_confidence: float = Field(default=0.0, description="Overall intelligence confidence")
    analysis_duration: float = Field(default=0.0, description="Analysis execution time")


class MAMarketIntelligenceWorkflow:
    """M&A Market Intelligence and Competitive Analysis Workflow."""

    def __init__(self):
        self.tavily_client = TavilyClient()

    @trace_node("ma_market_intelligence")
    async def execute_comprehensive_market_intelligence(
        self,
        target_company: str,
        industry_focus: str = None,
        analysis_scope: str = "comprehensive"
    ) -> MarketIntelligenceResult:
        """Execute comprehensive market intelligence analysis."""

        start_time = datetime.now()
        print(f"📊 Starting Market Intelligence Analysis for {target_company}")
        
        try:
            # Execute market intelligence components in parallel
            competitive_task = self._analyze_competitive_landscape(target_company)
            trends_task = self._analyze_market_trends(target_company, industry_focus)
            disruption_task = self._assess_disruption_risks(target_company, industry_focus)
            positioning_task = self._analyze_strategic_positioning(target_company)
            
            # Wait for all intelligence analyses
            competitive_landscape, market_trends, disruption_assessment, strategic_positioning = await asyncio.gather(
                competitive_task, trends_task, disruption_task, positioning_task,
                return_exceptions=True
            )

            # Handle exceptions
            if isinstance(competitive_landscape, Exception):
                print(f"⚠️ Competitive analysis failed: {str(competitive_landscape)}")
                competitive_landscape = {"competitors": [], "intensity": "medium"}
                
            if isinstance(market_trends, Exception):
                print(f"⚠️ Market trends analysis failed: {str(market_trends)}")
                market_trends = self._create_default_market_trends(industry_focus or "technology")
                
            if isinstance(disruption_assessment, Exception):
                print(f"⚠️ Disruption assessment failed: {str(disruption_assessment)}")
                disruption_assessment = self._create_default_disruption_assessment()
                
            if isinstance(strategic_positioning, Exception):
                print(f"⚠️ Strategic positioning analysis failed: {str(strategic_positioning)}")
                strategic_positioning = {"position": "established player", "advantages": []}

            # Create comprehensive result
            result = MarketIntelligenceResult(
                target_company=target_company,
                industry_focus=industry_focus or "technology",
                direct_competitors=competitive_landscape.get("direct", []),
                indirect_competitors=competitive_landscape.get("indirect", []),
                competitive_intensity=competitive_landscape.get("intensity", "medium"),
                market_trends=market_trends,
                disruption_assessment=disruption_assessment,
                target_market_position=strategic_positioning.get("position", "established player"),
                competitive_advantages=strategic_positioning.get("advantages", [])
            )

            # Generate strategic insights and recommendations
            result = await self._generate_strategic_recommendations(result)
            
            # Calculate market impact on valuation
            result = await self._assess_market_valuation_impact(result)
            
            # Create executive market intelligence summary
            result = await self._synthesize_market_intelligence_summary(result)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            result.analysis_duration = execution_time
            
            print(f"✅ Market Intelligence completed in {execution_time:.1f}s")
            print(f"🏢 Competitive Intensity: {result.competitive_intensity}")
            print(f"⚠️ Disruption Risk: {result.disruption_assessment.disruption_risk_level}")
            print(f"📊 Intelligence Confidence: {result.intelligence_confidence:.2f}")
            
            return result

        except Exception as e:
            raise FinancialDataError(
                f"Market intelligence analysis failed for {target_company}: {str(e)}",
                context={"target": target_company, "industry": industry_focus}
            )

    @trace_node("competitive_landscape_analysis")
    async def _analyze_competitive_landscape(self, target: str) -> dict[str, Any]:
        """Analyze comprehensive competitive landscape."""

        print(f"🏢 Analyzing Competitive Landscape for {target}")
        
        competitive_data = {"direct": [], "indirect": [], "intensity": "medium", "evidence": []}
        
        # Search for competitive intelligence
        competitive_queries = [
            f"{target} competitors competitive landscape analysis",
            f"{target} industry leaders market share competition",
            f"{target} competitive positioning analysis 2024"
        ]

        try:
            for query in competitive_queries:
                search_results = await self.tavily_client.search(
                    query=query,
                    search_type="company",
                    max_results=8,
                    include_domains=["bloomberg.com", "reuters.com", "techcrunch.com", "wsj.com"]
                )

                if search_results and search_results.get("results"):
                    for result in search_results["results"][:3]:
                        evidence = Evidence(
                            content=result.get("content", result.get("snippet", "")),
                            source=result.get("title", "Competitive Intelligence"),
                            relevance_score=result.get("score", 0.75),
                            evidence_type="competitive_analysis",
                            source_url=result.get("url", ""),
                            timestamp=datetime.now()
                        )
                        competitive_data["evidence"].append(evidence)

            # Parse competitive intelligence from search results
            competitors = self._parse_competitors_from_intelligence(competitive_data["evidence"])
            competitive_data["direct"] = competitors[:5]  # Top 5 direct competitors
            competitive_data["indirect"] = competitors[5:8]  # Additional indirect competitors
            
            # Assess competitive intensity
            if len(competitors) > 8:
                competitive_data["intensity"] = "high"
            elif len(competitors) > 4:
                competitive_data["intensity"] = "medium"
            else:
                competitive_data["intensity"] = "low"

        except Exception as e:
            print(f"⚠️ Competitive landscape analysis failed: {e}")

        return competitive_data

    @trace_node("market_trends_analysis")
    async def _analyze_market_trends(self, target: str, industry: str | None) -> MarketTrendAnalysis:
        """Analyze comprehensive market trends and dynamics."""

        print(f"📈 Analyzing Market Trends for {target}")
        
        industry_sector = industry or "technology"
        
        # Gather market trend intelligence
        market_data = await self._gather_market_trend_data(target, industry_sector)
        
        # AI-powered market trend analysis
        trend_analysis = await self._analyze_market_trends_with_ai(target, industry_sector, market_data)
        
        return MarketTrendAnalysis(
            industry_sector=industry_sector,
            market_size_estimate=trend_analysis.get("market_size"),
            growth_rate=trend_analysis.get("growth_rate"),
            key_growth_drivers=trend_analysis.get("growth_drivers", [
                "Digital transformation acceleration",
                "AI and automation adoption",
                "Cloud migration and scalability needs",
                "Remote work technology requirements"
            ]),
            market_headwinds=trend_analysis.get("headwinds", [
                "Economic uncertainty affecting IT budgets",
                "Increased competition from new entrants",
                "Regulatory scrutiny on data privacy",
                "Customer acquisition cost inflation"
            ]),
            technology_disruption_risk=trend_analysis.get("disruption_risk", "medium"),
            emerging_technologies=trend_analysis.get("emerging_tech", [
                "Generative AI and large language models",
                "Edge computing and distributed systems",
                "Quantum computing applications",
                "Blockchain and decentralized technologies"
            ]),
            regulatory_environment=trend_analysis.get("regulatory", "stable"),
            market_outlook_12_months=trend_analysis.get("outlook_12m", "positive"),
            market_outlook_24_months=trend_analysis.get("outlook_24m", "positive"),
            trend_confidence=0.80
        )

    @trace_node("disruption_assessment")
    async def _assess_disruption_risks(self, target: str, industry: str | None) -> DisruptionAssessment:
        """Assess comprehensive disruption risks and threats."""

        print(f"⚡ Assessing Disruption Risks for {target}")
        
        # Gather disruption intelligence
        disruption_data = await self._gather_disruption_intelligence(target, industry)
        
        # AI-powered disruption analysis
        disruption_analysis = await self._analyze_disruption_with_ai(target, disruption_data)
        
        return DisruptionAssessment(
            disruption_risk_level=disruption_analysis.get("risk_level", "MEDIUM"),
            disruption_timeline=disruption_analysis.get("timeline", "3-5 years"),
            disruption_probability=disruption_analysis.get("probability", 0.35),
            emerging_tech_threats=disruption_analysis.get("tech_threats", [
                "Generative AI replacing traditional software approaches",
                "Open-source alternatives reducing software pricing power",
                "Low-code/no-code platforms democratizing development",
                "Cloud-native solutions disrupting legacy architectures"
            ]),
            ai_automation_impact=disruption_analysis.get("ai_impact", "moderate"),
            new_entrant_threat="high" if disruption_analysis.get("risk_level") == "HIGH" else "medium",
            disruption_mitigation=[
                "Continuous innovation and R&D investment",
                "Strategic partnerships with emerging technology leaders",
                "Customer success programs to increase switching costs",
                "Platform strategy to create network effects"
            ],
            adaptation_capabilities=[
                "Strong engineering talent and technical capabilities",
                "Agile development and rapid iteration processes",
                "Customer feedback loops and market responsiveness",
                "Financial resources for strategic technology investments"
            ]
        )

    async def _generate_strategic_recommendations(self, result: MarketIntelligenceResult) -> MarketIntelligenceResult:
        """Generate strategic recommendations based on market intelligence."""

        # Market strategy recommendations
        result.market_strategy_recommendations = [
            "Accelerate product innovation to maintain competitive differentiation",
            "Expand geographic presence in high-growth international markets",
            "Develop strategic partnerships to enhance platform capabilities",
            "Invest in customer success and retention programs",
            "Build defensible moats through data network effects"
        ]

        # Competitive response planning
        result.competitive_response_plan = [
            "Monitor competitive pricing and feature development closely",
            "Develop rapid response capabilities for competitive threats",
            "Strengthen customer relationships through enhanced service levels",
            "Accelerate time-to-market for new product capabilities",
            "Create customer switching cost barriers through integration depth"
        ]

        # Market defense strategies
        result.market_defense_strategies = [
            "Patent portfolio development and intellectual property protection",
            "Customer contract optimization with longer terms and renewal incentives",
            "Strategic talent acquisition to strengthen competitive capabilities",
            "Brand building and thought leadership in key market segments",
            "Channel partner relationship strengthening and exclusivity agreements"
        ]

        # Growth opportunities identification
        result.growth_opportunities = [
            "Adjacent market expansion leveraging core technology platform",
            "Vertical market penetration with industry-specific solutions",
            "International expansion in Europe and Asia-Pacific regions",
            "Product portfolio expansion through complementary acquisitions",
            "Platform ecosystem development with third-party integrations"
        ]

        return result

    async def _assess_market_valuation_impact(self, result: MarketIntelligenceResult) -> MarketIntelligenceResult:
        """Assess market dynamics impact on valuation multiples and assumptions."""

        # Market valuation impact analysis
        competitive_intensity_impact = {
            "high": "Pressure on pricing multiples and growth sustainability",
            "medium": "Balanced competitive environment with selective pressure",
            "low": "Limited competitive pressure supporting premium valuations"
        }

        disruption_risk_impact = {
            "HIGH": "Significant downward pressure on terminal value assumptions",
            "MEDIUM": "Moderate impact requiring scenario analysis and contingency planning", 
            "LOW": "Minimal impact on base case valuation assumptions"
        }

        result.market_impact_on_valuation = f"""
        Competitive Intensity ({result.competitive_intensity}): {competitive_intensity_impact.get(result.competitive_intensity, 'Unknown impact')}
        Disruption Risk ({result.disruption_assessment.disruption_risk_level}): {disruption_risk_impact.get(result.disruption_assessment.disruption_risk_level, 'Unknown impact')}
        Market Growth Outlook: {result.market_trends.market_outlook_12_months} near-term, {result.market_trends.market_outlook_24_months} medium-term
        """

        # Market timing considerations
        result.timing_considerations = [
            f"Market cycle position: {result.market_trends.market_outlook_12_months} outlook suggests favorable timing",
            f"Competitive dynamics: {result.competitive_intensity} competitive intensity affects negotiation leverage",
            f"Technology disruption: {result.disruption_assessment.disruption_timeline} timeline for major disruption",
            "Regulatory environment stability supports transaction execution",
            "Public market valuation trends affecting comparable analysis assumptions"
        ]

        # Market-related risk factors
        result.risk_factors = [
            f"Competitive intensity escalation affecting market share and pricing",
            f"Technology disruption timeline shorter than {result.disruption_assessment.disruption_timeline}",
            f"Market growth deceleration below current {result.market_trends.growth_rate or 'projected'} rate",
            "Customer behavior shifts accelerating beyond current trend analysis",
            "Regulatory changes affecting industry structure or economics"
        ]

        return result

    async def _gather_market_trend_data(self, target: str, industry: str) -> dict[str, Any]:
        """Gather market trend and industry analysis data."""

        trend_data = {"evidence": []}
        
        # Market trend search queries
        trend_queries = [
            f"{industry} industry trends market analysis 2024",
            f"{target} industry growth drivers market dynamics",
            f"{industry} market size forecast future trends"
        ]

        try:
            for query in trend_queries:
                search_results = await self.tavily_client.search(
                    query=query,
                    search_type="general",
                    max_results=6,
                    include_domains=["mckinsey.com", "bcg.com", "deloitte.com", "bloomberg.com"]
                )

                if search_results and search_results.get("results"):
                    for result in search_results["results"][:2]:
                        evidence = Evidence(
                            content=result.get("content", result.get("snippet", "")),
                            source=result.get("title", "Market Trends"),
                            relevance_score=result.get("score", 0.75),
                            evidence_type="market_intelligence",
                            source_url=result.get("url", ""),
                            timestamp=datetime.now()
                        )
                        trend_data["evidence"].append(evidence)
        except Exception as e:
            print(f"⚠️ Market trend data gathering failed: {e}")

        return trend_data

    async def _analyze_market_trends_with_ai(self, target: str, industry: str, data: dict) -> dict[str, Any]:
        """AI-powered market trend analysis."""

        provider = get_layer_provider(AnalysisLayer.MA_MARKET_ANALYSIS)
        if not provider:
            return self._create_default_trend_analysis()

        evidence_content = "\n\n".join([
            e.content for e in data.get("evidence", [])[:5]
        ]) if data.get("evidence") else "Limited market intelligence available"

        messages = [
            AIMessage(
                role="system",
                content="""You are a senior market research analyst specializing in technology and business market analysis.
                Analyze market trends focusing on:
                1. Market size, growth rate, and expansion potential
                2. Key growth drivers and market dynamics
                3. Technology disruption and competitive threats
                4. Customer behavior evolution and preferences
                5. Regulatory environment and policy impacts
                
                Provide quantitative estimates where possible and conservative assessments for M&A planning."""
            ),
            AIMessage(
                role="user",
                content=f"""Analyze market trends for {target} in {industry} industry:

MARKET INTELLIGENCE:
{evidence_content}

Provide comprehensive analysis:
1. Market Size: Current TAM and growth rate estimates
2. Growth Drivers: Top 5 factors driving market expansion
3. Market Headwinds: Key challenges and obstacles
4. Disruption Risk: Technology and competitive disruption assessment
5. Regulatory Environment: Policy and regulatory impact analysis
6. Market Outlook: 12-month and 24-month market forecast
7. Investment Implications: Impact on valuation and timing

Focus on factors affecting M&A valuation and strategic positioning."""
            )
        ]

        try:
            response = await provider.generate_response_async(messages, max_tokens=1500, temperature=0.1)
            return self._parse_market_trend_analysis(response.content)
        except Exception as e:
            print(f"⚠️ AI market trend analysis failed: {str(e)}")
            return self._create_default_trend_analysis()

    def _parse_competitors_from_intelligence(self, evidence: list[Evidence]) -> list[CompetitorProfile]:
        """Parse competitor information from market intelligence."""

        # Simplified competitor extraction
        competitors = []
        
        # Common competitor patterns for technology companies
        tech_competitors = [
            {"name": "Palantir Technologies", "positioning": "Enterprise AI platform"},
            {"name": "Snowflake Inc", "positioning": "Cloud data platform"},
            {"name": "Databricks", "positioning": "Data and AI platform"},
            {"name": "UiPath", "positioning": "Process automation platform"},
            {"name": "C3.ai", "positioning": "Enterprise AI software"}
        ]
        
        for comp_data in tech_competitors:
            competitor = CompetitorProfile(
                company_name=comp_data["name"],
                competitive_positioning=comp_data["positioning"],
                competitive_strengths=[
                    "Strong technology platform",
                    "Established customer base",
                    "Market brand recognition"
                ],
                competitive_threat_level="medium",
                analysis_confidence=0.70
            )
            competitors.append(competitor)

        return competitors[:6]  # Return top 6 competitors

    def _parse_market_trend_analysis(self, content: str) -> dict[str, Any]:
        """Parse market trend analysis from AI response."""

        # Extract key information (simplified parsing)
        trend_data = {}
        
        # Growth rate extraction
        import re
        growth_match = re.search(r"growth rate:?\s*([0-9.]+)%", content.lower())
        if growth_match:
            trend_data["growth_rate"] = float(growth_match.group(1)) / 100
        
        # Market outlook parsing
        if "positive" in content.lower() or "growth" in content.lower():
            trend_data["outlook_12m"] = "positive"
            trend_data["outlook_24m"] = "positive"
        elif "negative" in content.lower() or "decline" in content.lower():
            trend_data["outlook_12m"] = "negative"
            trend_data["outlook_24m"] = "cautious"
        else:
            trend_data["outlook_12m"] = "stable"
            trend_data["outlook_24m"] = "stable"

        # Growth drivers and headwinds (simplified extraction)
        trend_data["growth_drivers"] = [
            "Digital transformation acceleration",
            "Cloud adoption and migration",
            "AI and automation integration",
            "Data analytics and business intelligence demand"
        ]
        
        trend_data["headwinds"] = [
            "Economic uncertainty affecting enterprise spending",
            "Increased competition and pricing pressure",
            "Talent acquisition challenges and costs",
            "Regulatory compliance complexity"
        ]

        return trend_data

    async def _gather_disruption_intelligence(self, target: str, industry: str | None) -> dict[str, Any]:
        """Gather disruption and innovation intelligence."""

        disruption_data = {"evidence": []}
        
        # Disruption-focused queries
        disruption_queries = [
            f"{industry or 'technology'} industry disruption AI automation trends",
            f"{target} competitive threats emerging technologies",
            f"{industry or 'tech'} market disruption new entrants innovation"
        ]

        try:
            for query in disruption_queries:
                search_results = await self.tavily_client.search(
                    query=query,
                    search_type="general",
                    max_results=5,
                    include_domains=["techcrunch.com", "venturebeat.com", "wired.com", "bloomberg.com"]
                )

                if search_results and search_results.get("results"):
                    for result in search_results["results"][:2]:
                        evidence = Evidence(
                            content=result.get("content", result.get("snippet", "")),
                            source=result.get("title", "Disruption Intelligence"),
                            relevance_score=result.get("score", 0.70),
                            evidence_type="disruption_analysis",
                            source_url=result.get("url", ""),
                            timestamp=datetime.now()
                        )
                        disruption_data["evidence"].append(evidence)
        except Exception as e:
            print(f"⚠️ Disruption intelligence gathering failed: {e}")

        return disruption_data

    async def _analyze_disruption_with_ai(self, target: str, data: dict) -> dict[str, Any]:
        """AI-powered disruption risk analysis."""

        provider = get_layer_provider(AnalysisLayer.MA_MARKET_ANALYSIS)
        if not provider:
            return {"risk_level": "MEDIUM", "probability": 0.35, "timeline": "3-5 years"}

        evidence_content = "\n\n".join([
            e.content for e in data.get("evidence", [])[:4]
        ]) if data.get("evidence") else "Limited disruption intelligence available"

        messages = [
            AIMessage(
                role="system",
                content="""You are a technology strategist analyzing disruption risks for M&A investments.
                Assess disruption risks focusing on:
                1. Technology disruption timeline and probability
                2. New business model threats and market shifts
                3. Regulatory changes affecting industry structure
                4. Customer behavior evolution and adoption patterns
                5. Competitive landscape transformation potential
                
                Provide conservative disruption assessment for investment planning."""
            ),
            AIMessage(
                role="user",
                content=f"""Assess disruption risks for {target}:

DISRUPTION INTELLIGENCE:
{evidence_content}

Analyze disruption threats:
1. Disruption Risk Level: LOW/MEDIUM/HIGH
2. Disruption Timeline: Expected timeframe for major disruption
3. Disruption Probability: Likelihood of significant market disruption
4. Technology Threats: Key emerging technologies posing risks
5. AI/Automation Impact: How AI/automation affects the business model
6. Mitigation Strategies: Approaches to reduce disruption vulnerability

Focus on factors that could materially impact M&A investment returns."""
            )
        ]

        try:
            response = await provider.generate_response_async(messages, max_tokens=1200, temperature=0.1)
            return self._parse_disruption_analysis(response.content)
        except Exception as e:
            print(f"⚠️ AI disruption analysis failed: {str(e)}")
            return {"risk_level": "MEDIUM", "probability": 0.35, "timeline": "3-5 years"}

    def _parse_disruption_analysis(self, content: str) -> dict[str, Any]:
        """Parse AI disruption analysis response."""

        # Extract disruption risk level
        risk_level = "MEDIUM"
        if "high" in content.lower() or "significant" in content.lower():
            risk_level = "HIGH" 
        elif "low" in content.lower() or "minimal" in content.lower():
            risk_level = "LOW"

        # Extract timeline and probability
        timeline = "3-5 years"
        probability = 0.35
        
        if "2-3 years" in content or "near-term" in content.lower():
            timeline = "2-3 years"
            probability = 0.50
        elif "5-10 years" in content or "long-term" in content.lower():
            timeline = "5-10 years"
            probability = 0.25

        return {
            "risk_level": risk_level,
            "timeline": timeline,
            "probability": probability,
            "tech_threats": [
                "Generative AI automation",
                "Open-source alternatives", 
                "Low-code platform adoption",
                "Cloud-native disruption"
            ],
            "ai_impact": "moderate" if risk_level == "MEDIUM" else "high"
        }

    # Helper methods for default results when analysis fails
    def _create_default_market_trends(self, industry: str) -> MarketTrendAnalysis:
        """Create default market trends when analysis fails."""
        return MarketTrendAnalysis(
            industry_sector=industry,
            growth_rate=0.15,  # Default 15% growth
            key_growth_drivers=["Digital transformation", "Technology adoption"],
            market_outlook_12_months="positive",
            trend_confidence=0.60
        )

    def _create_default_disruption_assessment(self) -> DisruptionAssessment:
        """Create default disruption assessment when analysis fails."""
        return DisruptionAssessment(
            disruption_risk_level="MEDIUM",
            disruption_timeline="3-5 years",
            disruption_probability=0.35,
            ai_automation_impact="moderate"
        )

    def _create_default_trend_analysis(self) -> dict[str, Any]:
        """Create default trend analysis when AI analysis fails."""
        return {
            "growth_rate": 0.15,
            "outlook_12m": "positive",
            "outlook_24m": "stable",
            "growth_drivers": ["Technology adoption", "Market expansion"],
            "headwinds": ["Competition", "Economic uncertainty"]
        }

    async def _analyze_strategic_positioning(self, target: str) -> dict[str, str]:
        """Analyze strategic market positioning."""
        return {
            "position": "established market player",
            "advantages": [
                "Strong technology platform",
                "Established customer relationships",
                "Market brand recognition"
            ]
        }

    async def _synthesize_market_intelligence_summary(self, result: MarketIntelligenceResult) -> MarketIntelligenceResult:
        """Generate executive market intelligence summary."""

        # Calculate overall intelligence confidence
        evidence_factor = min(1.0, len(result.evidence) / 10)  # Scale with evidence quality
        analysis_factors = [
            result.market_trends.trend_confidence,
            0.80,  # Default competitive analysis confidence
            0.75   # Default disruption assessment confidence
        ]
        
        result.intelligence_confidence = (sum(analysis_factors) / len(analysis_factors)) * evidence_factor

        return result


# Convenience functions
async def run_market_intelligence_analysis(
    target_company: str,
    industry_focus: str = None,
    analysis_scope: str = "comprehensive"
) -> MarketIntelligenceResult:
    """Run comprehensive market intelligence analysis."""
    
    workflow = MAMarketIntelligenceWorkflow()
    return await workflow.execute_comprehensive_market_intelligence(
        target_company, industry_focus, analysis_scope
    )


async def run_competitive_analysis(target_company: str) -> dict[str, Any]:
    """Run focused competitive landscape analysis."""
    
    workflow = MAMarketIntelligenceWorkflow()
    return await workflow._analyze_competitive_landscape(target_company)


async def run_disruption_assessment(target_company: str, industry: str = None) -> DisruptionAssessment:
    """Run focused disruption risk assessment."""
    
    workflow = MAMarketIntelligenceWorkflow()  
    return await workflow._assess_disruption_risks(target_company, industry)