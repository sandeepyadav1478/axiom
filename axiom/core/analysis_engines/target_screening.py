"""
M&A Target Identification & Screening Workflow

Systematic identification, evaluation, and prioritization of M&A targets
based on strategic criteria, financial thresholds, and market opportunities.
"""

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


class TargetCriteria(BaseModel):
    """Criteria for M&A target identification and screening."""

    # Strategic Criteria
    industry_sectors: list[str] = Field(..., description="Target industry sectors")
    geographic_regions: list[str] = Field(
        default=["US", "EU"], description="Target geographic regions"
    )
    strategic_rationale: str = Field(
        ..., description="Strategic reason for acquisition"
    )

    # Financial Criteria
    min_revenue: float = Field(
        default=10_000_000, description="Minimum annual revenue ($)"
    )
    max_revenue: float = Field(
        default=10_000_000_000, description="Maximum annual revenue ($)"
    )
    min_ebitda_margin: float = Field(
        default=0.10, description="Minimum EBITDA margin (%)"
    )
    max_debt_to_equity: float = Field(
        default=2.0, description="Maximum debt-to-equity ratio"
    )
    min_growth_rate: float = Field(
        default=0.05, description="Minimum revenue growth rate (%)"
    )

    # Market Criteria
    min_market_share: float = Field(
        default=0.01, description="Minimum market share (%)"
    )
    market_position: str = Field(
        default="top_5", description="Required market position"
    )
    competitive_moat: list[str] = Field(
        default=[], description="Required competitive advantages"
    )

    # Deal Criteria
    max_valuation: float = Field(
        default=5_000_000_000, description="Maximum target valuation ($)"
    )
    deal_type: str = Field(default="acquisition", description="Type of transaction")
    integration_complexity: str = Field(
        default="medium", description="Acceptable integration complexity"
    )


class TargetProfile(BaseModel):
    """Profile of a potential M&A target."""

    # Company Information
    company_name: str = Field(..., description="Company name")
    ticker: str | None = Field(None, description="Stock ticker if public")
    industry: str = Field(..., description="Primary industry")
    sector: str = Field(..., description="Market sector")
    headquarters: str = Field(..., description="Company headquarters location")
    founded_year: int | None = Field(None, description="Year founded")
    employee_count: int | None = Field(None, description="Number of employees")

    # Financial Metrics
    annual_revenue: float | None = Field(None, description="Latest annual revenue")
    ebitda: float | None = Field(None, description="Latest EBITDA")
    ebitda_margin: float | None = Field(None, description="EBITDA margin")
    net_income: float | None = Field(None, description="Latest net income")
    revenue_growth: float | None = Field(None, description="Revenue growth rate")
    market_cap: float | None = Field(None, description="Market capitalization")
    enterprise_value: float | None = Field(None, description="Enterprise value")

    # Strategic Information
    business_model: str | None = Field(None, description="Primary business model")
    key_products: list[str] = Field(default=[], description="Key products/services")
    customer_base: str | None = Field(None, description="Customer base description")
    competitive_advantages: list[str] = Field(
        default=[], description="Competitive moats"
    )
    market_position: str | None = Field(None, description="Market position description")

    # Deal Relevance
    strategic_fit_score: float = Field(
        default=0.0, description="Strategic fit score (0-1)"
    )
    financial_attractiveness: float = Field(
        default=0.0, description="Financial attractiveness (0-1)"
    )
    acquisition_probability: float = Field(
        default=0.0, description="Acquisition feasibility (0-1)"
    )
    estimated_valuation: float | None = Field(
        None, description="Estimated valuation range"
    )

    # Metadata
    last_updated: datetime = Field(default_factory=datetime.now)
    data_sources: list[str] = Field(default=[], description="Information sources")
    confidence_level: float = Field(default=0.0, description="Data confidence (0-1)")


class ScreeningResult(BaseModel):
    """Result of M&A target screening process."""

    criteria: TargetCriteria = Field(..., description="Screening criteria used")
    targets_identified: list[TargetProfile] = Field(
        default=[], description="Identified targets"
    )
    targets_screened: int = Field(default=0, description="Total targets screened")
    targets_qualified: int = Field(default=0, description="Targets meeting criteria")

    # Analysis Summary
    industry_coverage: list[str] = Field(default=[], description="Industries analyzed")
    geographic_coverage: list[str] = Field(default=[], description="Regions analyzed")
    key_findings: list[str] = Field(default=[], description="Key screening insights")
    market_insights: list[str] = Field(default=[], description="Market intelligence")

    # Supporting Data
    evidence: list[Evidence] = Field(default=[], description="Supporting evidence")
    citations: list[Citation] = Field(default=[], description="Source citations")

    # Metadata
    execution_time: float = Field(default=0.0, description="Workflow execution time")
    confidence_level: float = Field(default=0.0, description="Overall confidence")
    timestamp: datetime = Field(default_factory=datetime.now)


class MATargetScreeningWorkflow:
    """M&A Target Identification and Screening Workflow."""

    def __init__(self):
        self.tavily_client = TavilyClient()
        self.firecrawl_client = FirecrawlClient()

    @trace_node("ma_target_screening")
    async def execute(self, criteria: TargetCriteria) -> ScreeningResult:
        """Execute comprehensive M&A target screening workflow."""

        start_time = datetime.now()
        result = ScreeningResult(criteria=criteria, timestamp=start_time)

        try:
            # Step 1: Industry Analysis & Market Mapping
            print("🔍 Phase 1: Industry Analysis & Market Mapping...")
            market_intelligence = await self._analyze_industry_landscape(criteria)
            result.market_insights.extend(market_intelligence["insights"])
            result.evidence.extend(market_intelligence["evidence"])

            # Step 2: Target Identification
            print("🎯 Phase 2: Target Identification...")
            potential_targets = await self._identify_potential_targets(criteria)
            result.targets_screened = len(potential_targets)

            # Step 3: Financial Screening
            print("💰 Phase 3: Financial Screening...")
            qualified_targets = await self._apply_financial_screening(
                potential_targets, criteria
            )

            # Step 4: Strategic Fit Analysis
            print("🎯 Phase 4: Strategic Fit Analysis...")
            analyzed_targets = await self._analyze_strategic_fit(
                qualified_targets, criteria
            )

            # Step 5: Prioritization & Ranking
            print("📊 Phase 5: Target Prioritization...")
            final_targets = self._prioritize_targets(analyzed_targets)

            result.targets_identified = final_targets
            result.targets_qualified = len(final_targets)
            result.industry_coverage = criteria.industry_sectors
            result.geographic_coverage = criteria.geographic_regions

            # Generate summary insights
            result.key_findings = await self._generate_screening_insights(
                result, criteria
            )

            # Calculate overall confidence
            result.confidence_level = self._calculate_screening_confidence(result)

            execution_time = (datetime.now() - start_time).total_seconds()
            result.execution_time = execution_time

            print(
                f"✅ Screening completed: {result.targets_qualified}/{result.targets_screened} targets qualified"
            )

            return result

        except Exception as e:
            raise FinancialDataError(
                f"M&A target screening failed: {str(e)}",
                context={"criteria": criteria.model_dump()},
            )

    async def _analyze_industry_landscape(
        self, criteria: TargetCriteria
    ) -> dict[str, Any]:
        """Analyze industry landscape to understand market dynamics."""

        market_intelligence = {"insights": [], "evidence": []}

        for sector in criteria.industry_sectors:
            # Search for industry analysis and trends
            search_query = (
                f"{sector} industry analysis market leaders companies M&A trends 2024"
            )
            search_results = await self.tavily_client.search(
                query=search_query,
                search_type="company",
                max_results=15,
                include_domains=["bloomberg.com", "reuters.com", "wsj.com", "ft.com"],
            )

            if search_results and search_results.get("results"):
                for result in search_results["results"][:5]:
                    evidence = Evidence(
                        content=result.get("content", result.get("snippet", "")),
                        source=result.get("title", "Industry Analysis"),
                        relevance_score=result.get("score", 0.8),
                        evidence_type="market_intelligence",
                        source_url=result.get("url", ""),
                        timestamp=datetime.now(),
                    )
                    market_intelligence["evidence"].append(evidence)

                # Extract key market insights
                insights = self._extract_market_insights(search_results, sector)
                market_intelligence["insights"].extend(insights)

        return market_intelligence

    async def _identify_potential_targets(self, criteria: TargetCriteria) -> list[dict]:
        """Identify potential M&A targets based on criteria."""

        potential_targets = []

        for sector in criteria.industry_sectors:
            # Search for companies in each sector
            for region in criteria.geographic_regions:
                search_queries = [
                    f"top {sector} companies {region} revenue market leaders",
                    f"{sector} industry {region} fastest growing companies",
                    f"private {sector} companies {region} acquisition targets",
                    f"{sector} startups {region} Series B Series C funding",
                ]

                for query in search_queries:
                    search_results = await self.tavily_client.search(
                        query=query, search_type="company", max_results=10
                    )

                    if search_results and search_results.get("results"):
                        targets = self._parse_company_results(
                            search_results["results"], sector, region
                        )
                        potential_targets.extend(targets)

        # Remove duplicates by company name
        unique_targets = {}
        for target in potential_targets:
            company_name = target.get("company_name", "").strip().lower()
            if company_name and company_name not in unique_targets:
                unique_targets[company_name] = target

        return list(unique_targets.values())

    async def _apply_financial_screening(
        self, targets: list[dict], criteria: TargetCriteria
    ) -> list[TargetProfile]:
        """Apply financial screening criteria to filter targets."""

        qualified_targets = []

        for target_data in targets:
            try:
                # Create target profile
                target = self._create_target_profile(target_data)

                # Apply financial criteria
                if self._meets_financial_criteria(target, criteria):
                    # Enhance with additional financial data
                    enhanced_target = await self._enhance_financial_data(target)
                    qualified_targets.append(enhanced_target)

            except Exception as e:
                print(
                    f"⚠️  Error processing target {target_data.get('company_name')}: {str(e)}"
                )
                continue

        return qualified_targets

    async def _analyze_strategic_fit(
        self, targets: list[TargetProfile], criteria: TargetCriteria
    ) -> list[TargetProfile]:
        """Analyze strategic fit for each qualified target."""

        analyzed_targets = []
        provider = get_layer_provider(AnalysisLayer.MA_STRATEGIC_FIT)

        if not provider:
            print("⚠️  No AI provider available for strategic fit analysis")
            return targets

        for target in targets:
            try:
                # Create strategic fit analysis prompt
                messages = [
                    AIMessage(
                        role="system",
                        content="""You are an M&A strategic advisor analyzing potential acquisition targets.
                        Evaluate strategic fit based on:
                        1. Business model synergies
                        2. Market position enhancement
                        3. Technology and capability gaps
                        4. Customer base expansion
                        5. Geographic expansion opportunities
                        6. Cultural and operational alignment

                        Provide a strategic fit score (0-1) and detailed rationale.""",
                    ),
                    AIMessage(
                        role="user",
                        content=f"""Analyze strategic fit for potential acquisition:

TARGET COMPANY: {target.company_name}
Industry: {target.industry}
Revenue: ${target.annual_revenue:,.0f} (if available)
Business Model: {target.business_model}
Key Products: {', '.join(target.key_products)}
Market Position: {target.market_position}

ACQUISITION CRITERIA:
Strategic Rationale: {criteria.strategic_rationale}
Target Industries: {', '.join(criteria.industry_sectors)}
Deal Type: {criteria.deal_type}

Please provide:
1. Strategic fit score (0.0-1.0)
2. Key synergy opportunities
3. Integration challenges
4. Strategic recommendation""",
                    ),
                ]

                # Get strategic fit analysis
                response = await provider.generate_response_async(
                    messages,
                    max_tokens=1500,
                    temperature=0.1,  # Conservative for M&A decisions
                )

                # Parse strategic fit score
                strategic_score = self._extract_strategic_fit_score(response.content)
                target.strategic_fit_score = strategic_score

                analyzed_targets.append(target)

            except Exception as e:
                print(
                    f"⚠️  Strategic fit analysis failed for {target.company_name}: {str(e)}"
                )
                target.strategic_fit_score = 0.5  # Default neutral score
                analyzed_targets.append(target)

        return analyzed_targets

    def _prioritize_targets(self, targets: list[TargetProfile]) -> list[TargetProfile]:
        """Prioritize and rank targets based on overall attractiveness."""

        for target in targets:
            # Calculate composite attractiveness score
            financial_score = self._calculate_financial_score(target)
            strategic_score = target.strategic_fit_score

            # Weighted average (60% strategic fit, 40% financial attractiveness)
            target.acquisition_probability = (strategic_score * 0.6) + (
                financial_score * 0.4
            )
            target.financial_attractiveness = financial_score

        # Sort by acquisition probability (highest first)
        targets.sort(key=lambda t: t.acquisition_probability, reverse=True)

        return targets[:20]  # Return top 20 targets

    async def _generate_screening_insights(
        self, result: ScreeningResult, criteria: TargetCriteria
    ) -> list[str]:
        """Generate key insights from the screening process."""

        if not result.targets_identified:
            return ["No qualified targets identified meeting specified criteria"]

        insights = []

        # Market insights
        avg_revenue = sum(
            t.annual_revenue or 0 for t in result.targets_identified
        ) / len(result.targets_identified)
        insights.append(f"Average target revenue: ${avg_revenue:,.0f}")

        top_sectors = {}
        for target in result.targets_identified:
            sector = target.sector or target.industry
            top_sectors[sector] = top_sectors.get(sector, 0) + 1

        if top_sectors:
            top_sector = max(top_sectors, key=top_sectors.get)
            insights.append(
                f"Most represented sector: {top_sector} ({top_sectors[top_sector]} targets)"
            )

        # Strategic insights
        high_fit_targets = [
            t for t in result.targets_identified if t.strategic_fit_score > 0.7
        ]
        if high_fit_targets:
            insights.append(
                f"{len(high_fit_targets)} targets show high strategic fit (>0.7)"
            )

        # Financial insights
        profitable_targets = [
            t for t in result.targets_identified if (t.ebitda or 0) > 0
        ]
        insights.append(
            f"{len(profitable_targets)}/{len(result.targets_identified)} targets are profitable"
        )

        return insights

    def _meets_financial_criteria(
        self, target: TargetProfile, criteria: TargetCriteria
    ) -> bool:
        """Check if target meets financial screening criteria."""

        # Revenue criteria
        if target.annual_revenue:
            if not (
                criteria.min_revenue <= target.annual_revenue <= criteria.max_revenue
            ):
                return False

        # EBITDA margin criteria
        if target.ebitda_margin:
            if target.ebitda_margin < criteria.min_ebitda_margin:
                return False

        # Growth rate criteria
        if target.revenue_growth:
            if target.revenue_growth < criteria.min_growth_rate:
                return False

        return True

    def _calculate_financial_score(self, target: TargetProfile) -> float:
        """Calculate financial attractiveness score."""

        score = 0.0
        factors = 0

        # Revenue size (normalized)
        if target.annual_revenue:
            revenue_score = min(
                target.annual_revenue / 1_000_000_000, 1.0
            )  # Cap at $1B
            score += revenue_score * 0.2
            factors += 0.2

        # Profitability
        if target.ebitda_margin:
            margin_score = min(target.ebitda_margin / 0.3, 1.0)  # Cap at 30% margin
            score += margin_score * 0.3
            factors += 0.3

        # Growth
        if target.revenue_growth:
            growth_score = min(target.revenue_growth / 0.5, 1.0)  # Cap at 50% growth
            score += growth_score * 0.3
            factors += 0.3

        # Market position (qualitative scoring)
        if target.market_position:
            position_score = 0.7  # Default good score if market position exists
            if "leader" in target.market_position.lower():
                position_score = 1.0
            elif "top" in target.market_position.lower():
                position_score = 0.8

            score += position_score * 0.2
            factors += 0.2

        return score / factors if factors > 0 else 0.5

    def _calculate_screening_confidence(self, result: ScreeningResult) -> float:
        """Calculate overall confidence in screening results."""

        if not result.targets_identified:
            return 0.0

        # Average target confidence
        avg_target_confidence = sum(
            t.confidence_level for t in result.targets_identified
        ) / len(result.targets_identified)

        # Data coverage factor
        coverage_factor = min(
            len(result.evidence) / 20, 1.0
        )  # Ideal: 20+ evidence pieces

        # Source diversity factor
        unique_sources = set()
        for evidence in result.evidence:
            if evidence.source_url:
                domain = (
                    evidence.source_url.split("/")[2]
                    if "/" in evidence.source_url
                    else evidence.source_url
                )
                unique_sources.add(domain)

        source_factor = min(len(unique_sources) / 10, 1.0)  # Ideal: 10+ unique sources

        # Weighted confidence calculation
        overall_confidence = (
            (avg_target_confidence * 0.6)
            + (coverage_factor * 0.2)
            + (source_factor * 0.2)
        )

        return round(overall_confidence, 2)

    def _extract_strategic_fit_score(self, analysis_content: str) -> float:
        """Extract strategic fit score from AI analysis."""

        import re

        # Look for score patterns
        score_patterns = [
            r"strategic fit score:?\s*([0-9.]+)",
            r"score:?\s*([0-9.]+)",
            r"fit score:?\s*([0-9.]+)",
            r"rating:?\s*([0-9.]+)",
        ]

        for pattern in score_patterns:
            match = re.search(pattern, analysis_content.lower())
            if match:
                try:
                    score = float(match.group(1))
                    return min(max(score, 0.0), 1.0)  # Clamp to 0-1 range
                except ValueError:
                    continue

        # Default scoring based on content analysis
        positive_terms = [
            "strong",
            "excellent",
            "high",
            "good",
            "synergy",
            "strategic",
            "valuable",
        ]
        negative_terms = [
            "weak",
            "poor",
            "low",
            "limited",
            "risk",
            "challenge",
            "difficult",
        ]

        positive_count = sum(
            1 for term in positive_terms if term in analysis_content.lower()
        )
        negative_count = sum(
            1 for term in negative_terms if term in analysis_content.lower()
        )

        base_score = 0.5 + (positive_count - negative_count) * 0.1
        return min(max(base_score, 0.0), 1.0)

    def _create_target_profile(self, target_data: dict) -> TargetProfile:
        """Create TargetProfile from raw target data."""

        return TargetProfile(
            company_name=target_data.get("company_name", "Unknown"),
            industry=target_data.get("industry", "Unknown"),
            sector=target_data.get("sector", target_data.get("industry", "Unknown")),
            headquarters=target_data.get(
                "headquarters", target_data.get("location", "Unknown")
            ),
            annual_revenue=target_data.get("revenue"),
            market_cap=target_data.get("market_cap"),
            business_model=target_data.get("business_model"),
            confidence_level=target_data.get("confidence", 0.6),
        )

    def _parse_company_results(
        self, search_results: list, sector: str, region: str
    ) -> list[dict]:
        """Parse search results to extract company information."""

        companies = []

        for result in search_results:
            content = result.get("content", result.get("snippet", "")).lower()
            title = result.get("title", "")

            # Extract company names (simplified pattern matching)
            company_indicators = [
                "company",
                "corp",
                "inc",
                "ltd",
                "technologies",
                "systems",
            ]

            if any(indicator in content for indicator in company_indicators):
                company_data = {
                    "company_name": title.split(" - ")[0] if " - " in title else title,
                    "industry": sector,
                    "sector": sector,
                    "headquarters": region,
                    "source_url": result.get("url"),
                    "confidence": result.get("score", 0.6),
                }

                # Try to extract basic financial info from content
                revenue_match = re.search(r"\$([0-9.]+)\s*(billion|million)", content)
                if revenue_match:
                    amount = float(revenue_match.group(1))
                    multiplier = (
                        1_000_000_000
                        if revenue_match.group(2) == "billion"
                        else 1_000_000
                    )
                    company_data["revenue"] = amount * multiplier

                companies.append(company_data)

        return companies

    async def _enhance_financial_data(self, target: TargetProfile) -> TargetProfile:
        """Enhance target profile with additional financial data."""

        # Search for detailed financial information
        financial_query = (
            f"{target.company_name} financial statements revenue EBITDA profit"
        )

        try:
            search_results = await self.tavily_client.search(
                query=financial_query,
                search_type="company",
                max_results=5,
                include_domains=["sec.gov", "investor.", "ir.", "bloomberg.com"],
            )

            if search_results and search_results.get("results"):
                # Parse financial data from search results
                financial_data = self._extract_financial_metrics(
                    search_results["results"]
                )

                # Update target profile with enhanced data
                if financial_data.get("revenue"):
                    target.annual_revenue = financial_data["revenue"]
                if financial_data.get("ebitda"):
                    target.ebitda = financial_data["ebitda"]
                if financial_data.get("ebitda_margin"):
                    target.ebitda_margin = financial_data["ebitda_margin"]
                if financial_data.get("growth_rate"):
                    target.revenue_growth = financial_data["growth_rate"]

                target.confidence_level = min(target.confidence_level + 0.2, 1.0)

        except Exception as e:
            print(
                f"⚠️  Could not enhance financial data for {target.company_name}: {str(e)}"
            )

        return target

    def _extract_market_insights(self, search_results: dict, sector: str) -> list[str]:
        """Extract market insights from search results."""

        insights = []

        if search_results and search_results.get("results"):
            for result in search_results["results"][:3]:
                content = result.get("content", result.get("snippet", "")).lower()

                # Extract growth insights
                if "growth" in content and any(
                    word in content for word in ["percent", "%", "billion", "million"]
                ):
                    insights.append(f"{sector} sector showing strong growth momentum")

                # Extract consolidation insights
                if any(
                    word in content
                    for word in ["acquisition", "merger", "consolidation", "buyout"]
                ):
                    insights.append(
                        f"{sector} industry experiencing M&A activity and consolidation"
                    )

                # Extract market size insights
                if "market size" in content or "industry size" in content:
                    insights.append(f"{sector} market size and dynamics analyzed")

        return insights[:5]  # Limit insights

    def _extract_financial_metrics(self, search_results: list) -> dict:
        """Extract financial metrics from search result content."""

        financial_data = {}

        for result in search_results:
            content = result.get("content", result.get("snippet", "")).lower()

            # Revenue extraction
            revenue_patterns = [
                r"revenue of \$([0-9.]+)\s*(billion|million)",
                r"annual revenue \$([0-9.]+)\s*(billion|million)",
                r"sales of \$([0-9.]+)\s*(billion|million)",
            ]

            for pattern in revenue_patterns:
                match = re.search(pattern, content)
                if match:
                    amount = float(match.group(1))
                    multiplier = (
                        1_000_000_000 if match.group(2) == "billion" else 1_000_000
                    )
                    financial_data["revenue"] = amount * multiplier
                    break

            # EBITDA extraction
            ebitda_patterns = [
                r"ebitda of \$([0-9.]+)\s*(billion|million)",
                r"ebitda \$([0-9.]+)\s*(billion|million)",
            ]

            for pattern in ebitda_patterns:
                match = re.search(pattern, content)
                if match:
                    amount = float(match.group(1))
                    multiplier = (
                        1_000_000_000 if match.group(2) == "billion" else 1_000_000
                    )
                    financial_data["ebitda"] = amount * multiplier
                    break

            # Growth rate extraction
            growth_patterns = [
                r"growth of ([0-9.]+)%",
                r"grew ([0-9.]+)%",
                r"increased ([0-9.]+)%",
            ]

            for pattern in growth_patterns:
                match = re.search(pattern, content)
                if match:
                    financial_data["growth_rate"] = float(match.group(1)) / 100
                    break

        # Calculate EBITDA margin if both revenue and EBITDA available
        if "revenue" in financial_data and "ebitda" in financial_data:
            if financial_data["revenue"] > 0:
                financial_data["ebitda_margin"] = (
                    financial_data["ebitda"] / financial_data["revenue"]
                )

        return financial_data


# Convenience function for workflow execution
async def run_target_screening(criteria: TargetCriteria) -> ScreeningResult:
    """Run M&A target screening workflow."""

    workflow = MATargetScreeningWorkflow()
    return await workflow.execute(criteria)
