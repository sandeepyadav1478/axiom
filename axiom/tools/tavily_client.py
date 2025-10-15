"""Investment Banking Tavily Search Client - Financial Data and M&A Intelligence."""

import asyncio
from typing import Any

from tavily import TavilyClient as BaseTavilyClient

from axiom.config.settings import settings
from axiom.tracing.langsmith_tracer import trace_tool


class TavilyClient:
    """Enhanced Tavily wrapper optimized for investment banking and financial research."""

    def __init__(self):
        self.client = BaseTavilyClient(api_key=settings.tavily_api_key)

        # Financial data source priorities
        self.financial_domains = [
            "sec.gov",
            "edgar.sec.gov",
            "investor.gov",
            "bloomberg.com",
            "reuters.com",
            "wsj.com",
            "ft.com",
            "finance.yahoo.com",
            "marketwatch.com",
            "cnbc.com",
            "fool.com",
            "seekingalpha.com",
            "zacks.com",
            "morningstar.com",
            "investopedia.com",
        ]

    @trace_tool("investment_banking_search")
    async def search(
        self,
        query: str,
        max_results: int = 10,
        search_depth: str = "advanced",
        include_domains: list[str] = None,
        include_raw_content: bool = False,
        time_range: str = None,
    ) -> dict[str, Any] | None:
        """Perform investment banking optimized search using Tavily API."""

        try:
            # Enhance query with financial context
            enhanced_query = self._enhance_financial_query(query)

            # Prioritize financial domains if none specified
            domains = include_domains or self.financial_domains

            # Tavily client is synchronous, so we run it in a thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.search(
                    query=enhanced_query,
                    max_results=max_results,
                    search_depth=search_depth,
                    include_answer=True,
                    include_raw_content=include_raw_content,
                    include_domains=domains[:10] if domains else None,  # Limit domains
                    days=self._parse_time_range(time_range) if time_range else None,
                ),
            )

            # Post-process results for financial relevance
            if response and "results" in response:
                response["results"] = self._rank_financial_results(
                    response["results"], query
                )

            return response

        except Exception as e:
            print(f"Investment banking search error: {e}")
            return None

    @trace_tool("financial_company_search")
    async def company_search(
        self, company_name: str, analysis_type: str = "overview"
    ) -> dict[str, Any] | None:
        """Specialized company search for investment banking analysis."""

        # Create targeted company queries based on analysis type
        query_templates = {
            "overview": f"{company_name} company profile business model revenue financials",
            "financials": f"{company_name} financial statements earnings revenue EBITDA debt cash flow",
            "valuation": f"{company_name} valuation market cap enterprise value trading multiples",
            "ma_analysis": f"{company_name} M&A acquisition merger strategic buyer synergies",
            "risk_analysis": f"{company_name} business risks regulatory compliance ESG issues",
        }

        query = query_templates.get(analysis_type, query_templates["overview"])

        return await self.search(
            query=query,
            max_results=12,
            search_depth="advanced",
            include_raw_content=True,
            time_range="6m",  # Focus on recent data
        )

    @trace_tool("sector_intelligence_search")
    async def sector_search(
        self, sector: str, focus: str = "trends"
    ) -> dict[str, Any] | None:
        """Search for sector-specific intelligence and trends."""

        focus_queries = {
            "trends": f"{sector} industry trends market outlook growth projections",
            "ma_activity": f"{sector} M&A activity mergers acquisitions consolidation trends",
            "valuation": f"{sector} industry valuation multiples trading comparables",
            "regulation": f"{sector} regulatory environment compliance requirements changes",
        }

        query = focus_queries.get(focus, focus_queries["trends"])

        return await self.search(
            query=query, max_results=15, search_depth="advanced", time_range="3m"
        )

    @trace_tool("tavily_qna_search")
    async def qna_search(self, query: str) -> str | None:
        """Get direct financial analysis answer using Tavily QnA."""

        try:
            # Enhance query for financial context
            enhanced_query = self._enhance_financial_query(query)

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, lambda: self.client.qna_search(query=enhanced_query)
            )

            return response

        except Exception as e:
            print(f"Financial QnA search error: {e}")
            return None

    def _enhance_financial_query(self, query: str) -> str:
        """Enhance query with financial and investment banking context."""

        # Financial keywords to boost relevance
        financial_terms = {
            "acquisition": ["M&A", "merger", "due diligence", "synergies"],
            "valuation": ["DCF", "multiples", "enterprise value", "fair value"],
            "analysis": ["financial analysis", "investment banking", "credit analysis"],
            "performance": [
                "financial performance",
                "EBITDA",
                "revenue growth",
                "profitability",
            ],
            "market": ["market analysis", "competitive positioning", "industry trends"],
        }

        query_lower = query.lower()
        enhancements = []

        # Add relevant financial context
        for term, related in financial_terms.items():
            if term in query_lower:
                enhancements.extend(related[:2])  # Add top 2 related terms

        # Combine original query with enhancements
        if enhancements:
            enhanced_query = (
                f"{query} {' '.join(enhancements[:3])}"  # Limit enhancements
            )
        else:
            enhanced_query = query

        return enhanced_query

    def _parse_time_range(self, time_range: str) -> int | None:
        """Parse time range string to days for Tavily API."""

        time_mappings = {
            "1d": 1,
            "1w": 7,
            "2w": 14,
            "1m": 30,
            "3m": 90,
            "6m": 180,
            "1y": 365,
            "2y": 730,
        }

        return time_mappings.get(time_range.lower())

    def _rank_financial_results(
        self, results: list[dict], original_query: str
    ) -> list[dict]:
        """Rank search results by financial relevance and authority."""

        def calculate_financial_score(result: dict) -> float:
            score = result.get("score", 0.0)
            url = result.get("url", "").lower()
            title = result.get("title", "").lower()
            content = result.get("content", "").lower()

            # Boost authoritative financial sources
            authority_boost = 0.0
            for domain in ["sec.gov", "bloomberg.com", "reuters.com", "wsj.com"]:
                if domain in url:
                    authority_boost += 0.3
                    break

            # Boost results with financial keywords
            financial_keywords = [
                "financial",
                "earnings",
                "revenue",
                "valuation",
                "m&a",
                "merger",
                "acquisition",
                "ebitda",
                "dcf",
            ]
            keyword_boost = sum(
                0.1
                for keyword in financial_keywords
                if keyword in title or keyword in content[:500]
            )

            # Boost recent content (simple heuristic)
            recency_boost = (
                0.1
                if any(term in content for term in ["2024", "2023", "recent", "latest"])
                else 0
            )

            return min(1.0, score + authority_boost + keyword_boost + recency_boost)

        # Re-rank results
        for result in results:
            result["financial_score"] = calculate_financial_score(result)

        # Sort by financial relevance
        return sorted(results, key=lambda x: x.get("financial_score", 0), reverse=True)
