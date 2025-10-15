"""Investment Banking Task Runner - Parallel Execution of Financial Analysis Tasks."""

import asyncio
import json
from typing import Dict, Any, List
from langchain_core.messages import HumanMessage

from axiom.config.settings import settings
from axiom.config.schemas import SearchResult, Evidence
from axiom.config.ai_layer_config import AnalysisLayer
from axiom.graph.state import AxiomState
from axiom.tools.tavily_client import TavilyClient
from axiom.tools.firecrawl_client import FirecrawlClient
from axiom.tracing.langsmith_tracer import trace_node
from axiom.ai_client_integrations import get_layer_provider, AIMessage


@trace_node("investment_banking_task_runner")
async def task_runner_node(state: AxiomState) -> Dict[str, Any]:
    """Execute investment banking research tasks in parallel with financial data focus."""

    try:
        # Initialize financial data tools
        tavily = TavilyClient()
        firecrawl = FirecrawlClient()

        search_results = []
        crawl_results = []
        evidence = []

        # Get optimal AI provider for task execution
        provider = get_layer_provider(AnalysisLayer.TASK_RUNNER)
        if not provider:
            raise Exception("No available AI provider for task execution")

        # Execute searches for all investment banking task plans
        search_tasks = []
        for task_plan in state["task_plans"]:
            for query in task_plan.queries:
                # Add investment banking specific search parameters
                search_task = tavily.search(
                    query.query,
                    include_domains=[
                        "sec.gov",
                        "bloomberg.com",
                        "reuters.com",
                        "wsj.com",
                        "ft.com",
                    ],
                    max_results=8,
                    include_raw_content=True,
                )
                search_tasks.append((search_task, task_plan.task_id, query.query))

        # Execute searches with financial data source priority
        semaphore = asyncio.Semaphore(settings.max_parallel_analysis_tasks)

        async def bounded_financial_search(search_task_info):
            search_task, task_id, original_query = search_task_info
            async with semaphore:
                try:
                    result = await search_task
                    return (result, task_id, original_query)
                except Exception as e:
                    print(f"Search failed for {original_query}: {str(e)}")
                    return (None, task_id, original_query)

        search_responses = await asyncio.gather(
            *[bounded_financial_search(task_info) for task_info in search_tasks],
            return_exceptions=True,
        )

        # Process financial search results
        task_specific_results = {}
        for response_data in search_responses:
            if isinstance(response_data, Exception):
                continue

            response, task_id, original_query = response_data
            if response and hasattr(response, "results"):
                for result in response.results:
                    search_result = SearchResult(
                        title=result.get("title", ""),
                        url=result.get("url", ""),
                        snippet=result.get("content", ""),
                        score=result.get("score", 0.0),
                        timestamp=result.get("published_at"),
                    )
                    search_results.append(search_result)

                    # Group results by task for analysis
                    if task_id not in task_specific_results:
                        task_specific_results[task_id] = []
                    task_specific_results[task_id].append(search_result)

        # Extract investment banking evidence using AI analysis
        evidence = await extract_financial_evidence(
            provider, state["query"], task_specific_results, state["task_plans"]
        )

        # Optional: Escalate to full content crawling for critical financial documents
        critical_urls = identify_critical_financial_sources(search_results)
        if critical_urls:
            crawl_results = await crawl_financial_documents(
                firecrawl, critical_urls[:3]
            )

        return {
            "search_results": state["search_results"] + search_results,
            "crawl_results": state["crawl_results"] + crawl_results,
            "evidence": state["evidence"] + evidence,
            "step_count": state["step_count"] + 1,
            "messages": state["messages"]
            + [
                HumanMessage(
                    content=f"Investment banking task execution complete: {len(search_results)} search results, {len(evidence)} evidence pieces, {len(crawl_results)} documents crawled"
                )
            ],
        }

    except Exception as e:
        error_msg = f"Investment banking task runner error: {str(e)}"
        return {
            "error_messages": state["error_messages"] + [error_msg],
            "step_count": state["step_count"] + 1,
        }


async def extract_financial_evidence(
    provider,
    original_query: str,
    task_results: Dict[str, List[SearchResult]],
    task_plans: List,
) -> List[Evidence]:
    """Extract structured financial evidence using investment banking AI analysis."""

    evidence = []

    for task_id, results in task_results.items():
        if not results:
            continue

        # Find corresponding task plan
        task_plan = next((tp for tp in task_plans if tp.task_id == task_id), None)
        if not task_plan:
            continue

        # Create investment banking evidence extraction prompt
        results_summary = "\n".join(
            [
                f"• {result.title}\n  Source: {result.url}\n  Content: {result.snippet[:300]}...\n"
                for result in results[:8]
            ]
        )

        evidence_messages = [
            AIMessage(
                role="system",
                content="""You are a senior investment banking analyst extracting key evidence for M&A and financial analysis.

**Task:** Extract 3-5 high-quality evidence pieces from search results.

**Evidence Requirements:**
- Focus on quantitative financial data (revenue, EBITDA, ratios, multiples)
- Include strategic insights (market position, competitive advantages)
- Identify risk factors and regulatory considerations
- Prioritize recent, authoritative sources (SEC filings, analyst reports, financial news)

**Output Format:** For each evidence piece, provide:
- Clear, specific financial or strategic insight
- Source credibility assessment (0.0-1.0 confidence)
- Relevance to investment banking decision-making (0.0-1.0)
- Brief explanation of why this evidence matters for M&A/valuation

Focus on actionable intelligence for investment committee consumption.""",
            ),
            AIMessage(
                role="user",
                content=f"""Investment Banking Query: {original_query}
Task: {task_plan.description}

Search Results:
{results_summary}

Extract key investment banking evidence:""",
            ),
        ]

        try:
            response = await provider.generate_response_async(
                evidence_messages,
                max_tokens=1500,
                temperature=0.05,  # Very conservative for financial evidence
            )

            # Parse AI response and create evidence objects
            # For now, create structured evidence from top results
            for i, result in enumerate(results[:3]):
                evidence.append(
                    Evidence(
                        content=f"[{task_id.replace('_', ' ').title()}] {result.snippet[:400]}",
                        source_url=result.url,
                        source_title=result.title,
                        confidence=min(0.9, 0.7 + (result.score * 0.2)),
                        relevance_score=result.score,
                    )
                )

        except Exception as e:
            print(f"Evidence extraction failed for {task_id}: {str(e)}")
            # Fallback: create basic evidence
            for result in results[:2]:
                evidence.append(
                    Evidence(
                        content=result.snippet[:300],
                        source_url=result.url,
                        source_title=result.title,
                        confidence=0.7,
                        relevance_score=result.score,
                    )
                )

    return evidence


def identify_critical_financial_sources(
    search_results: List[SearchResult],
) -> List[str]:
    """Identify critical financial documents that should be fully crawled."""
    critical_domains = ["sec.gov", "investor.", "ir."]
    critical_keywords = [
        "10-k",
        "10-q",
        "8-k",
        "earnings",
        "investor relations",
        "annual report",
    ]

    critical_urls = []
    for result in search_results:
        url_lower = result.url.lower()
        title_lower = result.title.lower()

        # Prioritize SEC filings and investor relations pages
        if any(domain in url_lower for domain in critical_domains):
            critical_urls.append(result.url)
        elif any(keyword in title_lower for keyword in critical_keywords):
            critical_urls.append(result.url)

    return critical_urls[:5]  # Limit to top 5 critical sources


async def crawl_financial_documents(
    firecrawl: FirecrawlClient, urls: List[str]
) -> List:
    """Crawl critical financial documents for detailed analysis."""
    crawl_results = []

    for url in urls:
        try:
            # Crawl with financial document optimization
            result = await firecrawl.crawl(
                url,
                include_selectors=["table", ".financial-data", ".metrics"],
                exclude_selectors=["nav", "footer", ".ads"],
                max_depth=1,
            )
            if result:
                crawl_results.append(result)
        except Exception as e:
            print(f"Failed to crawl {url}: {str(e)}")

    return crawl_results
