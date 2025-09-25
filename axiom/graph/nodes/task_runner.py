"""Task runner node for parallel execution of research tasks."""

import asyncio
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from axiom.config.settings import settings
from axiom.config.schemas import SearchResult, Evidence
from axiom.graph.state import AxiomState
from axiom.tools.tavily_client import TavilyClient
from axiom.tools.firecrawl_client import FirecrawlClient
from axiom.tracing.langsmith_tracer import trace_node


@trace_node("task_runner")
async def task_runner_node(state: AxiomState) -> Dict[str, Any]:
    """Execute research tasks in parallel."""

    tavily = TavilyClient()
    firecrawl = FirecrawlClient()

    search_results = []
    crawl_results = []
    evidence = []

    try:
        # Execute searches for all task plans
        search_tasks = []
        for task_plan in state["task_plans"]:
            for query in task_plan.queries:
                search_tasks.append(tavily.search(query.query))

        # Run searches in parallel (limited by max_parallel_tasks)
        semaphore = asyncio.Semaphore(settings.max_parallel_tasks)

        async def bounded_search(search_task):
            async with semaphore:
                return await search_task

        search_responses = await asyncio.gather(*[bounded_search(task) for task in search_tasks])

        # Process search results
        for response in search_responses:
            if response and hasattr(response, 'results'):
                for result in response.results:
                    search_result = SearchResult(
                        title=result.get('title', ''),
                        url=result.get('url', ''),
                        snippet=result.get('content', ''),
                        score=result.get('score', 0.0)
                    )
                    search_results.append(search_result)

        # Extract evidence from search results
        llm = ChatOpenAI(
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
            model=settings.openai_model_name,
            temperature=0.1
        )

        evidence_prompt = f"""Extract key evidence from these search results for the query: {state['query']}

Search Results:
{chr(10).join([f"- {r.title}: {r.snippet[:200]}..." for r in search_results[:10]])}

Extract 3-5 pieces of evidence with confidence scores (0-1):"""

        evidence_response = await llm.ainvoke([HumanMessage(content=evidence_prompt)])

        # For now, create sample evidence - this would be enhanced with structured parsing
        for i, result in enumerate(search_results[:5]):
            evidence.append(Evidence(
                content=result.snippet[:300],
                source_url=result.url,
                source_title=result.title,
                confidence=0.8,
                relevance_score=result.score
            ))

        return {
            "search_results": state["search_results"] + search_results,
            "crawl_results": state["crawl_results"] + crawl_results,
            "evidence": state["evidence"] + evidence,
            "step_count": state["step_count"] + 1,
            "messages": state["messages"] + [HumanMessage(content=f"Task execution complete: {len(search_results)} results")]
        }

    except Exception as e:
        error_msg = f"Task runner error: {str(e)}"
        return {
            "error_messages": state["error_messages"] + [error_msg],
            "step_count": state["step_count"] + 1
        }
