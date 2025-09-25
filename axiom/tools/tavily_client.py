"""Tavily search client integration."""

import asyncio
from typing import List, Optional, Dict, Any
from tavily import TavilyClient as BaseTavilyClient

from axiom.config.settings import settings
from axiom.tracing.langsmith_tracer import trace_tool


class TavilyClient:
    """Wrapper for Tavily search API with tracing."""

    def __init__(self):
        self.client = BaseTavilyClient(api_key=settings.tavily_api_key)

    @trace_tool("tavily_search")
    async def search(
        self, 
        query: str, 
        max_results: int = 10,
        search_depth: str = "basic"
    ) -> Optional[Dict[str, Any]]:
        """Perform a search using Tavily API."""

        try:
            # Tavily client is synchronous, so we run it in a thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.search(
                    query=query,
                    max_results=max_results,
                    search_depth=search_depth,
                    include_answer=True,
                    include_raw_content=False
                )
            )

            return response

        except Exception as e:
            print(f"Tavily search error: {e}")
            return None

    @trace_tool("tavily_qna_search") 
    async def qna_search(self, query: str) -> Optional[str]:
        """Get a direct answer using Tavily QnA."""

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.qna_search(query=query)
            )

            return response

        except Exception as e:
            print(f"Tavily QnA error: {e}")
            return None
