"""Firecrawl web crawling client integration."""

import asyncio
from typing import List, Optional, Dict, Any
from firecrawl import FirecrawlApp

from axiom.config.settings import settings
from axiom.tracing.langsmith_tracer import trace_tool


class FirecrawlClient:
    """Wrapper for Firecrawl API with tracing."""

    def __init__(self):
        self.client = FirecrawlApp(api_key=settings.firecrawl_api_key)

    @trace_tool("firecrawl_scrape")
    async def scrape(
        self, 
        url: str,
        wait_for: int = 0,
        formats: List[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Scrape a single URL using Firecrawl."""

        if formats is None:
            formats = ["markdown", "html"]

        try:
            # Firecrawl client is synchronous, so we run it in a thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.scrape_url(
                    url=url,
                    params={
                        "waitFor": wait_for,
                        "formats": formats
                    }
                )
            )

            return response

        except Exception as e:
            print(f"Firecrawl scrape error: {e}")
            return None

    @trace_tool("firecrawl_crawl")
    async def crawl(
        self,
        url: str,
        max_pages: int = 5,
        allowed_domains: List[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Crawl a website using Firecrawl."""

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.crawl_url(
                    url=url,
                    params={
                        "limit": max_pages,
                        "allowedDomains": allowed_domains or [],
                        "formats": ["markdown"]
                    }
                )
            )

            return response

        except Exception as e:
            print(f"Firecrawl crawl error: {e}")
            return None
