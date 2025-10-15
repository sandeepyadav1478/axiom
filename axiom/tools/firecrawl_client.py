"""Investment Banking Firecrawl Client - Financial Document Processing."""

import asyncio
import re
from typing import List, Optional, Dict, Any
from firecrawl import FirecrawlApp

from axiom.config.settings import settings
from axiom.tracing.langsmith_tracer import trace_tool


class FirecrawlClient:
    """Enhanced Firecrawl wrapper optimized for financial document processing and SEC filings."""

    def __init__(self):
        self.client = FirecrawlApp(api_key=settings.firecrawl_api_key)

        # Financial document selectors
        self.financial_selectors = {
            "tables": "table, .financial-table, .data-table, .financial-data",
            "metrics": ".metrics, .financial-metrics, .key-metrics, .performance-data",
            "highlights": ".highlights, .key-highlights, .executive-summary",
            "sections": '.financial-section, .investor-section, section[class*="financial"]',
        }

    @trace_tool("financial_document_scrape")
    async def scrape(
        self,
        url: str,
        wait_for: int = 3000,  # Longer wait for financial sites
        formats: List[str] = None,
        include_selectors: List[str] = None,
        exclude_selectors: List[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Scrape financial documents with enhanced processing."""

        if formats is None:
            formats = ["markdown", "html"]  # Both for financial data extraction

        # Enhanced selectors for financial documents
        if include_selectors is None and self._is_financial_document(url):
            include_selectors = [
                self.financial_selectors["tables"],
                self.financial_selectors["metrics"],
                self.financial_selectors["highlights"],
            ]

        # Exclude common non-financial elements
        default_excludes = ["nav", "footer", ".ads", ".sidebar", ".social", ".comments"]
        exclude_selectors = (exclude_selectors or []) + default_excludes

        try:
            # Firecrawl client is synchronous, so we run it in a thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.scrape_url(
                    url=url,
                    params={
                        "waitFor": wait_for,
                        "formats": formats,
                        "includeTags": include_selectors,
                        "excludeTags": exclude_selectors,
                        "onlyMainContent": True,  # Focus on main content for financial docs
                    },
                ),
            )

            # Post-process financial documents
            if response:
                response = self._enhance_financial_content(response, url)

            return response

        except Exception as e:
            print(f"Financial document scrape error for {url}: {e}")
            return None

    @trace_tool("sec_filing_scrape")
    async def scrape_sec_filing(self, url: str) -> Optional[Dict[str, Any]]:
        """Specialized scraping for SEC filings (10-K, 10-Q, 8-K, etc.)."""

        try:
            # SEC-specific configuration
            sec_params = {
                "waitFor": 5000,  # SEC site can be slow
                "formats": ["markdown", "html"],
                "includeTags": [
                    "table[class*='financial']",
                    "div[class*='financialHighlight']",
                    "section[id*='financial']",
                    ".AccordionSection",  # SEC EDGAR specific
                    ".FormData",
                ],
                "excludeTags": [
                    "nav",
                    "footer",
                    ".header",
                    ".navigation",
                    "script",
                    "style",
                    "noscript",
                ],
                "onlyMainContent": True,
            }

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, lambda: self.client.scrape_url(url=url, params=sec_params)
            )

            # Extract SEC filing metadata
            if response:
                response = self._extract_sec_metadata(response, url)

            return response

        except Exception as e:
            print(f"SEC filing scrape error for {url}: {e}")
            return None

    @trace_tool("investor_relations_crawl")
    async def crawl_investor_relations(
        self, base_url: str, max_pages: int = 8, focus_areas: List[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Crawl investor relations sections for comprehensive financial data."""

        if focus_areas is None:
            focus_areas = [
                "earnings",
                "financial",
                "investor",
                "reports",
                "sec-filings",
            ]

        try:
            crawl_params = {
                "limit": max_pages,
                "formats": ["markdown"],
                "allowedDomains": [self._extract_domain(base_url)],
                "includeTags": [
                    self.financial_selectors["tables"],
                    self.financial_selectors["metrics"],
                    "a[href*='earnings']",
                    "a[href*='financial']",
                    "a[href*='10-']",  # SEC filing links
                ],
                "excludeTags": ["nav", "footer", ".ads", ".social", "iframe"],
                "crawlerOptions": {"includes": [f"*{area}*" for area in focus_areas]},
            }

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, lambda: self.client.crawl_url(url=base_url, params=crawl_params)
            )

            # Process crawled investor data
            if response:
                response = self._process_investor_crawl_results(response)

            return response

        except Exception as e:
            print(f"Investor relations crawl error for {base_url}: {e}")
            return None

    @trace_tool("financial_table_extraction")
    async def extract_financial_tables(self, url: str) -> Optional[Dict[str, Any]]:
        """Extract and structure financial tables from documents."""

        try:
            # Focus specifically on tables
            table_params = {
                "waitFor": 4000,
                "formats": ["html"],  # HTML better for table structure
                "includeTags": [
                    "table",
                    ".financial-table",
                    ".data-table",
                    "[class*='table']",
                ],
                "excludeTags": ["nav", "footer", "script", "style"],
                "onlyMainContent": True,
            }

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, lambda: self.client.scrape_url(url=url, params=table_params)
            )

            # Parse and structure tables
            if response:
                response = self._parse_financial_tables(response)

            return response

        except Exception as e:
            print(f"Financial table extraction error for {url}: {e}")
            return None

    def _is_financial_document(self, url: str) -> bool:
        """Check if URL points to a financial document."""

        financial_indicators = [
            "sec.gov",
            "edgar",
            "investor",
            "earnings",
            "financial",
            "10-k",
            "10-q",
            "8-k",
            "annual-report",
            "quarterly",
        ]

        url_lower = url.lower()
        return any(indicator in url_lower for indicator in financial_indicators)

    def _enhance_financial_content(
        self, response: Dict[str, Any], url: str
    ) -> Dict[str, Any]:
        """Enhance scraped content with financial document processing."""

        if "markdown" in response:
            # Add financial document metadata
            response["financial_metadata"] = {
                "document_type": self._detect_document_type(
                    url, response.get("markdown", "")
                ),
                "extraction_timestamp": asyncio.get_event_loop().time(),
                "source_authority": self._assess_source_authority(url),
                "contains_tables": "table" in response.get("html", "").lower(),
                "financial_keywords": self._extract_financial_keywords(
                    response.get("markdown", "")
                ),
            }

        return response

    def _extract_sec_metadata(
        self, response: Dict[str, Any], url: str
    ) -> Dict[str, Any]:
        """Extract SEC filing specific metadata."""

        content = response.get("markdown", "")

        # Extract SEC filing information
        filing_info = {
            "filing_type": self._extract_filing_type(url, content),
            "cik": self._extract_cik(url, content),
            "document_date": self._extract_document_date(content),
            "company_name": self._extract_company_name(content),
            "is_sec_filing": True,
        }

        response["sec_metadata"] = filing_info
        return response

    def _process_investor_crawl_results(
        self, response: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process results from investor relations crawling."""

        if "results" in response:
            # Categorize crawled pages
            categorized_results = {
                "earnings_reports": [],
                "financial_statements": [],
                "sec_filings": [],
                "presentations": [],
                "other": [],
            }

            for result in response["results"]:
                url = result.get("url", "")
                title = result.get("title", "").lower()

                if "earnings" in url or "earnings" in title:
                    categorized_results["earnings_reports"].append(result)
                elif any(term in url for term in ["10-k", "10-q", "8-k"]):
                    categorized_results["sec_filings"].append(result)
                elif "presentation" in url or "presentation" in title:
                    categorized_results["presentations"].append(result)
                elif "financial" in url or "financial" in title:
                    categorized_results["financial_statements"].append(result)
                else:
                    categorized_results["other"].append(result)

            response["categorized_results"] = categorized_results

        return response

    def _parse_financial_tables(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse financial tables from HTML content."""

        html_content = response.get("html", "")
        if not html_content:
            return response

        # Simple table extraction (would use proper HTML parser in production)
        table_pattern = r"<table[^>]*>(.*?)</table>"
        tables = re.findall(table_pattern, html_content, re.DOTALL | re.IGNORECASE)

        response["extracted_tables"] = {
            "count": len(tables),
            "tables": tables[:5] if tables else [],  # Limit to first 5 tables
            "extraction_method": "regex_basic",
        }

        return response

    def _detect_document_type(self, url: str, content: str) -> str:
        """Detect the type of financial document."""

        url_lower = url.lower()
        content_lower = content.lower()

        if "10-k" in url_lower or "annual report" in content_lower:
            return "annual_report"
        elif "10-q" in url_lower or "quarterly" in content_lower:
            return "quarterly_report"
        elif "8-k" in url_lower:
            return "current_report"
        elif "earnings" in url_lower:
            return "earnings_report"
        elif "investor" in url_lower:
            return "investor_relations"
        else:
            return "financial_document"

    def _assess_source_authority(self, url: str) -> str:
        """Assess the authority level of the source."""

        url_lower = url.lower()

        if "sec.gov" in url_lower:
            return "regulatory"
        elif any(
            domain in url_lower
            for domain in ["bloomberg.com", "reuters.com", "wsj.com"]
        ):
            return "high_authority_news"
        elif "investor." in url_lower or "/investor" in url_lower:
            return "official_company"
        else:
            return "standard"

    def _extract_financial_keywords(self, content: str) -> List[str]:
        """Extract financial keywords from content."""

        financial_keywords = [
            "revenue",
            "ebitda",
            "net income",
            "cash flow",
            "debt",
            "valuation",
            "dcf",
            "multiples",
            "enterprise value",
            "merger",
            "acquisition",
            "m&a",
            "synergies",
        ]

        content_lower = content.lower()
        found_keywords = [kw for kw in financial_keywords if kw in content_lower]

        return found_keywords[:10]  # Limit to top 10

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        from urllib.parse import urlparse

        return urlparse(url).netloc

    def _extract_filing_type(self, url: str, content: str) -> str:
        """Extract SEC filing type."""
        filing_types = ["10-K", "10-Q", "8-K", "DEF 14A", "13D", "13G"]
        url_content = (url + " " + content).upper()

        for filing_type in filing_types:
            if filing_type in url_content:
                return filing_type
        return "unknown"

    def _extract_cik(self, url: str, content: str) -> str:
        """Extract CIK number from SEC documents."""
        cik_pattern = r"CIK[:\s]*(\d{10})"
        match = re.search(cik_pattern, content, re.IGNORECASE)
        return match.group(1) if match else ""

    def _extract_document_date(self, content: str) -> str:
        """Extract document date from content."""
        date_pattern = r"(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4})"
        match = re.search(date_pattern, content)
        return match.group(1) if match else ""

    def _extract_company_name(self, content: str) -> str:
        """Extract company name from SEC filing."""
        # Look for company name patterns in SEC filings
        company_patterns = [
            r"COMPANY NAME[:\s]+(.*?)(?:\n|$)",
            r"REGISTRANT[:\s]+(.*?)(?:\n|$)",
            r"<COMPANY-NAME>(.*?)</COMPANY-NAME>",
        ]

        for pattern in company_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return ""
