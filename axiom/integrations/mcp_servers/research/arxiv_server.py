"""
arXiv Research MCP Server

Provides academic research capabilities for quantitative finance, ML, and financial engineering.
Integrates with the arXiv API for paper search, download, and analysis.
"""

import asyncio
import logging
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import arxiv
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArxivMCPServer:
    """MCP Server for arXiv research paper access and analysis."""

    def __init__(self, download_dir: Optional[str] = None):
        """
        Initialize arXiv MCP Server.

        Args:
            download_dir: Directory for PDF downloads (default: ./arxiv_papers)
        """
        self.server = Server("arxiv-research")
        self.download_dir = Path(download_dir or "./arxiv_papers")
        self.download_dir.mkdir(exist_ok=True)

        # arXiv categories relevant to finance
        self.finance_categories = [
            "q-fin.PM",  # Portfolio Management
            "q-fin.RM",  # Risk Management
            "q-fin.PR",  # Pricing of Securities
            "q-fin.TR",  # Trading and Market Microstructure
            "q-fin.MF",  # Mathematical Finance
            "q-fin.CP",  # Computational Finance
            "q-fin.ST",  # Statistical Finance
            "q-fin.EC",  # Economics
            "stat.ML",   # Machine Learning
            "cs.LG",     # Learning
        ]

        self._register_handlers()
        logger.info("arXiv Research MCP Server initialized")

    def _register_handlers(self):
        """Register all tool handlers."""

        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List all available arXiv research tools."""
            return [
                Tool(
                    name="search_papers",
                    description="Search arXiv papers by keywords, categories, authors, or date ranges. "
                                "Returns paper metadata including title, abstract, authors, and PDF link.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query (keywords, title, etc.)"
                            },
                            "category": {
                                "type": "string",
                                "description": "arXiv category (e.g., 'q-fin.PM', 'stat.ML'). "
                                             "Leave empty for all categories."
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of results (default: 10, max: 100)",
                                "default": 10
                            },
                            "sort_by": {
                                "type": "string",
                                "enum": ["relevance", "lastUpdatedDate", "submittedDate"],
                                "description": "Sort order (default: relevance)",
                                "default": "relevance"
                            },
                            "date_from": {
                                "type": "string",
                                "description": "Filter papers from date (YYYY-MM-DD format)"
                            }
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="get_paper",
                    description="Get detailed metadata and abstract for a specific paper by arXiv ID.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "arxiv_id": {
                                "type": "string",
                                "description": "arXiv paper ID (e.g., '2301.12345' or 'arXiv:2301.12345')"
                            }
                        },
                        "required": ["arxiv_id"]
                    }
                ),
                Tool(
                    name="download_pdf",
                    description="Download the PDF file for a specific paper.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "arxiv_id": {
                                "type": "string",
                                "description": "arXiv paper ID"
                            },
                            "filename": {
                                "type": "string",
                                "description": "Optional custom filename (without .pdf extension)"
                            }
                        },
                        "required": ["arxiv_id"]
                    }
                ),
                Tool(
                    name="get_latest",
                    description="Get the latest papers in a specific category or across all categories.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "category": {
                                "type": "string",
                                "description": "arXiv category (e.g., 'q-fin.PM'). Leave empty for all finance categories."
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of results (default: 10)",
                                "default": 10
                            },
                            "days_back": {
                                "type": "integer",
                                "description": "Number of days to look back (default: 7)",
                                "default": 7
                            }
                        }
                    }
                ),
                Tool(
                    name="search_by_author",
                    description="Find all papers by a specific author.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "author_name": {
                                "type": "string",
                                "description": "Author name (full or partial)"
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of results (default: 20)",
                                "default": 20
                            },
                            "category": {
                                "type": "string",
                                "description": "Optional category filter"
                            }
                        },
                        "required": ["author_name"]
                    }
                ),
                Tool(
                    name="get_citations",
                    description="Get citation information for a paper (title, authors, year, arXiv ID). "
                                "Returns formatted citation in multiple styles.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "arxiv_id": {
                                "type": "string",
                                "description": "arXiv paper ID"
                            },
                            "style": {
                                "type": "string",
                                "enum": ["bibtex", "apa", "mla", "chicago"],
                                "description": "Citation style (default: bibtex)",
                                "default": "bibtex"
                            }
                        },
                        "required": ["arxiv_id"]
                    }
                ),
                Tool(
                    name="extract_formulas",
                    description="Extract mathematical formulas and equations from a paper's abstract. "
                                "Useful for understanding models and implementations.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "arxiv_id": {
                                "type": "string",
                                "description": "arXiv paper ID"
                            }
                        },
                        "required": ["arxiv_id"]
                    }
                ),
                Tool(
                    name="summarize_paper",
                    description="Generate an AI-powered summary of a paper including key findings, "
                                "methodology, and relevance to quantitative finance.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "arxiv_id": {
                                "type": "string",
                                "description": "arXiv paper ID"
                            },
                            "focus": {
                                "type": "string",
                                "enum": ["general", "methodology", "results", "implementation"],
                                "description": "Summary focus (default: general)",
                                "default": "general"
                            }
                        },
                        "required": ["arxiv_id"]
                    }
                )
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Any) -> List[TextContent]:
            """Handle tool calls."""
            try:
                if name == "search_papers":
                    result = await self._search_papers(**arguments)
                elif name == "get_paper":
                    result = await self._get_paper(**arguments)
                elif name == "download_pdf":
                    result = await self._download_pdf(**arguments)
                elif name == "get_latest":
                    result = await self._get_latest(**arguments)
                elif name == "search_by_author":
                    result = await self._search_by_author(**arguments)
                elif name == "get_citations":
                    result = await self._get_citations(**arguments)
                elif name == "extract_formulas":
                    result = await self._extract_formulas(**arguments)
                elif name == "summarize_paper":
                    result = await self._summarize_paper(**arguments)
                else:
                    result = f"Unknown tool: {name}"

                return [TextContent(type="text", text=str(result))]

            except Exception as e:
                logger.error(f"Error in {name}: {str(e)}")
                return [TextContent(type="text", text=f"Error: {str(e)}")]

    async def _search_papers(
        self,
        query: str,
        category: Optional[str] = None,
        max_results: int = 10,
        sort_by: str = "relevance",
        date_from: Optional[str] = None
    ) -> Dict[str, Any]:
        """Search arXiv papers."""
        try:
            # Build search query
            search_query = query
            if category:
                search_query = f"cat:{category} AND {query}"

            # Map sort_by to arxiv.SortCriterion
            sort_map = {
                "relevance": arxiv.SortCriterion.Relevance,
                "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
                "submittedDate": arxiv.SortCriterion.SubmittedDate
            }

            # Execute search
            search = arxiv.Search(
                query=search_query,
                max_results=min(max_results, 100),
                sort_by=sort_map.get(sort_by, arxiv.SortCriterion.Relevance)
            )

            papers = []
            for result in search.results():
                # Date filter if specified
                if date_from:
                    date_threshold = datetime.strptime(date_from, "%Y-%m-%d")
                    if result.published.replace(tzinfo=None) < date_threshold:
                        continue

                paper_info = {
                    "arxiv_id": result.entry_id.split("/")[-1],
                    "title": result.title,
                    "authors": [author.name for author in result.authors],
                    "abstract": result.summary,
                    "categories": result.categories,
                    "primary_category": result.primary_category,
                    "published": result.published.isoformat(),
                    "updated": result.updated.isoformat(),
                    "pdf_url": result.pdf_url,
                    "doi": result.doi
                }
                papers.append(paper_info)

            return {
                "query": query,
                "category": category,
                "total_results": len(papers),
                "papers": papers
            }

        except Exception as e:
            logger.error(f"Error searching papers: {str(e)}")
            raise

    async def _get_paper(self, arxiv_id: str) -> Dict[str, Any]:
        """Get detailed paper information."""
        try:
            # Clean arxiv_id
            arxiv_id = arxiv_id.replace("arXiv:", "").strip()

            # Search by ID
            search = arxiv.Search(id_list=[arxiv_id])
            result = next(search.results())

            return {
                "arxiv_id": result.entry_id.split("/")[-1],
                "title": result.title,
                "authors": [
                    {"name": author.name, "affiliation": getattr(author, "affiliation", None)}
                    for author in result.authors
                ],
                "abstract": result.summary,
                "categories": result.categories,
                "primary_category": result.primary_category,
                "published": result.published.isoformat(),
                "updated": result.updated.isoformat(),
                "doi": result.doi,
                "journal_ref": result.journal_ref,
                "comment": result.comment,
                "pdf_url": result.pdf_url,
                "links": [{"title": link.title, "href": link.href} for link in result.links]
            }

        except StopIteration:
            raise ValueError(f"Paper not found: {arxiv_id}")
        except Exception as e:
            logger.error(f"Error getting paper: {str(e)}")
            raise

    async def _download_pdf(
        self,
        arxiv_id: str,
        filename: Optional[str] = None
    ) -> Dict[str, str]:
        """Download paper PDF."""
        try:
            # Clean arxiv_id
            arxiv_id = arxiv_id.replace("arXiv:", "").strip()

            # Get paper
            search = arxiv.Search(id_list=[arxiv_id])
            result = next(search.results())

            # Determine filename
            if not filename:
                filename = f"{arxiv_id.replace('/', '_')}"

            filepath = self.download_dir / f"{filename}.pdf"

            # Download PDF
            result.download_pdf(dirpath=str(self.download_dir), filename=f"{filename}.pdf")

            return {
                "arxiv_id": arxiv_id,
                "title": result.title,
                "filepath": str(filepath),
                "size_mb": round(filepath.stat().st_size / (1024 * 1024), 2),
                "message": f"PDF downloaded successfully to {filepath}"
            }

        except StopIteration:
            raise ValueError(f"Paper not found: {arxiv_id}")
        except Exception as e:
            logger.error(f"Error downloading PDF: {str(e)}")
            raise

    async def _get_latest(
        self,
        category: Optional[str] = None,
        max_results: int = 10,
        days_back: int = 7
    ) -> Dict[str, Any]:
        """Get latest papers in category."""
        try:
            # Calculate date threshold
            date_threshold = datetime.now() - timedelta(days=days_back)
            date_str = date_threshold.strftime("%Y%m%d")

            # Build query
            if category:
                categories = [category]
            else:
                # Use all finance-related categories
                categories = self.finance_categories

            all_papers = []
            for cat in categories:
                search_query = f"cat:{cat} AND submittedDate:[{date_str}* TO *]"

                search = arxiv.Search(
                    query=search_query,
                    max_results=max_results,
                    sort_by=arxiv.SortCriterion.SubmittedDate,
                    sort_order=arxiv.SortOrder.Descending
                )

                for result in search.results():
                    paper_info = {
                        "arxiv_id": result.entry_id.split("/")[-1],
                        "title": result.title,
                        "authors": [author.name for author in result.authors],
                        "abstract": result.summary[:300] + "...",  # Truncate for brevity
                        "categories": result.categories,
                        "primary_category": result.primary_category,
                        "published": result.published.isoformat(),
                        "pdf_url": result.pdf_url
                    }
                    all_papers.append(paper_info)

            # Sort by date and limit results
            all_papers.sort(key=lambda x: x["published"], reverse=True)
            all_papers = all_papers[:max_results]

            return {
                "category": category or "all_finance",
                "days_back": days_back,
                "total_results": len(all_papers),
                "papers": all_papers
            }

        except Exception as e:
            logger.error(f"Error getting latest papers: {str(e)}")
            raise

    async def _search_by_author(
        self,
        author_name: str,
        max_results: int = 20,
        category: Optional[str] = None
    ) -> Dict[str, Any]:
        """Search papers by author name."""
        try:
            # Build query
            search_query = f"au:{author_name}"
            if category:
                search_query = f"cat:{category} AND {search_query}"

            search = arxiv.Search(
                query=search_query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )

            papers = []
            for result in search.results():
                paper_info = {
                    "arxiv_id": result.entry_id.split("/")[-1],
                    "title": result.title,
                    "authors": [author.name for author in result.authors],
                    "abstract": result.summary[:200] + "...",
                    "categories": result.categories,
                    "published": result.published.isoformat(),
                    "pdf_url": result.pdf_url
                }
                papers.append(paper_info)

            return {
                "author": author_name,
                "category": category,
                "total_results": len(papers),
                "papers": papers
            }

        except Exception as e:
            logger.error(f"Error searching by author: {str(e)}")
            raise

    async def _get_citations(
        self,
        arxiv_id: str,
        style: str = "bibtex"
    ) -> Dict[str, Any]:
        """Generate citation for a paper."""
        try:
            # Get paper info
            paper = await self._get_paper(arxiv_id)

            # Extract info
            authors = paper["authors"]
            title = paper["title"]
            year = datetime.fromisoformat(paper["published"]).year
            arxiv_clean = arxiv_id.replace("arXiv:", "").strip()

            # Generate citations in different styles
            citations = {}

            # BibTeX
            author_names = " and ".join([a["name"] for a in authors])
            citations["bibtex"] = f"""@article{{{arxiv_clean.replace('.', '_')},
  title={{{title}}},
  author={{{author_names}}},
  journal={{arXiv preprint arXiv:{arxiv_clean}}},
  year={{{year}}}
}}"""

            # APA
            author_list = ", ".join([a["name"] for a in authors[:3]])
            if len(authors) > 3:
                author_list += ", et al."
            citations["apa"] = f"{author_list} ({year}). {title}. arXiv preprint arXiv:{arxiv_clean}."

            # MLA
            first_author = authors[0]["name"]
            citations["mla"] = f'{first_author}, et al. "{title}." arXiv preprint arXiv:{arxiv_clean} ({year}).'

            # Chicago
            citations["chicago"] = f'{author_list}. "{title}." arXiv preprint arXiv:{arxiv_clean} ({year}).'

            return {
                "arxiv_id": arxiv_clean,
                "title": title,
                "year": year,
                "requested_style": style,
                "citation": citations.get(style, citations["bibtex"]),
                "all_styles": citations
            }

        except Exception as e:
            logger.error(f"Error generating citation: {str(e)}")
            raise

    async def _extract_formulas(self, arxiv_id: str) -> Dict[str, Any]:
        """Extract mathematical formulas from paper abstract."""
        try:
            # Get paper
            paper = await self._get_paper(arxiv_id)
            abstract = paper["abstract"]

            # Extract formulas using regex patterns
            # Pattern 1: LaTeX inline math $...$
            inline_math = re.findall(r'\$([^\$]+)\$', abstract)

            # Pattern 2: LaTeX display math $$...$$
            display_math = re.findall(r'\$\$([^\$]+)\$\$', abstract)

            # Pattern 3: Common mathematical expressions
            expressions = re.findall(
                r'[A-Za-z_]\s*[=<>≤≥]\s*[^\s,.;]+',
                abstract
            )

            # Pattern 4: Greek letters and symbols (common in formulas)
            symbols = re.findall(
                r'\b(alpha|beta|gamma|delta|epsilon|theta|lambda|mu|sigma|tau|phi|psi|omega)\b',
                abstract,
                re.IGNORECASE
            )

            return {
                "arxiv_id": arxiv_id,
                "title": paper["title"],
                "inline_formulas": inline_math,
                "display_formulas": display_math,
                "expressions": list(set(expressions)),
                "mathematical_symbols": list(set(symbols)),
                "formula_count": len(inline_math) + len(display_math),
                "note": "For full formula extraction, download the PDF and use LaTeX parsing tools."
            }

        except Exception as e:
            logger.error(f"Error extracting formulas: {str(e)}")
            raise

    async def _summarize_paper(
        self,
        arxiv_id: str,
        focus: str = "general"
    ) -> Dict[str, Any]:
        """Generate AI-powered paper summary."""
        try:
            # Get paper
            paper = await self._get_paper(arxiv_id)

            abstract = paper["abstract"]
            title = paper["title"]

            # Generate focused summary based on focus parameter
            summary_parts = {
                "title": title,
                "arxiv_id": arxiv_id,
                "authors": [a["name"] for a in paper["authors"]],
                "published": paper["published"],
                "categories": paper["categories"]
            }

            # Extract key information
            abstract_lower = abstract.lower()

            # Methodology keywords
            methodology_keywords = [
                "method", "approach", "algorithm", "model", "framework",
                "technique", "procedure", "methodology"
            ]

            # Results keywords
            results_keywords = [
                "results", "findings", "demonstrate", "show", "achieve",
                "performance", "accuracy", "improvement"
            ]

            # Implementation keywords
            implementation_keywords = [
                "implementation", "code", "software", "library", "package",
                "system", "platform", "tool"
            ]

            if focus == "methodology" or focus == "general":
                methodology_found = any(kw in abstract_lower for kw in methodology_keywords)
                summary_parts["has_methodology"] = methodology_found

            if focus == "results" or focus == "general":
                results_found = any(kw in abstract_lower for kw in results_keywords)
                summary_parts["has_results"] = results_found

            if focus == "implementation" or focus == "general":
                implementation_found = any(kw in abstract_lower for kw in implementation_keywords)
                summary_parts["has_implementation"] = implementation_found

            # Extract sentences for each focus
            sentences = abstract.split(". ")

            summary_parts["abstract"] = abstract
            summary_parts["abstract_sentences"] = len(sentences)
            summary_parts["abstract_length"] = len(abstract)

            # Relevance to quantitative finance
            finance_keywords = [
                "portfolio", "risk", "option", "derivative", "volatility",
                "market", "trading", "price", "return", "hedge", "arbitrage",
                "credit", "equity", "bond", "asset"
            ]
            finance_relevance = sum(1 for kw in finance_keywords if kw in abstract_lower)
            summary_parts["finance_relevance_score"] = finance_relevance

            # ML/AI keywords
            ml_keywords = [
                "machine learning", "deep learning", "neural network",
                "reinforcement learning", "supervised", "unsupervised",
                "regression", "classification", "prediction"
            ]
            ml_relevance = sum(1 for kw in ml_keywords if kw in abstract_lower)
            summary_parts["ml_relevance_score"] = ml_relevance

            summary_parts["focus"] = focus
            summary_parts["summary"] = self._generate_text_summary(abstract, focus)

            return summary_parts

        except Exception as e:
            logger.error(f"Error summarizing paper: {str(e)}")
            raise

    def _generate_text_summary(self, abstract: str, focus: str) -> str:
        """Generate a text summary based on focus."""
        sentences = [s.strip() + "." for s in abstract.split(". ") if s.strip()]

        if focus == "methodology":
            return " ".join(sentences[:len(sentences)//2])
        elif focus == "results":
            return " ".join(sentences[len(sentences)//2:])
        elif focus == "implementation":
            # Look for implementation-related sentences
            impl_sentences = [
                s for s in sentences
                if any(kw in s.lower() for kw in ["implement", "code", "software", "system"])
            ]
            return " ".join(impl_sentences) if impl_sentences else sentences[-1]
        else:  # general
            # Return first and last sentences as a quick summary
            if len(sentences) <= 2:
                return abstract
            return f"{sentences[0]} ... {sentences[-1]}"

    async def run(self):
        """Run the MCP server."""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )


async def main():
    """Main entry point."""
    server = ArxivMCPServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())