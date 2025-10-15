"""Investment Banking MCP (Model Context Protocol) Adapter - Financial Tool Integration."""

import asyncio
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod

from axiom.tools.tavily_client import TavilyClient
from axiom.tools.firecrawl_client import FirecrawlClient


class MCPTool(ABC):
    """Abstract base class for MCP-compatible investment banking tools."""

    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool with given parameters."""
        pass

    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """Get the tool schema for MCP."""
        pass


class InvestmentBankingSearchTool(MCPTool):
    """MCP adapter for investment banking optimized search."""

    def __init__(self):
        self.client = TavilyClient()

    async def execute(self, query: str, max_results: int = 10, search_type: str = "general", **kwargs) -> Dict[str, Any]:
        """Execute investment banking search with financial optimization."""

        try:
            if search_type == "company":
                company_name = kwargs.get("company_name", query)
                analysis_type = kwargs.get("analysis_type", "overview")
                result = await self.client.company_search(company_name, analysis_type)
            elif search_type == "sector":
                sector = kwargs.get("sector", query)
                focus = kwargs.get("focus", "trends")
                result = await self.client.sector_search(sector, focus)
            else:
                # General investment banking search
                result = await self.client.search(
                    query=query,
                    max_results=max_results,
                    search_depth="advanced",
                    include_raw_content=True,
                    time_range=kwargs.get("time_range", "6m")
                )

            if result:
                return {
                    "success": True,
                    "results": result.get("results", []),
                    "answer": result.get("answer", ""),
                    "query": query,
                    "search_type": search_type,
                    "financial_ranking": True
                }

            return {
                "success": False,
                "error": "Investment banking search failed",
                "query": query
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Investment banking search error: {str(e)}",
                "query": query
            }

    def get_schema(self) -> Dict[str, Any]:
        """Get MCP schema for investment banking search."""

        return {
            "name": "investment_banking_search",
            "description": "Search financial and investment banking data with authority ranking and domain filtering",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The investment banking search query (company, M&A, valuation, etc.)"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 20
                    },
                    "search_type": {
                        "type": "string",
                        "enum": ["general", "company", "sector"],
                        "description": "Type of investment banking search",
                        "default": "general"
                    },
                    "company_name": {
                        "type": "string",
                        "description": "Company name for company-specific searches"
                    },
                    "analysis_type": {
                        "type": "string",
                        "enum": ["overview", "financials", "valuation", "ma_analysis", "risk_analysis"],
                        "description": "Type of company analysis",
                        "default": "overview"
                    },
                    "sector": {
                        "type": "string",
                        "description": "Industry sector for sector analysis"
                    },
                    "focus": {
                        "type": "string",
                        "enum": ["trends", "ma_activity", "valuation", "regulation"],
                        "description": "Focus area for sector analysis",
                        "default": "trends"
                    },
                    "time_range": {
                        "type": "string",
                        "enum": ["1d", "1w", "1m", "3m", "6m", "1y"],
                        "description": "Time range for recent data",
                        "default": "6m"
                    }
                },
                "required": ["query"]
            }
        }


class FinancialDocumentTool(MCPTool):
    """MCP adapter for financial document processing."""

    def __init__(self):
        self.client = FirecrawlClient()

    async def execute(self, url: str, document_type: str = "financial", **kwargs) -> Dict[str, Any]:
        """Execute financial document processing."""

        try:
            if document_type == "sec_filing":
                result = await self.client.scrape_sec_filing(url)
            elif document_type == "financial_tables":
                result = await self.client.extract_financial_tables(url)
            elif document_type == "investor_relations":
                max_pages = kwargs.get("max_pages", 5)
                result = await self.client.crawl_investor_relations(url, max_pages)
            else:
                # General financial document
                formats = kwargs.get("formats", ["markdown", "html"])
                include_selectors = kwargs.get("include_selectors")
                result = await self.client.scrape(
                    url=url,
                    formats=formats,
                    include_selectors=include_selectors
                )

            if result:
                return {
                    "success": True,
                    "url": url,
                    "content": result.get("markdown", result.get("content", "")),
                    "html_content": result.get("html", ""),
                    "metadata": result.get("metadata", {}),
                    "financial_metadata": result.get("financial_metadata", {}),
                    "sec_metadata": result.get("sec_metadata", {}),
                    "document_type": document_type,
                    "extracted_tables": result.get("extracted_tables", {}),
                    "categorized_results": result.get("categorized_results", {})
                }

            return {
                "success": False,
                "error": "Financial document processing failed",
                "url": url
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Financial document processing error: {str(e)}",
                "url": url
            }

    def get_schema(self) -> Dict[str, Any]:
        """Get MCP schema for financial document processing."""

        return {
            "name": "financial_document_processor",
            "description": "Process financial documents including SEC filings, earnings reports, and investor relations pages",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL of the financial document to process",
                        "format": "uri"
                    },
                    "document_type": {
                        "type": "string",
                        "enum": ["financial", "sec_filing", "financial_tables", "investor_relations"],
                        "description": "Type of financial document processing",
                        "default": "financial"
                    },
                    "formats": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Output formats (markdown, html)",
                        "default": ["markdown", "html"]
                    },
                    "include_selectors": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "CSS selectors for specific content extraction"
                    },
                    "max_pages": {
                        "type": "integer",
                        "description": "Maximum pages for investor relations crawling",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 15
                    }
                },
                "required": ["url"]
            }
        }


class FinancialQATool(MCPTool):
    """MCP adapter for financial Q&A using Tavily."""

    def __init__(self):
        self.client = TavilyClient()

    async def execute(self, question: str, context: str = "", **kwargs) -> Dict[str, Any]:
        """Execute financial Q&A search."""

        try:
            # Enhance question with financial context
            enhanced_question = f"{question} {context}".strip()
            result = await self.client.qna_search(enhanced_question)

            if result:
                return {
                    "success": True,
                    "question": question,
                    "answer": result,
                    "context": context,
                    "financial_optimized": True
                }

            return {
                "success": False,
                "error": "Financial Q&A search failed",
                "question": question
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Financial Q&A error: {str(e)}",
                "question": question
            }

    def get_schema(self) -> Dict[str, Any]:
        """Get MCP schema for financial Q&A."""

        return {
            "name": "financial_qa",
            "description": "Get direct answers to financial and investment banking questions",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The financial or investment banking question"
                    },
                    "context": {
                        "type": "string",
                        "description": "Additional context for the question (company name, sector, etc.)",
                        "default": ""
                    }
                },
                "required": ["question"]
            }
        }


class InvestmentBankingMCPAdapter:
    """Enhanced MCP adapter for investment banking tools."""

    def __init__(self):
        self.tools = {
            "investment_banking_search": InvestmentBankingSearchTool(),
            "financial_document_processor": FinancialDocumentTool(),
            "financial_qa": FinancialQATool()
        }

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get schemas for all available investment banking tools."""
        return [tool.get_schema() for tool in self.tools.values()]

    async def execute_tool(self, tool_name: str, **parameters) -> Dict[str, Any]:
        """Execute an investment banking tool by name with parameters."""

        if tool_name not in self.tools:
            return {
                "success": False,
                "error": f"Investment banking tool '{tool_name}' not found",
                "available_tools": list(self.tools.keys())
            }

        try:
            result = await self.tools[tool_name].execute(**parameters)
            
            # Add investment banking metadata
            if result.get("success"):
                result["investment_banking_optimized"] = True
                result["execution_timestamp"] = asyncio.get_event_loop().time()
            
            return result

        except Exception as e:
            return {
                "success": False,
                "error": f"Investment banking tool execution failed: {str(e)}",
                "tool": tool_name,
                "parameters": parameters
            }

    def validate_parameters(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate parameters against investment banking tool schema."""

        if tool_name not in self.tools:
            return {
                "valid": False,
                "error": f"Investment banking tool '{tool_name}' not found"
            }

        schema = self.tools[tool_name].get_schema()
        required_params = schema["parameters"].get("required", [])

        # Validate required parameters
        missing_params = [param for param in required_params if param not in parameters]

        if missing_params:
            return {
                "valid": False,
                "error": f"Missing required parameters: {missing_params}",
                "required": required_params,
                "provided": list(parameters.keys())
            }

        # Validate enum values
        properties = schema["parameters"].get("properties", {})
        enum_errors = []
        
        for param_name, param_value in parameters.items():
            if param_name in properties:
                param_schema = properties[param_name]
                if "enum" in param_schema and param_value not in param_schema["enum"]:
                    enum_errors.append(f"{param_name} must be one of {param_schema['enum']}")

        if enum_errors:
            return {
                "valid": False,
                "error": f"Invalid parameter values: {'; '.join(enum_errors)}"
            }

        return {"valid": True}

    def get_tool_capabilities(self) -> Dict[str, Any]:
        """Get capabilities of all investment banking tools."""
        
        capabilities = {}
        for tool_name, tool in self.tools.items():
            schema = tool.get_schema()
            capabilities[tool_name] = {
                "description": schema["description"],
                "parameters": list(schema["parameters"]["properties"].keys()),
                "required_parameters": schema["parameters"].get("required", []),
                "investment_banking_focused": True
            }
        
        return capabilities


# Global investment banking MCP adapter instance
mcp_adapter = InvestmentBankingMCPAdapter()

# Backward compatibility
MCPAdapter = InvestmentBankingMCPAdapter
