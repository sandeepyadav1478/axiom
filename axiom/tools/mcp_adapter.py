"""MCP (Model Context Protocol) adapter for tool integration."""

import asyncio
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod

from axiom.tools.tavily_client import TavilyClient
from axiom.tools.firecrawl_client import FirecrawlClient


class MCPTool(ABC):
    """Abstract base class for MCP-compatible tools."""

    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool with given parameters."""
        pass

    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """Get the tool schema for MCP."""
        pass


class TavilySearchTool(MCPTool):
    """MCP adapter for Tavily search."""

    def __init__(self):
        self.client = TavilyClient()

    async def execute(self, query: str, max_results: int = 10, **kwargs) -> Dict[str, Any]:
        """Execute Tavily search."""

        result = await self.client.search(query, max_results)

        if result:
            return {
                "success": True,
                "results": result.get("results", []),
                "answer": result.get("answer", ""),
                "query": query
            }

        return {
            "success": False,
            "error": "Search failed",
            "query": query
        }

    def get_schema(self) -> Dict[str, Any]:
        """Get MCP schema for Tavily search."""

        return {
            "name": "tavily_search",
            "description": "Search the web using Tavily API for high-quality, real-time results",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 20
                    }
                },
                "required": ["query"]
            }
        }


class FirecrawlScrapeTool(MCPTool):
    """MCP adapter for Firecrawl scraping."""

    def __init__(self):
        self.client = FirecrawlClient()

    async def execute(self, url: str, formats: List[str] = None, **kwargs) -> Dict[str, Any]:
        """Execute Firecrawl scrape."""

        if formats is None:
            formats = ["markdown"]

        result = await self.client.scrape(url, formats=formats)

        if result:
            return {
                "success": True,
                "url": url,
                "content": result.get("content", ""),
                "metadata": result.get("metadata", {}),
                "formats": formats
            }

        return {
            "success": False,
            "error": "Scraping failed",
            "url": url
        }

    def get_schema(self) -> Dict[str, Any]:
        """Get MCP schema for Firecrawl scrape."""

        return {
            "name": "firecrawl_scrape",
            "description": "Scrape content from a URL using Firecrawl API",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to scrape",
                        "format": "uri"
                    },
                    "formats": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Output formats (markdown, html, etc.)",
                        "default": ["markdown"]
                    }
                },
                "required": ["url"]
            }
        }


class MCPAdapter:
    """Main MCP adapter for Axiom tools."""

    def __init__(self):
        self.tools = {
            "tavily_search": TavilySearchTool(),
            "firecrawl_scrape": FirecrawlScrapeTool()
        }

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get schemas for all available tools."""

        return [tool.get_schema() for tool in self.tools.values()]

    async def execute_tool(self, tool_name: str, **parameters) -> Dict[str, Any]:
        """Execute a tool by name with parameters."""

        if tool_name not in self.tools:
            return {
                "success": False,
                "error": f"Tool '{tool_name}' not found",
                "available_tools": list(self.tools.keys())
            }

        try:
            result = await self.tools[tool_name].execute(**parameters)
            return result

        except Exception as e:
            return {
                "success": False,
                "error": f"Tool execution failed: {str(e)}",
                "tool": tool_name,
                "parameters": parameters
            }

    def validate_parameters(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate parameters against tool schema."""

        if tool_name not in self.tools:
            return {
                "valid": False,
                "error": f"Tool '{tool_name}' not found"
            }

        schema = self.tools[tool_name].get_schema()
        required_params = schema["parameters"].get("required", [])

        # Basic validation - check required parameters
        missing_params = [param for param in required_params if param not in parameters]

        if missing_params:
            return {
                "valid": False,
                "error": f"Missing required parameters: {missing_params}"
            }

        return {"valid": True}


# Global MCP adapter instance
mcp_adapter = MCPAdapter()
