"""Document Processing MCP Servers.

Provides document processing capabilities through MCP:
- PDF extraction and parsing
- Excel/spreadsheet operations
- Word document processing
- Markdown conversion
"""

from typing import Any

__all__ = ["pdf_server", "excel_server"]


def get_available_servers() -> list[str]:
    """Get list of available document servers.
    
    Returns:
        List of server names
    """
    return ["pdf", "excel"]