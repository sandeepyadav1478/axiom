# Deprecated Financial Data Providers

These providers have been **REPLACED** by external MCP servers and are no longer maintained.

## Deprecated Files (Use OpenBB MCP Instead)

**Replaced by OpenBB MCP Server**:
- ❌ `alpha_vantage_provider.py` (180 lines) → Use OpenBB MCP
- ❌ `fmp_provider.py` (200 lines) → Use OpenBB MCP  
- ❌ `finnhub_provider.py` (150 lines) → Use OpenBB MCP
- ❌ `iex_cloud_provider.py` (150 lines) → Use OpenBB MCP

**Total**: 680 lines of maintenance code eliminated

## Migration Guide

### Old Way (REST API - DEPRECATED)
```python
from axiom.integrations.data_sources.finance.alpha_vantage_provider import AlphaVantageProvider

provider = AlphaVantageProvider()
data = provider.get_quote("AAPL")
```

### New Way (OpenBB MCP - RECOMMENDED)
```python
from axiom.integrations.search_tools.mcp_adapter import use_mcp_tool

data = await use_mcp_tool(
    server_name="openbb",
    tool_name="get_quote",
    symbol="AAPL"
)
```

## Benefits of OpenBB MCP

✅ **Zero Maintenance** - Community-maintained
✅ **Auto-Updates** - Automatic bug fixes and features
✅ **More Features** - 50+ tools vs 5-10 per provider
✅ **Better Performance** - Optimized by OpenBB team
✅ **Standardized** - MCP protocol integration
✅ **Free Tier** - Most features available for free

## Status

These files are kept for **backward compatibility only** and will be removed in v2.0.0.

**Sunset Date**: 2025-12-31
**Replacement**: OpenBB MCP Server (already deployed)
