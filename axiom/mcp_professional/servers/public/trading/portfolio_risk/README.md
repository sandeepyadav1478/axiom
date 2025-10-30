# Portfolio Risk MCP Server

**Industry-standard MCP server for real-time portfolio risk management.**

## 🎯 Overview

Provides comprehensive portfolio risk analysis via Model Context Protocol (MCP).

- **Performance:** <5ms complete risk (1000+ positions)
- **Methods:** Parametric VaR, Historical VaR, Monte Carlo VaR
- **Approach:** Conservative (better to overestimate risk)
- **Protocol:** MCP 1.0.0 compliant
- **Transport:** STDIO (Claude Desktop), HTTP, SSE

## 🔧 Installation

```bash
pip install -r requirements.txt
```

## 🚀 Usage

### With Claude Desktop

Add to config:

```json
{
  "mcpServers": {
    "portfolio-risk": {
      "command": "python",
      "args": ["-m", "axiom.mcp_servers.trading.portfolio_risk.server"]
    }
  }
}
```

## 📚 Tools

### `calculate_risk`
Calculate complete portfolio risk with multiple VaR methods.

**Input:**
```json
{
  "positions": [
    {"strike": 100, "time_to_maturity": 0.25, "quantity": 100, "entry_price": 5.0}
  ],
  "market_data": {"spot": 100.0, "vol": 0.25, "rate": 0.03}
}
```

**Output:**
```json
{
  "success": true,
  "risk_metrics": {
    "total_delta": 2500.0,
    "total_gamma": 150.0,
    "var_1day_parametric": 125000,
    "var_1day_historical": 130000,
    "var_1day_monte_carlo": 132500,
    "cvar_1day": 175000
  },
  "within_limits": true,
  "limit_breaches": [],
  "calculation_time_ms": 4.2
}
```

### `stress_test`
Run stress tests (market crash scenarios).

### `check_limits`
Check if portfolio is within risk limits.

## 📊 Resources

- `risk://metrics` - Current risk metrics
- `risk://limits` - Risk limit configuration
- `risk://history` - Historical risk data

## 💬 Prompts

- `explain_risk` - Explain portfolio risk
- `var_explanation` - Explain VaR calculations

## 🐳 Docker

```bash
docker build -t portfolio-risk-mcp-server .
docker run -i portfolio-risk-mcp-server
```

## 📈 Performance

- **Complete risk:** <5ms
- **Stress test:** <100ms
- **VaR methods:** 3 (cross-validation)
- **Approach:** Conservative

---

**Industry-standard MCP server compatible with Claude Desktop and any MCP client.**