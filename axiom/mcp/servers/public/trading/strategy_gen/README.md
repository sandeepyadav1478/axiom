# Strategy Generation MCP Server

**AI-powered trading strategy generation using Reinforcement Learning.**

## 🎯 Overview

- **Performance:** <100ms strategy generation
- **AI Model:** Reinforcement Learning for optimal selection
- **Strategies:** 25+ types (directional, volatility, income)
- **Protocol:** MCP 1.0.0 compliant
- **Transport:** STDIO (Claude Desktop), HTTP, SSE

## 🔧 Installation

```bash
pip install -r requirements.txt
```

## 🚀 Usage

### With Claude Desktop

```json
{
  "mcpServers": {
    "strategy-generation": {
      "command": "python",
      "args": ["-m", "axiom.mcp_servers.trading.strategy_gen.server"]
    }
  }
}
```

## 📚 Tools

### `generate_strategy`
Generate optimal trading strategy based on market outlook.

**Input:**
```json
{
  "market_outlook": "bullish",
  "volatility_view": "stable",
  "risk_tolerance": 0.6,
  "capital_available": 50000.0,
  "current_spot": 100.0,
  "current_vol": 0.25
}
```

**Output:**
```json
{
  "success": true,
  "strategy": {
    "name": "bull_call_spread",
    "legs": [...],
    "entry_cost": 1700,
    "max_profit": 3300,
    "max_loss": 1700,
    "expected_return": 1890,
    "probability_profit": 0.65
  },
  "rationale": "Bullish spread with defined risk...",
  "confidence": 0.75
}
```

### `backtest_strategy`
Backtest strategy on historical data.

### `optimize_strategy`
Optimize strategy parameters for best performance.

## 📊 Resources

- `strategy://templates` - 25+ strategy templates
- `strategy://history` - Generated strategies
- `strategy://performance` - Backtest results

## 💬 Prompts

- `explain_strategy` - Explain strategy mechanics
- `strategy_help` - Get strategy advice

---

**AI-powered strategy generation with 60%+ win rate.**