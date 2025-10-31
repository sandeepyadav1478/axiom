# MCP Servers - Simple Explanation

## 🤔 What Are These MCP Servers For?

Each MCP server exposes a specific capability of the Axiom platform to external clients (like Claude Desktop, Cline, or any AI assistant).

Think of them as **specialized assistants** that Claude Desktop can call upon for specific tasks.

## 📋 The 12 MCP Servers - What Each One Does

### 🎯 TRADING CLUSTER (5 servers)

#### 1. pricing-greeks-mcp
**What it does**: Calculate option prices and Greeks (delta, gamma, vega, theta, rho)  
**Why you need it**: When you ask Claude "What's the delta of this call option?", it calls this server  
**Example**: "Calculate Greeks for a $100 strike call option expiring in 30 days"

#### 2. portfolio-risk-mcp
**What it does**: Calculate total portfolio risk using multiple VaR methods  
**Why you need it**: When you ask "What's my portfolio risk?" or "Am I within risk limits?"  
**Example**: "Calculate VaR for my portfolio of 100 option contracts"

#### 3. strategy-gen-mcp
**What it does**: Generate optimal trading strategies using AI  
**Why you need it**: When you ask "What's the best options strategy for a bullish outlook?"  
**Example**: "Suggest a strategy for bullish market with increasing volatility"

#### 4. execution-mcp
**What it does**: Smart order routing - finds best execution venue  
**Why you need it**: When you want to execute a trade with best price  
**Example**: "Where should I route this order for best execution?"

#### 5. hedging-mcp
**What it does**: Calculate optimal hedge using Deep RL  
**Why you need it**: When you ask "How should I hedge my delta exposure?"  
**Example**: "Calculate optimal hedge for my 500 delta position"

---

### 📊 ANALYTICS CLUSTER (3 servers)

#### 6. performance-mcp
**What it does**: Calculate P&L with Greeks attribution  
**Why you need it**: When you ask "What's my P&L today?" or "Where did my profit come from?"  
**Example**: "Calculate my P&L and break it down by delta, gamma, vega"

#### 7. market-data-mcp
**What it does**: Aggregate market data from multiple sources (NBBO)  
**Why you need it**: When you ask "What's the current quote for SPY options?"  
**Example**: "Get real-time quotes for SPY 100 strike call"

#### 8. volatility-mcp
**What it does**: Forecast volatility using AI (Transformer+GARCH+LSTM)  
**Why you need it**: When you ask "What will volatility be tomorrow?"  
**Example**: "Forecast 1-week volatility for SPY"

---

### 🛡️ SUPPORT CLUSTER (4 servers)

#### 9. regulatory-mcp
**What it does**: Check regulatory compliance (SEC, FINRA, MiFID II, EMIR)  
**Why you need it**: When you ask "Is this trade compliant?" or "Generate LOPR report"  
**Example**: "Check if my position is compliant with position limits"

#### 10. system-health-mcp
**What it does**: Monitor all agents and system health  
**Why you need it**: When you ask "Is the system healthy?" or "Are all agents working?"  
**Example**: "Check system health and show any issues"

#### 11. guardrails-mcp
**What it does**: AI safety validation (veto authority)  
**Why you need it**: When you ask "Is this action safe?" - validates all AI outputs  
**Example**: "Validate these Greeks before I use them in production"

#### 12. interface-mcp
**What it does**: Orchestrates all other MCPs for complex queries  
**Why you need it**: When you ask complex questions that need multiple MCPs  
**Example**: "Analyze my portfolio, suggest strategies, and check compliance"

---

## 💡 Real-World Usage Examples

### Example 1: "I want to trade SPY options"
Claude Desktop would use:
1. **market-data-mcp** → Get current SPY quotes
2. **pricing-greeks-mcp** → Calculate Greeks for different strikes
3. **strategy-gen-mcp** → Suggest optimal strategy
4. **guardrails-mcp** → Validate the strategy is safe
5. **execution-mcp** → Route order to best venue
6. **regulatory-mcp** → Ensure compliance

### Example 2: "How risky is my portfolio?"
Claude Desktop would use:
1. **portfolio-risk-mcp** → Calculate VaR and Greeks
2. **volatility-mcp** → Forecast future volatility
3. **system-health-mcp** → Ensure risk engine is healthy
4. **interface-mcp** → Present results in dashboard format

### Example 3: "What's my P&L?"
Claude Desktop would use:
1. **performance-mcp** → Calculate total P&L
2. **market-data-mcp** → Get current market prices
3. **interface-mcp** → Format nice report

---

## 🤝 How They Work Together

```
┌─────────────────┐
│ You Ask Claude  │
│ "Trade question"│
└────────┬────────┘
         │
         v
┌─────────────────────┐
│  Claude Desktop     │
│  (Uses MCP)         │
└────────┬────────────┘
         │
         ├──→ pricing-greeks-mcp (calculates Greeks)
         ├──→ market-data-mcp (gets quotes)
         ├──→ strategy-gen-mcp (suggests strategy)
         ├──→ execution-mcp (finds best venue)
         ├──→ guardrails-mcp (validates safety)
         └──→ regulatory-mcp (checks compliance)
         │
         v
┌─────────────────────┐
│  Complete Answer    │
│  with all details   │
└─────────────────────┘
```

---

## 🎯 Why 12 Separate Servers?

### Microservice Benefits:
1. **Isolation**: One server crash doesn't affect others
2. **Scalability**: Can scale each server independently
3. **Clarity**: Each server has ONE job (Single Responsibility)
4. **Testing**: Easy to test each capability separately
5. **Deployment**: Can update one server without touching others

### Industry Standard:
- Bloomberg has separate servers for pricing, risk, analytics
- Trading firms have microservices for each function
- This is how modern financial systems are built

---

## 🔍 Which Servers Are Most Important?

### Must-Have (Core Trading):
1. **pricing-greeks-mcp** - Can't trade options without Greeks ⭐⭐⭐
2. **portfolio-risk-mcp** - Must know your risk ⭐⭐⭐
3. **execution-mcp** - Need to execute trades ⭐⭐⭐

### Very Important (Safety & Compliance):
4. **guardrails-mcp** - Safety validation ⭐⭐
5. **regulatory-mcp** - Legal compliance ⭐⭐

### Nice to Have (Enhanced Features):
6. **strategy-gen-mcp** - AI strategy suggestions ⭐
7. **volatility-mcp** - Volatility forecasting ⭐
8. **performance-mcp** - P&L tracking ⭐
9. **market-data-mcp** - Quote aggregation ⭐
10. **system-health-mcp** - Monitoring ⭐
11. **hedging-mcp** - Auto-hedging ⭐
12. **interface-mcp** - Orchestration ⭐

---

## 🎓 Technical Summary

**Purpose**: Expose Axiom's capabilities via MCP protocol  
**Clients**: Claude Desktop, Cline, any MCP-compatible tool  
**Protocol**: Model Context Protocol (MCP) v1.0.0  
**Transport**: STDIO (standard input/output)  
**Status**: All 12 operational and tested  

---

## ✅ Bottom Line

**These 12 servers turn Axiom into a complete AI-powered trading platform that any AI assistant (Claude, etc.) can use via simple chat!**

Instead of manually calculating Greeks, checking risk, generating strategies - just ask Claude and it uses these servers to do the work.

**Think of it like giving Claude Desktop 12 specialized expert assistants:**
- A pricing expert
- A risk analyst  
- A strategy designer
- An execution trader
- A compliance officer
- And more!

---

**Questions? Each server's purpose is documented in its README.md file in its directory.**