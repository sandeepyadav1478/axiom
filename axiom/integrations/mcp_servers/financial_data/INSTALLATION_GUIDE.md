# 🚀 Axiom Financial MCP Servers - Installation Guide

This guide shows how to install and configure **multiple professional financial MCP servers** for Axiom without modifying your global MCP settings.

## 📊 **Available Financial MCP Servers:**

### 1. 🏆 **Official Polygon.io MCP Server**
- **Provider**: Official Polygon.io company
- **Cost**: FREE tier (5 calls/minute) + $25/month premium
- **Features**: Real-time market data, options, crypto, forex
- **Repository**: https://github.com/polygon-io/mcp_polygon

### 2. 🚀 **Professional Yahoo Finance MCP Server**  
- **Provider**: gregorizeidler/MCP-yahoofinance-ai
- **Cost**: 100% FREE (no API key needed)
- **Features**: 27 professional tools including portfolio management, risk analysis, technical indicators
- **Repository**: https://github.com/gregorizeidler/MCP-yahoofinance-ai

### 3. 📈 **Comprehensive Yahoo Finance MCP Server**
- **Provider**: Alex2Yang97/yahoo-finance-mcp
- **Cost**: 100% FREE (no API key needed)  
- **Features**: Complete fundamental analysis and research tools
- **Repository**: https://github.com/Alex2Yang97/yahoo-finance-mcp

## 🔧 **Installation Options:**

### **Option 1: Official uvx Installation (Recommended)**

This uses the official installation method recommended by each MCP server:

#### **Install Polygon.io MCP Server:**
```bash
# Get FREE API key from: https://polygon.io/
export POLYGON_API_KEY="your_polygon_api_key_here"

# Install using official uvx method
uvx --from git+https://github.com/polygon-io/mcp_polygon@v0.5.1 mcp_polygon
```

#### **Install Yahoo Finance Professional MCP Server:**
```bash
# Clone and install (100% FREE - no API key needed)
git clone https://github.com/gregorizeidler/MCP-yahoofinance-ai.git
cd MCP-yahoofinance-ai
pip install -r requirements.txt

# Add to your MCP client configuration
```

#### **Install Yahoo Finance Comprehensive MCP Server:**
```bash
# Clone and install using uv (100% FREE)
git clone https://github.com/Alex2Yang97/yahoo-finance-mcp.git
cd yahoo-finance-mcp
uv venv && source .venv/bin/activate
uv pip install -e .
```

### **Option 2: Project-Level MCP Configuration**

Use the provided project-level configuration without modifying your global settings:

#### **1. Copy Project MCP Settings:**
```bash
# Copy our project MCP settings as a reference
cp axiom/integrations/mcp_servers/financial_data/project_mcp_settings.json ~/axiom_financial_mcp_settings.json

# Edit the file to add your API keys:
# - Replace "your_polygon_api_key_here" with real Polygon.io key
# - Update HOME directory path for your system
```

#### **2. Add to Your MCP Client (Optional):**

**For Claude Desktop:**
Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "polygon-io": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/polygon-io/mcp_polygon@v0.5.1", "mcp_polygon"],
      "env": {"POLYGON_API_KEY": "your_polygon_key", "HOME": "/Users/your_username"}
    }
  }
}
```

**For VS Code with Roo:**
Add financial servers to your MCP settings file (manually, when desired)

### **Option 3: Docker External Reference (Development)**

For development and testing with always-updated external repositories:

```bash
# Navigate to financial MCP directory
cd axiom/integrations/mcp_servers/financial_data/

# Use external reference docker-compose  
docker-compose -f external_mcp_servers.yml up --build -d

# This builds from external GitHub repos (always latest)
```

## 🔑 **API Key Setup:**

### **Polygon.io API Key (Optional - FREE tier available):**
1. Visit: https://polygon.io/
2. Sign up for **FREE account** (5 calls/minute)
3. Get API key from dashboard
4. Add to environment: `export POLYGON_API_KEY="your_key"`

### **API Key Rotation for FREE Tier Optimization:**
```bash
# Create multiple FREE Polygon.io accounts for rotation
export POLYGON_API_KEY_1="free_account_1_key"
export POLYGON_API_KEY_2="free_account_2_key"  
export POLYGON_API_KEY_3="free_account_3_key"

# Total capacity: 3 accounts × 5 calls/minute = 15 calls/minute FREE
```

### **Yahoo Finance Servers:**
- **No API keys required** - 100% FREE unlimited access!

## 🎯 **Usage Examples:**

Once installed, you can use these MCP financial tools:

### **Polygon.io Tools (if API key configured):**
```
"Get real-time market data for AAPL using Polygon.io"
"Show me the latest options chains for Tesla" 
"Get crypto market data for BTC-USD"
```

### **Yahoo Finance Professional Tools (100% FREE):**
```
"Create a portfolio with 40% Apple, 30% Microsoft, 20% Google, 10% Tesla and analyze risk metrics"
"Calculate VaR and Sharpe ratio for my tech portfolio"
"Generate technical analysis with RSI and MACD for NVIDIA"
"Show me sector rotation analysis with ETF performance"
```

### **Yahoo Finance Comprehensive Tools (100% FREE):**  
```
"Get Apple's quarterly income statement"
"Show me Tesla's institutional holders"
"Get options expiration dates for Microsoft"
"Analyze recent news for Amazon stock"
```

## 💡 **Recommended Setup:**

### **For Most Users (100% FREE):**
1. Use existing yahoo-finance-comprehensive MCP server (already working!)
2. No additional setup needed - already tested and functional

### **For Enhanced Capabilities:**
1. Install **Official Polygon.io MCP** (FREE tier + optional premium)
2. Install **Professional Yahoo Finance MCP** (27 advanced tools)
3. Total cost: $0/month FREE or $25/month with Polygon.io premium

### **Cost Comparison:**
- **FREE Setup**: $0/month (Yahoo Finance MCP servers only)
- **Enhanced Setup**: $25/month (+ Polygon.io premium)
- **Bloomberg Terminal**: $2,000/month 
- **Savings**: 98%+ cost reduction with professional capabilities

## 📋 **Maintenance:**

### **Updates:**
- **uvx installations**: Automatically update from GitHub on reinstall
- **External Docker builds**: Always pull latest code on rebuild
- **No embedded repos**: No maintenance burden on Axiom project

### **Clean Architecture Benefits:**
✅ No external repositories embedded in Axiom project  
✅ Always uses latest versions from upstream  
✅ Easy updates and security patches  
✅ Clean separation of concerns  
✅ Professional MCP server capabilities maintained  

---

**🎯 Result: World-class financial data access with clean, maintainable architecture!**