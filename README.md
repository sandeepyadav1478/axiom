# Axiom Investment Banking Analytics

AI-Powered Investment Banking Platform — M&A Due Diligence, Company Valuation, Market Intelligence with DSPy Optimization, LangGraph Orchestration, and Financial Data Integration.

## 🎉 Status: Core Implementation Complete!

✅ **Multi-AI Provider System**: OpenAI, Claude, SGLang with intelligent routing
✅ **Investment Banking Workflows**: M&A due diligence, valuation, strategic analysis
✅ **Financial Data Integration**: Enhanced Tavily + Firecrawl for SEC filings
✅ **DSPy Optimization**: Investment banking query enrichment and optimization
✅ **Comprehensive Validation**: Financial metrics, compliance, data quality
✅ **Error Handling**: Investment banking grade error management

## 🚀 Quick Start

### 1. Quick Setup (Auto-Activation ⚡)
```bash
# Option 1: Automatic setup with pyenv + direnv (Recommended)
./scripts/setup-development-environment.sh
# Sets up Python 3.13, auto-activation, and dependencies

# Option 2: Manual setup
source .venv/bin/activate

# Verify installation
python tests/validate_system.py
# Expected: 7/7 validations passed ✅

# Test core M&A functionality
python demos/demo_ma_analysis.py
# Expected: 5/5 demos successful ✅
```

### 🐍 Auto-Activation Features
```bash
# After setup, environment auto-activates when entering directory
cd axiom/  # Automatically activates Python 3.13 + dependencies
python --version  # Shows: Python 3.13.7
# No more manual: source .venv/bin/activate
```

### 2. Environment Setup
```bash
# Configure API keys
cp .env.example .env
# Edit .env with your API keys (minimum required):
# TAVILY_API_KEY=your_tavily_api_key_here
# FIRECRAWL_API_KEY=your_firecrawl_api_key_here
# OPENAI_API_KEY=sk-your_openai_key_here  # OR Claude
```

### 3. Run Investment Analytics
```bash
# M&A Due Diligence Analysis
python -m axiom.main "Analyze Microsoft's acquisition potential for OpenAI"

# Company Valuation Analysis
axiom "Comprehensive financial analysis of NVIDIA for IPO readiness"

# Market Intelligence
axiom "Investment banking analysis of the AI infrastructure market"
```

## Investment Banking Architecture

- **LangGraph Workflow**: Financial Planner → Parallel Analysis Engines → Investment Validator
- **DSPy Optimization**: Multi-source financial query optimization and valuation model enhancement
- **Data Integration**: Tavily for market intelligence, Firecrawl for SEC filings and financial reports
- **AI Infrastructure**: SGLang for structured financial analysis or cloud AI endpoints
- **Audit Trail**: Complete LangSmith tracing for regulatory compliance and decision auditing

## Development

### Running Tests
```bash
pytest
```

### Code Formatting
```bash
black .
ruff check --fix .
```

### Evaluation
```bash
python -m axiom.eval.run_eval
```

## Financial Data Configuration

Key environment variables:
- `TAVILY_API_KEY`: For market intelligence and news analysis
- `FIRECRAWL_API_KEY`: For SEC filings and financial reports extraction
- `OPENAI_BASE_URL`: AI inference endpoint (default: https://api.openai.com/v1)
- `OPENAI_MODEL_NAME`: Financial analysis model (default: gpt-4o-mini)
- `LANGCHAIN_API_KEY`: For audit trails and compliance tracing (recommended)

## 📚 Comprehensive Documentation

### **Complete M&A Workflow System**
- 📋 **[M&A Workflows Documentation](docs/README.md)** - Complete documentation index and navigation
- 🎯 **[M&A Workflow Guide](docs/ma-workflows/M&A_WORKFLOW_GUIDE.md)** - Usage examples and API documentation
- 🏗️ **[M&A System Architecture](docs/ma-workflows/M&A_SYSTEM_OVERVIEW.md)** - Technical architecture and deployment
- 💼 **[Business Rationale](docs/ma-workflows/M&A_WORKFLOWS_BUSINESS_RATIONALE.md)** - Why each M&A workflow is essential

### **GitHub Actions for M&A Operations**
- 🚀 **[M&A Workflow Execution Guide](docs/ma-workflows/M&A_WORKFLOW_EXECUTION_GUIDE.md)** - How to trigger M&A workflows
- ⚙️ **[GitHub Actions Architecture](docs/architecture/WHY_GITHUB_ACTIONS_FOR_MA.md)** - Strategic rationale for GitHub-based M&A automation

### **🔮 Future Deployment (AWS Free Tier)**
- 💡 **[AWS Deployment Planning](docs/deployment/README.md)** - Cost-free AWS Lambda/EC2 migration guide (planned implementation)

## Investment Banking Platform Structure

```
axiom/
├── workflows/      # 🎯 M&A lifecycle workflows (target screening, DD, valuation)
├── graph/          # 🔄 Financial analysis workflows and decision trees
├── tools/          # 🔍 Market data, SEC filings, and financial news integration
├── dspy_modules/   # 🤖 Financial query optimization and valuation models
├── tracing/        # 📋 Audit trails and compliance tracking
├── config/         # ⚙️ Financial data sources and analysis parameters
├── utils/          # 🛠️ Validation, error handling, compliance frameworks
└── eval/           # 📊 Investment decision accuracy and performance metrics

demos/              # 🎮 Demo files and examples
├── demo_complete_ma_workflow.py     # Complete M&A workflow demonstration
├── demo_enhanced_ma_workflows.py    # Enhanced M&A workflows
├── demo_ma_analysis.py              # M&A analysis examples
├── simple_demo.py                   # Quick start demo
└── README.md                        # Demo documentation

guides/             # 📚 Setup and configuration guides
├── FINANCIAL_MCP_SERVERS_GUIDE.md   # Financial MCP servers setup
├── INSTALLATION_GUIDE.md            # Installation instructions
└── README.md                        # Guides documentation

docs/
├── ma-workflows/   # 💼 M&A workflow documentation and guides
├── architecture/   # 🏗️ System architecture and design rationale
└── deployment/     # 🚀 Deployment guides and AWS planning

.github/workflows/
├── ma-deal-pipeline.yml         # 🏦 Complete M&A deal execution automation
├── ma-risk-assessment.yml       # ⚠️ Risk management and regulatory compliance
├── ma-valuation-validation.yml  # 💎 Financial model validation and stress testing
└── ma-deal-management.yml       # 📊 Executive portfolio oversight and coordination
```




