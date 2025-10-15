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

### 1. Activate Virtual Environment & Install
```bash
# IMPORTANT: Activate the virtual environment first
source .venv/bin/activate

# Verify installation
python tests/validate_system.py
# Expected: 7/7 validations passed ✅

# Test core M&A functionality
python demo_ma_analysis.py
# Expected: 5/5 demos successful ✅
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

## Investment Banking Platform Structure

```
axiom/
├── graph/          # Financial analysis workflows and decision trees
├── tools/          # Market data, SEC filings, and financial news integration
├── dspy_modules/   # Financial query optimization and valuation models
├── tracing/        # Audit trails and compliance tracking
├── config/         # Financial data sources and analysis parameters
└── eval/           # Investment decision accuracy and performance metrics
```
