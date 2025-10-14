# Axiom Investment Banking Analytics

AI-Powered Investment Banking Platform — M&A Due Diligence, Company Valuation, Market Intelligence with DSPy Optimization, LangGraph Orchestration, and Financial Data Integration.

## Quick Start

### 1. Install Dependencies
```bash
pip install -e .
# or for development
pip install -e ".[dev]"
# or with SGLang for local inference (NVIDIA systems only)
pip install -e ".[dev,sglang]"
```

### 2. Environment Setup
```bash
cp .env.example .env
# Edit .env with your API keys
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
