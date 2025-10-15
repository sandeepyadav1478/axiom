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

### 3. Run M&A Analytics (Priority Development Phase)
```bash
# M&A Due Diligence Analysis (Phase 1)
python -m axiom.main "M&A due diligence analysis of Microsoft acquiring OpenAI"

# M&A Valuation Analysis (Phase 1)
axiom "M&A valuation analysis for NVIDIA acquisition with synergy assessment"

# M&A Strategic Fit Analysis (Phase 1)
axiom "Strategic fit analysis for Tesla acquisition by traditional automaker"

# M&A Market Impact Analysis (Phase 1)
axiom "Market impact analysis of proposed Disney-Netflix merger"
```

## M&A-Focused Investment Banking Architecture

- **LangGraph Workflow**: M&A Planner → Parallel M&A Analysis Engines → M&A Validator
- **DSPy Optimization**: M&A-specific query optimization for due diligence, valuation, and strategic analysis
- **Data Integration**: Tavily for M&A market intelligence, Firecrawl for SEC filings and acquisition reports
- **Multi-AI System**: Claude for M&A reasoning, OpenAI for structured analysis, SGLang for local inference
- **M&A Audit Trail**: Complete deal tracing for regulatory compliance and acquisition decision auditing

### Phase 1 Priority: M&A Analytics
**Current Focus**: Mergers & Acquisitions analysis automation
**Future Phases**: General investment banking, IPO analysis, restructuring

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
