# Axiom Research Agent

Research and Web Intelligence Agent — Input‑Enriched, Evidence‑Grounded, LangGraph‑Orchestrated with DSPy Optimization, Tavily/Firecrawl Tools, SGLang Inference, and LangSmith Tracing.

## Quick Start

### 1. Install Dependencies
```bash
pip install -e .
# or for development
pip install -e ".[dev]"
# or with SGLang for local inference
pip install -e ".[dev,sglang]"
```

### 2. Environment Setup
```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Run the Agent
```bash
# Basic usage
python -m axiom.main "What are the latest developments in quantum computing?"

# Or via CLI
axiom "Research the impact of AI on software development"
```

## Architecture

- **LangGraph Orchestration**: Planner → Parallel Task Runners → Observer/Validator
- **Input Enrichment**: Multi-query expansion and HyDE via DSPy optimization
- **Evidence Gathering**: Tavily for snippet-first reasoning, Firecrawl for deep crawling
- **OpenAI-Compatible**: Works with SGLang locally or any OpenAI-compatible endpoint
- **Observability**: Full tracing via LangSmith with token/cost tracking

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

## Configuration

Key environment variables:
- `TAVILY_API_KEY`: For search API access
- `FIRECRAWL_API_KEY`: For web crawling
- `OPENAI_BASE_URL`: SGLang endpoint (default: http://localhost:30000/v1)
- `OPENAI_MODEL_NAME`: Model name for inference
- `LANGCHAIN_API_KEY`: For LangSmith tracing (optional)

## Project Structure

```
axiom/
├── graph/          # LangGraph nodes and state management
├── tools/          # Tavily/Firecrawl clients and MCP adapters
├── dspy_modules/   # Query enrichment and optimization
├── tracing/        # LangSmith integration
├── config/         # Settings and schemas
└── eval/           # Evaluation metrics and datasets
```
