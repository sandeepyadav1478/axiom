# Axiom Investment Banking Analytics - Setup Guide

## Quick Start for Investment Banking Analytics

### Prerequisites
- Python 3.10+ 
- Git
- API keys for chosen AI providers

### Installation
```bash
# Clone repository
git clone <repository-url>
cd axiom

# Run installation script
chmod +x install.sh
./install.sh

# Choose option 1 for development setup
```

### Configure AI Providers
```bash
# Edit .env file with your API keys
# Uncomment the providers you want to use:

# For Claude (recommended for investment banking reasoning)
CLAUDE_API_KEY=sk-ant-your_claude_api_key

# For OpenAI (good for structured analysis)  
OPENAI_API_KEY=sk-your_openai_api_key

# For local inference on NVIDIA systems
SGLANG_BASE_URL=http://localhost:30000/v1
```

### Test Investment Banking Analysis
```bash
# Activate environment
source .venv/bin/activate

# Run M&A analysis
python -m axiom.main "Analyze Tesla acquisition potential for strategic value"

# Run due diligence
python -m axiom.main "Comprehensive due diligence analysis of Apple financial health"
```

### AI Provider Layer Configuration
The system intelligently routes analysis types to optimal AI providers:
- **Due Diligence** → Claude (superior reasoning)
- **Valuation** → OpenAI (structured analysis)  
- **Market Intelligence** → Claude (comprehensive synthesis)

Users can override these assignments in `axiom/config/ai_layer_config.py`

### Development Workflow
```bash
# Run tests
pytest

# Format code
black . && ruff check --fix .

# Run evaluation
python -m axiom.eval.run_eval
```

## Investment Banking Use Cases
- M&A due diligence automation
- Company valuation analysis
- Market intelligence gathering
- Risk assessment workflows
- Competitive analysis
- Regulatory compliance analysis

Ready for professional Investment Banking Analytics!