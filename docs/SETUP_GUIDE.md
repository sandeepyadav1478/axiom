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

### Quantitative Finance Configuration

#### Configuration System Overview
Axiom provides 47+ configuration parameters across all financial models:
- **VaR Models**: Confidence levels, methods, simulation counts
- **Time Series**: ARIMA orders, GARCH parameters, EWMA decay factors
- **Portfolio**: Risk-free rates, constraints, optimization methods
- **Credit Risk**: Basel III parameters, PD/LGD/EAD settings
- **Options**: Monte Carlo paths, Greeks calculation, IV solvers

#### Configuration File Setup

**Step 1: Create configuration file**
```bash
# Create config directory
mkdir -p ~/.axiom

# Create custom configuration
cat > ~/.axiom/config.json << 'EOF'
{
  "var": {
    "default_confidence_level": 0.99,
    "default_method": "monte_carlo",
    "default_simulations": 50000,
    "parallel_mc": true
  },
  "time_series": {
    "ewma_decay_factor": 0.94,
    "forecast_horizon": 5,
    "arima_auto_select": true
  },
  "portfolio": {
    "default_risk_free_rate": 0.03,
    "optimization_method": "max_sharpe",
    "long_only": true
  }
}
EOF
```

**Step 2: Load configuration in code**
```python
from axiom.config.model_config import ModelConfig

# Load from file
config = ModelConfig.from_file("~/.axiom/config.json")

# Or use presets
config = ModelConfig.for_basel_iii_compliance()  # Regulatory compliance
config = ModelConfig.for_high_performance()      # Speed-optimized
config = ModelConfig.for_high_precision()        # Accuracy-optimized
```

#### Environment Variable Overrides

**Add to your `.env` file:**
```env
# VaR Configuration
VAR_CONFIDENCE=0.99
VAR_METHOD=historical
VAR_MIN_OBS=252

# Time Series Configuration
TS_EWMA_LAMBDA=0.94
TS_FORECAST_HORIZON=5

# Portfolio Configuration
PORTFOLIO_RISK_FREE_RATE=0.03
PORTFOLIO_METHOD=max_sharpe
PORTFOLIO_LONG_ONLY=true

# Credit Risk Configuration
CREDIT_BASEL_CONFIDENCE=0.999
CREDIT_DOWNTURN_MULTIPLIER=1.25
CREDIT_CAPITAL_APPROACH=ADVANCED_IRB
```

**Load from environment:**
```python
from axiom.config.model_config import ModelConfig

# Automatically loads from environment variables
config = ModelConfig.from_env()
```

#### Custom Configuration Examples

**Example 1: High-Frequency Trading**
```python
from axiom.config.model_config import TimeSeriesConfig, VaRConfig

# Intraday trading configuration
config = TimeSeriesConfig.for_intraday_trading()
# - EWMA decay: 0.99 (very reactive)
# - Min periods: 10
# - Forecast horizon: 1

# Fast VaR for real-time risk
var_config = VaRConfig(
    default_method="parametric",  # Fastest method
    cache_results=True,
    parallel_mc=False  # Not needed for parametric
)
```

**Example 2: Basel III Compliance**
```python
from axiom.config.model_config import ModelConfig

# Full Basel III configuration
config = ModelConfig.for_basel_iii_compliance()
# - VaR confidence: 99.9%
# - Time horizon: 10 days
# - Min observations: 252 (1 year)
# - Credit downturn LGD: 1.25x
# - Capital approach: Advanced IRB
```

**Example 3: Backtesting & Research**
```python
from axiom.config.model_config import ModelConfig

# High precision for research
config = ModelConfig.for_high_precision()
# - Monte Carlo: 100K paths
# - Binomial steps: 500
# - Black-Scholes precision: 1e-10
# - Min observations: 500
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