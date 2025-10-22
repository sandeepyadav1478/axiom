# Axiom Investment Banking Analytics - Project Structure
## Modern Directory Organization (Post-Restructuring)

### 🏗️ New Directory Structure

```
axiom/
├── core/                                    # 🎯 Core Business Logic
│   ├── __init__.py                         # Core module exports
│   ├── api_management/                     # 🔄 API key rotation & quota management
│   │   └── __init__.py                     
│   ├── orchestration/                      # 🔄 LangGraph workflow orchestration
│   │   ├── __init__.py                     # Orchestration exports
│   │   ├── graph.py                        # Main LangGraph workflow
│   │   ├── state.py                        # State management
│   │   └── nodes/                          # Workflow nodes
│   │       ├── __init__.py
│   │       ├── observer.py                 # Investment banking observer
│   │       ├── planner.py                  # M&A analysis planner
│   │       └── task_runner.py              # Parallel task execution
│   ├── analysis_engines/                   # 📊 M&A Analysis Workflows
│   │   ├── __init__.py                     
│   │   ├── due_diligence.py                # Financial, Commercial, Operational DD
│   │   ├── target_screening.py             # M&A target identification
│   │   ├── valuation.py                    # DCF, Comparables, Synergies
│   │   ├── advanced_modeling.py            # Monte Carlo, Stress testing
│   │   ├── risk_assessment.py              # Multi-dimensional risk analysis
│   │   ├── regulatory_compliance.py        # HSR filing, Antitrust analysis
│   │   ├── market_intelligence.py          # Competitive analysis
│   │   ├── executive_dashboards.py         # Portfolio KPIs, Executive reporting
│   │   ├── esg_analysis.py                 # Environmental, Social, Governance
│   │   ├── pmi_planning.py                 # Post-merger integration
│   │   ├── deal_execution.py               # Contract analysis, Closing coordination
│   │   └── cross_border_ma.py              # International M&A, Currency hedging
│   └── validation/                         # ✅ Data validation & compliance
│       ├── __init__.py
│       ├── error_handling.py               # Investment banking error classes
│       └── validation.py                   # Financial data validation
│
├── integrations/                           # 🔌 External Service Integrations
│   ├── __init__.py                         # Integration module exports
│   ├── ai_providers/                       # 🤖 AI provider abstractions
│   │   ├── __init__.py                     # AI provider exports
│   │   ├── base_ai_provider.py             # Abstract base class
│   │   ├── claude_provider.py              # Claude integration
│   │   ├── openai_provider.py              # OpenAI integration
│   │   ├── sglang_provider.py              # SGLang local inference
│   │   └── provider_factory.py             # Provider management
│   ├── data_sources/                       # 📊 Financial data providers
│   │   ├── __init__.py
│   │   └── finance/                        # Financial data sources
│   │       ├── __init__.py
│   │       ├── base_financial_provider.py  # Financial provider abstraction
│   │       ├── openbb_provider.py          # OpenBB integration (FREE)
│   │       ├── alpha_vantage_provider.py   # Alpha Vantage ($49/month)
│   │       ├── sec_edgar_provider.py       # SEC Edgar (FREE government data)
│   │       ├── bloomberg_provider.py       # Bloomberg (enterprise)
│   │       └── factset_provider.py         # FactSet (enterprise)
│   ├── search_tools/                       # 🔍 Search & crawl tools
│   │   ├── __init__.py
│   │   ├── tavily_client.py                # Financial search optimization
│   │   ├── firecrawl_client.py             # SEC filing processing
│   │   └── mcp_adapter.py                  # MCP tool standardization
│   └── mcp_servers/                        # 🌐 MCP server integrations
│       └── __init__.py
│
├── models/                                 # 📈 Quantitative Finance Models
│   ├── __init__.py
│   ├── pricing/                            # 💰 Pricing models
│   │   ├── __init__.py
│   │   ├── black_scholes.py                # Black-Scholes-Merton
│   │   ├── binomial_trees.py               # Binomial/Trinomial trees
│   │   └── monte_carlo.py                  # Monte Carlo simulation
│   ├── risk/                               # ⚠️ Risk models
│   │   ├── __init__.py
│   │   ├── var_models.py                   # Value at Risk models
│   │   ├── credit_risk.py                  # Credit risk models
│   │   └── stress_testing.py               # Stress testing framework
│   ├── portfolio/                          # 📊 Portfolio models
│   │   ├── __init__.py
│   │   ├── markowitz.py                    # Mean-variance optimization
│   │   ├── black_litterman.py              # Black-Litterman model
│   │   └── risk_parity.py                  # Risk parity allocation
│   └── time_series/                        # 📈 Time series models
│       ├── __init__.py
│       ├── arima.py                        # ARIMA/ARMA models
│       ├── garch.py                        # GARCH volatility models
│       └── state_space.py                  # State-space models
│
├── infrastructure/                         # 🚀 Deployment & Operations
│   ├── __init__.py
│   ├── terraform/                          # Infrastructure as Code
│   │   ├── __init__.py
│   │   ├── main.tf                         # Main Terraform config
│   │   ├── variables.tf                    # Variable definitions
│   │   └── modules/                        # Reusable modules
│   ├── docker/                             # 🐳 Container configurations
│   │   ├── __init__.py
│   │   ├── Dockerfile                      # Main application container
│   │   └── docker-compose.yml              # Multi-service deployment
│   └── monitoring/                         # 📊 Observability
│       ├── __init__.py
│       ├── prometheus.yml                  # Metrics configuration
│       └── grafana/                        # Dashboard configurations
│
├── config/                                 # ⚙️ Configuration management
│   ├── __init__.py
│   ├── settings.py                         # Application settings
│   ├── ai_layer_config.py                  # AI provider mappings
│   └── schemas.py                          # Data schemas
│
├── dspy_modules/                           # 🤖 DSPy optimization
│   ├── __init__.py
│   ├── hyde.py                             # HyDE module
│   ├── multi_query.py                      # Multi-query expansion
│   └── optimizer.py                        # Investment banking optimizer
│
├── tracing/                                # 📋 Audit trails & compliance
│   ├── __init__.py
│   └── langsmith_tracer.py                 # LangSmith integration
│
├── eval/                                   # 📊 Evaluation & metrics
│   ├── __init__.py
│   ├── dataset.py                          # Evaluation datasets
│   ├── metrics.py                          # Performance metrics
│   └── run_eval.py                         # Evaluation runner
│
└── main.py                                 # 🎯 Main entry point
```

### 🔄 Migration Summary

#### **From → To Mapping:**
```
OLD STRUCTURE                 →  NEW STRUCTURE
axiom/ai_client_integrations  →  axiom/integrations/ai_providers/
axiom/data_source_integrations →  axiom/integrations/data_sources/
axiom/tools/                  →  axiom/integrations/search_tools/
axiom/graph/                  →  axiom/core/orchestration/
axiom/workflows/              →  axiom/core/analysis_engines/
axiom/utils/                  →  axiom/core/validation/
```

#### **Import Statement Updates:**
```python
# OLD IMPORTS (deprecated - do not use)
from axiom.ai_client_integrations import get_layer_provider  # ❌ OLD
from axiom.graph.state import AxiomState  # ❌ OLD
from axiom.utils.validation import FinancialValidator  # ❌ OLD
from axiom.workflows.due_diligence import run_comprehensive_dd  # ❌ OLD

# NEW IMPORTS (current structure - use these)
from axiom.integrations.ai_providers import get_layer_provider  # ✅ NEW
from axiom.core.orchestration.state import AxiomState  # ✅ NEW
from axiom.core.validation.validation import FinancialValidator  # ✅ NEW
from axiom.core.analysis_engines.due_diligence import run_comprehensive_dd  # ✅ NEW
```

### 🎯 Benefits of New Structure

#### **Improved Organization:**
- **Clear Separation**: Core business logic vs external integrations
- **Logical Grouping**: Related functionality grouped together
- **Scalability**: Easy to add new models, integrations, and infrastructure
- **Maintainability**: Cleaner import paths and dependencies

#### **Developer Experience:**
- **Intuitive Navigation**: Developers can easily find relevant code
- **Module Discovery**: Clear module boundaries and responsibilities
- **Future-Proofing**: Structure supports growth into enterprise platform

#### **Enterprise Readiness:**
- **Infrastructure Support**: Dedicated infrastructure/ directory for Terraform and Docker
- **Model Library**: Dedicated models/ directory for quantitative finance
- **Integration Framework**: Standardized approach for external services
- **Configuration Management**: Centralized config with environment-specific support

### 🔍 Validation Results Post-Restructuring

✅ **System Validation: 7/7 Passed**
- Project Structure: ✅ All directories exist
- Core Files: ✅ All key files present  
- Module Imports: ✅ All imports working with new structure
- Configuration: ✅ AI layer config functional
- AI Providers: ✅ Provider factory operational
- Graph Components: ✅ LangGraph workflow functional
- Tool Integrations: ✅ MCP adapter and tools working

### 🚀 Next Steps

1. **Immediate**: Commit restructured codebase to feature branch
2. **Phase 1**: Implement API key rotation in `core/api_management/`
3. **Phase 2**: Add quantitative models in `models/` directories
4. **Phase 3**: Create Terraform infrastructure in `infrastructure/terraform/`

The restructuring maintains full backward compatibility while providing a modern, scalable foundation for enterprise investment banking analytics.