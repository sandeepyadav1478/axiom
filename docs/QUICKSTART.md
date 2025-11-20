# Axiom Platform - Quick Start Guide

## ⚠️ IMPORTANT: Setup Order Matters!

**Follow these steps IN ORDER - especially step 3 (.env configuration)!**

## Complete Setup (10 minutes)

```bash
# 1. Clone repository
git clone https://github.com/your-org/axiom.git
cd axiom

# 2. Create virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. 🚨 CRITICAL STEP: Configure Environment
python setup_environment.py
# This script will:
#  - Check if .env exists (create from template if not)
#  - Validate system dependencies
#  - Guide you through configuration
#  - Ensure API keys are set

# 4. Install Python dependencies
uv pip install numpy
uv pip install --no-build-isolation pmdarima
uv pip install -r requirements.txt
uv pip install neo4j
uv pip install -e .

# 5. Start databases
cd axiom/database
docker compose up -d postgres
docker compose --profile cache up -d redis
docker compose --profile vector-db-light up -d chromadb
docker compose --profile graph-db up -d neo4j
cd ../..

# 6. Verify setup
python demos/demo_complete_data_infrastructure.py
python demos/demo_multi_database_architecture.py
python test_gpu.py  # If you have GPU
```

## ⚠️ Why .env Configuration is Critical

**Without .env file:**
- ❌ No API keys for AI providers (OpenAI/Anthropic)
- ❌ Security risk (hardcoded default passwords)
- ❌ Can't customize database settings
- ❌ Production deployment will fail

**The setup_environment.py script ensures you don't skip this critical step!**

---

## First Portfolio Optimization

```python
from axiom.models.base.factory import ModelFactory, ModelType
import numpy as np

# Load model
model = ModelFactory.create(ModelType.PORTFOLIO_TRANSFORMER)

# Your data (5 assets, 30 days)
market_data = np.random.randn(30, 25)  # 30 days, 5 assets × 5 features

# Get optimal allocation
weights = model.allocate(market_data)
print(f"Optimal weights: {weights}")
```

---

## First Credit Assessment

```python
from axiom.models.base.factory import ModelFactory, ModelType

# Load credit models
ensemble = ModelFactory.create(ModelType.ENSEMBLE_CREDIT)
llm_scorer = ModelFactory.create(ModelType.LLM_CREDIT_SCORING)

# Borrower data
borrower_features = np.array([...])  # Your features

# Get predictions
default_prob = ensemble.predict_proba(borrower_features)
print(f"Default probability: {default_prob:.1%}")
```

---

## First M&A Analysis

```python
from axiom.models.ma.ml_target_screener import MLTargetScreener

# Screen targets
screener = MLTargetScreener()
acquirer = {'name': 'Your Company', 'revenue': 2_000_000_000}
targets = [...]  # List of target profiles

# Get ranked targets
ranked = screener.screen_targets(acquirer, targets)
print(f"Top target: {ranked[0][0].company_name}")
```

---

## Deploy to Production

```bash
# Using Docker Compose
./scripts/deploy_production.sh

# Or Kubernetes
kubectl apply -f kubernetes/deployment.yaml
```

---

## Access Services

- **API:** http://localhost:8000
- **MLflow:** http://localhost:5000
- **Grafana:** http://localhost:3000
- **Prometheus:** http://localhost:9090

---

## Generate Client Reports

```python
from axiom.client_interface.portfolio_dashboard import PortfolioDashboard

dashboard = PortfolioDashboard(your_data)
fig = dashboard.create_dashboard()
fig.write_html('client_report.html')
```

---

**60 ML Models | Complete Infrastructure | Client-Ready**

See full docs in `/docs` directory.