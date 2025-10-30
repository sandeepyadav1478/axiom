# Axiom Platform - Quick Start Guide

## 5-Minute Setup

```bash
# 1. Clone repository
git clone https://github.com/your-org/axiom.git
cd axiom

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment
cp .env.example .env
# Edit .env with your API keys

# 4. Run platform
python demos/demo_complete_platform_42_models.py
```

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