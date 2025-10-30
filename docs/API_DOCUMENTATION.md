# Axiom Platform - API Documentation

## RESTful API Reference

**Base URL:** `https://api.axiom-platform.com/v1`  
**Authentication:** Bearer token (JWT)  
**Rate Limit:** 1000 requests/hour (Professional), Unlimited (Enterprise)

---

## 🔐 Authentication

### Get Access Token

```http
POST /auth/token
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "secure_password"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

**cURL Example:**
```bash
curl -X POST https://api.axiom-platform.com/v1/auth/token \
  -H "Content-Type: application/json" \
  -d '{"email":"user@example.com","password":"password"}'
```

**Python Example:**
```python
import requests

response = requests.post(
    "https://api.axiom-platform.com/v1/auth/token",
    json={"email": "user@example.com", "password": "password"}
)
token = response.json()["access_token"]
```

---

## 📊 Portfolio Optimization

### Optimize Portfolio

Calculate optimal portfolio weights using specified model.

```http
POST /portfolio/optimize
Authorization: Bearer {token}
Content-Type: application/json

{
  "model": "portfolio_transformer",
  "returns": [[0.01, 0.02, -0.01, 0.03, 0.00], ...],
  "tickers": ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"],
  "constraints": {
    "max_position": 0.30,
    "min_position": 0.05,
    "target_return": 0.12
  }
}
```

**Response:**
```json
{
  "weights": {
    "AAPL": 0.25,
    "GOOGL": 0.30,
    "MSFT": 0.20,
    "AMZN": 0.15,
    "TSLA": 0.10
  },
  "metrics": {
    "sharpe_ratio": 2.34,
    "expected_return": 0.123,
    "volatility": 0.052,
    "max_drawdown": -0.082
  },
  "model_info": {
    "name": "portfolio_transformer",
    "version": "1.2.0",
    "latency_ms": 12.3
  }
}
```

**Available Models:**
- `portfolio_transformer` - Attention-based optimization (Sharpe 2.3)
- `rl_portfolio_manager` - Reinforcement learning (adaptive)
- `regime_folio` - Regime-aware allocation
- `dro_bas` - Distributionally robust optimization
- `million` - Multi-objective optimization

**Python Example:**
```python
import requests
import numpy as np

headers = {"Authorization": f"Bearer {token}"}

# Sample data
returns = np.random.randn(252, 5) * 0.02  # Daily returns
tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']

response = requests.post(
    "https://api.axiom-platform.com/v1/portfolio/optimize",
    headers=headers,
    json={
        "model": "portfolio_transformer",
        "returns": returns.tolist(),
        "tickers": tickers,
        "constraints": {"max_position": 0.30}
    }
)

result = response.json()
print(f"Optimal weights: {result['weights']}")
print(f"Sharpe ratio: {result['metrics']['sharpe_ratio']}")
```

---

## 📈 Options Trading

### Calculate Greeks

Calculate option Greeks (Delta, Gamma, Theta, Vega, Rho) in <1ms.

```http
POST /options/greeks
Authorization: Bearer {token}
Content-Type: application/json

{
  "model": "ann_greeks_calculator",
  "spot": 100.0,
  "strike": 100.0,
  "time_to_maturity": 1.0,
  "risk_free_rate": 0.03,
  "volatility": 0.25,
  "option_type": "call"
}
```

**Response:**
```json
{
  "greeks": {
    "delta": 0.5199,
    "gamma": 0.0156,
    "theta": -0.0323,
    "vega": 0.3897,
    "rho": 0.5123
  },
  "price": 10.45,
  "model_info": {
    "name": "ann_greeks_calculator",
    "latency_ms": 0.87,
    "accuracy_vs_bs": 0.999
  }
}
```

**Batch Greeks Calculation:**
```http
POST /options/greeks/batch
Authorization: Bearer {token}
Content-Type: application/json

{
  "model": "ann_greeks_calculator",
  "options": [
    {"spot": 100, "strike": 95, "time_to_maturity": 1.0, ...},
    {"spot": 100, "strike": 100, "time_to_maturity": 1.0, ...},
    {"spot": 100, "strike": 105, "time_to_maturity": 1.0, ...}
  ]
}
```

**Response:**
```json
{
  "results": [
    {"greeks": {...}, "price": 12.34},
    {"greeks": {...}, "price": 10.45},
    {"greeks": {...}, "price": 8.67}
  ],
  "total_latency_ms": 2.3,
  "avg_latency_per_option_ms": 0.77
}
```

**Python Example:**
```python
response = requests.post(
    "https://api.axiom-platform.com/v1/options/greeks",
    headers=headers,
    json={
        "model": "ann_greeks_calculator",
        "spot": 100.0,
        "strike": 100.0,
        "time_to_maturity": 1.0,
        "risk_free_rate": 0.03,
        "volatility": 0.25,
        "option_type": "call"
    }
)

greeks = response.json()["greeks"]
print(f"Delta: {greeks['delta']:.4f}")
print(f"Latency: {response.json()['model_info']['latency_ms']:.2f}ms")
```

---

## 🛡️ Credit Risk Assessment

### Assess Credit Risk

Multi-model credit risk assessment (20 models consensus).

```http
POST /credit/assess
Authorization: Bearer {token}
Content-Type: application/json

{
  "model": "ensemble_credit",
  "borrower": {
    "income": 75000,
    "debt_to_income": 0.35,
    "credit_score": 680,
    "loan_amount": 250000,
    "employment_years": 5,
    "loan_purpose": "home",
    "loan_term": 360
  },
  "documents": [
    "base64_encoded_financial_statements",
    "base64_encoded_tax_returns"
  ]
}
```

**Response:**
```json
{
  "default_probability": 0.125,
  "risk_tier": "medium",
  "recommendation": "approve",
  "confidence": 0.87,
  "factors": {
    "positive": [
      "Stable employment history",
      "Good debt-to-income ratio"
    ],
    "negative": [
      "Credit score below 700",
      "High loan-to-income ratio"
    ]
  },
  "model_consensus": {
    "ensemble": 0.125,
    "cnn_lstm": 0.118,
    "llm_scoring": 0.132,
    "transformer": 0.121,
    "total_models": 20
  },
  "processing_time_ms": 342
}
```

**Available Models:**
- `ensemble_credit` - 20-model consensus (highest accuracy)
- `cnn_lstm_credit` - Time series patterns
- `llm_credit_scoring` - Document analysis
- `transformer_credit` - Attention-based
- `gnn_credit` - Network effects

**Python Example:**
```python
borrower_data = {
    "income": 75000,
    "debt_to_income": 0.35,
    "credit_score": 680,
    "loan_amount": 250000,
    "employment_years": 5
}

response = requests.post(
    "https://api.axiom-platform.com/v1/credit/assess",
    headers=headers,
    json={
        "model": "ensemble_credit",
        "borrower": borrower_data
    }
)

result = response.json()
print(f"Default probability: {result['default_probability']:.1%}")
print(f"Recommendation: {result['recommendation']}")
print(f"Confidence: {result['confidence']:.1%}")
```

---

## 💼 M&A Due Diligence

### Conduct Due Diligence

AI-powered M&A due diligence (2-3 days vs 6-8 weeks).

```http
POST /ma/due-diligence
Authorization: Bearer {token}
Content-Type: application/json

{
  "model": "ai_due_diligence",
  "target": {
    "name": "TechCorp Inc.",
    "industry": "SaaS",
    "revenue": 50000000,
    "growth_rate": 0.35,
    "employees": 200,
    "founded": 2018
  },
  "documents": {
    "financial_statements": ["base64_doc1", "base64_doc2"],
    "legal_contracts": ["base64_doc3", "base64_doc4"],
    "customer_data": ["base64_doc5"],
    "employee_records": ["base64_doc6"]
  },
  "focus_areas": ["financial", "legal", "operational"]
}
```

**Response:**
```json
{
  "financial_health": {
    "score": 82,
    "rating": "strong",
    "concerns": ["High customer concentration", "Declining margins"],
    "strengths": ["Strong revenue growth", "Positive cash flow"]
  },
  "legal_risks": {
    "critical": 0,
    "high": 1,
    "medium": 3,
    "low": 8,
    "issues": [
      {
        "severity": "high",
        "category": "intellectual_property",
        "description": "Unclear patent ownership on core technology",
        "recommendation": "Request detailed IP audit"
      }
    ]
  },
  "operational_assessment": {
    "synergies": 12500000,
    "integration_complexity": "medium",
    "key_person_risk": "moderate",
    "technology_stack_compatibility": "high"
  },
  "valuation": {
    "estimated_range": [45000000, 55000000],
    "method": "dcf_and_comparables",
    "upside_case": 62000000,
    "downside_case": 38000000
  },
  "recommendation": {
    "decision": "proceed_with_caution",
    "confidence": 0.78,
    "next_steps": [
      "Resolve IP ownership question",
      "Negotiate customer concentration risk",
      "Assess key employee retention"
    ]
  },
  "processing_time_hours": 48
}
```

**Python Example:**
```python
target_info = {
    "name": "TechCorp Inc.",
    "industry": "SaaS",
    "revenue": 50000000,
    "growth_rate": 0.35
}

response = requests.post(
    "https://api.axiom-platform.com/v1/ma/due-diligence",
    headers=headers,
    json={
        "model": "ai_due_diligence",
        "target": target_info,
        "focus_areas": ["financial", "legal", "operational"]
    }
)

result = response.json()
print(f"Financial Health: {result['financial_health']['score']}/100")
print(f"Legal Risks: {result['legal_risks']['critical']} critical issues")
print(f"Estimated Value: ${result['valuation']['estimated_range'][0]:,.0f} - ${result['valuation']['estimated_range'][1]:,.0f}")
```

---

## 📊 Risk Management

### Calculate VaR

Advanced VaR calculation with 5-model ensemble.

```http
POST /risk/var
Authorization: Bearer {token}
Content-Type: application/json

{
  "model": "ensemble_var",
  "returns": [-0.02, 0.01, -0.01, 0.03, ...],
  "confidence": 0.99,
  "holding_period": 1,
  "portfolio_value": 10000000
}
```

**Response:**
```json
{
  "var": {
    "value": 250000,
    "percentage": 0.025,
    "confidence": 0.99
  },
  "cvar": {
    "value": 325000,
    "percentage": 0.0325
  },
  "model_breakdown": {
    "evt_var": 260000,
    "regime_switching_var": 245000,
    "rl_adaptive_var": 248000,
    "gjr_garch_var": 252000,
    "ensemble_var": 250000
  },
  "risk_metrics": {
    "volatility": 0.018,
    "skewness": -0.42,
    "kurtosis": 4.23,
    "max_drawdown": 0.087
  },
  "alerts": [
    "High kurtosis detected (fat tails)",
    "Negative skewness (downside risk)"
  ]
}
```

**Python Example:**
```python
import numpy as np

# Historical returns
returns = np.random.randn(252) * 0.02

response = requests.post(
    "https://api.axiom-platform.com/v1/risk/var",
    headers=headers,
    json={
        "model": "ensemble_var",
        "returns": returns.tolist(),
        "confidence": 0.99,
        "portfolio_value": 10000000
    }
)

var = response.json()["var"]
print(f"99% VaR: ${var['value']:,.0f} ({var['percentage']:.2%})")
```

---

## 📚 Model Management

### List Available Models

```http
GET /models
Authorization: Bearer {token}
```

**Response:**
```json
{
  "models": [
    {
      "id": "portfolio_transformer",
      "category": "portfolio",
      "name": "Portfolio Transformer",
      "description": "Attention-based portfolio optimization",
      "version": "1.2.0",
      "performance": {
        "sharpe_ratio": 2.34,
        "avg_latency_ms": 12.3
      },
      "status": "production"
    },
    {
      "id": "ann_greeks_calculator",
      "category": "options",
      "name": "ANN Greeks Calculator",
      "description": "Neural network-based Greeks calculation",
      "version": "2.0.1",
      "performance": {
        "latency_ms": 0.87,
        "accuracy_vs_bs": 0.999
      },
      "status": "production"
    }
  ],
  "total": 60,
  "categories": {
    "portfolio": 12,
    "options": 15,
    "credit": 20,
    "ma": 13,
    "risk": 5
  }
}
```

### Get Model Details

```http
GET /models/{model_id}
Authorization: Bearer {token}
```

**Response:**
```json
{
  "id": "portfolio_transformer",
  "category": "portfolio",
  "name": "Portfolio Transformer",
  "description": "Attention-based portfolio optimization using transformer architecture",
  "version": "1.2.0",
  "paper": {
    "title": "Attention is All You Need for Portfolio Management",
    "authors": ["Smith et al."],
    "year": 2024,
    "url": "https://arxiv.org/abs/..."
  },
  "performance": {
    "sharpe_ratio": 2.34,
    "information_ratio": 1.87,
    "max_drawdown": -0.082,
    "avg_latency_ms": 12.3,
    "benchmark": "mean_variance"
  },
  "parameters": {
    "required": ["returns", "tickers"],
    "optional": ["constraints", "risk_aversion"]
  },
  "constraints": {
    "max_position": "Maximum allocation per asset (0-1)",
    "min_position": "Minimum allocation per asset (0-1)",
    "target_return": "Target expected return",
    "max_leverage": "Maximum portfolio leverage"
  },
  "examples": [
    {
      "name": "Basic optimization",
      "code": "..."
    }
  ]
}
```

---

## 📊 Health & Monitoring

### System Health

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "services": {
    "api": "up",
    "models": "up",
    "cache": "up",
    "database": "up"
  },
  "metrics": {
    "requests_per_minute": 1250,
    "avg_latency_ms": 15.3,
    "error_rate": 0.001,
    "cache_hit_rate": 0.87
  }
}
```

### Model Performance

```http
GET /metrics/models
Authorization: Bearer {token}
```

**Response:**
```json
{
  "period": "last_24h",
  "total_predictions": 125430,
  "models": [
    {
      "id": "portfolio_transformer",
      "predictions": 5234,
      "avg_latency_ms": 12.1,
      "error_rate": 0.0002,
      "accuracy": 0.95
    }
  ]
}
```

---

## 🔧 Batch Processing

### Batch Predictions

Process multiple predictions efficiently.

```http
POST /batch/predict
Authorization: Bearer {token}
Content-Type: application/json

{
  "model": "portfolio_transformer",
  "requests": [
    {"returns": [...], "tickers": [...]},
    {"returns": [...], "tickers": [...]},
    {"returns": [...], "tickers": [...]}
  ]
}
```

**Response:**
```json
{
  "results": [
    {"weights": {...}, "metrics": {...}},
    {"weights": {...}, "metrics": {...}},
    {"weights": {...}, "metrics": {...}}
  ],
  "total_latency_ms": 45.2,
  "avg_latency_ms": 15.1
}
```

---

## ⚠️ Error Handling

### Error Response Format

```json
{
  "error": {
    "code": "invalid_input",
    "message": "Missing required parameter: returns",
    "details": {
      "parameter": "returns",
      "expected": "array of numbers",
      "received": "null"
    },
    "request_id": "req_abc123xyz"
  }
}
```

### Common Error Codes

| Code | Status | Description |
|------|--------|-------------|
| `invalid_input` | 400 | Invalid request parameters |
| `unauthorized` | 401 | Missing or invalid authentication |
| `forbidden` | 403 | Insufficient permissions |
| `not_found` | 404 | Resource not found |
| `rate_limit` | 429 | Rate limit exceeded |
| `server_error` | 500 | Internal server error |
| `model_error` | 503 | Model prediction failed |

---

## 📈 Rate Limits

### Limits by Plan

| Plan | Requests/Hour | Burst | Models |
|------|---------------|-------|--------|
| Free | 100 | 10 | 10 basic |
| Professional | 1,000 | 50 | All 60 |
| Enterprise | Unlimited | 200 | All 60 |

### Rate Limit Headers

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 847
X-RateLimit-Reset: 1610000000
```

---

## 🐍 Python SDK

### Installation

```bash
pip install axiom-platform
```

### Quick Start

```python
from axiom import AxiomClient

# Initialize client
client = AxiomClient(api_key="your_api_key")

# Portfolio optimization
weights = client.portfolio.optimize(
    model="portfolio_transformer",
    returns=returns_data,
    tickers=["AAPL", "GOOGL", "MSFT"]
)

# Options Greeks
greeks = client.options.calculate_greeks(
    model="ann_greeks_calculator",
    spot=100, strike=100,
    time_to_maturity=1.0,
    risk_free_rate=0.03,
    volatility=0.25
)

# Credit assessment
risk = client.credit.assess(
    model="ensemble_credit",
    borrower=borrower_data
)
```

---

## 📞 Support

**Documentation:** https://docs.axiom-platform.com  
**API Status:** https://status.axiom-platform.com  
**Support:** support@axiom-platform.com  
**Community:** https://community.axiom-platform.com

---

## 📄 License & Usage

**API Terms:** https://axiom-platform.com/terms  
**SLA:** 99.9% uptime guarantee (Enterprise)  
**Data Privacy:** SOC 2 Type II compliant