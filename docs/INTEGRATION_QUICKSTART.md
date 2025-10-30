# Integration Quickstart - Using All Implemented Models

**Quick Reference:** How to use all 7 implemented ML models + open-source tools

---

## 🚀 1. RL-GARCH VaR Model

```python
from axiom.models.base.factory import ModelFactory, ModelType

# Create model via factory
var_model = ModelFactory.create(ModelType.RL_GARCH_VAR)

# Train on historical returns
var_model.train(returns_data, episodes=1000)

# Calculate VaR
var_95 = var_model.calculate_var(current_returns, confidence_level=0.95)
print(f"95% VaR: ${var_95:,.0f}")
```

**When to use:** Real-time VaR estimation, adaptive risk management

---

## 🚀 2. RL Portfolio Manager (PPO)

```python
from axiom.models.base.factory import ModelFactory, ModelType

# Create via factory
manager = ModelFactory.create(ModelType.RL_PORTFOLIO_MANAGER)

# Train
manager.train(historical_data, total_timesteps=100000)

# Get optimal allocation
weights = manager.allocate(current_state)
print(f"Allocation: {weights}")

# Backtest
results = manager.backtest(test_data)
print(f"Sharpe: {results['sharpe_ratio']:.2f}")
```

**When to use:** Systematic portfolio rebalancing, maximizing risk-adjusted returns

---

## 🚀 3. VAE+MLP Option Pricer

```python
from axiom.models.base.factory import ModelFactory, ModelType
from axiom.models.pricing.vae_option_pricer import OptionType

# Create via factory
pricer = ModelFactory.create(ModelType.VAE_OPTION_PRICER)

# Train VAE on volatility surfaces
pricer.train_vae(volatility_surfaces, epochs=100)

# Train MLP on options
pricer.train_pricer(vol_surfaces, option_params, prices, epochs=100)

# Price American put
price = pricer.price_option(
    volatility_surface=current_vol_surface,
    strike=100, maturity=1.0, spot=100,
    rate=0.03, dividend_yield=0.02,
    option_type=OptionType.AMERICAN_PUT
)
print(f"American Put: ${price:.2f}")
```

**When to use:** Fast exotic options pricing, volatility surface modeling

---

## 🚀 4. CNN-LSTM-Attention Credit Model

```python
from axiom.models.base.factory import ModelFactory, ModelType

# Create via factory
credit_model = ModelFactory.create(ModelType.CNN_LSTM_CREDIT)

# Train on credit history sequences
credit_model.train(X_train, y_train, X_val, y_val, epochs=100)

# Predict default probability
prob = credit_model.predict_proba(customer_history)
print(f"Default Probability: {prob[0]:.1%}")

# Get attention weights (interpretability)
attention = credit_model.get_attention_weights(customer_history)
most_important_month = attention[0, -1, :].argmax()
print(f"Most critical: {most_important_month} months ago")
```

**When to use:** Credit card default prediction, early warning systems

---

## 🚀 5. Ensemble XGBoost+LightGBM Credit

```python
from axiom.models.base.factory import ModelFactory, ModelType

# Create via factory
ensemble = ModelFactory.create(ModelType.ENSEMBLE_CREDIT)

# Train all models (XGB, LGB, RF, GB)
ensemble.train(X_train, y_train, X_val, y_val)

# Predict using stacking ensemble
probs = ensemble.predict_proba(X_test, use_ensemble=True)

# Get feature importance
importance = ensemble.get_feature_importance(top_n=10, feature_names=feature_names)
print(importance)
```

**When to use:** Production credit scoring, high-accuracy predictions

---

## 🚀 6. LSTM+CNN Portfolio (3 Frameworks)

```python
from axiom.models.base.factory import ModelFactory, ModelType
from axiom.models.portfolio.lstm_cnn_predictor import PortfolioFramework

# Create via factory
predictor = ModelFactory.create(ModelType.LSTM_CNN_PORTFOLIO)

# Train LSTM
predictor.train_lstm(X_train, y_train, X_val, y_val, epochs=100)

# Optimize with MVF framework (moderate risk)
mvf_result = predictor.optimize_portfolio(
    current_data,
    historical_returns,
    framework=PortfolioFramework.MVF
)

# Or Risk Parity (balanced)
rpp_result = predictor.optimize_portfolio(
    current_data,
    historical_returns,
    framework=PortfolioFramework.RPP
)

# Or Maximum Drawdown (conservative)
mdp_result = predictor.optimize_portfolio(
    current_data,
    historical_returns,
    framework=PortfolioFramework.MDP
)

print(f"MVF Sharpe: {mvf_result['sharpe_ratio']:.2f}")
```

**When to use:** Risk-aligned portfolios, different investor preferences

---

## 🚀 7. Portfolio Transformer

```python
from axiom.models.base.factory import ModelFactory, ModelType

# Create via factory
transformer = ModelFactory.create(ModelType.PORTFOLIO_TRANSFORMER)

# Train end-to-end on Sharpe ratio
transformer.train(X_train, returns_train, X_val, returns_val, epochs=100)

# Get optimal weights
weights = transformer.allocate(current_market_data)
print(f"Weights: {weights}")

# Backtest
results = transformer.backtest(
    market_data,
    returns_data,
    rebalance_frequency=5
)
print(f"Sharpe: {results['sharpe_ratio']:.2f}")
```

**When to use:** End-to-end portfolio optimization, regime adaptation

---

## 📊 Open Source Tool Integration

### **MLflow Experiment Tracking**

```python
from axiom.infrastructure.mlops.experiment_tracking import AxiomMLflowTracker

# Initialize
tracker = AxiomMLflowTracker("portfolio_experiments")

# Track training
with tracker.start_run("transformer_v1"):
    tracker.log_params(config.__dict__)
    
    for epoch in range(epochs):
        loss = train_one_epoch()
        tracker.log_metrics({"loss": loss}, step=epoch)
    
    tracker.log_model(model, "model", registered_model_name="PortfolioTransformer")

# View: mlflow ui --port 5000
```

### **QuantStats Risk Analytics**

```python
from axiom.infrastructure.analytics.risk_metrics import quick_analysis, generate_performance_report

# Quick metrics
metrics = quick_analysis(portfolio_returns)
print(f"Sharpe: {metrics['sharpe']:.2f}")
print(f"Max DD: {metrics['max_drawdown']:.1%}")
print(f"Calmar: {metrics['calmar']:.2f}")

# Full HTML report
generate_performance_report(
    returns=portfolio_returns,
    benchmark=spy_returns,
    output_file="strategy_report.html"
)
```

### **Evidently Drift Detection**

```python
from axiom.infrastructure.monitoring.drift_detection import AxiomDriftMonitor

# Initialize with training data
monitor = AxiomDriftMonitor(
    reference_data=training_df,
    target_column="default",
    prediction_column="prediction"
)

# Check production data for drift
drift_results = monitor.detect_drift(production_df)

if drift_results['drift_detected']:
    print(f"⚠️ Drift in: {drift_results['drifted_features']}")
    # Trigger retraining

# Generate monitoring report
monitor.generate_report(production_df, "drift_report.html")
```

### **PyPortfolioOpt (Traditional Optimization)**

```python
from pypfopt import EfficientFrontier, risk_models, expected_returns

# Use PyPortfolioOpt for traditional methods
mu = expected_returns.ema_historical_return(prices)
S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()

# Markowitz
ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe()
performance = ef.portfolio_performance(verbose=True)

# Or Black-Litterman with views
from pypfopt import black_litterman
viewdict = {"AAPL": 0.15, "TSLA": 0.10}  # Investor views
bl = black_litterman.BlackLittermanModel(S, pi=mu, absolute_views=viewdict)
bl_weights = bl.bl_weights()
```

**Use PyPortfolioOpt for:** Traditional Markowitz, Black-Litterman, HRP  
**Use our models for:** RL-based, transformer-based, ML-predicted allocations

---

## 🎯 Model Selection Guide

| Objective | Recommended Model | Why |
|-----------|------------------|-----|
| **Max Sharpe Ratio** | Portfolio Transformer | End-to-end optimization |
| **Risk-Aligned (Moderate)** | LSTM+CNN MVF | Predicted returns + MV |
| **Risk-Aligned (Balanced)** | LSTM+CNN RPP | Equal risk contribution |
| **Risk-Aligned (Conservative)** | LSTM+CNN MDP | Min drawdown |
| **Continuous Adaptation** | RL Portfolio Manager | Learns from market |
| **Traditional + Views** | PyPortfolioOpt BL | Incorporate opinions |
| **Fast Exotic Options** | VAE+MLP Pricer | 1000x faster |
| **American/Asian Options** | VAE+MLP Pricer | Handles path-dependent |
| **Credit Card Defaults** | CNN-LSTM-Attention | 16% improvement |
| **Loan Defaults** | Ensemble XGB+LGB | Production-proven |
| **Dynamic VaR** | RL-GARCH | Adapts to volatility |

---

## 📈 Complete Workflow Example

```python
# 1. Track experiment with MLflow
from axiom.infrastructure.mlops.experiment_tracking import AxiomMLflowTracker
tracker = AxiomMLflowTracker("complete_workflow")

with tracker.start_run("daily_allocation"):
    # 2. Get optimal allocation from transformer
    from axiom.models.base.factory import ModelFactory, ModelType
    transformer = ModelFactory.create(ModelType.PORTFOLIO_TRANSFORMER)
    weights = transformer.allocate(current_market_data)
    
    tracker.log_params({"model": "transformer", "date": str(datetime.now())})
    
    # 3. Calculate expected risk metrics with QuantStats
    from axiom.infrastructure.analytics.risk_metrics import quick_analysis
    metrics = quick_analysis(historical_returns)
    tracker.log_metrics(metrics)
    
    # 4. Check for drift with Evidently
    from axiom.infrastructure.monitoring.drift_detection import AxiomDriftMonitor
    monitor = AxiomDriftMonitor(reference_data, target_column="return")
    drift = monitor.detect_drift(current_data)
    
    if drift['drift_detected']:
        print("⚠️ Drift detected - consider retraining")
        tracker.log_metrics({"drift_detected": 1})
    
    # 5. Execute allocation
    execute_trades(weights)
    
    print("✓ Complete workflow executed with full tracking")
```

---

## 📦 Installation

```bash
# Install all dependencies
pip install -r requirements.txt

# Or with uv (faster)
uv pip install -r requirements.txt

# Verify installations
python -c "import mlflow, quantstats, evidently; print('✓ All tools ready')"
```

---

## 🎓 Best Practices

### **1. Always Use MLflow for Training**
Track every experiment - costs nothing, saves hours of debugging

### **2. Use QuantStats for Metrics**
Don't calculate Sharpe/Sortino manually - use battle-tested calculations

### **3. Monitor with Evidently**
Catch drift before model performance degrades

### **4. Leverage PyPortfolioOpt**
Use for traditional optimization - we already have it!

### **5. Keep Custom Models for Novel Research**
Our 7 models implement cutting-edge research not available elsewhere

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Axiom ML Platform                     │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  CUSTOM MODELS (Novel Research 2024-2025)               │
│  ├── RL-GARCH VaR                                       │
│  ├── RL Portfolio Manager (PPO)                         │
│  ├── VAE+MLP Option Pricer                             │
│  ├── CNN-LSTM-Attention Credit                         │
│  ├── Ensemble XGBoost+LightGBM                         │
│  ├── LSTM+CNN Portfolio (3 frameworks)                 │
│  └── Portfolio Transformer                              │
│                                                          │
│  LEVERAGED TOOLS (Open Source)                          │
│  ├── MLflow - Experiment Tracking                       │
│  ├── QuantStats - Risk Analytics                        │
│  ├── Evidently - Drift Detection                        │
│  ├── PyPortfolioOpt - Traditional Optimization          │
│  ├── Optuna - Hyperparameter Tuning                    │
│  ├── XGBoost/LightGBM - Gradient Boosting              │
│  ├── PyTorch - Deep Learning                            │
│  └── stable-baselines3 - RL Algorithms                  │
│                                                          │
│  INTEGRATION LAYER                                       │
│  └── ModelFactory - Unified Model Creation              │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## ✅ Verification

Run all demos to verify implementations:

```bash
# 1. VaR
python demos/demo_rl_garch_var_2025.py

# 2-3. Portfolio
python demos/demo_rl_portfolio_manager.py
python demos/demo_lstm_cnn_portfolio.py

# 4. Options
python demos/demo_vae_option_pricer.py

# 5-6. Credit
python demos/demo_cnn_lstm_credit_model.py
python demos/demo_ensemble_credit_model.py

# All should run successfully
```

---

**Status:** Complete integration guide for all models + open-source tools
**Usage:** Reference this document when implementing new strategies