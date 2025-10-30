# Portfolio Optimization - Deep Research Notes

**Research Session: Portfolio Optimization (Topic 2/7)**
**Date:** 2025-10-29
**Objective:** Discover cutting-edge portfolio optimization techniques for quantitative finance platform

---

## Key Papers Found

### 1. MILLION Framework (December 2024) ⭐⭐⭐
**Paper:** arXiv:2412.03038 - "MILLION: A General Multi-Objective Framework with Controllable Risk for Portfolio Management"
**Authors:** Liwei Deng, Tianfu Wang, Yan Zhao, Kai Zheng
**Status:** Accepted by VLDB 2025
**Date:** Originally announced December 2024

**Key Contributions:**
- **Multi-Objective Portfolio Management (MILLION)** - comprehensive framework
- **Two-Phase Architecture:**
  1. **Return-Related Maximization Phase:**
     - Return rate prediction
     - Return rate ranking
     - Combined with portfolio optimization to prevent overfitting
     - Improves generalization to future markets
  
  2. **Risk Control Phase:**
     - Portfolio interpolation method - theoretically proven to perfectly control risk if target risk level is in proper interval
     - Portfolio improvement method - achieves higher returns while maintaining same risk level
     - Fast risk adaptation to user-specified risk levels

- **Mathematical Properties:**
  - Proof that adjusted portfolio after interpolation has min-variance no greater than original min-variance optimization
  - Fine-grained risk control capabilities
  - Better return rates compared to interpolation alone

- **Validation:**
  - Extensive experiments on 3 real-world datasets
  - Demonstrates effectiveness and efficiency

**Implementation Potential:** HIGH
- Could implement as new portfolio optimizer in `axiom/models/portfolio/`
- Multi-objective approach aligns with institutional needs
- Risk control mechanisms crucial for real-world deployment

---

### 2. DRO-BAS with Bayesian Ambiguity Sets (November 2024) ⭐⭐
**Paper:** arXiv:2411.16829 - "Decision Making under the Exponential Family: Distributionally Robust Optimisation with Bayesian Ambiguity Sets"
**Authors:** Charita Dellaporta, Patrick O'Hara, Theodoros Damoulas  
**Status:** Accepted for publication (spotlight) at ICML 2025
**Date:** Submitted June 2025, v1 November 2024

**Key Contributions:**
- **Distributionally Robust Optimization (DRO-BAS)**
- Bayesian approach to handling uncertainty in portfolio problems
- Generates robust portfolios using exponential family distributions
- Outperforms existing Bayesian DRO on Newsvendor problem
- Faster solve times with comparable robustness
- Addresses portfolio problems with distributional ambiguity

**Implementation Potential:** MEDIUM
- More theoretical/mathematical focus
- Useful for robust portfolio construction under uncertainty
- Could complement existing risk models

---

### 3. Deep Learning for Risk-Aligned Portfolio Optimization (August 2025) ⭐⭐⭐
**Paper:** "Advanced investing with deep learning for risk-aligned portfolio optimization"
**Author:** Minh Duc Nguyen (MD Nguyen)
**Journal:** PLoS One 20(8): e0330547
**Date:** Published August 19, 2025
**DOI:** https://doi.org/10.1371/journal.pone.0330547

**Key Contributions:**
- **Deep Learning Prediction Models:**
  - **LSTM (Long Short-Term Memory)** for time series forecasting
  - **1D-CNN (One-Dimensional Convolutional Neural Network)** for pattern recognition
  - Combined for improved return predictions

- **Three Portfolio Optimization Frameworks:**
  1. **MVF (Mean-Variance with Forecasting)** - Return-seeking, moderate-risk
  2. **RPP (Risk Parity Portfolio)** - Balanced risk allocation
  3. **MDP (Maximum Drawdown Portfolio)** - Conservative, risk-averse

- **Real-World Testing:**
  - Dataset: Daily returns of VN-100 stocks (Vietnam)
  - Period: 2017 to 2024
  - Test period: 2023-2024 showed strong performance

- **Performance Results:**
  - **LSTM outperformed CNN** in all portfolios for both accuracy and stability
  - **LSTM+MVF:** Best risk-adjusted returns
  - **LSTM+MDP:** Highest total return
  - Highlights value of aligning predictive models with optimization strategies

- **Future Directions:**
  - Other asset classes
  - Transaction cost modeling
  - Dynamic rebalancing
  - Macroeconomic/alternative data integration

**Implementation Potential:** VERY HIGH
- Practical, tested approach
- Clear framework architecture
- LSTM models work well in practice
- Can implement all three portfolio types
- Vietnam market data available for validation

---

### 4. RegimeFolio - ML for Sectoral Portfolio Optimization (2025) ⭐⭐
**Paper:** "RegimeFolio: A Regime Aware ML System for Sectoral Portfolio Optimization in Dynamic Markets"
**Authors:** Y Zhang, D Goel, H Ahmad, C Szabo
**Conference:** IEEE Access, 2025
**Status:** Published at ieee.org

**Key Contributions:**
- **Regime-Aware Portfolio Management**
- Captures US equities from 2020 to 2024
- Framework achieves competitive returns AND advanced machine learning benchmarks
- Designed for dynamic markets with changing regimes
- Sectoral approach to portfolio allocation

**Implementation Potential:** MEDIUM-HIGH
- Regime detection is crucial for real markets
- Sector-based allocation useful for institutional portfolios
- 2020-2024 period includes COVID and recovery
- Machine learning benchmarking

---

### 5. RL Portfolio Management with Continuous Actions (May 2024) ⭐⭐⭐
**Paper:** "Portfolio management based on a reinforcement learning framework"
**Authors:** Wu Junfeng, Li Yaoming, Tan Wenqing, Chen Yun
**Journal:** Journal of Forecasting, Volume 43, Issue 7, pp. 2792-2808
**Date:** First published May 19, 2024
**DOI:** https://doi.org/10.1002/for.3155

**Key Contributions:**
- **Reinforcement Learning with Continuous Action Space**
  - Feature extraction network + fully connected network
  - Addresses continuous action space problem (sum of portfolio weights = 1)
  - Proximal Policy Optimization (PPO) algorithm

- **Feature Extraction Networks Compared:**
  1. **CNN (Convolutional Neural Network)** - BEST performer in test set
  2. **LSTM (Long Short-Term Memory)**
  3. **Convolutional LSTM** - hybrid approach

- **Experimental Setup:**
  - 6 kinds of assets
  - 16 features per asset
  - Extensive experiments comparing architectures

- **Key Findings:**
  - **CNN performed best** in test set
  - **Monthly trading frequency** achieved highest Sharpe ratio vs other frequencies
  - Addresses major limitation of previous RL portfolio work (discrete action spaces)

- **Technical Details:**
  - Continuous action space with constraint (weights sum to 1)
  - Policy gradient methods (PPO)
  - Feature engineering critical for performance

**Implementation Potential:** VERY HIGH
- Solves practical problem (continuous weights)
- CNN architecture simpler than LSTM
- PPO algorithm well-established
- Clear performance metrics (Sharpe ratio)
- Trading frequency analysis useful

---

## Research Areas to Explore Further

### Completed Searches:
✅ Initial arXiv search for "portfolio optimization machine learning 2024 2025"  
✅ Google Scholar: "deep learning portfolio optimization 2024 2025"
✅ Google Scholar: "reinforcement learning portfolio management 2024"

### Next Searches:
- [ ] Transaction cost optimization portfolio
- [ ] Multi-period portfolio optimization stochastic
- [ ] Factor investing smart beta 2024
- [ ] Black-Litterman model modern improvements
- [ ] ESG portfolio optimization 2024
- [ ] High-frequency portfolio rebalancing
- [ ] Transformer models portfolio management
- [ ] Graph neural networks portfolio
- [ ] Alternative data portfolio construction
- [ ] Robust portfolio optimization

---

## Technical Implementation Notes

### Current Platform Gaps:
1. No multi-objective portfolio optimizer
2. No regime-aware portfolio management
3. Limited risk control mechanisms  
4. Basic mean-variance only
5. No deep learning predictions integrated with optimization
6. No RL-based portfolio management (continuous action space)
7. No transaction cost optimization
8. No factor-based portfolio construction

### Integration Points:
- `axiom/models/portfolio/` - add new optimizers
- `axiom/core/analysis_engines/` - integrate with existing workflows
- `axiom/models/base/factory.py` - register new model types
- `axiom/models/time_series/` - add LSTM/CNN predictors

### Dependencies Needed:
- cvxpy (convex optimization) - ALREADY HAVE
- scipy.optimize (constrained optimization) - ALREADY HAVE
- tensorflow or pytorch (deep learning) - HAVE PYTORCH for RL-GARCH
- stable-baselines3 (RL algorithms - PPO) - ALREADY HAVE
- Additional libraries TBD based on approaches

### Priority Implementations:
1. **RL Portfolio Manager with PPO** (from Wu et al. 2024) ⭐⭐⭐
   - CNN feature extraction
   - Continuous action space (weights sum to 1)
   - PPO algorithm from stable-baselines3
   - Monthly rebalancing
   
2. **LSTM+CNN Portfolio Predictor** (from Nguyen 2025) ⭐⭐⭐
   - Implement LSTM time series forecasting
   - Implement 1D-CNN pattern recognition
   - Ensemble predictions
   
3. **Three Portfolio Frameworks** (from Nguyen 2025)
   - Mean-Variance with Forecasting (MVF)
   - Risk Parity Portfolio (RPP)
   - Maximum Drawdown Portfolio (MDP)

4. **MILLION Multi-Objective Framework** (from arXiv 2024) ⭐⭐
   - Two-phase optimization
   - Risk control mechanisms
   - Portfolio interpolation/improvement

5. **Regime Detection** (from RegimeFolio 2025)
   - Market regime identification
   - Regime-conditional optimization

---

## Next Steps:
1. ✅ Find 5 cutting-edge papers on portfolio optimization
2. Continue systematic literature search on transaction costs, factors, transformers
3. Search for open-source implementations and code repos
4. Identify top 2-3 approaches to implement first
5. Design integration architecture
6. Implement RL PPO Portfolio Manager first (most practical)
7. Then implement LSTM+CNN predictor framework
8. Test and validate on real data

**Status:** In Progress - Continue Research (45 min into 1.5-2 hour session)
**Papers Found:** 5/10+ target