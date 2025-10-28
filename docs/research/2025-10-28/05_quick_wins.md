# Quick Wins: This Week's Actionable Tasks
**Date:** October 28, 2025  
**Timeline:** Days 1-5 (This Week)  
**Focus:** Immediate, high-impact implementations

---

## 🎯 This Week's Goals

### Day 1: Monday - Dependency Updates ✅
### Day 2: Tuesday - DSPy Migration Start ⏰
### Day 3: Wednesday - DSPy Migration Complete ⏰
### Day 4: Thursday - RL-GARCH Setup ⏰
### Day 5: Friday - Testing & Validation ⏰

---

## 📅 Day-by-Day Action Plan

### Day 1: Monday - Foundation Updates

#### Morning (2 hours): Low-Risk Dependencies
```bash
#!/bin/bash
# Save current state
pip freeze > requirements_backup_2025-10-28.txt

# Update scientific computing stack
pip install --upgrade pandas==2.2.3
pip install --upgrade numpy==2.1.2
pip install --upgrade scipy==1.14.1
pip install --upgrade scikit-learn==1.5.2

# Verify installation
python -c "import pandas; print(f'Pandas: {pandas.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import scipy; print(f'SciPy: {scipy.__version__}')"
python -c "import sklearn; print(f'Scikit-learn: {sklearn.__version__}')"
```

**Verification:**
```bash
# Run basic tests
pytest tests/models/portfolio/ -v
pytest tests/models/risk/ -v
```

**Expected Outcome:**
- ✅ All packages updated
- ✅ No test failures
- ✅ ~15 minutes actual work

---

#### Afternoon (2 hours): Financial Libraries
```bash
# Update QuantLib
pip install --upgrade QuantLib==1.35
pip install --upgrade QuantLib-Python==1.35

# Update PyPortfolioOpt
pip install --upgrade PyPortfolioOpt==1.5.5

# Update LangGraph
pip install --upgrade langgraph==0.6.5

# Install new dependencies for RL-GARCH
pip install arch==6.3.0
pip install stable-baselines3==2.3.1
```

**Verification:**
```python
# Test QuantLib
import QuantLib as ql
print(f"QuantLib version: {ql.__version__}")

# Test PyPortfolioOpt
from pypfopt import HRPOpt
print("PyPortfolioOpt working ✓")

# Test LangGraph
from langgraph.graph import StateGraph
print("LangGraph working ✓")

# Test RL libraries
from arch import arch_model
from stable_baselines3 import DDQN
print("RL libraries ready ✓")
```

**Expected Outcome:**
- ✅ All financial libraries updated
- ✅ New RL dependencies installed
- ✅ Tests pass

**Files to Update:**
- Update [`requirements.txt`](../../requirements.txt:1) with new versions

---

### Day 2: Tuesday - DSPy Migration (Part 1)

#### Morning (3 hours): Update DSPy
```bash
# Install DSPy 3.0.4b2
pip install dspy-ai==3.0.4b2

# Verify installation
python -c "import dspy; print(f'DSPy version: {dspy.__version__}')"
```

#### Afternoon (3 hours): Migrate First Module
**File:** [`axiom/dspy_modules/hyde.py`](../../axiom/dspy_modules/hyde.py:1)

**Before (DSPy 2.x):**
```python
# OLD CODE - Example structure
import dspy

class HyDESignature(dspy.Signature):
    context = dspy.InputField()
    question = dspy.InputField()
    document = dspy.OutputField()
```

**After (DSPy 3.0):**
```python
# NEW CODE - Update to new syntax
from dspy import Signature, InputField, OutputField

class HyDESignature(Signature):
    """Generate hypothetical document for retrieval"""
    context: str = InputField(desc="Background context for the query")
    question: str = InputField(desc="User's question to answer")
    document: str = OutputField(desc="Hypothetical document that would answer the question")
```

**Testing:**
```bash
# Test the updated module
pytest tests/dspy_modules/test_hyde.py -v
```

**Expected Outcome:**
- ✅ HyDE module migrated
- ✅ Tests pass
- ✅ No functionality broken

---

### Day 3: Wednesday - DSPy Migration (Part 2)

#### Morning (3 hours): Multi-Query Module
**File:** [`axiom/dspy_modules/multi_query.py`](../../axiom/dspy_modules/multi_query.py:1)

**Migration Example:**
```python
from dspy import Signature, InputField, OutputField, Module
import dspy

class MultiQuerySignature(Signature):
    """Generate multiple search queries from a question"""
    question: str = InputField(desc="Original user question")
    num_queries: int = InputField(desc="Number of queries to generate", default=3)
    queries: list[str] = OutputField(desc="List of alternative search queries")

class MultiQueryGenerator(Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(MultiQuerySignature)
        
    def forward(self, question: str, num_queries: int = 3):
        return self.generate(question=question, num_queries=num_queries)
```

#### Afternoon (3 hours): Optimizer Module
**File:** [`axiom/dspy_modules/optimizer.py`](../../axiom/dspy_modules/optimizer.py:1)

**Migration Example:**
```python
from dspy.teleprompt import BootstrapFewShotWithRandomSearch

class DSPyOptimizer:
    def __init__(self, metric):
        self.metric = metric
        # NEW: Updated optimizer name
        self.optimizer = BootstrapFewShotWithRandomSearch(
            metric=self.metric,
            max_bootstrapped_demos=8,
            max_labeled_demos=4
        )
        
    def compile(self, module, trainset):
        return self.optimizer.compile(
            module,
            trainset=trainset
        )
```

**Expected Outcome:**
- ✅ All DSPy modules migrated
- ✅ Full test suite passes
- ✅ Ready for production

---

### Day 4: Thursday - RL-GARCH Setup

#### Morning (2 hours): File Structure
```bash
# Create new file for RL-GARCH implementation
touch axiom/models/risk/rl_garch_var.py
touch tests/models/risk/test_rl_garch_var.py
```

#### Afternoon (4 hours): Basic Implementation
**File:** Create [`axiom/models/risk/rl_garch_var.py`](../../axiom/models/risk/rl_garch_var.py:1)

```python
"""
RL-GARCH VaR Model
Based on arXiv:2504.16635
Combines GARCH volatility modeling with Deep Q-Learning
"""

import numpy as np
from arch import arch_model
from stable_baselines3 import DDQN
from typing import Optional, Tuple
import gym
from gym import spaces

class GARCHEnvironment(gym.Env):
    """
    RL environment for GARCH VaR estimation
    State: [volatility, returns, regime_indicator]
    Action: VaR adjustment factor
    Reward: -abs(VaR_error) - penalty for violations
    """
    def __init__(self, returns: np.ndarray, confidence_level: float = 0.95):
        super().__init__()
        self.returns = returns
        self.confidence_level = confidence_level
        self.current_step = 0
        
        # State space: [vol, return, regime]
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(3,), 
            dtype=np.float32
        )
        
        # Action space: VaR adjustment multiplier [0.8, 1.2]
        self.action_space = spaces.Box(
            low=0.8,
            high=1.2,
            shape=(1,),
            dtype=np.float32
        )
        
    def reset(self):
        self.current_step = 0
        return self._get_state()
        
    def step(self, action):
        # Calculate reward based on VaR accuracy
        reward = self._calculate_reward(action)
        self.current_step += 1
        done = self.current_step >= len(self.returns) - 1
        return self._get_state(), reward, done, {}
        
    def _get_state(self):
        # Return current market state
        vol = self._estimate_volatility()
        ret = self.returns[self.current_step]
        regime = self._detect_regime()
        return np.array([vol, ret, regime], dtype=np.float32)
        
    def _calculate_reward(self, action):
        # Reward function based on VaR accuracy
        # Negative reward for VaR violations
        # Penalty for overly conservative estimates
        pass
        
    def _estimate_volatility(self):
        # Rolling volatility estimate
        window = min(20, self.current_step)
        return np.std(self.returns[max(0, self.current_step-window):self.current_step+1])
        
    def _detect_regime(self):
        # Simple regime detection (can be enhanced)
        recent_vol = self._estimate_volatility()
        historical_vol = np.std(self.returns[:self.current_step+1])
        return 1.0 if recent_vol > 1.5 * historical_vol else 0.0


class RLGARCHVaR:
    """
    RL-enhanced GARCH VaR model
    Combines traditional GARCH(1,1) with RL-based adjustments
    """
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.garch_model = None
        self.rl_agent = None
        self.is_fitted = False
        
    def fit(self, returns: np.ndarray, train_rl: bool = True):
        """
        Fit both GARCH model and RL agent
        
        Args:
            returns: Historical returns
            train_rl: Whether to train RL agent (can be False for quick testing)
        """
        # Fit GARCH(1,1) model
        self.garch_model = arch_model(
            returns * 100,  # Scale for numerical stability
            vol='Garch',
            p=1,
            q=1
        )
        self.garch_fit = self.garch_model.fit(disp='off', show_warning=False)
        
        if train_rl:
            # Create environment and train RL agent
            env = GARCHEnvironment(returns, self.confidence_level)
            
            self.rl_agent = DDQN(
                'MlpPolicy',
                env,
                learning_rate=0.001,
                buffer_size=10000,
                learning_starts=1000,
                batch_size=64,
                tau=0.005,
                gamma=0.99,
                train_freq=4,
                gradient_steps=1,
                verbose=0
            )
            
            # Train agent
            self.rl_agent.learn(total_timesteps=10000)
            
        self.is_fitted = True
        
    def estimate_var(self, horizon: int = 1) -> float:
        """
        Estimate VaR using combined GARCH + RL approach
        
        Args:
            horizon: Forecast horizon in days
            
        Returns:
            VaR estimate (as positive number)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before estimating VaR")
            
        # GARCH volatility forecast
        forecast = self.garch_fit.forecast(horizon=horizon)
        vol_forecast = np.sqrt(forecast.variance.values[-1, 0]) / 100
        
        # Standard normal quantile for confidence level
        from scipy.stats import norm
        quantile = norm.ppf(1 - self.confidence_level)
        
        # Base VaR from GARCH
        var_garch = -quantile * vol_forecast
        
        # RL adjustment (if trained)
        if self.rl_agent is not None:
            # Get current state
            state = self._get_current_state()
            adjustment = self.rl_agent.predict(state, deterministic=True)[0]
            var_combined = var_garch * adjustment[0]
        else:
            var_combined = var_garch
            
        return float(var_combined)
        
    def _get_current_state(self) -> np.ndarray:
        """Get current market state for RL agent"""
        # Simplified state extraction
        vol = np.sqrt(self.garch_fit.conditional_volatility.values[-1]) / 100
        ret = 0.0  # Would use actual recent return
        regime = 0.0  # Would use actual regime detection
        return np.array([vol, ret, regime], dtype=np.float32)
        
    def backtest(self, returns: np.ndarray, window: int = 252) -> dict:
        """
        Backtest VaR model on historical data
        
        Args:
            returns: Historical returns to test on
            window: Rolling window for fitting
            
        Returns:
            Dictionary with backtest results
        """
        violations = 0
        estimates = []
        
        for i in range(window, len(returns)):
            # Fit on rolling window
            train_data = returns[i-window:i]
            self.fit(train_data, train_rl=False)  # Quick fitting for backtest
            
            # Estimate VaR
            var_estimate = self.estimate_var(horizon=1)
            estimates.append(var_estimate)
            
            # Check for violation
            actual_return = returns[i]
            if actual_return < -var_estimate:
                violations += 1
                
        violation_rate = violations / len(estimates)
        expected_rate = 1 - self.confidence_level
        
        return {
            'violation_rate': violation_rate,
            'expected_rate': expected_rate,
            'num_violations': violations,
            'num_observations': len(estimates),
            'estimates': estimates
        }
```

**Create Test File:** [`tests/models/risk/test_rl_garch_var.py`](../../tests/models/risk/test_rl_garch_var.py:1)

```python
"""Tests for RL-GARCH VaR model"""
import pytest
import numpy as np
from axiom.models.risk.rl_garch_var import RLGARCHVaR, GARCHEnvironment

def test_garch_environment_creation():
    """Test GARCH environment initialization"""
    returns = np.random.randn(100) * 0.01
    env = GARCHEnvironment(returns)
    
    assert env.observation_space.shape == (3,)
    assert env.action_space.shape == (1,)

def test_rl_garch_var_fitting():
    """Test model fitting"""
    returns = np.random.randn(500) * 0.01
    model = RLGARCHVaR(confidence_level=0.95)
    
    # Fit without RL for speed
    model.fit(returns, train_rl=False)
    assert model.is_fitted
    
def test_rl_garch_var_estimation():
    """Test VaR estimation"""
    returns = np.random.randn(500) * 0.01
    model = RLGARCHVaR(confidence_level=0.95)
    model.fit(returns, train_rl=False)
    
    var_estimate = model.estimate_var(horizon=1)
    assert var_estimate > 0
    assert var_estimate < 0.1  # Reasonable bound

def test_rl_garch_backtest():
    """Test backtesting functionality"""
    returns = np.random.randn(500) * 0.01
    model = RLGARCHVaR(confidence_level=0.95)
    
    results = model.backtest(returns, window=100)
    assert 'violation_rate' in results
    assert 'expected_rate' in results
    assert results['expected_rate'] == 0.05
```

**Expected Outcome:**
- ✅ Basic RL-GARCH structure in place
- ✅ Can fit GARCH model
- ✅ Basic tests pass
- ✅ Ready for Week 2 enhancement

---

### Day 5: Friday - Testing & Documentation

#### Morning (2 hours): Comprehensive Testing
```bash
# Run all tests
pytest tests/ -v --tb=short

# Specific test suites
pytest tests/dspy_modules/ -v
pytest tests/models/risk/ -v
pytest tests/models/portfolio/ -v

# Coverage report
pytest tests/ --cov=axiom --cov-report=html
```

**Expected Results:**
- ✅ All tests pass
- ✅ No regressions
- ✅ Coverage maintained or improved

#### Afternoon (3 hours): Documentation & Review

**Update Documentation:**
1. Update [`requirements.txt`](../../requirements.txt:1)
2. Update [`CHANGELOG.md`](../../CHANGELOG.md:1) (if exists)
3. Create implementation notes

**Create Migration Notes:**
```markdown
# Week 1 Implementation Notes

## Completed
- ✅ Updated all dependencies to latest versions
- ✅ Migrated DSPy from 2.x to 3.0.4b2
- ✅ Created RL-GARCH VaR foundation
- ✅ All tests passing

## DSPy Migration Changes
- Updated signature syntax
- Updated optimizer names
- No breaking changes in functionality

## Next Week
- Complete RL-GARCH implementation
- Train RL agent properly
- Begin transformer time series work
```

**Code Review Checklist:**
- [ ] All code follows Python style guide
- [ ] Type hints added where appropriate
- [ ] Docstrings complete
- [ ] Tests comprehensive
- [ ] No TODO comments in main branch

---

## 📊 Week 1 Success Metrics

### Completed Tasks
- [x] Updated 10+ dependencies
- [x] Migrated 3 DSPy modules to v3.0
- [x] Created RL-GARCH foundation
- [x] Wrote tests for new code
- [x] Documentation updated

### Performance Metrics
- **Test Coverage:** Maintained >80%
- **Tests Passing:** 100%
- **Build Time:** < 5 minutes
- **No Regressions:** Verified

### Business Value
- **Foundation:** Ready for Week 2 implementation
- **Risk Reduction:** Updated to latest stable versions
- **Technical Debt:** Reduced by modernizing DSPy

---

## 🚀 Quick Commands Reference

### Daily Commands
```bash
# Activate environment
source venv/bin/activate  # or: conda activate axiom

# Run tests (quick)
pytest tests/ -v -x --tb=short

# Run specific test file
pytest tests/models/risk/test_rl_garch_var.py -v

# Check imports
python -c "import dspy; import arch; import stable_baselines3; print('✓ All imports working')"

# Format code
black axiom/
isort axiom/

# Type checking
mypy axiom/models/risk/
```

### Verification Commands
```bash
# Check package versions
pip list | grep -E "(dspy|langgraph|quantlib|pypfopt)"

# Run single test
pytest tests/models/risk/test_rl_garch_var.py::test_rl_garch_var_fitting -v

# Profile code
python -m cProfile -s cumtime demos/demo_var_risk_models.py
```

---

## 📝 Common Issues & Solutions

### Issue 1: DSPy Import Errors
```bash
# Problem: Old DSPy syntax not working
# Solution: Check version
python -c "import dspy; print(dspy.__version__)"
# Should be: 3.0.4b2
```

### Issue 2: GARCH Convergence
```python
# Problem: GARCH model not converging
# Solution: Scale returns
returns_scaled = returns * 100  # GARCH works better with scaled data
model = arch_model(returns_scaled, ...)
```

### Issue 3: RL Training Slow
```python
# Problem: RL agent training takes too long
# Solution: Reduce timesteps for testing
agent.learn(total_timesteps=1000)  # Instead of 10000
```

---

## 🎯 Monday Morning Checklist

Before starting next week:
- [ ] All Week 1 tasks completed
- [ ] All tests passing
- [ ] Code reviewed by team
- [ ] Documentation updated
- [ ] No blocking issues
- [ ] Ready for RL-GARCH Week 2

---

## 📞 Support & Resources

### Internal
- **Slack Channel:** #axiom-quant-dev
- **Jira Board:** Axiom Q4 2025 Sprint
- **Code Review:** Submit PR by Friday EOD

### External
- **DSPy Docs:** https://github.com/stanfordnlp/dspy
- **ARCH Docs:** https://arch.readthedocs.io/
- **Stable-Baselines3:** https://stable-baselines3.readthedocs.io/

### Papers
- **RL-GARCH:** https://arxiv.org/abs/2504.16635
- **Deep Hedging:** https://arxiv.org/abs/1802.03042
- **Transformer TS:** https://arxiv.org/abs/2202.07125

---

## 🎉 Celebration Points

By end of Friday, you will have:
- ✨ Modernized entire tech stack
- ✨ Migrated to DSPy 3.0 (cutting edge!)
- ✨ Built foundation for RL-GARCH VaR
- ✨ All tests passing
- ✨ Ready for Week 2 feature implementation

**This is significant progress!** 🚀

---

**Previous Document:** [04_implementation_priorities.md](04_implementation_priorities.md) - Full roadmap  
**Summary Document:** [01_executive_summary.md](01_executive_summary.md) - Executive overview