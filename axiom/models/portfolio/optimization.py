"""
Portfolio Optimization Models for Quantitative Trading and Asset Management

Implements industry-standard portfolio optimization techniques:
1. Markowitz Mean-Variance Optimization - Classic portfolio theory
2. Efficient Frontier - Risk-return trade-off visualization
3. Portfolio Performance Metrics - Sharpe, Sortino, Calmar ratios
4. Risk-Return Optimization - Maximize Sharpe ratio, minimize volatility
5. Asset Allocation - Strategic and tactical allocation strategies

Designed for:
- Quantitative traders and portfolio managers
- Hedge funds and institutional investors
- Wealth managers and financial advisors
- Risk managers and compliance teams

Integrates with VaR models for comprehensive risk management.
"""

import numpy as np
import pandas as pd
from scipy import optimize
from scipy import stats
from typing import List, Dict, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import warnings


class OptimizationMethod(Enum):
    """Portfolio optimization objectives."""
    MAX_SHARPE = "max_sharpe"  # Maximize Sharpe ratio
    MIN_VOLATILITY = "min_volatility"  # Minimize portfolio volatility
    MAX_RETURN = "max_return"  # Maximize expected return
    EFFICIENT_RETURN = "efficient_return"  # Target specific return
    EFFICIENT_RISK = "efficient_risk"  # Target specific risk
    RISK_PARITY = "risk_parity"  # Equal risk contribution
    MIN_CVaR = "min_cvar"  # Minimize Conditional VaR


class ConstraintType(Enum):
    """Portfolio constraint types."""
    LONG_ONLY = "long_only"  # No short selling (weights >= 0)
    FULLY_INVESTED = "fully_invested"  # Sum of weights = 1
    BOX = "box"  # Lower and upper bounds per asset
    SECTOR = "sector"  # Sector exposure limits
    TURNOVER = "turnover"  # Limit portfolio turnover


@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics."""
    
    expected_return: float  # Annualized expected return
    volatility: float  # Annualized volatility (standard deviation)
    sharpe_ratio: float  # Risk-adjusted return
    sortino_ratio: float  # Downside risk-adjusted return
    calmar_ratio: Optional[float] = None  # Return to max drawdown ratio
    max_drawdown: Optional[float] = None  # Maximum peak-to-trough decline
    var_95: Optional[float] = None  # 95% Value at Risk
    cvar_95: Optional[float] = None  # 95% Conditional VaR (Expected Shortfall)
    beta: Optional[float] = None  # Market beta (if benchmark provided)
    alpha: Optional[float] = None  # Jensen's alpha (if benchmark provided)
    information_ratio: Optional[float] = None  # Tracking error adjusted return
    treynor_ratio: Optional[float] = None  # Return per unit of systematic risk
    
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary."""
        return {
            "expected_return": self.expected_return,
            "volatility": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "max_drawdown": self.max_drawdown,
            "var_95": self.var_95,
            "cvar_95": self.cvar_95,
            "beta": self.beta,
            "alpha": self.alpha,
            "information_ratio": self.information_ratio,
            "treynor_ratio": self.treynor_ratio
        }
    
    def __str__(self) -> str:
        """Formatted string representation."""
        return (
            f"Portfolio Metrics:\n"
            f"  Expected Return: {self.expected_return*100:.2f}%\n"
            f"  Volatility: {self.volatility*100:.2f}%\n"
            f"  Sharpe Ratio: {self.sharpe_ratio:.3f}\n"
            f"  Sortino Ratio: {self.sortino_ratio:.3f}\n"
            f"  Max Drawdown: {self.max_drawdown*100:.2f}%" if self.max_drawdown else ""
        )


@dataclass
class OptimizationResult:
    """Portfolio optimization result."""
    
    weights: np.ndarray  # Optimal portfolio weights
    assets: List[str]  # Asset identifiers
    metrics: PortfolioMetrics  # Portfolio performance metrics
    method: OptimizationMethod  # Optimization method used
    success: bool  # Whether optimization succeeded
    message: str = ""  # Optimization status message
    constraints_satisfied: bool = True  # Whether all constraints are satisfied
    computation_time: float = 0.0  # Time taken for optimization
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def get_weights_dict(self) -> Dict[str, float]:
        """Get weights as dictionary mapping asset to weight."""
        return dict(zip(self.assets, self.weights))
    
    def to_dict(self) -> Dict:
        """Convert result to dictionary."""
        return {
            "weights": self.weights.tolist(),
            "assets": self.assets,
            "weights_dict": self.get_weights_dict(),
            "metrics": self.metrics.to_dict(),
            "method": self.method.value,
            "success": self.success,
            "message": self.message,
            "constraints_satisfied": self.constraints_satisfied,
            "computation_time": self.computation_time,
            "timestamp": self.timestamp
        }
    
    def __str__(self) -> str:
        """Formatted string representation."""
        weights_str = "\n".join([
            f"  {asset}: {weight*100:.2f}%"
            for asset, weight in self.get_weights_dict().items()
            if weight > 0.001  # Show only significant weights
        ])
        return (
            f"Optimization Result ({self.method.value}):\n"
            f"Status: {'Success' if self.success else 'Failed'}\n"
            f"Weights:\n{weights_str}\n"
            f"{self.metrics}"
        )


@dataclass
class EfficientFrontier:
    """Efficient frontier results."""
    
    returns: np.ndarray  # Expected returns for each portfolio
    risks: np.ndarray  # Volatilities for each portfolio
    sharpe_ratios: np.ndarray  # Sharpe ratios for each portfolio
    weights: np.ndarray  # Weights for each portfolio (2D array)
    assets: List[str]  # Asset identifiers
    risk_free_rate: float = 0.0  # Risk-free rate used
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def get_max_sharpe_portfolio(self) -> OptimizationResult:
        """Get portfolio with maximum Sharpe ratio."""
        max_sharpe_idx = np.argmax(self.sharpe_ratios)
        return OptimizationResult(
            weights=self.weights[max_sharpe_idx],
            assets=self.assets,
            metrics=PortfolioMetrics(
                expected_return=self.returns[max_sharpe_idx],
                volatility=self.risks[max_sharpe_idx],
                sharpe_ratio=self.sharpe_ratios[max_sharpe_idx],
                sortino_ratio=0.0  # Not computed in efficient frontier
            ),
            method=OptimizationMethod.MAX_SHARPE,
            success=True,
            message="Maximum Sharpe ratio portfolio from efficient frontier"
        )
    
    def get_min_volatility_portfolio(self) -> OptimizationResult:
        """Get minimum volatility portfolio."""
        min_vol_idx = np.argmin(self.risks)
        return OptimizationResult(
            weights=self.weights[min_vol_idx],
            assets=self.assets,
            metrics=PortfolioMetrics(
                expected_return=self.returns[min_vol_idx],
                volatility=self.risks[min_vol_idx],
                sharpe_ratio=self.sharpe_ratios[min_vol_idx],
                sortino_ratio=0.0
            ),
            method=OptimizationMethod.MIN_VOLATILITY,
            success=True,
            message="Minimum volatility portfolio from efficient frontier"
        )
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert efficient frontier to DataFrame."""
        return pd.DataFrame({
            'return': self.returns,
            'risk': self.risks,
            'sharpe_ratio': self.sharpe_ratios
        })
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "returns": self.returns.tolist(),
            "risks": self.risks.tolist(),
            "sharpe_ratios": self.sharpe_ratios.tolist(),
            "assets": self.assets,
            "risk_free_rate": self.risk_free_rate,
            "timestamp": self.timestamp
        }


class PortfolioOptimizer:
    """
    Portfolio optimization using Modern Portfolio Theory (MPT).
    
    Implements Markowitz mean-variance optimization and extensions.
    
    Features:
    - Multiple optimization objectives
    - Flexible constraint system
    - Efficient frontier generation
    - Integration with VaR models
    - Transaction cost modeling
    """
    
    def __init__(
        self,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ):
        """
        Initialize portfolio optimizer.
        
        Args:
            risk_free_rate: Annual risk-free rate (default: 2%)
            periods_per_year: Trading periods per year (252 for daily, 52 for weekly)
        """
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
        self.optimization_history: List[OptimizationResult] = []
    
    def optimize(
        self,
        returns: Union[pd.DataFrame, np.ndarray],
        assets: Optional[List[str]] = None,
        method: OptimizationMethod = OptimizationMethod.MAX_SHARPE,
        constraints: Optional[Dict] = None,
        target_return: Optional[float] = None,
        target_risk: Optional[float] = None,
        bounds: Optional[Tuple[float, float]] = (0.0, 1.0),
        initial_weights: Optional[np.ndarray] = None
    ) -> OptimizationResult:
        """
        Optimize portfolio weights.
        
        Args:
            returns: Historical returns (assets in columns, time in rows)
            assets: List of asset names (if returns is numpy array)
            method: Optimization objective
            constraints: Additional constraints dict
            target_return: Target return for efficient return optimization
            target_risk: Target risk for efficient risk optimization
            bounds: Weight bounds per asset (min, max)
            initial_weights: Starting weights for optimization
        
        Returns:
            OptimizationResult with optimal weights and metrics
        """
        import time
        start_time = time.time()
        
        # Convert returns to DataFrame if needed
        if isinstance(returns, np.ndarray):
            if assets is None:
                assets = [f"Asset_{i}" for i in range(returns.shape[1])]
            returns = pd.DataFrame(returns, columns=assets)
        else:
            assets = list(returns.columns)
        
        n_assets = len(assets)
        
        # Calculate expected returns and covariance
        mean_returns = returns.mean().values * self.periods_per_year
        cov_matrix = returns.cov().values * self.periods_per_year
        
        # Initial weights
        if initial_weights is None:
            initial_weights = np.array([1.0 / n_assets] * n_assets)
        
        # Set up constraints
        cons = self._setup_constraints(
            n_assets, constraints, target_return, target_risk, mean_returns
        )
        
        # Set up bounds
        if bounds is not None:
            bounds_tuple = [bounds] * n_assets
        else:
            bounds_tuple = [(0.0, 1.0)] * n_assets
        
        # Optimization objective function
        if method == OptimizationMethod.MAX_SHARPE:
            # Minimize negative Sharpe ratio
            objective = lambda w: -self._portfolio_sharpe(w, mean_returns, cov_matrix)
        elif method == OptimizationMethod.MIN_VOLATILITY:
            objective = lambda w: self._portfolio_volatility(w, cov_matrix)
        elif method == OptimizationMethod.MAX_RETURN:
            objective = lambda w: -self._portfolio_return(w, mean_returns)
        elif method == OptimizationMethod.EFFICIENT_RETURN:
            # Minimize volatility for target return
            objective = lambda w: self._portfolio_volatility(w, cov_matrix)
        elif method == OptimizationMethod.EFFICIENT_RISK:
            # Maximize return for target risk
            objective = lambda w: -self._portfolio_return(w, mean_returns)
        elif method == OptimizationMethod.RISK_PARITY:
            objective = lambda w: self._risk_parity_objective(w, cov_matrix)
        elif method == OptimizationMethod.MIN_CVaR:
            # Minimize Conditional VaR
            objective = lambda w: self._portfolio_cvar(w, returns.values)
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        # Run optimization
        try:
            result = optimize.minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds_tuple,
                constraints=cons,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            
            optimal_weights = result.x
            success = result.success
            message = result.message
            
        except Exception as e:
            warnings.warn(f"Optimization failed: {str(e)}")
            optimal_weights = initial_weights
            success = False
            message = f"Optimization error: {str(e)}"
        
        # Calculate portfolio metrics
        metrics = self.calculate_metrics(
            optimal_weights,
            returns.values,
            mean_returns,
            cov_matrix
        )
        
        # Create result
        optimization_result = OptimizationResult(
            weights=optimal_weights,
            assets=assets,
            metrics=metrics,
            method=method,
            success=success,
            message=message,
            constraints_satisfied=self._check_constraints(optimal_weights),
            computation_time=time.time() - start_time
        )
        
        # Store in history
        self.optimization_history.append(optimization_result)
        
        return optimization_result
    
    def calculate_efficient_frontier(
        self,
        returns: Union[pd.DataFrame, np.ndarray],
        assets: Optional[List[str]] = None,
        n_points: int = 100,
        bounds: Optional[Tuple[float, float]] = (0.0, 1.0)
    ) -> EfficientFrontier:
        """
        Calculate the efficient frontier.
        
        Args:
            returns: Historical returns
            assets: Asset names
            n_points: Number of points on the frontier
            bounds: Weight bounds per asset
        
        Returns:
            EfficientFrontier object with frontier portfolios
        """
        # Convert to DataFrame if needed
        if isinstance(returns, np.ndarray):
            if assets is None:
                assets = [f"Asset_{i}" for i in range(returns.shape[1])]
            returns = pd.DataFrame(returns, columns=assets)
        else:
            assets = list(returns.columns)
        
        # Calculate statistics
        mean_returns = returns.mean().values * self.periods_per_year
        cov_matrix = returns.cov().values * self.periods_per_year
        
        # Find min and max returns
        min_vol_result = self.optimize(
            returns, assets, OptimizationMethod.MIN_VOLATILITY, bounds=bounds
        )
        max_return_result = self.optimize(
            returns, assets, OptimizationMethod.MAX_RETURN, bounds=bounds
        )
        
        min_return = min_vol_result.metrics.expected_return
        max_return = max_return_result.metrics.expected_return
        
        # Generate target returns
        target_returns = np.linspace(min_return, max_return, n_points)
        
        # Calculate efficient portfolios
        frontier_returns = []
        frontier_risks = []
        frontier_sharpe = []
        frontier_weights = []
        
        for target_ret in target_returns:
            try:
                result = self.optimize(
                    returns,
                    assets,
                    OptimizationMethod.EFFICIENT_RETURN,
                    target_return=target_ret,
                    bounds=bounds
                )
                
                if result.success:
                    frontier_returns.append(result.metrics.expected_return)
                    frontier_risks.append(result.metrics.volatility)
                    frontier_sharpe.append(result.metrics.sharpe_ratio)
                    frontier_weights.append(result.weights)
            except:
                continue
        
        return EfficientFrontier(
            returns=np.array(frontier_returns),
            risks=np.array(frontier_risks),
            sharpe_ratios=np.array(frontier_sharpe),
            weights=np.array(frontier_weights),
            assets=assets,
            risk_free_rate=self.risk_free_rate
        )
    
    def calculate_metrics(
        self,
        weights: np.ndarray,
        returns: np.ndarray,
        mean_returns: Optional[np.ndarray] = None,
        cov_matrix: Optional[np.ndarray] = None,
        benchmark_returns: Optional[np.ndarray] = None
    ) -> PortfolioMetrics:
        """
        Calculate comprehensive portfolio performance metrics.
        
        Args:
            weights: Portfolio weights
            returns: Historical returns (time x assets)
            mean_returns: Expected returns (if pre-computed)
            cov_matrix: Covariance matrix (if pre-computed)
            benchmark_returns: Benchmark returns for relative metrics
        
        Returns:
            PortfolioMetrics with all calculated metrics
        """
        # Calculate mean returns and covariance if not provided
        if mean_returns is None:
            mean_returns = np.mean(returns, axis=0) * self.periods_per_year
        if cov_matrix is None:
            cov_matrix = np.cov(returns.T) * self.periods_per_year
        
        # Portfolio returns
        portfolio_returns = returns @ weights
        
        # Expected return and volatility
        expected_return = self._portfolio_return(weights, mean_returns)
        volatility = self._portfolio_volatility(weights, cov_matrix)
        
        # Sharpe ratio
        sharpe_ratio = (expected_return - self.risk_free_rate) / volatility if volatility > 0 else 0.0
        
        # Sortino ratio (downside deviation)
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_std = np.std(downside_returns) * np.sqrt(self.periods_per_year) if len(downside_returns) > 0 else volatility
        sortino_ratio = (expected_return - self.risk_free_rate) / downside_std if downside_std > 0 else 0.0
        
        # Maximum drawdown
        cumulative_returns = np.cumprod(1 + portfolio_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0.0
        
        # Calmar ratio
        calmar_ratio = -expected_return / max_drawdown if max_drawdown < 0 else 0.0
        
        # VaR and CVaR
        var_95 = np.percentile(portfolio_returns, 5)
        cvar_95 = np.mean(portfolio_returns[portfolio_returns <= var_95]) if np.any(portfolio_returns <= var_95) else var_95
        
        # Benchmark-relative metrics
        beta = None
        alpha = None
        information_ratio = None
        treynor_ratio = None
        
        if benchmark_returns is not None and len(benchmark_returns) == len(portfolio_returns):
            # Beta
            covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
            benchmark_variance = np.var(benchmark_returns)
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0
            
            # Alpha (Jensen's alpha)
            benchmark_return = np.mean(benchmark_returns) * self.periods_per_year
            alpha = expected_return - (self.risk_free_rate + beta * (benchmark_return - self.risk_free_rate))
            
            # Information ratio
            active_returns = portfolio_returns - benchmark_returns
            tracking_error = np.std(active_returns) * np.sqrt(self.periods_per_year)
            information_ratio = np.mean(active_returns) * self.periods_per_year / tracking_error if tracking_error > 0 else 0.0
            
            # Treynor ratio
            treynor_ratio = (expected_return - self.risk_free_rate) / beta if beta > 0 else 0.0
        
        return PortfolioMetrics(
            expected_return=expected_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            var_95=var_95,
            cvar_95=cvar_95,
            beta=beta,
            alpha=alpha,
            information_ratio=information_ratio,
            treynor_ratio=treynor_ratio
        )
    
    def _portfolio_return(self, weights: np.ndarray, mean_returns: np.ndarray) -> float:
        """Calculate portfolio expected return."""
        return np.dot(weights, mean_returns)
    
    def _portfolio_volatility(self, weights: np.ndarray, cov_matrix: np.ndarray) -> float:
        """Calculate portfolio volatility."""
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    def _portfolio_sharpe(self, weights: np.ndarray, mean_returns: np.ndarray, cov_matrix: np.ndarray) -> float:
        """Calculate portfolio Sharpe ratio."""
        ret = self._portfolio_return(weights, mean_returns)
        vol = self._portfolio_volatility(weights, cov_matrix)
        return (ret - self.risk_free_rate) / vol if vol > 0 else 0.0
    
    def _portfolio_cvar(self, weights: np.ndarray, returns: np.ndarray, alpha: float = 0.05) -> float:
        """Calculate portfolio Conditional VaR."""
        portfolio_returns = returns @ weights
        var = np.percentile(portfolio_returns, alpha * 100)
        cvar = np.mean(portfolio_returns[portfolio_returns <= var])
        return -cvar  # Return positive value for minimization
    
    def _risk_parity_objective(self, weights: np.ndarray, cov_matrix: np.ndarray) -> float:
        """Risk parity optimization objective."""
        # Risk contribution of each asset
        portfolio_vol = self._portfolio_volatility(weights, cov_matrix)
        marginal_contrib = np.dot(cov_matrix, weights)
        risk_contrib = weights * marginal_contrib / portfolio_vol
        
        # Minimize variance of risk contributions (equal risk)
        target_risk = portfolio_vol / len(weights)
        return np.sum((risk_contrib - target_risk) ** 2)
    
    def _setup_constraints(
        self,
        n_assets: int,
        constraints: Optional[Dict],
        target_return: Optional[float],
        target_risk: Optional[float],
        mean_returns: np.ndarray
    ) -> List[Dict]:
        """Setup optimization constraints."""
        cons = []
        
        # Weights sum to 1 (fully invested)
        cons.append({
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1.0
        })
        
        # Target return constraint
        if target_return is not None:
            cons.append({
                'type': 'eq',
                'fun': lambda w: self._portfolio_return(w, mean_returns) - target_return
            })
        
        # Add custom constraints if provided
        if constraints:
            for constraint_type, constraint_value in constraints.items():
                if constraint_type == 'max_weight':
                    # Maximum weight per asset
                    for i in range(n_assets):
                        cons.append({
                            'type': 'ineq',
                            'fun': lambda w, i=i: constraint_value - w[i]
                        })
                elif constraint_type == 'min_weight':
                    # Minimum weight per asset
                    for i in range(n_assets):
                        cons.append({
                            'type': 'ineq',
                            'fun': lambda w, i=i: w[i] - constraint_value
                        })
        
        return cons
    
    def _check_constraints(self, weights: np.ndarray) -> bool:
        """Check if portfolio weights satisfy basic constraints."""
        # Check weights sum to approximately 1
        if not np.isclose(np.sum(weights), 1.0, atol=1e-3):
            return False
        
        # Check non-negative weights (long-only)
        if np.any(weights < -1e-6):
            return False
        
        return True


# Convenience functions
def markowitz_optimization(
    returns: Union[pd.DataFrame, np.ndarray],
    risk_free_rate: float = 0.02,
    method: OptimizationMethod = OptimizationMethod.MAX_SHARPE
) -> OptimizationResult:
    """
    Quick Markowitz mean-variance optimization.
    
    Args:
        returns: Historical returns
        risk_free_rate: Risk-free rate
        method: Optimization objective
    
    Returns:
        OptimizationResult with optimal portfolio
    """
    optimizer = PortfolioOptimizer(risk_free_rate=risk_free_rate)
    return optimizer.optimize(returns, method=method)


def calculate_sharpe_ratio(
    returns: Union[np.ndarray, pd.Series],
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sharpe ratio for a return series.
    
    Args:
        returns: Return series
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading periods per year
    
    Returns:
        Sharpe ratio
    """
    returns_array = np.array(returns)
    excess_returns = returns_array - (risk_free_rate / periods_per_year)
    return np.mean(excess_returns) / np.std(returns_array) * np.sqrt(periods_per_year)


def calculate_sortino_ratio(
    returns: Union[np.ndarray, pd.Series],
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sortino ratio (downside risk-adjusted return).
    
    Args:
        returns: Return series
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading periods per year
    
    Returns:
        Sortino ratio
    """
    returns_array = np.array(returns)
    excess_returns = returns_array - (risk_free_rate / periods_per_year)
    downside_returns = excess_returns[excess_returns < 0]
    downside_std = np.std(downside_returns) if len(downside_returns) > 0 else np.std(returns_array)
    return np.mean(excess_returns) / downside_std * np.sqrt(periods_per_year)


def calculate_max_drawdown(returns: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate maximum drawdown.
    
    Args:
        returns: Return series
    
    Returns:
        Maximum drawdown (negative value)
    """
    returns_array = np.array(returns)
    cumulative = np.cumprod(1 + returns_array)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    return np.min(drawdown)


# Export all components
__all__ = [
    "OptimizationMethod",
    "ConstraintType",
    "PortfolioMetrics",
    "OptimizationResult",
    "EfficientFrontier",
    "PortfolioOptimizer",
    "markowitz_optimization",
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
    "calculate_max_drawdown"
]