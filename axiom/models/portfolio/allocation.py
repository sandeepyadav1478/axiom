"""
Asset Allocation Models for Portfolio Management

Implements strategic and tactical asset allocation strategies:
1. Strategic Asset Allocation - Long-term policy portfolio
2. Tactical Asset Allocation - Active deviations from policy
3. Risk Parity - Equal risk contribution allocation
4. Black-Litterman - Combines market equilibrium with investor views
5. Hierarchical Risk Parity - Cluster-based diversification
6. VaR-Constrained Allocation - Risk-budget aware allocation

Designed for:
- Institutional investors and pension funds
- Wealth managers and financial advisors
- Multi-asset portfolio managers
- Risk-controlled investment strategies

Integrates with portfolio optimization and VaR models.
"""

import numpy as np
import pandas as pd
from scipy import cluster
from scipy.spatial.distance import squareform
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .optimization import (
    PortfolioOptimizer,
    OptimizationMethod,
    PortfolioMetrics,
    OptimizationResult
)


class AllocationStrategy(Enum):
    """Asset allocation strategies."""
    EQUAL_WEIGHT = "equal_weight"  # 1/N allocation
    MARKET_CAP = "market_cap"  # Market capitalization weighted
    RISK_PARITY = "risk_parity"  # Equal risk contribution
    MIN_VARIANCE = "min_variance"  # Minimum variance
    MAX_SHARPE = "max_sharpe"  # Maximum Sharpe ratio
    BLACK_LITTERMAN = "black_litterman"  # Black-Litterman model
    HIERARCHICAL_RISK_PARITY = "hrp"  # Hierarchical risk parity
    VAR_CONSTRAINED = "var_constrained"  # VaR budget constrained


@dataclass
class AssetClass:
    """Asset class definition."""
    
    name: str  # Asset class name (e.g., "US Equities", "Bonds")
    symbols: List[str]  # Securities in this asset class
    strategic_weight: float  # Long-term policy weight
    min_weight: float = 0.0  # Minimum allowed weight
    max_weight: float = 1.0  # Maximum allowed weight
    expected_return: Optional[float] = None  # Expected annual return
    expected_volatility: Optional[float] = None  # Expected annual volatility
    
    def __post_init__(self):
        """Validate asset class parameters."""
        if not 0 <= self.strategic_weight <= 1:
            raise ValueError(f"Strategic weight must be between 0 and 1, got {self.strategic_weight}")
        if not 0 <= self.min_weight <= self.max_weight <= 1:
            raise ValueError(f"Invalid weight bounds: [{self.min_weight}, {self.max_weight}]")


@dataclass
class AllocationResult:
    """Asset allocation result."""
    
    weights: Dict[str, float]  # Asset weights
    strategy: AllocationStrategy  # Strategy used
    asset_class_weights: Optional[Dict[str, float]] = None  # Asset class level weights
    metrics: Optional[PortfolioMetrics] = None  # Portfolio metrics
    rebalancing_required: bool = False  # Whether rebalancing is needed
    tracking_error: Optional[float] = None  # Tracking error vs policy
    var_utilization: Optional[float] = None  # VaR budget utilization
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "weights": self.weights,
            "strategy": self.strategy.value,
            "asset_class_weights": self.asset_class_weights,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "rebalancing_required": self.rebalancing_required,
            "tracking_error": self.tracking_error,
            "var_utilization": self.var_utilization,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }
    
    def get_top_holdings(self, n: int = 10) -> Dict[str, float]:
        """Get top N holdings by weight."""
        sorted_weights = sorted(self.weights.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_weights[:n])


class AssetAllocator:
    """
    Multi-strategy asset allocation engine.
    
    Features:
    - Multiple allocation strategies
    - Asset class constraints
    - Rebalancing logic
    - VaR budget integration
    - Performance attribution
    """
    
    def __init__(
        self,
        asset_classes: Optional[List[AssetClass]] = None,
        risk_free_rate: float = 0.02,
        rebalancing_threshold: float = 0.05
    ):
        """
        Initialize asset allocator.
        
        Args:
            asset_classes: List of asset class definitions
            risk_free_rate: Annual risk-free rate
            rebalancing_threshold: Threshold for rebalancing trigger
        """
        self.asset_classes = asset_classes or []
        self.risk_free_rate = risk_free_rate
        self.rebalancing_threshold = rebalancing_threshold
        self.optimizer = PortfolioOptimizer(risk_free_rate=risk_free_rate)
        self.allocation_history: List[AllocationResult] = []
    
    def allocate(
        self,
        returns: Union[pd.DataFrame, np.ndarray],
        strategy: AllocationStrategy = AllocationStrategy.RISK_PARITY,
        current_weights: Optional[Dict[str, float]] = None,
        market_caps: Optional[Dict[str, float]] = None,
        var_budget: Optional[float] = None,
        **kwargs
    ) -> AllocationResult:
        """
        Calculate asset allocation using specified strategy.
        
        Args:
            returns: Historical returns (assets in columns)
            strategy: Allocation strategy
            current_weights: Current portfolio weights
            market_caps: Market capitalizations for market-cap weighting
            var_budget: VaR budget constraint
            **kwargs: Strategy-specific parameters
        
        Returns:
            AllocationResult with optimal allocation
        """
        # Convert to DataFrame if needed
        if isinstance(returns, np.ndarray):
            assets = [f"Asset_{i}" for i in range(returns.shape[1])]
            returns = pd.DataFrame(returns, columns=assets)
        
        assets = list(returns.columns)
        n_assets = len(assets)
        
        # Calculate allocation based on strategy
        if strategy == AllocationStrategy.EQUAL_WEIGHT:
            weights_array = self._equal_weight(n_assets)
            
        elif strategy == AllocationStrategy.MARKET_CAP:
            if market_caps is None:
                raise ValueError("Market caps required for market-cap weighting")
            weights_array = self._market_cap_weight(assets, market_caps)
            
        elif strategy == AllocationStrategy.RISK_PARITY:
            weights_array = self._risk_parity(returns)
            
        elif strategy == AllocationStrategy.MIN_VARIANCE:
            result = self.optimizer.optimize(
                returns, method=OptimizationMethod.MIN_VOLATILITY
            )
            weights_array = result.weights
            
        elif strategy == AllocationStrategy.MAX_SHARPE:
            result = self.optimizer.optimize(
                returns, method=OptimizationMethod.MAX_SHARPE
            )
            weights_array = result.weights
            
        elif strategy == AllocationStrategy.BLACK_LITTERMAN:
            market_caps = market_caps or {}
            views = kwargs.get('views', {})
            view_confidences = kwargs.get('view_confidences', {})
            weights_array = self._black_litterman(
                returns, assets, market_caps, views, view_confidences
            )
            
        elif strategy == AllocationStrategy.HIERARCHICAL_RISK_PARITY:
            weights_array = self._hierarchical_risk_parity(returns)
            
        elif strategy == AllocationStrategy.VAR_CONSTRAINED:
            if var_budget is None:
                raise ValueError("VaR budget required for VaR-constrained allocation")
            weights_array = self._var_constrained(returns, var_budget)
            
        else:
            raise ValueError(f"Unknown allocation strategy: {strategy}")
        
        # Convert to dictionary
        weights_dict = dict(zip(assets, weights_array))
        
        # Calculate metrics
        metrics = self.optimizer.calculate_metrics(
            weights_array, returns.values
        )
        
        # Check rebalancing need
        rebalancing_required = False
        tracking_error = None
        if current_weights:
            rebalancing_required = self._check_rebalancing(
                current_weights, weights_dict
            )
            tracking_error = self._calculate_tracking_error(
                current_weights, weights_dict, returns
            )
        
        # Calculate asset class weights if defined
        asset_class_weights = None
        if self.asset_classes:
            asset_class_weights = self._aggregate_to_asset_classes(weights_dict)
        
        result = AllocationResult(
            weights=weights_dict,
            strategy=strategy,
            asset_class_weights=asset_class_weights,
            metrics=metrics,
            rebalancing_required=rebalancing_required,
            tracking_error=tracking_error,
            metadata={
                "n_assets": n_assets,
                "strategy_params": kwargs
            }
        )
        
        self.allocation_history.append(result)
        return result
    
    def _equal_weight(self, n_assets: int) -> np.ndarray:
        """Equal weight (1/N) allocation."""
        return np.ones(n_assets) / n_assets
    
    def _market_cap_weight(
        self,
        assets: List[str],
        market_caps: Dict[str, float]
    ) -> np.ndarray:
        """Market capitalization weighted allocation."""
        caps = np.array([market_caps.get(asset, 0) for asset in assets])
        total_cap = np.sum(caps)
        return caps / total_cap if total_cap > 0 else self._equal_weight(len(assets))
    
    def _risk_parity(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Risk parity allocation (equal risk contribution).
        
        Each asset contributes equally to portfolio risk.
        """
        result = self.optimizer.optimize(
            returns, method=OptimizationMethod.RISK_PARITY
        )
        return result.weights
    
    def _black_litterman(
        self,
        returns: pd.DataFrame,
        assets: List[str],
        market_caps: Dict[str, float],
        views: Dict[str, float],
        view_confidences: Dict[str, float]
    ) -> np.ndarray:
        """
        Black-Litterman model allocation.
        
        Combines market equilibrium with investor views.
        
        Args:
            returns: Historical returns
            assets: Asset names
            market_caps: Market capitalizations
            views: Dict of {asset: expected_return}
            view_confidences: Dict of {asset: confidence_level}
        
        Returns:
            Optimal weights
        """
        # Market equilibrium weights (market cap weighted)
        market_weights = self._market_cap_weight(assets, market_caps)
        
        # Calculate implied returns (reverse optimization)
        cov_matrix = returns.cov().values * 252  # Annualized
        risk_aversion = 2.5  # Typical risk aversion parameter
        
        # Implied returns: Π = λ Σ w_market
        implied_returns = risk_aversion * cov_matrix @ market_weights
        
        # If no views, return market portfolio
        if not views:
            return market_weights
        
        # Construct views matrix and vector
        n_assets = len(assets)
        n_views = len(views)
        
        P = np.zeros((n_views, n_assets))  # Views matrix
        Q = np.zeros(n_views)  # Views vector
        
        for i, (asset, view_return) in enumerate(views.items()):
            if asset in assets:
                asset_idx = assets.index(asset)
                P[i, asset_idx] = 1.0
                Q[i] = view_return
        
        # View uncertainty (Omega)
        # Higher confidence = lower uncertainty
        tau = 0.025  # Scaling factor
        Omega = np.diag([
            tau * cov_matrix[assets.index(asset), assets.index(asset)] / view_confidences.get(asset, 1.0)
            for asset in views.keys() if asset in assets
        ])
        
        # Black-Litterman formula
        # Posterior returns = [(τΣ)^-1 + P'Ω^-1P]^-1 [(τΣ)^-1 Π + P'Ω^-1 Q]
        tau_cov_inv = np.linalg.inv(tau * cov_matrix)
        P_omega_inv_P = P.T @ np.linalg.inv(Omega) @ P
        
        posterior_cov_inv = tau_cov_inv + P_omega_inv_P
        posterior_cov = np.linalg.inv(posterior_cov_inv)
        
        posterior_returns = posterior_cov @ (
            tau_cov_inv @ implied_returns + P.T @ np.linalg.inv(Omega) @ Q
        )
        
        # Optimize with posterior returns
        # w* = (λΣ)^-1 E[R]
        optimal_weights = np.linalg.inv(risk_aversion * cov_matrix) @ posterior_returns
        
        # Normalize to sum to 1
        optimal_weights = optimal_weights / np.sum(optimal_weights)
        
        # Ensure non-negative (long-only)
        optimal_weights = np.maximum(optimal_weights, 0)
        optimal_weights = optimal_weights / np.sum(optimal_weights)
        
        return optimal_weights
    
    def _hierarchical_risk_parity(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Hierarchical Risk Parity (HRP) allocation.
        
        Uses hierarchical clustering for diversification.
        More robust than traditional mean-variance.
        """
        # Calculate correlation matrix
        corr_matrix = returns.corr().values
        
        # Convert correlation to distance
        dist_matrix = np.sqrt((1 - corr_matrix) / 2)
        
        # Hierarchical clustering
        dist_condensed = squareform(dist_matrix, checks=False)
        linkage = cluster.hierarchy.linkage(dist_condensed, method='single')
        
        # Get sorted assets (quasi-diagonalization)
        sorted_indices = cluster.hierarchy.leaves_list(linkage)
        
        # Calculate covariance
        cov_matrix = returns.cov().values
        
        # Recursive bisection for weight allocation
        weights = self._hrp_recursive_bisection(
            cov_matrix, sorted_indices
        )
        
        return weights
    
    def _hrp_recursive_bisection(
        self,
        cov_matrix: np.ndarray,
        cluster_indices: np.ndarray
    ) -> np.ndarray:
        """Recursive bisection for HRP weights."""
        n_assets = len(cluster_indices)
        weights = np.zeros(cov_matrix.shape[0])  # Full size for indexing
        
        # Base case: single asset
        if n_assets == 1:
            weights[cluster_indices[0]] = 1.0
            return weights
        
        # Split cluster in half
        mid = n_assets // 2
        left_indices = cluster_indices[:mid]
        right_indices = cluster_indices[mid:]
        
        # Calculate cluster variances
        left_var = self._cluster_variance(cov_matrix, left_indices)
        right_var = self._cluster_variance(cov_matrix, right_indices)
        
        # Allocate weights inversely proportional to variance
        left_weight = 1.0 - left_var / (left_var + right_var)
        right_weight = 1.0 - left_weight
        
        # Recursively allocate within each cluster
        left_sub_weights = self._hrp_recursive_bisection(cov_matrix, left_indices)
        right_sub_weights = self._hrp_recursive_bisection(cov_matrix, right_indices)
        
        weights = left_weight * left_sub_weights + right_weight * right_sub_weights
        
        return weights
    
    def _cluster_variance(
        self,
        cov_matrix: np.ndarray,
        indices: np.ndarray
    ) -> float:
        """Calculate variance of equally-weighted cluster."""
        n = len(indices)
        cluster_weights = np.zeros(cov_matrix.shape[0])
        cluster_weights[indices] = 1.0 / n
        
        variance = cluster_weights @ cov_matrix @ cluster_weights
        return variance
    
    def _var_constrained(
        self,
        returns: pd.DataFrame,
        var_budget: float
    ) -> np.ndarray:
        """
        VaR-constrained allocation.
        
        Maximizes expected return subject to VaR constraint.
        """
        from ..risk.var_models import VaRCalculator, VaRMethod
        
        n_assets = len(returns.columns)
        
        # Objective: maximize expected return
        mean_returns = returns.mean().values * 252
        
        def objective(weights):
            return -np.dot(weights, mean_returns)
        
        # VaR constraint
        var_calculator = VaRCalculator()
        
        def var_constraint(weights):
            portfolio_returns = returns.values @ weights
            var_result = var_calculator.calculate_var(
                portfolio_value=1.0,
                returns=portfolio_returns,
                method=VaRMethod.HISTORICAL,
                confidence_level=0.95
            )
            # Constraint: VaR <= budget
            return var_budget - var_result.var_percentage
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},  # Fully invested
            {'type': 'ineq', 'fun': var_constraint}  # VaR constraint
        ]
        
        # Bounds (long only)
        bounds = [(0.0, 1.0)] * n_assets
        
        # Initial guess (equal weight)
        initial_weights = np.ones(n_assets) / n_assets
        
        # Optimize
        from scipy import optimize as scipy_optimize
        result = scipy_optimize.minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        return result.x if result.success else initial_weights
    
    def _check_rebalancing(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float]
    ) -> bool:
        """Check if rebalancing is required."""
        for asset in current_weights.keys():
            current = current_weights.get(asset, 0)
            target = target_weights.get(asset, 0)
            
            if abs(current - target) > self.rebalancing_threshold:
                return True
        
        return False
    
    def _calculate_tracking_error(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        returns: pd.DataFrame
    ) -> float:
        """Calculate tracking error between current and target portfolios."""
        assets = list(returns.columns)
        
        current_array = np.array([current_weights.get(asset, 0) for asset in assets])
        target_array = np.array([target_weights.get(asset, 0) for asset in assets])
        
        # Active returns
        current_returns = returns.values @ current_array
        target_returns = returns.values @ target_array
        active_returns = current_returns - target_returns
        
        # Tracking error (annualized)
        tracking_error = np.std(active_returns) * np.sqrt(252)
        
        return tracking_error
    
    def _aggregate_to_asset_classes(
        self,
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """Aggregate security weights to asset class level."""
        asset_class_weights = {}
        
        for asset_class in self.asset_classes:
            total_weight = sum(
                weights.get(symbol, 0) for symbol in asset_class.symbols
            )
            asset_class_weights[asset_class.name] = total_weight
        
        return asset_class_weights
    
    def rebalance(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        transaction_cost: float = 0.001
    ) -> Dict:
        """
        Calculate rebalancing trades.
        
        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            transaction_cost: Transaction cost per trade (0.1% default)
        
        Returns:
            Dictionary with rebalancing details
        """
        trades = {}
        total_turnover = 0.0
        
        all_assets = set(current_weights.keys()) | set(target_weights.keys())
        
        for asset in all_assets:
            current = current_weights.get(asset, 0)
            target = target_weights.get(asset, 0)
            trade = target - current
            
            if abs(trade) > 1e-6:
                trades[asset] = trade
                total_turnover += abs(trade)
        
        # Calculate costs
        total_cost = total_turnover * transaction_cost
        
        return {
            "trades": trades,
            "turnover": total_turnover,
            "transaction_cost": total_cost,
            "net_benefit": self._calculate_rebalancing_benefit(
                current_weights, target_weights
            ) - total_cost
        }
    
    def _calculate_rebalancing_benefit(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float]
    ) -> float:
        """Estimate benefit of rebalancing (simplified)."""
        # This is a simplified calculation
        # In practice, would use expected returns and risk reduction
        weight_differences = [
            abs(target_weights.get(asset, 0) - current_weights.get(asset, 0))
            for asset in set(current_weights.keys()) | set(target_weights.keys())
        ]
        
        return sum(weight_differences) * 0.01  # Assume 1% benefit per unit of rebalancing


# Convenience functions
def equal_weight_allocation(assets: List[str]) -> Dict[str, float]:
    """Simple equal-weight allocation."""
    weight = 1.0 / len(assets)
    return {asset: weight for asset in assets}


def risk_parity_allocation(
    returns: Union[pd.DataFrame, np.ndarray]
) -> Dict[str, float]:
    """Quick risk parity allocation."""
    allocator = AssetAllocator()
    result = allocator.allocate(returns, strategy=AllocationStrategy.RISK_PARITY)
    return result.weights


# Export all components
__all__ = [
    "AllocationStrategy",
    "AssetClass",
    "AllocationResult",
    "AssetAllocator",
    "equal_weight_allocation",
    "risk_parity_allocation"
]