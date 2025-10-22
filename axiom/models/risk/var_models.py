"""
Value at Risk (VaR) Models for Quantitative Trading and Risk Management

Implements three industry-standard VaR methodologies:
1. Parametric VaR - Analytical method using normal distribution
2. Historical Simulation VaR - Empirical method using historical returns
3. Monte Carlo VaR - Simulation method for complex portfolios

Designed for:
- Quantitative traders and hedge funds
- Risk management and compliance (Basel requirements)
- Portfolio managers and institutional investors
- Real-time risk monitoring and alerts
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum


class VaRMethod(Enum):
    """VaR calculation methodologies."""
    PARAMETRIC = "parametric"
    HISTORICAL = "historical"
    MONTE_CARLO = "monte_carlo"


class ConfidenceLevel(Enum):
    """Standard confidence levels for VaR calculations."""
    LEVEL_90 = 0.90  # 90% confidence (10% VaR)
    LEVEL_95 = 0.95  # 95% confidence (5% VaR) - Most common
    LEVEL_99 = 0.99  # 99% confidence (1% VaR) - Regulatory standard


@dataclass
class VaRResult:
    """Value at Risk calculation result."""
    
    var_amount: float  # VaR in currency units
    var_percentage: float  # VaR as percentage of portfolio value
    confidence_level: float  # Confidence level used (e.g., 0.95 for 95%)
    time_horizon_days: int  # Time horizon in days
    method: VaRMethod  # Calculation method used
    portfolio_value: float  # Current portfolio value
    expected_shortfall: Optional[float] = None  # ES/CVaR (average loss beyond VaR)
    calculation_timestamp: str = None
    metadata: Dict = None
    
    def __post_init__(self):
        if self.calculation_timestamp is None:
            self.calculation_timestamp = datetime.now().isoformat()
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict:
        """Convert VaR result to dictionary."""
        return {
            "var_amount": self.var_amount,
            "var_percentage": self.var_percentage,
            "confidence_level": self.confidence_level,
            "time_horizon_days": self.time_horizon_days,
            "method": self.method.value,
            "portfolio_value": self.portfolio_value,
            "expected_shortfall": self.expected_shortfall,
            "calculation_timestamp": self.calculation_timestamp,
            "metadata": self.metadata
        }
    
    def __str__(self) -> str:
        """Formatted string representation."""
        return (
            f"VaR ({self.method.value}, {self.confidence_level*100:.0f}% confidence, "
            f"{self.time_horizon_days}d): ${self.var_amount:,.2f} "
            f"({self.var_percentage*100:.2f}% of portfolio)"
        )


class ParametricVaR:
    """
    Parametric VaR (Variance-Covariance Method)
    
    Assumes normal distribution of returns.
    Fast and analytical, but less accurate for fat-tailed distributions.
    
    Formula: VaR = Portfolio_Value × Z_score × Volatility × √(Time_Horizon)
    """
    
    @staticmethod
    def calculate(
        portfolio_value: float,
        returns: Union[np.ndarray, pd.Series, List[float]],
        confidence_level: float = 0.95,
        time_horizon_days: int = 1
    ) -> VaRResult:
        """
        Calculate Parametric VaR using normal distribution assumption.
        
        Args:
            portfolio_value: Current portfolio value in currency units
            returns: Historical returns (can be daily, weekly, etc.)
            confidence_level: Confidence level (0.90, 0.95, 0.99)
            time_horizon_days: VaR time horizon in days
        
        Returns:
            VaRResult with calculated VaR and metadata
        """
        # Convert to numpy array
        returns_array = np.array(returns)
        
        # Calculate statistics
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array, ddof=1)  # Sample std dev
        
        # Get Z-score for confidence level
        z_score = stats.norm.ppf(1 - confidence_level)  # Negative value
        
        # Scale volatility for time horizon
        scaled_volatility = std_return * np.sqrt(time_horizon_days)
        
        # Calculate VaR (as positive value representing potential loss)
        var_percentage = abs((mean_return * time_horizon_days) + (z_score * scaled_volatility))
        var_amount = portfolio_value * var_percentage
        
        # Calculate Expected Shortfall (CVaR)
        es_percentage = ParametricVaR._calculate_expected_shortfall(
            mean_return, std_return, confidence_level, time_horizon_days
        )
        es_amount = portfolio_value * es_percentage
        
        return VaRResult(
            var_amount=var_amount,
            var_percentage=var_percentage,
            confidence_level=confidence_level,
            time_horizon_days=time_horizon_days,
            method=VaRMethod.PARAMETRIC,
            portfolio_value=portfolio_value,
            expected_shortfall=es_amount,
            metadata={
                "mean_return": mean_return,
                "volatility": std_return,
                "scaled_volatility": scaled_volatility,
                "z_score": z_score,
                "sample_size": len(returns_array)
            }
        )
    
    @staticmethod
    def _calculate_expected_shortfall(
        mean_return: float,
        std_return: float,
        confidence_level: float,
        time_horizon: int
    ) -> float:
        """Calculate Expected Shortfall (CVaR) for parametric method."""
        z_score = stats.norm.ppf(1 - confidence_level)
        scaled_vol = std_return * np.sqrt(time_horizon)
        
        # ES = μ + σ × φ(z) / (1 - α) where φ is PDF, α is confidence
        pdf_at_z = stats.norm.pdf(z_score)
        es = abs((mean_return * time_horizon) - (scaled_vol * pdf_at_z / (1 - confidence_level)))
        
        return es


class HistoricalSimulationVaR:
    """
    Historical Simulation VaR
    
    Uses actual historical returns distribution.
    No distribution assumptions, captures fat tails and skewness.
    Simple and intuitive, but limited by historical data availability.
    """
    
    @staticmethod
    def calculate(
        portfolio_value: float,
        returns: Union[np.ndarray, pd.Series, List[float]],
        confidence_level: float = 0.95,
        time_horizon_days: int = 1
    ) -> VaRResult:
        """
        Calculate Historical Simulation VaR using empirical distribution.
        
        Args:
            portfolio_value: Current portfolio value
            returns: Historical returns
            confidence_level: Confidence level
            time_horizon_days: VaR time horizon
        
        Returns:
            VaRResult with calculated VaR
        """
        returns_array = np.array(returns)
        
        # For multi-day horizon, aggregate returns if needed
        if time_horizon_days > 1:
            # Compound returns for multi-day periods
            aggregated_returns = HistoricalSimulationVaR._aggregate_returns(
                returns_array, time_horizon_days
            )
        else:
            aggregated_returns = returns_array
        
        # Calculate VaR as percentile of historical losses
        var_percentile = (1 - confidence_level) * 100
        var_percentage = abs(np.percentile(aggregated_returns, var_percentile))
        var_amount = portfolio_value * var_percentage
        
        # Calculate Expected Shortfall (average of losses beyond VaR)
        es_percentage = HistoricalSimulationVaR._calculate_expected_shortfall(
            aggregated_returns, confidence_level
        )
        es_amount = portfolio_value * es_percentage
        
        return VaRResult(
            var_amount=var_amount,
            var_percentage=var_percentage,
            confidence_level=confidence_level,
            time_horizon_days=time_horizon_days,
            method=VaRMethod.HISTORICAL,
            portfolio_value=portfolio_value,
            expected_shortfall=es_amount,
            metadata={
                "sample_size": len(returns_array),
                "aggregated_sample_size": len(aggregated_returns),
                "min_return": float(np.min(aggregated_returns)),
                "max_return": float(np.max(aggregated_returns)),
                "mean_return": float(np.mean(aggregated_returns)),
                "median_return": float(np.median(aggregated_returns))
            }
        )
    
    @staticmethod
    def _aggregate_returns(returns: np.ndarray, period_days: int) -> np.ndarray:
        """Aggregate returns for multi-day periods."""
        if period_days == 1:
            return returns
        
        # Rolling window aggregation
        aggregated = []
        for i in range(len(returns) - period_days + 1):
            period_return = np.prod(1 + returns[i:i+period_days]) - 1
            aggregated.append(period_return)
        
        return np.array(aggregated)
    
    @staticmethod
    def _calculate_expected_shortfall(
        returns: np.ndarray,
        confidence_level: float
    ) -> float:
        """Calculate Expected Shortfall for historical method."""
        var_percentile = (1 - confidence_level) * 100
        var_threshold = np.percentile(returns, var_percentile)
        
        # ES is average of all losses worse than VaR
        tail_losses = returns[returns <= var_threshold]
        es = abs(np.mean(tail_losses)) if len(tail_losses) > 0 else abs(var_threshold)
        
        return es


class MonteCarloVaR:
    """
    Monte Carlo VaR
    
    Simulates future portfolio values using random scenarios.
    Flexible for complex portfolios with derivatives and non-linear payoffs.
    Most computationally intensive but most accurate for complex portfolios.
    """
    
    @staticmethod
    def calculate(
        portfolio_value: float,
        returns: Union[np.ndarray, pd.Series, List[float]],
        confidence_level: float = 0.95,
        time_horizon_days: int = 1,
        num_simulations: int = 10000,
        random_seed: Optional[int] = None
    ) -> VaRResult:
        """
        Calculate Monte Carlo VaR using simulated scenarios.
        
        Args:
            portfolio_value: Current portfolio value
            returns: Historical returns for parameter estimation
            confidence_level: Confidence level
            time_horizon_days: VaR time horizon
            num_simulations: Number of Monte Carlo simulations
            random_seed: Random seed for reproducibility
        
        Returns:
            VaRResult with calculated VaR
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        returns_array = np.array(returns)
        
        # Estimate parameters from historical data
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array, ddof=1)
        
        # Simulate portfolio returns
        simulated_returns = MonteCarloVaR._simulate_returns(
            mean_return,
            std_return,
            time_horizon_days,
            num_simulations
        )
        
        # Calculate portfolio values
        simulated_values = portfolio_value * (1 + simulated_returns)
        simulated_losses = portfolio_value - simulated_values
        
        # Calculate VaR as percentile of simulated losses
        var_percentile = confidence_level * 100
        var_amount = np.percentile(simulated_losses, 100 - var_percentile)
        var_percentage = var_amount / portfolio_value
        
        # Calculate Expected Shortfall
        es_amount = MonteCarloVaR._calculate_expected_shortfall(
            simulated_losses, var_amount
        )
        
        return VaRResult(
            var_amount=var_amount,
            var_percentage=var_percentage,
            confidence_level=confidence_level,
            time_horizon_days=time_horizon_days,
            method=VaRMethod.MONTE_CARLO,
            portfolio_value=portfolio_value,
            expected_shortfall=es_amount,
            metadata={
                "num_simulations": num_simulations,
                "mean_return": mean_return,
                "volatility": std_return,
                "min_simulated_loss": float(np.min(simulated_losses)),
                "max_simulated_loss": float(np.max(simulated_losses)),
                "mean_simulated_loss": float(np.mean(simulated_losses)),
                "median_simulated_loss": float(np.median(simulated_losses))
            }
        )
    
    @staticmethod
    def _simulate_returns(
        mean_return: float,
        std_return: float,
        time_horizon: int,
        num_simulations: int
    ) -> np.ndarray:
        """Simulate portfolio returns using geometric Brownian motion."""
        # Generate random returns for each simulation
        daily_returns = np.random.normal(
            mean_return,
            std_return,
            (num_simulations, time_horizon)
        )
        
        # Compound returns for multi-day horizon
        cumulative_returns = np.prod(1 + daily_returns, axis=1) - 1
        
        return cumulative_returns
    
    @staticmethod
    def _calculate_expected_shortfall(
        simulated_losses: np.ndarray,
        var_threshold: float
    ) -> float:
        """Calculate Expected Shortfall for Monte Carlo method."""
        # ES is average of simulated losses exceeding VaR
        tail_losses = simulated_losses[simulated_losses >= var_threshold]
        es = np.mean(tail_losses) if len(tail_losses) > 0 else var_threshold
        
        return es


class VaRCalculator:
    """
    Unified VaR Calculator with multiple methodologies.
    
    Features:
    - Support for all three VaR methods
    - Multi-method comparison and validation
    - Portfolio-level and position-level VaR
    - Backtesting and model validation
    - Risk decomposition and attribution
    """
    
    def __init__(self, default_confidence: float = 0.95):
        """
        Initialize VaR calculator.
        
        Args:
            default_confidence: Default confidence level for calculations
        """
        self.default_confidence = default_confidence
        self.calculation_history: List[VaRResult] = []
    
    def calculate_var(
        self,
        portfolio_value: float,
        returns: Union[np.ndarray, pd.Series, List[float]],
        method: VaRMethod = VaRMethod.HISTORICAL,
        confidence_level: Optional[float] = None,
        time_horizon_days: int = 1,
        **kwargs
    ) -> VaRResult:
        """
        Calculate VaR using specified method.
        
        Args:
            portfolio_value: Current portfolio value
            returns: Historical returns data
            method: VaR calculation method
            confidence_level: Confidence level (uses default if not specified)
            time_horizon_days: Time horizon in days
            **kwargs: Additional method-specific parameters
        
        Returns:
            VaRResult with calculated VaR
        """
        conf_level = confidence_level or self.default_confidence
        
        if method == VaRMethod.PARAMETRIC:
            result = ParametricVaR.calculate(
                portfolio_value, returns, conf_level, time_horizon_days
            )
        elif method == VaRMethod.HISTORICAL:
            result = HistoricalSimulationVaR.calculate(
                portfolio_value, returns, conf_level, time_horizon_days
            )
        elif method == VaRMethod.MONTE_CARLO:
            num_sims = kwargs.get('num_simulations', 10000)
            seed = kwargs.get('random_seed', None)
            result = MonteCarloVaR.calculate(
                portfolio_value, returns, conf_level, time_horizon_days,
                num_sims, seed
            )
        else:
            raise ValueError(f"Unknown VaR method: {method}")
        
        # Store calculation history
        self.calculation_history.append(result)
        
        return result
    
    def calculate_all_methods(
        self,
        portfolio_value: float,
        returns: Union[np.ndarray, pd.Series, List[float]],
        confidence_level: Optional[float] = None,
        time_horizon_days: int = 1,
        num_simulations: int = 10000
    ) -> Dict[str, VaRResult]:
        """
        Calculate VaR using all three methods for comparison.
        
        Returns:
            Dictionary mapping method name to VaRResult
        """
        conf_level = confidence_level or self.default_confidence
        
        results = {}
        
        # Parametric VaR
        results["parametric"] = ParametricVaR.calculate(
            portfolio_value, returns, conf_level, time_horizon_days
        )
        
        # Historical Simulation VaR
        results["historical"] = HistoricalSimulationVaR.calculate(
            portfolio_value, returns, conf_level, time_horizon_days
        )
        
        # Monte Carlo VaR
        results["monte_carlo"] = MonteCarloVaR.calculate(
            portfolio_value, returns, conf_level, time_horizon_days, num_simulations
        )
        
        return results
    
    def get_var_summary(
        self,
        all_results: Dict[str, VaRResult]
    ) -> Dict:
        """
        Generate summary statistics across all VaR methods.
        
        Args:
            all_results: Dictionary of VaR results from calculate_all_methods()
        
        Returns:
            Summary dictionary with statistics and recommendations
        """
        var_amounts = [r.var_amount for r in all_results.values()]
        var_percentages = [r.var_percentage for r in all_results.values()]
        
        return {
            "portfolio_value": all_results["parametric"].portfolio_value,
            "confidence_level": all_results["parametric"].confidence_level,
            "time_horizon_days": all_results["parametric"].time_horizon_days,
            "var_range": {
                "min": min(var_amounts),
                "max": max(var_amounts),
                "mean": np.mean(var_amounts),
                "median": np.median(var_amounts)
            },
            "var_percentage_range": {
                "min": min(var_percentages),
                "max": max(var_percentages),
                "mean": np.mean(var_percentages)
            },
            "method_comparison": {
                method: {
                    "var_amount": result.var_amount,
                    "var_percentage": result.var_percentage,
                    "expected_shortfall": result.expected_shortfall
                }
                for method, result in all_results.items()
            },
            "recommendation": "Use historical or Monte Carlo for non-normal distributions",
            "calculation_timestamp": datetime.now().isoformat()
        }
    
    def backtest_var(
        self,
        historical_var_results: List[VaRResult],
        actual_returns: Union[np.ndarray, pd.Series, List[float]],
        portfolio_values: Union[np.ndarray, pd.Series, List[float]]
    ) -> Dict:
        """
        Backtest VaR model accuracy.
        
        Calculates number of VaR breaches and validates model performance.
        
        Returns:
            Backtest statistics including breach rate and accuracy
        """
        actual_returns_array = np.array(actual_returns)
        portfolio_values_array = np.array(portfolio_values)
        
        # Calculate actual losses
        actual_losses = -portfolio_values_array * actual_returns_array
        
        breaches = 0
        total_observations = min(len(historical_var_results), len(actual_losses))
        
        for i in range(total_observations):
            var_threshold = historical_var_results[i].var_amount
            if actual_losses[i] > var_threshold:
                breaches += 1
        
        breach_rate = breaches / total_observations if total_observations > 0 else 0
        expected_breach_rate = 1 - historical_var_results[0].confidence_level if historical_var_results else 0
        
        # Model is accurate if breach rate is close to expected
        accuracy_score = 1 - abs(breach_rate - expected_breach_rate)
        
        return {
            "total_observations": total_observations,
            "var_breaches": breaches,
            "breach_rate": breach_rate,
            "expected_breach_rate": expected_breach_rate,
            "accuracy_score": accuracy_score,
            "model_performance": "GOOD" if accuracy_score > 0.8 else "REVIEW",
            "confidence_level": historical_var_results[0].confidence_level if historical_var_results else None
        }


def calculate_portfolio_var(
    positions: Dict[str, Dict[str, float]],
    returns_data: Dict[str, Union[np.ndarray, pd.Series]],
    method: VaRMethod = VaRMethod.HISTORICAL,
    confidence_level: float = 0.95,
    time_horizon_days: int = 1,
    correlation_matrix: Optional[np.ndarray] = None
) -> VaRResult:
    """
    Calculate VaR for a multi-asset portfolio.
    
    Args:
        positions: Dict of {symbol: {"value": amount, "weight": percentage}}
        returns_data: Dict of {symbol: returns_array}
        method: VaR calculation method
        confidence_level: Confidence level
        time_horizon_days: Time horizon
        correlation_matrix: Optional correlation matrix (for parametric only)
    
    Returns:
        Portfolio VaR result
    """
    # Calculate total portfolio value
    total_value = sum(pos["value"] for pos in positions.values())
    
    # For parametric method with correlation
    if method == VaRMethod.PARAMETRIC and correlation_matrix is not None:
        # Use matrix multiplication for correlated portfolio VaR
        weights = np.array([pos["weight"] for pos in positions.values()])
        volatilities = np.array([
            np.std(returns_data[symbol], ddof=1) 
            for symbol in positions.keys()
        ])
        
        # Portfolio volatility: σ_p = √(w' Σ w)
        portfolio_var_matrix = weights @ correlation_matrix @ (weights * volatilities**2)
        portfolio_std = np.sqrt(portfolio_var_matrix)
        
        # Calculate VaR
        z_score = stats.norm.ppf(1 - confidence_level)
        mean_return = np.dot(weights, [np.mean(returns_data[s]) for s in positions.keys()])
        
        scaled_vol = portfolio_std * np.sqrt(time_horizon_days)
        var_percentage = abs((mean_return * time_horizon_days) + (z_score * scaled_vol))
        var_amount = total_value * var_percentage
        
        return VaRResult(
            var_amount=var_amount,
            var_percentage=var_percentage,
            confidence_level=confidence_level,
            time_horizon_days=time_horizon_days,
            method=VaRMethod.PARAMETRIC,
            portfolio_value=total_value,
            metadata={
                "portfolio_volatility": portfolio_std,
                "num_positions": len(positions),
                "correlation_adjusted": True
            }
        )
    
    else:
        # For historical or MC, aggregate position returns
        portfolio_returns = []
        symbols = list(positions.keys())
        
        # Ensure all return series have same length
        min_length = min(len(returns_data[s]) for s in symbols)
        
        for i in range(min_length):
            # Calculate portfolio return for day i
            day_return = sum(
                positions[symbol]["weight"] * returns_data[symbol][i]
                for symbol in symbols
            )
            portfolio_returns.append(day_return)
        
        portfolio_returns_array = np.array(portfolio_returns)
        
        # Calculate VaR using selected method
        calculator = VaRCalculator(confidence_level)
        return calculator.calculate_var(
            total_value,
            portfolio_returns_array,
            method,
            confidence_level,
            time_horizon_days
        )


def calculate_marginal_var(
    portfolio_value: float,
    portfolio_returns: Union[np.ndarray, pd.Series],
    position_returns: Union[np.ndarray, pd.Series],
    position_weight: float,
    confidence_level: float = 0.95,
    time_horizon_days: int = 1
) -> float:
    """
    Calculate Marginal VaR (contribution of a position to portfolio VaR).
    
    Useful for:
    - Risk attribution
    - Position sizing
    - Risk-adjusted performance
    
    Args:
        portfolio_value: Total portfolio value
        portfolio_returns: Portfolio returns
        position_returns: Individual position returns
        position_weight: Position weight in portfolio
        confidence_level: Confidence level
        time_horizon_days: Time horizon
    
    Returns:
        Marginal VaR amount (contribution to portfolio VaR)
    """
    # Calculate portfolio VaR
    portfolio_var = HistoricalSimulationVaR.calculate(
        portfolio_value, portfolio_returns, confidence_level, time_horizon_days
    )
    
    # Calculate position VaR
    position_value = portfolio_value * position_weight
    position_var = HistoricalSimulationVaR.calculate(
        position_value, position_returns, confidence_level, time_horizon_days
    )
    
    # Marginal VaR is the incremental VaR from the position
    # Approximation: MVaR ≈ position_weight × correlation × position_VaR
    correlation = np.corrcoef(portfolio_returns, position_returns)[0, 1]
    marginal_var = position_weight * correlation * position_var.var_amount
    
    return marginal_var


def calculate_component_var(
    portfolio_var: float,
    position_marginal_vars: Dict[str, float],
    position_weights: Dict[str, float]
) -> Dict[str, float]:
    """
    Calculate Component VaR (each position's contribution to total VaR).
    
    Component VaR sums to total portfolio VaR.
    
    Args:
        portfolio_var: Total portfolio VaR
        position_marginal_vars: Marginal VaR for each position
        position_weights: Weight of each position
    
    Returns:
        Dictionary of Component VaR for each position
    """
    component_vars = {}
    
    for symbol in position_marginal_vars.keys():
        # Component VaR = Marginal VaR × Position Weight
        component_vars[symbol] = position_marginal_vars[symbol] * position_weights[symbol]
    
    # Normalize to sum to portfolio VaR
    total_component = sum(component_vars.values())
    if total_component > 0:
        scale_factor = portfolio_var / total_component
        component_vars = {k: v * scale_factor for k, v in component_vars.items()}
    
    return component_vars


# Convenience functions
def quick_var(
    portfolio_value: float,
    returns: Union[np.ndarray, pd.Series, List[float]],
    confidence_level: float = 0.95
) -> float:
    """Quick VaR calculation using Historical Simulation (most common)."""
    result = HistoricalSimulationVaR.calculate(
        portfolio_value, returns, confidence_level, time_horizon_days=1
    )
    return result.var_amount


def regulatory_var(
    portfolio_value: float,
    returns: Union[np.ndarray, pd.Series, List[float]]
) -> VaRResult:
    """
    Calculate VaR for regulatory reporting (99% confidence, 10-day horizon).
    
    Standard for Basel III and many regulatory frameworks.
    """
    return HistoricalSimulationVaR.calculate(
        portfolio_value,
        returns,
        confidence_level=0.99,
        time_horizon_days=10
    )


# Export all components
__all__ = [
    "VaRMethod",
    "ConfidenceLevel",
    "VaRResult",
    "ParametricVaR",
    "HistoricalSimulationVaR",
    "MonteCarloVaR",
    "VaRCalculator",
    "calculate_portfolio_var",
    "calculate_marginal_var",
    "calculate_component_var",
    "quick_var",
    "regulatory_var"
]