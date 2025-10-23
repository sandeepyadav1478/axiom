"""
Market Impact Models for Optimal Execution
===========================================

Institutional-grade market impact estimation and optimal execution models:
- Kyle's Lambda Model (price impact per unit volume)
- Almgren-Chriss Model (optimal execution with temporary/permanent impact)
- Square-Root Law (empirical market impact formula)
- Bertsimas-Lo Model (dynamic trading strategies)

Performance Target: <15ms for impact estimation and optimal trajectory

Mathematical References:
- Kyle, A. S. (1985). "Continuous Auctions and Insider Trading"
- Almgren, R., & Chriss, N. (2001). "Optimal execution of portfolio transactions"
- Huberman, G., & Stanzl, W. (2004). "Price manipulation and quasi-arbitrage"
- Bertsimas, D., & Lo, A. W. (1998). "Optimal control of execution costs"
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass
from scipy.optimize import minimize
import time

from axiom.models.microstructure.base_model import (
    BaseMarketMicrostructureModel,
    TickData,
    MicrostructureMetrics,
    ModelResult
)
from axiom.models.base.mixins import NumericalMethodsMixin


@dataclass
class MarketImpactEstimate:
    """Container for market impact estimation results."""
    # Impact coefficients
    kyle_lambda: float  # Kyle's lambda (price impact per unit volume)
    temporary_impact_coef: float  # Temporary impact coefficient
    permanent_impact_coef: float  # Permanent impact coefficient
    
    # Square-root law parameters
    sqrt_coef: float  # Coefficient in square-root law
    participation_rate: float  # Optimal participation rate
    
    # Expected costs
    expected_price_impact_bps: float  # Expected impact in bps
    expected_cost_bps: float  # Total expected execution cost
    timing_risk_bps: float  # Cost due to price volatility
    
    # Optimal execution
    optimal_execution_time: float  # Optimal time to complete (seconds)
    optimal_slice_size: float  # Optimal order size per slice
    n_slices: int  # Number of slices
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'kyle_lambda': self.kyle_lambda,
            'temporary_impact_coef': self.temporary_impact_coef,
            'permanent_impact_coef': self.permanent_impact_coef,
            'sqrt_coef': self.sqrt_coef,
            'participation_rate': self.participation_rate,
            'expected_price_impact_bps': self.expected_price_impact_bps,
            'expected_cost_bps': self.expected_cost_bps,
            'timing_risk_bps': self.timing_risk_bps,
            'optimal_execution_time': self.optimal_execution_time,
            'optimal_slice_size': self.optimal_slice_size,
            'n_slices': self.n_slices
        }


@dataclass
class OptimalTrajectory:
    """Optimal execution trajectory."""
    times: np.ndarray  # Time points
    holdings: np.ndarray  # Remaining shares at each time
    trade_rates: np.ndarray  # Trading rate at each time
    expected_cost: float  # Total expected cost
    execution_shortfall: float  # Implementation shortfall
    
    @property
    def total_time(self) -> float:
        """Total execution time."""
        return self.times[-1] if len(self.times) > 0 else 0.0
    
    @property
    def n_steps(self) -> int:
        """Number of time steps."""
        return len(self.times)


class KyleLambdaModel(BaseMarketMicrostructureModel):
    """
    Kyle's Lambda Model for Market Impact.
    
    Estimates the price impact per unit volume traded (Kyle's lambda).
    
    Model:
        ΔP = λ × Q + ε
    
    where:
        ΔP = price change
        λ = Kyle's lambda (impact coefficient)
        Q = order size (signed volume)
        ε = noise
    
    Reference:
    Kyle, A. S. (1985). "Continuous Auctions and Insider Trading"
    
    Usage:
        model = KyleLambdaModel()
        lambda_estimate = model.estimate_lambda(tick_data)
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        enable_logging: bool = True,
        enable_performance_tracking: bool = True
    ):
        """
        Initialize Kyle's Lambda Model.
        
        Args:
            config: Configuration dictionary
            enable_logging: Enable detailed logging
            enable_performance_tracking: Track execution time
        """
        super().__init__(config, enable_logging, enable_performance_tracking)
        
        self.estimation_window = self.config.get('estimation_window', 100)
    
    def calculate_metrics(self, tick_data: TickData, **kwargs) -> MicrostructureMetrics:
        """
        Calculate Kyle's lambda from tick data.
        
        Args:
            tick_data: Tick data with trade directions
            **kwargs: Additional parameters
            
        Returns:
            MicrostructureMetrics with kyle_lambda
        """
        start_time = time.perf_counter()
        
        lambda_estimate = self.estimate_lambda(tick_data)
        
        execution_time = (time.perf_counter() - start_time) * 1000
        
        if self.enable_logging:
            self.logger.debug(
                f"Kyle's lambda estimation completed",
                execution_time_ms=round(execution_time, 3),
                kyle_lambda=round(lambda_estimate, 6)
            )
        
        return MicrostructureMetrics(
            kyle_lambda=lambda_estimate
        )
    
    def estimate_lambda(self, tick_data: TickData) -> float:
        """
        Estimate Kyle's lambda using regression.
        
        Regression: ΔP_t = λ × SignedVolume_t + ε_t
        
        Args:
            tick_data: Tick data with classified trades
            
        Returns:
            Estimated Kyle's lambda
        """
        if tick_data.trade_direction is None or tick_data.n_ticks < 2:
            return 0.0
        
        # Calculate price changes
        price_changes = np.diff(tick_data.price)
        
        # Signed volumes (volume × direction)
        signed_volumes = tick_data.volume[1:] * tick_data.trade_direction[1:]
        
        # Use most recent window
        if len(price_changes) > self.estimation_window:
            price_changes = price_changes[-self.estimation_window:]
            signed_volumes = signed_volumes[-self.estimation_window:]
        
        # Estimate lambda via OLS: λ = Cov(ΔP, Q) / Var(Q)
        if np.std(signed_volumes) > 0:
            kyle_lambda = np.cov(price_changes, signed_volumes)[0, 1] / np.var(signed_volumes)
        else:
            kyle_lambda = 0.0
        
        return float(kyle_lambda)
    
    def calculate_informed_trading_probability(
        self,
        tick_data: TickData,
        kyle_lambda: Optional[float] = None
    ) -> float:
        """
        Estimate probability of informed trading.
        
        In Kyle's model, higher lambda indicates more informed trading.
        
        Args:
            tick_data: Tick data
            kyle_lambda: Pre-computed lambda (optional)
            
        Returns:
            Probability of informed trading [0, 1]
        """
        if kyle_lambda is None:
            kyle_lambda = self.estimate_lambda(tick_data)
        
        # Normalize lambda to probability scale
        # Higher lambda = higher probability of informed trading
        # This is a simplified heuristic
        avg_price = np.mean(tick_data.price)
        avg_volume = np.mean(tick_data.volume)
        
        if avg_price > 0 and avg_volume > 0:
            normalized_lambda = abs(kyle_lambda) * avg_volume / avg_price
            # Map to [0, 1] using sigmoid-like function
            prob = 1 - np.exp(-normalized_lambda)
        else:
            prob = 0.0
        
        return float(min(prob, 1.0))


class AlmgrenChrissModel(BaseMarketMicrostructureModel, NumericalMethodsMixin):
    """
    Almgren-Chriss Model for Optimal Execution.
    
    Calculates optimal execution trajectory balancing:
    - Market impact costs (temporary and permanent)
    - Timing risk (volatility while holding)
    
    Model minimizes:
        E[Cost] + λ × Var[Cost]
    
    where λ is the risk aversion parameter.
    
    Reference:
    Almgren, R., & Chriss, N. (2001). "Optimal execution of portfolio transactions"
    
    Features:
    - Optimal trajectory calculation
    - Trade-off between impact and risk
    - Customizable risk aversion
    - Time-dependent strategies
    
    Usage:
        model = AlmgrenChrissModel(config={
            'risk_aversion': 1e-6,
            'permanent_impact': 0.1,
            'temporary_impact': 0.5
        })
        
        trajectory = model.calculate_optimal_trajectory(
            total_shares=10000,
            total_time=3600,  # 1 hour
            volatility=0.02
        )
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        enable_logging: bool = True,
        enable_performance_tracking: bool = True
    ):
        """
        Initialize Almgren-Chriss Model.
        
        Args:
            config: Configuration dictionary with:
                - risk_aversion: Risk aversion parameter λ (default: 1e-6)
                - permanent_impact: Permanent impact coefficient η (default: 0.1)
                - temporary_impact: Temporary impact coefficient ε (default: 0.5)
                - n_steps: Number of time steps (default: 100)
            enable_logging: Enable detailed logging
            enable_performance_tracking: Track execution time
        """
        super().__init__(config, enable_logging, enable_performance_tracking)
        
        self.risk_aversion = self.config.get('risk_aversion', 1e-6)
        self.permanent_impact = self.config.get('permanent_impact', 0.1)
        self.temporary_impact = self.config.get('temporary_impact', 0.5)
        self.n_steps = self.config.get('n_steps', 100)
    
    def calculate_metrics(self, tick_data: TickData, **kwargs) -> MicrostructureMetrics:
        """
        Calculate optimal execution metrics.
        
        Args:
            tick_data: Tick data for parameter estimation
            **kwargs: Must include 'total_shares', 'total_time', 'volatility'
            
        Returns:
            MicrostructureMetrics with impact estimates
        """
        start_time = time.perf_counter()
        
        # Extract parameters
        total_shares = kwargs.get('total_shares', 10000)
        total_time = kwargs.get('total_time', 3600)
        volatility = kwargs.get('volatility', 0.02)
        
        # Calculate optimal trajectory
        trajectory = self.calculate_optimal_trajectory(
            total_shares=total_shares,
            total_time=total_time,
            volatility=volatility
        )
        
        execution_time = (time.perf_counter() - start_time) * 1000
        
        if self.enable_logging:
            self.logger.debug(
                f"Almgren-Chriss optimization completed",
                execution_time_ms=round(execution_time, 3),
                expected_cost=round(trajectory.expected_cost, 2)
            )
        
        return MicrostructureMetrics(
            market_impact_bps=trajectory.expected_cost,
            temporary_impact=self.temporary_impact,
            permanent_impact=self.permanent_impact
        )
    
    def calculate_optimal_trajectory(
        self,
        total_shares: float,
        total_time: float,
        volatility: float,
        current_price: float = 100.0
    ) -> OptimalTrajectory:
        """
        Calculate optimal execution trajectory.
        
        The optimal strategy is characterized by:
            n(t) = sinh(κ(T-t)) / sinh(κT) × X
        
        where κ depends on model parameters and risk aversion.
        
        Args:
            total_shares: Total shares to execute
            total_time: Total time for execution (seconds)
            volatility: Price volatility (annualized)
            current_price: Current price (default: 100)
            
        Returns:
            OptimalTrajectory with execution schedule
        """
        # Time discretization
        dt = total_time / self.n_steps
        times = np.linspace(0, total_time, self.n_steps + 1)
        
        # Calculate kappa (trade-off parameter)
        # κ² = λ × σ² / ε
        kappa = np.sqrt(self.risk_aversion * volatility**2 / self.temporary_impact)
        
        # Optimal holdings trajectory
        # n(t) = sinh(κ(T-t)) / sinh(κT) × X
        kappa_T = kappa * total_time
        
        if kappa_T > 1e-10:
            holdings = total_shares * np.sinh(kappa * (total_time - times)) / np.sinh(kappa_T)
        else:
            # Linear trajectory for small kappa (risk-neutral case)
            holdings = total_shares * (1 - times / total_time)
        
        # Trading rates (derivative of holdings)
        trade_rates = -np.diff(holdings) / dt
        trade_rates = np.append(trade_rates, 0)  # Zero rate at end
        
        # Calculate expected cost
        # Cost = Σ[temporary impact + permanent impact + risk cost]
        permanent_cost = self.permanent_impact * np.sum(np.abs(trade_rates)) * dt * current_price
        temporary_cost = self.temporary_impact * np.sum(trade_rates**2) * dt * current_price
        risk_cost = 0.5 * self.risk_aversion * volatility**2 * np.sum(holdings[:-1]**2) * dt
        
        expected_cost = permanent_cost + temporary_cost + risk_cost
        
        # Implementation shortfall (as percentage of notional)
        notional_value = total_shares * current_price
        execution_shortfall = (expected_cost / notional_value) * 10000  # bps
        
        return OptimalTrajectory(
            times=times,
            holdings=holdings,
            trade_rates=trade_rates,
            expected_cost=expected_cost,
            execution_shortfall=execution_shortfall
        )
    
    def estimate_parameters_from_data(self, tick_data: TickData) -> Dict[str, float]:
        """
        Estimate impact parameters from historical data.
        
        Args:
            tick_data: Historical tick data
            
        Returns:
            Dictionary with estimated parameters
        """
        if tick_data.trade_direction is None or tick_data.n_ticks < 2:
            return {
                'permanent_impact': self.permanent_impact,
                'temporary_impact': self.temporary_impact,
                'volatility': 0.02
            }
        
        # Estimate volatility from returns
        returns = np.diff(tick_data.price) / tick_data.price[:-1]
        volatility = np.std(returns) * np.sqrt(252 * 6.5 * 3600)  # Annualized
        
        # Estimate impact coefficients from price changes and volumes
        price_changes = np.abs(np.diff(tick_data.price))
        volumes = tick_data.volume[1:]
        avg_price = np.mean(tick_data.price)
        
        # Simple linear regression for impact
        if np.std(volumes) > 0:
            impact_per_volume = np.mean(price_changes / volumes) / avg_price
            # Split into permanent (30%) and temporary (70%) - empirical split
            permanent = impact_per_volume * 0.3
            temporary = impact_per_volume * 0.7
        else:
            permanent = self.permanent_impact
            temporary = self.temporary_impact
        
        return {
            'permanent_impact': permanent,
            'temporary_impact': temporary,
            'volatility': volatility
        }


class SquareRootLawModel(BaseMarketMicrostructureModel):
    """
    Square-Root Law of Market Impact.
    
    Empirical model showing that market impact scales with the square root
    of order size relative to daily volume.
    
    Model:
        I = σ × (Q / V)^0.5
    
    where:
        I = market impact
        σ = daily volatility
        Q = order size
        V = daily volume
    
    Reference:
    Huberman, G., & Stanzl, W. (2004). "Price manipulation and quasi-arbitrage"
    
    Features:
    - Empirically validated across markets
    - Simple to compute
    - Useful for quick impact estimates
    
    Usage:
        model = SquareRootLawModel()
        impact = model.calculate_impact(
            order_size=10000,
            daily_volume=1000000,
            volatility=0.02
        )
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        enable_logging: bool = True,
        enable_performance_tracking: bool = True
    ):
        """
        Initialize Square-Root Law Model.
        
        Args:
            config: Configuration dictionary with:
                - impact_coefficient: Calibration coefficient (default: 1.0)
            enable_logging: Enable detailed logging
            enable_performance_tracking: Track execution time
        """
        super().__init__(config, enable_logging, enable_performance_tracking)
        
        self.impact_coefficient = self.config.get('impact_coefficient', 1.0)
    
    def calculate_metrics(self, tick_data: TickData, **kwargs) -> MicrostructureMetrics:
        """
        Calculate market impact using square-root law.
        
        Args:
            tick_data: Tick data
            **kwargs: Must include 'order_size'
            
        Returns:
            MicrostructureMetrics with impact estimate
        """
        start_time = time.perf_counter()
        
        order_size = kwargs.get('order_size', 1000)
        daily_volume = np.sum(tick_data.volume)
        
        # Estimate volatility
        if tick_data.n_ticks > 1:
            returns = np.diff(tick_data.price) / tick_data.price[:-1]
            volatility = np.std(returns) * np.sqrt(252 * 6.5 * 3600)
        else:
            volatility = 0.02
        
        # Calculate impact
        impact_bps = self.calculate_impact(order_size, daily_volume, volatility)
        
        execution_time = (time.perf_counter() - start_time) * 1000
        
        if self.enable_logging:
            self.logger.debug(
                f"Square-root law impact calculated",
                execution_time_ms=round(execution_time, 3),
                impact_bps=round(impact_bps, 2)
            )
        
        return MicrostructureMetrics(
            market_impact_bps=impact_bps
        )
    
    def calculate_impact(
        self,
        order_size: float,
        daily_volume: float,
        volatility: float
    ) -> float:
        """
        Calculate market impact using square-root law.
        
        I = c × σ × (Q / V)^0.5
        
        Args:
            order_size: Size of order to execute
            daily_volume: Daily trading volume
            volatility: Daily volatility
            
        Returns:
            Expected market impact in basis points
        """
        if daily_volume <= 0:
            return 0.0
        
        # Participation rate
        participation = order_size / daily_volume
        
        # Square-root law
        impact = self.impact_coefficient * volatility * np.sqrt(participation)
        
        # Convert to basis points
        impact_bps = impact * 10000
        
        return float(impact_bps)
    
    def calculate_optimal_participation_rate(
        self,
        total_shares: float,
        daily_volume: float,
        max_participation: float = 0.25
    ) -> float:
        """
        Calculate optimal participation rate to minimize cost.
        
        Args:
            total_shares: Total shares to execute
            daily_volume: Expected daily volume
            max_participation: Maximum allowed participation rate
            
        Returns:
            Optimal participation rate
        """
        # Optimal rate balances market impact vs execution time
        # For square-root law, trade-off is roughly linear
        naive_participation = total_shares / daily_volume
        
        # Apply maximum constraint
        optimal_rate = min(naive_participation, max_participation)
        
        # Ensure minimum rate (don't take forever)
        optimal_rate = max(optimal_rate, 0.01)  # At least 1%
        
        return float(optimal_rate)


class MarketImpactAnalyzer(BaseMarketMicrostructureModel):
    """
    Comprehensive Market Impact Analysis.
    
    Combines multiple impact models for robust estimation:
    - Kyle's Lambda
    - Almgren-Chriss
    - Square-Root Law
    
    Provides:
    - Impact estimation from multiple models
    - Optimal execution recommendations
    - Cost-benefit analysis
    - Risk assessment
    
    Usage:
        analyzer = MarketImpactAnalyzer()
        
        estimate = analyzer.analyze_impact(
            tick_data=market_data,
            order_size=10000,
            execution_time=3600
        )
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        enable_logging: bool = True,
        enable_performance_tracking: bool = True
    ):
        """Initialize Market Impact Analyzer."""
        super().__init__(config, enable_logging, enable_performance_tracking)
        
        # Initialize component models
        self.kyle_model = KyleLambdaModel(config=config)
        self.almgren_chriss = AlmgrenChrissModel(config=config)
        self.sqrt_law = SquareRootLawModel(config=config)
    
    def calculate_metrics(self, tick_data: TickData, **kwargs) -> MicrostructureMetrics:
        """
        Calculate comprehensive market impact metrics.
        
        Args:
            tick_data: Historical tick data
            **kwargs: Must include 'order_size', optionally 'execution_time'
            
        Returns:
            MicrostructureMetrics with all impact estimates
        """
        start_time = time.perf_counter()
        
        # Get parameters
        order_size = kwargs.get('order_size', 1000)
        execution_time = kwargs.get('execution_time', 3600)
        
        # Estimate from all models
        kyle_lambda = self.kyle_model.estimate_lambda(tick_data)
        
        # Almgren-Chriss requires volatility
        if tick_data.n_ticks > 1:
            returns = np.diff(tick_data.price) / tick_data.price[:-1]
            volatility = np.std(returns) * np.sqrt(252 * 6.5 * 3600)
        else:
            volatility = 0.02
        
        trajectory = self.almgren_chriss.calculate_optimal_trajectory(
            total_shares=order_size,
            total_time=execution_time,
            volatility=volatility
        )
        
        daily_volume = np.sum(tick_data.volume)
        sqrt_impact = self.sqrt_law.calculate_impact(order_size, daily_volume, volatility)
        
        execution_time_ms = (time.perf_counter() - start_time) * 1000
        
        if self.enable_logging:
            self.logger.info(
                f"Comprehensive market impact analysis completed",
                execution_time_ms=round(execution_time_ms, 3),
                kyle_lambda=round(kyle_lambda, 6),
                almgren_chriss_cost=round(trajectory.expected_cost, 2),
                sqrt_law_impact_bps=round(sqrt_impact, 2)
            )
        
        return MicrostructureMetrics(
            kyle_lambda=kyle_lambda,
            market_impact_bps=sqrt_impact,
            temporary_impact=self.almgren_chriss.temporary_impact,
            permanent_impact=self.almgren_chriss.permanent_impact
        )
    
    def analyze_impact(
        self,
        tick_data: TickData,
        order_size: float,
        execution_time: float = 3600,
        current_price: float = 100.0
    ) -> MarketImpactEstimate:
        """
        Comprehensive market impact analysis.
        
        Args:
            tick_data: Historical market data
            order_size: Size of order to execute
            execution_time: Time available for execution (seconds)
            current_price: Current market price
            
        Returns:
            MarketImpactEstimate with all metrics
        """
        # Kyle's Lambda
        kyle_lambda = self.kyle_model.estimate_lambda(tick_data)
        
        # Estimate parameters from data
        params = self.almgren_chriss.estimate_parameters_from_data(tick_data)
        
        # Almgren-Chriss optimal trajectory
        trajectory = self.almgren_chriss.calculate_optimal_trajectory(
            total_shares=order_size,
            total_time=execution_time,
            volatility=params['volatility'],
            current_price=current_price
        )
        
        # Square-root law
        daily_volume = np.sum(tick_data.volume)
        sqrt_impact = self.sqrt_law.calculate_impact(
            order_size,
            daily_volume,
            params['volatility']
        )
        optimal_participation = self.sqrt_law.calculate_optimal_participation_rate(
            order_size,
            daily_volume
        )
        
        # Calculate expected costs
        notional = order_size * current_price
        impact_cost = sqrt_impact  # Use square-root law as baseline
        timing_risk = params['volatility'] * np.sqrt(execution_time / (252 * 6.5 * 3600)) * 10000
        total_cost = impact_cost + timing_risk
        
        # Optimal slicing
        n_slices = max(int(execution_time / 60), 1)  # At least 1 slice per minute
        optimal_slice_size = order_size / n_slices
        
        return MarketImpactEstimate(
            kyle_lambda=kyle_lambda,
            temporary_impact_coef=params['temporary_impact'],
            permanent_impact_coef=params['permanent_impact'],
            sqrt_coef=self.sqrt_law.impact_coefficient,
            participation_rate=optimal_participation,
            expected_price_impact_bps=impact_cost,
            expected_cost_bps=total_cost,
            timing_risk_bps=timing_risk,
            optimal_execution_time=execution_time,
            optimal_slice_size=optimal_slice_size,
            n_slices=n_slices
        )


__all__ = [
    "KyleLambdaModel",
    "AlmgrenChrissModel",
    "SquareRootLawModel",
    "MarketImpactAnalyzer",
    "MarketImpactEstimate",
    "OptimalTrajectory",
]