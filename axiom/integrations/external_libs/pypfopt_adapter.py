"""
PyPortfolioOpt Adapter for Portfolio Optimization

This module provides an adapter around PyPortfolioOpt for production-grade portfolio optimization.
PyPortfolioOpt offers modern portfolio theory implementations including:
- Efficient Frontier optimization
- Black-Litterman model
- Hierarchical Risk Parity (HRP)
- Critical Line Algorithm (CLA)
- Multiple risk models and expected return estimators

Features:
- Mean-variance optimization
- Risk parity portfolios
- Black-Litterman with market views
- Discrete allocation for real trading
- Multiple objective functions (Sharpe, volatility, return)
- Covariance shrinkage and robust estimation
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .config import get_config, LibraryAvailability

logger = logging.getLogger(__name__)

# Check if PyPortfolioOpt is available
PYPFOPT_AVAILABLE = LibraryAvailability.check_library('pypfopt')

if PYPFOPT_AVAILABLE:
    from pypfopt import (
        EfficientFrontier,
        BlackLittermanModel,
        HRPOpt,
        CLA,
        risk_models,
        expected_returns,
        objective_functions,
        DiscreteAllocation,
    )
    from pypfopt.discrete_allocation import get_latest_prices


class OptimizationObjective(Enum):
    """Optimization objectives."""
    MAX_SHARPE = "max_sharpe"
    MIN_VOLATILITY = "min_volatility"
    MAX_QUADRATIC_UTILITY = "max_quadratic_utility"
    EFFICIENT_RISK = "efficient_risk"
    EFFICIENT_RETURN = "efficient_return"


class RiskModel(Enum):
    """Risk model types."""
    SAMPLE_COV = "sample_cov"
    SEMICOVARIANCE = "semicovariance"
    EXP_COV = "exp_cov"
    LEDOIT_WOLF = "ledoit_wolf"
    LEDOIT_WOLF_CONSTANT_VARIANCE = "ledoit_wolf_constant_variance"
    LEDOIT_WOLF_SINGLE_FACTOR = "ledoit_wolf_single_factor"
    LEDOIT_WOLF_CONSTANT_CORRELATION = "ledoit_wolf_constant_correlation"
    ORACLE_APPROXIMATING = "oracle_approximating"


class ExpectedReturnModel(Enum):
    """Expected return estimation models."""
    MEAN_HISTORICAL_RETURN = "mean_historical_return"
    EMA_HISTORICAL_RETURN = "ema_historical_return"
    CAPM_RETURN = "capm_return"


@dataclass
class OptimizationResult:
    """Portfolio optimization result."""
    weights: Dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    portfolio_value: Optional[float] = None
    discrete_allocation: Optional[Dict[str, int]] = None
    leftover_cash: Optional[float] = None


@dataclass
class BlackLittermanInputs:
    """Inputs for Black-Litterman model."""
    market_caps: Dict[str, float]  # Market capitalization of each asset
    views: Dict[str, float]  # Absolute views on expected returns
    view_confidences: Optional[Dict[str, float]] = None  # Confidence in each view
    risk_aversion: float = 1.0
    tau: float = 0.05  # Scaling factor for uncertainty


class PyPortfolioOptAdapter:
    """Adapter for PyPortfolioOpt portfolio optimization.
    
    This class provides a clean interface to PyPortfolioOpt's optimization capabilities,
    making it easy to optimize portfolios using modern portfolio theory.
    
    Example:
        >>> adapter = PyPortfolioOptAdapter()
        >>> prices = pd.DataFrame(...)  # Historical price data
        >>> result = adapter.optimize_portfolio(
        ...     prices,
        ...     objective=OptimizationObjective.MAX_SHARPE
        ... )
        >>> print(f"Expected Return: {result.expected_return:.2%}")
        >>> print(f"Volatility: {result.volatility:.2%}")
        >>> print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    """
    
    def __init__(self):
        """Initialize the PyPortfolioOpt adapter."""
        if not PYPFOPT_AVAILABLE:
            raise ImportError(
                "PyPortfolioOpt is not available. Install it with: pip install PyPortfolioOpt"
            )
        
        self.config = get_config()
        logger.info("PyPortfolioOpt adapter initialized")
    
    def _calculate_expected_returns(
        self,
        prices: pd.DataFrame,
        method: ExpectedReturnModel = ExpectedReturnModel.MEAN_HISTORICAL_RETURN,
        **kwargs
    ) -> pd.Series:
        """Calculate expected returns using specified method.
        
        Args:
            prices: DataFrame of historical prices
            method: Expected return estimation method
            **kwargs: Additional arguments for the method
            
        Returns:
            Series of expected returns
        """
        if method == ExpectedReturnModel.MEAN_HISTORICAL_RETURN:
            return expected_returns.mean_historical_return(prices, **kwargs)
        elif method == ExpectedReturnModel.EMA_HISTORICAL_RETURN:
            return expected_returns.ema_historical_return(prices, **kwargs)
        elif method == ExpectedReturnModel.CAPM_RETURN:
            return expected_returns.capm_return(prices, **kwargs)
        else:
            return expected_returns.mean_historical_return(prices)
    
    def _calculate_risk_model(
        self,
        prices: pd.DataFrame,
        method: RiskModel = RiskModel.SAMPLE_COV,
        **kwargs
    ) -> pd.DataFrame:
        """Calculate covariance matrix using specified method.
        
        Args:
            prices: DataFrame of historical prices
            method: Risk model method
            **kwargs: Additional arguments for the method
            
        Returns:
            Covariance matrix
        """
        if method == RiskModel.SAMPLE_COV:
            return risk_models.sample_cov(prices, **kwargs)
        elif method == RiskModel.SEMICOVARIANCE:
            return risk_models.semicovariance(prices, **kwargs)
        elif method == RiskModel.EXP_COV:
            return risk_models.exp_cov(prices, **kwargs)
        elif method == RiskModel.LEDOIT_WOLF:
            return risk_models.CovarianceShrinkage(prices, **kwargs).ledoit_wolf()
        elif method == RiskModel.LEDOIT_WOLF_CONSTANT_VARIANCE:
            return risk_models.CovarianceShrinkage(prices, **kwargs).ledoit_wolf(
                shrinkage_target="constant_variance"
            )
        elif method == RiskModel.LEDOIT_WOLF_SINGLE_FACTOR:
            return risk_models.CovarianceShrinkage(prices, **kwargs).ledoit_wolf(
                shrinkage_target="single_factor"
            )
        elif method == RiskModel.LEDOIT_WOLF_CONSTANT_CORRELATION:
            return risk_models.CovarianceShrinkage(prices, **kwargs).ledoit_wolf(
                shrinkage_target="constant_correlation"
            )
        elif method == RiskModel.ORACLE_APPROXIMATING:
            return risk_models.CovarianceShrinkage(prices, **kwargs).oracle_approximating()
        else:
            return risk_models.sample_cov(prices)
    
    def optimize_portfolio(
        self,
        prices: pd.DataFrame,
        objective: OptimizationObjective = OptimizationObjective.MAX_SHARPE,
        risk_model_method: RiskModel = RiskModel.SAMPLE_COV,
        return_model_method: ExpectedReturnModel = ExpectedReturnModel.MEAN_HISTORICAL_RETURN,
        weight_bounds: Tuple[float, float] = (0, 1),
        target_return: Optional[float] = None,
        target_volatility: Optional[float] = None,
        risk_free_rate: float = 0.02,
        **kwargs
    ) -> OptimizationResult:
        """Optimize portfolio using mean-variance optimization.
        
        Args:
            prices: DataFrame of historical prices (assets as columns)
            objective: Optimization objective
            risk_model_method: Method for covariance estimation
            return_model_method: Method for expected return estimation
            weight_bounds: Min and max weights for assets
            target_return: Target return for efficient_return objective
            target_volatility: Target volatility for efficient_risk objective
            risk_free_rate: Risk-free rate for Sharpe ratio
            **kwargs: Additional arguments passed to optimizer
            
        Returns:
            OptimizationResult with weights and performance metrics
        """
        try:
            # Calculate expected returns and covariance
            mu = self._calculate_expected_returns(prices, return_model_method)
            S = self._calculate_risk_model(prices, risk_model_method)
            
            # Create efficient frontier
            ef = EfficientFrontier(mu, S, weight_bounds=weight_bounds, **kwargs)
            
            # Optimize based on objective
            if objective == OptimizationObjective.MAX_SHARPE:
                weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
            elif objective == OptimizationObjective.MIN_VOLATILITY:
                weights = ef.min_volatility()
            elif objective == OptimizationObjective.MAX_QUADRATIC_UTILITY:
                weights = ef.max_quadratic_utility(risk_aversion=kwargs.get('risk_aversion', 1))
            elif objective == OptimizationObjective.EFFICIENT_RISK:
                if target_volatility is None:
                    raise ValueError("target_volatility required for efficient_risk objective")
                weights = ef.efficient_risk(target_volatility)
            elif objective == OptimizationObjective.EFFICIENT_RETURN:
                if target_return is None:
                    raise ValueError("target_return required for efficient_return objective")
                weights = ef.efficient_return(target_return)
            else:
                weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
            
            # Clean weights (remove tiny weights)
            cleaned_weights = ef.clean_weights()
            
            # Get performance metrics
            performance = ef.portfolio_performance(
                verbose=False,
                risk_free_rate=risk_free_rate
            )
            
            result = OptimizationResult(
                weights=cleaned_weights,
                expected_return=performance[0],
                volatility=performance[1],
                sharpe_ratio=performance[2]
            )
            
            if self.config.log_library_usage:
                logger.info(
                    f"Portfolio optimized: Return={result.expected_return:.2%}, "
                    f"Vol={result.volatility:.2%}, Sharpe={result.sharpe_ratio:.2f}"
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Error optimizing portfolio: {e}")
            raise
    
    def optimize_hrp(
        self,
        prices: pd.DataFrame,
        returns_data: bool = False
    ) -> OptimizationResult:
        """Optimize portfolio using Hierarchical Risk Parity.
        
        HRP is a modern portfolio optimization method that doesn't require
        expected returns and is more stable than mean-variance optimization.
        
        Args:
            prices: DataFrame of historical prices or returns
            returns_data: If True, input is returns; if False, input is prices
            
        Returns:
            OptimizationResult with HRP weights
        """
        try:
            hrp = HRPOpt(prices, returns_data=returns_data)
            weights = hrp.optimize()
            
            # Calculate performance metrics
            if not returns_data:
                returns = prices.pct_change().dropna()
            else:
                returns = prices
            
            portfolio_return = np.sum(returns.mean() * pd.Series(weights)) * 252
            portfolio_vol = np.sqrt(
                np.dot(pd.Series(weights), np.dot(returns.cov() * 252, pd.Series(weights)))
            )
            sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
            
            result = OptimizationResult(
                weights=weights,
                expected_return=portfolio_return,
                volatility=portfolio_vol,
                sharpe_ratio=sharpe
            )
            
            if self.config.log_library_usage:
                logger.info(f"HRP optimization complete: {len(weights)} assets")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in HRP optimization: {e}")
            raise
    
    def optimize_black_litterman(
        self,
        prices: pd.DataFrame,
        bl_inputs: BlackLittermanInputs,
        objective: OptimizationObjective = OptimizationObjective.MAX_SHARPE,
        risk_model_method: RiskModel = RiskModel.SAMPLE_COV,
        weight_bounds: Tuple[float, float] = (0, 1),
        risk_free_rate: float = 0.02
    ) -> OptimizationResult:
        """Optimize portfolio using Black-Litterman model.
        
        Black-Litterman allows you to incorporate market views into portfolio optimization.
        
        Args:
            prices: DataFrame of historical prices
            bl_inputs: Black-Litterman model inputs
            objective: Optimization objective
            risk_model_method: Method for covariance estimation
            weight_bounds: Min and max weights for assets
            risk_free_rate: Risk-free rate
            
        Returns:
            OptimizationResult with Black-Litterman weights
        """
        try:
            # Calculate covariance matrix
            S = self._calculate_risk_model(prices, risk_model_method)
            
            # Set up Black-Litterman model
            bl = BlackLittermanModel(
                S,
                pi="market",
                market_caps=bl_inputs.market_caps,
                risk_aversion=bl_inputs.risk_aversion,
                tau=bl_inputs.tau
            )
            
            # Add views
            for asset, view_return in bl_inputs.views.items():
                confidence = (
                    bl_inputs.view_confidences.get(asset, 0.5)
                    if bl_inputs.view_confidences
                    else 0.5
                )
                bl.add_view(asset, view_return, confidence)
            
            # Get posterior estimates
            ret_bl = bl.bl_returns()
            S_bl = bl.bl_cov()
            
            # Optimize with Black-Litterman returns
            ef = EfficientFrontier(ret_bl, S_bl, weight_bounds=weight_bounds)
            
            if objective == OptimizationObjective.MAX_SHARPE:
                weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
            elif objective == OptimizationObjective.MIN_VOLATILITY:
                weights = ef.min_volatility()
            else:
                weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
            
            cleaned_weights = ef.clean_weights()
            performance = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
            
            result = OptimizationResult(
                weights=cleaned_weights,
                expected_return=performance[0],
                volatility=performance[1],
                sharpe_ratio=performance[2]
            )
            
            if self.config.log_library_usage:
                logger.info("Black-Litterman optimization complete")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in Black-Litterman optimization: {e}")
            raise
    
    def discrete_allocation(
        self,
        weights: Dict[str, float],
        latest_prices: Union[pd.Series, Dict[str, float]],
        total_portfolio_value: float
    ) -> OptimizationResult:
        """Convert continuous weights to discrete share allocations.
        
        This is useful for actual trading - converts optimal weights to
        integer number of shares that can be purchased.
        
        Args:
            weights: Dictionary of asset weights
            latest_prices: Latest prices for each asset
            total_portfolio_value: Total value of portfolio to allocate
            
        Returns:
            OptimizationResult with discrete allocation
        """
        try:
            if isinstance(latest_prices, dict):
                latest_prices = pd.Series(latest_prices)
            
            da = DiscreteAllocation(
                weights,
                latest_prices,
                total_portfolio_value=total_portfolio_value
            )
            
            allocation, leftover = da.greedy_portfolio()
            
            # Calculate actual portfolio value and weights
            actual_value = sum(
                shares * latest_prices[ticker]
                for ticker, shares in allocation.items()
            )
            
            actual_weights = {
                ticker: (shares * latest_prices[ticker]) / actual_value
                for ticker, shares in allocation.items()
            }
            
            result = OptimizationResult(
                weights=actual_weights,
                expected_return=0.0,  # Not calculated for discrete allocation
                volatility=0.0,
                sharpe_ratio=0.0,
                portfolio_value=actual_value,
                discrete_allocation=allocation,
                leftover_cash=leftover
            )
            
            if self.config.log_library_usage:
                logger.info(
                    f"Discrete allocation: ${actual_value:.2f} allocated, "
                    f"${leftover:.2f} leftover"
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in discrete allocation: {e}")
            raise
    
    def efficient_frontier_curve(
        self,
        prices: pd.DataFrame,
        risk_model_method: RiskModel = RiskModel.SAMPLE_COV,
        return_model_method: ExpectedReturnModel = ExpectedReturnModel.MEAN_HISTORICAL_RETURN,
        n_points: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate efficient frontier curve.
        
        Args:
            prices: DataFrame of historical prices
            risk_model_method: Method for covariance estimation
            return_model_method: Method for expected return estimation
            n_points: Number of points on the frontier
            
        Returns:
            Tuple of (returns, volatilities) arrays
        """
        try:
            mu = self._calculate_expected_returns(prices, return_model_method)
            S = self._calculate_risk_model(prices, risk_model_method)
            
            # Use Critical Line Algorithm for efficient frontier
            cla = CLA(mu, S)
            
            returns, volatilities, _ = cla.efficient_frontier(points=n_points)
            
            return np.array(returns), np.array(volatilities)
            
        except Exception as e:
            logger.error(f"Error generating efficient frontier: {e}")
            raise


def check_pypfopt_availability() -> bool:
    """Check if PyPortfolioOpt is available.
    
    Returns:
        True if PyPortfolioOpt is available, False otherwise
    """
    return PYPFOPT_AVAILABLE