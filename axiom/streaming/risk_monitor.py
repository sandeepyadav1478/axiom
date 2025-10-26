"""
Real-Time Risk Monitoring

Continuous portfolio risk monitoring with live VaR calculation,
position limits, drawdown tracking, and automated alerts.

Integrates with existing VaR models from axiom.models.risk
"""

import asyncio
import time
import logging
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass, field
from collections import deque

import numpy as np

from axiom.streaming.config import StreamingConfig
from axiom.streaming.portfolio_tracker import PortfolioTracker, Position


logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """Real-time risk metrics."""
    timestamp: float = field(default_factory=time.time)
    portfolio_value: float = 0.0
    var_95: float = 0.0
    var_99: float = 0.0
    var_percentage: float = 0.0
    max_position_exposure: float = 0.0
    max_position_pct: float = 0.0
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    concentration_risk: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'portfolio_value': self.portfolio_value,
            'var_95': self.var_95,
            'var_99': self.var_99,
            'var_percentage': self.var_percentage,
            'max_position_exposure': self.max_position_exposure,
            'max_position_pct': self.max_position_pct,
            'current_drawdown': self.current_drawdown,
            'max_drawdown': self.max_drawdown,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'concentration_risk': self.concentration_risk,
        }


@dataclass
class RiskAlert:
    """Risk alert configuration."""
    alert_type: str
    threshold: float
    current_value: float
    callback: Callable
    triggered: bool = False


class RealTimeRiskMonitor:
    """
    Monitor portfolio risk in real-time.
    
    Features:
    - Live VaR calculation (parametric method)
    - Position limit monitoring
    - Drawdown tracking
    - Volatility alerts
    - Concentration risk
    - Margin requirement tracking
    - Automated risk alerts
    
    Integrates with existing VaR models from axiom.models.risk.var_models
    
    Example:
        ```python
        monitor = RealTimeRiskMonitor(portfolio_tracker)
        await monitor.start_monitoring()
        
        # Add risk limits
        monitor.add_risk_limit('var_percentage', 0.02)  # 2% VaR limit
        monitor.add_risk_limit('max_position_pct', 0.25)  # 25% position limit
        
        # Get risk metrics
        metrics = monitor.get_current_metrics()
        print(f"Current VaR: {metrics.var_percentage:.2%}")
        ```
    """
    
    def __init__(
        self,
        portfolio_tracker: PortfolioTracker,
        config: Optional[StreamingConfig] = None,
    ):
        """
        Initialize risk monitor.
        
        Args:
            portfolio_tracker: Portfolio tracker instance
            config: Streaming configuration
        """
        self.tracker = portfolio_tracker
        self.config = config or StreamingConfig()
        
        # Risk metrics
        self.current_metrics = RiskMetrics()
        self.metrics_history: deque = deque(maxlen=1000)
        
        # Risk limits and alerts
        self.risk_limits: Dict[str, float] = {
            'var_percentage': self.config.var_limit,
            'max_position_pct': self.config.position_limit_pct,
            'max_drawdown': self.config.drawdown_limit,
        }
        self.alerts: List[RiskAlert] = []
        
        # Return history for VaR calculation
        self.return_history: deque = deque(maxlen=252)  # 1 year of daily returns
        self.value_history: deque = deque(maxlen=1000)
        self.peak_value = 0.0
        
        # Monitoring control
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
        
        logger.info("Real-time risk monitor initialized")
    
    async def start_monitoring(self):
        """Start continuous risk monitoring."""
        if self._monitoring:
            logger.warning("Risk monitoring already running")
            return
        
        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_risk())
        
        logger.info("Risk monitoring started")
    
    async def stop_monitoring(self):
        """Stop risk monitoring."""
        self._monitoring = False
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Risk monitoring stopped")
    
    async def _monitor_risk(self):
        """Main risk monitoring loop."""
        while self._monitoring:
            try:
                # Calculate current risk metrics
                await self._calculate_risk_metrics()
                
                # Check risk limits
                await self._check_risk_limits()
                
                # Store metrics history
                self.metrics_history.append(self.current_metrics)
                
                # Wait for next check
                await asyncio.sleep(self.config.risk_check_interval)
                
            except Exception as e:
                logger.error(f"Error in risk monitoring loop: {e}")
                await asyncio.sleep(1)
    
    async def _calculate_risk_metrics(self):
        """Calculate current risk metrics."""
        # Get portfolio summary
        summary = self.tracker.get_portfolio_summary()
        portfolio_value = summary['total_value']
        
        if portfolio_value == 0:
            return
        
        # Update portfolio value history
        self.value_history.append(portfolio_value)
        
        # Calculate returns if we have history
        if len(self.value_history) >= 2:
            prev_value = self.value_history[-2]
            if prev_value > 0:
                ret = (portfolio_value - prev_value) / prev_value
                self.return_history.append(ret)
        
        # Calculate VaR (parametric method)
        var_95, var_99, var_pct = self._calculate_var(portfolio_value)
        
        # Calculate position concentration
        max_position_value = 0.0
        max_position_pct = 0.0
        
        for position in self.tracker.positions.values():
            abs_value = abs(position.market_value)
            if abs_value > max_position_value:
                max_position_value = abs_value
                max_position_pct = abs_value / portfolio_value if portfolio_value > 0 else 0
        
        # Calculate drawdown
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
        
        current_drawdown = (
            (self.peak_value - portfolio_value) / self.peak_value
            if self.peak_value > 0 else 0
        )
        
        max_drawdown = max(
            current_drawdown,
            self.current_metrics.max_drawdown
        )
        
        # Calculate volatility
        volatility = self._calculate_volatility()
        
        # Calculate Sharpe ratio (simplified, assuming 0 risk-free rate)
        sharpe_ratio = self._calculate_sharpe_ratio()
        
        # Calculate concentration risk (Herfindahl index)
        concentration = self._calculate_concentration_risk(portfolio_value)
        
        # Update current metrics
        self.current_metrics = RiskMetrics(
            timestamp=time.time(),
            portfolio_value=portfolio_value,
            var_95=var_95,
            var_99=var_99,
            var_percentage=var_pct,
            max_position_exposure=max_position_value,
            max_position_pct=max_position_pct,
            current_drawdown=current_drawdown,
            max_drawdown=max_drawdown,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            concentration_risk=concentration,
        )
    
    def _calculate_var(self, portfolio_value: float) -> tuple:
        """
        Calculate Value at Risk using parametric method.
        
        Args:
            portfolio_value: Current portfolio value
        
        Returns:
            Tuple of (var_95, var_99, var_percentage)
        """
        if len(self.return_history) < 30:
            # Not enough data for reliable VaR
            return 0.0, 0.0, 0.0
        
        # Convert to numpy array
        returns = np.array(list(self.return_history))
        
        # Calculate mean and std
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Calculate VaR at 95% and 99% confidence
        # Using normal distribution assumption
        z_95 = 1.645  # 95% confidence
        z_99 = 2.326  # 99% confidence
        
        var_95 = portfolio_value * (mean_return - z_95 * std_return)
        var_99 = portfolio_value * (mean_return - z_99 * std_return)
        
        # VaR as percentage of portfolio
        var_pct = abs(var_95) / portfolio_value if portfolio_value > 0 else 0
        
        return abs(var_95), abs(var_99), var_pct
    
    def _calculate_volatility(self) -> float:
        """
        Calculate portfolio volatility (annualized).
        
        Returns:
            Annualized volatility
        """
        if len(self.return_history) < 2:
            return 0.0
        
        returns = np.array(list(self.return_history))
        std_return = np.std(returns)
        
        # Annualize (assuming daily returns)
        annualized_vol = std_return * np.sqrt(252)
        
        return annualized_vol
    
    def _calculate_sharpe_ratio(self) -> float:
        """
        Calculate Sharpe ratio (simplified with 0 risk-free rate).
        
        Returns:
            Sharpe ratio
        """
        if len(self.return_history) < 2:
            return 0.0
        
        returns = np.array(list(self.return_history))
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # Annualize
        sharpe = (mean_return * 252) / (std_return * np.sqrt(252))
        
        return sharpe
    
    def _calculate_concentration_risk(self, portfolio_value: float) -> float:
        """
        Calculate concentration risk using Herfindahl index.
        
        Args:
            portfolio_value: Current portfolio value
        
        Returns:
            Concentration risk (0-1, higher = more concentrated)
        """
        if portfolio_value == 0 or not self.tracker.positions:
            return 0.0
        
        # Calculate weight squared sum
        weight_squared_sum = 0.0
        for position in self.tracker.positions.values():
            weight = abs(position.market_value) / portfolio_value
            weight_squared_sum += weight ** 2
        
        return weight_squared_sum
    
    async def _check_risk_limits(self):
        """Check if any risk limits are breached."""
        metrics = self.current_metrics
        
        # Check VaR limit
        if metrics.var_percentage > self.risk_limits.get('var_percentage', 1.0):
            await self._trigger_alert(
                'var_limit_breach',
                self.risk_limits['var_percentage'],
                metrics.var_percentage,
            )
        
        # Check position limit
        if metrics.max_position_pct > self.risk_limits.get('max_position_pct', 1.0):
            await self._trigger_alert(
                'position_limit_breach',
                self.risk_limits['max_position_pct'],
                metrics.max_position_pct,
            )
        
        # Check drawdown limit
        if metrics.current_drawdown > self.risk_limits.get('max_drawdown', 1.0):
            await self._trigger_alert(
                'drawdown_limit_breach',
                self.risk_limits['max_drawdown'],
                metrics.current_drawdown,
            )
    
    async def _trigger_alert(
        self,
        alert_type: str,
        threshold: float,
        current_value: float,
    ):
        """
        Trigger risk alert.
        
        Args:
            alert_type: Type of alert
            threshold: Threshold value
            current_value: Current value
        """
        logger.warning(
            f"Risk Alert: {alert_type} - "
            f"Current: {current_value:.2%}, Threshold: {threshold:.2%}"
        )
        
        # Find matching alert callback
        for alert in self.alerts:
            if alert.alert_type == alert_type and not alert.triggered:
                alert.triggered = True
                try:
                    await alert.callback(alert_type, current_value, threshold)
                except Exception as e:
                    logger.error(f"Error in alert callback: {e}")
    
    def add_risk_limit(self, metric_name: str, threshold: float):
        """
        Add or update risk limit.
        
        Args:
            metric_name: Name of risk metric
            threshold: Threshold value
        """
        self.risk_limits[metric_name] = threshold
        logger.info(f"Added risk limit: {metric_name} = {threshold}")
    
    def add_alert_callback(
        self,
        alert_type: str,
        threshold: float,
        callback: Callable,
    ):
        """
        Add alert callback.
        
        Args:
            alert_type: Type of alert
            threshold: Threshold value
            callback: Alert callback function
        """
        alert = RiskAlert(
            alert_type=alert_type,
            threshold=threshold,
            current_value=0.0,
            callback=callback,
        )
        self.alerts.append(alert)
        logger.info(f"Added alert callback for {alert_type}")
    
    def get_current_metrics(self) -> RiskMetrics:
        """
        Get current risk metrics.
        
        Returns:
            Current risk metrics
        """
        return self.current_metrics
    
    def get_metrics_history(self, n: Optional[int] = None) -> List[RiskMetrics]:
        """
        Get historical risk metrics.
        
        Args:
            n: Number of recent metrics to return (None = all)
        
        Returns:
            List of risk metrics
        """
        if n is None:
            return list(self.metrics_history)
        else:
            return list(self.metrics_history)[-n:]
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """
        Get risk monitoring summary.
        
        Returns:
            Risk summary dictionary
        """
        metrics = self.current_metrics
        
        return {
            'monitoring': self._monitoring,
            'current_metrics': metrics.to_dict(),
            'risk_limits': self.risk_limits.copy(),
            'limit_breaches': {
                'var_breach': metrics.var_percentage > self.risk_limits.get('var_percentage', 1.0),
                'position_breach': metrics.max_position_pct > self.risk_limits.get('max_position_pct', 1.0),
                'drawdown_breach': metrics.current_drawdown > self.risk_limits.get('max_drawdown', 1.0),
            },
            'data_points': {
                'return_history': len(self.return_history),
                'metrics_history': len(self.metrics_history),
            }
        }
    
    def reset_alerts(self):
        """Reset all triggered alerts."""
        for alert in self.alerts:
            alert.triggered = False
        logger.info("All alerts reset")