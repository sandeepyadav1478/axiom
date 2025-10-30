"""
Risk Analytics Integration using QuantStats

Leverages QuantStats open-source library for comprehensive risk and performance metrics
instead of building custom calculations from scratch.

QuantStats provides 40+ metrics, visual reports, and comparison tools used by
professional quant traders worldwide.
"""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from datetime import datetime

try:
    import quantstats as qs
    QUANTSTATS_AVAILABLE = True
except ImportError:
    QUANTSTATS_AVAILABLE = False


class AxiomRiskAnalytics:
    """
    Unified risk analytics using QuantStats
    
    Provides institutional-grade risk metrics without custom implementation.
    
    Usage:
        analytics = AxiomRiskAnalytics()
        
        # Get all metrics
        metrics = analytics.calculate_metrics(portfolio_returns)
        
        # Generate HTML report
        analytics.generate_report(
            returns=portfolio_returns,
            benchmark=spy_returns,
            output_file="portfolio_report.html"
        )
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        if not QUANTSTATS_AVAILABLE:
            raise ImportError("QuantStats required: pip install quantstats")
        
        self.risk_free_rate = risk_free_rate
        
    def calculate_metrics(
        self,
        returns: Union[pd.Series, np.ndarray],
        benchmark: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive risk metrics
        
        Args:
            returns: Portfolio returns (daily)
            benchmark: Benchmark returns (optional)
            
        Returns:
            Dictionary with 20+ risk metrics
        """
        # Convert to pandas Series if needed
        if isinstance(returns, np.ndarray):
            returns = pd.Series(returns)
        
        metrics = {
            # Return metrics
            'cumulative_return': qs.stats.comp(returns),
            'cagr': qs.stats.cagr(returns),
            'avg_return': qs.stats.avg_return(returns),
            'avg_win': qs.stats.avg_win(returns),
            'avg_loss': qs.stats.avg_loss(returns),
            
            # Risk metrics
            'volatility': qs.stats.volatility(returns),
            'max_drawdown': qs.stats.max_drawdown(returns),
            'calmar': qs.stats.calmar(returns),
            'var': qs.stats.var(returns),
            'cvar': qs.stats.cvar(returns),
            
            # Risk-adjusted returns
            'sharpe': qs.stats.sharpe(returns, rf=self.risk_free_rate),
            'sortino': qs.stats.sortino(returns, rf=self.risk_free_rate),
            'omega': qs.stats.omega(returns),
            
            # Win/Loss metrics
            'win_rate': qs.stats.win_rate(returns),
            'profit_factor': qs.stats.profit_factor(returns),
            'payoff_ratio': qs.stats.payoff_ratio(returns),
            
            # Tail risk
            'skew': qs.stats.skew(returns),
            'kurtosis': qs.stats.kurtosis(returns),
            'tail_ratio': qs.stats.tail_ratio(returns),
            
            # Recovery metrics
            'recovery_factor': qs.stats.recovery_factor(returns),
            'ulcer_index': qs.stats.ulcer_index(returns),
        }
        
        # Benchmark-relative metrics
        if benchmark is not None:
            if isinstance(benchmark, np.ndarray):
                benchmark = pd.Series(benchmark)
            
            metrics.update({
                'alpha': qs.stats.alpha(returns, benchmark, rf=self.risk_free_rate),
                'beta': qs.stats.beta(returns, benchmark),
                'r_squared': qs.stats.r_squared(returns, benchmark),
                'information_ratio': qs.stats.information_ratio(returns, benchmark),
                'treynor_ratio': qs.stats.treynor_ratio(returns, benchmark, rf=self.risk_free_rate)
            })
        
        return metrics
        
    def generate_report(
        self,
        returns: Union[pd.Series, np.ndarray],
        benchmark: Optional[Union[pd.Series, np.ndarray]] = None,
        output_file: str = "portfolio_report.html",
        title: str = "Axiom Portfolio Performance"
    ):
        """
        Generate comprehensive HTML report
        
        Args:
            returns: Portfolio returns
            benchmark: Benchmark returns (optional)
            output_file: Output HTML file path
            title: Report title
        """
        if isinstance(returns, np.ndarray):
            returns = pd.Series(returns)
        if benchmark is not None and isinstance(benchmark, np.ndarray):
            benchmark = pd.Series(benchmark)
        
        # Generate full report
        if benchmark is not None:
            qs.reports.html(
                returns,
                benchmark=benchmark,
                output=output_file,
                title=title
            )
        else:
            qs.reports.html(
                returns,
                output=output_file,
                title=title
            )
            
    def plot_performance(
        self,
        returns: Union[pd.Series, np.ndarray],
        benchmark: Optional[Union[pd.Series, np.ndarray]] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot performance overview
        
        Creates matplotlib figure with key visualizations.
        """
        if isinstance(returns, np.ndarray):
            returns = pd.Series(returns)
        
        if benchmark is not None and isinstance(benchmark, np.ndarray):
            benchmark = pd.Series(benchmark)
        
        # Create plots
        import matplotlib.pyplot as plt
        
        if save_path:
            qs.plots.snapshot(returns, savefig=save_path)
        else:
            qs.plots.snapshot(returns)
            plt.show()
            
    def compare_strategies(
        self,
        strategies: Dict[str, pd.Series],
        benchmark: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Compare multiple strategies
        
        Args:
            strategies: Dict of {name: returns_series}
            benchmark: Optional benchmark
            
        Returns:
            Comparison DataFrame
        """
        comparison_data = []
        
        for name, returns in strategies.items():
            metrics = self.calculate_metrics(returns, benchmark)
            metrics['strategy'] = name
            comparison_data.append(metrics)
        
        df = pd.DataFrame(comparison_data)
        df = df.set_index('strategy')
        
        return df


# Convenience functions
def quick_analysis(
    returns: Union[pd.Series, np.ndarray, List[float]],
    benchmark: Optional[Union[pd.Series, np.ndarray]] = None
) -> Dict[str, float]:
    """
    Quick risk analysis - convenience function
    
    Usage:
        metrics = quick_analysis(portfolio_returns)
        print(f"Sharpe: {metrics['sharpe']:.2f}")
    """
    analytics = AxiomRiskAnalytics()
    return analytics.calculate_metrics(returns, benchmark)


def generate_performance_report(
    returns: Union[pd.Series, np.ndarray],
    benchmark: Optional[Union[pd.Series, np.ndarray]] = None,
    output_file: str = "performance_report.html"
):
    """
    Generate performance report - convenience function
    
    Usage:
        generate_performance_report(
            returns=portfolio_returns,
            benchmark=spy_returns,
            output_file="my_strategy.html"
        )
    """
    analytics = AxiomRiskAnalytics()
    analytics.generate_report(returns, benchmark, output_file)


# Example usage
if __name__ == "__main__":
    print("QuantStats Risk Analytics - Example")
    print("=" * 60)
    
    if not QUANTSTATS_AVAILABLE:
        print("Install: pip install quantstats")
    else:
        # Sample returns
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))  # 1 year daily
        
        # Calculate metrics
        analytics = AxiomRiskAnalytics()
        metrics = analytics.calculate_metrics(returns)
        
        print("\nKey Metrics:")
        print(f"  Sharpe Ratio: {metrics['sharpe']:.3f}")
        print(f"  Sortino Ratio: {metrics['sortino']:.3f}")
        print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"  CAGR: {metrics['cagr']:.2%}")
        print(f"  Volatility: {metrics['volatility']:.2%}")
        print(f"  Win Rate: {metrics['win_rate']:.2%}")
        
        print("\n" + "=" * 60)
        print("✓ Leveraging QuantStats instead of custom metrics")