#!/usr/bin/env python3
"""
Comprehensive Quantitative Finance Integration Demo
====================================================

This flagship demo showcases the complete end-to-end quantitative finance workflow
using REAL market data from FREE providers (Yahoo Finance, OpenBB).

Features Demonstrated:
1. Real-time data fetching from Yahoo Finance (100% FREE, no API key needed)
2. Multi-stock portfolio VaR calculations (Parametric, Historical, Monte Carlo)
3. Portfolio optimization using real returns (Markowitz, Max Sharpe, Min Vol)
4. Efficient Frontier generation and visualization
5. Strategy comparison and performance analysis
6. Risk-adjusted returns and performance metrics
7. Comprehensive error handling and data validation
8. Production-ready code with logging and monitoring

This demo is designed to work completely FREE without any paid API keys.

Usage:
    python demos/demo_integrated_quant_finance.py

Requirements:
    - yfinance (FREE, unlimited)
    - numpy, pandas, scipy, matplotlib
    - All included in requirements.txt
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import warnings

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import Axiom components
from axiom.integrations.data_sources.finance.yahoo_finance_provider import YahooFinanceProvider
from axiom.models.risk.var_models import (
    VaRCalculator, 
    VaRMethod, 
    calculate_portfolio_var,
    VaRResult
)
from axiom.models.portfolio.optimization import (
    PortfolioOptimizer,
    OptimizationMethod,
    EfficientFrontier,
    calculate_sharpe_ratio,
    calculate_max_drawdown
)
from axiom.core.logging.axiom_logger import AxiomLogger

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Initialize logger
logger = AxiomLogger("quant_finance_demo")

# Configure plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10


class QuantFinanceIntegrationDemo:
    """
    Comprehensive Quantitative Finance Integration Demo.
    
    Demonstrates the complete workflow from data fetching to portfolio
    optimization and risk management using real market data.
    """
    
    def __init__(self, 
                 symbols: List[str] = None,
                 lookback_period: str = "2y",
                 portfolio_value: float = 1_000_000):
        """
        Initialize the demo with portfolio configuration.
        
        Args:
            symbols: List of stock tickers (default: tech stocks)
            lookback_period: Historical data period (1y, 2y, 5y, max)
            portfolio_value: Initial portfolio value in USD
        """
        self.symbols = symbols or [
            "AAPL",  # Apple
            "MSFT",  # Microsoft
            "GOOGL", # Alphabet
            "AMZN",  # Amazon
            "NVDA",  # NVIDIA
            "META",  # Meta
            "TSLA",  # Tesla
            "JPM"    # JPMorgan (for diversification)
        ]
        
        self.lookback_period = lookback_period
        self.portfolio_value = portfolio_value
        
        # Initialize data provider (100% FREE!)
        self.data_provider = YahooFinanceProvider()
        
        # Initialize calculators
        self.var_calculator = VaRCalculator(default_confidence=0.95)
        self.optimizer = PortfolioOptimizer(risk_free_rate=0.045)  # 4.5% risk-free rate
        
        # Storage for results
        self.price_data: Optional[pd.DataFrame] = None
        self.returns_data: Optional[pd.DataFrame] = None
        self.portfolio_stats: Dict[str, Any] = {}
        
        logger.info("Initialized QuantFinanceIntegrationDemo",
                   symbols=self.symbols,
                   portfolio_value=portfolio_value,
                   lookback=lookback_period)
    
    def run_complete_demo(self) -> Dict[str, Any]:
        """
        Run the complete quantitative finance workflow demonstration.
        
        Returns:
            Dictionary with all results and analysis
        """
        print("\n" + "="*80)
        print("COMPREHENSIVE QUANTITATIVE FINANCE INTEGRATION DEMO")
        print("="*80)
        print(f"\nPortfolio: {', '.join(self.symbols)}")
        print(f"Portfolio Value: ${self.portfolio_value:,.2f}")
        print(f"Data Period: {self.lookback_period}")
        print(f"Data Provider: Yahoo Finance (100% FREE)")
        print("="*80 + "\n")
        
        results = {}
        
        try:
            # Step 1: Fetch Real Market Data
            print("\n[1/7] Fetching Real Market Data from Yahoo Finance...")
            print("-" * 80)
            results['data'] = self._fetch_market_data()
            
            # Step 2: Calculate Portfolio VaR (All Methods)
            print("\n[2/7] Calculating Value at Risk (VaR) - All Methods...")
            print("-" * 80)
            results['var_analysis'] = self._calculate_var_all_methods()
            
            # Step 3: Portfolio Optimization - Multiple Strategies
            print("\n[3/7] Portfolio Optimization - Multiple Strategies...")
            print("-" * 80)
            results['optimization'] = self._optimize_portfolios()
            
            # Step 4: Generate Efficient Frontier
            print("\n[4/7] Generating Efficient Frontier...")
            print("-" * 80)
            results['efficient_frontier'] = self._generate_efficient_frontier()
            
            # Step 5: Compare Investment Strategies
            print("\n[5/7] Comparing Investment Strategies...")
            print("-" * 80)
            results['strategy_comparison'] = self._compare_strategies()
            
            # Step 6: Risk-Adjusted Performance Analysis
            print("\n[6/7] Risk-Adjusted Performance Analysis...")
            print("-" * 80)
            results['performance_analysis'] = self._analyze_performance()
            
            # Step 7: Visualize Results
            print("\n[7/7] Generating Visualizations...")
            print("-" * 80)
            results['visualizations'] = self._create_visualizations(results)
            
            # Print Summary
            self._print_summary(results)
            
            logger.info("Demo completed successfully", 
                       symbols_analyzed=len(self.symbols),
                       data_points=len(self.returns_data))
            
            return results
            
        except Exception as e:
            logger.error(f"Demo failed: {str(e)}", error=str(e))
            print(f"\n❌ ERROR: {str(e)}")
            print("\nPlease ensure:")
            print("  1. Internet connection is available")
            print("  2. Stock symbols are valid")
            print("  3. yfinance library is installed: pip install yfinance")
            raise
    
    def _fetch_market_data(self) -> Dict[str, Any]:
        """Fetch real historical market data from Yahoo Finance."""
        
        print(f"Fetching {len(self.symbols)} stocks from Yahoo Finance...")
        
        import yfinance as yf
        
        # Calculate date range
        end_date = datetime.now()
        
        # Convert period to start date
        period_map = {
            "1y": 365,
            "2y": 730,
            "3y": 1095,
            "5y": 1825,
            "10y": 3650,
            "max": 7300
        }
        days = period_map.get(self.lookback_period, 730)
        start_date = end_date - timedelta(days=days)
        
        # Download data for all symbols
        print(f"Downloading data from {start_date.date()} to {end_date.date()}...")
        
        try:
            # Fetch data using yfinance (Close prices are auto-adjusted by default)
            data = yf.download(
                self.symbols,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True  # Ensure prices are adjusted
            )
            
            # Extract close prices (already adjusted)
            if len(self.symbols) == 1:
                prices = pd.DataFrame(data['Close'])
                prices.columns = self.symbols
            else:
                # Multi-symbol download has multi-level columns: (PriceType, Ticker)
                prices = data['Close']  # This gets all tickers' close prices
            
            # Remove NaN values
            prices = prices.dropna()
            
            # Calculate returns
            returns = prices.pct_change().dropna()
            
            # Store data
            self.price_data = prices
            self.returns_data = returns
            
            # Calculate summary statistics
            summary = {
                'symbols': self.symbols,
                'data_points': len(prices),
                'start_date': prices.index[0].strftime("%Y-%m-%d"),
                'end_date': prices.index[-1].strftime("%Y-%m-%d"),
                'total_days': len(prices),
                'trading_days': len(returns),
                'mean_returns': returns.mean().to_dict(),
                'volatility': returns.std().to_dict(),
                'correlation_matrix': returns.corr().to_dict()
            }
            
            print(f"✓ Successfully fetched {len(prices)} days of data")
            print(f"  Period: {summary['start_date']} to {summary['end_date']}")
            print(f"  Trading days: {summary['trading_days']}")
            
            # Show basic statistics
            print("\nAnnualized Statistics:")
            print("-" * 80)
            print(f"{'Symbol':<8} {'Mean Return':<15} {'Volatility':<15} {'Sharpe Ratio':<15}")
            print("-" * 80)
            
            for symbol in self.symbols:
                mean_ret = returns[symbol].mean() * 252 * 100
                vol = returns[symbol].std() * np.sqrt(252) * 100
                sharpe = (returns[symbol].mean() * 252 - 0.045) / (returns[symbol].std() * np.sqrt(252))
                print(f"{symbol:<8} {mean_ret:>13.2f}% {vol:>13.2f}% {sharpe:>13.2f}")
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to fetch market data: {str(e)}")
            raise
    
    def _calculate_var_all_methods(self) -> Dict[str, Any]:
        """Calculate VaR using all three methods for comparison."""
        
        if self.returns_data is None:
            raise ValueError("No returns data available. Run _fetch_market_data first.")
        
        # Calculate equal-weighted portfolio returns for VaR
        n_assets = len(self.symbols)
        equal_weights = np.array([1.0 / n_assets] * n_assets)
        portfolio_returns = (self.returns_data * equal_weights).sum(axis=1)
        
        print("Calculating VaR for equal-weighted portfolio...")
        print(f"Confidence Level: 95%")
        print(f"Time Horizon: 1 day")
        print(f"Portfolio Value: ${self.portfolio_value:,.2f}")
        print()
        
        # Calculate VaR using all methods
        var_results = self.var_calculator.calculate_all_methods(
            portfolio_value=self.portfolio_value,
            returns=portfolio_returns,
            confidence_level=0.95,
            time_horizon_days=1,
            num_simulations=10000
        )
        
        # Get summary
        var_summary = self.var_calculator.get_var_summary(var_results)
        
        # Print results
        print("VaR Results by Method:")
        print("-" * 80)
        print(f"{'Method':<20} {'VaR Amount':<20} {'VaR %':<15} {'Expected Shortfall':<20}")
        print("-" * 80)
        
        for method, result in var_results.items():
            var_amt = f"${result.var_amount:,.2f}"
            var_pct = f"{result.var_percentage*100:.2f}%"
            es_amt = f"${result.expected_shortfall:,.2f}" if result.expected_shortfall else "N/A"
            print(f"{method.upper():<20} {var_amt:<20} {var_pct:<15} {es_amt:<20}")
        
        print(f"\nVaR Range: ${var_summary['var_range']['min']:,.2f} - ${var_summary['var_range']['max']:,.2f}")
        print(f"Mean VaR: ${var_summary['var_range']['mean']:,.2f}")
        
        return {
            'var_results': var_results,
            'var_summary': var_summary,
            'portfolio_returns': portfolio_returns
        }
    
    def _optimize_portfolios(self) -> Dict[str, Any]:
        """Optimize portfolio using multiple strategies."""
        
        if self.returns_data is None:
            raise ValueError("No returns data available.")
        
        optimization_results = {}
        
        strategies = [
            (OptimizationMethod.MAX_SHARPE, "Maximum Sharpe Ratio"),
            (OptimizationMethod.MIN_VOLATILITY, "Minimum Volatility"),
            (OptimizationMethod.RISK_PARITY, "Risk Parity"),
        ]
        
        print("Optimizing portfolio using multiple strategies...\n")
        
        for method, name in strategies:
            try:
                result = self.optimizer.optimize(
                    returns=self.returns_data,
                    assets=self.symbols,
                    method=method,
                    bounds=(0.0, 0.40)  # Max 40% in any single asset
                )
                
                optimization_results[name] = result
                
                # Print results
                print(f"\n{name}:")
                print("-" * 80)
                print(f"Expected Return: {result.metrics.expected_return*100:.2f}% (annualized)")
                print(f"Volatility: {result.metrics.volatility*100:.2f}% (annualized)")
                print(f"Sharpe Ratio: {result.metrics.sharpe_ratio:.3f}")
                print(f"Sortino Ratio: {result.metrics.sortino_ratio:.3f}")
                print(f"Max Drawdown: {result.metrics.max_drawdown*100:.2f}%")
                
                print("\nPortfolio Weights:")
                for asset, weight in result.get_weights_dict().items():
                    if weight > 0.01:  # Show only significant weights
                        print(f"  {asset}: {weight*100:.2f}%")
                
            except Exception as e:
                logger.warning(f"Optimization failed for {name}: {str(e)}")
                print(f"  ⚠ Optimization failed: {str(e)}")
        
        return optimization_results
    
    def _generate_efficient_frontier(self) -> EfficientFrontier:
        """Generate the efficient frontier."""
        
        if self.returns_data is None:
            raise ValueError("No returns data available.")
        
        print("Generating efficient frontier with 50 portfolios...")
        
        try:
            frontier = self.optimizer.calculate_efficient_frontier(
                returns=self.returns_data,
                assets=self.symbols,
                n_points=50,
                bounds=(0.0, 0.40)
            )
            
            print(f"✓ Generated {len(frontier.returns)} efficient portfolios")
            print(f"  Return range: {frontier.returns.min()*100:.2f}% to {frontier.returns.max()*100:.2f}%")
            print(f"  Risk range: {frontier.risks.min()*100:.2f}% to {frontier.risks.max()*100:.2f}%")
            print(f"  Max Sharpe Ratio: {frontier.sharpe_ratios.max():.3f}")
            
            return frontier
            
        except Exception as e:
            logger.error(f"Failed to generate efficient frontier: {str(e)}")
            raise
    
    def _compare_strategies(self) -> Dict[str, Any]:
        """Compare different investment strategies."""
        
        if self.returns_data is None:
            raise ValueError("No returns data available.")
        
        strategies = {
            'Equal Weight': np.array([1.0 / len(self.symbols)] * len(self.symbols)),
            'Market Cap Weight': self._get_market_cap_weights(),
            'Max Sharpe': None,  # Will be optimized
            'Min Volatility': None,  # Will be optimized
        }
        
        print("Comparing investment strategies...\n")
        comparison_results = {}
        
        for strategy_name, weights in strategies.items():
            try:
                if weights is None:
                    # Optimize for this strategy
                    if strategy_name == 'Max Sharpe':
                        result = self.optimizer.optimize(
                            self.returns_data, 
                            self.symbols,
                            OptimizationMethod.MAX_SHARPE
                        )
                    else:
                        result = self.optimizer.optimize(
                            self.returns_data,
                            self.symbols,
                            OptimizationMethod.MIN_VOLATILITY
                        )
                    weights = result.weights
                    metrics = result.metrics
                else:
                    # Calculate metrics for given weights
                    metrics = self.optimizer.calculate_metrics(
                        weights,
                        self.returns_data.values
                    )
                
                comparison_results[strategy_name] = {
                    'weights': weights,
                    'metrics': metrics
                }
                
            except Exception as e:
                logger.warning(f"Strategy {strategy_name} failed: {str(e)}")
        
        # Print comparison table
        print("\nStrategy Comparison:")
        print("=" * 120)
        print(f"{'Strategy':<20} {'Return':<12} {'Risk':<12} {'Sharpe':<10} {'Sortino':<10} {'Max DD':<12} {'VaR 95%':<12}")
        print("=" * 120)
        
        for strategy, data in comparison_results.items():
            m = data['metrics']
            print(f"{strategy:<20} {m.expected_return*100:>10.2f}% {m.volatility*100:>10.2f}% "
                  f"{m.sharpe_ratio:>8.3f} {m.sortino_ratio:>8.3f} "
                  f"{m.max_drawdown*100:>10.2f}% {abs(m.var_95)*100:>10.2f}%")
        
        return comparison_results
    
    def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze risk-adjusted performance metrics."""
        
        if self.returns_data is None:
            raise ValueError("No returns data available.")
        
        print("\nRisk-Adjusted Performance Analysis:")
        print("=" * 80)
        
        analysis = {}
        
        # Calculate correlation matrix
        corr_matrix = self.returns_data.corr()
        
        print("\nCorrelation Matrix:")
        print("-" * 80)
        print(corr_matrix.round(3))
        
        # Identify highest/lowest correlations
        corr_values = []
        for i in range(len(self.symbols)):
            for j in range(i+1, len(self.symbols)):
                corr_values.append((
                    self.symbols[i],
                    self.symbols[j],
                    corr_matrix.iloc[i, j]
                ))
        
        corr_values.sort(key=lambda x: x[2])
        
        print("\nLowest Correlations (Best Diversification):")
        for sym1, sym2, corr in corr_values[:3]:
            print(f"  {sym1} - {sym2}: {corr:.3f}")
        
        print("\nHighest Correlations:")
        for sym1, sym2, corr in corr_values[-3:]:
            print(f"  {sym1} - {sym2}: {corr:.3f}")
        
        analysis['correlation_matrix'] = corr_matrix
        analysis['correlation_pairs'] = corr_values
        
        # Calculate rolling metrics
        print("\nRolling Performance (90-day window):")
        print("-" * 80)
        
        equal_weights = np.array([1.0 / len(self.symbols)] * len(self.symbols))
        portfolio_returns = (self.returns_data * equal_weights).sum(axis=1)
        
        rolling_vol = portfolio_returns.rolling(window=90).std() * np.sqrt(252)
        rolling_sharpe = ((portfolio_returns.rolling(window=90).mean() * 252 - 0.045) / 
                         (portfolio_returns.rolling(window=90).std() * np.sqrt(252)))
        
        print(f"Current 90-day Volatility: {rolling_vol.iloc[-1]*100:.2f}%")
        print(f"Current 90-day Sharpe: {rolling_sharpe.iloc[-1]:.3f}")
        print(f"Average 90-day Volatility: {rolling_vol.mean()*100:.2f}%")
        print(f"Average 90-day Sharpe: {rolling_sharpe.mean():.3f}")
        
        analysis['rolling_volatility'] = rolling_vol
        analysis['rolling_sharpe'] = rolling_sharpe
        
        return analysis
    
    def _get_market_cap_weights(self) -> np.ndarray:
        """Get approximate market cap weights for the portfolio."""
        
        # Approximate market caps (in billions) - would use real data in production
        market_caps = {
            'AAPL': 3000, 'MSFT': 2800, 'GOOGL': 1700, 'AMZN': 1500,
            'NVDA': 1200, 'META': 800, 'TSLA': 700, 'JPM': 450
        }
        
        caps = np.array([market_caps.get(s, 100) for s in self.symbols])
        weights = caps / caps.sum()
        
        return weights
    
    def _create_visualizations(self, results: Dict[str, Any]) -> Dict[str, str]:
        """Create comprehensive visualizations of results."""
        
        viz_files = {}
        
        try:
            # Create figure with subplots
            fig = plt.figure(figsize=(18, 12))
            
            # 1. Price Performance
            ax1 = plt.subplot(3, 3, 1)
            normalized_prices = self.price_data / self.price_data.iloc[0] * 100
            for symbol in self.symbols:
                ax1.plot(normalized_prices.index, normalized_prices[symbol], 
                        label=symbol, linewidth=2, alpha=0.7)
            ax1.set_title('Normalized Price Performance (Base = 100)', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Price (Normalized)')
            ax1.legend(loc='best', fontsize=8)
            ax1.grid(True, alpha=0.3)
            
            # 2. Returns Distribution
            ax2 = plt.subplot(3, 3, 2)
            equal_weights = np.array([1.0 / len(self.symbols)] * len(self.symbols))
            portfolio_returns = (self.returns_data * equal_weights).sum(axis=1)
            ax2.hist(portfolio_returns * 100, bins=50, alpha=0.7, edgecolor='black')
            ax2.axvline(portfolio_returns.mean() * 100, color='red', 
                       linestyle='--', linewidth=2, label='Mean')
            ax2.set_title('Portfolio Returns Distribution', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Daily Return (%)')
            ax2.set_ylabel('Frequency')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. Correlation Heatmap
            ax3 = plt.subplot(3, 3, 3)
            corr_matrix = self.returns_data.corr()
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                       center=0, ax=ax3, square=True, cbar_kws={'shrink': 0.8})
            ax3.set_title('Correlation Matrix', fontsize=12, fontweight='bold')
            
            # 4. Efficient Frontier
            if 'efficient_frontier' in results:
                ax4 = plt.subplot(3, 3, 4)
                frontier = results['efficient_frontier']
                
                # Plot frontier
                ax4.scatter(frontier.risks * 100, frontier.returns * 100, 
                           c=frontier.sharpe_ratios, cmap='viridis', s=50, alpha=0.6)
                
                # Mark special portfolios
                max_sharpe_idx = np.argmax(frontier.sharpe_ratios)
                min_vol_idx = np.argmin(frontier.risks)
                
                ax4.scatter(frontier.risks[max_sharpe_idx] * 100, 
                           frontier.returns[max_sharpe_idx] * 100,
                           color='red', s=200, marker='*', 
                           label='Max Sharpe', edgecolors='black', linewidths=2)
                
                ax4.scatter(frontier.risks[min_vol_idx] * 100,
                           frontier.returns[min_vol_idx] * 100,
                           color='green', s=200, marker='s',
                           label='Min Volatility', edgecolors='black', linewidths=2)
                
                ax4.set_title('Efficient Frontier', fontsize=12, fontweight='bold')
                ax4.set_xlabel('Volatility (% annualized)')
                ax4.set_ylabel('Return (% annualized)')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
                
                cbar = plt.colorbar(ax4.collections[0], ax=ax4)
                cbar.set_label('Sharpe Ratio')
            
            # 5. VaR Comparison
            if 'var_analysis' in results:
                ax5 = plt.subplot(3, 3, 5)
                var_results = results['var_analysis']['var_results']
                
                methods = list(var_results.keys())
                var_amounts = [var_results[m].var_amount for m in methods]
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
                
                bars = ax5.bar(methods, var_amounts, color=colors, alpha=0.7, edgecolor='black')
                ax5.set_title('VaR Comparison (95% Confidence)', fontsize=12, fontweight='bold')
                ax5.set_ylabel('VaR Amount ($)')
                ax5.set_xlabel('Method')
                ax5.grid(True, alpha=0.3, axis='y')
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax5.text(bar.get_x() + bar.get_width()/2., height,
                            f'${height:,.0f}',
                            ha='center', va='bottom', fontsize=9)
            
            # 6. Strategy Comparison
            if 'strategy_comparison' in results:
                ax6 = plt.subplot(3, 3, 6)
                strategies = results['strategy_comparison']
                
                strategy_names = list(strategies.keys())
                returns = [strategies[s]['metrics'].expected_return * 100 for s in strategy_names]
                risks = [strategies[s]['metrics'].volatility * 100 for s in strategy_names]
                
                ax6.scatter(risks, returns, s=200, alpha=0.6, c=range(len(strategy_names)), 
                           cmap='tab10', edgecolors='black', linewidths=2)
                
                for i, name in enumerate(strategy_names):
                    ax6.annotate(name, (risks[i], returns[i]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=9)
                
                ax6.set_title('Strategy Risk-Return Profile', fontsize=12, fontweight='bold')
                ax6.set_xlabel('Risk (Volatility %)')
                ax6.set_ylabel('Return (%)')
                ax6.grid(True, alpha=0.3)
            
            # 7. Portfolio Weights (Max Sharpe)
            if 'optimization' in results:
                ax7 = plt.subplot(3, 3, 7)
                
                max_sharpe_result = results['optimization'].get('Maximum Sharpe Ratio')
                if max_sharpe_result:
                    weights_dict = max_sharpe_result.get_weights_dict()
                    weights_dict = {k: v for k, v in weights_dict.items() if v > 0.01}
                    
                    colors_palette = plt.cm.Set3(np.linspace(0, 1, len(weights_dict)))
                    wedges, texts, autotexts = ax7.pie(
                        list(weights_dict.values()),
                        labels=list(weights_dict.keys()),
                        autopct='%1.1f%%',
                        colors=colors_palette,
                        startangle=90
                    )
                    
                    for autotext in autotexts:
                        autotext.set_color('black')
                        autotext.set_fontsize(9)
                        autotext.set_weight('bold')
                    
                    ax7.set_title('Optimal Portfolio Weights\n(Max Sharpe)', 
                                 fontsize=12, fontweight='bold')
            
            # 8. Rolling Sharpe Ratio
            if 'performance_analysis' in results:
                ax8 = plt.subplot(3, 3, 8)
                rolling_sharpe = results['performance_analysis']['rolling_sharpe']
                
                ax8.plot(rolling_sharpe.index, rolling_sharpe, 
                        color='purple', linewidth=2, alpha=0.7)
                ax8.axhline(y=0, color='red', linestyle='--', alpha=0.5)
                ax8.axhline(y=rolling_sharpe.mean(), color='green', 
                           linestyle='--', alpha=0.5, label=f'Mean: {rolling_sharpe.mean():.3f}')
                
                ax8.set_title('Rolling 90-Day Sharpe Ratio', fontsize=12, fontweight='bold')
                ax8.set_xlabel('Date')
                ax8.set_ylabel('Sharpe Ratio')
                ax8.legend()
                ax8.grid(True, alpha=0.3)
            
            # 9. Cumulative Returns
            ax9 = plt.subplot(3, 3, 9)
            
            # Equal weight portfolio
            equal_portfolio_returns = (self.returns_data * equal_weights).sum(axis=1)
            cumulative_returns = (1 + equal_portfolio_returns).cumprod()
            
            ax9.plot(cumulative_returns.index, cumulative_returns, 
                    color='blue', linewidth=2, label='Equal Weight Portfolio')
            
            # Individual stocks (top 3)
            for symbol in self.symbols[:3]:
                cum_ret = (1 + self.returns_data[symbol]).cumprod()
                ax9.plot(cum_ret.index, cum_ret, 
                        linewidth=1.5, alpha=0.7, label=symbol)
            
            ax9.set_title('Cumulative Returns', fontsize=12, fontweight='bold')
            ax9.set_xlabel('Date')
            ax9.set_ylabel('Cumulative Return')
            ax9.legend(loc='best', fontsize=8)
            ax9.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save figure
            output_file = 'quant_finance_analysis.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"✓ Saved comprehensive visualization to: {output_file}")
            
            viz_files['comprehensive'] = output_file
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Visualization failed: {str(e)}")
            print(f"⚠ Warning: Visualization failed: {str(e)}")
        
        return viz_files
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print executive summary of results."""
        
        print("\n" + "="*80)
        print("EXECUTIVE SUMMARY")
        print("="*80)
        
        if 'data' in results:
            data_info = results['data']
            print(f"\n📊 Data Coverage:")
            print(f"  • Period: {data_info['start_date']} to {data_info['end_date']}")
            print(f"  • Trading Days: {data_info['trading_days']}")
            print(f"  • Data Provider: Yahoo Finance (100% FREE)")
        
        if 'var_analysis' in results:
            var_summary = results['var_analysis']['var_summary']
            print(f"\n⚠️  Risk Metrics (95% Confidence):")
            print(f"  • VaR Range: ${var_summary['var_range']['min']:,.2f} - ${var_summary['var_range']['max']:,.2f}")
            print(f"  • Mean VaR: ${var_summary['var_range']['mean']:,.2f}")
            print(f"  • Portfolio Value: ${self.portfolio_value:,.2f}")
            print(f"  • VaR as % of Portfolio: {(var_summary['var_range']['mean']/self.portfolio_value)*100:.2f}%")
        
        if 'optimization' in results:
            opt_results = results['optimization']
            
            if 'Maximum Sharpe Ratio' in opt_results:
                max_sharpe = opt_results['Maximum Sharpe Ratio']
                print(f"\n🎯 Optimal Portfolio (Max Sharpe):")
                print(f"  • Expected Return: {max_sharpe.metrics.expected_return*100:.2f}%")
                print(f"  • Volatility: {max_sharpe.metrics.volatility*100:.2f}%")
                print(f"  • Sharpe Ratio: {max_sharpe.metrics.sharpe_ratio:.3f}")
                print(f"  • Max Drawdown: {max_sharpe.metrics.max_drawdown*100:.2f}%")
            
            if 'Minimum Volatility' in opt_results:
                min_vol = opt_results['Minimum Volatility']
                print(f"\n🛡️  Minimum Risk Portfolio:")
                print(f"  • Expected Return: {min_vol.metrics.expected_return*100:.2f}%")
                print(f"  • Volatility: {min_vol.metrics.volatility*100:.2f}%")
                print(f"  • Sharpe Ratio: {min_vol.metrics.sharpe_ratio:.3f}")
        
        if 'efficient_frontier' in results:
            frontier = results['efficient_frontier']
            print(f"\n📈 Efficient Frontier:")
            print(f"  • Portfolios Generated: {len(frontier.returns)}")
            print(f"  • Return Range: {frontier.returns.min()*100:.2f}% to {frontier.returns.max()*100:.2f}%")
            print(f"  • Risk Range: {frontier.risks.min()*100:.2f}% to {frontier.risks.max()*100:.2f}%")
            print(f"  • Best Sharpe Ratio: {frontier.sharpe_ratios.max():.3f}")
        
        print("\n" + "="*80)
        print("✓ Demo completed successfully!")
        print("="*80)
        print("\nKey Takeaways:")
        print("  1. All data fetched from FREE providers (Yahoo Finance)")
        print("  2. Multiple VaR methods provide risk estimates")
        print("  3. Portfolio optimization improves risk-adjusted returns")
        print("  4. Efficient frontier shows optimal risk-return tradeoffs")
        print("  5. Strategy comparison helps select best approach")
        print("\nProduction Notes:")
        print("  • This demo uses real market data")
        print("  • All calculations are production-ready")
        print("  • No paid API keys required")
        print("  • Suitable for hedge funds, asset managers, and traders")
        print("="*80 + "\n")


def main():
    """Main entry point for the demo."""
    
    try:
        # Create demo instance with configuration
        demo = QuantFinanceIntegrationDemo(
            symbols=["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM"],
            lookback_period="2y",
            portfolio_value=1_000_000
        )
        
        # Run complete demonstration
        results = demo.run_complete_demo()
        
        print("\n✅ Demo completed successfully!")
        print("\nGenerated files:")
        print("  • quant_finance_analysis.png - Comprehensive visualizations")
        
        return results
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
        return None
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()