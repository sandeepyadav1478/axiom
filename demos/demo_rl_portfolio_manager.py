"""
Demo: RL Portfolio Manager with PPO

This demo showcases the Reinforcement Learning-based Portfolio Manager
using Proximal Policy Optimization (PPO) for optimal asset allocation.

Based on research from:
Wu Junfeng, Li Yaoming, Tan Wenqing, Chen Yun (2024)
"Portfolio management based on a reinforcement learning framework"
Journal of Forecasting, Volume 43, Issue 7
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

try:
    from axiom.models.portfolio.rl_portfolio_manager import (
        RLPortfolioManager,
        PortfolioConfig,
        create_sample_data
    )
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    IMPORTS_AVAILABLE = False


def generate_realistic_market_data(
    n_timesteps: int = 1000,
    n_assets: int = 6,
    start_date: str = "2020-01-01"
) -> pd.DataFrame:
    """
    Generate realistic synthetic market data for portfolio optimization
    
    Creates correlated asset returns with different characteristics:
    - Tech stocks (high return, high volatility)
    - Blue chips (moderate return, low volatility)  
    - Bonds (low return, very low volatility)
    """
    np.random.seed(42)
    
    # Generate dates
    dates = pd.date_range(start=start_date, periods=n_timesteps, freq='D')
    
    # Asset characteristics
    assets = {
        'TECH_A': {'mu': 0.0008, 'sigma': 0.025},  # High risk, high return
        'TECH_B': {'mu': 0.0007, 'sigma': 0.023},
        'BLUE_A': {'mu': 0.0005, 'sigma': 0.015},  # Moderate risk/return
        'BLUE_B': {'mu': 0.0004, 'sigma': 0.013},
        'BOND_A': {'mu': 0.0002, 'sigma': 0.005},  # Low risk/return
        'BOND_B': {'mu': 0.0002, 'sigma': 0.004},
    }
    
    # Generate correlated returns
    correlation_matrix = np.array([
        [1.00, 0.70, 0.30, 0.25, 0.05, 0.03],
        [0.70, 1.00, 0.25, 0.30, 0.03, 0.05],
        [0.30, 0.25, 1.00, 0.60, 0.15, 0.12],
        [0.25, 0.30, 0.60, 1.00, 0.12, 0.15],
        [0.05, 0.03, 0.15, 0.12, 1.00, 0.50],
        [0.03, 0.05, 0.12, 0.15, 0.50, 1.00],
    ])
    
    # Generate correlated random returns
    L = np.linalg.cholesky(correlation_matrix)
    uncorrelated = np.random.randn(n_timesteps, n_assets)
    correlated_returns = uncorrelated @ L.T
    
    # Scale by volatility and add drift
    prices = {}
    for i, (name, params) in enumerate(assets.items()):
        returns = correlated_returns[:, i] * params['sigma'] + params['mu']
        # Convert to prices
        prices[name] = 100 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame(prices, index=dates)
    
    # Add technical indicators as features
    features_df = pd.DataFrame(index=dates)
    
    for asset in df.columns:
        # Price
        features_df[f'{asset}_price'] = df[asset]
        
        # Returns  
        features_df[f'{asset}_return'] = df[asset].pct_change().fillna(0)
        
        # Moving averages
        features_df[f'{asset}_ma5'] = df[asset].rolling(5).mean().fillna(method='bfill')
        features_df[f'{asset}_ma20'] = df[asset].rolling(20).mean().fillna(method='bfill')
        
        # Volatility
        features_df[f'{asset}_vol'] = df[asset].pct_change().rolling(20).std().fillna(0)
        
        # RSI-like indicator
        delta = df[asset].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        features_df[f'{asset}_rsi'] = 100 - (100 / (1 + rs)).fillna(50)
        
        # Momentum
        features_df[f'{asset}_momentum'] = df[asset] / df[asset].shift(20).fillna(method='bfill') - 1
        
        # Volume proxy (random but correlated with price changes)
        price_changes = df[asset].pct_change().abs()
        features_df[f'{asset}_volume'] = (price_changes * np.random.uniform(0.8, 1.2, len(df))).fillna(1.0)
        
        # Bollinger band position
        ma = df[asset].rolling(20).mean()
        std = df[asset].rolling(20).std()
        features_df[f'{asset}_bb'] = ((df[asset] - ma) / (2 * std + 1e-8)).fillna(0)
        
        # MACD-like
        ema12 = df[asset].ewm(span=12).mean()
        ema26 = df[asset].ewm(span=26).mean()
        features_df[f'{asset}_macd'] = (ema12 - ema26).fillna(0)
        
        # Price relative to max
        rolling_max = df[asset].rolling(50).max()
        features_df[f'{asset}_rel_max'] = (df[asset] / rolling_max).fillna(1.0)
        
        # Trend strength
        x = np.arange(20)
        def calc_slope(y):
            if len(y) < 20:
                return 0
            return np.polyfit(x, y[-20:], 1)[0]
        features_df[f'{asset}_trend'] = df[asset].rolling(20).apply(calc_slope, raw=True).fillna(0)
        
        # Average true range proxy
        high_low = df[asset].rolling(2).max() - df[asset].rolling(2).min()
        features_df[f'{asset}_atr'] = high_low.rolling(14).mean().fillna(0)
        
        # On-balance volume proxy
        obv = (features_df[f'{asset}_volume'] * np.sign(features_df[f'{asset}_return'])).cumsum()
        features_df[f'{asset}_obv'] = (obv - obv.rolling(20).mean()).fillna(0)
        
        # Stochastic oscillator
        low14 = df[asset].rolling(14).min()
        high14 = df[asset].rolling(14).max()
        features_df[f'{asset}_stoch'] = ((df[asset] - low14) / (high14 - low14 + 1e-8) * 100).fillna(50)
        
        # Williams %R
        features_df[f'{asset}_williams'] = ((high14 - df[asset]) / (high14 - low14 + 1e-8) * -100).fillna(-50)
    
    # Normalize features
    for col in features_df.columns:
        mean = features_df[col].mean()
        std = features_df[col].std()
        if std > 0:
            features_df[col] = (features_df[col] - mean) / std
        else:
            features_df[col] = 0
    
    return features_df


def plot_portfolio_performance(results: dict, title: str = "Portfolio Performance"):
    """Plot portfolio backtest results"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Portfolio value over time
    ax1 = axes[0]
    portfolio_values = results['portfolio_values']
    ax1.plot(portfolio_values, linewidth=2, label='Portfolio Value')
    ax1.set_title(f'{title} - Portfolio Value', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Trading Days')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Cumulative returns
    ax2 = axes[1]
    returns = np.array(results['returns'])
    cumulative_returns = (1 + returns).cumprod() - 1
    ax2.plot(cumulative_returns * 100, linewidth=2, color='green', label='Cumulative Return')
    ax2.set_title('Cumulative Returns', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Trading Days')
    ax2.set_ylabel('Return (%)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Portfolio weights over time
    ax3 = axes[2]
    weights_history = np.array(results['weights_history'])
    n_assets = weights_history.shape[1]
    
    # Create stacked area plot
    ax3.stackplot(
        range(len(weights_history)),
        *[weights_history[:, i] for i in range(n_assets)],
        labels=[f'Asset {i+1}' for i in range(n_assets)],
        alpha=0.7
    )
    ax3.set_title('Portfolio Allocation Over Time', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Trading Days')
    ax3.set_ylabel('Weight')
    ax3.set_ylim([0, 1])
    ax3.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def main():
    """Main demo function"""
    print("=" * 70)
    print("RL Portfolio Manager Demo")
    print("Reinforcement Learning for Optimal Asset Allocation")
    print("=" * 70)
    print()
    
    if not IMPORTS_AVAILABLE:
        print("ERROR: Required modules not available. Please install dependencies:")
        print("  pip install torch gymnasium stable-baselines3")
        return
    
    # Configuration
    print("1. Configuration")
    print("-" * 70)
    config = PortfolioConfig(
        n_assets=6,
        n_features=16,
        lookback_window=30,
        transaction_cost=0.001,  # 0.1%
        risk_free_rate=0.02,
        rebalance_frequency="monthly"
    )
    print(f"  Number of assets: {config.n_assets}")
    print(f"  Features per asset: {config.n_features}")
    print(f"  Lookback window: {config.lookback_window} days")
    print(f"  Transaction cost: {config.transaction_cost:.2%}")
    print(f"  Risk-free rate: {config.risk_free_rate:.2%}")
    print()
    
    # Generate data
    print("2. Generating Market Data")
    print("-" * 70)
    print("  Creating realistic synthetic market data...")
    print("  Assets: 2 Tech stocks, 2 Blue chips, 2 Bonds")
    
    # Full dataset
    full_data = generate_realistic_market_data(n_timesteps=800, n_assets=6)
    
    # Split into train/test
    train_size = 600
    train_data = full_data.iloc[:train_size]
    test_data = full_data.iloc[train_size:]
    
    print(f"  Training period: {train_size} days")
    print(f"  Testing period: {len(test_data)} days")
    print(f"  Total features: {full_data.shape[1]}")
    print()
    
    # Initialize manager
    print("3. Initializing RL Portfolio Manager")
    print("-" * 70)
    manager = RLPortfolioManager(config)
    print("  ✓ Manager initialized with CNN feature extractor")
    print("  ✓ PPO algorithm configured")
    print()
    
    # Train
    print("4. Training Portfolio Manager")
    print("-" * 70)
    print("  Training with PPO algorithm...")
    print("  This may take a few minutes...")
    
    try:
        training_history = manager.train(
            train_data=train_data,
            total_timesteps=50000,
            verbose=0
        )
        print("  ✓ Training completed successfully")
    except Exception as e:
        print(f"  ✗ Training failed: {e}")
        return
    print()
    
    # Backtest
    print("5. Backtesting on Test Data")
    print("-" * 70)
    print("  Running backtest on unseen data...")
    
    try:
        results = manager.backtest(
            test_data=test_data,
            initial_capital=10000.0
        )
        print("  ✓ Backtest completed")
    except Exception as e:
        print(f"  ✗ Backtest failed: {e}")
        return
    print()
    
    # Performance metrics
    print("6. Performance Metrics")
    print("-" * 70)
    print(f"  Initial Capital:    ${10000.00:,.2f}")
    print(f"  Final Value:        ${results['final_value']:,.2f}")
    print(f"  Total Return:       {results['total_return']:.2%}")
    print(f"  Annualized Return:  {results['total_return'] * 252/len(test_data):.2%}")
    print(f"  Sharpe Ratio:       {results['sharpe_ratio']:.3f}")
    print(f"  Max Drawdown:       {results['max_drawdown']:.2%}")
    print()
    
    # Portfolio statistics
    returns = np.array(results['returns'])
    print("7. Return Statistics")
    print("-" * 70)
    print(f"  Mean Daily Return:  {np.mean(returns):.4%}")
    print(f"  Std Dev:            {np.std(returns):.4%}")
    print(f"  Min Return:         {np.min(returns):.4%}")
    print(f"  Max Return:         {np.max(returns):.4%}")
    print(f"  Win Rate:           {np.sum(returns > 0) / len(returns):.2%}")
    print()
    
    # Asset allocation
    final_weights = results['weights_history'][-1]
    print("8. Final Portfolio Allocation")
    print("-" * 70)
    assets = ['TECH_A', 'TECH_B', 'BLUE_A', 'BLUE_B', 'BOND_A', 'BOND_B']
    for asset, weight in zip(assets, final_weights):
        print(f"  {asset:8s}: {weight:6.2%} {'█' * int(weight * 50)}")
    print()
    
    # Comparison with equal-weight portfolio
    print("9. Comparison with Baseline Strategies")
    print("-" * 70)
    
    # Calculate equal-weight portfolio performance
    test_prices = full_data.iloc[train_size:]
    price_columns = [col for col in test_prices.columns if '_price' in col]
    equal_weight_returns = test_prices[price_columns].pct_change().mean(axis=1).dropna()
    equal_weight_value = 10000 * (1 + equal_weight_returns).cumprod().iloc[-1]
    equal_weight_sharpe = np.mean(equal_weight_returns) / (np.std(equal_weight_returns) + 1e-8) * np.sqrt(252)
    
    print(f"  Equal-Weight Portfolio:")
    print(f"    Final Value:  ${equal_weight_value:,.2f}")
    print(f"    Return:       {(equal_weight_value - 10000) / 10000:.2%}")
    print(f"    Sharpe Ratio: {equal_weight_sharpe:.3f}")
    print()
    print(f"  RL Portfolio (PPO):")
    print(f"    Final Value:  ${results['final_value']:,.2f}")
    print(f"    Return:       {results['total_return']:.2%}")
    print(f"    Sharpe Ratio: {results['sharpe_ratio']:.3f}")
    print()
    
    improvement = (results['final_value'] - equal_weight_value) / equal_weight_value
    print(f"  Performance Improvement: {improvement:.2%}")
    print()
    
    # Visualization
    print("10. Generating Visualizations")
    print("-" * 70)
    try:
        fig = plot_portfolio_performance(results, title="RL Portfolio Manager (PPO)")
        plt.savefig('rl_portfolio_performance.png', dpi=150, bbox_inches='tight')
        print("  ✓ Performance chart saved: rl_portfolio_performance.png")
    except Exception as e:
        print(f"  ✗ Visualization failed: {e}")
    print()
    
    # Summary
    print("=" * 70)
    print("Demo completed successfully!")
    print()
    print("Key Takeaways:")
    print("  • RL-based portfolio optimization with PPO algorithm")
    print("  • CNN feature extraction for market patterns")
    print("  • Continuous action space with portfolio weight constraints")
    print("  • Automatic rebalancing with transaction cost consideration")
    print(f"  • Achieved {results['sharpe_ratio']:.2f} Sharpe ratio on test data")
    print()
    print("Based on: Wu et al. (2024) Journal of Forecasting research")
    print("=" * 70)


if __name__ == "__main__":
    main()