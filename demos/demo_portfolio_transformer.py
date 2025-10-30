"""
Demo: Portfolio Transformer for Attention-Based Asset Allocation

This demo showcases the Portfolio Transformer model that uses transformer
encoder-decoder architecture with specialized time encoding and gating for
end-to-end portfolio optimization.

Based on research from:
Damian Kisiel, Denise Gorse (2023)
"Portfolio Transformer for Attention-Based Asset Allocation"
International Conference on Artificial Intelligence and Soft Computing (ICAISC 2022)
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from axiom.models.portfolio.portfolio_transformer import (
        PortfolioTransformer,
        TransformerConfig,
        create_sample_transformer_data
    )
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    IMPORTS_AVAILABLE = False


def plot_transformer_results(history: dict, results: dict):
    """Plot training and backtest results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training Sharpe over epochs
    ax1 = axes[0, 0]
    ax1.plot(history['train_sharpe'], label='Train Sharpe', linewidth=2)
    if history['val_sharpe']:
        ax1.plot(history['val_sharpe'], label='Val Sharpe', linewidth=2)
    ax1.set_title('Training Progress - Sharpe Ratio', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Sharpe Ratio')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Portfolio value
    ax2 = axes[0, 1]
    ax2.plot(results['portfolio_values'], linewidth=2, color='green')
    ax2.set_title('Portfolio Value Over Time', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Trading Days')
    ax2.set_ylabel('Portfolio Value ($)')
    ax2.grid(True, alpha=0.3)
    
    # Cumulative returns
    ax3 = axes[1, 0]
    returns = np.array(results['returns'])
    cumulative = (1 + returns).cumprod() - 1
    ax3.plot(cumulative * 100, linewidth=2, color='blue')
    ax3.set_title('Cumulative Returns', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Trading Days')
    ax3.set_ylabel('Return (%)')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Portfolio weights (final allocation)
    ax4 = axes[1, 1]
    final_weights = results['weights_history'][-1]
    assets = [f'Asset {i+1}' for i in range(len(final_weights))]
    colors = plt.cm.Set3(np.linspace(0, 1, len(final_weights)))
    ax4.pie(final_weights, labels=assets, autopct='%1.1f%%', colors=colors, startangle=90)
    ax4.set_title('Final Portfolio Allocation', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig


def main():
    """Main demo function"""
    print("=" * 70)
    print("Portfolio Transformer Demo")
    print("End-to-End Attention-Based Asset Allocation")
    print("=" * 70)
    print()
    
    if not IMPORTS_AVAILABLE:
        print("ERROR: Required modules not available. Please install dependencies:")
        print("  pip install torch")
        return
    
    # Configuration
    print("1. Configuration")
    print("-" * 70)
    config = TransformerConfig(
        n_assets=6,
        lookback_window=50,
        d_model=128,
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=2,
        use_time_encoding=True,
        use_gating=True,
        max_position=0.40,
        learning_rate=1e-4
    )
    print(f"  Number of assets: {config.n_assets}")
    print(f"  Lookback window: {config.lookback_window} days")
    print(f"  Model dimension: {config.d_model}")
    print(f"  Attention heads: {config.nhead}")
    print(f"  Encoder layers: {config.num_encoder_layers}")
    print(f"  Decoder layers: {config.num_decoder_layers}")
    print(f"  Time encoding: {'✓' if config.use_time_encoding else '✗'}")
    print(f"  Gating mechanism: {'✓' if config.use_gating else '✗'}")
    print(f"  Max position: {config.max_position:.0%}")
    print()
    
    # Generate data
    print("2. Generating Market Data")
    print("-" * 70)
    print("  Creating synthetic market data with diverse asset characteristics...")
    
    X, returns = create_sample_transformer_data(
        n_samples=400,
        lookback=config.lookback_window,
        n_assets=config.n_assets
    )
    
    # Split data
    train_size = 320
    X_train, X_val = X[:train_size], X[train_size:]
    ret_train, ret_val = returns[:train_size], returns[train_size:]
    
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print(f"  Sequence shape: {X_train.shape}")
    print()
    
    # Initialize transformer
    print("3. Initializing Portfolio Transformer")
    print("-" * 70)
    pt = PortfolioTransformer(config)
    print("  ✓ Encoder-decoder architecture initialized")
    print("  ✓ Multi-head attention configured")
    print("  ✓ Time encoding enabled")
    print("  ✓ Gating mechanism enabled")
    print()
    
    # Train
    print("4. Training Portfolio Transformer")
    print("-" * 70)
    print("  Optimizing for maximum Sharpe ratio...")
    print("  This may take a few minutes...")
    
    try:
        history = pt.train(
            X_train, ret_train,
            X_val, ret_val,
            epochs=50,
            verbose=1
        )
        print("  ✓ Training completed successfully")
    except Exception as e:
        print(f"  ✗ Training failed: {e}")
        return
    print()
    
    # Test allocation
    print("5. Testing Portfolio Allocation")
    print("-" * 70)
    
    # Get allocation for a sample
    sample_weights = pt.allocate(X_val[0])
    print(f"  Sample allocation:")
    for i, weight in enumerate(sample_weights):
        print(f"    Asset {i+1}: {weight:6.2%} {'█' * int(weight * 50)}")
    print(f"  Sum of weights: {sample_weights.sum():.4f}")
    print()
    
    # Backtest
    print("6. Backtesting on Validation Data")
    print("-" * 70)
    print("  Running backtest with transformer allocations...")
    
    # Create test data (need to reconstruct market data for backtest)
    np.random.seed(42)
    n_timesteps = len(X) + config.lookback_window + 10
    prices = np.zeros((n_timesteps, config.n_assets))
    prices[0] = 100
    
    drifts = np.array([0.0008, 0.0006, 0.0005, 0.0003, 0.0002, 0.0002])
    vols = np.array([0.025, 0.020, 0.018, 0.015, 0.010, 0.008])
    
    for t in range(1, n_timesteps):
        rets = np.random.normal(drifts, vols)
        prices[t] = prices[t-1] * (1 + rets)
    
    # Create market data
    market_data = np.zeros((n_timesteps, config.n_assets, 5))
    for i in range(config.n_assets):
        market_data[:, i, 0] = prices[:, i]
        market_data[:, i, 1] = prices[:, i] * (1 + np.random.normal(0, 0.002, n_timesteps))
        market_data[:, i, 2] = prices[:, i] * (1 + np.abs(np.random.normal(0, 0.005, n_timesteps)))
        market_data[:, i, 3] = prices[:, i] * (1 - np.abs(np.random.normal(0, 0.005, n_timesteps)))
        market_data[:, i, 4] = np.random.lognormal(0, 0.5, n_timesteps)
    
    # Normalize
    for i in range(config.n_assets):
        for j in range(5):
            mean = market_data[:, i, j].mean()
            std = market_data[:, i, j].std()
            if std > 0:
                market_data[:, i, j] = (market_data[:, i, j] - mean) / std
    
    # Calculate returns
    returns_data = np.zeros((n_timesteps, config.n_assets))
    for i in range(config.n_assets):
        returns_data[1:, i] = (prices[1:, i] - prices[:-1, i]) / prices[:-1, i]
    
    # Use validation portion
    test_start = train_size + config.lookback_window
    test_market = market_data[test_start:]
    test_returns = returns_data[test_start:]
    
    try:
        results = pt.backtest(
            test_market,
            test_returns,
            rebalance_frequency=5,
            initial_capital=10000.0,
            transaction_cost=0.001
        )
        print("  ✓ Backtest completed")
    except Exception as e:
        print(f"  ✗ Backtest failed: {e}")
        return
    print()
    
    # Performance metrics
    print("7. Performance Metrics")
    print("-" * 70)
    print(f"  Initial Capital:    ${10000.00:,.2f}")
    print(f"  Final Value:        ${results['final_value']:,.2f}")
    print(f"  Total Return:       {results['total_return']:.2%}")
    print(f"  Sharpe Ratio:       {results['sharpe_ratio']:.3f}")
    print(f"  Max Drawdown:       {results['max_drawdown']:.2%}")
    print()
    
    # Training metrics
    print("8. Training Metrics")
    print("-" * 70)
    print(f"  Final Train Sharpe: {history['train_sharpe'][-1]:.3f}")
    if history['val_sharpe']:
        print(f"  Final Val Sharpe:   {history['val_sharpe'][-1]:.3f}")
    print(f"  Training epochs:    {len(history['train_sharpe'])}")
    print()
    
    # Model architecture details
    print("9. Model Architecture")
    print("-" * 70)
    print(f"  Total parameters:   ~{sum(p.numel() for p in pt.model.parameters()):,}")
    print(f"  Encoder depth:      {config.num_encoder_layers} layers")
    print(f"  Decoder depth:      {config.num_decoder_layers} layers")
    print(f"  Attention heads:    {config.nhead}")
    print(f"  FFN dimension:      {config.dim_feedforward}")
    print()
    
    # Visualization
    print("10. Generating Visualizations")
    print("-" * 70)
    try:
        fig = plot_transformer_results(history, results)
        plt.savefig('portfolio_transformer_performance.png', dpi=150, bbox_inches='tight')
        print("  ✓ Performance charts saved: portfolio_transformer_performance.png")
    except Exception as e:
        print(f"  ✗ Visualization failed: {e}")
    print()
    
    # Summary
    print("=" * 70)
    print("Demo completed successfully!")
    print()
    print("Key Features:")
    print("  • End-to-end learning from prices to portfolio weights")
    print("  • Multi-head attention for capturing asset relationships")
    print("  • Specialized time encoding for financial time series")
    print("  • Gating mechanism for controlled information flow")
    print("  • Direct Sharpe ratio optimization")
    print("  • Automatic position limit enforcement")
    print()
    print(f"Achieved Performance:")
    print(f"  • Sharpe Ratio: {results['sharpe_ratio']:.3f}")
    print(f"  • Total Return: {results['total_return']:.2%}")
    print(f"  • Max Drawdown: {results['max_drawdown']:.2%}")
    print()
    print("Based on: Kisiel & Gorse (2023) ICAISC 2022")
    print("Outperforms LSTM-based SOTA on 3 datasets")
    print("=" * 70)


if __name__ == "__main__":
    main()