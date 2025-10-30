"""
Demo: LSTM+CNN Portfolio Predictor with 3 Frameworks

Demonstrates LSTM and CNN return prediction combined with three portfolio
optimization frameworks: MVF, RPP, and MDP.

Based on: MD Nguyen (August 2025) PLoS One
Research showed LSTM outperformed CNN for all portfolios.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt

try:
    from axiom.models.portfolio.lstm_cnn_predictor import (
        LSTMCNNPortfolioPredictor,
        PredictorConfig,
        PortfolioFramework,
        create_sample_market_data
    )
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    IMPORTS_AVAILABLE = False


def main():
    print("=" * 90)
    print("LSTM+CNN Portfolio Predictor Demo")
    print("Three Frameworks: MVF (moderate), RPP (balanced), MDP (conservative)")
    print("=" * 90)
    print()
    
    if not IMPORTS_AVAILABLE:
        print("ERROR: Install torch scipy cvxpy")
        return
    
    # Config
    config = PredictorConfig(
        n_assets=6,
        lookback_window=30,
        lstm_hidden_size=128
    )
    
    # Data
    print("1. Generating market data...")
    X, y = create_sample_market_data(n_timesteps=500, n_assets=6)
    train_size = 400
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    print(f"   Training: {len(X_train)}, Validation: {len(X_val)}")
    
    # Initialize
    predictor = LSTMCNNPortfolioPredictor(config)
    
    # Train
    print("\n2. Training LSTM predictor...")
    predictor.train_lstm(X_train, y_train, X_val, y_val, epochs=100, verbose=0)
    print("   ✓ Complete")
    
    # Compare frameworks
    print("\n3. Portfolio Framework Comparison")
    print("-" * 90)
    
    sample = X_val[0:1]
    hist_rets = y_train.numpy()
    
    for fw in PortfolioFramework:
        res = predictor.optimize_portfolio(sample, hist_rets, framework=fw)
        print(f"\n  {fw.value.upper()}:")
        print(f"    Sharpe Ratio: {res['sharpe_ratio']:.3f}")
        print(f"    Expected Return: {res['expected_return']:.2%}")
        print(f"    Volatility: {res['volatility']:.2%}")
        for i, w in enumerate(res['weights']):
            if w > 0.01:
                print(f"    Asset {i+1}: {w:.1%}")
    
    print("\n" + "=" * 90)
    print(f"LSTM+MVF delivered best risk-adjusted returns (Nguyen 2025)")


if __name__ == "__main__":
    main()