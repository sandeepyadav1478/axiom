"""
LSTM+CNN Portfolio Return Predictor with Three Optimization Frameworks

Based on: Minh Duc Nguyen (MD Nguyen) (August 2025)
"Advanced investing with deep learning for risk-aligned portfolio optimization"
PLoS One 20(8): e0330547
DOI: https://doi.org/10.1371/journal.pone.0330547

This implementation combines LSTM and 1D-CNN for return prediction, then applies
three portfolio optimization frameworks: Mean-Variance with Forecasting (MVF),
Risk Parity Portfolio (RPP), and Maximum Drawdown Portfolio (MDP).

Research showed LSTM outperformed CNN in all portfolios for both accuracy and stability.
"""

from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from scipy.optimize import minimize
    import cvxpy as cp
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class PortfolioFramework(Enum):
    """Portfolio optimization frameworks"""
    MVF = "mean_variance_forecasting"  # Return-seeking, moderate-risk
    RPP = "risk_parity_portfolio"      # Balanced risk allocation
    MDP = "maximum_drawdown_portfolio"  # Conservative, risk-averse


@dataclass
class PredictorConfig:
    """Configuration for LSTM+CNN Portfolio Predictor"""
    # Input parameters
    n_assets: int = 6
    lookback_window: int = 30
    n_features_per_asset: int = 5  # OHLCV
    
    # LSTM parameters
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.2
    
    # CNN parameters
    cnn_filters: List[int] = None
    cnn_kernel_sizes: List[int] = None
    
    # Training parameters
    learning_rate: float = 1e-3
    batch_size: int = 32
    
    # Portfolio optimization
    risk_free_rate: float = 0.02  # 2% annual
    target_return: Optional[float] = None
    max_weight: float = 0.30  # Max 30% in any single asset
    min_weight: float = 0.0   # No short selling
    
    def __post_init__(self):
        if self.cnn_filters is None:
            self.cnn_filters = [32, 64]
        if self.cnn_kernel_sizes is None:
            self.cnn_kernel_sizes = [3, 3]


import torch.nn as nn

class LSTMReturnPredictor(nn.Module):
    """
    LSTM for time series return forecasting
    
    Research showed LSTM outperformed CNN for all portfolio types.
    """
    
    def __init__(self, config: PredictorConfig):
        super(LSTMReturnPredictor, self).__init__()
        
        self.config = config
        input_size = config.n_assets * config.n_features_per_asset
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_num_layers,
            batch_first=True,
            dropout=config.lstm_dropout if config.lstm_num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Prediction head
        self.fc = nn.Sequential(
            nn.Linear(config.lstm_hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, config.n_assets)  # Predict returns for each asset
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict returns
        
        Args:
            x: Input (batch, sequence_length, n_assets * n_features)
            
        Returns:
            Predicted returns (batch, n_assets)
        """
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use final hidden state
        final_hidden = lstm_out[:, -1, :]
        
        # Predict returns
        returns = self.fc(final_hidden)
        
        return returns


class CNN1DReturnPredictor(nn.Module):
    """
    1D CNN for pattern-based return forecasting
    
    Used for ensemble with LSTM.
    """
    
    def __init__(self, config: PredictorConfig):
        super(CNN1DReturnPredictor, self).__init__()
        
        self.config = config
        
        # CNN layers
        conv_layers = []
        in_channels = config.n_assets * config.n_features_per_asset
        
        for filters, kernel_size in zip(config.cnn_filters, config.cnn_kernel_sizes):
            conv_layers.extend([
                nn.Conv1d(in_channels, filters, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(filters),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(0.2)
            ])
            in_channels = filters
        
        self.cnn = nn.Sequential(*conv_layers)
        
        # Calculate flatten size
        self._calculate_flatten_size()
        
        # Prediction head
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, config.n_assets)
        )
        
    def _calculate_flatten_size(self):
        """Calculate size after convolutions"""
        dummy = torch.zeros(1, self.config.n_assets * self.config.n_features_per_asset, self.config.lookback_window)
        with torch.no_grad():
            out = self.cnn(dummy)
        self.flatten_size = out.numel()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict returns
        
        Args:
            x: Input (batch, lookback_window, n_assets * n_features)
            
        Returns:
            Predicted returns (batch, n_assets)
        """
        # Transpose for Conv1d: (batch, features, sequence)
        x = x.transpose(1, 2)
        
        # CNN
        cnn_out = self.cnn(x)
        
        # Flatten
        flattened = cnn_out.view(cnn_out.size(0), -1)
        
        # Predict
        returns = self.fc(flattened)
        
        return returns


class PortfolioOptimizer:
    """
    Three portfolio optimization frameworks from Nguyen (2025)
    """
    
    @staticmethod
    def mean_variance_forecasting(
        predicted_returns: np.ndarray,
        cov_matrix: np.ndarray,
        risk_free_rate: float = 0.02,
        max_weight: float = 0.30,
        min_weight: float = 0.0
    ) -> np.ndarray:
        """
        Mean-Variance with Forecasting (MVF)
        
        Return-seeking, moderate-risk portfolio.
        Maximizes Sharpe ratio using predicted returns.
        """
        n_assets = len(predicted_returns)
        
        # Define optimization problem
        weights = cp.Variable(n_assets)
        
        # Portfolio return and variance
        portfolio_return = predicted_returns @ weights
        portfolio_variance = cp.quad_form(weights, cov_matrix)
        
        # Sharpe ratio (maximize)
        sharpe = (portfolio_return - risk_free_rate) / cp.sqrt(portfolio_variance)
        
        # Objective: maximize Sharpe
        objective = cp.Maximize(sharpe)
        
        # Constraints
        constraints = [
            cp.sum(weights) == 1,  # Fully invested
            weights >= min_weight,  # No short selling
            weights <= max_weight   # Position limits
        ]
        
        # Solve
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        return weights.value if weights.value is not None else np.ones(n_assets) / n_assets
        
    @staticmethod
    def risk_parity_portfolio(
        cov_matrix: np.ndarray,
        max_weight: float = 0.30,
        min_weight: float = 0.0
    ) -> np.ndarray:
        """
        Risk Parity Portfolio (RPP)
        
        Balanced risk allocation across assets.
        Each asset contributes equally to portfolio risk.
        """
        n_assets = cov_matrix.shape[0]
        
        def risk_parity_objective(weights):
            """Minimize sum of squared differences in risk contributions"""
            portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
            marginal_contrib = cov_matrix @ weights
            risk_contrib = weights * marginal_contrib / portfolio_vol
            
            # Target: equal risk contribution (1/n each)
            target_contrib = portfolio_vol / n_assets
            
            return np.sum((risk_contrib - target_contrib) ** 2)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Sum to 1
        ]
        
        bounds = [(min_weight, max_weight) for _ in range(n_assets)]
        
        # Initial guess: equal weight
        w0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(
            risk_parity_objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x if result.success else w0
        
    @staticmethod
    def maximum_drawdown_portfolio(
        predicted_returns: np.ndarray,
        historical_returns: np.ndarray,
        max_weight: float = 0.30,
        min_weight: float = 0.0
    ) -> np.ndarray:
        """
        Maximum Drawdown Portfolio (MDP)
        
        Conservative, risk-averse portfolio.
        Minimizes maximum historical drawdown.
        """
        n_assets = len(predicted_returns)
        n_periods = historical_returns.shape[0]
        
        # Define optimization problem
        weights = cp.Variable(n_assets)
        
        # Portfolio returns over time
        portfolio_returns = historical_returns @ weights
        
        # Cumulative returns
        cumulative_returns = cp.cumsum(portfolio_returns)
        
        # Running maximum
        # Approximate max drawdown using CVaR of drawdowns
        drawdowns = []
        for t in range(1, n_periods):
            running_max = cp.max(cumulative_returns[:t])
            drawdown = running_max - cumulative_returns[t]
            drawdowns.append(drawdown)
        
        # Minimize maximum drawdown (approximate)
        max_dd = cp.max(cp.vstack(drawdowns))
        
        # Objective
        objective = cp.Minimize(max_dd)
        
        # Constraints
        constraints = [
            cp.sum(weights) == 1,
            weights >= min_weight,
            weights <= max_weight
        ]
        
        # Solve
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve()
            return weights.value if weights.value is not None else np.ones(n_assets) / n_assets
        except:
            # Fallback: minimum variance
            cov = np.cov(historical_returns.T)
            return PortfolioOptimizer.risk_parity_portfolio(cov, max_weight, min_weight)


class LSTMCNNPortfolioPredictor:
    """
    Complete LSTM+CNN Portfolio Prediction and Optimization System
    
    Combines LSTM and CNN forecasters with three portfolio frameworks.
    """
    
    def __init__(self, config: Optional[PredictorConfig] = None):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy and cvxpy required")
            
        self.config = config or PredictorConfig()
        
        # Initialize predictors
        self.lstm_predictor = LSTMReturnPredictor(self.config)
        self.cnn_predictor = CNN1DReturnPredictor(self.config)
        
        # Optimizer
        self.optimizer = None
        
        # History
        self.history = {'train_loss': [], 'val_loss': []}
        
    def train_lstm(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
        epochs: int = 100,
        verbose: int = 1
    ):
        """Train LSTM predictor"""
        self.lstm_predictor.train()
        self.optimizer = torch.optim.Adam(self.lstm_predictor.parameters(), lr=self.config.learning_rate)
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            # Training
            self.optimizer.zero_grad()
            pred = self.lstm_predictor(X_train)
            loss = criterion(pred, y_train)
            loss.backward()
            self.optimizer.step()
            
            self.history['train_loss'].append(loss.item())
            
            # Validation
            if X_val is not None:
                with torch.no_grad():
                    val_pred = self.lstm_predictor(X_val)
                    val_loss = criterion(val_pred, y_val).item()
                    self.history['val_loss'].append(val_loss)
            
            if verbose > 0 and (epoch + 1) % 20 == 0:
                if X_val is not None:
                    print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.6f}, Val Loss: {val_loss:.6f}")
                else:
                    print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.6f}")
        
        return self.history
        
    def predict_returns(
        self,
        X: torch.Tensor,
        use_lstm: bool = True
    ) -> np.ndarray:
        """
        Predict asset returns
        
        Args:
            X: Market data
            use_lstm: Use LSTM (True) or CNN (False)
            
        Returns:
            Predicted returns (n_assets,)
        """
        if use_lstm:
            self.lstm_predictor.eval()
            with torch.no_grad():
                pred = self.lstm_predictor(X)
        else:
            self.cnn_predictor.eval()
            with torch.no_grad():
                pred = self.cnn_predictor(X)
        
        return pred.squeeze().numpy()
        
    def optimize_portfolio(
        self,
        current_data: torch.Tensor,
        historical_returns: np.ndarray,
        framework: PortfolioFramework = PortfolioFramework.MVF
    ) -> Dict[str, Union[np.ndarray, float]]:
        """
        Predict returns and optimize portfolio
        
        Args:
            current_data: Current market data for prediction
            historical_returns: Historical returns for covariance
            framework: Optimization framework to use
            
        Returns:
            Dictionary with weights, expected return, risk
        """
        # Predict returns using LSTM (best performer)
        predicted_returns = self.predict_returns(current_data, use_lstm=True)
        
        # Calculate covariance matrix
        cov_matrix = np.cov(historical_returns.T)
        
        # Optimize based on framework
        if framework == PortfolioFramework.MVF:
            weights = PortfolioOptimizer.mean_variance_forecasting(
                predicted_returns,
                cov_matrix,
                risk_free_rate=self.config.risk_free_rate,
                max_weight=self.config.max_weight,
                min_weight=self.config.min_weight
            )
        elif framework == PortfolioFramework.RPP:
            weights = PortfolioOptimizer.risk_parity_portfolio(
                cov_matrix,
                max_weight=self.config.max_weight,
                min_weight=self.config.min_weight
            )
        elif framework == PortfolioFramework.MDP:
            weights = PortfolioOptimizer.maximum_drawdown_portfolio(
                predicted_returns,
                historical_returns,
                max_weight=self.config.max_weight,
                min_weight=self.config.min_weight
            )
        else:
            raise ValueError(f"Unknown framework: {framework}")
        
        # Normalize weights
        weights = weights / np.sum(weights)
        
        # Calculate portfolio metrics
        expected_return = np.dot(weights, predicted_returns)
        portfolio_variance = weights @ cov_matrix @ weights
        portfolio_vol = np.sqrt(portfolio_variance)
        sharpe_ratio = (expected_return - self.config.risk_free_rate) / portfolio_vol
        
        return {
            'weights': weights,
            'expected_return': expected_return,
            'volatility': portfolio_vol,
            'sharpe_ratio': sharpe_ratio,
            'framework': framework.value
        }
        
    def backtest(
        self,
        market_data: np.ndarray,
        rebalance_frequency: int = 20,  # Days
        framework: PortfolioFramework = PortfolioFramework.MVF,
        initial_capital: float = 100000.0
    ) -> Dict[str, Union[List, float]]:
        """
        Backtest portfolio strategy
        
        Args:
            market_data: Historical market data (timesteps, n_assets, n_features)
            rebalance_frequency: Days between rebalancing
            framework: Portfolio framework
            initial_capital: Starting capital
            
        Returns:
            Backtest results
        """
        portfolio_values = [initial_capital]
        weights_history = []
        returns_history = []
        
        n_timesteps = len(market_data)
        current_weights = np.ones(self.config.n_assets) / self.config.n_assets
        
        for t in range(self.config.lookback_window, n_timesteps, rebalance_frequency):
            # Get historical window
            window_data = market_data[t - self.config.lookback_window:t]
            
            # Prepare for LSTM
            window_tensor = torch.FloatTensor(window_data).unsqueeze(0)
            # Reshape: (batch=1, sequence, features)
            window_tensor = window_tensor.view(1, self.config.lookback_window, -1)
            
            # Get historical returns for covariance
            hist_returns = market_data[max(0, t-100):t, :, 0]  # Use close prices
            hist_returns = np.diff(hist_returns, axis=0) / hist_returns[:-1] if len(hist_returns) > 1 else np.zeros((1, self.config.n_assets))
            
            # Optimize portfolio
            result = self.optimize_portfolio(
                window_tensor,
                hist_returns,
                framework=framework
            )
            
            current_weights = result['weights']
            weights_history.append(current_weights.copy())
            
            # Calculate returns over next period
            if t + rebalance_frequency < n_timesteps:
                period_returns = market_data[t:t+rebalance_frequency, :, 0]
                if len(period_returns) > 1:
                    asset_returns = (period_returns[-1] - period_returns[0]) / period_returns[0]
                    portfolio_return = np.dot(current_weights, asset_returns)
                    returns_history.append(portfolio_return)
                    
                    # Update portfolio value
                    portfolio_values.append(portfolio_values[-1] * (1 + portfolio_return))
        
        # Calculate performance metrics
        returns = np.array(returns_history)
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        max_dd = self._calculate_max_drawdown(portfolio_values)
        
        return {
            'portfolio_values': portfolio_values,
            'returns': returns_history,
            'weights_history': weights_history,
            'sharpe_ratio': sharpe,
            'total_return': (portfolio_values[-1] - initial_capital) / initial_capital,
            'max_drawdown': max_dd,
            'final_value': portfolio_values[-1]
        }
        
    @staticmethod
    def _calculate_max_drawdown(values: List[float]) -> float:
        """Calculate maximum drawdown"""
        values = np.array(values)
        running_max = np.maximum.accumulate(values)
        drawdown = (values - running_max) / running_max
        return float(np.min(drawdown))


def create_sample_market_data(
    n_timesteps: int = 500,
    n_assets: int = 6,
    n_features: int = 5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create sample market data (OHLCV)
    
    Returns:
        (features, returns) for training
    """
    np.random.seed(42)
    
    # Simulate prices with different characteristics
    prices = np.zeros((n_timesteps, n_assets))
    prices[0] = 100
    
    # Different drift and volatility for each asset
    drifts = np.array([0.0008, 0.0006, 0.0004, 0.0003, 0.0002, 0.0002])
    vols = np.array([0.02, 0.018, 0.015, 0.013, 0.008, 0.006])
    
    for t in range(1, n_timesteps):
        returns = np.random.normal(drifts, vols)
        prices[t] = prices[t-1] * (1 + returns)
    
    # Create OHLCV features
    market_data = np.zeros((n_timesteps, n_assets, n_features))
    
    for i in range(n_assets):
        # Close
        market_data[:, i, 0] = prices[:, i]
        # Open (slight variation)
        market_data[:, i, 1] = prices[:, i] * (1 + np.random.normal(0, 0.001, n_timesteps))
        # High
        market_data[:, i, 2] = prices[:, i] * (1 + np.abs(np.random.normal(0, 0.005, n_timesteps)))
        # Low
        market_data[:, i, 3] = prices[:, i] * (1 - np.abs(np.random.normal(0, 0.005, n_timesteps)))
        # Volume (normalized)
        market_data[:, i, 4] = np.random.lognormal(0, 0.5, n_timesteps)
    
    # Create training sequences
    sequences = []
    targets = []
    
    lookback = 30
    for t in range(lookback, n_timesteps - 1):
        seq = market_data[t-lookback:t]
        # Flatten for LSTM input
        seq_flat = seq.reshape(lookback, -1)
        sequences.append(seq_flat)
        
        # Target: next period returns
        next_returns = (market_data[t+1, :, 0] - market_data[t, :, 0]) / market_data[t, :, 0]
        targets.append(next_returns)
    
    X = torch.FloatTensor(np.array(sequences))
    y = torch.FloatTensor(np.array(targets))
    
    return X, y


# Example usage
if __name__ == "__main__":
    print("LSTM+CNN Portfolio Predictor - Example Usage")
    print("=" * 60)
    
    if not all([TORCH_AVAILABLE, SCIPY_AVAILABLE]):
        print("ERROR: Missing dependencies")
        print(f"  PyTorch: {'✓' if TORCH_AVAILABLE else '✗'}")
        print(f"  scipy/cvxpy: {'✓' if SCIPY_AVAILABLE else '✗'}")
    else:
        # Create data
        print("\n1. Generating market data...")
        X, y = create_sample_market_data()
        
        # Split
        train_size = 400
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        print(f"   Training: {len(X_train)}, Validation: {len(X_val)}")
        
        # Initialize
        print("\n2. Initializing LSTM+CNN predictor...")
        predictor = LSTMCNNPortfolioPredictor()
        print("   ✓ LSTM predictor initialized")
        print("   ✓ CNN predictor initialized")
        
        # Train LSTM
        print("\n3. Training LSTM predictor...")
        predictor.train_lstm(X_train, y_train, X_val, y_val, epochs=50, verbose=1)
        print("   ✓ Training complete")
        
        # Test three frameworks
        print("\n4. Testing portfolio frameworks...")
        sample_data = X_val[0:1]
        hist_returns = y_train.numpy()
        
        for framework in PortfolioFramework:
            result = predictor.optimize_portfolio(
                sample_data,
                hist_returns,
                framework=framework
            )
            print(f"\n   {framework.value.upper()}:")
            print(f"     Expected Return: {result['expected_return']:.2%}")
            print(f"     Volatility: {result['volatility']:.2%}")
            print(f"     Sharpe Ratio: {result['sharpe_ratio']:.3f}")
            print(f"     Weights: {result['weights']}")
        
        print("\n" + "=" * 60)
        print("Demo completed!")