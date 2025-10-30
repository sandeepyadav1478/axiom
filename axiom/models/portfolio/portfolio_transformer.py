"""
Portfolio Transformer for End-to-End Asset Allocation

Based on: Damian Kisiel, Denise Gorse (2023)
"Portfolio Transformer for Attention-Based Asset Allocation"
International Conference on Artificial Intelligence and Soft Computing (ICAISC 2022)
Springer LNCS 13588, Published: January 24, 2023

This implementation uses transformer encoder-decoder architecture with specialized
time encoding and gating components to directly optimize portfolio weights for
maximum Sharpe ratio, bypassing traditional forecasting steps.

Research showed it outperforms LSTM-based state-of-the-art on 3 datasets and
handles market regime changes (COVID-19).
"""

from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import math
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class TransformerConfig:
    """Configuration for Portfolio Transformer"""
    # Input parameters
    n_assets: int = 6
    lookback_window: int = 50
    n_features_per_asset: int = 5  # OHLCV or engineered features
    
    # Transformer parameters
    d_model: int = 128  # Model dimension
    nhead: int = 8      # Number of attention heads
    num_encoder_layers: int = 3
    num_decoder_layers: int = 2
    dim_feedforward: int = 512
    dropout: float = 0.1
    
    # Time encoding
    use_time_encoding: bool = True
    time_encoding_dim: int = 32
    
    # Gating
    use_gating: bool = True
    
    # Training
    learning_rate: float = 1e-4
    batch_size: int = 16
    weight_decay: float = 1e-5
    
    # Portfolio constraints
    allow_short: bool = False
    max_position: float = 0.40
    min_position: float = 0.0


class SpecializedTimeEncoding(nn.Module):
    """
    Specialized time encoding for financial time series
    
    Captures both periodic patterns (weekly, monthly) and trend information.
    """
    
    def __init__(self, d_model: int, time_encoding_dim: int = 32):
        super(SpecializedTimeEncoding, self).__init__()
        
        self.d_model = d_model
        self.time_encoding_dim = time_encoding_dim
        
        # Learnable time embeddings
        self.time_embedding = nn.Embedding(366, time_encoding_dim)  # Day of year
        
        # Projection to model dimension
        self.projection = nn.Linear(time_encoding_dim, d_model)
        
    def forward(self, x: torch.Tensor, time_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Add time encoding to input
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            time_indices: Optional time indices (batch, seq_len)
            
        Returns:
            Time-encoded tensor
        """
        batch_size, seq_len, _ = x.size()
        
        if time_indices is None:
            # Default: sequential indices
            time_indices = torch.arange(seq_len, device=x.device).unsqueeze(0).repeat(batch_size, 1)
        
        # Get time embeddings
        time_embeds = self.time_embedding(time_indices % 366)  # Modulo for day of year
        
        # Project to model dimension
        time_encoded = self.projection(time_embeds)
        
        # Add to input
        return x + time_encoded


class GatingComponent(nn.Module):
    """
    Gating mechanism for controlling information flow
    
    Helps model focus on relevant time periods and assets.
    """
    
    def __init__(self, d_model: int):
        super(GatingComponent, self).__init__()
        
        self.gate_fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply gating to input
        
        Args:
            x: Input tensor
            
        Returns:
            Gated tensor
        """
        gate = self.gate_fc(x)
        return x * gate


class PortfolioTransformerModel(nn.Module):
    """
    Complete Portfolio Transformer architecture
    
    End-to-end learning from price history to optimal portfolio weights.
    Directly optimizes Sharpe ratio without separate forecasting step.
    """
    
    def __init__(self, config: TransformerConfig):
        super(PortfolioTransformerModel, self).__init__()
        
        self.config = config
        
        # Input projection
        input_dim = config.n_assets * config.n_features_per_asset
        self.input_projection = nn.Linear(input_dim, config.d_model)
        
        # Time encoding
        if config.use_time_encoding:
            self.time_encoding = SpecializedTimeEncoding(
                config.d_model,
                config.time_encoding_dim
            )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_encoder_layers
        )
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.num_decoder_layers
        )
        
        # Gating component
        if config.use_gating:
            self.gating = GatingComponent(config.d_model)
        
        # Output head - portfolio weights
        self.weight_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, config.n_assets),
            nn.Softmax(dim=-1) if not config.allow_short else nn.Tanh()
        )
        
    def forward(self, x: torch.Tensor, time_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass: market data → portfolio weights
        
        Args:
            x: Market data (batch, seq_len, n_assets * n_features)
            time_indices: Optional time indices
            
        Returns:
            Portfolio weights (batch, n_assets)
        """
        # Project input to model dimension
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        
        # Add time encoding
        if self.config.use_time_encoding:
            x = self.time_encoding(x, time_indices)
        
        # Encoder
        memory = self.transformer_encoder(x)  # (batch, seq_len, d_model)
        
        # Decoder (use last encoder state as query)
        tgt = memory[:, -1:, :]  # (batch, 1, d_model)
        decoder_output = self.transformer_decoder(tgt, memory)  # (batch, 1, d_model)
        
        # Gating
        if self.config.use_gating:
            decoder_output = self.gating(decoder_output)
        
        # Generate portfolio weights
        weights = self.weight_head(decoder_output.squeeze(1))  # (batch, n_assets)
        
        # Normalize to sum to 1 if using softmax alternative
        if self.config.allow_short:
            # Tanh output: normalize to sum to 1
            weights = F.softmax(weights, dim=-1)
        
        # Apply position limits
        weights = torch.clamp(weights, self.config.min_position, self.config.max_position)
        weights = weights / weights.sum(dim=-1, keepdim=True)  # Renormalize
        
        return weights


class PortfolioTransformer:
    """
    Complete Portfolio Transformer system
    
    Main interface for training and using the transformer for portfolio allocation.
    """
    
    def __init__(self, config: Optional[TransformerConfig] = None):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for PortfolioTransformer")
            
        self.config = config or TransformerConfig()
        self.model = PortfolioTransformerModel(self.config)
        self.optimizer = None
        
        # History
        self.history = {'train_loss': [], 'val_loss': [], 'train_sharpe': [], 'val_sharpe': []}
        
    def sharpe_ratio_loss(
        self,
        weights: torch.Tensor,
        returns: torch.Tensor,
        risk_free_rate: float = 0.02
    ) -> torch.Tensor:
        """
        Calculate negative Sharpe ratio as loss
        
        Args:
            weights: Portfolio weights (batch, n_assets)
            returns: Asset returns (batch, n_assets)
            risk_free_rate: Risk-free rate (annual)
            
        Returns:
            Negative Sharpe ratio (to minimize)
        """
        # Portfolio returns
        portfolio_returns = (weights * returns).sum(dim=-1)  # (batch,)
        
        # Mean and std
        mean_return = portfolio_returns.mean()
        std_return = portfolio_returns.std() + 1e-8
        
        # Sharpe ratio (annualized)
        sharpe = (mean_return * 252 - risk_free_rate) / (std_return * math.sqrt(252))
        
        # Return negative (we minimize)
        return -sharpe
        
    def train(
        self,
        X_train: torch.Tensor,
        returns_train: torch.Tensor,
        X_val: Optional[torch.Tensor] = None,
        returns_val: Optional[torch.Tensor] = None,
        epochs: int = 100,
        verbose: int = 1
    ) -> Dict[str, List[float]]:
        """
        Train Portfolio Transformer
        
        Args:
            X_train: Training market data (n_samples, seq_len, n_assets * n_features)
            returns_train: Actual returns for each sample (n_samples, n_assets)
            X_val: Validation data
            returns_val: Validation returns
            epochs: Training epochs
            verbose: Verbosity
            
        Returns:
            Training history
        """
        self.model.train()
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        n_samples = X_train.size(0)
        batch_size = self.config.batch_size
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            # Mini-batch training
            indices = torch.randperm(n_samples)
            
            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i:i + batch_size]
                batch_X = X_train[batch_indices]
                batch_returns = returns_train[batch_indices]
                
                # Forward
                weights = self.model(batch_X)
                
                # Loss: negative Sharpe ratio
                loss = self.sharpe_ratio_loss(
                    weights,
                    batch_returns,
                    risk_free_rate=self.config.risk_free_rate
                )
                
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            # Average loss
            avg_loss = epoch_loss / n_batches
            self.history['train_loss'].append(avg_loss)
            self.history['train_sharpe'].append(-avg_loss)  # Negative of negative = positive
            
            # Validation
            if X_val is not None and returns_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_weights = self.model(X_val)
                    val_loss = self.sharpe_ratio_loss(
                        val_weights,
                        returns_val,
                        risk_free_rate=self.config.risk_free_rate
                    )
                    self.history['val_loss'].append(val_loss.item())
                    self.history['val_sharpe'].append(-val_loss.item())
                self.model.train()
                
                if verbose > 0 and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Train Sharpe: {-avg_loss:.4f}, "
                          f"Val Sharpe: {-val_loss.item():.4f}")
            else:
                if verbose > 0 and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Train Sharpe: {-avg_loss:.4f}")
        
        return self.history
        
    def allocate(
        self,
        market_data: Union[torch.Tensor, np.ndarray],
        time_indices: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """
        Get optimal portfolio allocation
        
        Args:
            market_data: Current market data (seq_len, n_assets * n_features)
                        or (batch, seq_len, n_assets * n_features)
            time_indices: Optional time indices
            
        Returns:
            Portfolio weights (n_assets,) or (batch, n_assets)
        """
        self.model.eval()
        
        # Convert to tensor
        if isinstance(market_data, np.ndarray):
            market_data = torch.FloatTensor(market_data)
        
        # Add batch dimension if needed
        squeeze_output = False
        if len(market_data.shape) == 2:
            market_data = market_data.unsqueeze(0)
            squeeze_output = True
        
        with torch.no_grad():
            weights = self.model(market_data, time_indices)
        
        # Convert to numpy
        weights = weights.cpu().numpy()
        
        if squeeze_output:
            weights = weights.squeeze(0)
        
        return weights
        
    def backtest(
        self,
        market_data: np.ndarray,
        returns_data: np.ndarray,
        rebalance_frequency: int = 5,  # Days
        initial_capital: float = 100000.0,
        transaction_cost: float = 0.001  # 0.1%
    ) -> Dict[str, Union[List, float]]:
        """
        Backtest portfolio transformer
        
        Args:
            market_data: Historical data (timesteps, n_assets, n_features)
            returns_data: Historical returns (timesteps, n_assets)
            rebalance_frequency: Days between rebalancing
            initial_capital: Starting capital
            transaction_cost: Transaction cost rate
            
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
            window = market_data[t - self.config.lookback_window:t]
            # Flatten for transformer
            window_flat = window.reshape(self.config.lookback_window, -1)
            
            # Get optimal allocation
            new_weights = self.allocate(window_flat)
            
            # Calculate transaction costs
            weight_change = np.abs(new_weights - current_weights).sum()
            tc_cost = weight_change * transaction_cost
            
            # Calculate returns over next period
            period_end = min(t + rebalance_frequency, n_timesteps)
            if period_end > t:
                period_returns = returns_data[t:period_end]
                # Use new weights for this period
                for period_ret in period_returns:
                    portfolio_ret = np.dot(new_weights, period_ret) - tc_cost
                    returns_history.append(portfolio_ret)
                    portfolio_values.append(portfolio_values[-1] * (1 + portfolio_ret))
                
                tc_cost = 0  # Only charge once per rebalance
            
            current_weights = new_weights
            weights_history.append(current_weights.copy())
        
        # Calculate metrics
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
        
    def save(self, path: str):
        """Save model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'history': self.history
        }, path)
        
    def load(self, path: str):
        """Load model"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint.get('history', {'train_loss': [], 'val_loss': [], 'train_sharpe': [], 'val_sharpe': []})


def create_sample_transformer_data(
    n_samples: int = 500,
    lookback: int = 50,
    n_assets: int = 6,
    n_features: int = 5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create sample data for Portfolio Transformer
    
    Returns:
        (X, returns) where X is market data and returns is next-period returns
    """
    np.random.seed(42)
    
    # Generate price series
    total_timesteps = n_samples + lookback + 10
    prices = np.zeros((total_timesteps, n_assets))
    prices[0] = 100
    
    # Asset characteristics
    drifts = np.array([0.0008, 0.0006, 0.0005, 0.0003, 0.0002, 0.0002])
    vols = np.array([0.025, 0.020, 0.018, 0.015, 0.010, 0.008])
    
    for t in range(1, total_timesteps):
        rets = np.random.normal(drifts, vols)
        prices[t] = prices[t-1] * (1 + rets)
    
    # Create features
    market_data = np.zeros((total_timesteps, n_assets, n_features))
    
    for i in range(n_assets):
        market_data[:, i, 0] = prices[:, i]  # Close
        market_data[:, i, 1] = prices[:, i] * (1 + np.random.normal(0, 0.002, total_timesteps))  # Open
        market_data[:, i, 2] = prices[:, i] * (1 + np.abs(np.random.normal(0, 0.005, total_timesteps)))  # High
        market_data[:, i, 3] = prices[:, i] * (1 - np.abs(np.random.normal(0, 0.005, total_timesteps)))  # Low
        market_data[:, i, 4] = np.random.lognormal(0, 0.5, total_timesteps)  # Volume
    
    # Normalize
    for i in range(n_assets):
        for j in range(n_features):
            mean = market_data[:, i, j].mean()
            std = market_data[:, i, j].std()
            if std > 0:
                market_data[:, i, j] = (market_data[:, i, j] - mean) / std
    
    # Create sequences
    X_sequences = []
    returns_sequences = []
    
    for t in range(lookback, total_timesteps - 1):
        # Historical window
        window = market_data[t-lookback:t]
        window_flat = window.reshape(lookback, -1)
        X_sequences.append(window_flat)
        
        # Next period returns
        next_rets = (prices[t+1] - prices[t]) / prices[t]
        returns_sequences.append(next_rets)
    
    X = torch.FloatTensor(np.array(X_sequences))
    returns = torch.FloatTensor(np.array(returns_sequences))
    
    return X, returns


# Example usage
if __name__ == "__main__":
    print("Portfolio Transformer - Example Usage")
    print("=" * 60)
    
    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch required")
    else:
        # Config
        config = TransformerConfig(
            n_assets=6,
            lookback_window=50,
            d_model=128,
            nhead=8
        )
        
        # Data
        print("\n1. Generating data...")
        X, returns = create_sample_transformer_data(n_samples=400, lookback=50)
        
        train_size = 320
        X_train, X_val = X[:train_size], X[train_size:]
        ret_train, ret_val = returns[:train_size], returns[train_size:]
        
        print(f"   Train: {len(X_train)}, Val: {len(X_val)}")
        
        # Initialize
        print("\n2. Initializing transformer...")
        pt = PortfolioTransformer(config)
        print("   ✓ Encoder-decoder architecture ready")
        
        # Train
        print("\n3. Training...")
        hist = pt.train(X_train, ret_train, X_val, ret_val, epochs=50, verbose=1)
        print("   ✓ Complete")
        
        # Test
        print("\n4. Testing allocation...")
        sample_weights = pt.allocate(X_val[0])
        print(f"   Weights: {sample_weights}")
        print(f"   Sum: {sample_weights.sum():.4f}")
        
        print("\n" + "=" * 60)
        print("Based on: Kisiel & Gorse (2023)")