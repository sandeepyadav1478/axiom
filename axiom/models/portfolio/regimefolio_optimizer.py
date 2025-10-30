"""
RegimeFolio: Regime-Aware ML Portfolio Optimization

Based on: Y. Zhang, D. Goel, H. Ahmad, C. Szabo (2025)
"RegimeFolio: A Regime Aware ML System for Sectoral Portfolio Optimization in Dynamic Markets"
IEEE Access, 2025
arXiv:2510.14986

Captures US equities 2020-2024 including COVID era market regime changes.
Uses ML to detect market regimes and optimize sector allocation dynamically.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class MarketRegime(Enum):
    """Identified market regimes"""
    BULL_LOW_VOL = "bull_low_volatility"
    BULL_HIGH_VOL = "bull_high_volatility"
    BEAR_HIGH_VOL = "bear_high_volatility"  # Crisis
    SIDEWAYS = "sideways_consolidation"
    RECOVERY = "recovery_rebound"


@dataclass
class RegimeFolioConfig:
    """Configuration for RegimeFolio"""
    n_regimes: int = 4  # Number of market regimes to detect
    n_sectors: int = 11  # GICS sectors
    lookback_window: int = 60  # Days for regime detection
    
    # Regime detection
    regime_features: List[str] = None  # volatility, returns, correlation, volume
    
    # Portfolio optimization per regime
    optimize_per_regime: bool = True
    max_sector_weight: float = 0.25  # Max 25% per sector
    min_sector_weight: float = 0.02  # Min 2% per sector
    
    # ML architecture
    hidden_dim: int = 128
    dropout: float = 0.3
    
    # Training
    learning_rate: float = 1e-3
    batch_size: int = 32
    
    def __post_init__(self):
        if self.regime_features is None:
            self.regime_features = ['volatility', 'returns', 'correlation', 'volume_trend']


class RegimeDetector:
    """
    Detects market regimes using Gaussian Mixture Model
    
    Features:
    - Volatility level
    - Return trend
    - Cross-asset correlation
    - Volume patterns
    """
    
    def __init__(self, n_regimes: int = 4):
        self.n_regimes = n_regimes
        self.gmm = GaussianMixture(
            n_components=n_regimes,
            covariance_type='full',
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, market_features: np.ndarray):
        """
        Fit regime detector on historical market data
        
        Args:
            market_features: (n_timesteps, n_features) array
        """
        # Normalize features
        features_scaled = self.scaler.fit_transform(market_features)
        
        # Fit GMM
        self.gmm.fit(features_scaled)
        self.is_fitted = True
    
    def detect_regime(self, current_features: np.ndarray) -> int:
        """
        Detect current market regime
        
        Args:
            current_features: Current market features
            
        Returns:
            Regime index (0 to n_regimes-1)
        """
        if not self.is_fitted:
            return 0  # Default regime
        
        # Normalize
        features_scaled = self.scaler.transform(current_features.reshape(1, -1))
        
        # Predict regime
        regime = self.gmm.predict(features_scaled)[0]
        
        return int(regime)
    
    def get_regime_probabilities(self, current_features: np.ndarray) -> np.ndarray:
        """Get probabilities for all regimes"""
        if not self.is_fitted:
            return np.ones(self.n_regimes) / self.n_regimes
        
        features_scaled = self.scaler.transform(current_features.reshape(1, -1))
        probs = self.gmm.predict_proba(features_scaled)[0]
        
        return probs


class RegimeAwarePortfolioNetwork(nn.Module):
    """
    Neural network for regime-specific portfolio allocation
    
    Different optimal allocations for different market regimes.
    """
    
    def __init__(self, config: RegimeFolioConfig):
        super(RegimeAwarePortfolioNetwork, self).__init__()
        
        self.config = config
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(config.n_sectors * 5, config.hidden_dim),  # 5 features per sector
            nn.BatchNorm1d(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # Regime-specific allocation heads
        self.regime_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim // 2, config.n_sectors),
                nn.Softmax(dim=-1)  # Ensure weights sum to 1
            )
            for _ in range(config.n_regimes)
        ])
    
    def forward(self, sector_features: torch.Tensor, regime: int) -> torch.Tensor:
        """
        Get optimal sector allocation for current regime
        
        Args:
            sector_features: (batch, n_sectors * 5) features
            regime: Current regime index
            
        Returns:
            Sector weights (batch, n_sectors)
        """
        # Extract features
        features = self.feature_extractor(sector_features)
        
        # Get regime-specific allocation
        weights = self.regime_heads[regime](features)
        
        # Apply position limits
        weights = torch.clamp(weights, self.config.min_sector_weight, self.config.max_sector_weight)
        
        # Renormalize
        weights = weights / weights.sum(dim=-1, keepdim=True)
        
        return weights


class RegimeFolioOptimizer:
    """
    Complete RegimeFolio system
    
    Dynamically adjusts sector allocation based on detected market regime.
    """
    
    def __init__(self, config: Optional[RegimeFolioConfig] = None):
        if not TORCH_AVAILABLE or not SKLEARN_AVAILABLE:
            raise ImportError("PyTorch and scikit-learn required")
        
        self.config = config or RegimeFolioConfig()
        
        # Components
        self.regime_detector = RegimeDetector(self.config.n_regimes)
        self.portfolio_network = RegimeAwarePortfolioNetwork(self.config)
        
        self.optimizer = None
        self.history = {'train_loss': [], 'regime_accuracy': []}
    
    def train(
        self,
        market_data: np.ndarray,
        sector_returns: np.ndarray,
        epochs: int = 100,
        verbose: int = 1
    ):
        """
        Train RegimeFolio on historical data
        
        Args:
            market_data: (timesteps, n_features) market features
            sector_returns: (timesteps, n_sectors) sector returns
            epochs: Training epochs
            verbose: Verbosity
        """
        # Step 1: Fit regime detector
        self.regime_detector.fit(market_data)
        
        if verbose > 0:
            print(f"Regime detector fitted on {len(market_data)} timesteps")
        
        # Step 2: Train portfolio network
        self.portfolio_network.train()
        self.optimizer = torch.optim.Adam(
            self.portfolio_network.parameters(),
            lr=self.config.learning_rate
        )
        
        # Create training sequences
        X_train, y_train, regimes = self._create_training_data(
            market_data, sector_returns
        )
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            # Train on batches
            indices = torch.randperm(len(X_train))
            
            for i in range(0, len(X_train), self.config.batch_size):
                batch_idx = indices[i:i + self.config.batch_size]
                batch_X = X_train[batch_idx]
                batch_y = y_train[batch_idx]
                batch_regimes = regimes[batch_idx]
                
                # Forward for each regime in batch
                total_loss = 0
                for j, regime in enumerate(batch_regimes):
                    weights = self.portfolio_network(batch_X[j:j+1], regime.item())
                    returns = batch_y[j]
                    
                    # Portfolio return
                    port_return = (weights * returns).sum()
                    
                    # Loss: negative return (we maximize)
                    loss = -port_return
                    total_loss += loss
                
                avg_loss = total_loss / len(batch_regimes)
                
                # Backward
                self.optimizer.zero_grad()
                avg_loss.backward()
                self.optimizer.step()
                
                epoch_loss += avg_loss.item()
                n_batches += 1
            
            avg_epoch_loss = epoch_loss / n_batches
            self.history['train_loss'].append(avg_epoch_loss)
            
            if verbose > 0 and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_epoch_loss:.6f}")
    
    def allocate(self, current_market_features: np.ndarray, current_sector_features: np.ndarray) -> np.ndarray:
        """
        Get optimal sector allocation based on current regime
        
        Args:
            current_market_features: Current market state for regime detection
            current_sector_features: Current sector data for allocation
            
        Returns:
            Optimal sector weights
        """
        # Detect regime
        regime = self.regime_detector.detect_regime(current_market_features)
        
        # Get allocation for this regime
        self.portfolio_network.eval()
        
        with torch.no_grad():
            sector_tensor = torch.FloatTensor(current_sector_features).unsqueeze(0)
            weights = self.portfolio_network(sector_tensor, regime)
        
        return weights.squeeze().cpu().numpy()
    
    def _create_training_data(
        self,
        market_data: np.ndarray,
        sector_returns: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create training sequences"""
        
        X_list = []
        y_list = []
        regime_list = []
        
        lookback = self.config.lookback_window
        
        for t in range(lookback, len(market_data)):
            # Market features for regime detection
            market_window = market_data[t-lookback:t]
            regime = self.regime_detector.detect_regime(market_window.mean(axis=0))
            
            # Sector features
            sector_window = sector_returns[t-lookback:t]
            sector_features = sector_window.flatten()  # Simplified
            
            # Next period returns
            next_returns = sector_returns[t]
            
            X_list.append(sector_features[:self.config.n_sectors * 5])  # Limit features
            y_list.append(next_returns)
            regime_list.append(regime)
        
        return (
            torch.FloatTensor(np.array(X_list)),
            torch.FloatTensor(np.array(y_list)),
            torch.LongTensor(regime_list)
        )
    
    def save(self, path: str):
        """Save models"""
        import joblib
        torch.save({
            'portfolio_network': self.portfolio_network.state_dict(),
            'config': self.config,
            'history': self.history
        }, path)
        joblib.dump(self.regime_detector, path + '.regime')
    
    def load(self, path: str):
        """Load models"""
        import joblib
        checkpoint = torch.load(path)
        self.portfolio_network.load_state_dict(checkpoint['portfolio_network'])
        self.history = checkpoint.get('history', {})
        self.regime_detector = joblib.load(path + '.regime')


# Example
if __name__ == "__main__":
    print("RegimeFolio - Example")
    print("=" * 60)
    
    if not all([TORCH_AVAILABLE, SKLEARN_AVAILABLE]):
        print("Missing dependencies")
    else:
        # Sample data
        market_data = np.random.randn(300, 4)  # 4 market features
        sector_returns = np.random.randn(300, 11) * 0.01  # 11 sectors
        
        config = RegimeFolioConfig(n_regimes=4, n_sectors=11)
        regimefolio = RegimeFolioOptimizer(config)
        
        print("\nTraining RegimeFolio...")
        regimefolio.train(market_data, sector_returns, epochs=50, verbose=1)
        
        print("\n✓ RegimeFolio trained")
        print("  • Regime-aware allocation")
        print("  • Sector optimization")
        print("  • Dynamic adaptation")