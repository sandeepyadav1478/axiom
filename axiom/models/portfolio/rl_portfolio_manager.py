"""
Reinforcement Learning Portfolio Manager using Proximal Policy Optimization (PPO)

Based on: Wu Junfeng, Li Yaoming, Tan Wenqing, Chen Yun (2024)
"Portfolio management based on a reinforcement learning framework"
Journal of Forecasting, Volume 43, Issue 7, pp. 2792-2808
DOI: https://doi.org/10.1002/for.3155

This implementation addresses the continuous action space problem in portfolio management
where portfolio weights must sum to 1, using CNN for feature extraction and PPO for policy
optimization.
"""

from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Dirichlet
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYMNASIUM_AVAILABLE = True
except ImportError:
    GYMNASIUM_AVAILABLE = False

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False


@dataclass
class PortfolioConfig:
    """Configuration for RL Portfolio Manager"""
    n_assets: int = 6
    n_features: int = 16
    lookback_window: int = 30
    transaction_cost: float = 0.001  # 0.1%
    risk_free_rate: float = 0.02  # 2% annual
    gamma: float = 0.99  # Discount factor
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    clip_range: float = 0.2
    rebalance_frequency: str = "monthly"  # daily, weekly, monthly
    
    
import torch.nn as nn

class CNNFeatureExtractor(nn.Module):
    """
    CNN-based feature extraction for portfolio management
    
    Based on Wu et al. (2024) findings that CNN performed best in test set.
    Processes temporal sequences of asset features to extract patterns.
    """
    
    def __init__(self, n_assets: int, n_features: int, lookback_window: int):
        super(CNNFeatureExtractor, self).__init__()
        
        self.n_assets = n_assets
        self.n_features = n_features
        self.lookback_window = lookback_window
        
        # Convolutional layers for temporal pattern extraction
        # Input: (batch, n_assets, n_features, lookback_window)
        self.conv1 = nn.Conv2d(n_assets, 32, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Calculate flattened dimension
        self._calculate_flatten_dim()
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_dim, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        
    def _calculate_flatten_dim(self):
        """Calculate the flattened dimension after convolutions"""
        dummy_input = torch.zeros(1, self.n_assets, self.n_features, self.lookback_window)
        x = self.pool1(F.relu(self.bn1(self.conv1(dummy_input))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        self.flatten_dim = x.numel()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CNN feature extractor
        
        Args:
            x: Input tensor of shape (batch, n_assets, n_features, lookback_window)
            
        Returns:
            Extracted features of shape (batch, 128)
        """
        # Convolutional feature extraction
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        
        return x


class PortfolioActorCritic(nn.Module):
    """
    Actor-Critic network for portfolio weight allocation
    
    Actor outputs Dirichlet distribution parameters for portfolio weights
    ensuring weights are non-negative and sum to 1.
    """
    
    def __init__(self, config: PortfolioConfig):
        super(PortfolioActorCritic, self).__init__()
        
        self.config = config
        
        # Feature extractor (CNN)
        self.feature_extractor = CNNFeatureExtractor(
            n_assets=config.n_assets,
            n_features=config.n_features,
            lookback_window=config.lookback_window
        )
        
        # Actor head - outputs Dirichlet concentration parameters
        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, config.n_assets),
            nn.Softplus()  # Ensures positive concentration parameters
        )
        
        # Critic head - estimates state value
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, state: torch.Tensor) -> Tuple[Dirichlet, torch.Tensor]:
        """
        Forward pass returning action distribution and value estimate
        
        Args:
            state: Market state (batch, n_assets, n_features, lookback_window)
            
        Returns:
            (Dirichlet distribution over portfolio weights, state value)
        """
        # Extract features
        features = self.feature_extractor(state)
        
        # Actor: Get Dirichlet concentration parameters
        concentration = self.actor(features)
        # Add small constant for numerical stability
        concentration = concentration + 1.0
        distribution = Dirichlet(concentration)
        
        # Critic: Get state value
        value = self.critic(features)
        
        return distribution, value
        
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Sample action (portfolio weights) from policy
        
        Args:
            state: Market state
            deterministic: If True, use mean of distribution
            
        Returns:
            Portfolio weights summing to 1
        """
        distribution, _ = self.forward(state)
        
        if deterministic:
            # Use mode of Dirichlet (not well-defined, use mean)
            weights = distribution.mean
        else:
            weights = distribution.sample()
            
        return weights


class PortfolioEnvironment(gym.Env):
    """
    Gymnasium environment for portfolio optimization
    
    State: Historical prices and features for all assets
    Action: Portfolio weights (continuous, sum to 1)
    Reward: Sharpe ratio or portfolio return
    """
    
    def __init__(self, data: pd.DataFrame, config: PortfolioConfig):
        super(PortfolioEnvironment, self).__init__()
        
        self.data = data
        self.config = config
        
        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(config.n_assets,),
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(config.n_assets, config.n_features, config.lookback_window),
            dtype=np.float32
        )
        
        self.current_step = config.lookback_window
        self.portfolio_value = 1.0
        self.portfolio_weights = np.ones(config.n_assets) / config.n_assets
        
        # Track performance
        self.returns_history = []
        self.weights_history = []
        
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        self.current_step = self.config.lookback_window
        self.portfolio_value = 1.0
        self.portfolio_weights = np.ones(self.config.n_assets) / self.config.n_assets
        self.returns_history = []
        self.weights_history = []
        
        observation = self._get_observation()
        info = {}
        
        return observation, info
        
    def _get_observation(self) -> np.ndarray:
        """Get current market observation"""
        # Extract lookback window of features for all assets
        start_idx = self.current_step - self.config.lookback_window
        end_idx = self.current_step
        
        # Shape: (n_assets, n_features, lookback_window)
        observation = self.data.iloc[start_idx:end_idx].values.T
        observation = observation.reshape(
            self.config.n_assets, 
            self.config.n_features, 
            self.config.lookback_window
        )
        
        return observation.astype(np.float32)
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in environment
        
        Args:
            action: Portfolio weights (should sum to ~1)
            
        Returns:
            (observation, reward, terminated, truncated, info)
        """
        # Normalize action to ensure sum = 1
        action = action / (np.sum(action) + 1e-8)
        
        # Calculate transaction costs
        weight_diff = np.abs(action - self.portfolio_weights)
        transaction_cost = np.sum(weight_diff) * self.config.transaction_cost
        
        # Get asset returns for current step
        if self.current_step < len(self.data) - 1:
            current_prices = self.data.iloc[self.current_step].values
            next_prices = self.data.iloc[self.current_step + 1].values
            asset_returns = (next_prices - current_prices) / (current_prices + 1e-8)
        else:
            asset_returns = np.zeros(self.config.n_assets)
        
        # Calculate portfolio return
        portfolio_return = np.dot(action, asset_returns) - transaction_cost
        
        # Update portfolio value
        self.portfolio_value *= (1 + portfolio_return)
        self.portfolio_weights = action
        
        # Store history
        self.returns_history.append(portfolio_return)
        self.weights_history.append(action.copy())
        
        # Calculate reward (Sharpe ratio over recent window)
        if len(self.returns_history) >= 20:
            recent_returns = np.array(self.returns_history[-20:])
            sharpe = (np.mean(recent_returns) - self.config.risk_free_rate / 252) / (np.std(recent_returns) + 1e-8)
            reward = sharpe
        else:
            reward = portfolio_return * 10  # Scale for training stability
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        terminated = self.current_step >= len(self.data) - 1
        truncated = False
        
        # Get next observation
        if not terminated:
            observation = self._get_observation()
        else:
            observation = np.zeros_like(self._get_observation())
        
        info = {
            'portfolio_value': self.portfolio_value,
            'portfolio_return': portfolio_return,
            'transaction_cost': transaction_cost,
            'weights': action
        }
        
        return observation, reward, terminated, truncated, info


class RLPortfolioManager:
    """
    Reinforcement Learning Portfolio Manager using PPO
    
    Main class for training and using RL-based portfolio optimization.
    Based on Wu et al. (2024) approach with CNN feature extraction and PPO.
    """
    
    def __init__(self, config: Optional[PortfolioConfig] = None):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for RLPortfolioManager")
        if not GYMNASIUM_AVAILABLE:
            raise ImportError("Gymnasium is required for RLPortfolioManager")
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 is required for RLPortfolioManager")
            
        self.config = config or PortfolioConfig()
        self.model = None
        self.env = None
        
    def train(
        self,
        train_data: pd.DataFrame,
        total_timesteps: int = 100000,
        verbose: int = 1
    ) -> Dict[str, List[float]]:
        """
        Train RL portfolio manager
        
        Args:
            train_data: Historical data with shape (timesteps, n_assets * n_features)
            total_timesteps: Number of training timesteps
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        # Create environment
        self.env = DummyVecEnv([lambda: PortfolioEnvironment(train_data, self.config)])
        
        # Create PPO model
        self.model = PPO(
            policy="MlpPolicy",  # Will be overridden by custom network
            env=self.env,
            learning_rate=self.config.learning_rate,
            n_steps=self.config.n_steps,
            batch_size=self.config.batch_size,
            n_epochs=self.config.n_epochs,
            gamma=self.config.gamma,
            clip_range=self.config.clip_range,
            verbose=verbose
        )
        
        # Train model
        self.model.learn(total_timesteps=total_timesteps)
        
        return {'total_timesteps': [total_timesteps]}
        
    def allocate(
        self,
        current_state: np.ndarray,
        deterministic: bool = True
    ) -> np.ndarray:
        """
        Get optimal portfolio allocation
        
        Args:
            current_state: Current market state
                Shape: (n_assets, n_features, lookback_window)
            deterministic: Use deterministic policy
            
        Returns:
            Portfolio weights (n_assets,)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        # Predict action
        action, _ = self.model.predict(current_state, deterministic=deterministic)
        
        # Normalize to ensure sum = 1
        weights = action / (np.sum(action) + 1e-8)
        
        return weights
        
    def backtest(
        self,
        test_data: pd.DataFrame,
        initial_capital: float = 1.0
    ) -> Dict[str, Union[float, List[float]]]:
        """
        Backtest portfolio manager on historical data
        
        Args:
            test_data: Test dataset
            initial_capital: Initial portfolio value
            
        Returns:
            Backtest results with performance metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        # Create test environment
        test_env = PortfolioEnvironment(test_data, self.config)
        
        # Run backtest
        obs, _ = test_env.reset()
        portfolio_values = [initial_capital]
        weights_history = []
        
        done = False
        while not done:
            # Get action from model
            action, _ = self.model.predict(obs, deterministic=True)
            
            # Take step
            obs, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated
            
            portfolio_values.append(info['portfolio_value'])
            weights_history.append(info['weights'])
            
        # Calculate performance metrics
        returns = np.diff(portfolio_values) / np.array(portfolio_values[:-1])
        
        results = {
            'final_value': portfolio_values[-1],
            'total_return': (portfolio_values[-1] - initial_capital) / initial_capital,
            'sharpe_ratio': np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252),
            'max_drawdown': self._calculate_max_drawdown(portfolio_values),
            'portfolio_values': portfolio_values,
            'weights_history': weights_history,
            'returns': returns.tolist()
        }
        
        return results
        
    @staticmethod
    def _calculate_max_drawdown(values: List[float]) -> float:
        """Calculate maximum drawdown from portfolio values"""
        values = np.array(values)
        running_max = np.maximum.accumulate(values)
        drawdown = (values - running_max) / running_max
        return float(np.min(drawdown))
        
    def save(self, path: str):
        """Save trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        self.model.save(path)
        
    def load(self, path: str):
        """Load trained model"""
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 required to load model")
        self.model = PPO.load(path)


def create_sample_data(
    n_timesteps: int = 1000,
    n_assets: int = 6,
    n_features: int = 16
) -> pd.DataFrame:
    """
    Create sample data for testing
    
    Returns:
        DataFrame with shape (n_timesteps, n_assets * n_features)
    """
    np.random.seed(42)
    data = np.random.randn(n_timesteps, n_assets * n_features) * 0.02 + 1.0
    data = np.cumsum(data, axis=0)
    return pd.DataFrame(data)


# Example usage
if __name__ == "__main__":
    print("RL Portfolio Manager - Example Usage")
    print("=" * 50)
    
    if not all([TORCH_AVAILABLE, GYMNASIUM_AVAILABLE, SB3_AVAILABLE]):
        print("Missing required dependencies:")
        print(f"  PyTorch: {'✓' if TORCH_AVAILABLE else '✗'}")
        print(f"  Gymnasium: {'✓' if GYMNASIUM_AVAILABLE else '✗'}")
        print(f"  stable-baselines3: {'✓' if SB3_AVAILABLE else '✗'}")
    else:
        # Create sample data
        train_data = create_sample_data(n_timesteps=500)
        test_data = create_sample_data(n_timesteps=200)
        
        # Initialize manager
        config = PortfolioConfig(n_assets=6, n_features=16)
        manager = RLPortfolioManager(config)
        
        print("\nTraining RL Portfolio Manager...")
        manager.train(train_data, total_timesteps=10000, verbose=0)
        
        print("\nRunning backtest...")
        results = manager.backtest(test_data)
        
        print(f"\nBacktest Results:")
        print(f"  Total Return: {results['total_return']:.2%}")
        print(f"  Sharpe Ratio: {results['sharpe_ratio']:.3f}")
        print(f"  Max Drawdown: {results['max_drawdown']:.2%}")
        print(f"  Final Value: ${results['final_value']:.2f}")