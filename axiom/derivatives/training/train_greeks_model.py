"""
Training Script for Ultra-Fast Greeks Neural Network

Trains the neural network to calculate Greeks with 99.99% accuracy vs Black-Scholes.
Uses synthetic data generation from analytical solutions.

Training approach:
1. Generate 1M+ training examples from Black-Scholes
2. Train neural network to replicate
3. Validate accuracy (99.99% target)
4. Export optimized model
5. Benchmark performance

Run: python axiom/derivatives/training/train_greeks_model.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.stats import norm
from typing import Tuple
import time
from pathlib import Path


class GreeksDataset(Dataset):
    """
    Dataset of option Greeks calculated from Black-Scholes
    
    Generates synthetic training data covering:
    - Spot: 50-200
    - Strike: 50-200
    - Time: 0.01-5 years
    - Rate: 0-10%
    - Vol: 5%-100%
    """
    
    def __init__(self, num_samples: int = 1_000_000):
        """Generate dataset"""
        print(f"Generating {num_samples:,} training samples...")
        
        np.random.seed(42)
        
        # Generate random parameters
        self.spots = np.random.uniform(50, 200, num_samples)
        self.strikes = np.random.uniform(50, 200, num_samples)
        self.times = np.random.uniform(0.01, 5.0, num_samples)
        self.rates = np.random.uniform(0.0, 0.10, num_samples)
        self.vols = np.random.uniform(0.05, 1.0, num_samples)
        
        # Calculate analytical Greeks
        print("Calculating analytical Greeks (Black-Scholes)...")
        self.greeks = self._calculate_greeks_vectorized()
        
        print(f"✓ Dataset ready: {num_samples:,} samples")
    
    def _calculate_greeks_vectorized(self) -> np.ndarray:
        """Calculate Greeks for all samples using Black-Scholes"""
        S, K, T, r, sigma = self.spots, self.strikes, self.times, self.rates, self.vols
        
        # Black-Scholes Greeks (vectorized)
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        delta = norm.cdf(d1)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                 - r * K * np.exp(-r*T) * norm.cdf(d2)) / 365
        rho = K * T * np.exp(-r*T) * norm.cdf(d2) / 100
        price = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
        
        # Stack into array [delta, gamma, theta, vega, rho, price]
        greeks = np.column_stack([delta, gamma, theta, vega, rho, price])
        
        return greeks.astype(np.float32)
    
    def __len__(self) -> int:
        return len(self.spots)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get single sample"""
        # Input: [spot, strike, time, rate, vol]
        x = torch.tensor([
            self.spots[idx],
            self.strikes[idx],
            self.times[idx],
            self.rates[idx],
            self.vols[idx]
        ], dtype=torch.float32)
        
        # Target: [delta, gamma, theta, vega, rho, price]
        y = torch.tensor(self.greeks[idx], dtype=torch.float32)
        
        return x, y


def train_greeks_model(
    num_samples: int = 1_000_000,
    batch_size: int = 1024,
    epochs: int = 50,
    learning_rate: float = 0.001,
    use_gpu: bool = True
):
    """
    Train Greeks neural network
    
    Target accuracy: 99.99% vs Black-Scholes
    """
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    
    # Create dataset
    dataset = GreeksDataset(num_samples=num_samples)
    
    # Split train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    from axiom.derivatives.ultra_fast_greeks import QuantizedGreeksNetwork
    model = QuantizedGreeksNetwork()
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Forward
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss: {val_loss:.6f}")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'greeks_model_best.pth')
            print(f"  ✓ New best model saved (val_loss: {val_loss:.6f})")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved to: greeks_model_best.pth")
    
    # Test final accuracy
    model.load_state_dict(torch.load('greeks_model_best.pth'))
    model.eval()
    
    test_errors = []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            
            # Relative error
            error = torch.abs(outputs - batch_y) / (torch.abs(batch_y) + 1e-10)
            test_errors.append(error.cpu().numpy())
    
    test_errors = np.concatenate(test_errors)
    mean_error = np.mean(test_errors)
    
    print(f"\nFinal Accuracy:")
    print(f"  Mean relative error: {mean_error:.6f} ({(1-mean_error)*100:.4f}% accuracy)")
    print(f"  Target: 99.99% accuracy")
    print(f"  Status: {'✓ ACHIEVED' if mean_error < 0.0001 else '✗ NEEDS IMPROVEMENT'}")
    
    return model


if __name__ == "__main__":
    # Train model
    model = train_greeks_model(
        num_samples=1_000_000,
        batch_size=1024,
        epochs=50,
        use_gpu=True
    )
    
    print("\n✓ Training complete")
    print("\nNext steps:")
    print("  1. Load trained model into ultra_fast_greeks.py")
    print("  2. Apply quantization (4x speedup)")
    print("  3. Apply TorchScript compilation (2x speedup)")
    print("  4. Benchmark final performance")
    print("  5. Deploy to production")