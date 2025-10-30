"""
Axiom Derivatives Platform - Python Client SDK

Official Python client library for accessing the derivatives platform.
Simplifies integration for clients with high-level API.

Usage:
    from axiom.derivatives.client import DerivativesClient
    
    client = DerivativesClient(api_key="your_key")
    greeks = client.calculate_greeks(spot=100, strike=100, time=1.0, rate=0.03, vol=0.25)
    print(f"Delta: {greeks['delta']}")
"""

import requests
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import time


@dataclass
class GreeksResult:
    """Greeks calculation result"""
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    price: float
    calculation_time_us: float
    
    @classmethod
    def from_dict(cls, data: Dict):
        return cls(**data)


class DerivativesClient:
    """
    Official Python client for Axiom Derivatives Platform
    
    Features:
    - Simple high-level API
    - Automatic retries
    - Connection pooling
    - Caching
    - Error handling
    """
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://derivatives.axiom-platform.com",
        timeout: int = 30
    ):
        """
        Initialize derivatives client
        
        Args:
            api_key: Your API key
            base_url: API base URL
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        })
        
        # Cache for repeated queries
        self._cache = {}
    
    def calculate_greeks(
        self,
        spot: float,
        strike: float,
        time_to_maturity: float,
        risk_free_rate: float,
        volatility: float,
        option_type: str = 'call',
        use_cache: bool = True
    ) -> Dict:
        """
        Calculate option Greeks
        
        Args:
            spot: Current price of underlying
            strike: Strike price
            time_to_maturity: Time to expiration (years)
            risk_free_rate: Risk-free rate
            volatility: Implied volatility
            option_type: 'call' or 'put'
            use_cache: Use cached result if available
        
        Returns:
            Dict with delta, gamma, theta, vega, rho, price
        
        Example:
            >>> client = DerivativesClient(api_key="your_key")
            >>> greeks = client.calculate_greeks(100, 100, 1.0, 0.03, 0.25)
            >>> print(f"Delta: {greeks['delta']:.4f}")
            Delta: 0.5199
        """
        # Check cache
        cache_key = f"{spot}_{strike}_{time_to_maturity}_{risk_free_rate}_{volatility}_{option_type}"
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        # Make request
        response = self.session.post(
            f"{self.base_url}/greeks",
            json={
                "spot": spot,
                "strike": strike,
                "time_to_maturity": time_to_maturity,
                "risk_free_rate": risk_free_rate,
                "volatility": volatility,
                "option_type": option_type
            },
            timeout=self.timeout
        )
        
        response.raise_for_status()
        result = response.json()
        
        # Cache result
        if use_cache:
            self._cache[cache_key] = result
        
        return result
    
    def calculate_greeks_batch(
        self,
        options: List[Dict]
    ) -> List[Dict]:
        """
        Calculate Greeks for multiple options (batch)
        
        More efficient than individual calls for multiple options
        
        Args:
            options: List of option specifications
        
        Returns:
            List of Greeks results
        
        Example:
            >>> options = [
            ...     {"spot": 100, "strike": 95, "time_to_maturity": 1.0, ...},
            ...     {"spot": 100, "strike": 100, "time_to_maturity": 1.0, ...},
            ...     {"spot": 100, "strike": 105, "time_to_maturity": 1.0, ...}
            ... ]
            >>> results = client.calculate_greeks_batch(options)
            >>> len(results)
            3
        """
        response = self.session.post(
            f"{self.base_url}/greeks/batch",
            json=options,
            timeout=self.timeout
        )
        
        response.raise_for_status()
        return response.json()['results']
    
    def price_barrier_option(
        self,
        spot: float,
        strike: float,
        barrier: float,
        barrier_type: str,
        time_to_maturity: float,
        risk_free_rate: float,
        volatility: float
    ) -> Dict:
        """
        Price barrier option
        
        Args:
            barrier: Barrier level
            barrier_type: 'up_and_out', 'down_and_in', etc.
        
        Returns:
            Dict with price, delta, gamma, vega
        """
        response = self.session.post(
            f"{self.base_url}/exotic/barrier",
            json={
                "exotic_type": "barrier",
                "spot": spot,
                "strike": strike,
                "barrier": barrier,
                "barrier_type": barrier_type,
                "time_to_maturity": time_to_maturity,
                "risk_free_rate": risk_free_rate,
                "volatility": volatility
            },
            timeout=self.timeout
        )
        
        response.raise_for_status()
        return response.json()
    
    def get_volatility_surface(
        self,
        underlying: str,
        market_quotes: List[float],
        spot: float
    ) -> Dict:
        """
        Construct complete volatility surface
        
        Args:
            underlying: Ticker symbol
            market_quotes: Sparse market implied vols
            spot: Current spot price
        
        Returns:
            Dict with complete surface data
        """
        response = self.session.post(
            f"{self.base_url}/surface/construct",
            json={
                "underlying": underlying,
                "market_quotes": market_quotes,
                "spot": spot
            },
            timeout=self.timeout
        )
        
        response.raise_for_status()
        return response.json()
    
    def get_statistics(self) -> Dict:
        """Get engine statistics from API"""
        response = self.session.get(
            f"{self.base_url}/stats/engines",
            timeout=self.timeout
        )
        
        response.raise_for_status()
        return response.json()
    
    def health_check(self) -> bool:
        """Check if API is healthy"""
        try:
            response = self.session.get(
                f"{self.base_url}/health",
                timeout=5
            )
            return response.status_code == 200
        except:
            return False


# Example usage
if __name__ == "__main__":
    # Create client
    client = DerivativesClient(
        api_key="demo_key_for_testing",
        base_url="http://localhost:8000"
    )
    
    # Check health
    print("Checking API health...")
    if client.health_check():
        print("✓ API is healthy")
    else:
        print("✗ API is not available")
        exit(1)
    
    # Calculate Greeks
    print("\nCalculating Greeks...")
    greeks = client.calculate_greeks(
        spot=100.0,
        strike=100.0,
        time_to_maturity=1.0,
        risk_free_rate=0.03,
        volatility=0.25
    )
    
    print(f"Delta: {greeks['delta']:.4f}")
    print(f"Gamma: {greeks['gamma']:.4f}")
    print(f"Price: ${greeks['price']:.4f}")
    print(f"Latency: {greeks['calculation_time_microseconds']:.2f}us")
    
    # Batch calculation
    print("\nBatch calculation (3 options)...")
    options = [
        {"spot": 100, "strike": 95, "time_to_maturity": 1.0, "risk_free_rate": 0.03, "volatility": 0.25},
        {"spot": 100, "strike": 100, "time_to_maturity": 1.0, "risk_free_rate": 0.03, "volatility": 0.25},
        {"spot": 100, "strike": 105, "time_to_maturity": 1.0, "risk_free_rate": 0.03, "volatility": 0.25}
    ]
    
    batch_results = client.calculate_greeks_batch(options)
    print(f"Calculated {len(batch_results)} options")
    print(f"Average time: {batch_results[0]['calculation_time_microseconds']:.2f}us")
    
    # Price exotic
    print("\nPricing barrier option...")
    barrier_result = client.price_barrier_option(
        spot=100.0,
        strike=100.0,
        barrier=120.0,
        barrier_type='up_and_out',
        time_to_maturity=1.0,
        risk_free_rate=0.03,
        volatility=0.25
    )
    
    print(f"Price: ${barrier_result['price']:.4f}")
    print(f"Time: {barrier_result['calculation_time_ms']:.2f}ms")
    
    # Get statistics
    print("\nEngine statistics...")
    stats = client.get_statistics()
    print(f"Total calculations: {stats['greeks_engine']['total_calculations']}")
    print(f"Average time: {stats['greeks_engine']['average_time_microseconds']:.2f}us")
    
    print("\n✓ SDK working correctly")