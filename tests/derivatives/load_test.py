"""
Load Testing for Derivatives Platform

Tests system under realistic market maker load:
- 10,000+ requests/second
- Mix of Greeks, exotic, and surface requests
- Sustained load for 10+ minutes
- Latency p95 must stay <1ms

Run: locust -f tests/derivatives/load_test.py --host=http://localhost:8000
"""

from locust import HttpUser, task, between, events
import random
import numpy as np


class MarketMakerUser(HttpUser):
    """
    Simulates market maker API usage pattern
    
    Profile:
    - 90% Greeks calculations (high frequency)
    - 8% Exotic options pricing
    - 2% Volatility surface construction
    """
    
    # Realistic wait time between requests
    wait_time = between(0.001, 0.01)  # 1-10ms (HFT speed)
    
    def on_start(self):
        """Initialize user session"""
        self.spot_price = 100.0
        self.volatility = 0.25
    
    @task(90)
    def calculate_greeks(self):
        """
        Greeks calculation (most frequent operation)
        
        Target: <100us, so API latency should be <1ms
        """
        # Random option parameters (realistic)
        strike = self.spot_price * random.uniform(0.9, 1.1)
        time_to_maturity = random.choice([0.083, 0.25, 0.5, 1.0, 2.0])  # 1m, 3m, 6m, 1y, 2y
        
        self.client.post("/greeks", json={
            "spot": self.spot_price,
            "strike": round(strike, 2),
            "time_to_maturity": time_to_maturity,
            "risk_free_rate": 0.03,
            "volatility": self.volatility,
            "option_type": random.choice(['call', 'put'])
        }, name="/greeks")
    
    @task(8)
    def price_exotic(self):
        """
        Exotic option pricing
        
        Target: <2ms for exotic, so API should be <5ms
        """
        strike = self.spot_price * random.uniform(0.9, 1.1)
        barrier = self.spot_price * random.uniform(1.1, 1.3)
        
        self.client.post("/exotic/barrier", json={
            "exotic_type": "barrier",
            "spot": self.spot_price,
            "strike": round(strike, 2),
            "barrier": round(barrier, 2),
            "barrier_type": random.choice(['up_and_out', 'down_and_out', 'up_and_in', 'down_and_in']),
            "time_to_maturity": 1.0,
            "risk_free_rate": 0.03,
            "volatility": self.volatility
        }, name="/exotic/barrier")
    
    @task(2)
    def get_volatility_surface(self):
        """
        Volatility surface construction
        
        Target: <1ms construction, so API should be <3ms
        """
        # Generate market quotes
        quotes = [random.uniform(0.18, 0.32) for _ in range(20)]
        
        self.client.post("/surface/construct", json={
            "underlying": "SPY",
            "market_quotes": quotes,
            "spot": self.spot_price
        }, name="/surface/construct")


# Event handlers for custom metrics
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    print("="*60)
    print("DERIVATIVES PLATFORM LOAD TEST")
    print("="*60)
    print("Target: 10,000+ req/s with p95 <1ms")
    print("Profile: 90% Greeks, 8% Exotic, 2% Surface")
    print("="*60)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    print("\n" + "="*60)
    print("LOAD TEST COMPLETE")
    print("="*60)
    print("\nCheck Locust web UI for:")
    print("  - Total requests/sec")
    print("  - Response time percentiles")
    print("  - Failure rate")
    print("\nSLA Requirements:")
    print("  - Throughput: >10,000 req/s")
    print("  - Latency p95: <1ms")
    print("  - Latency p99: <5ms")
    print("  - Error rate: <0.01%")
    print("="*60)


# Custom load test profiles
class HighFrequencyTrader(HttpUser):
    """
    Ultra-high frequency profile
    
    Simulates HFT firm making 100K+ requests/day
    Even shorter wait times, pure Greeks focus
    """
    wait_time = between(0.0001, 0.001)  # 0.1-1ms between requests
    
    @task(100)
    def greeks_ultra_fast(self):
        """HFT only cares about Greeks speed"""
        self.client.post("/greeks", json={
            "spot": 100.0,
            "strike": 100.0,
            "time_to_maturity": 0.083,  # 1 month
            "risk_free_rate": 0.03,
            "volatility": 0.25,
            "option_type": "call"
        })


# Run configurations:

# Development test (low load):
# locust -f load_test.py --users 10 --spawn-rate 1 --host=http://localhost:8000

# Staging test (medium load):
# locust -f load_test.py --users 100 --spawn-rate 10 --host=https://derivatives-staging.axiom-platform.com

# Production test (high load):
# locust -f load_test.py --users 1000 --spawn-rate 100 --run-time 10m --host=https://derivatives.axiom-platform.com

# HFT simulation (extreme load):
# locust -f load_test.py --user-classes HighFrequencyTrader --users 500 --spawn-rate 50 --host=http://localhost:8000