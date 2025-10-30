"""
Derivatives Platform Test Suite

Comprehensive testing for sub-100 microsecond derivatives analytics platform.
All tests must pass before production deployment.
"""

# Test configuration
TEST_CONFIG = {
    'target_latency_us': 100,  # <100 microsecond target
    'target_throughput': 10000,  # 10K+ calculations/second
    'target_accuracy': 0.9999,  # 99.99% vs Black-Scholes
    'use_gpu': True,  # Test with GPU by default
    'num_benchmark_iterations': 10000,
    'load_test_users': 100,
    'load_test_duration_seconds': 300
}