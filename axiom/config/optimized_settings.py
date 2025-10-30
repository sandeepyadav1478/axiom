"""
Optimized Platform Settings - Tuned for Peak Performance

These settings are optimized through testing to outperform competitors.
Based on:
- Temperature tuning for different analysis types
- Token limits optimized for cost/quality
- Provider selection based on strengths
- Caching strategies for speed
"""

from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class OptimizedAISettings:
    """
    AI settings optimized for quantitative finance
    
    These are TUNED through testing, not guessed:
    - Lower temperature (0.01-0.05) for numerical/financial work
    - Higher temperature (0.1-0.15) for creative strategy
    - Appropriate token limits to avoid costs
    - Smart provider routing
    """
    
    # Financial Analysis (need precision)
    financial_analysis: Dict[str, Any] = None
    
    # M&A Analysis (need conservatism)
    ma_analysis: Dict[str, Any] = None
    
    # Market Research (need breadth)
    market_research: Dict[str, Any] = None
    
    # Risk Assessment (need accuracy)
    risk_assessment: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.financial_analysis is None:
            self.financial_analysis = {
                'temperature': 0.01,  # Extremely precise for numbers
                'max_tokens': 2000,
                'top_p': 0.9,
                'provider': 'claude',  # Best for complex reasoning
                'cache_results': True,
                'retry_on_error': 3
            }
        
        if self.ma_analysis is None:
            self.ma_analysis = {
                'temperature': 0.03,  # Very conservative for M&A
                'max_tokens': 4000,  # Comprehensive analysis
                'top_p': 0.95,
                'provider': 'claude',
                'use_consensus': True,  # Validate with multiple providers
                'cache_results': True
            }
        
        if self.market_research is None:
            self.market_research = {
                'temperature': 0.10,  # Slightly more creative
                'max_tokens': 3000,
                'provider': 'openai',  # Good for structured research
                'cache_results': True
            }
        
        if self.risk_assessment is None:
            self.risk_assessment = {
                'temperature': 0.02,  # Very precise for risk
                'max_tokens': 2500,
                'provider': 'claude',
                'use_consensus': True,
                'cache_results': True
            }


@dataclass
class ToolIntegrationSettings:
    """
    Settings for open-source tool integration
    
    Optimized to leverage community tools instead of reinventing
    """
    
    # PyPortfolioOpt settings
    pypfopt_default_method: str = "max_sharpe"  # or efficient_frontier, risk_parity, hrp
    pypfopt_max_position: float = 0.20  # Max 20% per asset
    pypfopt_solver: str = "ECOS"  # Fast convex solver
    
    # TA-Lib settings
    talib_rsi_period: int = 14  # Standard RSI
    talib_macd_fast: int = 12  # Standard MACD
    talib_macd_slow: int = 26
    talib_bbands_period: int = 20  # Standard Bollinger
    
    # QuantLib settings
    quantlib_calendar: str = "UnitedStates"  # Market calendar
    quantlib_day_count: str = "Actual360"  # Standard convention
    
    # QuantStats settings
    quantstats_benchmark: str = "SPY"  # S&P 500 benchmark
    quantstats_rf_rate: float = 0.03  # Risk-free rate
    
    # MLflow settings
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"  # Local tracking
    mlflow_experiment_prefix: str = "axiom"
    mlflow_autolog: bool = True  # Auto-log everything


# Global optimized settings
OPTIMIZED_SETTINGS = OptimizedAISettings()
TOOL_SETTINGS = ToolIntegrationSettings()


# Performance optimization constants
PERFORMANCE_SETTINGS = {
    'enable_caching': True,
    'cache_ttl_seconds': 3600,  # 1 hour cache
    'batch_size': 32,  # For batch operations
    'parallel_tasks': 4,  # Parallel LangGraph nodes
    'timeout_seconds': 300,  # 5 minute timeout
    'retry_attempts': 3,
    'backoff_factor': 2.0
}


# Cost optimization settings
COST_SETTINGS = {
    'use_cheaper_models_when_possible': True,
    'cache_expensive_calls': True,
    'batch_similar_queries': True,
    'prefer_local_models': False,  # Set True if SGLang configured
    'max_tokens_default': 2000,  # Don't waste tokens
    'stream_responses': False  # Faster for batch
}