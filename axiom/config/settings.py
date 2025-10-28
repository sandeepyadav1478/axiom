"""Configuration settings for Axiom Analytics Platform."""

from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Keys (Optional for testing, required for production)
    tavily_api_key: str = Field("test_tavily_key", env="TAVILY_API_KEY")
    firecrawl_api_key: str = Field("test_firecrawl_key", env="FIRECRAWL_API_KEY")

    # Multi-AI Provider Support - Dynamic Configuration with Failover
    # Users can configure multiple keys per provider for quota failover

    # OpenAI Configuration (supports multiple keys)
    openai_api_key: str | None = Field(None, env="OPENAI_API_KEY")
    openai_api_keys: str | None = Field(None, env="OPENAI_API_KEYS")  # Comma-separated multiple keys
    openai_base_url: str = Field("https://api.openai.com/v1", env="OPENAI_BASE_URL")
    openai_model_name: str = Field("gpt-4o-mini", env="OPENAI_MODEL_NAME")
    openai_rotation_enabled: bool = Field(True, env="OPENAI_ROTATION_ENABLED")

    # Claude Configuration (supports multiple keys)
    claude_api_key: str | None = Field(None, env="CLAUDE_API_KEY")
    claude_api_keys: str | None = Field(None, env="CLAUDE_API_KEYS")  # Comma-separated multiple keys
    claude_base_url: str = Field("https://api.anthropic.com", env="CLAUDE_BASE_URL")
    claude_model_name: str = Field("claude-3-5-sonnet-20241022", env="CLAUDE_MODEL_NAME")  # Latest Claude 3.5
    claude_rotation_enabled: bool = Field(True, env="CLAUDE_ROTATION_ENABLED")

    # SGLang Configuration (optional - for local inference)
    sglang_api_key: str | None = Field(
        None, env="SGLANG_API_KEY"
    )  # Usually None for local
    sglang_base_url: str = Field("http://localhost:30000/v1", env="SGLANG_BASE_URL")
    sglang_model_name: str = Field(
        "meta-llama/Meta-Llama-3.1-8B-Instruct", env="SGLANG_MODEL_NAME"
    )

    # Hugging Face Configuration (optional)
    huggingface_api_key: str | None = Field(None, env="HUGGINGFACE_API_KEY")
    huggingface_base_url: str = Field(
        "https://api-inference.huggingface.co", env="HUGGINGFACE_BASE_URL"
    )
    huggingface_model_name: str = Field(
        "microsoft/DialoGPT-medium", env="HUGGINGFACE_MODEL_NAME"
    )

    # Google Gemini Configuration (optional)
    gemini_api_key: str | None = Field(None, env="GEMINI_API_KEY")
    gemini_base_url: str = Field(
        "https://generativelanguage.googleapis.com/v1beta", env="GEMINI_BASE_URL"
    )
    gemini_model_name: str = Field("gemini-1.5-pro", env="GEMINI_MODEL_NAME")

    # Data Provider APIs (supports multiple keys)
    tavily_api_key: str = Field("test_tavily_key", env="TAVILY_API_KEY")
    tavily_api_keys: str | None = Field(None, env="TAVILY_API_KEYS")  # Multiple Tavily keys
    tavily_rotation_enabled: bool = Field(True, env="TAVILY_ROTATION_ENABLED")
    
    firecrawl_api_key: str = Field("test_firecrawl_key", env="FIRECRAWL_API_KEY")
    firecrawl_api_keys: str | None = Field(None, env="FIRECRAWL_API_KEYS")  # Multiple Firecrawl keys
    firecrawl_rotation_enabled: bool = Field(True, env="FIRECRAWL_ROTATION_ENABLED")

    # Global API Rotation Setting
    api_key_rotation_enabled: bool = Field(True, env="API_KEY_ROTATION_ENABLED")

    # LangSmith tracing
    langchain_tracing_v2: bool = Field(True, env="LANGCHAIN_TRACING_V2")
    langchain_endpoint: str = Field(
        "https://api.smith.langchain.com", env="LANGCHAIN_ENDPOINT"
    )
    langchain_api_key: str | None = Field(None, env="LANGCHAIN_API_KEY")
    langchain_project: str = Field("axiom-investment-banking", env="LANGCHAIN_PROJECT")

    # Investment Banking Application settings
    debug: bool = Field(False, env="DEBUG")
    max_parallel_analysis_tasks: int = Field(5, env="MAX_PARALLEL_ANALYSIS_TASKS")
    financial_data_depth: str = Field("comprehensive", env="FINANCIAL_DATA_DEPTH")
    due_diligence_confidence_threshold: float = Field(
        0.65, env="DUE_DILIGENCE_CONFIDENCE_THRESHOLD"  # Lowered for demo/testing
    )

    # Financial Analysis Parameters
    valuation_model_types: str = Field(
        "dcf,comparable,precedent", env="VALUATION_MODEL_TYPES"
    )
    risk_analysis_enabled: bool = Field(True, env="RISK_ANALYSIS_ENABLED")
    regulatory_compliance_check: bool = Field(True, env="REGULATORY_COMPLIANCE_CHECK")
    market_volatility_assessment: bool = Field(True, env="MARKET_VOLATILITY_ASSESSMENT")

    # Financial Data API Keys (Optional - with rotation support)
    alpha_vantage_api_key: str | None = Field(None, env="ALPHA_VANTAGE_API_KEY")
    financial_modeling_prep_api_key: str | None = Field(
        None, env="FINANCIAL_MODELING_PREP_API_KEY"
    )
    fmp_api_key: str | None = Field(None, env="FMP_API_KEY")  # Alias for FMP
    finnhub_api_key: str | None = Field(None, env="FINNHUB_API_KEY")
    polygon_api_key: str | None = Field(None, env="POLYGON_API_KEY")
    sec_edgar_user_agent: str | None = Field(None, env="SEC_EDGAR_USER_AGENT")
    
    # Financial API Rotation Settings
    financial_api_rotation_enabled: bool = Field(True, env="FINANCIAL_API_ROTATION_ENABLED")
    alpha_vantage_api_rotation_enabled: bool = Field(True, env="ALPHA_VANTAGE_API_ROTATION_ENABLED")
    polygon_api_rotation_enabled: bool = Field(True, env="POLYGON_API_ROTATION_ENABLED")
    finnhub_api_rotation_enabled: bool = Field(True, env="FINNHUB_API_ROTATION_ENABLED")
    fmp_api_rotation_enabled: bool = Field(True, env="FMP_API_ROTATION_ENABLED")

    # Quantitative Finance Model Settings
    # VaR (Value at Risk) Configuration
    var_default_confidence_level: float = Field(0.95, env="VAR_DEFAULT_CONFIDENCE_LEVEL")
    var_default_time_horizon_days: int = Field(1, env="VAR_DEFAULT_TIME_HORIZON_DAYS")
    var_monte_carlo_simulations: int = Field(10000, env="VAR_MONTE_CARLO_SIMULATIONS")
    var_calculation_method: str = Field("historical", env="VAR_CALCULATION_METHOD")  # parametric, historical, monte_carlo
    
    # Portfolio Optimization Configuration
    # Basic Settings
    portfolio_risk_free_rate: float = Field(0.02, env="PORTFOLIO_RISK_FREE_RATE")
    portfolio_default_optimization: str = Field("max_sharpe", env="PORTFOLIO_DEFAULT_OPTIMIZATION")
    portfolio_rebalancing_threshold: float = Field(0.05, env="PORTFOLIO_REBALANCING_THRESHOLD")
    portfolio_transaction_cost: float = Field(0.001, env="PORTFOLIO_TRANSACTION_COST")
    portfolio_efficient_frontier_points: int = Field(50, env="PORTFOLIO_EFFICIENT_FRONTIER_POINTS")
    
    # Position Constraints
    portfolio_allow_short_selling: bool = Field(False, env="PORTFOLIO_ALLOW_SHORT_SELLING")
    portfolio_min_position_weight: float = Field(0.0, env="PORTFOLIO_MIN_POSITION_WEIGHT")
    portfolio_max_position_weight: float = Field(1.0, env="PORTFOLIO_MAX_POSITION_WEIGHT")
    portfolio_max_concentration: float = Field(0.3, env="PORTFOLIO_MAX_CONCENTRATION")  # Max weight in single position
    portfolio_min_positions: int = Field(5, env="PORTFOLIO_MIN_POSITIONS")  # Diversification requirement
    
    # Risk Parameters
    portfolio_target_volatility: float = Field(0.15, env="PORTFOLIO_TARGET_VOLATILITY")  # 15% annual
    portfolio_max_volatility: float = Field(0.25, env="PORTFOLIO_MAX_VOLATILITY")  # 25% annual
    portfolio_target_return: float = Field(0.10, env="PORTFOLIO_TARGET_RETURN")  # 10% annual
    portfolio_min_sharpe_ratio: float = Field(0.5, env="PORTFOLIO_MIN_SHARPE_RATIO")
    portfolio_max_drawdown_limit: float = Field(0.20, env="PORTFOLIO_MAX_DRAWDOWN_LIMIT")  # 20%
    
    # Covariance Matrix Settings
    portfolio_cov_estimation_method: str = Field("sample", env="PORTFOLIO_COV_ESTIMATION_METHOD")  # sample, shrinkage, ledoit_wolf
    portfolio_cov_shrinkage_delta: float = Field(0.5, env="PORTFOLIO_COV_SHRINKAGE_DELTA")
    portfolio_cov_lookback_days: int = Field(252, env="PORTFOLIO_COV_LOOKBACK_DAYS")  # 1 year
    portfolio_cov_exponential_decay: float = Field(0.94, env="PORTFOLIO_COV_EXPONENTIAL_DECAY")  # EWMA decay
    
    # Return Forecasting
    portfolio_return_estimation: str = Field("historical", env="PORTFOLIO_RETURN_ESTIMATION")  # historical, black_litterman, factor_model
    portfolio_return_lookback_days: int = Field(252, env="PORTFOLIO_RETURN_LOOKBACK_DAYS")
    portfolio_use_momentum: bool = Field(False, env="PORTFOLIO_USE_MOMENTUM")
    portfolio_momentum_window_days: int = Field(90, env="PORTFOLIO_MOMENTUM_WINDOW_DAYS")
    
    # Optimization Solver Settings
    portfolio_optimizer_method: str = Field("SLSQP", env="PORTFOLIO_OPTIMIZER_METHOD")  # SLSQP, trust-constr, COBYLA
    portfolio_optimizer_max_iterations: int = Field(1000, env="PORTFOLIO_OPTIMIZER_MAX_ITERATIONS")
    portfolio_optimizer_tolerance: float = Field(1e-8, env="PORTFOLIO_OPTIMIZER_TOLERANCE")
    portfolio_use_gradient: bool = Field(True, env="PORTFOLIO_USE_GRADIENT")
    
    # Asset Allocation Preferences
    portfolio_equal_weight_baseline: bool = Field(False, env="PORTFOLIO_EQUAL_WEIGHT_BASELINE")
    portfolio_market_cap_weighted: bool = Field(False, env="PORTFOLIO_MARKET_CAP_WEIGHTED")
    portfolio_sector_constraints_enabled: bool = Field(False, env="PORTFOLIO_SECTOR_CONSTRAINTS_ENABLED")
    portfolio_max_sector_weight: float = Field(0.4, env="PORTFOLIO_MAX_SECTOR_WEIGHT")
    
    # Rebalancing Policy
    portfolio_rebalancing_frequency: str = Field("monthly", env="PORTFOLIO_REBALANCING_FREQUENCY")  # daily, weekly, monthly, quarterly
    portfolio_max_turnover: float = Field(0.5, env="PORTFOLIO_MAX_TURNOVER")  # 50% max turnover
    portfolio_min_trade_size: float = Field(0.01, env="PORTFOLIO_MIN_TRADE_SIZE")  # 1% minimum
    portfolio_tax_aware_rebalancing: bool = Field(False, env="PORTFOLIO_TAX_AWARE_REBALANCING")
    
    # Risk Parity Settings
    portfolio_risk_parity_method: str = Field("equal_risk", env="PORTFOLIO_RISK_PARITY_METHOD")  # equal_risk, equal_vol
    portfolio_risk_parity_leverage: float = Field(1.0, env="PORTFOLIO_RISK_PARITY_LEVERAGE")
    
    # Black-Litterman Settings
    portfolio_bl_tau: float = Field(0.025, env="PORTFOLIO_BL_TAU")  # Prior uncertainty
    portfolio_bl_risk_aversion: float = Field(2.5, env="PORTFOLIO_BL_RISK_AVERSION")
    portfolio_bl_views_confidence: float = Field(0.5, env="PORTFOLIO_BL_VIEWS_CONFIDENCE")
    
    # CVaR/Downside Risk Settings
    portfolio_cvar_alpha: float = Field(0.95, env="PORTFOLIO_CVAR_ALPHA")  # CVaR confidence level
    portfolio_focus_downside_risk: bool = Field(True, env="PORTFOLIO_FOCUS_DOWNSIDE_RISK")
    portfolio_mar_threshold: float = Field(0.0, env="PORTFOLIO_MAR_THRESHOLD")  # Minimum acceptable return
    
    # Market Regime Settings
    portfolio_market_regime: str = Field("normal", env="PORTFOLIO_MARKET_REGIME")  # normal, bull, bear, volatile
    portfolio_crisis_mode: bool = Field(False, env="PORTFOLIO_CRISIS_MODE")
    portfolio_defensive_allocation: bool = Field(False, env="PORTFOLIO_DEFENSIVE_ALLOCATION")
    
    # Performance Attribution
    portfolio_benchmark_tracking: bool = Field(False, env="PORTFOLIO_BENCHMARK_TRACKING")
    portfolio_max_tracking_error: float = Field(0.05, env="PORTFOLIO_MAX_TRACKING_ERROR")
    portfolio_target_information_ratio: float = Field(0.5, env="PORTFOLIO_TARGET_INFORMATION_RATIO")
    
    # Advanced Options
    portfolio_use_robust_estimation: bool = Field(False, env="PORTFOLIO_USE_ROBUST_ESTIMATION")
    portfolio_winsorize_returns: bool = Field(False, env="PORTFOLIO_WINSORIZE_RETURNS")
    portfolio_winsorize_percentile: float = Field(0.05, env="PORTFOLIO_WINSORIZE_PERCENTILE")
    portfolio_correlation_floor: float = Field(-0.99, env="PORTFOLIO_CORRELATION_FLOOR")
    portfolio_correlation_ceiling: float = Field(0.99, env="PORTFOLIO_CORRELATION_CEILING")
    
    # Risk Management Global Settings
    risk_monitoring_enabled: bool = Field(True, env="RISK_MONITORING_ENABLED")
    regulatory_var_enabled: bool = Field(True, env="REGULATORY_VAR_ENABLED")
    
    # Legacy settings (for backward compatibility)
    max_parallel_tasks: int = Field(5, env="MAX_PARALLEL_TASKS")
    snippet_reasoning_threshold: int = Field(5, env="SNIPPET_REASONING_THRESHOLD")
    crawl_escalation_threshold: float = Field(0.6, env="CRAWL_ESCALATION_THRESHOLD")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore extra fields from .env

    def get_configured_providers(self) -> list[str]:
        """Get list of providers that have valid API keys configured"""
        providers = []

        # Check each provider for valid credentials (single or multiple keys)
        if self._has_valid_keys("openai"):
            providers.append("openai")

        if self._has_valid_keys("claude"):
            providers.append("claude")

        # SGLang doesn't need API key for local inference
        if self.sglang_base_url:
            providers.append("sglang")

        if self.huggingface_api_key and self.huggingface_api_key != "hf_placeholder":
            providers.append("huggingface")

        if self.gemini_api_key and self.gemini_api_key != "gemini_placeholder":
            providers.append("gemini")

        return providers
    
    def _has_valid_keys(self, provider: str) -> bool:
        """Check if provider has valid API keys (single or multiple)."""
        
        if provider == "openai":
            single_key = self.openai_api_key and self.openai_api_key not in ["sk-placeholder", "test_key"]
            multiple_keys = self.openai_api_keys and self.openai_api_keys.strip()
            return single_key or multiple_keys
            
        elif provider == "claude":
            single_key = self.claude_api_key and self.claude_api_key not in ["sk-placeholder", "test_key"]
            multiple_keys = self.claude_api_keys and self.claude_api_keys.strip()
            return single_key or multiple_keys
        
        return False
    
    def get_provider_keys(self, provider: str) -> list[str]:
        """Get all API keys for a provider (single or multiple)."""
        
        if provider == "openai":
            # Check for multiple keys first, then single key
            if self.openai_api_keys and self.openai_api_keys.strip():
                return [k.strip() for k in self.openai_api_keys.split(",") if k.strip()]
            elif self.openai_api_key and self.openai_api_key not in ["sk-placeholder", "test_key"]:
                return [self.openai_api_key]
            
        elif provider == "claude":
            # Check for multiple keys first, then single key
            if self.claude_api_keys and self.claude_api_keys.strip():
                return [k.strip() for k in self.claude_api_keys.split(",") if k.strip()]
            elif self.claude_api_key and self.claude_api_key not in ["sk-placeholder", "test_key"]:
                return [self.claude_api_key]
        
        elif provider == "tavily":
            if self.tavily_api_keys and self.tavily_api_keys.strip():
                return [k.strip() for k in self.tavily_api_keys.split(",") if k.strip()]
            elif self.tavily_api_key and self.tavily_api_key not in ["test_tavily_key", "placeholder"]:
                return [self.tavily_api_key]
        
        elif provider == "firecrawl":
            if self.firecrawl_api_keys and self.firecrawl_api_keys.strip():
                return [k.strip() for k in self.firecrawl_api_keys.split(",") if k.strip()]
            elif self.firecrawl_api_key and self.firecrawl_api_key not in ["test_firecrawl_key", "placeholder"]:
                return [self.firecrawl_api_key]
        
        return []
    
    def is_rotation_enabled(self, provider: str) -> bool:
        """Check if rotation is enabled for specific provider."""
        
        if not self.api_key_rotation_enabled:
            return False
        
        if provider == "openai":
            return self.openai_rotation_enabled
        elif provider == "claude":
            return self.claude_rotation_enabled
        elif provider == "tavily":
            return self.tavily_rotation_enabled
        elif provider == "firecrawl":
            return self.firecrawl_rotation_enabled
        
        return True  # Default to enabled

    def get_provider_config(self, provider: str) -> dict[str, Any]:
        """Get configuration for specific AI provider (only if configured)"""
        provider = provider.lower()
        configured_providers = self.get_configured_providers()

        if provider not in configured_providers:
            raise ValueError(
                f"AI provider '{provider}' is not configured. Available: {configured_providers}"
            )

        configs = {
            "openai": {
                "api_key": self.openai_api_key,
                "base_url": self.openai_base_url,
                "model_name": self.openai_model_name,
            },
            "claude": {
                "api_key": self.claude_api_key,
                "base_url": self.claude_base_url,
                "model_name": self.claude_model_name,
            },
            "sglang": {
                "api_key": self.sglang_api_key or "local-inference",
                "base_url": self.sglang_base_url,
                "model_name": self.sglang_model_name,
            },
            "huggingface": {
                "api_key": self.huggingface_api_key,
                "base_url": self.huggingface_base_url,
                "model_name": self.huggingface_model_name,
            },
            "gemini": {
                "api_key": self.gemini_api_key,
                "base_url": self.gemini_base_url,
                "model_name": self.gemini_model_name,
            },
        }

        return configs.get(provider, {})

    def get_all_available_configs(self) -> dict[str, dict[str, Any]]:
        """Get configurations for all available providers"""
        return {
            provider: self.get_provider_config(provider)
            for provider in self.get_configured_providers()
        }

    def has_multiple_providers(self) -> bool:
        """Check if multiple AI providers are configured"""
        return len(self.get_configured_providers()) > 1


# Global settings instance
settings = Settings()
