"""Configuration settings for Axiom Investment Banking Analytics Platform."""

from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Keys (Optional for testing, required for production)
    tavily_api_key: str = Field("test_tavily_key", env="TAVILY_API_KEY")
    firecrawl_api_key: str = Field("test_firecrawl_key", env="FIRECRAWL_API_KEY")

    # Multi-AI Provider Support - Dynamic Configuration
    # Users configure only the providers they want to use

    # OpenAI Configuration (optional)
    openai_api_key: str | None = Field(None, env="OPENAI_API_KEY")
    openai_base_url: str = Field("https://api.openai.com/v1", env="OPENAI_BASE_URL")
    openai_model_name: str = Field("gpt-4o-mini", env="OPENAI_MODEL_NAME")

    # Claude Configuration (optional)
    claude_api_key: str | None = Field(None, env="CLAUDE_API_KEY")
    claude_base_url: str = Field("https://api.anthropic.com", env="CLAUDE_BASE_URL")
    claude_model_name: str = Field("claude-3-sonnet-20240229", env="CLAUDE_MODEL_NAME")

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
        0.8, env="DUE_DILIGENCE_CONFIDENCE_THRESHOLD"
    )

    # Financial Analysis Parameters
    valuation_model_types: str = Field(
        "dcf,comparable,precedent", env="VALUATION_MODEL_TYPES"
    )
    risk_analysis_enabled: bool = Field(True, env="RISK_ANALYSIS_ENABLED")
    regulatory_compliance_check: bool = Field(True, env="REGULATORY_COMPLIANCE_CHECK")
    market_volatility_assessment: bool = Field(True, env="MARKET_VOLATILITY_ASSESSMENT")

    # Financial Data API Keys (Optional)
    alpha_vantage_api_key: str | None = Field(None, env="ALPHA_VANTAGE_API_KEY")
    financial_modeling_prep_api_key: str | None = Field(
        None, env="FINANCIAL_MODELING_PREP_API_KEY"
    )
    polygon_api_key: str | None = Field(None, env="POLYGON_API_KEY")
    sec_edgar_user_agent: str | None = Field(None, env="SEC_EDGAR_USER_AGENT")

    # Legacy settings (for backward compatibility)
    max_parallel_tasks: int = Field(5, env="MAX_PARALLEL_TASKS")
    snippet_reasoning_threshold: int = Field(5, env="SNIPPET_REASONING_THRESHOLD")
    crawl_escalation_threshold: float = Field(0.6, env="CRAWL_ESCALATION_THRESHOLD")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def get_configured_providers(self) -> list[str]:
        """Get list of providers that have valid API keys configured"""
        providers = []

        # Check each provider for valid credentials
        if self.openai_api_key and self.openai_api_key != "sk-placeholder":
            providers.append("openai")

        if self.claude_api_key and self.claude_api_key != "sk-placeholder":
            providers.append("claude")

        # SGLang doesn't need API key for local inference
        if self.sglang_base_url:
            providers.append("sglang")

        if self.huggingface_api_key and self.huggingface_api_key != "hf_placeholder":
            providers.append("huggingface")

        if self.gemini_api_key and self.gemini_api_key != "gemini_placeholder":
            providers.append("gemini")

        return providers

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
