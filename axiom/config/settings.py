"""Configuration settings for Axiom research agent."""

import os
from pydantic import BaseSettings, Field
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Keys
    tavily_api_key: str = Field(..., env="TAVILY_API_KEY")
    firecrawl_api_key: str = Field(..., env="FIRECRAWL_API_KEY")

    # OpenAI-compatible endpoint
    openai_api_key: str = Field("sk-placeholder", env="OPENAI_API_KEY")
    openai_base_url: str = Field("http://localhost:30000/v1", env="OPENAI_BASE_URL")
    openai_model_name: str = Field("meta-llama/Meta-Llama-3.1-8B-Instruct", env="OPENAI_MODEL_NAME")

    # LangSmith tracing
    langchain_tracing_v2: bool = Field(True, env="LANGCHAIN_TRACING_V2")
    langchain_endpoint: str = Field("https://api.smith.langchain.com", env="LANGCHAIN_ENDPOINT")
    langchain_api_key: Optional[str] = Field(None, env="LANGCHAIN_API_KEY")
    langchain_project: str = Field("axiom-research-agent", env="LANGCHAIN_PROJECT")

    # Application settings
    debug: bool = Field(False, env="DEBUG")
    max_parallel_tasks: int = Field(3, env="MAX_PARALLEL_TASKS")
    snippet_reasoning_threshold: int = Field(5, env="SNIPPET_REASONING_THRESHOLD")
    crawl_escalation_threshold: float = Field(0.6, env="CRAWL_ESCALATION_THRESHOLD")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()
