"""MCP Ecosystem Configuration.

Centralized configuration for all MCP servers across all categories.
Supports environment variables, API key rotation, and fallback strategies.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


@dataclass
class MCPEcosystemConfig:
    """Comprehensive MCP ecosystem configuration.
    
    Controls which MCP servers are enabled across all categories
    and provides fallback strategies.
    """

    # ===== EXTERNAL DATA PROVIDERS (Community-Maintained) =====
    # These replace custom REST API wrappers - zero maintenance!
    use_openbb_mcp: bool = True           # Replaces 5 providers (880 lines)
    use_sec_edgar_mcp: bool = True        # Replaces SEC wrapper (150 lines)
    use_fred_mcp: bool = True             # Replaces FRED integration (120 lines)
    use_coingecko_mcp: bool = True        # Replaces crypto wrapper (100 lines)
    use_newsapi_mcp: bool = True          # Replaces news wrapper (100 lines)
    
    # ===== INTERNAL DATA PROVIDERS =====
    # Financial data MCP servers (still maintained internally)
    use_polygon_mcp: bool = True
    use_yahoo_finance_mcp: bool = True
    
    # ===== DEPRECATED PROVIDERS =====
    # These are deprecated in favor of external MCPs
    use_alpha_vantage_mcp: bool = False   # → Use openbb_mcp
    use_finnhub_mcp: bool = False         # → Use openbb_mcp
    use_fmp_mcp: bool = False             # → Use openbb_mcp
    use_iex_cloud_mcp: bool = False       # → Use openbb_mcp

    # ===== STORAGE & DATABASES =====
    use_postgres_mcp: bool = True
    use_redis_mcp: bool = True
    use_mongodb_mcp: bool = False
    use_vector_db_mcp: bool = False  # Pinecone, Weaviate, ChromaDB
    use_s3_storage_mcp: bool = False

    # ===== FILESYSTEM & DOCUMENTS =====
    use_filesystem_mcp: bool = True
    use_pdf_processing_mcp: bool = True
    use_excel_mcp: bool = True
    use_markdown_mcp: bool = True
    use_docx_mcp: bool = False

    # ===== DEVOPS =====
    use_git_mcp: bool = True
    use_docker_mcp: bool = True
    use_kubernetes_mcp: bool = False
    use_github_actions_mcp: bool = True
    use_gitlab_ci_mcp: bool = False

    # ===== CLOUD INFRASTRUCTURE =====
    use_aws_mcp: bool = False
    use_gcp_mcp: bool = False
    use_azure_mcp: bool = False
    use_terraform_mcp: bool = False

    # ===== COMMUNICATION =====
    use_slack_mcp: bool = True
    use_email_mcp: bool = True
    use_sms_mcp: bool = False
    use_discord_mcp: bool = False
    use_teams_mcp: bool = False

    # ===== MONITORING & OBSERVABILITY =====
    use_prometheus_mcp: bool = True
    use_grafana_mcp: bool = False
    use_sentry_mcp: bool = False
    use_datadog_mcp: bool = False
    use_logging_mcp: bool = True

    # ===== ML OPERATIONS =====
    use_model_serving_mcp: bool = False
    use_training_pipeline_mcp: bool = False
    use_mlflow_mcp: bool = False
    use_wandb_mcp: bool = False

    # ===== CODE QUALITY =====
    use_linting_mcp: bool = True
    use_testing_mcp: bool = True
    use_security_scan_mcp: bool = False
    use_code_review_mcp: bool = False

    # ===== BUSINESS INTELLIGENCE =====
    use_analytics_mcp: bool = False
    use_tableau_mcp: bool = False
    use_powerbi_mcp: bool = False

    # ===== RESEARCH =====
    use_research_paper_mcp: bool = False
    use_patent_search_mcp: bool = False
    use_legal_research_mcp: bool = False

    # ===== GENERAL SETTINGS =====
    prefer_mcp_over_direct: bool = True  # Always prefer MCP when available
    fallback_to_direct: bool = True  # Fallback to direct API if MCP unavailable
    enable_mcp_caching: bool = True
    mcp_cache_ttl: int = 300  # seconds
    mcp_request_timeout: int = 30  # seconds
    mcp_max_retries: int = 3
    mcp_health_check_interval: int = 60  # seconds

    # API key rotation
    enable_api_key_rotation: bool = True
    rotation_on_rate_limit: bool = True
    rotation_on_error: bool = True

    # Performance
    max_concurrent_mcp_calls: int = 10
    enable_request_pooling: bool = True

    # Monitoring
    enable_mcp_metrics: bool = True
    enable_mcp_tracing: bool = True
    log_mcp_requests: bool = True


class MCPServerSettings(BaseSettings):
    """MCP server configuration from environment variables.
    
    Extends the main settings with MCP-specific configurations.
    """

    # ===== MCP ECOSYSTEM CONFIG =====
    mcp_ecosystem_config: MCPEcosystemConfig = Field(
        default_factory=MCPEcosystemConfig
    )

    # ===== FILESYSTEM MCP =====
    filesystem_mcp_root_path: str = Field("/", env="FILESYSTEM_MCP_ROOT_PATH")
    filesystem_mcp_allowed_paths: Optional[str] = Field(
        None, env="FILESYSTEM_MCP_ALLOWED_PATHS"
    )
    filesystem_mcp_max_file_size: int = Field(
        104857600, env="FILESYSTEM_MCP_MAX_FILE_SIZE"
    )  # 100MB

    # ===== GIT MCP =====
    git_mcp_default_branch: str = Field("main", env="GIT_MCP_DEFAULT_BRANCH")
    git_mcp_user_name: Optional[str] = Field(None, env="GIT_MCP_USER_NAME")
    git_mcp_user_email: Optional[str] = Field(None, env="GIT_MCP_USER_EMAIL")
    git_mcp_ssh_key_path: Optional[str] = Field(None, env="GIT_MCP_SSH_KEY_PATH")

    # ===== POSTGRESQL MCP =====
    postgres_mcp_host: str = Field("localhost", env="POSTGRES_MCP_HOST")
    postgres_mcp_port: int = Field(5432, env="POSTGRES_MCP_PORT")
    postgres_mcp_database: Optional[str] = Field(None, env="POSTGRES_MCP_DATABASE")
    postgres_mcp_user: Optional[str] = Field(None, env="POSTGRES_MCP_USER")
    postgres_mcp_password: Optional[str] = Field(None, env="POSTGRES_MCP_PASSWORD")
    postgres_mcp_max_connections: int = Field(10, env="POSTGRES_MCP_MAX_CONNECTIONS")

    # ===== REDIS MCP =====
    redis_mcp_host: str = Field("localhost", env="REDIS_MCP_HOST")
    redis_mcp_port: int = Field(6379, env="REDIS_MCP_PORT")
    redis_mcp_db: int = Field(0, env="REDIS_MCP_DB")
    redis_mcp_password: Optional[str] = Field(None, env="REDIS_MCP_PASSWORD")
    redis_mcp_max_connections: int = Field(50, env="REDIS_MCP_MAX_CONNECTIONS")

    # ===== SLACK MCP =====
    slack_mcp_token: Optional[str] = Field(None, env="SLACK_MCP_TOKEN")
    slack_mcp_webhook_url: Optional[str] = Field(None, env="SLACK_MCP_WEBHOOK_URL")
    slack_mcp_default_channel: str = Field("#general", env="SLACK_MCP_DEFAULT_CHANNEL")
    slack_mcp_bot_name: str = Field("Axiom Bot", env="SLACK_MCP_BOT_NAME")

    # ===== EMAIL MCP =====
    email_mcp_smtp_host: Optional[str] = Field(None, env="EMAIL_MCP_SMTP_HOST")
    email_mcp_smtp_port: int = Field(587, env="EMAIL_MCP_SMTP_PORT")
    email_mcp_smtp_user: Optional[str] = Field(None, env="EMAIL_MCP_SMTP_USER")
    email_mcp_smtp_password: Optional[str] = Field(None, env="EMAIL_MCP_SMTP_PASSWORD")
    email_mcp_from_address: Optional[str] = Field(None, env="EMAIL_MCP_FROM_ADDRESS")
    email_mcp_use_tls: bool = Field(True, env="EMAIL_MCP_USE_TLS")

    # ===== DOCKER MCP =====
    docker_mcp_socket: str = Field(
        "unix:///var/run/docker.sock", env="DOCKER_MCP_SOCKET"
    )
    docker_mcp_registry_url: Optional[str] = Field(None, env="DOCKER_MCP_REGISTRY_URL")
    docker_mcp_registry_user: Optional[str] = Field(
        None, env="DOCKER_MCP_REGISTRY_USER"
    )
    docker_mcp_registry_password: Optional[str] = Field(
        None, env="DOCKER_MCP_REGISTRY_PASSWORD"
    )

    # ===== PROMETHEUS MCP =====
    prometheus_mcp_url: str = Field(
        "http://localhost:9090", env="PROMETHEUS_MCP_URL"
    )
    prometheus_mcp_auth_token: Optional[str] = Field(
        None, env="PROMETHEUS_MCP_AUTH_TOKEN"
    )

    # ===== PDF PROCESSING MCP =====
    pdf_mcp_ocr_enabled: bool = Field(True, env="PDF_MCP_OCR_ENABLED")
    pdf_mcp_ocr_language: str = Field("eng", env="PDF_MCP_OCR_LANGUAGE")
    pdf_mcp_extract_tables: bool = Field(True, env="PDF_MCP_EXTRACT_TABLES")
    pdf_mcp_extract_images: bool = Field(False, env="PDF_MCP_EXTRACT_IMAGES")

    # ===== EXCEL MCP =====
    excel_mcp_max_rows: int = Field(100000, env="EXCEL_MCP_MAX_ROWS")
    excel_mcp_max_columns: int = Field(1000, env="EXCEL_MCP_MAX_COLUMNS")
    excel_mcp_evaluate_formulas: bool = Field(True, env="EXCEL_MCP_EVALUATE_FORMULAS")

    # ===== AWS MCP =====
    aws_mcp_region: str = Field("us-east-1", env="AWS_MCP_REGION")
    aws_mcp_access_key_id: Optional[str] = Field(None, env="AWS_MCP_ACCESS_KEY_ID")
    aws_mcp_secret_access_key: Optional[str] = Field(
        None, env="AWS_MCP_SECRET_ACCESS_KEY"
    )
    aws_mcp_profile: Optional[str] = Field(None, env="AWS_MCP_PROFILE")

    # ===== GCP MCP =====
    gcp_mcp_project_id: Optional[str] = Field(None, env="GCP_MCP_PROJECT_ID")
    gcp_mcp_credentials_path: Optional[str] = Field(None, env="GCP_MCP_CREDENTIALS_PATH")

    # ===== NOTIFICATION MCP =====
    notification_mcp_smtp_server: str = Field("smtp.gmail.com", env="NOTIFICATION_MCP_SMTP_SERVER")
    notification_mcp_smtp_port: int = Field(587, env="NOTIFICATION_MCP_SMTP_PORT")
    notification_mcp_smtp_user: Optional[str] = Field(None, env="NOTIFICATION_MCP_SMTP_USER")
    notification_mcp_smtp_password: Optional[str] = Field(None, env="NOTIFICATION_MCP_SMTP_PASSWORD")
    notification_mcp_smtp_from_address: Optional[str] = Field(None, env="NOTIFICATION_MCP_SMTP_FROM_ADDRESS")
    notification_mcp_smtp_use_tls: bool = Field(True, env="NOTIFICATION_MCP_SMTP_USE_TLS")
    notification_mcp_sendgrid_api_key: Optional[str] = Field(None, env="NOTIFICATION_MCP_SENDGRID_API_KEY")
    notification_mcp_mailgun_api_key: Optional[str] = Field(None, env="NOTIFICATION_MCP_MAILGUN_API_KEY")
    notification_mcp_mailgun_domain: Optional[str] = Field(None, env="NOTIFICATION_MCP_MAILGUN_DOMAIN")
    notification_mcp_twilio_account_sid: Optional[str] = Field(None, env="NOTIFICATION_MCP_TWILIO_ACCOUNT_SID")
    notification_mcp_twilio_auth_token: Optional[str] = Field(None, env="NOTIFICATION_MCP_TWILIO_AUTH_TOKEN")
    notification_mcp_twilio_from_number: Optional[str] = Field(None, env="NOTIFICATION_MCP_TWILIO_FROM_NUMBER")
    notification_mcp_default_channel: str = Field("email", env="NOTIFICATION_MCP_DEFAULT_CHANNEL")

    # ===== VECTOR DB MCP =====
    vector_db_mcp_provider: str = Field(
        "pinecone", env="VECTOR_DB_MCP_PROVIDER"
    )  # pinecone, weaviate, chromadb, qdrant
    vector_db_mcp_dimension: int = Field(1536, env="VECTOR_DB_MCP_DIMENSION")
    vector_db_mcp_api_key: Optional[str] = Field(None, env="VECTOR_DB_MCP_API_KEY")
    vector_db_mcp_environment: Optional[str] = Field(
        None, env="VECTOR_DB_MCP_ENVIRONMENT"
    )
    vector_db_mcp_index_name: str = Field(
        "axiom-index", env="VECTOR_DB_MCP_INDEX_NAME"
    )
    # Weaviate
    vector_db_mcp_weaviate_url: str = Field("http://localhost:8080", env="VECTOR_DB_MCP_WEAVIATE_URL")
    # ChromaDB
    vector_db_mcp_chromadb_path: str = Field("./chroma_db", env="VECTOR_DB_MCP_CHROMADB_PATH")
    vector_db_mcp_chromadb_host: Optional[str] = Field(None, env="VECTOR_DB_MCP_CHROMADB_HOST")
    vector_db_mcp_chromadb_port: Optional[int] = Field(None, env="VECTOR_DB_MCP_CHROMADB_PORT")
    # Qdrant
    vector_db_mcp_qdrant_url: str = Field("http://localhost:6333", env="VECTOR_DB_MCP_QDRANT_URL")
    vector_db_mcp_qdrant_collection: str = Field("axiom-collection", env="VECTOR_DB_MCP_QDRANT_COLLECTION")

    # ===== KUBERNETES MCP =====
    k8s_mcp_config_path: Optional[str] = Field(None, env="K8S_MCP_CONFIG_PATH")
    k8s_mcp_namespace: str = Field("default", env="K8S_MCP_NAMESPACE")
    k8s_mcp_context: Optional[str] = Field(None, env="K8S_MCP_CONTEXT")

    # ===== MLFLOW MCP =====
    mlflow_mcp_tracking_uri: Optional[str] = Field(None, env="MLFLOW_MCP_TRACKING_URI")
    mlflow_mcp_experiment_name: str = Field(
        "axiom-experiments", env="MLFLOW_MCP_EXPERIMENT_NAME"
    )
    mlflow_mcp_artifact_location: Optional[str] = Field(
        None, env="MLFLOW_MCP_ARTIFACT_LOCATION"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    def get_server_config(self, server_name: str) -> dict[str, Any]:
        """Get configuration for a specific MCP server.

        Args:
            server_name: Name of the MCP server

        Returns:
            Configuration dictionary for the server
        """
        configs = {
            "filesystem": {
                "root_path": self.filesystem_mcp_root_path,
                "allowed_paths": (
                    self.filesystem_mcp_allowed_paths.split(",")
                    if self.filesystem_mcp_allowed_paths
                    else None
                ),
                "max_file_size": self.filesystem_mcp_max_file_size,
            },
            "git": {
                "default_branch": self.git_mcp_default_branch,
                "user_name": self.git_mcp_user_name,
                "user_email": self.git_mcp_user_email,
                "ssh_key_path": self.git_mcp_ssh_key_path,
            },
            "postgres": {
                "host": self.postgres_mcp_host,
                "port": self.postgres_mcp_port,
                "database": self.postgres_mcp_database,
                "user": self.postgres_mcp_user,
                "password": self.postgres_mcp_password,
                "max_connections": self.postgres_mcp_max_connections,
            },
            "redis": {
                "host": self.redis_mcp_host,
                "port": self.redis_mcp_port,
                "db": self.redis_mcp_db,
                "password": self.redis_mcp_password,
                "max_connections": self.redis_mcp_max_connections,
            },
            "slack": {
                "token": self.slack_mcp_token,
                "webhook_url": self.slack_mcp_webhook_url,
                "default_channel": self.slack_mcp_default_channel,
                "bot_name": self.slack_mcp_bot_name,
            },
            "email": {
                "smtp_host": self.email_mcp_smtp_host,
                "smtp_port": self.email_mcp_smtp_port,
                "smtp_user": self.email_mcp_smtp_user,
                "smtp_password": self.email_mcp_smtp_password,
                "from_address": self.email_mcp_from_address,
                "use_tls": self.email_mcp_use_tls,
            },
            "docker": {
                "socket": self.docker_mcp_socket,
                "registry_url": self.docker_mcp_registry_url,
                "registry_user": self.docker_mcp_registry_user,
                "registry_password": self.docker_mcp_registry_password,
            },
            "prometheus": {
                "url": self.prometheus_mcp_url,
                "auth_token": self.prometheus_mcp_auth_token,
            },
            "pdf": {
                "ocr_enabled": self.pdf_mcp_ocr_enabled,
                "ocr_language": self.pdf_mcp_ocr_language,
                "extract_tables": self.pdf_mcp_extract_tables,
                "extract_images": self.pdf_mcp_extract_images,
            },
            "excel": {
                "max_rows": self.excel_mcp_max_rows,
                "max_columns": self.excel_mcp_max_columns,
                "evaluate_formulas": self.excel_mcp_evaluate_formulas,
            },
            "aws": {
                "region": self.aws_mcp_region,
                "access_key_id": self.aws_mcp_access_key_id,
                "secret_access_key": self.aws_mcp_secret_access_key,
                "profile": self.aws_mcp_profile,
            },
            "gcp": {
                "project_id": self.gcp_mcp_project_id,
                "credentials_path": self.gcp_mcp_credentials_path,
            },
            "notification": {
                "smtp_server": self.notification_mcp_smtp_server,
                "smtp_port": self.notification_mcp_smtp_port,
                "smtp_user": self.notification_mcp_smtp_user,
                "smtp_password": self.notification_mcp_smtp_password,
                "smtp_from_address": self.notification_mcp_smtp_from_address,
                "smtp_use_tls": self.notification_mcp_smtp_use_tls,
                "sendgrid_api_key": self.notification_mcp_sendgrid_api_key,
                "mailgun_api_key": self.notification_mcp_mailgun_api_key,
                "mailgun_domain": self.notification_mcp_mailgun_domain,
                "twilio_account_sid": self.notification_mcp_twilio_account_sid,
                "twilio_auth_token": self.notification_mcp_twilio_auth_token,
                "twilio_from_number": self.notification_mcp_twilio_from_number,
                "default_channel": self.notification_mcp_default_channel,
            },
            "vector_db": {
                "provider": self.vector_db_mcp_provider,
                "dimension": self.vector_db_mcp_dimension,
                "api_key": self.vector_db_mcp_api_key,
                "environment": self.vector_db_mcp_environment,
                "index_name": self.vector_db_mcp_index_name,
                "weaviate_url": self.vector_db_mcp_weaviate_url,
                "chromadb_path": self.vector_db_mcp_chromadb_path,
                "chromadb_host": self.vector_db_mcp_chromadb_host,
                "chromadb_port": self.vector_db_mcp_chromadb_port,
                "qdrant_url": self.vector_db_mcp_qdrant_url,
                "qdrant_collection": self.vector_db_mcp_qdrant_collection,
            },
            "kubernetes": {
                "config_path": self.k8s_mcp_config_path,
                "namespace": self.k8s_mcp_namespace,
                "context": self.k8s_mcp_context,
            },
            "mlflow": {
                "tracking_uri": self.mlflow_mcp_tracking_uri,
                "experiment_name": self.mlflow_mcp_experiment_name,
                "artifact_location": self.mlflow_mcp_artifact_location,
            },
        }

        return configs.get(server_name, {})

    def is_server_enabled(self, server_name: str) -> bool:
        """Check if an MCP server is enabled.

        Args:
            server_name: Name of the MCP server

        Returns:
            True if server is enabled
        """
        config = self.mcp_ecosystem_config
        server_flags = {
            # External data providers (community-maintained)
            "openbb": config.use_openbb_mcp,
            "sec_edgar": config.use_sec_edgar_mcp,
            "fred": config.use_fred_mcp,
            "coingecko": config.use_coingecko_mcp,
            "newsapi": config.use_newsapi_mcp,
            # Internal data providers
            "polygon": config.use_polygon_mcp,
            "yahoo_finance": config.use_yahoo_finance_mcp,
            # Deprecated providers
            "alpha_vantage": config.use_alpha_vantage_mcp,
            "finnhub": config.use_finnhub_mcp,
            "fmp": config.use_fmp_mcp,
            "iex_cloud": config.use_iex_cloud_mcp,
            # Storage
            "postgres": config.use_postgres_mcp,
            "redis": config.use_redis_mcp,
            "mongodb": config.use_mongodb_mcp,
            "vector_db": config.use_vector_db_mcp,
            # Filesystem
            "filesystem": config.use_filesystem_mcp,
            "pdf": config.use_pdf_processing_mcp,
            "excel": config.use_excel_mcp,
            "markdown": config.use_markdown_mcp,
            # DevOps
            "git": config.use_git_mcp,
            "docker": config.use_docker_mcp,
            "kubernetes": config.use_kubernetes_mcp,
            "github_actions": config.use_github_actions_mcp,
            # Cloud
            "aws": config.use_aws_mcp,
            "gcp": config.use_gcp_mcp,
            "azure": config.use_azure_mcp,
            # Communication
            "slack": config.use_slack_mcp,
            "email": config.use_email_mcp,
            "sms": config.use_sms_mcp,
            "notification": config.use_email_mcp or config.use_sms_mcp,
            # Monitoring
            "prometheus": config.use_prometheus_mcp,
            "grafana": config.use_grafana_mcp,
            "sentry": config.use_sentry_mcp,
            "logging": config.use_logging_mcp,
            # ML Ops
            "model_serving": config.use_model_serving_mcp,
            "mlflow": config.use_mlflow_mcp,
            # Code Quality
            "linting": config.use_linting_mcp,
            "testing": config.use_testing_mcp,
            "security_scan": config.use_security_scan_mcp,
        }

        return server_flags.get(server_name, False)

    def get_enabled_servers(self) -> list[str]:
        """Get list of all enabled MCP servers.

        Returns:
            List of enabled server names
        """
        config = self.mcp_ecosystem_config
        enabled = []

        # Check each server flag
        # External financial data providers
        if config.use_openbb_mcp:
            enabled.append("openbb")
        if config.use_sec_edgar_mcp:
            enabled.append("sec_edgar")
        if config.use_fred_mcp:
            enabled.append("fred")
        if config.use_coingecko_mcp:
            enabled.append("coingecko")
        if config.use_newsapi_mcp:
            enabled.append("newsapi")
        # Internal servers
        if config.use_filesystem_mcp:
            enabled.append("filesystem")
        if config.use_git_mcp:
            enabled.append("git")
        if config.use_postgres_mcp:
            enabled.append("postgres")
        if config.use_redis_mcp:
            enabled.append("redis")
        if config.use_slack_mcp:
            enabled.append("slack")
        if config.use_email_mcp:
            enabled.append("email")
        if config.use_docker_mcp:
            enabled.append("docker")
        if config.use_prometheus_mcp:
            enabled.append("prometheus")
        if config.use_pdf_processing_mcp:
            enabled.append("pdf")
        if config.use_excel_mcp:
            enabled.append("excel")
        if config.use_linting_mcp:
            enabled.append("linting")
        if config.use_testing_mcp:
            enabled.append("testing")
        # Week 3 servers
        if config.use_aws_mcp:
            enabled.append("aws")
        if config.use_gcp_mcp:
            enabled.append("gcp")
        if config.use_vector_db_mcp:
            enabled.append("vector_db")
        if config.use_kubernetes_mcp:
            enabled.append("kubernetes")
        # Notification server (uses communication category flags)
        if config.use_email_mcp or config.use_sms_mcp:
            enabled.append("notification")

        return enabled


# Global settings instance
mcp_settings = MCPServerSettings()