# Axiom Investment Banking Analytics Platform
# Terraform Variables Configuration
# AWS Deployment Parameters

# ============================================================================
# General Configuration
# ============================================================================

variable "environment" {
  description = "Deployment environment (dev, staging, production)"
  type        = string
  default     = "dev"
  
  validation {
    condition     = contains(["dev", "staging", "production"], var.environment)
    error_message = "Environment must be dev, staging, or production."
  }
}

variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name for resource naming"
  type        = string
  default     = "axiom-investment-banking"
}

# ============================================================================
# VPC Configuration
# ============================================================================

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "availability_zones" {
  description = "Availability zones for multi-AZ deployment"
  type        = list(string)
  default     = ["us-east-1a", "us-east-1b"]
}

variable "private_subnet_cidrs" {
  description = "CIDR blocks for private subnets"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24"]
}

variable "public_subnet_cidrs" {
  description = "CIDR blocks for public subnets"
  type        = list(string)
  default     = ["10.0.101.0/24", "10.0.102.0/24"]
}

variable "enable_nat_gateway" {
  description = "Enable NAT Gateway for private subnet internet access"
  type        = bool
  default     = true
}

variable "single_nat_gateway" {
  description = "Use single NAT Gateway for cost optimization (not HA)"
  type        = bool
  default     = true # Cost optimization for dev/staging
}

# ============================================================================
# Lambda Configuration
# ============================================================================

variable "lambda_memory_size" {
  description = "Lambda function memory in MB (128-10240)"
  type        = number
  default     = 1024 # 1GB for M&A analysis workloads
  
  validation {
    condition     = var.lambda_memory_size >= 128 && var.lambda_memory_size <= 10240
    error_message = "Lambda memory must be between 128 and 10240 MB."
  }
}

variable "lambda_timeout" {
  description = "Lambda function timeout in seconds (1-900)"
  type        = number
  default     = 900 # 15 minutes for complex M&A analysis
  
  validation {
    condition     = var.lambda_timeout >= 1 && var.lambda_timeout <= 900
    error_message = "Lambda timeout must be between 1 and 900 seconds."
  }
}

variable "lambda_reserved_concurrent_executions" {
  description = "Reserved concurrent executions for Lambda (null = unreserved)"
  type        = number
  default     = null
}

# ============================================================================
# RDS PostgreSQL Configuration
# ============================================================================

variable "db_master_username" {
  description = "Master username for RDS PostgreSQL"
  type        = string
  default     = "axiom_admin"
}

variable "db_master_password" {
  description = "Master password for RDS PostgreSQL"
  type        = string
  sensitive   = true
  
  validation {
    condition     = length(var.db_master_password) >= 8
    error_message = "Database password must be at least 8 characters."
  }
}

variable "db_min_capacity" {
  description = "Minimum ACU (Aurora Capacity Units) for RDS Serverless V2"
  type        = number
  default     = 0.5 # $0.12/hour minimum
}

variable "db_max_capacity" {
  description = "Maximum ACU for RDS Serverless V2"
  type        = number
  default     = 4.0 # Scale up to 4 ACU
}

variable "db_backup_retention" {
  description = "Database backup retention period in days"
  type        = number
  default     = 7
}

variable "db_preferred_backup_window" {
  description = "Preferred backup window (UTC)"
  type        = string
  default     = "03:00-04:00"
}

variable "db_preferred_maintenance_window" {
  description = "Preferred maintenance window (UTC)"
  type        = string
  default     = "sun:04:00-sun:05:00"
}

# ============================================================================
# ElastiCache Redis Configuration
# ============================================================================

variable "redis_data_storage_gb" {
  description = "Redis data storage limit in GB"
  type        = number
  default     = 1 # 1GB for financial data caching
}

variable "redis_ecpu_per_second" {
  description = "ElastiCache Processing Units per second"
  type        = number
  default     = 1000
}

# ============================================================================
# S3 Configuration
# ============================================================================

variable "s3_data_lifecycle_days" {
  description = "Days before moving data to Glacier (0 = disabled)"
  type        = number
  default     = 90
}

variable "s3_artifacts_lifecycle_days" {
  description = "Days before moving artifacts to Glacier"
  type        = number
  default     = 30
}

variable "s3_logs_lifecycle_days" {
  description = "Days before deleting logs"
  type        = number
  default     = 30
}

# ============================================================================
# API Gateway Configuration
# ============================================================================

variable "api_throttle_burst" {
  description = "API Gateway burst limit (requests)"
  type        = number
  default     = 2000
}

variable "api_throttle_rate" {
  description = "API Gateway rate limit (requests per second)"
  type        = number
  default     = 1000
}

variable "custom_domain_name" {
  description = "Custom domain name for API (leave empty for default)"
  type        = string
  default     = ""
}

variable "certificate_arn" {
  description = "ACM certificate ARN for custom domain"
  type        = string
  default     = ""
}

# ============================================================================
# CloudWatch Configuration
# ============================================================================

variable "enable_cloudwatch_alarms" {
  description = "Enable CloudWatch alarms"
  type        = bool
  default     = true
}

variable "alarm_notification_email" {
  description = "Email for CloudWatch alarm notifications"
  type        = string
  default     = ""
}

variable "cloudwatch_log_retention_days" {
  description = "CloudWatch log retention in days"
  type        = number
  default     = 30
}

variable "log_level" {
  description = "Application log level"
  type        = string
  default     = "INFO"
  
  validation {
    condition     = contains(["DEBUG", "INFO", "WARNING", "ERROR"], var.log_level)
    error_message = "Log level must be DEBUG, INFO, WARNING, or ERROR."
  }
}

# ============================================================================
# API Keys & Secrets
# ============================================================================

variable "tavily_api_key" {
  description = "Tavily API key for financial search"
  type        = string
  sensitive   = true
  default     = ""
}

variable "firecrawl_api_key" {
  description = "Firecrawl API key for SEC filing extraction"
  type        = string
  sensitive   = true
  default     = ""
}

variable "openai_api_key" {
  description = "OpenAI API key (single key)"
  type        = string
  sensitive   = true
  default     = ""
}

variable "openai_api_keys" {
  description = "OpenAI API keys (comma-separated for rotation)"
  type        = string
  sensitive   = true
  default     = ""
}

variable "claude_api_key" {
  description = "Claude API key (single key)"
  type        = string
  sensitive   = true
  default     = ""
}

variable "claude_api_keys" {
  description = "Claude API keys (comma-separated for rotation)"
  type        = string
  sensitive   = true
  default     = ""
}

variable "sglang_base_url" {
  description = "SGLang server base URL (if using local inference)"
  type        = string
  default     = ""
}

# ============================================================================
# Tagging Configuration
# ============================================================================

variable "additional_tags" {
  description = "Additional tags to apply to all resources"
  type        = map(string)
  default     = {}
}

variable "cost_center" {
  description = "Cost center tag for billing"
  type        = string
  default     = "InvestmentBanking"
}

variable "owner" {
  description = "Resource owner"
  type        = string
  default     = "AxiomTeam"
}

# ============================================================================
# Feature Flags
# ============================================================================

variable "enable_waf" {
  description = "Enable AWS WAF for API Gateway"
  type        = bool
  default     = false # Cost optimization for dev
}

variable "enable_xray_tracing" {
  description = "Enable AWS X-Ray tracing"
  type        = bool
  default     = false # Cost optimization for dev
}

variable "enable_auto_scaling" {
  description = "Enable auto-scaling for Lambda and RDS"
  type        = bool
  default     = true
}

variable "enable_multi_az" {
  description = "Enable Multi-AZ deployment for high availability"
  type        = bool
  default     = false # Cost optimization for dev
}

# ============================================================================
# Cost Optimization
# ============================================================================

variable "enable_cost_optimization" {
  description = "Enable aggressive cost optimization (may reduce availability)"
  type        = bool
  default     = true
}

variable "scale_to_zero" {
  description = "Scale resources to zero during off-hours"
  type        = bool
  default     = false
}

variable "reserved_capacity" {
  description = "Use reserved capacity for cost savings (requires commitment)"
  type        = bool
  default     = false
}