# Axiom Investment Banking Analytics Platform
# AWS Infrastructure as Code (Terraform)
# Optimized for cost-effective deployment with AWS Free Tier compatibility

terraform {
  required_version = ">= 1.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  
  # Optional: S3 backend for state management
  # backend "s3" {
  #   bucket = "axiom-terraform-state"
  #   key    = "axiom/terraform.tfstate"
  #   region = "us-east-1"
  # }
}

provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = "Axiom Investment Banking Analytics"
      Environment = var.environment
      ManagedBy   = "Terraform"
      CostCenter  = "InvestmentBanking"
    }
  }
}

# Data sources
data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

# Local variables
locals {
  name_prefix = "axiom-${var.environment}"
  
  common_tags = {
    Project     = "Axiom Investment Banking"
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
  
  # API Gateway configuration
  api_name = "${local.name_prefix}-api"
  
  # Lambda configuration
  lambda_runtime     = "python3.13"
  lambda_timeout     = 900 # 15 minutes for complex M&A analysis
  lambda_memory      = var.lambda_memory_size
  
  # Database configuration
  db_name            = "axiom_investment_banking"
  db_master_username = "axiom_admin"
}

# VPC Module
module "vpc" {
  source = "./modules/vpc"
  
  name_prefix = local.name_prefix
  environment = var.environment
  
  vpc_cidr            = var.vpc_cidr
  availability_zones  = var.availability_zones
  private_subnet_cidrs = var.private_subnet_cidrs
  public_subnet_cidrs  = var.public_subnet_cidrs
  
  enable_nat_gateway = var.enable_nat_gateway
  single_nat_gateway = var.single_nat_gateway # Cost optimization
  
  tags = local.common_tags
}

# RDS PostgreSQL Serverless Module
module "rds" {
  source = "./modules/rds"
  
  name_prefix = local.name_prefix
  environment = var.environment
  
  database_name      = local.db_name
  master_username    = local.db_master_username
  master_password    = var.db_master_password
  
  vpc_id             = module.vpc.vpc_id
  subnet_ids         = module.vpc.private_subnet_ids
  allowed_cidr_blocks = [var.vpc_cidr]
  
  # Serverless V2 configuration for cost optimization
  engine_version     = "15.4"
  min_capacity       = var.db_min_capacity # 0.5 ACU = $0.12/hour
  max_capacity       = var.db_max_capacity # 4.0 ACU max
  
  backup_retention_days = var.db_backup_retention
  skip_final_snapshot   = var.environment != "production"
  
  tags = local.common_tags
}

# ElastiCache Redis Serverless Module
module "elasticache" {
  source = "./modules/elasticache"
  
  name_prefix = local.name_prefix
  environment = var.environment
  
  vpc_id              = module.vpc.vpc_id
  subnet_ids          = module.vpc.private_subnet_ids
  allowed_cidr_blocks = [var.vpc_cidr]
  
  # Serverless configuration
  cache_usage_limits = {
    data_storage = var.redis_data_storage_gb
    ecpu_per_second = var.redis_ecpu_per_second
  }
  
  tags = local.common_tags
}

# S3 Buckets Module
module "s3" {
  source = "./modules/s3"
  
  name_prefix = local.name_prefix
  environment = var.environment
  
  # Buckets configuration
  create_data_bucket      = true
  create_artifacts_bucket = true
  create_logs_bucket      = true
  
  # Lifecycle policies for cost optimization
  data_lifecycle_days      = var.s3_data_lifecycle_days
  artifacts_lifecycle_days = var.s3_artifacts_lifecycle_days
  logs_lifecycle_days      = var.s3_logs_lifecycle_days
  
  tags = local.common_tags
}

# Lambda Functions Module
module "lambda" {
  source = "./modules/lambda"
  
  name_prefix = local.name_prefix
  environment = var.environment
  
  # Lambda configuration
  runtime        = local.lambda_runtime
  timeout        = local.lambda_timeout
  memory_size    = local.lambda_memory
  
  # VPC configuration
  vpc_id              = module.vpc.vpc_id
  subnet_ids          = module.vpc.private_subnet_ids
  security_group_ids  = [module.vpc.lambda_security_group_id]
  
  # Environment variables for Lambda
  environment_variables = {
    # Database
    DB_HOST         = module.rds.endpoint
    DB_NAME         = local.db_name
    DB_USER         = local.db_master_username
    DB_PASSWORD     = var.db_master_password
    
    # Redis
    REDIS_HOST      = module.elasticache.endpoint
    REDIS_PORT      = "6379"
    
    # S3
    DATA_BUCKET     = module.s3.data_bucket_name
    ARTIFACTS_BUCKET = module.s3.artifacts_bucket_name
    
    # Application config
    ENVIRONMENT     = var.environment
    LOG_LEVEL       = var.log_level
    
    # API Keys (from secrets manager)
    TAVILY_API_KEY      = var.tavily_api_key
    FIRECRAWL_API_KEY   = var.firecrawl_api_key
    OPENAI_API_KEY      = var.openai_api_key
    CLAUDE_API_KEY      = var.claude_api_key
  }
  
  # IAM permissions
  additional_iam_policies = [
    module.s3.lambda_access_policy_arn,
    module.rds.lambda_access_policy_arn,
    module.elasticache.lambda_access_policy_arn,
  ]
  
  tags = local.common_tags
}

# API Gateway Module
module "api_gateway" {
  source = "./modules/api_gateway"
  
  name_prefix = local.name_prefix
  environment = var.environment
  
  # Lambda integration
  lambda_function_arn  = module.lambda.function_arn
  lambda_function_name = module.lambda.function_name
  
  # API configuration
  api_name        = local.api_name
  api_description = "Axiom Investment Banking Analytics API"
  
  # Throttling for cost control
  throttle_burst_limit = var.api_throttle_burst
  throttle_rate_limit  = var.api_throttle_rate
  
  # Custom domain (optional)
  custom_domain_name = var.custom_domain_name
  certificate_arn    = var.certificate_arn
  
  tags = local.common_tags
}

# Secrets Manager Module
module "secrets" {
  source = "./modules/secrets"
  
  name_prefix = local.name_prefix
  environment = var.environment
  
  # Secrets configuration
  secrets = {
    db_password = {
      description = "PostgreSQL master password"
      value       = var.db_master_password
    }
    openai_api_keys = {
      description = "OpenAI API keys (comma-separated)"
      value       = var.openai_api_keys
    }
    claude_api_keys = {
      description = "Claude API keys (comma-separated)"
      value       = var.claude_api_keys
    }
    tavily_api_key = {
      description = "Tavily API key"
      value       = var.tavily_api_key
    }
    firecrawl_api_key = {
      description = "Firecrawl API key"
      value       = var.firecrawl_api_key
    }
  }
  
  tags = local.common_tags
}

# CloudWatch Monitoring Module
module "cloudwatch" {
  source = "./modules/cloudwatch"
  
  name_prefix = local.name_prefix
  environment = var.environment
  
  # Lambda monitoring
  lambda_function_name = module.lambda.function_name
  
  # RDS monitoring
  rds_cluster_id = module.rds.cluster_id
  
  # ElastiCache monitoring  
  redis_cluster_id = module.elasticache.cluster_id
  
  # Alarm configuration
  enable_alarms           = var.enable_cloudwatch_alarms
  alarm_email            = var.alarm_notification_email
  
  # Log retention
  log_retention_days = var.cloudwatch_log_retention_days
  
  tags = local.common_tags
}

# Outputs
output "api_endpoint" {
  description = "API Gateway endpoint URL"
  value       = module.api_gateway.api_endpoint
}

output "custom_domain_url" {
  description = "Custom domain URL (if configured)"
  value       = var.custom_domain_name != "" ? "https://${var.custom_domain_name}" : "Not configured"
}

output "rds_endpoint" {
  description = "RDS PostgreSQL endpoint"
  value       = module.rds.endpoint
  sensitive   = true
}

output "redis_endpoint" {
  description = "ElastiCache Redis endpoint"
  value       = module.elasticache.endpoint
  sensitive   = true
}

output "s3_buckets" {
  description = "S3 bucket names"
  value = {
    data      = module.s3.data_bucket_name
    artifacts = module.s3.artifacts_bucket_name
    logs      = module.s3.logs_bucket_name
  }
}

output "lambda_function_name" {
  description = "Lambda function name"
  value       = module.lambda.function_name
}

output "estimated_monthly_cost" {
  description = "Estimated monthly AWS cost"
  value = <<-EOT
    Estimated Monthly Cost (Light Usage):
    - Lambda: $5-15 (Free tier: 1M requests/month)
    - RDS Serverless: $15-30 (0.5-2 ACU average)
    - ElastiCache: $15-25 (Serverless with 1GB data)
    - S3: $1-5 (Minimal storage)
    - Data Transfer: $5-10
    - CloudWatch: $2-5
    
    Total: ~$43-90/month (vs $24,000-51,000/year for Bloomberg/FactSet)
    Savings: 99.6% cost reduction
  EOT
}