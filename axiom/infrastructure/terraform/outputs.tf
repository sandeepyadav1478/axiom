# Axiom Investment Banking Analytics Platform
# Terraform Outputs
# AWS Deployment Output Values

# ============================================================================
# API Outputs
# ============================================================================

output "api_gateway_endpoint" {
  description = "API Gateway invoke URL"
  value       = module.api_gateway.api_endpoint
}

output "api_gateway_id" {
  description = "API Gateway REST API ID"
  value       = module.api_gateway.api_id
}

output "custom_domain_url" {
  description = "Custom domain URL (if configured)"
  value       = var.custom_domain_name != "" ? "https://${var.custom_domain_name}" : "Not configured"
}

# ============================================================================
# Lambda Outputs
# ============================================================================

output "lambda_function_name" {
  description = "Lambda function name"
  value       = module.lambda.function_name
}

output "lambda_function_arn" {
  description = "Lambda function ARN"
  value       = module.lambda.function_arn
}

output "lambda_role_arn" {
  description = "Lambda execution role ARN"
  value       = module.lambda.role_arn
}

# ============================================================================
# Database Outputs
# ============================================================================

output "rds_endpoint" {
  description = "RDS PostgreSQL endpoint"
  value       = module.rds.endpoint
  sensitive   = true
}

output "rds_cluster_id" {
  description = "RDS cluster identifier"
  value       = module.rds.cluster_id
}

output "rds_database_name" {
  description = "RDS database name"
  value       = module.rds.database_name
}

output "redis_endpoint" {
  description = "ElastiCache Redis endpoint"
  value       = module.elasticache.endpoint
  sensitive   = true
}

output "redis_port" {
  description = "ElastiCache Redis port"
  value       = "6379"
}

# ============================================================================
# Storage Outputs
# ============================================================================

output "s3_data_bucket" {
  description = "S3 bucket for data storage"
  value       = module.s3.data_bucket_name
}

output "s3_artifacts_bucket" {
  description = "S3 bucket for build artifacts"
  value       = module.s3.artifacts_bucket_name
}

output "s3_logs_bucket" {
  description = "S3 bucket for logs"
  value       = module.s3.logs_bucket_name
}

# ============================================================================
# Network Outputs
# ============================================================================

output "vpc_id" {
  description = "VPC ID"
  value       = module.vpc.vpc_id
}

output "private_subnet_ids" {
  description = "Private subnet IDs"
  value       = module.vpc.private_subnet_ids
}

output "public_subnet_ids" {
  description = "Public subnet IDs"
  value       = module.vpc.public_subnet_ids
}

# ============================================================================
# Monitoring Outputs
# ============================================================================

output "cloudwatch_log_group" {
  description = "CloudWatch log group name"
  value       = module.cloudwatch.log_group_name
}

output "cloudwatch_dashboard_url" {
  description = "CloudWatch dashboard URL"
  value       = "https://console.aws.amazon.com/cloudwatch/home?region=${var.aws_region}#dashboards:name=${module.cloudwatch.dashboard_name}"
}

# ============================================================================
# Cost Estimation Output
# ============================================================================

output "estimated_monthly_cost_breakdown" {
  description = "Estimated monthly AWS costs by service"
  value = {
    lambda = {
      description = "Lambda compute costs"
      estimate    = "$5-15/month (1M requests free tier)"
    }
    rds = {
      description = "RDS Serverless V2 costs"
      estimate    = "$15-30/month (0.5-2 ACU average)"
    }
    elasticache = {
      description = "ElastiCache Serverless costs"
      estimate    = "$15-25/month (1GB data)"
    }
    s3 = {
      description = "S3 storage costs"
      estimate    = "$1-5/month (minimal storage)"
    }
    data_transfer = {
      description = "Data transfer costs"
      estimate    = "$5-10/month"
    }
    cloudwatch = {
      description = "CloudWatch logs and metrics"
      estimate    = "$2-5/month"
    }
    total = {
      description = "Total monthly cost"
      estimate    = "$43-90/month"
    }
    savings = {
      description = "vs Bloomberg/FactSet"
      estimate    = "99.6% cost reduction ($24K-51K/year → $43-90/month)"
    }
  }
}

# ============================================================================
# Connection Information
# ============================================================================

output "connection_details" {
  description = "Connection details for application configuration"
  value = {
    api_url       = module.api_gateway.api_endpoint
    database_host = module.rds.endpoint
    redis_host    = module.elasticache.endpoint
    data_bucket   = module.s3.data_bucket_name
  }
  sensitive = true
}

# ============================================================================
# Deployment Information
# ============================================================================

output "deployment_info" {
  description = "Deployment information and next steps"
  value = <<-EOT
    ╔════════════════════════════════════════════════════════════════╗
    ║  Axiom Investment Banking Analytics - AWS Deployment Complete  ║
    ╚════════════════════════════════════════════════════════════════╝
    
    Environment: ${var.environment}
    Region: ${var.aws_region}
    
    📊 Deployed Services:
    - ✅ AWS Lambda (Serverless Compute)
    - ✅ RDS Serverless V2 (PostgreSQL Database)
    - ✅ ElastiCache Serverless (Redis Cache)
    - ✅ S3 (Object Storage)
    - ✅ API Gateway (REST API)
    - ✅ CloudWatch (Monitoring & Logs)
    - ✅ Secrets Manager (API Keys)
    
    🔗 Access Information:
    - API Endpoint: ${module.api_gateway.api_endpoint}
    - CloudWatch Dashboard: https://console.aws.amazon.com/cloudwatch
    - Lambda Console: https://console.aws.amazon.com/lambda
    
    📋 Next Steps:
    1. Update application environment variables with output values
    2. Deploy Lambda code: ./scripts/deploy-lambda.sh
    3. Run database migrations: ./scripts/run-migrations.sh
    4. Test API endpoints
    5. Configure CloudWatch alarms
    
    💰 Estimated Monthly Cost: $43-90/month
    💎 Cost Savings: 99.6% vs Bloomberg/FactSet
    
    📚 Documentation: docs/deployment/AWS_DEPLOYMENT_GUIDE.md
  EOT
}