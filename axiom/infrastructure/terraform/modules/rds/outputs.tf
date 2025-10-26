# RDS Module Outputs

output "cluster_id" {
  description = "RDS cluster identifier"
  value       = aws_rds_cluster.main.cluster_identifier
}

output "cluster_arn" {
  description = "RDS cluster ARN"
  value       = aws_rds_cluster.main.arn
}

output "endpoint" {
  description = "RDS cluster endpoint (writer)"
  value       = aws_rds_cluster.main.endpoint
}

output "reader_endpoint" {
  description = "RDS cluster reader endpoint"
  value       = aws_rds_cluster.main.reader_endpoint
}

output "port" {
  description = "RDS cluster port"
  value       = aws_rds_cluster.main.port
}

output "database_name" {
  description = "Database name"
  value       = aws_rds_cluster.main.database_name
}

output "master_username" {
  description = "Master username"
  value       = aws_rds_cluster.main.master_username
  sensitive   = true
}

output "security_group_id" {
  description = "RDS security group ID"
  value       = aws_security_group.main.id
}

output "lambda_access_policy_arn" {
  description = "IAM policy ARN for Lambda access to RDS"
  value       = aws_iam_policy.lambda_access.arn
}