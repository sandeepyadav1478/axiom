# RDS Serverless V2 Module for Axiom Investment Banking Platform
# PostgreSQL database with Aurora Serverless V2 for cost optimization

resource "aws_db_subnet_group" "main" {
  name       = "${var.name_prefix}-db-subnet"
  subnet_ids = var.subnet_ids
  
  tags = merge(
    var.tags,
    {
      Name = "${var.name_prefix}-db-subnet-group"
    }
  )
}

resource "aws_rds_cluster_parameter_group" "main" {
  name        = "${var.name_prefix}-db-params"
  family      = "aurora-postgresql15"
  description = "Axiom Investment Banking PostgreSQL parameters"
  
  # Optimized for financial analytics workloads
  parameter {
    name  = "shared_preload_libraries"
    value = "pg_stat_statements"
  }
  
  parameter {
    name  = "log_statement"
    value = "all"
  }
  
  parameter {
    name  = "log_min_duration_statement"
    value = "1000" # Log queries >1s
  }
  
  tags = var.tags
}

resource "aws_rds_cluster" "main" {
  cluster_identifier      = "${var.name_prefix}-db-cluster"
  engine                  = "aurora-postgresql"
  engine_mode             = "provisioned" # Required for Serverless V2
  engine_version          = var.engine_version
  database_name           = var.database_name
  master_username         = var.master_username
  master_password         = var.master_password
  
  # Serverless V2 scaling configuration
  serverlessv2_scaling_configuration {
    min_capacity = var.min_capacity
    max_capacity = var.max_capacity
  }
  
  # Network configuration
  db_subnet_group_name            = aws_db_subnet_group.main.name
  vpc_security_group_ids          = [aws_security_group.main.id]
  db_cluster_parameter_group_name = aws_rds_cluster_parameter_group.main.name
  
  # Backup configuration
  backup_retention_period      = var.backup_retention_days
  preferred_backup_window      = var.preferred_backup_window
  preferred_maintenance_window = var.preferred_maintenance_window
  
  # Snapshot configuration
  skip_final_snapshot       = var.skip_final_snapshot
  final_snapshot_identifier = var.skip_final_snapshot ? null : "${var.name_prefix}-final-snapshot-${formatdate("YYYY-MM-DD-hhmm", timestamp())}"
  
  # Encryption
  storage_encrypted = true
  kms_key_id        = var.kms_key_id
  
  # Performance insights
  enabled_cloudwatch_logs_exports = ["postgresql"]
  
  # Deletion protection for production
  deletion_protection = var.environment == "production" ? true : false
  
  tags = merge(
    var.tags,
    {
      Name        = "${var.name_prefix}-db-cluster"
      Environment = var.environment
      Component   = "Database"
    }
  )
}

resource "aws_rds_cluster_instance" "main" {
  count = var.instance_count
  
  identifier         = "${var.name_prefix}-db-instance-${count.index + 1}"
  cluster_identifier = aws_rds_cluster.main.id
  instance_class     = "db.serverless" # Serverless V2 instance class
  engine             = aws_rds_cluster.main.engine
  engine_version     = aws_rds_cluster.main.engine_version
  
  # Performance Insights
  performance_insights_enabled    = var.enable_performance_insights
  performance_insights_kms_key_id = var.kms_key_id
  
  # Monitoring
  monitoring_interval = var.monitoring_interval
  monitoring_role_arn = var.monitoring_interval > 0 ? aws_iam_role.rds_monitoring[0].arn : null
  
  tags = merge(
    var.tags,
    {
      Name = "${var.name_prefix}-db-instance-${count.index + 1}"
    }
  )
}

# Security Group for RDS
resource "aws_security_group" "main" {
  name        = "${var.name_prefix}-rds-sg"
  description = "Security group for RDS PostgreSQL cluster"
  vpc_id      = var.vpc_id
  
  # Allow inbound PostgreSQL
  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidr_blocks
    description = "PostgreSQL access from allowed CIDRs"
  }
  
  # Allow all outbound
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow all outbound"
  }
  
  tags = merge(
    var.tags,
    {
      Name = "${var.name_prefix}-rds-sg"
    }
  )
}

# IAM role for enhanced monitoring
resource "aws_iam_role" "rds_monitoring" {
  count = var.monitoring_interval > 0 ? 1 : 0
  
  name = "${var.name_prefix}-rds-monitoring-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "monitoring.rds.amazonaws.com"
        }
      }
    ]
  })
  
  tags = var.tags
}

resource "aws_iam_role_policy_attachment" "rds_monitoring" {
  count = var.monitoring_interval > 0 ? 1 : 0
  
  role       = aws_iam_role.rds_monitoring[0].name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonRDSEnhancedMonitoringRole"
}

# IAM policy for Lambda access to RDS
resource "aws_iam_policy" "lambda_access" {
  name        = "${var.name_prefix}-rds-lambda-access"
  description = "Allow Lambda to access RDS"
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "rds:DescribeDBClusters",
          "rds:DescribeDBInstances",
          "rds:ListTagsForResource"
        ]
        Resource = [
          aws_rds_cluster.main.arn,
          "${aws_rds_cluster.main.arn}:*"
        ]
      }
    ]
  })
  
  tags = var.tags
}

# CloudWatch alarms for RDS
resource "aws_cloudwatch_metric_alarm" "rds_cpu" {
  count = var.enable_alarms ? 1 : 0
  
  alarm_name          = "${var.name_prefix}-rds-cpu-utilization"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "CPUUtilization"
  namespace           = "AWS/RDS"
  period              = "300"
  statistic           = "Average"
  threshold           = "80"
  alarm_description   = "RDS CPU utilization is too high"
  
  dimensions = {
    DBClusterIdentifier = aws_rds_cluster.main.cluster_identifier
  }
  
  tags = var.tags
}

resource "aws_cloudwatch_metric_alarm" "rds_connections" {
  count = var.enable_alarms ? 1 : 0
  
  alarm_name          = "${var.name_prefix}-rds-connections"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "DatabaseConnections"
  namespace           = "AWS/RDS"
  period              = "300"
  statistic           = "Average"
  threshold           = "80"
  alarm_description   = "RDS connection count is too high"
  
  dimensions = {
    DBClusterIdentifier = aws_rds_cluster.main.cluster_identifier
  }
  
  tags = var.tags
}