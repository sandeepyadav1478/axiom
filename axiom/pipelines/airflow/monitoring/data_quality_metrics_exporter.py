"""
Data Quality Metrics Exporter for Prometheus
Exposes data quality metrics for production monitoring

Metrics Tracked:
- Quality check pass/fail rates
- Data freshness
- Anomaly detection
- Schema validation results
- Record count trends
"""

from prometheus_client import Counter, Histogram, Gauge, start_http_server
from typing import Dict, Any, Optional
import psycopg2
import os
import time
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ================================================================
# Prometheus Metrics Definitions
# ================================================================

# Quality check metrics
quality_checks_total = Counter(
    'data_quality_checks_total',
    'Total quality checks executed',
    ['table_name', 'check_type', 'result']
)

quality_score = Gauge(
    'data_quality_score',
    'Current data quality score (0-100)',
    ['table_name']
)

quality_check_duration = Histogram(
    'data_quality_check_duration_seconds',
    'Quality check execution time',
    ['table_name', 'check_type'],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30]
)

# Data freshness metrics
data_freshness_minutes = Gauge(
    'data_freshness_minutes',
    'Minutes since last data update',
    ['table_name']
)

data_lag_seconds = Gauge(
    'data_lag_seconds',
    'Data lag in seconds',
    ['source', 'table_name']
)

# Anomaly detection metrics
anomalies_detected = Counter(
    'data_anomalies_detected_total',
    'Total anomalies detected',
    ['table_name', 'anomaly_type', 'severity']
)

anomaly_score = Gauge(
    'data_anomaly_score',
    'Current anomaly score (0-1)',
    ['table_name', 'metric']
)

# Schema validation metrics
schema_mismatches = Counter(
    'schema_validation_mismatches_total',
    'Schema validation mismatches',
    ['table_name', 'issue_type']
)

# Record count metrics
record_count = Gauge(
    'table_record_count',
    'Current record count',
    ['table_name']
)

record_count_change = Gauge(
    'table_record_count_change_24h',
    'Record count change in last 24 hours',
    ['table_name']
)

# Validation history metrics
validation_pass_rate = Gauge(
    'data_validation_pass_rate',
    'Validation pass rate (0-1)',
    ['table_name', 'time_window']
)

# Claude API cost metrics (integrated)
claude_cost_per_dag = Counter(
    'claude_api_cost_per_dag_usd',
    'Claude API cost per DAG run',
    ['dag_id', 'task_id']
)

claude_cost_daily = Gauge(
    'claude_api_cost_daily_usd',
    'Total Claude API cost today'
)

claude_cost_monthly = Gauge(
    'claude_api_cost_monthly_usd',
    'Total Claude API cost this month'
)


class DataQualityMetricsExporter:
    """
    Export data quality metrics to Prometheus.
    
    Monitors:
    - Quality check results
    - Data freshness
    - Anomaly detection
    - Schema validation
    - Claude API costs
    """
    
    def __init__(
        self,
        port: int = 9093,
        scrape_interval: int = 30,
        db_host: Optional[str] = None
    ):
        """Initialize metrics exporter."""
        self.port = port
        self.scrape_interval = scrape_interval
        self.db_host = db_host or os.getenv('POSTGRES_HOST', 'localhost')
        
        # Start metrics server
        try:
            start_http_server(port)
            logger.info(f"✅ Data quality metrics exporter started on port {port}")
        except Exception as e:
            logger.warning(f"Metrics server already running or port in use: {e}")
    
    def get_db_connection(self):
        """Get PostgreSQL connection."""
        return psycopg2.connect(
            host=self.db_host,
            user=os.getenv('POSTGRES_USER', 'axiom'),
            password=os.getenv('POSTGRES_PASSWORD'),
            database=os.getenv('POSTGRES_DB', 'axiom_finance')
        )
    
    def collect_quality_check_metrics(self):
        """Collect quality check results."""
        try:
            conn = self.get_db_connection()
            cur = conn.cursor()
            
            # Check if validation_history table exists
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'validation_history'
                )
            """)
            
            if not cur.fetchone()[0]:
                logger.debug("validation_history table not found, skipping")
                cur.close()
                conn.close()
                return
            
            # Get recent validation results
            cur.execute("""
                SELECT 
                    'price_data' as table_name,
                    records_checked,
                    records_passed,
                    records_failed,
                    window_start,
                    window_end
                FROM validation_history
                WHERE validation_time > NOW() - INTERVAL '1 hour'
                ORDER BY validation_time DESC
                LIMIT 1
            """)
            
            row = cur.fetchone()
            if row:
                table_name, total, passed, failed, start_time, end_time = row
                
                # Calculate quality score
                if total > 0:
                    score = (passed / total) * 100
                    quality_score.labels(table_name=table_name).set(score)
                    
                    # Update pass rate
                    pass_rate = passed / total
                    validation_pass_rate.labels(
                        table_name=table_name,
                        time_window='1h'
                    ).set(pass_rate)
            
            cur.close()
            conn.close()
            logger.debug("✅ Quality check metrics collected")
            
        except Exception as e:
            logger.error(f"❌ Failed to collect quality metrics: {e}")
    
    def collect_freshness_metrics(self):
        """Collect data freshness metrics."""
        try:
            conn = self.get_db_connection()
            cur = conn.cursor()
            
            # Check main tables for freshness
            tables = ['price_data', 'company_data', 'market_events']
            
            for table in tables:
                # Check if table exists
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = %s
                    )
                """, (table,))
                
                if not cur.fetchone()[0]:
                    continue
                
                # Get last update time
                cur.execute(f"""
                    SELECT 
                        EXTRACT(EPOCH FROM (NOW() - MAX(timestamp)))/60 as minutes_old
                    FROM {table}
                    WHERE timestamp IS NOT NULL
                """)
                
                row = cur.fetchone()
                if row and row[0] is not None:
                    minutes_old = row[0]
                    data_freshness_minutes.labels(table_name=table).set(minutes_old)
            
            cur.close()
            conn.close()
            logger.debug("✅ Freshness metrics collected")
            
        except Exception as e:
            logger.error(f"❌ Failed to collect freshness metrics: {e}")
    
    def collect_record_count_metrics(self):
        """Collect record count metrics."""
        try:
            conn = self.get_db_connection()
            cur = conn.cursor()
            
            tables = ['price_data', 'company_data', 'market_events']
            
            for table in tables:
                # Check if table exists
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = %s
                    )
                """, (table,))
                
                if not cur.fetchone()[0]:
                    continue
                
                # Current count
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                current_count = cur.fetchone()[0]
                record_count.labels(table_name=table).set(current_count)
                
                # Count from 24 hours ago
                cur.execute(f"""
                    SELECT COUNT(*) 
                    FROM {table}
                    WHERE timestamp < NOW() - INTERVAL '24 hours'
                """)
                old_count = cur.fetchone()[0]
                change = current_count - old_count
                record_count_change.labels(table_name=table).set(change)
            
            cur.close()
            conn.close()
            logger.debug("✅ Record count metrics collected")
            
        except Exception as e:
            logger.error(f"❌ Failed to collect record count metrics: {e}")
    
    def collect_claude_cost_metrics(self):
        """Collect Claude API cost metrics."""
        try:
            conn = self.get_db_connection()
            cur = conn.cursor()
            
            # Check if claude_usage_tracking table exists
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'claude_usage_tracking'
                )
            """)
            
            if not cur.fetchone()[0]:
                logger.debug("claude_usage_tracking table not found, skipping")
                cur.close()
                conn.close()
                return
            
            # Today's cost
            cur.execute("""
                SELECT COALESCE(SUM(cost_usd), 0)
                FROM claude_usage_tracking
                WHERE created_at::date = CURRENT_DATE
            """)
            daily_cost = cur.fetchone()[0]
            claude_cost_daily.set(float(daily_cost))
            
            # This month's cost
            cur.execute("""
                SELECT COALESCE(SUM(cost_usd), 0)
                FROM claude_usage_tracking
                WHERE DATE_TRUNC('month', created_at) = DATE_TRUNC('month', CURRENT_DATE)
            """)
            monthly_cost = cur.fetchone()[0]
            claude_cost_monthly.set(float(monthly_cost))
            
            # Cost per DAG (last 24 hours)
            cur.execute("""
                SELECT 
                    dag_id,
                    task_id,
                    SUM(cost_usd) as total_cost
                FROM claude_usage_tracking
                WHERE created_at > NOW() - INTERVAL '24 hours'
                GROUP BY dag_id, task_id
            """)
            
            for row in cur.fetchall():
                dag_id, task_id, cost = row
                claude_cost_per_dag.labels(
                    dag_id=dag_id,
                    task_id=task_id
                ).inc(float(cost))
            
            cur.close()
            conn.close()
            logger.debug("✅ Claude cost metrics collected")
            
        except Exception as e:
            logger.error(f"❌ Failed to collect Claude cost metrics: {e}")
    
    def collect_anomaly_metrics(self):
        """Collect anomaly detection metrics."""
        try:
            conn = self.get_db_connection()
            cur = conn.cursor()
            
            # Check for anomalies in price data (simple heuristic)
            cur.execute("""
                SELECT 
                    symbol,
                    COUNT(*) as anomaly_count
                FROM price_data
                WHERE timestamp > NOW() - INTERVAL '1 hour'
                    AND (
                        high < low 
                        OR close > high * 1.5 
                        OR close < low * 0.5
                        OR volume < 0
                    )
                GROUP BY symbol
            """)
            
            for row in cur.fetchall():
                symbol, count = row
                anomalies_detected.labels(
                    table_name='price_data',
                    anomaly_type='price_inconsistency',
                    severity='high'
                ).inc(count)
            
            cur.close()
            conn.close()
            logger.debug("✅ Anomaly metrics collected")
            
        except Exception as e:
            logger.error(f"❌ Failed to collect anomaly metrics: {e}")
    
    def collect_all_metrics(self):
        """Collect all metrics in one cycle."""
        logger.info("🔄 Collecting data quality metrics...")
        
        self.collect_quality_check_metrics()
        self.collect_freshness_metrics()
        self.collect_record_count_metrics()
        self.collect_claude_cost_metrics()
        self.collect_anomaly_metrics()
        
        logger.info("✅ Data quality metrics collection complete")
    
    def run_forever(self):
        """Run metrics collection loop."""
        logger.info(f"🚀 Starting continuous metrics collection (interval: {self.scrape_interval}s)")
        
        while True:
            try:
                self.collect_all_metrics()
                time.sleep(self.scrape_interval)
            except KeyboardInterrupt:
                logger.info("⏹️  Shutting down metrics exporter")
                break
            except Exception as e:
                logger.error(f"❌ Error in collection loop: {e}")
                time.sleep(self.scrape_interval)


def main():
    """Run the data quality metrics exporter."""
    exporter = DataQualityMetricsExporter(
        port=int(os.getenv('DATA_QUALITY_METRICS_PORT', '9093')),
        scrape_interval=int(os.getenv('METRICS_SCRAPE_INTERVAL', '30'))
    )
    
    exporter.run_forever()


if __name__ == '__main__':
    main()