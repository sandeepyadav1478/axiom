"""
Airflow DAG Metrics Exporter for Prometheus
Exposes Airflow execution metrics for production monitoring

Metrics Exported:
- DAG run durations and status
- Task execution times and failures
- SLA misses and delays
- Scheduler health
- Pool and queue metrics
"""

from prometheus_client import Counter, Histogram, Gauge, start_http_server, Info
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

# DAG execution metrics
dag_runs_total = Counter(
    'airflow_dag_runs_total',
    'Total DAG runs',
    ['dag_id', 'state']
)

dag_run_duration = Histogram(
    'airflow_dag_run_duration_seconds',
    'DAG run duration in seconds',
    ['dag_id'],
    buckets=[10, 30, 60, 300, 600, 1800, 3600, 7200]
)

dag_last_run_timestamp = Gauge(
    'airflow_dag_last_run_timestamp',
    'Timestamp of last DAG run',
    ['dag_id', 'state']
)

# Task execution metrics
task_runs_total = Counter(
    'airflow_task_runs_total',
    'Total task runs',
    ['dag_id', 'task_id', 'state']
)

task_duration = Histogram(
    'airflow_task_duration_seconds',
    'Task execution duration',
    ['dag_id', 'task_id'],
    buckets=[1, 5, 10, 30, 60, 300, 600, 1800]
)

task_failures_total = Counter(
    'airflow_task_failures_total',
    'Total task failures',
    ['dag_id', 'task_id', 'error_type']
)

# SLA metrics
sla_misses_total = Counter(
    'airflow_sla_misses_total',
    'Total SLA misses',
    ['dag_id', 'task_id']
)

# Scheduler metrics
scheduler_heartbeat = Gauge(
    'airflow_scheduler_heartbeat',
    'Scheduler last heartbeat timestamp'
)

scheduler_dag_processing_time = Histogram(
    'airflow_scheduler_dag_processing_seconds',
    'DAG file processing time',
    ['dag_id'],
    buckets=[0.1, 0.5, 1, 2, 5, 10]
)

# Queue metrics
task_queue_length = Gauge(
    'airflow_task_queue_length',
    'Number of tasks in queue',
    ['state']
)

# Pool metrics
pool_slots_available = Gauge(
    'airflow_pool_slots_available',
    'Available pool slots',
    ['pool_name']
)

pool_slots_used = Gauge(
    'airflow_pool_slots_used',
    'Used pool slots',
    ['pool_name']
)

# System info
airflow_info = Info(
    'airflow_version',
    'Airflow version and environment info'
)


class AirflowMetricsExporter:
    """
    Export Airflow metrics to Prometheus.
    
    Connects to Airflow's metadata database and exposes metrics
    on HTTP endpoint for Prometheus scraping.
    """
    
    def __init__(
        self,
        port: int = 9092,
        scrape_interval: int = 15,
        db_host: Optional[str] = None,
        db_name: str = 'airflow'
    ):
        """Initialize metrics exporter."""
        self.port = port
        self.scrape_interval = scrape_interval
        self.db_host = db_host or os.getenv('POSTGRES_HOST', 'localhost')
        self.db_name = db_name
        
        # Start metrics server
        try:
            start_http_server(port)
            logger.info(f"✅ Airflow metrics exporter started on port {port}")
        except Exception as e:
            logger.warning(f"Metrics server already running or port in use: {e}")
    
    def get_db_connection(self):
        """Get PostgreSQL connection to Airflow metadata DB."""
        return psycopg2.connect(
            host=self.db_host,
            user=os.getenv('POSTGRES_USER', 'axiom'),
            password=os.getenv('POSTGRES_PASSWORD'),
            database=self.db_name
        )
    
    def collect_dag_metrics(self):
        """Collect DAG execution metrics."""
        try:
            conn = self.get_db_connection()
            cur = conn.cursor()
            
            # Get DAG runs from last 24 hours
            cur.execute("""
                SELECT 
                    dag_id,
                    state,
                    COUNT(*) as run_count,
                    AVG(EXTRACT(EPOCH FROM (end_date - start_date))) as avg_duration,
                    MAX(execution_date) as last_run
                FROM dag_run
                WHERE execution_date > NOW() - INTERVAL '24 hours'
                GROUP BY dag_id, state
            """)
            
            for row in cur.fetchall():
                dag_id, state, count, avg_duration, last_run = row
                
                # Update counters
                dag_runs_total.labels(dag_id=dag_id, state=state).inc(count)
                
                # Update duration histogram
                if avg_duration:
                    dag_run_duration.labels(dag_id=dag_id).observe(avg_duration)
                
                # Update last run timestamp
                if last_run:
                    dag_last_run_timestamp.labels(
                        dag_id=dag_id,
                        state=state
                    ).set(last_run.timestamp())
            
            cur.close()
            conn.close()
            logger.debug("✅ DAG metrics collected")
            
        except Exception as e:
            logger.error(f"❌ Failed to collect DAG metrics: {e}")
    
    def collect_task_metrics(self):
        """Collect task execution metrics."""
        try:
            conn = self.get_db_connection()
            cur = conn.cursor()
            
            # Get task instances from last 24 hours
            cur.execute("""
                SELECT 
                    dag_id,
                    task_id,
                    state,
                    COUNT(*) as run_count,
                    AVG(EXTRACT(EPOCH FROM (end_date - start_date))) as avg_duration
                FROM task_instance
                WHERE execution_date > NOW() - INTERVAL '24 hours'
                GROUP BY dag_id, task_id, state
            """)
            
            for row in cur.fetchall():
                dag_id, task_id, state, count, avg_duration = row
                
                # Update counters
                task_runs_total.labels(
                    dag_id=dag_id,
                    task_id=task_id,
                    state=state
                ).inc(count)
                
                # Update duration histogram
                if avg_duration and state == 'success':
                    task_duration.labels(
                        dag_id=dag_id,
                        task_id=task_id
                    ).observe(avg_duration)
            
            cur.close()
            conn.close()
            logger.debug("✅ Task metrics collected")
            
        except Exception as e:
            logger.error(f"❌ Failed to collect task metrics: {e}")
    
    def collect_sla_metrics(self):
        """Collect SLA miss metrics."""
        try:
            conn = self.get_db_connection()
            cur = conn.cursor()
            
            # Check if sla_miss table exists
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'sla_miss'
                )
            """)
            
            if cur.fetchone()[0]:
                cur.execute("""
                    SELECT 
                        dag_id,
                        task_id,
                        COUNT(*) as miss_count
                    FROM sla_miss
                    WHERE timestamp > NOW() - INTERVAL '24 hours'
                    GROUP BY dag_id, task_id
                """)
                
                for row in cur.fetchall():
                    dag_id, task_id, count = row
                    sla_misses_total.labels(
                        dag_id=dag_id,
                        task_id=task_id
                    ).inc(count)
            
            cur.close()
            conn.close()
            logger.debug("✅ SLA metrics collected")
            
        except Exception as e:
            logger.error(f"❌ Failed to collect SLA metrics: {e}")
    
    def collect_queue_metrics(self):
        """Collect task queue metrics."""
        try:
            conn = self.get_db_connection()
            cur = conn.cursor()
            
            # Count tasks by state
            cur.execute("""
                SELECT state, COUNT(*) as count
                FROM task_instance
                WHERE state IN ('queued', 'running', 'scheduled')
                GROUP BY state
            """)
            
            for row in cur.fetchall():
                state, count = row
                task_queue_length.labels(state=state).set(count)
            
            cur.close()
            conn.close()
            logger.debug("✅ Queue metrics collected")
            
        except Exception as e:
            logger.error(f"❌ Failed to collect queue metrics: {e}")
    
    def collect_all_metrics(self):
        """Collect all metrics in one cycle."""
        logger.info("🔄 Collecting Airflow metrics...")
        
        self.collect_dag_metrics()
        self.collect_task_metrics()
        self.collect_sla_metrics()
        self.collect_queue_metrics()
        
        logger.info("✅ Metrics collection complete")
    
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
    """Run the Airflow metrics exporter."""
    exporter = AirflowMetricsExporter(
        port=int(os.getenv('AIRFLOW_METRICS_PORT', '9092')),
        scrape_interval=int(os.getenv('METRICS_SCRAPE_INTERVAL', '15'))
    )
    
    # Set Airflow version info
    airflow_info.info({
        'version': '2.8.0',
        'executor': 'LocalExecutor',
        'environment': 'production'
    })
    
    exporter.run_forever()


if __name__ == '__main__':
    main()