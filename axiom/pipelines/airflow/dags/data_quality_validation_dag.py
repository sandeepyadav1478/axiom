"""
Apache Airflow DAG: Data Quality Validation
BATCH VALIDATION: Processes 5-minute windows efficiently

FEATURES:
- ✅ Batch processing: Validates 5-minute windows instead of per-trigger
- ✅ Scheduled only: Runs every 5 minutes independently
- ✅ Centralized config: All thresholds configurable via YAML
- ✅ Comprehensive checks: Record-level, database-level, SQL-based
- ✅ Separate from ingestion: Complete decoupling
- ✅ Efficient validation: Single batch vs multiple triggers
- ✅ Stores validation results: Full tracking and trending

Schedule: Every 5 minutes (configurable via YAML)
Scope: Validates all data in the last 5-minute window
Purpose: Efficient batch quality assurance
"""
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, '/opt/airflow')

from dotenv import load_dotenv
load_dotenv('/opt/airflow/.env')

# Import operators from local path
operators_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, operators_path)

from operators.quality_check_operator import DataQualityOperator

# Import centralized configuration
utils_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, utils_path)
from utils.config_loader import (
    dag_config,
    build_postgres_conn_params
)

# ================================================================
# Load Configuration from YAML
# ================================================================
DAG_NAME = 'data_quality_validation'
config = dag_config.get_dag_config(DAG_NAME)
default_args = dag_config.get_default_args(DAG_NAME)
batch_config = dag_config.get_batch_config()
thresholds = dag_config.get_validation_thresholds()

# ================================================================
# Helper Functions
# ================================================================

def validate_batch_window(**context):
    """
    Validate all data in the last 5-minute window (batch processing).
    More efficient than per-trigger validation.
    """
    import psycopg2
    
    # Get batch window from config
    window_minutes = batch_config.get('window_minutes', 5)
    min_records = batch_config.get('min_records_to_validate', 1)
    
    current_time = datetime.now()
    window_start = current_time - timedelta(minutes=window_minutes)
    
    print(f"📊 Validating {window_minutes}-minute batch window:")
    print(f"   Window: {window_start} to {current_time}")
    
    # Use centralized config for DB connection
    db_params = build_postgres_conn_params()
    conn = psycopg2.connect(**db_params)
    
    cur = conn.cursor()
    
    # Fetch all data in the batch window
    cur.execute("""
        SELECT symbol, timestamp, open, high, low, close, volume
        FROM stock_prices
        WHERE timestamp >= %s AND timestamp < %s
        ORDER BY timestamp DESC
    """, (window_start, current_time))
    
    rows = cur.fetchall()
    
    if not rows or len(rows) < min_records:
        print(f"⏭️ Skipping validation - only {len(rows) if rows else 0} records in window (min: {min_records})")
        result = {
            'status': 'skipped',
            'records_checked': 0,
            'message': f'Insufficient data in {window_minutes}-minute window',
            'validation_window': {
                'start': window_start.isoformat(),
                'end': current_time.isoformat(),
                'window_minutes': window_minutes
            }
        }
        cur.close()
        conn.close()
        return result
    
    print(f"✅ Found {len(rows)} records to validate in {window_minutes}-minute window")
    
    # Import validation engine
    sys.path.insert(0, '/opt/airflow/axiom')
    from data_quality.validation.rules_engine import get_validation_engine
    
    engine = get_validation_engine()
    
    # Validate each record
    validation_results = []
    passed_count = 0
    failed_count = 0
    
    for row in rows:
        price_data = {
            'symbol': row[0],
            'timestamp': row[1],
            'open': row[2],
            'high': row[3],
            'low': row[4],
            'close': row[5],
            'volume': row[6]
        }
        
        try:
            results = engine.validate_data(
                price_data, 
                "price_data", 
                raise_on_critical=False  # Collect all issues
            )
            
            record_passed = all(r.passed for r in results)
            
            if record_passed:
                passed_count += 1
            else:
                failed_count += 1
                # Log failures
                failures = [r for r in results if not r.passed]
                for f in failures:
                    print(f"❌ Validation failure for {price_data['symbol']} at {price_data['timestamp']}")
                    print(f"   Rule: {f.rule_name} ({f.severity.value})")
                    print(f"   Message: {f.error_message}")
            
            validation_results.extend([r.to_dict() for r in results])
            
        except Exception as e:
            failed_count += 1
            print(f"Error validating record {price_data['symbol']}: {e}")
    
    # Store validation results in database
    cur.execute("""
        INSERT INTO validation_history (
            validation_run_time,
            records_checked,
            records_passed,
            records_failed,
            validation_period_start,
            validation_period_end,
            details
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (
        current_time,
        len(rows),
        passed_count,
        failed_count,
        window_start,
        current_time,
        str(validation_results[:100])  # Store sample of results
    ))
    
    conn.commit()
    cur.close()
    conn.close()
    
    result = {
        'status': 'success' if failed_count == 0 else 'warning',
        'records_checked': len(rows),
        'records_passed': passed_count,
        'records_failed': failed_count,
        'success_rate': (passed_count / len(rows) * 100) if len(rows) > 0 else 100,
        'validation_window': {
            'start': window_start.isoformat(),
            'end': current_time.isoformat(),
            'window_minutes': window_minutes
        }
    }
    
    print(f"\n📊 Validation Summary:")
    print(f"   Records Checked: {result['records_checked']}")
    print(f"   Passed: {result['records_passed']}")
    print(f"   Failed: {result['records_failed']}")
    print(f"   Success Rate: {result['success_rate']:.1f}%")
    
    return result


def comprehensive_database_checks(**context):
    """
    Run comprehensive database-level quality checks using config thresholds.
    """
    import psycopg2
    
    # Use centralized config for DB connection
    db_params = build_postgres_conn_params()
    conn = psycopg2.connect(**db_params)
    
    cur = conn.cursor()
    checks = []
    
    # Get thresholds from config
    freshness_threshold = thresholds.get('data_freshness_minutes', 120)
    min_symbols = thresholds.get('min_symbols_with_recent_data', 10)
    max_duplicates = thresholds.get('max_duplicate_tolerance', 0)
    price_min = thresholds.get('price_min', 0.01)
    price_max = thresholds.get('price_max', 100000)
    
    # Check 1: Data freshness (configurable)
    cur.execute("""
        SELECT MAX(timestamp) as latest_data,
               EXTRACT(EPOCH FROM (NOW() - MAX(timestamp)))/60 as minutes_old
        FROM stock_prices
    """)
    row = cur.fetchone()
    checks.append({
        'name': 'data_freshness',
        'passed': row[1] < freshness_threshold,
        'value': f"{row[1]:.0f} minutes old",
        'threshold': f'{freshness_threshold} minutes'
    })
    
    # Check 2: Symbol completeness (configurable threshold)
    cur.execute("""
        SELECT COUNT(DISTINCT symbol) as total_symbols,
               COUNT(DISTINCT CASE WHEN timestamp > NOW() - INTERVAL '1 hour'
                     THEN symbol END) as recent_symbols
        FROM stock_prices
    """)
    row = cur.fetchone()
    checks.append({
        'name': 'symbol_completeness',
        'passed': row[1] >= min_symbols,
        'value': f"{row[1]}/{row[0]} symbols updated",
        'threshold': f'At least {min_symbols} symbols'
    })
    
    # Check 3: No duplicate records (configurable tolerance)
    cur.execute("""
        SELECT COUNT(*) - COUNT(DISTINCT (symbol, timestamp, timeframe))
        FROM stock_prices
        WHERE timestamp > NOW() - INTERVAL '1 hour'
    """)
    duplicates = cur.fetchone()[0]
    checks.append({
        'name': 'no_duplicates',
        'passed': duplicates <= max_duplicates,
        'value': f"{duplicates} duplicates found",
        'threshold': f'{max_duplicates} duplicates max'
    })
    
    # Check 4: Price reasonableness (configurable range)
    cur.execute("""
        SELECT COUNT(*)
        FROM stock_prices
        WHERE timestamp > NOW() - INTERVAL '1 hour'
          AND (close < %s OR close > %s)
    """, (price_min, price_max))
    outliers = cur.fetchone()[0]
    checks.append({
        'name': 'price_reasonableness',
        'passed': outliers == 0,
        'value': f"{outliers} outliers found",
        'threshold': f'Price between ${price_min} and ${price_max}'
    })
    
    cur.close()
    conn.close()
    
    # Summary
    passed = sum(1 for c in checks if c['passed'])
    total = len(checks)
    
    print(f"\n📋 Comprehensive Database Checks:")
    for check in checks:
        status = "✅" if check['passed'] else "❌"
        print(f"   {status} {check['name']}: {check['value']} (threshold: {check['threshold']})")
    
    return {
        'checks': checks,
        'total': total,
        'passed': passed,
        'failed': total - passed
    }


def ensure_validation_table_exists(**context):
    """
    Ensure the validation_history table exists for tracking results.
    """
    import psycopg2
    
    # Use centralized config for DB connection
    db_params = build_postgres_conn_params()
    conn = psycopg2.connect(**db_params)
    
    cur = conn.cursor()
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS validation_history (
            id SERIAL PRIMARY KEY,
            validation_run_time TIMESTAMP NOT NULL,
            records_checked INTEGER NOT NULL,
            records_passed INTEGER NOT NULL,
            records_failed INTEGER NOT NULL,
            validation_period_start TIMESTAMP,
            validation_period_end TIMESTAMP,
            details TEXT,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    
    # Create index for quick lookups
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_validation_run_time 
        ON validation_history(validation_run_time DESC)
    """)
    
    conn.commit()
    cur.close()
    conn.close()
    
    print("✅ Validation history table ready")
    return {'status': 'success'}


# ================================================================
# Define the DAG (Config-Driven, Batch Processing)
# ================================================================
with DAG(
    dag_id=config.get('dag_id', 'data_quality_validation'),
    default_args=default_args,
    description=config.get('description', 'Batch validation of 5-minute windows'),
    schedule_interval=config.get('schedule_interval', '*/5 * * * *'),  # Every 5 minutes
    start_date=days_ago(1),
    catchup=dag_config.get_global('catchup', False),
    tags=config.get('tags', ['quality', 'validation', 'batch']),
    max_active_runs=dag_config.get_global('max_active_runs', 1),
) as dag:
    
    # Task 1: Ensure validation tracking table exists
    setup_validation_table = PythonOperator(
        task_id='setup_validation_table',
        python_callable=ensure_validation_table_exists,
        provide_context=True
    )
    
    # Task 2: Validate batch window (5-minute window)
    validate_batch = PythonOperator(
        task_id='validate_batch_window',
        python_callable=validate_batch_window,
        provide_context=True
    )
    
    # Task 3: Run comprehensive database checks
    database_checks = PythonOperator(
        task_id='comprehensive_database_checks',
        python_callable=comprehensive_database_checks,
        provide_context=True
    )
    
    # Task 4: Use DataQualityOperator for additional SQL-based checks (from config)
    volume_max = thresholds.get('volume_max', 1000000000000)
    
    additional_quality_checks = DataQualityOperator(
        task_id='additional_quality_checks',
        table_name='stock_prices',
        checks=[
            {
                'name': 'hourly_data_count',
                'type': 'custom_sql',
                'sql': """
                    SELECT COUNT(*) > 10
                    FROM stock_prices
                    WHERE timestamp > NOW() - INTERVAL '1 hour'
                """,
                'expected': True
            },
            {
                'name': 'no_stale_data',
                'type': 'custom_sql',
                'sql': f"""
                    SELECT MAX(timestamp) > NOW() - INTERVAL '{thresholds.get('data_freshness_minutes', 120)} minutes'
                    FROM stock_prices
                """,
                'expected': True
            },
            {
                'name': 'volume_sanity',
                'type': 'value_range',
                'column': 'volume',
                'min_value': 0,
                'max_value': volume_max
            }
        ],
        fail_on_error=config.get('alerts', {}).get('fail_on_error', False)
    )
    
    # ================================================================
    # Task Dependencies (Simplified Batch Flow)
    # ================================================================
    
    setup_validation_table >> validate_batch >> [database_checks, additional_quality_checks]


dag.doc_md = """
# Data Quality Validation DAG (Batch Processing)

## 🎯 Purpose

Efficient batch validation processing **5-minute windows** independently:
- **NO per-trigger validation**: Completely decoupled from ingestion
- **Scheduled only**: Runs every 5 minutes on its own schedule
- **Batch windows**: Validates all data in 5-minute windows
- **Centralized config**: All thresholds and parameters in YAML

## ⚡ Key Features

### 1. Batch Window Processing
- Validates **5-minute windows** of data
- More efficient than per-record or per-trigger
- Configurable window size via YAML
- Skips validation if insufficient data

### 2. Independent Scheduling
- **NO triggers from ingestion**: Completely independent
- **Scheduled batches**: Runs every 5 minutes (configurable)
- **No queue buildup**: Simple scheduled execution
- **Result**: Clean, predictable, efficient validation

### 3. Comprehensive Checks
- Record-level validation (using rules engine)
- Database-level checks (aggregates, outliers)
- SQL-based quality checks
- All thresholds configurable via YAML

### 4. Centralized Configuration
- All schedules in YAML
- All thresholds in YAML
- All parameters in YAML
- Easy to tune without code changes

## 📊 What Gets Validated

### Record-Level (Rules Engine)
- High >= Low price check
- Close/Open within High-Low range
- Positive prices and volume
- Reasonable intraday movement (configurable %)
- Timestamp validity

### Database-Level (All Configurable)
- Data freshness (configurable minutes)
- Symbol completeness (min symbols)
- Duplicate tolerance (configurable)
- Price reasonableness (min/max prices)

### SQL-Based (Config-Driven)
- Minimum hourly record count
- No stale data (configurable threshold)
- Volume sanity checks (configurable max)

## 🔄 Workflow (Simplified)

1. **Setup**: Ensure validation_history table exists
2. **Fetch Window**: Query all data in last 5-minute window
3. **Validate**: Run comprehensive quality checks on batch
4. **Store Results**: Save validation results with window info
5. **Alert**: Send notifications if quality issues found

## 💡 Why Batch Windows?

### Problem with Per-Trigger Approach
- Complex trigger logic
- Queue buildup issues
- Tight coupling with ingestion
- Repeated validation overhead
- State management complexity

### Benefits of Batch Windows
✅ Simple scheduled execution
✅ Complete decoupling from ingestion
✅ More efficient (single batch vs multiple triggers)
✅ Predictable resource usage
✅ Easy to configure and tune
✅ No queue management needed
✅ Clean separation of concerns

## 📈 Performance

- **Batch efficiency**: Validates 5-minute windows (~5-300 records per batch)
- **Scheduled execution**: Runs every 5 minutes (configurable)
- **No blocking**: Ingestion unaffected by validation
- **Configurable**: Window size, thresholds, schedules all in YAML
- **Storage**: Tracks validation history for trend analysis
- **Alerts**: Configurable alert behavior
- **Predictable**: Simple scheduled execution, no trigger complexity

## 🎛️ Configuration (All in YAML)

- **Schedule**: Every 5 minutes (`*/5 * * * *`) - configurable
- **Window size**: 5 minutes - configurable
- **Thresholds**: All validation thresholds in YAML
- **Timeout**: Configurable per DAG
- **Retries**: Configurable retry logic
- **Alerts**: Configurable alert behavior
- **Fail behavior**: Configurable (default: log warnings)

## 📊 Monitoring

Validation results stored in `validation_history` table:
- Validation run timestamp
- Records checked/passed/failed
- Validation period (start/end)
- Detailed results (sample)

Query validation trends:
```sql
SELECT 
    validation_run_time,
    records_checked,
    records_passed,
    ROUND(records_passed::numeric / records_checked * 100, 2) as success_rate
FROM validation_history
ORDER BY validation_run_time DESC
LIMIT 24;  -- Last 24 hours
```
"""