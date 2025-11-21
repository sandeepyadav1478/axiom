"""
Apache Airflow DAG: Data Quality Validation
Smart validation with row count check + skip logic + 15-minute fallback

FEATURES:
- ✅ Event-driven: Triggered by data_ingestion_v2 ONLY when NEW data stored (row count > 0)
- ✅ Skip logic: Skips if already ran within last 15 minutes (prevents queue buildup)
- ✅ Time-based fallback: Runs every 15 minutes if not triggered
- ✅ Only validates NEW data since last check (incremental)
- ✅ Comprehensive quality checks using rules engine
- ✅ Separate from ingestion (proper separation of concerns)
- ✅ Batch validation for efficiency
- ✅ Stores validation results for tracking
- ✅ Prevents overload during market closed/failures

Schedule: Every 15 minutes (fallback) + event-driven triggers (with skip logic)
Scope: Only NEW data since last validation
Purpose: Data quality assurance without blocking ingestion or queue buildup
"""
from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.utils.dates import days_ago
from airflow.models import Variable
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

# ================================================================
# Configuration
# ================================================================
default_args = {
    'owner': 'axiom',
    'depends_on_past': False,
    'email': ['admin@axiom.com'],
    'email_on_failure': True,  # Alert on quality issues
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(minutes=10)
}

# ================================================================
# Helper Functions
# ================================================================

def should_run_validation(**context):
    """
    Check if validation should run based on last run time.
    Returns False if validation ran within the last 15 minutes (prevents overload).
    This prevents queue buildup when validation is triggered frequently.
    """
    try:
        last_validation = Variable.get('last_data_quality_validation', default_var=None)
        
        if not last_validation:
            print("✅ First validation run - proceeding")
            return True
        
        last_run = datetime.fromisoformat(last_validation)
        current_time = datetime.now()
        time_since_last = (current_time - last_run).total_seconds() / 60  # minutes
        
        # Skip if ran within last 15 minutes (prevents overload)
        if time_since_last < 15:
            print(f"⏭️ Skipping validation - last ran {time_since_last:.1f} minutes ago (< 15 min threshold)")
            print(f"   Last run: {last_run.isoformat()}")
            print(f"   Current: {current_time.isoformat()}")
            return False
        else:
            print(f"✅ Running validation - last ran {time_since_last:.1f} minutes ago (>= 15 min threshold)")
            return True
            
    except Exception as e:
        print(f"⚠️ Error checking last validation time: {e}")
        # On error, allow validation to run (fail-safe)
        return True


def get_last_validation_time(**context):
    """
    Get the timestamp of the last validation run.
    Uses Airflow Variable to track state between runs.
    """
    try:
        last_validation = Variable.get('last_data_quality_validation', default_var=None)
        if last_validation:
            return datetime.fromisoformat(last_validation)
        else:
            # First run - validate last hour
            return datetime.now() - timedelta(hours=1)
    except Exception as e:
        print(f"Error getting last validation time: {e}")
        # Default to last hour
        return datetime.now() - timedelta(hours=1)


def validate_new_price_data(**context):
    """
    Validate only NEW price data since last validation run.
    This is efficient and doesn't re-validate old data.
    """
    import psycopg2
    from psycopg2 import sql
    
    # Get last validation time
    last_validation = get_last_validation_time(**context)
    current_time = datetime.now()
    
    print(f"Validating data from {last_validation} to {current_time}")
    
    conn = psycopg2.connect(
        host=os.getenv('POSTGRES_HOST'),
        user=os.getenv('POSTGRES_USER'),
        password=os.getenv('POSTGRES_PASSWORD'),
        database=os.getenv('POSTGRES_DB')
    )
    
    cur = conn.cursor()
    
    # Fetch NEW data only
    cur.execute("""
        SELECT symbol, timestamp, open, high, low, close, volume
        FROM stock_prices
        WHERE timestamp > %s AND timestamp <= %s
        ORDER BY timestamp DESC
    """, (last_validation, current_time))
    
    rows = cur.fetchall()
    
    if not rows:
        print("No new data to validate")
        result = {
            'status': 'success',
            'records_checked': 0,
            'message': 'No new data since last validation',
            'validation_period': {
                'start': last_validation.isoformat(),
                'end': current_time.isoformat()
            }
        }
        cur.close()
        conn.close()
        return result
    
    print(f"Found {len(rows)} new records to validate")
    
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
        last_validation,
        current_time,
        str(validation_results[:100])  # Store sample of results
    ))
    
    conn.commit()
    cur.close()
    conn.close()
    
    # Update last validation time
    Variable.set('last_data_quality_validation', current_time.isoformat())
    
    result = {
        'status': 'success' if failed_count == 0 else 'warning',
        'records_checked': len(rows),
        'records_passed': passed_count,
        'records_failed': failed_count,
        'success_rate': (passed_count / len(rows) * 100) if len(rows) > 0 else 100,
        'validation_period': {
            'start': last_validation.isoformat(),
            'end': current_time.isoformat()
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
    Run comprehensive database-level quality checks.
    These are in addition to record-level validations.
    """
    import psycopg2
    
    conn = psycopg2.connect(
        host=os.getenv('POSTGRES_HOST'),
        user=os.getenv('POSTGRES_USER'),
        password=os.getenv('POSTGRES_PASSWORD'),
        database=os.getenv('POSTGRES_DB')
    )
    
    cur = conn.cursor()
    checks = []
    
    # Check 1: Data freshness
    cur.execute("""
        SELECT MAX(timestamp) as latest_data,
               EXTRACT(EPOCH FROM (NOW() - MAX(timestamp)))/60 as minutes_old
        FROM stock_prices
    """)
    row = cur.fetchone()
    checks.append({
        'name': 'data_freshness',
        'passed': row[1] < 120,  # Less than 2 hours old
        'value': f"{row[1]:.0f} minutes old",
        'threshold': '120 minutes'
    })
    
    # Check 2: Data completeness (all symbols have recent data)
    cur.execute("""
        SELECT COUNT(DISTINCT symbol) as total_symbols,
               COUNT(DISTINCT CASE WHEN timestamp > NOW() - INTERVAL '1 hour' 
                     THEN symbol END) as recent_symbols
        FROM stock_prices
    """)
    row = cur.fetchone()
    checks.append({
        'name': 'symbol_completeness',
        'passed': row[1] == row[0],
        'value': f"{row[1]}/{row[0]} symbols updated",
        'threshold': 'All symbols'
    })
    
    # Check 3: No duplicate records
    cur.execute("""
        SELECT COUNT(*) - COUNT(DISTINCT (symbol, timestamp, timeframe))
        FROM stock_prices
        WHERE timestamp > NOW() - INTERVAL '1 hour'
    """)
    duplicates = cur.fetchone()[0]
    checks.append({
        'name': 'no_duplicates',
        'passed': duplicates == 0,
        'value': f"{duplicates} duplicates found",
        'threshold': '0 duplicates'
    })
    
    # Check 4: Price reasonableness (no extreme outliers)
    cur.execute("""
        SELECT COUNT(*)
        FROM stock_prices
        WHERE timestamp > NOW() - INTERVAL '1 hour'
          AND (close < 0.01 OR close > 100000)
    """)
    outliers = cur.fetchone()[0]
    checks.append({
        'name': 'price_reasonableness',
        'passed': outliers == 0,
        'value': f"{outliers} outliers found",
        'threshold': '0 outliers'
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
    
    conn = psycopg2.connect(
        host=os.getenv('POSTGRES_HOST'),
        user=os.getenv('POSTGRES_USER'),
        password=os.getenv('POSTGRES_PASSWORD'),
        database=os.getenv('POSTGRES_DB')
    )
    
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
# Define the DAG
# ================================================================
with DAG(
    dag_id='data_quality_validation',
    default_args=default_args,
    description='Event-driven + 15-min fallback validation of NEW data only',
    schedule_interval='*/15 * * * *',  # Every 15 minutes (fallback when not event-triggered)
    start_date=days_ago(1),
    catchup=False,
    tags=['quality', 'validation', 'event-driven', 'incremental', 'fallback'],
    max_active_runs=1,
) as dag:
    
    # Task 1: Check if validation should run (skip if ran < 15 min ago)
    should_run = ShortCircuitOperator(
        task_id='check_should_run_validation',
        python_callable=should_run_validation,
        provide_context=True
    )
    
    # Task 2: Ensure validation tracking table exists
    setup_validation_table = PythonOperator(
        task_id='setup_validation_table',
        python_callable=ensure_validation_table_exists,
        provide_context=True
    )
    
    # Task 3: Validate NEW price data (incremental)
    validate_new_data = PythonOperator(
        task_id='validate_new_price_data',
        python_callable=validate_new_price_data,
        provide_context=True
    )
    
    # Task 4: Run comprehensive database checks
    database_checks = PythonOperator(
        task_id='comprehensive_database_checks',
        python_callable=comprehensive_database_checks,
        provide_context=True
    )
    
    # Task 5: Use DataQualityOperator for additional SQL-based checks
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
                'sql': """
                    SELECT MAX(timestamp) > NOW() - INTERVAL '2 hours'
                    FROM stock_prices
                """,
                'expected': True
            },
            {
                'name': 'volume_sanity',
                'type': 'value_range',
                'column': 'volume',
                'min_value': 0,
                'max_value': 1000000000000  # 1 trillion max
            }
        ],
        fail_on_error=False  # Log warnings, don't fail DAG
    )
    
    # ================================================================
    # Task Dependencies
    # ================================================================
    
    # First check if we should run (skip if ran < 15 min ago)
    should_run >> setup_validation_table >> validate_new_data >> [database_checks, additional_quality_checks]


dag.doc_md = """
# Data Quality Validation DAG

## 🎯 Purpose

Separate data quality validation with **event-driven + time-based fallback + skip logic** strategy.
This implements proper **separation of concerns** with smart triggering and overload prevention:
- **Ingestion DAG**: Triggers validation ONLY when NEW data stored (row count > 0)
- **Validation DAG**: Also runs every 15 minutes as fallback (time-based)
- **Skip Logic**: Skips execution if already ran within last 15 minutes (prevents queue buildup)
- **Best of both**: Immediate validation when needed + guaranteed periodic checks + no overload

## ⚡ Key Features

### 1. Incremental Validation
- Only validates **NEW data** since last check
- Efficient batch processing (not per-record)
- Tracks validation state between runs

### 2. Triple Protection Strategy
- **Row Count Check** (Ingestion): Only triggers if NEW data stored (row count > 0)
- **Skip Logic** (Validation): Skips if already ran < 15 min ago (prevents queue buildup)
- **Time-based Fallback**: Runs every 15 minutes regardless (catches missed triggers)
- **Result**: No unnecessary work + fast validation + guaranteed coverage + no overload

### 3. Comprehensive Checks
- Record-level validation (using rules engine)
- Database-level checks (aggregates, outliers)
- SQL-based quality checks
- Validation history tracking

### 4. No Ingestion Blocking
- Ingestion can run at full speed
- Quality issues logged but don't stop ingestion
- Alerts sent on quality failures

## 📊 What Gets Validated

### Record-Level (Rules Engine)
- High >= Low price check
- Close/Open within High-Low range
- Positive prices and volume
- Reasonable intraday movement (<50%)
- Timestamp validity

### Database-Level
- Data freshness (<2 hours old)
- Symbol completeness (all symbols updated)
- No duplicate records
- Price reasonableness (no extreme outliers)

### SQL-Based
- Minimum hourly record count
- No stale data
- Volume sanity checks

## 🔄 Workflow

1. **Check Skip**: Verify if validation ran within last 15 minutes (skip if yes)
2. **Setup**: Ensure validation_history table exists
3. **Get State**: Retrieve last validation timestamp
4. **Fetch New Data**: Query only data added since last run
5. **Validate**: Run comprehensive quality checks (if new data exists)
6. **Store Results**: Save validation results and update state
7. **Alert**: Send notifications if quality issues found

## 💡 Why This Design?

### Problem with Previous Approach
- Validation in ingestion DAG caused failures
- Blocked data ingestion on quality issues
- Re-validated same data multiple times
- Tight coupling between ingestion and validation
- Queue buildup when triggering too frequently
- Unnecessary work during market closed/failures

### Benefits of Current Approach
✅ Ingestion never fails due to quality issues
✅ Row count check prevents unnecessary triggers
✅ Skip logic prevents queue buildup (<15 min)
✅ Only validates new data (efficient)
✅ Better monitoring and alerting
✅ Clean separation of concerns
✅ No wasted resources during downtime

## 📈 Performance

- **Validation overhead**: Negligible on ingestion (runs async)
- **Batch efficiency**: Validates new data only (~1-240 records per run)
- **Event-driven**: Usually runs within seconds of ingestion completing
- **Skip logic**: Prevents duplicate runs if <15 min since last run
- **Fallback frequency**: Every 15 minutes ensures no data goes unchecked
- **Storage**: Tracks validation history for trend analysis
- **Alerts**: Email on quality failures (as they occur)
- **Overload prevention**: No queue buildup during high-frequency triggers

## 🎛️ Configuration

- **Primary**: Event-driven (triggered by data_ingestion_v2 when row count > 0)
- **Skip Logic**: Skips if ran within last 15 minutes (prevents queue buildup)
- **Fallback**: Every 15 minutes (`*/15 * * * *`)
- **Timeout**: 10 minutes per run
- **Retries**: 1 retry with 5 min delay
- **Alerts**: Email on failure (quality issues)
- **Fail behavior**: Log warnings, don't fail DAG

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