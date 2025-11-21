"""
Apache Airflow DAG: Data Quality Validation (Simplified)

SIMPLE VALIDATION: Runs every 5 minutes, validates last 5 min of data.
No complex config, no skip logic, no fancy operators.
Make it work first, optimize later.
"""
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
import psycopg2
import os


def get_db_connection():
    """Get PostgreSQL connection using environment variables."""
    return psycopg2.connect(
        host=os.getenv('POSTGRES_HOST', 'localhost'),
        port=os.getenv('POSTGRES_PORT', '5432'),
        dbname=os.getenv('POSTGRES_DB', 'axiom'),
        user=os.getenv('POSTGRES_USER', 'axiom_user'),
        password=os.getenv('POSTGRES_PASSWORD', 'axiom_pass')
    )


def validate_last_5_minutes(**context):
    """
    Simple validation: Check last 5 minutes of data.
    No skip logic, no complex windowing, just validate.
    """
    current_time = datetime.now()
    window_start = current_time - timedelta(minutes=5)
    
    print(f"📊 Validating last 5 minutes of data:")
    print(f"   Window: {window_start} to {current_time}")
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Get all data from last 5 minutes
    cur.execute("""
        SELECT symbol, timestamp, open, high, low, close, volume
        FROM price_data
        WHERE timestamp >= %s AND timestamp < %s
        ORDER BY timestamp DESC
    """, (window_start, current_time))
    
    rows = cur.fetchall()
    record_count = len(rows)
    
    print(f"✅ Found {record_count} records in last 5 minutes")
    
    # Simple validation checks
    passed = 0
    failed = 0
    issues = []
    
    for row in rows:
        symbol, timestamp, open_p, high, low, close, volume = row
        record_ok = True
        
        # Check 1: High >= Low
        if high < low:
            issues.append(f"{symbol} at {timestamp}: High < Low")
            record_ok = False
        
        # Check 2: Close and Open within High-Low range
        if close > high or close < low:
            issues.append(f"{symbol} at {timestamp}: Close outside High-Low range")
            record_ok = False
        
        if open_p > high or open_p < low:
            issues.append(f"{symbol} at {timestamp}: Open outside High-Low range")
            record_ok = False
        
        # Check 3: Positive prices and volume
        if high <= 0 or low <= 0 or close <= 0 or open_p <= 0:
            issues.append(f"{symbol} at {timestamp}: Non-positive prices")
            record_ok = False
        
        if volume < 0:
            issues.append(f"{symbol} at {timestamp}: Negative volume")
            record_ok = False
        
        if record_ok:
            passed += 1
        else:
            failed += 1
    
    # Print issues if any
    if issues:
        print(f"\n❌ Found {len(issues)} validation issues:")
        for issue in issues[:10]:  # Print first 10
            print(f"   - {issue}")
        if len(issues) > 10:
            print(f"   ... and {len(issues) - 10} more")
    
    # Store results in simple validation_history table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS validation_history (
            id SERIAL PRIMARY KEY,
            validation_time TIMESTAMP NOT NULL,
            records_checked INTEGER NOT NULL,
            records_passed INTEGER NOT NULL,
            records_failed INTEGER NOT NULL,
            window_start TIMESTAMP,
            window_end TIMESTAMP,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    
    cur.execute("""
        INSERT INTO validation_history (
            validation_run_time, records_checked, records_passed, records_failed,
            window_start, window_end
        ) VALUES (%s, %s, %s, %s, %s, %s)
    """, (current_time, record_count, passed, failed, window_start, current_time))
    
    conn.commit()
    cur.close()
    conn.close()
    
    # Print summary
    success_rate = (passed / record_count * 100) if record_count > 0 else 100
    print(f"\n📊 Validation Summary:")
    print(f"   Records: {record_count}")
    print(f"   Passed: {passed}")
    print(f"   Failed: {failed}")
    print(f"   Success Rate: {success_rate:.1f}%")
    
    return {
        'records': record_count,
        'passed': passed,
        'failed': failed,
        'success_rate': success_rate
    }


def check_data_freshness(**context):
    """Simple check: Is data recent?"""
    conn = get_db_connection()
    cur = conn.cursor()
    
    cur.execute("""
        SELECT MAX(timestamp) as latest_data,
               EXTRACT(EPOCH FROM (NOW() - MAX(timestamp)))/60 as minutes_old
        FROM price_data
    """)
    
    row = cur.fetchone()
    latest_data = row[0]
    minutes_old = row[1] if row[1] else 999999
    
    cur.close()
    conn.close()
    
    is_fresh = minutes_old < 120  # Data should be less than 2 hours old
    
    status = "✅" if is_fresh else "❌"
    print(f"\n{status} Data Freshness Check:")
    print(f"   Latest data: {latest_data}")
    print(f"   Age: {minutes_old:.0f} minutes")
    print(f"   Status: {'FRESH' if is_fresh else 'STALE'}")
    
    return {
        'latest_data': str(latest_data),
        'minutes_old': minutes_old,
        'is_fresh': is_fresh
    }


# Define the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

with DAG(
    dag_id='data_quality_validation',
    default_args=default_args,
    description='Simple validation every 5 minutes',
    schedule_interval='*/5 * * * *',  # Every 5 minutes
    start_date=days_ago(1),
    catchup=False,
    tags=['quality', 'validation', 'simple'],
) as dag:
    
    # Task 1: Validate last 5 minutes of data
    validate_task = PythonOperator(
        task_id='validate_last_5_minutes',
        python_callable=validate_last_5_minutes,
        provide_context=True
    )
    
    # Task 2: Check data freshness
    freshness_task = PythonOperator(
        task_id='check_data_freshness',
        python_callable=check_data_freshness,
        provide_context=True
    )
    
    # Simple dependency: validate then check freshness
    validate_task >> freshness_task


dag.doc_md = """
# Data Quality Validation DAG (Simplified)

## Purpose
Simple, straightforward validation that runs every 5 minutes.

## What It Does
1. Gets all data from last 5 minutes
2. Validates basic quality checks:
   - High >= Low
   - Close/Open within High-Low range
   - Positive prices and volume
3. Checks data freshness
4. Stores results

## Configuration
- **Schedule**: Every 5 minutes (`*/5 * * * *`)
- **Window**: Last 5 minutes
- **No skip logic**: Always runs validation
- **No complex config**: Uses environment variables

## Making It Work
This is the simplified version. Make it work first, optimize later.
"""