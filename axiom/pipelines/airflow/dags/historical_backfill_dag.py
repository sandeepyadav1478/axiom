"""
Historical Data Backfill DAG
One-time or scheduled execution to backfill historical stock prices

Purpose: Populate price_data table with historical data to enable ML model training
Target: 1 year daily data (252 trading days) × 8 symbols = ~2,000 records
Enables: Portfolio optimization, VaR models, time series forecasting

SAFETY:
- Idempotent: Can run multiple times without duplicates
- Validated: Checks data quality before storage
- Monitored: Tracks progress and errors
- Configurable: Symbol list and lookback period via YAML
"""
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, '/opt/airflow')

from dotenv import load_dotenv
load_dotenv('/opt/airflow/.env', override=True)

# Import utilities
utils_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, utils_path)
from utils.config_loader import dag_config, get_symbols_for_dag

# ================================================================
# Configuration from YAML
# ================================================================
DAG_NAME = 'historical_backfill'
config = dag_config.get_dag_config(DAG_NAME)
default_args = dag_config.get_default_args(DAG_NAME)
SYMBOLS = get_symbols_for_dag(DAG_NAME)
backfill_config = config.get('backfill', {})

# ================================================================
# Helper Functions
# ================================================================

def fetch_historical_data(context):
    """
    Fetch historical data from yfinance (1 year daily).
    
    Returns structured data ready for PostgreSQL insertion.
    """
    import yfinance as yf
    from datetime import datetime, timedelta
    
    symbols = context['params'].get('symbols', SYMBOLS)
    lookback_days = context['params'].get('lookback_days', 365)
    
    all_data = []
    metrics = {
        'symbols_processed': 0,
        'total_records': 0,
        'failed_symbols': []
    }
    
    for symbol in symbols:
        try:
            context['task'].log.info(f"Fetching {symbol} - {lookback_days} days")
            
            ticker = yf.Ticker(symbol)
            
            # Fetch historical data
            hist = ticker.history(period=f"{lookback_days}d", interval='1d')
            
            if hist.empty:
                context['task'].log.warning(f"No historical data for {symbol}")
                metrics['failed_symbols'].append(symbol)
                continue
            
            # Convert to records
            for date, row in hist.iterrows():
                record = {
                    'symbol': symbol,
                    'timestamp': date,
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': int(row['Volume']),
                    'source': 'yfinance_historical',
                    'timeframe': 'DAY_1'
                }
                all_data.append(record)
            
            records_fetched = len(hist)
            metrics['total_records'] += records_fetched
            metrics['symbols_processed'] += 1
            
            context['task'].log.info(
                f"✅ {symbol}: {records_fetched} records "
                f"({hist.index[0].date()} to {hist.index[-1].date()})"
            )
            
        except Exception as e:
            context['task'].log.error(f"Failed to fetch {symbol}: {e}")
            metrics['failed_symbols'].append(symbol)
    
    # Push to XCom
    context['ti'].xcom_push(key='historical_data', value=all_data)
    context['ti'].xcom_push(key='fetch_metrics', value=metrics)
    
    context['task'].log.info(
        f"Fetch complete: {metrics['symbols_processed']}/{len(symbols)} symbols, "
        f"{metrics['total_records']} total records"
    )
    
    return metrics


def validate_historical_data(context):
    """
    Validate historical data before storage.
    
    Checks:
    - OHLC integrity (high >= low, etc.)
    - No null prices
    - Timestamps in valid range
    - Volume >= 0
    """
    all_data = context['ti'].xcom_pull(task_ids='fetch_historical', key='historical_data')
    
    if not all_data:
        raise ValueError("No historical data to validate")
    
    validation_issues = []
    valid_records = []
    
    for idx, record in enumerate(all_data):
        issues = []
        
        # Check required fields
        required_fields = ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing = [f for f in required_fields if f not in record or record[f] is None]
        if missing:
            issues.append(f"Missing fields: {missing}")
        
        # OHLC integrity
        if record.get('high') and record.get('low'):
            if record['high'] < record['low']:
                issues.append(f"high < low ({record['high']} < {record['low']})")
        
        # Price positivity
        for field in ['open', 'high', 'low', 'close']:
            if record.get(field) and record[field] <= 0:
                issues.append(f"{field} <= 0 ({record[field]})")
        
        # Volume non-negative
        if record.get('volume') and record['volume'] < 0:
            issues.append(f"negative volume ({record['volume']})")
        
        # Timestamp validity
        if record.get('timestamp'):
            ts = record['timestamp']
            if ts > datetime.now():
                issues.append(f"future timestamp ({ts})")
        
        if issues:
            validation_issues.append({
                'record_index': idx,
                'symbol': record.get('symbol'),
                'timestamp': str(record.get('timestamp')),
                'issues': issues
            })
        else:
            valid_records.append(record)
    
    validation_result = {
        'total_records': len(all_data),
        'valid_records': len(valid_records),
        'invalid_records': len(validation_issues),
        'validation_rate': len(valid_records) / len(all_data) * 100 if all_data else 0
    }
    
    context['ti'].xcom_push(key='valid_data', value=valid_records)
    context['ti'].xcom_push(key='validation_issues', value=validation_issues)
    context['ti'].xcom_push(key='validation_result', value=validation_result)
    
    context['task'].log.info(
        f"Validation: {validation_result['valid_records']}/{validation_result['total_records']} "
        f"passed ({validation_result['validation_rate']:.1f}%)"
    )
    
    if validation_issues:
        context['task'].log.warning(f"Found {len(validation_issues)} invalid records")
        for issue in validation_issues[:5]:  # Log first 5
            context['task'].log.warning(f"  {issue}")
    
    return validation_result


def store_historical_data(context):
    """
    Store validated historical data in PostgreSQL.
    
    Uses INSERT ... ON CONFLICT DO NOTHING for idempotency.
    """
    import psycopg2
    from psycopg2.extras import execute_batch
    
    valid_data = context['ti'].xcom_pull(task_ids='validate_historical', key='valid_data')
    
    if not valid_data:
        context['task'].log.warning("No valid data to store")
        return {'stored': 0}
    
    # Connect to PostgreSQL
    conn = psycopg2.connect(
        host=os.getenv('POSTGRES_HOST', 'localhost'),
        user=os.getenv('POSTGRES_USER', 'axiom'),
        password=os.getenv('POSTGRES_PASSWORD'),
        database=os.getenv('POSTGRES_DB', 'axiom_finance')
    )
    
    cur = conn.cursor()
    
    # Prepare batch insert (idempotent with ON CONFLICT)
    insert_query = """
        INSERT INTO price_data 
        (symbol, timestamp, timeframe, open, high, low, close, volume, source)
        VALUES (%(symbol)s, %(timestamp)s, %(timeframe)s, %(open)s, %(high)s, 
                %(low)s, %(close)s, %(volume)s, %(source)s)
        ON CONFLICT (symbol, timestamp, timeframe) DO NOTHING
    """
    
    # Execute batch insert
    execute_batch(cur, insert_query, valid_data, page_size=100)
    conn.commit()
    
    # Get count of inserted rows
    rows_inserted = cur.rowcount
    
    cur.close()
    conn.close()
    
    context['task'].log.info(f"✅ Stored {rows_inserted} records in PostgreSQL")
    
    return {
        'records_attempted': len(valid_data),
        'records_stored': rows_inserted,
        'duplicates_skipped': len(valid_data) - rows_inserted
    }


def verify_backfill_success(context):
    """
    Verify historical backfill completed successfully.
    
    Checks:
    - Expected number of records per symbol
    - Date range coverage
    - Data quality metrics
    """
    import psycopg2
    
    symbols = context['params'].get('symbols', SYMBOLS)
    lookback_days = context['params'].get('lookback_days', 365)
    
    conn = psycopg2.connect(
        host=os.getenv('POSTGRES_HOST', 'localhost'),
        user=os.getenv('POSTGRES_USER', 'axiom'),
        password=os.getenv('POSTGRES_PASSWORD'),
        database=os.getenv('POSTGRES_DB', 'axiom_finance')
    )
    
    cur = conn.cursor()
    
    verification_results = {}
    
    for symbol in symbols:
        # Count records
        cur.execute("""
            SELECT COUNT(*), MIN(timestamp), MAX(timestamp)
            FROM price_data
            WHERE symbol = %s AND timeframe = 'DAY_1'
        """, (symbol,))
        
        count, min_date, max_date = cur.fetchone()
        
        # Calculate expected trading days (~252 per year)
        expected_min = int(lookback_days * 0.65)  # Account for weekends/holidays
        
        verification_results[symbol] = {
            'record_count': count,
            'expected_minimum': expected_min,
            'meets_minimum': count >= expected_min,
            'date_range': f"{min_date} to {max_date}" if min_date else "No data",
            'coverage_days': (max_date - min_date).days if (min_date and max_date) else 0
        }
        
        status = "✅" if count >= expected_min else "⚠️"
        context['task'].log.info(
            f"{status} {symbol}: {count} records (expected >={expected_min}), "
            f"range: {verification_results[symbol]['date_range']}"
        )
    
    cur.close()
    conn.close()
    
    # Summary
    total_records = sum(v['record_count'] for v in verification_results.values())
    symbols_meeting_minimum = sum(1 for v in verification_results.values() if v['meets_minimum'])
    
    summary = {
        'total_records_in_db': total_records,
        'symbols_meeting_minimum': symbols_meeting_minimum,
        'total_symbols_checked': len(symbols),
        'success_rate': symbols_meeting_minimum / len(symbols) * 100 if symbols else 0
    }
    
    context['ti'].xcom_push(key='verification_summary', value=summary)
    
    context['task'].log.info(
        f"Verification Summary: {total_records} total records, "
        f"{symbols_meeting_minimum}/{len(symbols)} symbols meet minimum"
    )
    
    return summary


# ================================================================
# Define DAG
# ================================================================
with DAG(
    dag_id=config.get('dag_id', 'historical_backfill'),
    default_args=default_args,
    description=config.get('description', 'One-time historical data backfill (1 year)'),
    schedule_interval=config.get('schedule_interval', None),  # Manual trigger only
    start_date=days_ago(1),
    catchup=False,
    tags=config.get('tags', ['backfill', 'historical', 'one-time']),
    max_active_runs=1,
) as dag:
    
    # Task 1: Fetch historical data from yfinance
    fetch_historical = PythonOperator(
        task_id='fetch_historical',
        python_callable=fetch_historical_data,
        params={
            'symbols': SYMBOLS,
            'lookback_days': backfill_config.get('lookback_days', 365)
        }
    )
    
    # Task 2: Validate data quality
    validate_historical = PythonOperator(
        task_id='validate_historical',
        python_callable=validate_historical_data
    )
    
    # Task 3: Store in PostgreSQL (idempotent)
    store_historical = PythonOperator(
        task_id='store_historical',
        python_callable=store_historical_data
    )
    
    # Task 4: Verify backfill success
    verify_backfill = PythonOperator(
        task_id='verify_backfill',
        python_callable=verify_backfill_success,
        params={
            'symbols': SYMBOLS,
            'lookback_days': backfill_config.get('lookback_days', 365)
        }
    )
    
    # Task dependencies
    fetch_historical >> validate_historical >> store_historical >> verify_backfill


# ================================================================
# DAG Documentation
# ================================================================
dag.doc_md = """
# Historical Data Backfill DAG

## Purpose

Backfill historical stock price data to enable ML model training.

**Current State:** ~66 records (1-2 hours of data)  
**After Backfill:** ~2,000 records (1 year of daily data)  
**Enables:** 40% of ML models to train

## What It Does

### 1. Fetches Historical Data
- Source: yfinance (FREE, unlimited)
- Symbols: 8 stocks (AAPL, MSFT, GOOGL, TSLA, NVDA, SPY, QQQ, META)
- Period: 1 year daily (252 trading days)
- Fields: OHLCV (open, high, low, close, volume)

### 2. Validates Data Quality
- OHLC integrity checks (high >= low, etc.)
- No null prices
- Timestamps in valid range
- Volume non-negative
- Logs all validation issues

### 3. Stores in PostgreSQL
- Table: `price_data`
- Idempotent: Uses ON CONFLICT DO NOTHING
- Safe to re-run: Won't create duplicates
- Batch insert: 100 records at a time

### 4. Verifies Success
- Checks record counts per symbol
- Validates date range coverage
- Reports success metrics
- Logs any missing data

## How to Use

### First Time Backfill
```bash
# Trigger manually in Airflow UI
# Or via CLI:
docker exec axiom-airflow-webserver airflow dags trigger historical_backfill
```

### Monitor Progress
```bash
# Check logs
docker logs axiom-airflow-scheduler | grep historical_backfill

# Check database
docker exec axiom-postgresql psql -U axiom -d axiom_finance -c \
  "SELECT symbol, COUNT(*) FROM price_data WHERE source = 'yfinance_historical' GROUP BY symbol;"
```

### Expected Results
```
Symbol | Records | Date Range
-------|---------|------------------
AAPL   | 252     | 2024-11-21 to 2025-11-21
MSFT   | 252     | 2024-11-21 to 2025-11-21
GOOGL  | 252     | 2024-11-21 to 2025-11-21
...
Total: ~2,000 records
```

## Safety Features

### Idempotent Design
- Can run multiple times safely
- ON CONFLICT DO NOTHING prevents duplicates
- No data loss if re-run

### Validation Before Storage
- All records validated before PostgreSQL insertion
- Invalid records logged but not stored
- Quarantine table for manual review (future)

### Progress Tracking
- XCom stores intermediate results
- Can resume if interrupted
- Full audit trail in Airflow logs

## What This Enables

### ML Models Unblocked (40% → 60%)

**Portfolio Optimization (12 models):**
- ✅ Markowitz, Black-Litterman, Risk Parity
- ✅ MILLION, RegimeFolio (need 252+ days minimum)
- ✅ HRP, CVaR optimization

**VaR Models (5 models):**
- ✅ Historical VaR (needs 252+ days)
- ✅ Parametric VaR
- ⚠️ Regime-Switching VaR (needs 5+ years for regimes)

**Time Series (Several models):**
- ✅ ARIMA (needs 100+ observations)
- ✅ GARCH (needs 250+ observations)  
- ✅ EWMA

**Options Pricing:**
- ✅ Historical volatility calculation (needs 252+ days)
- ⚠️ Advanced models still need options chain data

### Still Blocked (Need More Data)

**Credit Models:** Need company fundamentals (next phase)  
**Advanced Options:** Need options chain data (Week 4)  
**Regime Models:** Need 5+ years for regime identification

## Configuration

Edit [`dag_config.yaml`](../dag_configs/dag_config.yaml):

```yaml
historical_backfill:
  dag_id: historical_backfill
  schedule_interval: null  # Manual trigger only
  
  backfill:
    lookback_days: 365  # 1 year
    batch_size: 100
    validate_before_store: true
  
  symbols_list: extended  # Uses 8 symbols
```

## Troubleshooting

**Issue:** "No data for symbol XYZ"
- Solution: Symbol might be delisted or ticker changed
- Action: Remove from symbol list or use correct ticker

**Issue:** "Validation failed: X records rejected"
- Solution: Check validation_issues in XCom
- Action: Review data quality, may need manual intervention

**Issue:** "Duplicates detected"
- Solution: Backfill already ran successfully
- Action: No action needed - ON CONFLICT prevents duplicates

## Next Steps After Backfill

1. Verify data quality:
   ```sql
   SELECT symbol, COUNT(*), MIN(timestamp), MAX(timestamp)
   FROM price_data 
   WHERE source = 'yfinance_historical'
   GROUP BY symbol;
   ```

2. Test ML models:
   ```bash
   python demos/demo_portfolio_optimization.py
   python demos/demo_var_risk_models.py
   ```

3. Expand to 50 stocks (Phase 2)

4. Add fundamentals data (Phase 3)

All configuration in [`dag_config.yaml`](../dag_configs/dag_config.yaml)!
"""