"""
Database-Integrated Data Pipeline Demo.

This demonstrates the CORRECT way to build data infrastructure:
1. Ingest data → Store in PostgreSQL (PriceData table)
2. Compute features → Store in PostgreSQL (FeatureData table)
3. Validate quality → Store in PostgreSQL (ValidationResult table)
4. Track pipeline → Store in PostgreSQL (PipelineRun table)
5. Track lineage → Store in PostgreSQL (DataLineage table)

This is NOT in-memory processing - this is REAL database persistence!
"""

import sys
import logging
from datetime import datetime, timedelta
from decimal import Decimal
import uuid

import pandas as pd
import numpy as np

# Database infrastructure (ALREADY EXISTS!)
from axiom.database import (
    get_db,
    MarketDataIntegration,
    FeatureIntegration,
    QualityIntegration,
    PipelineIntegration,
    MigrationManager,
)

# Feature computation (we built this)
from axiom.features.transformations.technical_indicators import TechnicalIndicators

# Data quality (we built this)
from axiom.data_quality.validation.rules_engine import (
    ValidationEngine, CompletessRule, RangeRule, ConsistencyRule
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_market_data() -> pd.DataFrame:
    """Create sample market data for demo."""
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    
    # Generate realistic price data
    np.random.seed(42)
    base_price = 100.0
    returns = np.random.normal(0.001, 0.02, len(dates))
    prices = base_price * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'Open': prices * (1 + np.random.normal(0, 0.005, len(dates))),
        'High': prices * (1 + np.abs(np.random.normal(0.01, 0.01, len(dates)))),
        'Low': prices * (1 - np.abs(np.random.normal(0.01, 0.01, len(dates)))),
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, len(dates)),
    }, index=dates)
    
    # Ensure High >= Low
    df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
    df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)
    
    return df


def main():
    """Run database-integrated pipeline demo."""
    
    print("\n" + "="*70)
    print("DATABASE-INTEGRATED DATA PIPELINE DEMO")
    print("Showing REAL PostgreSQL persistence (not in-memory!)")
    print("="*70 + "\n")
    
    # Configuration
    symbol = "AAPL"
    run_id = f"demo_{uuid.uuid4().hex[:8]}"
    
    try:
        # ============================================
        # STEP 1: Initialize Database
        # ============================================
        print("\n[STEP 1] Initializing Database Connection...")
        
        db = get_db()
        migration_manager = MigrationManager()
        
        # Initialize schema (creates all tables)
        print("  Creating database schema...")
        migration_manager.init_schema()
        print("  ✅ Database schema initialized")
        
        # Check database health
        if db.health_check():
            print("  ✅ PostgreSQL connection healthy")
        else:
            print("  ❌ PostgreSQL connection failed")
            return
        
        # ============================================
        # STEP 2: Start Pipeline Run Tracking
        # ============================================
        print("\n[STEP 2] Starting Pipeline Run Tracking...")
        
        pipeline_integration = PipelineIntegration()
        pipeline_run = pipeline_integration.start_pipeline_run(
            pipeline_name="market_data_ingestion",
            run_id=run_id,
            parameters={'symbol': symbol, 'periods': 100},
            source="demo_sample_data"
        )
        print(f"  ✅ Pipeline run started: {run_id}")
        print(f"     Status: {pipeline_run.status}")
        
        # ============================================
        # STEP 3: Ingest Market Data to PostgreSQL
        # ============================================
        print("\n[STEP 3] Ingesting Market Data to PostgreSQL (PriceData table)...")
        
        # Generate sample data
        price_df = create_sample_market_data()
        print(f"  Generated {len(price_df)} days of price data")
        
        # Store in PostgreSQL using MarketDataIntegration (ALREADY EXISTS!)
        market_data_integration = MarketDataIntegration()
        records_stored = market_data_integration.store_price_data(
            df=price_df,
            symbol=symbol,
            source="demo"
        )
        
        print(f"  ✅ Stored {records_stored} price records to PostgreSQL (PriceData table)")
        
        # ============================================
        # STEP 4: Compute Features & Store in PostgreSQL
        # ============================================
        print("\n[STEP 4] Computing Features & Storing in PostgreSQL (FeatureData table)...")
        
        # Compute technical indicators
        tech_indicators = TechnicalIndicators(price_df)
        
        features_computed = {
            'sma_20': tech_indicators.sma(period=20),
            'sma_50': tech_indicators.sma(period=50),
            'rsi_14': tech_indicators.rsi(period=14),
            'macd': tech_indicators.macd()[0],  # MACD line
            'bbands_upper': tech_indicators.bollinger_bands()[0],  # Upper band
        }
        
        print(f"  Computed {len(features_computed)} technical indicators")
        
        # Store features in PostgreSQL
        feature_integration = FeatureIntegration()
        
        features_stored = 0
        for feature_name, feature_series in features_computed.items():
            for timestamp, value in feature_series.items():
                if not pd.isna(value):
                    feature_integration.store_feature(
                        symbol=symbol,
                        timestamp=timestamp,
                        feature_name=feature_name,
                        value=float(value),
                        feature_category='technical',
                        feature_version='1.0.0',
                        computation_method=feature_name.split('_')[0],  # sma, rsi, etc.
                        source_table='price_data'
                    )
                    features_stored += 1
        
        print(f"  ✅ Stored {features_stored} feature values to PostgreSQL (FeatureData table)")
        
        # ============================================
        # STEP 5: Validate Data & Store Results in PostgreSQL
        # ============================================
        print("\n[STEP 5] Validating Data Quality & Storing Results (ValidationResult table)...")
        
        # Create validation engine
        validation_engine = ValidationEngine()
        
        # Add validation rules
        validation_engine.add_rule(CompletessRule("price_completeness", required_fields=['Close', 'Volume']))
        validation_engine.add_rule(RangeRule("price_range", field='Close', min_value=0, max_value=10000))
        validation_engine.add_rule(ConsistencyRule("ohlc_consistency"))
        
        # Run validation on price data
        validation_results = validation_engine.validate(price_df)
        
        print(f"  Ran {len(validation_results)} validation rules")
        
        # Store validation results in PostgreSQL
        quality_integration = QualityIntegration()
        
        for result in validation_results:
            quality_integration.store_validation_result(
                target_table='price_data',
                rule_name=result['rule'],
                passed=result['passed'],
                severity=result.get('severity', 'info'),
                symbol=symbol,
                rule_category='data_quality',
                message=result.get('message'),
                details=result.get('details'),
                quality_score=result.get('quality_score')
            )
        
        print(f"  ✅ Stored {len(validation_results)} validation results to PostgreSQL (ValidationResult table)")
        
        # ============================================
        # STEP 6: Track Data Lineage
        # ============================================
        print("\n[STEP 6] Tracking Data Lineage in PostgreSQL (DataLineage table)...")
        
        # Track lineage: price_data → feature_data
        for feature_name in features_computed.keys():
            pipeline_integration.track_lineage(
                source_table='price_data',
                target_table='feature_data',
                transformation_name=f'compute_{feature_name}',
                transformation_type='calculation',
                pipeline_run_id=pipeline_run.id,
                transformation_logic=f'Technical indicator: {feature_name}',
                metadata={'feature_category': 'technical', 'symbol': symbol}
            )
        
        print(f"  ✅ Tracked {len(features_computed)} lineage records in PostgreSQL")
        
        # ============================================
        # STEP 7: Complete Pipeline Run
        # ============================================
        print("\n[STEP 7] Completing Pipeline Run...")
        
        pipeline_integration.complete_pipeline_run(
            run_id=run_id,
            status='success',
            records_processed=records_stored,
            records_inserted=records_stored + features_stored,
            output_tables=['price_data', 'feature_data', 'validation_results', 'data_lineage'],
            output_record_count={
                'price_data': records_stored,
                'feature_data': features_stored,
                'validation_results': len(validation_results),
                'data_lineage': len(features_computed)
            }
        )
        
        print(f"  ✅ Pipeline run completed: {run_id}")
        
        # ============================================
        # STEP 8: Query Data from PostgreSQL
        # ============================================
        print("\n[STEP 8] Querying Data from PostgreSQL...")
        
        # Query price data
        from axiom.database import get_session
        from axiom.database.models import PriceData, FeatureData, ValidationResult, PipelineRun
        
        session = get_session()
        
        # Count records in each table
        price_count = session.query(PriceData).filter(PriceData.symbol == symbol).count()
        feature_count = session.query(FeatureData).filter(FeatureData.symbol == symbol).count()
        validation_count = session.query(ValidationResult).filter(ValidationResult.symbol == symbol).count()
        pipeline_count = session.query(PipelineRun).filter(PipelineRun.run_id == run_id).count()
        
        print(f"\n  📊 PostgreSQL Record Counts:")
        print(f"     PriceData table: {price_count} records for {symbol}")
        print(f"     FeatureData table: {feature_count} features for {symbol}")
        print(f"     ValidationResult table: {validation_count} validations for {symbol}")
        print(f"     PipelineRun table: {pipeline_count} runs")
        
        # Sample queries
        print(f"\n  📈 Sample Data from PostgreSQL:")
        
        # Latest 5 price records
        latest_prices = session.query(PriceData).filter(
            PriceData.symbol == symbol
        ).order_by(PriceData.timestamp.desc()).limit(5).all()
        
        print(f"\n  Latest 5 Price Records:")
        for price in latest_prices:
            print(f"     {price.timestamp.date()}: Close=${price.close}, Volume={price.volume}")
        
        # Latest 5 features
        latest_features = session.query(FeatureData).filter(
            FeatureData.symbol == symbol
        ).order_by(FeatureData.timestamp.desc()).limit(5).all()
        
        print(f"\n  Latest 5 Features:")
        for feature in latest_features:
            print(f"     {feature.timestamp.date()}: {feature.feature_name}={feature.value:.4f}")
        
        # Validation summary
        passed = session.query(ValidationResult).filter(
            ValidationResult.symbol == symbol,
            ValidationResult.passed == True
        ).count()
        
        failed = session.query(ValidationResult).filter(
            ValidationResult.symbol == symbol,
            ValidationResult.passed == False
        ).count()
        
        print(f"\n  Validation Summary:")
        print(f"     Passed: {passed}")
        print(f"     Failed: {failed}")
        print(f"     Pass Rate: {passed/(passed+failed)*100:.1f}%")
        
        session.close()
        
        # ============================================
        # SUCCESS SUMMARY
        # ============================================
        print("\n" + "="*70)
        print("✅ DATABASE-INTEGRATED PIPELINE COMPLETE!")
        print("="*70)
        
        print(f"\nData Flow:")
        print(f"  1. Market Data → PostgreSQL (PriceData) ✅")
        print(f"  2. Feature Computation → PostgreSQL (FeatureData) ✅")
        print(f"  3. Quality Validation → PostgreSQL (ValidationResult) ✅")
        print(f"  4. Pipeline Tracking → PostgreSQL (PipelineRun) ✅")
        print(f"  5. Lineage Tracking → PostgreSQL (DataLineage) ✅")
        
        print(f"\nPersistence Verified:")
        print(f"  - {price_count} price records in PostgreSQL")
        print(f"  - {feature_count} features in PostgreSQL")
        print(f"  - {validation_count} validation results in PostgreSQL")
        print(f"  - All data queryable via SQL")
        
        print(f"\nThis is REAL database persistence - not in-memory!")
        print(f"Run SQL queries to verify:")
        print(f"  SELECT * FROM price_data WHERE symbol = '{symbol}' LIMIT 5;")
        print(f"  SELECT * FROM feature_data WHERE symbol = '{symbol}' LIMIT 5;")
        print(f"  SELECT * FROM validation_results WHERE symbol = '{symbol}';")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        
        # Mark pipeline as failed
        try:
            pipeline_integration.complete_pipeline_run(
                run_id=run_id,
                status='failed',
                error_message=str(e)
            )
        except:
            pass
        
        raise


if __name__ == "__main__":
    print("\n" + "🎯 "* 15)
    print("IMPORTANT: This demo requires PostgreSQL running!")
    print("Start it with: cd axiom/database && docker-compose up -d postgres")
    print("🎯 " * 15 + "\n")
    
    response = input("Is PostgreSQL running? (y/n): ")
    if response.lower() != 'y':
        print("\n❌ Please start PostgreSQL first")
        print("   cd axiom/database && docker-compose up -d postgres")
        sys.exit(1)
    
    main()