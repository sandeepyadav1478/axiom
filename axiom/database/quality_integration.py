"""
Data Quality Integration Layer.

Connects Data Quality Framework to PostgreSQL for:
- Validation result persistence
- Quality metrics tracking
- Compliance reporting
- Anomaly tracking
"""

import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from .models import ValidationResult, PipelineRun, DataLineage
from .session import SessionManager

logger = logging.getLogger(__name__)


class QualityIntegration:
    """
    Integration layer for data quality results.
    
    Stores:
    - Validation results
    - Quality metrics
    - Anomaly detection results
    - Compliance status
    """
    
    def __init__(self, session_manager: Optional[SessionManager] = None):
        """Initialize quality integration."""
        self.session = session_manager or SessionManager()
    
    def store_validation_result(
        self,
        target_table: str,
        rule_name: str,
        passed: bool,
        severity: str = "info",
        symbol: Optional[str] = None,
        target_id: Optional[int] = None,
        rule_category: Optional[str] = None,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        quality_score: Optional[float] = None,
        quality_grade: Optional[str] = None,
        is_anomaly: bool = False,
        anomaly_score: Optional[float] = None,
        anomaly_method: Optional[str] = None
    ) -> ValidationResult:
        """
        Store validation result.
        
        Args:
            target_table: Table being validated
            rule_name: Validation rule name
            passed: Whether validation passed
            severity: Severity level
            symbol: Asset symbol
            target_id: Record ID being validated
            rule_category: Rule category
            message: Validation message
            details: Additional details
            quality_score: Quality score (0-100)
            quality_grade: Quality grade
            is_anomaly: Whether anomaly detected
            anomaly_score: Anomaly score
            anomaly_method: Anomaly detection method
            
        Returns:
            Stored ValidationResult
        """
        result = ValidationResult(
            validation_date=datetime.now(),
            target_table=target_table,
            target_id=target_id,
            symbol=symbol,
            rule_name=rule_name,
            rule_category=rule_category,
            severity=severity,
            passed=passed,
            message=message,
            details=details,
            quality_score=quality_score,
            quality_grade=quality_grade,
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_score,
            anomaly_method=anomaly_method,
            is_compliant=passed and severity not in ['critical', 'error']
        )
        
        self.session.add(result)
        self.session.commit()
        
        logger.debug(f"Stored validation result: {rule_name} for {target_table}/{symbol}")
        
        return result
    
    def bulk_store_validation_results(
        self,
        results: List[Dict[str, Any]]
    ) -> int:
        """
        Bulk store validation results.
        
        Args:
            results: List of validation result dictionaries
            
        Returns:
            Number of results stored
        """
        validation_results = []
        
        for res in results:
            validation_results.append(ValidationResult(
                validation_date=datetime.now(),
                target_table=res['target_table'],
                target_id=res.get('target_id'),
                symbol=res.get('symbol'),
                rule_name=res['rule_name'],
                rule_category=res.get('rule_category'),
                severity=res.get('severity', 'info'),
                passed=res['passed'],
                message=res.get('message'),
                details=res.get('details'),
                quality_score=res.get('quality_score'),
                quality_grade=res.get('quality_grade'),
                is_anomaly=res.get('is_anomaly', False),
                anomaly_score=res.get('anomaly_score'),
                anomaly_method=res.get('anomaly_method'),
                is_compliant=res['passed']
            ))
        
        self.session.bulk_insert(validation_results)
        self.session.commit()
        
        logger.info(f"Bulk stored {len(validation_results)} validation results")
        
        return len(validation_results)
    
    def get_validation_history(
        self,
        target_table: str,
        symbol: Optional[str] = None,
        rule_name: Optional[str] = None,
        days: int = 30
    ) -> List[ValidationResult]:
        """
        Get validation history.
        
        Args:
            target_table: Table name
            symbol: Filter by symbol
            rule_name: Filter by rule
            days: Days to look back
            
        Returns:
            List of validation results
        """
        from datetime import timedelta
        cutoff_date = datetime.now() - timedelta(days=days)
        
        query = self.session.query(ValidationResult).filter(
            ValidationResult.target_table == target_table,
            ValidationResult.validation_date >= cutoff_date
        )
        
        if symbol:
            query = query.filter(ValidationResult.symbol == symbol)
        
        if rule_name:
            query = query.filter(ValidationResult.rule_name == rule_name)
        
        return query.order_by(ValidationResult.validation_date.desc()).all()
    
    def get_quality_summary(
        self,
        target_table: str,
        symbol: Optional[str] = None,
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Get quality summary statistics.
        
        Args:
            target_table: Table name
            symbol: Filter by symbol
            days: Days to analyze
            
        Returns:
            Quality summary dict
        """
        from datetime import timedelta
        from sqlalchemy import func
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        query = self.session.query(ValidationResult).filter(
            ValidationResult.target_table == target_table,
            ValidationResult.validation_date >= cutoff_date
        )
        
        if symbol:
            query = query.filter(ValidationResult.symbol == symbol)
        
        results = query.all()
        
        if not results:
            return {
                'total_validations': 0,
                'passed': 0,
                'failed': 0,
                'pass_rate': 0.0,
                'avg_quality_score': 0.0,
                'anomalies_detected': 0
            }
        
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed
        anomalies = sum(1 for r in results if r.is_anomaly)
        
        quality_scores = [r.quality_score for r in results if r.quality_score is not None]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        
        return {
            'total_validations': total,
            'passed': passed,
            'failed': failed,
            'pass_rate': passed / total if total > 0 else 0.0,
            'avg_quality_score': avg_quality,
            'anomalies_detected': anomalies,
            'period_days': days
        }


class PipelineIntegration:
    """
    Integration layer for pipeline execution tracking.
    
    Tracks:
    - Pipeline runs
    - Performance metrics
    - Error handling
    - Data lineage
    """
    
    def __init__(self, session_manager: Optional[SessionManager] = None):
        """Initialize pipeline integration."""
        self.session = session_manager or SessionManager()
    
    def start_pipeline_run(
        self,
        pipeline_name: str,
        run_id: str,
        parameters: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None,
        pipeline_version: str = "1.0.0"
    ) -> PipelineRun:
        """
        Start tracking a pipeline run.
        
        Args:
            pipeline_name: Name of pipeline
            run_id: Unique run ID
            parameters: Pipeline parameters
            source: Data source
            pipeline_version: Pipeline version
            
        Returns:
            PipelineRun object
        """
        pipeline_run = PipelineRun(
            pipeline_name=pipeline_name,
            pipeline_version=pipeline_version,
            run_id=run_id,
            started_at=datetime.now(),
            status='running',
            parameters=parameters,
            source=source
        )
        
        self.session.add(pipeline_run)
        self.session.commit()
        
        logger.info(f"Started pipeline run: {pipeline_name} ({run_id})")
        
        return pipeline_run
    
    def complete_pipeline_run(
        self,
        run_id: str,
        status: str = 'success',
        records_processed: int = 0,
        records_inserted: int = 0,
        records_updated: int = 0,
        records_failed: int = 0,
        error_message: Optional[str] = None,
        error_details: Optional[Dict[str, Any]] = None,
        output_tables: Optional[List[str]] = None,
        output_record_count: Optional[Dict[str, int]] = None
    ) -> PipelineRun:
        """
        Complete pipeline run tracking.
        
        Args:
            run_id: Run ID
            status: Final status
            records_processed: Total records processed
            records_inserted: Records inserted
            records_updated: Records updated
            records_failed: Records failed
            error_message: Error message if failed
            error_details: Error details
            output_tables: Tables modified
            output_record_count: Record counts by table
            
        Returns:
            Updated PipelineRun
        """
        pipeline_run = self.session.query(PipelineRun).filter(
            PipelineRun.run_id == run_id
        ).first()
        
        if not pipeline_run:
            raise ValueError(f"Pipeline run not found: {run_id}")
        
        completed_at = datetime.now()
        duration = (completed_at - pipeline_run.started_at).total_seconds()
        
        pipeline_run.completed_at = completed_at
        pipeline_run.duration_seconds = duration
        pipeline_run.status = status
        pipeline_run.records_processed = records_processed
        pipeline_run.records_inserted = records_inserted
        pipeline_run.records_updated = records_updated
        pipeline_run.records_failed = records_failed
        pipeline_run.error_message = error_message
        pipeline_run.error_details = error_details
        pipeline_run.output_tables = output_tables
        pipeline_run.output_record_count = output_record_count
        
        if records_processed > 0 and duration > 0:
            pipeline_run.throughput_records_per_sec = records_processed / duration
        
        self.session.commit()
        
        logger.info(f"Completed pipeline run: {pipeline_run.pipeline_name} ({run_id}) - {status}")
        
        return pipeline_run
    
    def track_lineage(
        self,
        source_table: str,
        target_table: str,
        transformation_name: str,
        transformation_type: str = "calculation",
        source_id: Optional[int] = None,
        target_id: Optional[int] = None,
        pipeline_run_id: Optional[int] = None,
        transformation_logic: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DataLineage:
        """
        Track data lineage.
        
        Args:
            source_table: Source table name
            target_table: Target table name
            transformation_name: Name of transformation
            transformation_type: Type of transformation
            source_id: Source record ID
            target_id: Target record ID
            pipeline_run_id: Associated pipeline run
            transformation_logic: SQL/code for transformation
            metadata: Additional metadata
            
        Returns:
            DataLineage record
        """
        lineage = DataLineage(
            source_table=source_table,
            source_id=source_id,
            source_timestamp=datetime.now(),
            target_table=target_table,
            target_id=target_id,
            target_timestamp=datetime.now(),
            transformation_name=transformation_name,
            transformation_type=transformation_type,
            transformation_logic=transformation_logic,
            pipeline_run_id=pipeline_run_id,
            metadata=metadata
        )
        
        self.session.add(lineage)
        self.session.commit()
        
        logger.debug(f"Tracked lineage: {source_table} → {target_table} ({transformation_name})")
        
        return lineage
    
    def get_pipeline_history(
        self,
        pipeline_name: str,
        days: int = 30,
        status: Optional[str] = None
    ) -> List[PipelineRun]:
        """
        Get pipeline execution history.
        
        Args:
            pipeline_name: Pipeline name
            days: Days to look back
            status: Filter by status
            
        Returns:
            List of pipeline runs
        """
        from datetime import timedelta
        cutoff_date = datetime.now() - timedelta(days=days)
        
        query = self.session.query(PipelineRun).filter(
            PipelineRun.pipeline_name == pipeline_name,
            PipelineRun.started_at >= cutoff_date
        )
        
        if status:
            query = query.filter(PipelineRun.status == status)
        
        return query.order_by(PipelineRun.started_at.desc()).all()


# Export
__all__ = [
    "QualityIntegration",
    "PipelineIntegration",
]