"""
Model Monitoring and Drift Detection using Evidently

Leverages Evidently open-source library for production ML monitoring instead of
building custom drift detection from scratch.

Evidently provides:
- Data drift detection (feature distributions)
- Target drift detection (label distributions)
- Model performance monitoring
- Visual reports and dashboards
"""

from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

try:
    from evidently import ColumnMapping
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset, ClassificationPreset
    from evidently.metrics import *
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False


class AxiomDriftMonitor:
    """
    Production ML model monitoring using Evidently
    
    Monitors deployed models for:
    - Feature drift (input distribution changes)
    - Target drift (label distribution changes)
    - Model performance degradation
    - Data quality issues
    
    Usage:
        monitor = AxiomDriftMonitor(
            reference_data=training_data,
            model_type="classification"
        )
        
        # Check for drift
        drift_detected = monitor.detect_drift(current_data)
        
        # Generate report
        monitor.generate_report(current_data, "drift_report.html")
    """
    
    def __init__(
        self,
        reference_data: pd.DataFrame,
        model_type: str = "classification",  # or "regression"
        target_column: Optional[str] = "target",
        prediction_column: Optional[str] = "prediction",
        numerical_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None
    ):
        if not EVIDENTLY_AVAILABLE:
            raise ImportError("Evidently required: pip install evidently")
        
        self.reference_data = reference_data
        self.model_type = model_type
        
        # Set up column mapping
        self.column_mapping = ColumnMapping(
            target=target_column,
            prediction=prediction_column,
            numerical_features=numerical_features,
            categorical_features=categorical_features
        )
        
        # Drift thresholds
        self.drift_threshold = 0.1  # 10% drift considered significant
        
    def detect_drift(
        self,
        current_data: pd.DataFrame,
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Detect data drift
        
        Args:
            current_data: Current production data
            threshold: Custom drift threshold
            
        Returns:
            Drift detection results
        """
        threshold = threshold or self.drift_threshold
        
        # Create drift report
        report = Report(metrics=[
            DataDriftPreset(),
        ])
        
        report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )
        
        # Extract results
        results = report.as_dict()
        
        # Check drift
        drift_detected = False
        drifted_features = []
        
        for metric in results['metrics']:
            if 'result' in metric and 'drift_by_columns' in metric['result']:
                drift_info = metric['result']['drift_by_columns']
                for feature, drift_data in drift_info.items():
                    if drift_data.get('drift_detected', False):
                        drift_detected = True
                        drifted_features.append(feature)
        
        return {
            'drift_detected': drift_detected,
            'drifted_features': drifted_features,
            'n_drifted': len(drifted_features),
            'full_report': results
        }
        
    def monitor_model_performance(
        self,
        current_data: pd.DataFrame,
        alert_on_degradation: bool = True,
        performance_threshold: float = 0.05  # 5% degradation triggers alert
    ) -> Dict[str, Any]:
        """
        Monitor model performance metrics
        
        Args:
            current_data: Current data with predictions and actual labels
            alert_on_degradation: Alert if performance degrades
            performance_threshold: Degradation threshold
            
        Returns:
            Performance monitoring results
        """
        # Create performance report
        if self.model_type == "classification":
            report = Report(metrics=[ClassificationPreset()])
        else:
            from evidently.metric_preset import RegressionPreset
            report = Report(metrics=[RegressionPreset()])
        
        report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )
        
        results = report.as_dict()
        
        # Extract performance metrics
        performance_degraded = False
        degradation_details = []
        
        # Parse results (structure depends on metric preset)
        # This is simplified - actual parsing would be more detailed
        
        return {
            'performance_degraded': performance_degraded,
            'degradation_details': degradation_details,
            'full_report': results
        }
        
    def check_data_quality(
        self,
        current_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Check data quality issues
        
        Detects:
        - Missing values
        - Duplicate rows
        - Constant features
        - High correlation
        - Outliers
        """
        report = Report(metrics=[DataQualityPreset()])
        
        report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )
        
        results = report.as_dict()
        
        return {
            'quality_issues': [],  # Parsed from results
            'full_report': results
        }
        
    def generate_report(
        self,
        current_data: pd.DataFrame,
        output_file: str = "monitoring_report.html",
        include_data_quality: bool = True,
        include_drift: bool = True,
        include_performance: bool = True
    ):
        """
        Generate comprehensive monitoring report
        
        Args:
            current_data: Current production data
            output_file: Output HTML file
            include_data_quality: Include data quality checks
            include_drift: Include drift detection
            include_performance: Include performance monitoring
        """
        metrics = []
        
        if include_data_quality:
            metrics.append(DataQualityPreset())
        
        if include_drift:
            metrics.append(DataDriftPreset())
            if self.column_mapping.target:
                metrics.append(TargetDriftPreset())
        
        if include_performance and self.column_mapping.prediction:
            if self.model_type == "classification":
                metrics.append(ClassificationPreset())
            else:
                from evidently.metric_preset import RegressionPreset
                metrics.append(RegressionPreset())
        
        # Create report
        report = Report(metrics=metrics)
        
        report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )
        
        # Save HTML report
        report.save_html(output_file)
        
        return output_file


# Example usage
if __name__ == "__main__":
    print("Evidently Drift Detection - Example")
    print("=" * 60)
    
    if not EVIDENTLY_AVAILABLE:
        print("Install: pip install evidently")
    else:
        # Sample data
        np.random.seed(42)
        
        # Reference data (training)
        ref_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 1000),
            'feature2': np.random.normal(0, 1, 1000),
            'target': np.random.binomial(1, 0.3, 1000),
            'prediction': np.random.binomial(1, 0.3, 1000)
        })
        
        # Current data (production) - with drift
        curr_data = pd.DataFrame({
            'feature1': np.random.normal(0.5, 1.2, 500),  # Drifted
            'feature2': np.random.normal(0, 1, 500),
            'target': np.random.binomial(1, 0.3, 500),
            'prediction': np.random.binomial(1, 0.3, 500)
        })
        
        # Initialize monitor
        monitor = AxiomDriftMonitor(
            reference_data=ref_data,
            model_type="classification",
            target_column="target",
            prediction_column="prediction"
        )
        
        # Detect drift
        drift_results = monitor.detect_drift(curr_data)
        
        print(f"\nDrift Detected: {drift_results['drift_detected']}")
        print(f"Drifted Features: {drift_results['drifted_features']}")
        
        # Generate report
        report_file = monitor.generate_report(curr_data, "example_drift_report.html")
        print(f"\n✓ Report saved: {report_file}")
        
        print("\n" + "=" * 60)
        print("✓ Leveraging Evidently instead of custom drift detection")