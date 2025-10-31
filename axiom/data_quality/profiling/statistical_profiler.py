"""
Statistical Data Profiler - Institutional Grade

Comprehensive statistical profiling for financial data quality assessment.
Generates detailed profiles including distributions, outliers, and quality metrics.

Critical for data legitimacy and model reliability.
"""

from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import statistics
from collections import Counter


@dataclass
class ColumnProfile:
    """Statistical profile for a single data column/field."""
    
    column_name: str
    data_type: str
    
    # Completeness metrics
    total_count: int = 0
    null_count: int = 0
    null_percentage: float = 0.0
    unique_count: int = 0
    unique_percentage: float = 0.0
    
    # Numerical statistics (if applicable)
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mean: Optional[float] = None
    median: Optional[float] = None
    std_dev: Optional[float] = None
    q1: Optional[float] = None  # 25th percentile
    q3: Optional[float] = None  # 75th percentile
    iqr: Optional[float] = None  # Interquartile range
    
    # Distribution metrics
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    variance: Optional[float] = None
    
    # Categorical statistics (if applicable)
    most_common_values: List[tuple] = field(default_factory=list)  # [(value, count), ...]
    cardinality: int = 0
    
    # Quality indicators
    outlier_count: int = 0
    outlier_percentage: float = 0.0
    quality_score: float = 0.0  # 0-100
    
    # Validation flags
    has_negatives: bool = False
    has_zeros: bool = False
    has_duplicates: bool = False
    
    # Metadata
    profiled_at: datetime = field(default_factory=datetime.now)
    sample_values: List[Any] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert profile to dictionary for storage/reporting."""
        return {
            "column_name": self.column_name,
            "data_type": self.data_type,
            "completeness": {
                "total_count": self.total_count,
                "null_count": self.null_count,
                "null_percentage": self.null_percentage,
                "completeness_score": 100 - self.null_percentage
            },
            "uniqueness": {
                "unique_count": self.unique_count,
                "unique_percentage": self.unique_percentage,
                "cardinality": self.cardinality
            },
            "statistics": {
                "min": self.min_value,
                "max": self.max_value,
                "mean": self.mean,
                "median": self.median,
                "std_dev": self.std_dev,
                "q1": self.q1,
                "q3": self.q3,
                "iqr": self.iqr
            } if self.mean is not None else None,
            "distribution": {
                "skewness": self.skewness,
                "kurtosis": self.kurtosis,
                "variance": self.variance
            } if self.variance is not None else None,
            "categorical": {
                "most_common": self.most_common_values[:10],
                "cardinality": self.cardinality
            } if self.most_common_values else None,
            "quality": {
                "outlier_count": self.outlier_count,
                "outlier_percentage": self.outlier_percentage,
                "quality_score": self.quality_score,
                "has_negatives": self.has_negatives,
                "has_zeros": self.has_zeros,
                "has_duplicates": self.has_duplicates
            },
            "profiled_at": self.profiled_at.isoformat()
        }


@dataclass
class DatasetProfile:
    """Complete profile for an entire dataset."""
    
    dataset_name: str
    total_rows: int
    total_columns: int
    
    # Column profiles
    column_profiles: Dict[str, ColumnProfile] = field(default_factory=dict)
    
    # Dataset-level metrics
    overall_completeness: float = 0.0
    overall_quality_score: float = 0.0
    
    # Correlation analysis (for numerical columns)
    correlations: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Data quality issues
    critical_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Metadata
    profiled_at: datetime = field(default_factory=datetime.now)
    profile_version: str = "1.0"
    
    def to_dict(self) -> Dict:
        """Convert dataset profile to dictionary."""
        return {
            "dataset_name": self.dataset_name,
            "summary": {
                "total_rows": self.total_rows,
                "total_columns": self.total_columns,
                "overall_completeness": self.overall_completeness,
                "overall_quality_score": self.overall_quality_score
            },
            "column_profiles": {
                name: profile.to_dict() 
                for name, profile in self.column_profiles.items()
            },
            "data_quality": {
                "critical_issues": self.critical_issues,
                "warnings": self.warnings,
                "issue_count": len(self.critical_issues) + len(self.warnings)
            },
            "correlations": self.correlations,
            "metadata": {
                "profiled_at": self.profiled_at.isoformat(),
                "profile_version": self.profile_version
            }
        }


class StatisticalDataProfiler:
    """
    Statistical data profiler for financial datasets.
    
    Generates comprehensive statistical profiles including:
    - Descriptive statistics
    - Distribution analysis
    - Outlier detection
    - Quality metrics
    - Correlation analysis
    
    Used for:
    - Data quality assessment
    - Model input validation
    - Drift detection baseline
    - Data documentation
    """
    
    def __init__(self):
        """Initialize statistical profiler."""
        self.profile_cache: Dict[str, DatasetProfile] = {}
    
    def profile_dataset(
        self,
        data: List[Dict[str, Any]],
        dataset_name: str,
        numerical_columns: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None
    ) -> DatasetProfile:
        """
        Generate comprehensive statistical profile for dataset.
        
        Args:
            data: List of dictionaries representing rows
            dataset_name: Name/identifier for the dataset
            numerical_columns: List of numerical column names (auto-detected if None)
            categorical_columns: List of categorical column names
        
        Returns:
            Complete dataset profile with column-level and dataset-level metrics
        """
        if not data:
            raise ValueError("Cannot profile empty dataset")
        
        # Initialize dataset profile
        profile = DatasetProfile(
            dataset_name=dataset_name,
            total_rows=len(data),
            total_columns=len(data[0].keys()) if data else 0
        )
        
        # Auto-detect column types if not specified
        if numerical_columns is None or categorical_columns is None:
            numerical_columns, categorical_columns = self._detect_column_types(data)
        
        # Profile each column
        all_columns = set(data[0].keys()) if data else set()
        
        for column in all_columns:
            if column in numerical_columns:
                profile.column_profiles[column] = self._profile_numerical_column(
                    data, column
                )
            elif column in categorical_columns:
                profile.column_profiles[column] = self._profile_categorical_column(
                    data, column
                )
            else:
                # Generic profile
                profile.column_profiles[column] = self._profile_generic_column(
                    data, column
                )
        
        # Calculate dataset-level metrics
        profile.overall_completeness = self._calculate_overall_completeness(
            profile.column_profiles
        )
        profile.overall_quality_score = self._calculate_overall_quality(
            profile.column_profiles
        )
        
        # Detect dataset-level issues
        profile.critical_issues = self._detect_critical_issues(profile)
        profile.warnings = self._detect_warnings(profile)
        
        # Calculate correlations for numerical columns
        if len(numerical_columns) > 1:
            profile.correlations = self._calculate_correlations(
                data, numerical_columns
            )
        
        # Cache profile
        self.profile_cache[dataset_name] = profile
        
        return profile
    
    def _detect_column_types(
        self,
        data: List[Dict]
    ) -> tuple[List[str], List[str]]:
        """Auto-detect numerical vs categorical columns."""
        numerical = []
        categorical = []
        
        if not data:
            return numerical, categorical
        
        for column in data[0].keys():
            # Sample first non-null value
            sample_val = next(
                (row[column] for row in data if row.get(column) is not None),
                None
            )
            
            if sample_val is None:
                continue
            
            if isinstance(sample_val, (int, float)):
                numerical.append(column)
            else:
                categorical.append(column)
        
        return numerical, categorical
    
    def _profile_numerical_column(
        self,
        data: List[Dict],
        column: str
    ) -> ColumnProfile:
        """Generate statistical profile for numerical column."""
        
        # Extract values (non-null)
        values = [
            float(row[column]) for row in data 
            if row.get(column) is not None and row[column] != ''
        ]
        
        total_count = len(data)
        non_null_count = len(values)
        null_count = total_count - non_null_count
        
        profile = ColumnProfile(
            column_name=column,
            data_type="numerical",
            total_count=total_count,
            null_count=null_count,
            null_percentage=(null_count / total_count * 100) if total_count > 0 else 0,
            unique_count=len(set(values)),
            unique_percentage=(len(set(values)) / total_count * 100) if total_count > 0 else 0
        )
        
        if values:
            # Descriptive statistics
            profile.min_value = min(values)
            profile.max_value = max(values)
            profile.mean = statistics.mean(values)
            profile.median = statistics.median(values)
            
            if len(values) > 1:
                profile.std_dev = statistics.stdev(values)
                profile.variance = statistics.variance(values)
            
            # Quartiles
            sorted_vals = sorted(values)
            profile.q1 = sorted_vals[len(sorted_vals) // 4]
            profile.q3 = sorted_vals[3 * len(sorted_vals) // 4]
            profile.iqr = profile.q3 - profile.q1
            
            # Outlier detection (IQR method)
            lower_bound = profile.q1 - 1.5 * profile.iqr
            upper_bound = profile.q3 + 1.5 * profile.iqr
            outliers = [v for v in values if v < lower_bound or v > upper_bound]
            profile.outlier_count = len(outliers)
            profile.outlier_percentage = (len(outliers) / len(values) * 100)
            
            # Quality flags
            profile.has_negatives = any(v < 0 for v in values)
            profile.has_zeros = any(v == 0 for v in values)
            
            # Quality score (0-100)
            profile.quality_score = self._calculate_column_quality_score(profile)
            
            # Sample values
            profile.sample_values = values[:10]
        
        return profile
    
    def _profile_categorical_column(
        self,
        data: List[Dict],
        column: str
    ) -> ColumnProfile:
        """Generate profile for categorical column."""
        
        values = [
            str(row[column]) for row in data 
            if row.get(column) is not None and row[column] != ''
        ]
        
        total_count = len(data)
        
        profile = ColumnProfile(
            column_name=column,
            data_type="categorical",
            total_count=total_count,
            null_count=total_count - len(values),
            null_percentage=((total_count - len(values)) / total_count * 100) if total_count > 0 else 0
        )
        
        if values:
            # Count unique values
            profile.unique_count = len(set(values))
            profile.unique_percentage = (profile.unique_count / total_count * 100)
            profile.cardinality = profile.unique_count
            
            # Most common values
            counter = Counter(values)
            profile.most_common_values = counter.most_common(10)
            
            # Check for duplicates
            profile.has_duplicates = len(values) > profile.unique_count
            
            # Quality score
            profile.quality_score = self._calculate_column_quality_score(profile)
            
            # Sample values
            profile.sample_values = list(set(values))[:10]
        
        return profile
    
    def _profile_generic_column(
        self,
        data: List[Dict],
        column: str
    ) -> ColumnProfile:
        """Generic profile for unknown column types."""
        
        values = [row.get(column) for row in data if row.get(column) is not None]
        
        return ColumnProfile(
            column_name=column,
            data_type="unknown",
            total_count=len(data),
            null_count=len(data) - len(values),
            null_percentage=((len(data) - len(values)) / len(data) * 100) if data else 0,
            unique_count=len(set(str(v) for v in values)),
            sample_values=[str(v) for v in values[:10]]
        )
    
    def _calculate_column_quality_score(self, profile: ColumnProfile) -> float:
        """
        Calculate quality score (0-100) for a column.
        
        Scoring factors:
        - Completeness (40 points): % of non-null values
        - Validity (30 points): % of non-outlier values
        - Uniqueness (20 points): Appropriate uniqueness for column type
        - Consistency (10 points): Low variance in expected patterns
        """
        score = 0.0
        
        # Completeness (40 points)
        completeness = (100 - profile.null_percentage) / 100
        score += completeness * 40
        
        # Validity (30 points) - based on outlier percentage
        if profile.outlier_percentage is not None:
            validity = (100 - min(profile.outlier_percentage, 100)) / 100
            score += validity * 30
        else:
            score += 30  # No outliers detected (default pass)
        
        # Uniqueness (20 points)
        # For numerical: moderate uniqueness is good
        # For categorical: depends on cardinality
        if profile.data_type == "numerical":
            if 10 <= profile.unique_percentage <= 90:
                score += 20
            else:
                score += 10
        elif profile.data_type == "categorical":
            if profile.cardinality > 0:
                score += 20
            else:
                score += 5
        else:
            score += 15
        
        # Consistency (10 points) - based on presence of expected patterns
        if profile.data_type == "numerical":
            # Numerical data should have reasonable distribution
            if profile.std_dev is not None and profile.mean is not None:
                cv = profile.std_dev / abs(profile.mean) if profile.mean != 0 else float('inf')
                if cv < 2.0:  # Coefficient of variation < 200%
                    score += 10
                else:
                    score += 5
            else:
                score += 5
        else:
            score += 10
        
        return min(100.0, score)
    
    def _calculate_overall_completeness(
        self,
        column_profiles: Dict[str, ColumnProfile]
    ) -> float:
        """Calculate overall dataset completeness percentage."""
        if not column_profiles:
            return 0.0
        
        total_cells = sum(p.total_count for p in column_profiles.values())
        null_cells = sum(p.null_count for p in column_profiles.values())
        
        return ((total_cells - null_cells) / total_cells * 100) if total_cells > 0 else 0.0
    
    def _calculate_overall_quality(
        self,
        column_profiles: Dict[str, ColumnProfile]
    ) -> float:
        """Calculate overall dataset quality score (0-100)."""
        if not column_profiles:
            return 0.0
        
        scores = [p.quality_score for p in column_profiles.values()]
        return statistics.mean(scores) if scores else 0.0
    
    def _detect_critical_issues(self, profile: DatasetProfile) -> List[str]:
        """Detect critical data quality issues."""
        issues = []
        
        # Check for columns with >50% nulls
        for col_name, col_profile in profile.column_profiles.items():
            if col_profile.null_percentage > 50:
                issues.append(
                    f"Column '{col_name}' has {col_profile.null_percentage:.1f}% null values"
                )
        
        # Check for columns with all same values (except nulls)
        for col_name, col_profile in profile.column_profiles.items():
            if col_profile.unique_count == 1 and col_profile.total_count > 1:
                issues.append(
                    f"Column '{col_name}' has only one unique value"
                )
        
        # Check overall quality
        if profile.overall_quality_score < 60:
            issues.append(
                f"Overall quality score is low: {profile.overall_quality_score:.1f}/100"
            )
        
        return issues
    
    def _detect_warnings(self, profile: DatasetProfile) -> List[str]:
        """Detect data quality warnings."""
        warnings = []
        
        # Check for high outlier percentages
        for col_name, col_profile in profile.column_profiles.items():
            if col_profile.outlier_percentage > 10:
                warnings.append(
                    f"Column '{col_name}' has {col_profile.outlier_percentage:.1f}% outliers"
                )
        
        # Check for low uniqueness in expected unique columns
        for col_name, col_profile in profile.column_profiles.items():
            if 'id' in col_name.lower() and col_profile.unique_percentage < 95:
                warnings.append(
                    f"ID column '{col_name}' has duplicates ({col_profile.unique_percentage:.1f}% unique)"
                )
        
        return warnings
    
    def _calculate_correlations(
        self,
        data: List[Dict],
        numerical_columns: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate pairwise correlations for numerical columns."""
        correlations = {}
        
        # Simplified correlation (Pearson) - placeholder
        # In production, would use numpy/scipy
        for col1 in numerical_columns:
            correlations[col1] = {}
            for col2 in numerical_columns:
                if col1 == col2:
                    correlations[col1][col2] = 1.0
                else:
                    # Placeholder - would calculate actual correlation
                    correlations[col1][col2] = 0.0
        
        return correlations
    
    def compare_profiles(
        self,
        profile1: DatasetProfile,
        profile2: DatasetProfile
    ) -> Dict[str, Any]:
        """
        Compare two dataset profiles to detect drift/changes.
        
        Useful for:
        - Detecting data drift over time
        - Validating new data batches
        - Monitoring data quality trends
        """
        comparison = {
            "dataset1": profile1.dataset_name,
            "dataset2": profile2.dataset_name,
            "row_count_change": profile2.total_rows - profile1.total_rows,
            "quality_score_change": profile2.overall_quality_score - profile1.overall_quality_score,
            "column_changes": {},
            "drift_detected": False
        }
        
        # Compare common columns
        common_columns = set(profile1.column_profiles.keys()) & set(profile2.column_profiles.keys())
        
        for column in common_columns:
            col1 = profile1.column_profiles[column]
            col2 = profile2.column_profiles[column]
            
            col_comparison = {
                "null_percentage_change": col2.null_percentage - col1.null_percentage,
                "quality_score_change": col2.quality_score - col1.quality_score
            }
            
            # For numerical columns, check statistical drift
            if col1.data_type == "numerical" and col2.data_type == "numerical":
                if col1.mean is not None and col2.mean is not None:
                    col_comparison["mean_change_pct"] = (
                        (col2.mean - col1.mean) / abs(col1.mean) * 100 
                        if col1.mean != 0 else 0
                    )
                    
                    # Detect significant drift (>20% change in mean)
                    if abs(col_comparison["mean_change_pct"]) > 20:
                        col_comparison["drift_detected"] = True
                        comparison["drift_detected"] = True
            
            comparison["column_changes"][column] = col_comparison
        
        return comparison


# Singleton instance
_profiler_instance: Optional[StatisticalDataProfiler] = None


def get_data_profiler() -> StatisticalDataProfiler:
    """Get or create singleton data profiler instance."""
    global _profiler_instance
    
    if _profiler_instance is None:
        _profiler_instance = StatisticalDataProfiler()
    
    return _profiler_instance


if __name__ == "__main__":
    # Example usage
    profiler = get_data_profiler()
    
    # Sample price data
    sample_data = [
        {'symbol': 'AAPL', 'open': 150.0, 'high': 152.0, 'low': 149.0, 'close': 151.0, 'volume': 1000000},
        {'symbol': 'AAPL', 'open': 151.0, 'high': 153.0, 'low': 150.5, 'close': 152.5, 'volume': 1100000},
        {'symbol': 'AAPL', 'open': 152.5, 'high': 154.0, 'low': 151.0, 'close': 153.0, 'volume': 900000},
    ]
    
    profile = profiler.profile_dataset(
        sample_data,
        "AAPL_Price_Data",
        numerical_columns=['open', 'high', 'low', 'close', 'volume'],
        categorical_columns=['symbol']
    )
    
    print(f"Dataset: {profile.dataset_name}")
    print(f"Rows: {profile.total_rows}, Columns: {profile.total_columns}")
    print(f"Overall Completeness: {profile.overall_completeness:.1f}%")
    print(f"Overall Quality Score: {profile.overall_quality_score:.1f}/100")
    
    if profile.critical_issues:
        print(f"\n❌ Critical Issues ({len(profile.critical_issues)}):")
        for issue in profile.critical_issues:
            print(f"  - {issue}")
    
    if profile.warnings:
        print(f"\n⚠️  Warnings ({len(profile.warnings)}):")
        for warning in profile.warnings:
            print(f"  - {warning}")
    
    print("\n✅ Data profiling complete!")