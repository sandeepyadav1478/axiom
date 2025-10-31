"""
Data Exploration & Discovery System

Interactive data exploration and profiling tools for analysts.
Helps discover patterns, relationships, and insights in data.

Features:
- Automated data profiling
- Correlation analysis
- Distribution visualization
- Outlier identification
- Pattern detection
- Data summarization

Critical for understanding data before modeling.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import statistics


@dataclass
class DataSummary:
    """Summary statistics for a dataset."""
    
    dataset_name: str
    total_records: int
    total_fields: int
    
    # Field summaries
    numerical_fields: List[str] = field(default_factory=list)
    categorical_fields: List[str] = field(default_factory=list)
    datetime_fields: List[str] = field(default_factory=list)
    
    # Quick stats
    missing_values_pct: float = 0.0
    duplicate_records_pct: float = 0.0
    
    # Time range (if temporal data)
    earliest_date: Optional[datetime] = None
    latest_date: Optional[datetime] = None
    date_range_days: Optional[int] = None
    
    # Generated at
    explored_at: datetime = field(default_factory=datetime.now)


@dataclass
class CorrelationMatrix:
    """Correlation matrix for numerical fields."""
    
    fields: List[str]
    correlations: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    def get_high_correlations(self, threshold: float = 0.7) -> List[tuple]:
        """Get pairs with high correlation."""
        high_corr = []
        
        for field1 in self.fields:
            for field2 in self.fields:
                if field1 < field2:  # Avoid duplicates
                    corr = self.correlations.get(field1, {}).get(field2, 0)
                    if abs(corr) >= threshold:
                        high_corr.append((field1, field2, corr))
        
        return sorted(high_corr, key=lambda x: abs(x[2]), reverse=True)


class DataExplorer:
    """
    Data exploration and discovery tool.
    
    Provides interactive data exploration capabilities:
    - Quick summaries
    - Statistical analysis
    - Correlation analysis
    - Pattern detection
    - Data quality insights
    
    Helps analysts understand data before modeling.
    """
    
    def __init__(self):
        """Initialize data explorer."""
        self.exploration_cache: Dict[str, DataSummary] = {}
    
    def explore_dataset(
        self,
        data: List[Dict[str, Any]],
        dataset_name: str
    ) -> DataSummary:
        """
        Generate comprehensive summary of dataset.
        
        Args:
            data: Dataset to explore
            dataset_name: Name for this dataset
        
        Returns:
            Complete data summary
        """
        if not data:
            return DataSummary(dataset_name, 0, 0)
        
        summary = DataSummary(
            dataset_name=dataset_name,
            total_records=len(data),
            total_fields=len(data[0].keys()) if data else 0
        )
        
        # Categorize fields
        for field, value in data[0].items():
            if isinstance(value, (int, float)):
                summary.numerical_fields.append(field)
            elif isinstance(value, str):
                # Check if datetime
                if self._is_datetime_field(field, data):
                    summary.datetime_fields.append(field)
                else:
                    summary.categorical_fields.append(field)
        
        # Calculate missing values
        total_cells = len(data) * len(data[0].keys())
        missing_cells = sum(
            1 for record in data
            for value in record.values()
            if value is None or value == ''
        )
        summary.missing_values_pct = (missing_cells / total_cells * 100) if total_cells > 0 else 0
        
        # Find duplicates
        seen_records = set()
        duplicates = 0
        
        for record in data:
            record_str = str(sorted(record.items()))
            if record_str in seen_records:
                duplicates += 1
            seen_records.add(record_str)
        
        summary.duplicate_records_pct = (duplicates / len(data) * 100) if data else 0
        
        # Time range (if has timestamp)
        if 'timestamp' in data[0]:
            dates = [
                datetime.fromisoformat(str(r['timestamp']).replace('Z', '+00:00'))
                for r in data if r.get('timestamp')
            ]
            if dates:
                summary.earliest_date = min(dates)
                summary.latest_date = max(dates)
                summary.date_range_days = (summary.latest_date - summary.earliest_date).days
        
        # Cache summary
        self.exploration_cache[dataset_name] = summary
        
        return summary
    
    def compute_correlations(
        self,
        data: List[Dict[str, Any]],
        fields: Optional[List[str]] = None
    ) -> CorrelationMatrix:
        """
        Compute pairwise correlations for numerical fields.
        
        Args:
            data: Dataset
            fields: Specific fields to correlate (None = all numerical)
        
        Returns:
            Correlation matrix
        """
        # Get numerical fields
        if fields is None:
            fields = [
                k for k, v in data[0].items()
                if isinstance(v, (int, float))
            ]
        
        matrix = CorrelationMatrix(fields=fields)
        
        # Compute correlations (simplified Pearson)
        for field1 in fields:
            matrix.correlations[field1] = {}
            
            for field2 in fields:
                if field1 == field2:
                    matrix.correlations[field1][field2] = 1.0
                else:
                    # Extract values
                    values1 = [float(r.get(field1, 0)) for r in data if r.get(field1) is not None]
                    values2 = [float(r.get(field2, 0)) for r in data if r.get(field2) is not None]
                    
                    # Calculate correlation (simplified)
                    if len(values1) > 1 and len(values2) > 1:
                        corr = self._pearson_correlation(values1, values2)
                        matrix.correlations[field1][field2] = corr
                    else:
                        matrix.correlations[field1][field2] = 0.0
        
        return matrix
    
    def _pearson_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient."""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        # Calculate means
        mean_x = statistics.mean(x)
        mean_y = statistics.mean(y)
        
        # Calculate correlation
        numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        
        sum_sq_x = sum((xi - mean_x) ** 2 for xi in x)
        sum_sq_y = sum((yi - mean_y) ** 2 for yi in y)
        
        denominator = (sum_sq_x * sum_sq_y) ** 0.5
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def _is_datetime_field(self, field: str, data: List[Dict]) -> bool:
        """Check if field contains datetime values."""
        if 'date' in field.lower() or 'time' in field.lower():
            return True
        
        # Check first value
        try:
            value = next((r[field] for r in data if r.get(field)), None)
            if value and isinstance(value, str):
                datetime.fromisoformat(value.replace('Z', '+00:00'))
                return True
        except:
            pass
        
        return False
    
    def find_patterns(
        self,
        data: List[Dict[str, Any]],
        field: str
    ) -> Dict[str, Any]:
        """
        Find patterns in a specific field.
        
        Args:
            data: Dataset
            field: Field to analyze
        
        Returns:
            Pattern analysis results
        """
        values = [r.get(field) for r in data if r.get(field) is not None]
        
        if not values:
            return {"pattern": "no_data"}
        
        # Check if numerical
        if isinstance(values[0], (int, float)):
            return self._find_numerical_patterns(values)
        else:
            return self._find_categorical_patterns(values)
    
    def _find_numerical_patterns(self, values: List[float]) -> Dict:
        """Find patterns in numerical data."""
        
        if len(values) < 2:
            return {"pattern": "insufficient_data"}
        
        # Calculate trend
        trend = "increasing" if values[-1] > values[0] else "decreasing" if values[-1] < values[0] else "stable"
        
        # Calculate volatility
        returns = [(values[i] - values[i-1]) / values[i-1] for i in range(1, len(values)) if values[i-1] != 0]
        volatility = statistics.stdev(returns) if len(returns) > 1 else 0
        
        return {
            "pattern": "numerical",
            "trend": trend,
            "volatility": volatility,
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0,
            "range": (min(values), max(values))
        }
    
    def _find_categorical_patterns(self, values: List[Any]) -> Dict:
        """Find patterns in categorical data."""
        
        from collections import Counter
        
        counts = Counter(values)
        
        return {
            "pattern": "categorical",
            "unique_values": len(counts),
            "most_common": counts.most_common(5),
            "distribution": "uniform" if len(set(counts.values())) == 1 else "skewed"
        }


# Singleton
_explorer: Optional[DataExplorer] = None


def get_data_explorer() -> DataExplorer:
    """Get or create singleton data explorer."""
    global _explorer
    
    if _explorer is None:
        _explorer = DataExplorer()
    
    return _explorer


if __name__ == "__main__":
    # Example usage
    explorer = get_data_explorer()
    
    # Sample data
    sample_data = [
        {'symbol': 'AAPL', 'close': 150.0, 'volume': 1000000, 'timestamp': '2024-10-28'},
        {'symbol': 'AAPL', 'close': 151.0, 'volume': 1100000, 'timestamp': '2024-10-29'},
        {'symbol': 'AAPL', 'close': 153.0, 'volume': 900000, 'timestamp': '2024-10-30'},
        {'symbol': 'AAPL', 'close': 152.0, 'volume': 1050000, 'timestamp': '2024-10-31'},
    ]
    
    print("Data Exploration Demo")
    print("=" * 60)
    
    # Explore dataset
    summary = explorer.explore_dataset(sample_data, "AAPL_Prices")
    
    print(f"\nDataset: {summary.dataset_name}")
    print(f"Records: {summary.total_records}")
    print(f"Fields: {summary.total_fields}")
    print(f"Numerical fields: {summary.numerical_fields}")
    print(f"Categorical fields: {summary.categorical_fields}")
    print(f"DateTime fields: {summary.datetime_fields}")
    print(f"Missing values: {summary.missing_values_pct:.1f}%")
    print(f"Duplicates: {summary.duplicate_records_pct:.1f}%")
    
    if summary.date_range_days:
        print(f"Date range: {summary.date_range_days} days")
    
    # Compute correlations
    corr_matrix = explorer.compute_correlations(sample_data)
    print(f"\nCorrelation Matrix:")
    print(f"Fields: {corr_matrix.fields}")
    
    high_corr = corr_matrix.get_high_correlations(threshold=0.5)
    if high_corr:
        print(f"High correlations (>0.5):")
        for field1, field2, corr in high_corr:
            print(f"  {field1} <-> {field2}: {corr:.2f}")
    
    # Find patterns
    patterns = explorer.find_patterns(sample_data, 'close')
    print(f"\nPatterns in 'close':")
    print(f"  Trend: {patterns.get('trend', 'N/A')}")
    print(f"  Mean: {patterns.get('mean', 0):.2f}")
    print(f"  Volatility: {patterns.get('volatility', 0):.4f}")
    
    print("\n✅ Data exploration complete!")
    print("Comprehensive insights generated for analysis!")