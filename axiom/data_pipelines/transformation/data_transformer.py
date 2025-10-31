"""
Advanced Data Transformation System - Production Grade

Sophisticated data transformations for financial data.
Handles complex transformations, aggregations, and feature derivations.

Transformations:
- Time-series resampling (minute → daily, daily → weekly, etc.)
- Rolling window calculations
- Cross-sectional transformations
- Lag/lead features
- Differencing & returns
- Normalization & scaling

Critical for creating model-ready features from raw data.
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum


class TransformationType(Enum):
    """Types of data transformations."""
    RESAMPLE = "resample"              # Time-series resampling
    ROLLING_WINDOW = "rolling_window"  # Moving calculations
    LAG = "lag"                        # Lagged features
    DIFF = "diff"                      # Differencing
    RETURNS = "returns"                # Price returns
    NORMALIZE = "normalize"            # Normalization/scaling
    AGGREGATE = "aggregate"            # Aggregation
    CROSS_SECTIONAL = "cross_sectional"  # Cross-sectional features


@dataclass
class TransformationRule:
    """Definition of data transformation."""
    
    name: str
    description: str
    transformation_type: TransformationType
    transform_function: Callable
    
    # Configuration
    parameters: Dict[str, Any] = None
    
    # Dependencies
    required_fields: List[str] = None
    output_fields: List[str] = None
    
    def apply(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply transformation to data."""
        try:
            return self.transform_function(data, self.parameters or {})
        except Exception as e:
            raise RuntimeError(f"Transformation '{self.name}' failed: {e}")


class DataTransformer:
    """
    Advanced data transformation engine.
    
    Applies sophisticated transformations to prepare data for analysis and models.
    
    Features:
    - Time-series transformations
    - Rolling window calculations  
    - Feature derivations
    - Normalization
    - Cross-sectional features
    
    Critical for feature engineering!
    """
    
    def __init__(self):
        """Initialize data transformer."""
        self.transformations: List[TransformationRule] = []
        
        # Register built-in transformations
        self._register_builtin_transformations()
    
    def _register_builtin_transformations(self):
        """Register built-in transformation functions."""
        
        # 1. Calculate Returns
        self.register_transformation(
            TransformationRule(
                name="calculate_returns",
                description="Calculate price returns",
                transformation_type=TransformationType.RETURNS,
                transform_function=self._calculate_returns,
                required_fields=['close'],
                output_fields=['returns', 'log_returns']
            )
        )
        
        # 2. Rolling Mean
        self.register_transformation(
            TransformationRule(
                name="rolling_mean",
                description="Calculate rolling mean",
                transformation_type=TransformationType.ROLLING_WINDOW,
                transform_function=self._rolling_mean,
                parameters={'window': 20, 'field': 'close'},
                output_fields=['rolling_mean']
            )
        )
        
        # 3. Create Lag Features
        self.register_transformation(
            TransformationRule(
                name="create_lags",
                description="Create lagged features",
                transformation_type=TransformationType.LAG,
                transform_function=self._create_lags,
                parameters={'field': 'close', 'lags': [1, 2, 5]},
                output_fields=['close_lag_1', 'close_lag_2', 'close_lag_5']
            )
        )
    
    def register_transformation(self, rule: TransformationRule) -> None:
        """Register custom transformation rule."""
        self.transformations.append(rule)
    
    def transform_dataset(
        self,
        data: List[Dict[str, Any]],
        transformation_names: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Apply transformations to dataset.
        
        Args:
            data: Input data
            transformation_names: Specific transformations to apply (None = all)
        
        Returns:
            Transformed data
        """
        if not transformation_names:
            # Apply all transformations
            transformations = self.transformations
        else:
            # Apply specified transformations
            transformations = [
                t for t in self.transformations 
                if t.name in transformation_names
            ]
        
        transformed_data = data.copy()
        
        for transformation in transformations:
            transformed_data = transformation.apply(transformed_data)
        
        return transformed_data
    
    # ========================================================================
    # BUILT-IN TRANSFORMATION FUNCTIONS
    # ========================================================================
    
    @staticmethod
    def _calculate_returns(
        data: List[Dict[str, Any]],
        params: Dict
    ) -> List[Dict[str, Any]]:
        """Calculate simple and log returns."""
        import math
        
        enriched = []
        
        for i in range(len(data)):
            record = data[i].copy()
            
            if i == 0:
                record['returns'] = 0.0
                record['log_returns'] = 0.0
            else:
                curr_price = float(record.get('close', 0))
                prev_price = float(data[i-1].get('close', 1))
                
                if prev_price > 0:
                    record['returns'] = (curr_price - prev_price) / prev_price
                    record['log_returns'] = math.log(curr_price / prev_price) if curr_price > 0 else 0
                else:
                    record['returns'] = 0.0
                    record['log_returns'] = 0.0
            
            enriched.append(record)
        
        return enriched
    
    @staticmethod
    def _rolling_mean(
        data: List[Dict[str, Any]],
        params: Dict
    ) -> List[Dict[str, Any]]:
        """Calculate rolling mean over window."""
        
        window = params.get('window', 20)
        field = params.get('field', 'close')
        
        enriched = []
        
        for i in range(len(data)):
            record = data[i].copy()
            
            # Get window of values
            start_idx = max(0, i - window + 1)
            window_values = [
                float(data[j].get(field, 0))
                for j in range(start_idx, i + 1)
                if data[j].get(field) is not None
            ]
            
            if window_values:
                record['rolling_mean'] = sum(window_values) / len(window_values)
            else:
                record['rolling_mean'] = 0.0
            
            enriched.append(record)
        
        return enriched
    
    @staticmethod
    def _create_lags(
        data: List[Dict[str, Any]],
        params: Dict
    ) -> List[Dict[str, Any]]:
        """Create lagged features."""
        
        field = params.get('field', 'close')
        lags = params.get('lags', [1, 2, 5])
        
        enriched = []
        
        for i in range(len(data)):
            record = data[i].copy()
            
            for lag in lags:
                lag_idx = i - lag
                if lag_idx >= 0:
                    record[f'{field}_lag_{lag}'] = data[lag_idx].get(field)
                else:
                    record[f'{field}_lag_{lag}'] = None
            
            enriched.append(record)
        
        return enriched
    
    @staticmethod
    def resample_ohlcv(
        data: List[Dict[str, Any]],
        target_frequency: str = "1D"  # '1H', '1D', '1W', '1M'
    ) -> List[Dict[str, Any]]:
        """
        Resample OHLCV data to different frequency.
        
        Args:
            data: Input OHLCV data
            target_frequency: Target frequency ('1H', '1D', '1W', '1M')
        
        Returns:
            Resampled OHLCV data
        """
        # Simplified resampling - in production would use pandas
        # Group by date and aggregate
        
        from collections import defaultdict
        
        grouped = defaultdict(list)
        
        for record in data:
            timestamp = record.get('timestamp', '')
            date_key = timestamp[:10]  # YYYY-MM-DD
            grouped[date_key].append(record)
        
        resampled = []
        
        for date_key, records in grouped.items():
            if not records:
                continue
            
            # Aggregate OHLCV
            opens = [r.get('open') for r in records if r.get('open')]
            highs = [r.get('high') for r in records if r.get('high')]
            lows = [r.get('low') for r in records if r.get('low')]
            closes = [r.get('close') for r in records if r.get('close')]
            volumes = [r.get('volume', 0) for r in records]
            
            resampled_record = {
                'symbol': records[0].get('symbol'),
                'timestamp': date_key,
                'open': opens[0] if opens else None,
                'high': max(highs) if highs else None,
                'low': min(lows) if lows else None,
                'close': closes[-1] if closes else None,
                'volume': sum(volumes)
            }
            
            resampled.append(resampled_record)
        
        return resampled
    
    @staticmethod
    def z_score_normalize(
        data: List[Dict[str, Any]],
        fields: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Z-score normalization: (x - mean) / std.
        
        Args:
            data: Input data
            fields: Fields to normalize
        
        Returns:
            Data with normalized fields
        """
        import statistics
        
        normalized = []
        
        for field in fields:
            # Calculate mean and std
            values = [float(r.get(field, 0)) for r in data if r.get(field) is not None]
            
            if len(values) < 2:
                continue
            
            mean = statistics.mean(values)
            std = statistics.stdev(values)
            
            # Normalize
            for record in data:
                if field in record and record[field] is not None:
                    if std > 0:
                        record[f'{field}_zscore'] = (float(record[field]) - mean) / std
                    else:
                        record[f'{field}_zscore'] = 0.0
        
        return data


# Singleton
_transformer: Optional[DataTransformer] = None


def get_data_transformer() -> DataTransformer:
    """Get or create singleton transformer."""
    global _transformer
    
    if _transformer is None:
        _transformer = DataTransformer()
    
    return _transformer


if __name__ == "__main__":
    # Example usage
    transformer = get_data_transformer()
    
    # Sample price data
    sample_data = [
        {'symbol': 'AAPL', 'close': 150.0, 'timestamp': '2024-10-28'},
        {'symbol': 'AAPL', 'close': 151.0, 'timestamp': '2024-10-29'},
        {'symbol': 'AAPL', 'close': 153.0, 'timestamp': '2024-10-30'},
        {'symbol': 'AAPL', 'close': 152.0, 'timestamp': '2024-10-31'},
    ]
    
    print("Data Transformation Demo")
    print("=" * 60)
    
    # Apply transformations
    transformed = transformer.transform_dataset(
        sample_data,
        transformation_names=['calculate_returns', 'create_lags']
    )
    
    print("\n✅ Transformations Applied:")
    for record in transformed:
        print(f"Date: {record['timestamp']}, Close: {record['close']:.2f}, Return: {record.get('returns', 0)*100:.2f}%, Lag_1: {record.get('close_lag_1', 'N/A')}")
    
    print("\n✅ Data transformation system operational!")
    print("Sophisticated transformations ready for feature engineering!")