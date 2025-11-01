"""
Feature Data Integration Layer.

Connects Feature Store to PostgreSQL for persistence.
Enables:
- Feature computation and storage
- Feature versioning
- Feature retrieval for ML
- Feature quality tracking
"""

import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from decimal import Decimal

import numpy as np
import pandas as pd

from .models import FeatureData
from .session import SessionManager

logger = logging.getLogger(__name__)


class FeatureIntegration:
    """
    Integration layer for feature data persistence.
    
    Connects Feature Store with PostgreSQL to:
    - Store computed features
    - Retrieve features for ML training
    - Track feature quality
    - Version features
    """
    
    def __init__(self, session_manager: Optional[SessionManager] = None):
        """Initialize feature integration."""
        self.session = session_manager or SessionManager()
    
    def store_feature(
        self,
        symbol: str,
        timestamp: datetime,
        feature_name: str,
        value: float,
        feature_category: str = "technical",
        feature_version: str = "1.0.0",
        computation_method: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        quality_score: Optional[float] = None,
        source_table: str = "price_data",
        source_ids: Optional[List[int]] = None
    ) -> FeatureData:
        """
        Store a single computed feature.
        
        Args:
            symbol: Asset symbol
            timestamp: Feature timestamp
            feature_name: Name of feature (e.g., 'sma_50')
            value: Feature value
            feature_category: Category ('technical', 'fundamental', 'derived')
            feature_version: Feature version
            computation_method: Method used to compute
            parameters: Parameters used
            quality_score: Quality score (0-100)
            source_table: Source data table
            source_ids: Source record IDs
            
        Returns:
            Stored FeatureData object
        """
        feature = FeatureData(
            symbol=symbol,
            timestamp=timestamp,
            feature_name=feature_name,
            feature_category=feature_category,
            feature_version=feature_version,
            value=value,
            computation_method=computation_method,
            parameters=parameters,
            quality_score=quality_score,
            source_table=source_table,
            source_ids=source_ids,
            is_validated=quality_score is not None and quality_score >= 70,
            validation_status='passed' if quality_score and quality_score >= 70 else 'pending'
        )
        
        self.session.add(feature)
        self.session.commit()
        
        logger.debug(f"Stored feature {feature_name} for {symbol} at {timestamp}")
        
        return feature
    
    def bulk_store_features(
        self,
        features_df: pd.DataFrame,
        feature_category: str = "technical",
        feature_version: str = "1.0.0",
        source_table: str = "price_data"
    ) -> int:
        """
        Bulk store features from DataFrame.
        
        DataFrame should have:
        - Index: timestamp
        - Columns: symbol, feature1, feature2, ...
        
        Args:
            features_df: DataFrame with features
            feature_category: Category of features
            feature_version: Feature version
            source_table: Source data table
            
        Returns:
            Number of features stored
        """
        features = []
        
        for timestamp, row in features_df.iterrows():
            symbol = row.get('symbol', 'UNKNOWN')
            
            for col in features_df.columns:
                if col == 'symbol':
                    continue
                
                value = row[col]
                if pd.isna(value):
                    continue
                
                feature = FeatureData(
                    symbol=symbol,
                    timestamp=timestamp if isinstance(timestamp, datetime) else datetime.fromisoformat(str(timestamp)),
                    feature_name=col,
                    feature_category=feature_category,
                    feature_version=feature_version,
                    value=float(value),
                    source_table=source_table,
                )
                features.append(feature)
        
        self.session.bulk_insert(features)
        self.session.commit()
        
        logger.info(f"Bulk stored {len(features)} features")
        
        return len(features)
    
    def get_features(
        self,
        symbol: str,
        feature_names: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        feature_version: str = "1.0.0"
    ) -> pd.DataFrame:
        """
        Retrieve features from database.
        
        Args:
            symbol: Asset symbol
            feature_names: List of feature names (None = all)
            start_date: Start date filter
            end_date: End date filter
            feature_version: Feature version
            
        Returns:
            DataFrame with features (timestamp index, feature columns)
        """
        query = self.session.query(FeatureData).filter(
            FeatureData.symbol == symbol,
            FeatureData.feature_version == feature_version
        )
        
        if feature_names:
            query = query.filter(FeatureData.feature_name.in_(feature_names))
        
        if start_date:
            query = query.filter(FeatureData.timestamp >= start_date)
        
        if end_date:
            query = query.filter(FeatureData.timestamp <= end_date)
        
        features = query.all()
        
        if not features:
            return pd.DataFrame()
        
        # Convert to DataFrame
        data = []
        for f in features:
            data.append({
                'timestamp': f.timestamp,
                'feature_name': f.feature_name,
                'value': f.value
            })
        
        df = pd.DataFrame(data)
        
        # Pivot to wide format (timestamp x features)
        df_pivot = df.pivot(index='timestamp', columns='feature_name', values='value')
        
        logger.info(f"Retrieved {len(df_pivot)} rows, {len(df_pivot.columns)} features for {symbol}")
        
        return df_pivot
    
    def get_latest_features(
        self,
        symbol: str,
        feature_names: List[str],
        feature_version: str = "1.0.0"
    ) -> Dict[str, float]:
        """
        Get latest feature values for a symbol.
        
        Args:
            symbol: Asset symbol
            feature_names: List of feature names
            feature_version: Feature version
            
        Returns:
            Dictionary of {feature_name: value}
        """
        # Get latest timestamp for each feature
        results = {}
        
        for feature_name in feature_names:
            latest = self.session.query(FeatureData).filter(
                FeatureData.symbol == symbol,
                FeatureData.feature_name == feature_name,
                FeatureData.feature_version == feature_version
            ).order_by(FeatureData.timestamp.desc()).first()
            
            if latest:
                results[feature_name] = latest.value
        
        logger.info(f"Retrieved {len(results)} latest features for {symbol}")
        
        return results
    
    def delete_features(
        self,
        symbol: str,
        feature_names: Optional[List[str]] = None,
        before_date: Optional[datetime] = None
    ) -> int:
        """
        Delete features from database.
        
        Args:
            symbol: Asset symbol
            feature_names: Feature names to delete (None = all)
            before_date: Delete features before this date
            
        Returns:
            Number of features deleted
        """
        query = self.session.query(FeatureData).filter(
            FeatureData.symbol == symbol
        )
        
        if feature_names:
            query = query.filter(FeatureData.feature_name.in_(feature_names))
        
        if before_date:
            query = query.filter(FeatureData.timestamp < before_date)
        
        count = query.delete()
        self.session.commit()
        
        logger.info(f"Deleted {count} features for {symbol}")
        
        return count


# Export
__all__ = [
    "FeatureIntegration",
]