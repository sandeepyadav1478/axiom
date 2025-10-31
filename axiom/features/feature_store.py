"""
Feature Store - Production-Grade Feature Management

Centralized feature storage and serving for ML models.
Manages feature engineering, versioning, and real-time serving.

Critical for:
- Model performance (features make or break models)
- Consistency across training/serving
- Feature reusability
- Version control

This is essential for institutional-grade ML operations.
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hashlib
import json


class FeatureType(Enum):
    """Types of features in the store."""
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    TIMESTAMP = "timestamp"
    TEXT = "text"
    EMBEDDING = "embedding"


class ComputationType(Enum):
    """How feature is computed."""
    BATCH = "batch"           # Computed in batch jobs
    ONLINE = "online"         # Computed in real-time
    HYBRID = "hybrid"         # Both batch and online
    PRECOMPUTED = "precomputed"  # Stored, not computed


@dataclass
class FeatureDefinition:
    """
    Definition of a single feature.
    
    Includes all metadata needed for feature management:
    - Computation logic
    - Data types
    - Versioning
    - Dependencies
    """
    
    name: str
    description: str
    feature_type: FeatureType
    computation_type: ComputationType
    
    # Computation
    compute_function: Optional[Callable] = None
    dependencies: List[str] = field(default_factory=list)  # Other features needed
    source_tables: List[str] = field(default_factory=list)  # Data sources
    
    # Metadata
    version: str = "1.0"
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = "axiom"
    
    # Configuration
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Quality
    expected_range: Optional[tuple] = None  # (min, max) for numerical
    expected_values: Optional[List] = None  # For categorical
    null_allowed: bool = True
    
    # Performance
    computation_cost: str = "low"  # low, medium, high
    latency_ms: Optional[float] = None
    
    # Documentation
    example_value: Optional[Any] = None
    use_cases: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    
    def get_feature_id(self) -> str:
        """Generate unique feature ID from name and version."""
        return f"{self.name}_v{self.version}"
    
    def get_hash(self) -> str:
        """Generate hash for feature definition (for change detection)."""
        content = f"{self.name}{self.version}{str(self.compute_function)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            "name": self.name,
            "description": self.description,
            "type": self.feature_type.value,
            "computation_type": self.computation_type.value,
            "version": self.version,
            "dependencies": self.dependencies,
            "source_tables": self.source_tables,
            "parameters": self.parameters,
            "expected_range": self.expected_range,
            "expected_values": self.expected_values,
            "null_allowed": self.null_allowed,
            "computation_cost": self.computation_cost,
            "tags": self.tags,
            "use_cases": self.use_cases,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class FeatureValue:
    """Computed feature value with metadata."""
    
    feature_name: str
    value: Any
    computed_at: datetime
    entity_id: str  # Entity this feature belongs to (e.g., symbol, user_id)
    version: str = "1.0"
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "feature_name": self.feature_name,
            "value": self.value,
            "entity_id": self.entity_id,
            "version": self.version,
            "confidence": self.confidence,
            "computed_at": self.computed_at.isoformat(),
            "metadata": self.metadata
        }


class FeatureStore:
    """
    Centralized feature store for ML features.
    
    Manages:
    - Feature registration
    - Feature computation
    - Feature storage
    - Feature versioning
    - Feature serving (batch and online)
    
    Benefits:
    - Consistency across train/serve
    - Feature reusability
    - Version control
    - Performance optimization
    """
    
    def __init__(self):
        """Initialize feature store."""
        # Feature registry
        self.features: Dict[str, FeatureDefinition] = {}
        
        # Feature values cache (in-memory, would be Redis/DB in production)
        self.feature_cache: Dict[str, Dict[str, FeatureValue]] = {}
        
        # Feature groups (logical grouping)
        self.feature_groups: Dict[str, List[str]] = {}
        
        # Computation statistics
        self.computation_stats: Dict[str, Dict] = {}
    
    def register_feature(
        self,
        feature: FeatureDefinition
    ) -> None:
        """
        Register feature in the store.
        
        Args:
            feature: Feature definition to register
        """
        feature_id = feature.get_feature_id()
        
        if feature_id in self.features:
            # Check if definition changed
            existing_hash = self.features[feature_id].get_hash()
            new_hash = feature.get_hash()
            
            if existing_hash != new_hash:
                # Feature definition changed - version bump needed
                raise ValueError(
                    f"Feature definition changed but version not bumped: {feature.name}"
                )
        
        self.features[feature_id] = feature
        
        # Initialize cache for this feature
        if feature_id not in self.feature_cache:
            self.feature_cache[feature_id] = {}
    
    def compute_feature(
        self,
        feature_name: str,
        entity_id: str,
        input_data: Dict[str, Any],
        version: str = "1.0"
    ) -> FeatureValue:
        """
        Compute feature value for entity.
        
        Args:
            feature_name: Name of feature to compute
            entity_id: Entity identifier (e.g., 'AAPL')
            input_data: Input data for computation
            version: Feature version
        
        Returns:
            Computed feature value
        """
        feature_id = f"{feature_name}_v{version}"
        
        if feature_id not in self.features:
            raise ValueError(f"Feature not registered: {feature_id}")
        
        feature_def = self.features[feature_id]
        
        # Check if already cached
        cache_key = f"{entity_id}_{feature_name}"
        if cache_key in self.feature_cache.get(feature_id, {}):
            cached = self.feature_cache[feature_id][cache_key]
            # Check if cache is fresh (< 1 hour for demo, configurable in production)
            if (datetime.now() - cached.computed_at).seconds < 3600:
                return cached
        
        # Compute feature
        if feature_def.compute_function is None:
            raise ValueError(f"No compute function for feature: {feature_name}")
        
        start_time = datetime.now()
        
        try:
            value = feature_def.compute_function(input_data)
        except Exception as e:
            raise RuntimeError(f"Feature computation failed: {feature_name}: {e}")
        
        computation_time = (datetime.now() - start_time).total_seconds() * 1000  # ms
        
        # Create feature value
        feature_value = FeatureValue(
            feature_name=feature_name,
            value=value,
            entity_id=entity_id,
            version=version,
            computed_at=datetime.now(),
            metadata={"computation_time_ms": computation_time}
        )
        
        # Cache feature value
        if feature_id not in self.feature_cache:
            self.feature_cache[feature_id] = {}
        self.feature_cache[feature_id][cache_key] = feature_value
        
        # Update stats
        self._update_computation_stats(feature_id, computation_time)
        
        return feature_value
    
    def compute_feature_vector(
        self,
        feature_names: List[str],
        entity_id: str,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compute multiple features at once (feature vector).
        
        Args:
            feature_names: List of features to compute
            entity_id: Entity identifier
            input_data: Input data for computation
        
        Returns:
            Dictionary of feature_name -> value
        """
        feature_vector = {}
        
        for feature_name in feature_names:
            feature_value = self.compute_feature(
                feature_name, entity_id, input_data
            )
            feature_vector[feature_name] = feature_value.value
        
        return feature_vector
    
    def get_feature_group(
        self,
        group_name: str,
        entity_id: str,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get all features in a logical group.
        
        Args:
            group_name: Name of feature group
            entity_id: Entity identifier
            input_data: Input data
        
        Returns:
            Dictionary of all features in group
        """
        if group_name not in self.feature_groups:
            raise ValueError(f"Feature group not found: {group_name}")
        
        feature_names = self.feature_groups[group_name]
        return self.compute_feature_vector(feature_names, entity_id, input_data)
    
    def register_feature_group(
        self,
        group_name: str,
        feature_names: List[str]
    ) -> None:
        """Register a logical group of related features."""
        self.feature_groups[group_name] = feature_names
    
    def list_features(
        self,
        tags: Optional[List[str]] = None,
        feature_type: Optional[FeatureType] = None
    ) -> List[FeatureDefinition]:
        """
        List features with optional filtering.
        
        Args:
            tags: Filter by tags
            feature_type: Filter by type
        
        Returns:
            List of matching feature definitions
        """
        features = list(self.features.values())
        
        if tags:
            features = [
                f for f in features 
                if any(tag in f.tags for tag in tags)
            ]
        
        if feature_type:
            features = [
                f for f in features 
                if f.feature_type == feature_type
            ]
        
        return features
    
    def get_feature_stats(self, feature_name: str) -> Dict[str, Any]:
        """Get computation statistics for feature."""
        feature_id = f"{feature_name}_v1.0"  # Default version
        return self.computation_stats.get(feature_id, {
            "total_computations": 0,
            "avg_latency_ms": 0,
            "cache_hits": 0,
            "cache_misses": 0
        })
    
    def _update_computation_stats(
        self,
        feature_id: str,
        computation_time_ms: float
    ) -> None:
        """Update feature computation statistics."""
        if feature_id not in self.computation_stats:
            self.computation_stats[feature_id] = {
                "total_computations": 0,
                "total_time_ms": 0,
                "avg_latency_ms": 0
            }
        
        stats = self.computation_stats[feature_id]
        stats["total_computations"] += 1
        stats["total_time_ms"] += computation_time_ms
        stats["avg_latency_ms"] = stats["total_time_ms"] / stats["total_computations"]


# Singleton instance
_feature_store: Optional[FeatureStore] = None


def get_feature_store() -> FeatureStore:
    """Get or create singleton feature store instance."""
    global _feature_store
    
    if _feature_store is None:
        _feature_store = FeatureStore()
    
    return _feature_store


if __name__ == "__main__":
    # Example: Register and use features
    store = get_feature_store()
    
    # Define a simple feature
    def compute_price_momentum(data):
        """Compute price momentum (close - open) / open"""
        return (data['close'] - data['open']) / data['open']
    
    momentum_feature = FeatureDefinition(
        name="price_momentum",
        description="Intraday price momentum",
        feature_type=FeatureType.NUMERICAL,
        computation_type=ComputationType.ONLINE,
        compute_function=compute_price_momentum,
        source_tables=["price_data"],
        tags=["price", "momentum", "technical"],
        use_cases=["prediction", "signal_generation"]
    )
    
    store.register_feature(momentum_feature)
    
    # Compute feature
    price_data = {'open': 150.0, 'close': 152.0}
    result = store.compute_feature("price_momentum", "AAPL", price_data)
    
    print(f"Feature: {result.feature_name}")
    print(f"Value: {result.value:.4f}")
    print(f"Computed at: {result.computed_at}")
    
    # Get stats
    stats = store.get_feature_stats("price_momentum")
    print(f"Stats: {stats}")
    
    print("\n✅ Feature store operational!")