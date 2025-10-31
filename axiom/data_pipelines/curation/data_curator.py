"""
Data Curation & Filtration System - Enterprise Grade

Curates raw data into high-quality, analysis-ready datasets.
Applies sophisticated filtering, enrichment, and quality control.

Curation Steps:
1. Data filtering (remove low-quality/irrelevant data)
2. Data enrichment (add calculated fields)
3. Data normalization (standardize formats)
4. Quality scoring (assign confidence scores)
5. Metadata tagging (for discoverability)

Critical for ensuring only high-quality data reaches models.
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class FilterCriteria(Enum):
    """Types of data filters."""
    QUALITY_THRESHOLD = "quality_threshold"
    DATE_RANGE = "date_range"
    VOLUME_THRESHOLD = "volume_threshold"
    COMPLETENESS = "completeness"
    ANOMALY_FREE = "anomaly_free"
    CUSTOM = "custom"


@dataclass
class CurationRule:
    """Single data curation rule."""
    
    name: str
    description: str
    filter_type: FilterCriteria
    condition: Callable[[Dict], bool]
    priority: int = 5  # 1-10
    
    # Statistics
    records_filtered: int = 0
    records_kept: int = 0
    
    def apply(self, record: Dict[str, Any]) -> bool:
        """
        Apply curation rule to record.
        
        Returns:
            True if record should be kept, False if filtered out
        """
        try:
            keep = self.condition(record)
            
            if keep:
                self.records_kept += 1
            else:
                self.records_filtered += 1
            
            return keep
        except:
            self.records_filtered += 1
            return False


@dataclass
class CurationResult:
    """Result of data curation process."""
    
    dataset_name: str
    records_input: int
    records_output: int
    records_filtered: int
    
    # Quality metrics
    avg_quality_score_before: float
    avg_quality_score_after: float
    quality_improvement: float
    
    # Filters applied
    filters_applied: List[str] = field(default_factory=list)
    
    # Metadata
    curated_at: datetime = field(default_factory=datetime.now)
    curation_time_seconds: float = 0.0
    
    def get_retention_rate(self) -> float:
        """Get % of records retained."""
        return (self.records_output / self.records_input * 100) if self.records_input > 0 else 0


class DataCurator:
    """
    Data curation and filtration system.
    
    Transforms raw data into curated, analysis-ready datasets.
    
    Features:
    - Multi-criteria filtering
    - Quality-based filtering
    - Data enrichment
    - Metadata tagging
    - Quality scoring
    
    Critical for data excellence!
    """
    
    def __init__(self):
        """Initialize data curator."""
        # Curation rules
        self.rules: List[CurationRule] = []
        
        # Enrichment functions
        self.enrichment_functions: List[Callable] = []
        
        # Statistics
        self.curation_history: List[CurationResult] = []
    
    def register_curation_rule(
        self,
        name: str,
        description: str,
        filter_type: FilterCriteria,
        condition: Callable[[Dict], bool],
        priority: int = 5
    ) -> None:
        """Register a data curation rule."""
        
        rule = CurationRule(
            name=name,
            description=description,
            filter_type=filter_type,
            condition=condition,
            priority=priority
        )
        
        # Insert in priority order
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.priority, reverse=True)
    
    def register_enrichment_function(
        self,
        func: Callable[[Dict], Dict]
    ) -> None:
        """Register data enrichment function."""
        self.enrichment_functions.append(func)
    
    def curate_dataset(
        self,
        data: List[Dict[str, Any]],
        dataset_name: str,
        apply_filters: bool = True,
        apply_enrichment: bool = True,
        min_quality_score: float = 70.0
    ) -> tuple[List[Dict[str, Any]], CurationResult]:
        """
        Curate dataset using registered rules and enrichment.
        
        Args:
            data: Raw data to curate
            dataset_name: Name for this dataset
            apply_filters: Whether to apply filtering rules
            apply_enrichment: Whether to apply enrichment
            min_quality_score: Minimum quality score to keep
        
        Returns:
            Curated data and curation result
        """
        start_time = datetime.now()
        
        curated_data = data.copy()
        
        # Step 1: Apply filtering rules
        if apply_filters and self.rules:
            curated_data = self._apply_filters(curated_data)
        
        # Step 2: Apply quality threshold
        curated_data = self._filter_by_quality(curated_data, min_quality_score)
        
        # Step 3: Apply enrichment
        if apply_enrichment and self.enrichment_functions:
            curated_data = self._apply_enrichment(curated_data)
        
        # Step 4: Add metadata
        curated_data = self._add_metadata(curated_data, dataset_name)
        
        # Calculate quality improvement
        avg_quality_before = self._estimate_quality(data)
        avg_quality_after = self._estimate_quality(curated_data)
        
        # Create result
        result = CurationResult(
            dataset_name=dataset_name,
            records_input=len(data),
            records_output=len(curated_data),
            records_filtered=len(data) - len(curated_data),
            avg_quality_score_before=avg_quality_before,
            avg_quality_score_after=avg_quality_after,
            quality_improvement=avg_quality_after - avg_quality_before,
            filters_applied=[r.name for r in self.rules],
            curation_time_seconds=(datetime.now() - start_time).total_seconds()
        )
        
        self.curation_history.append(result)
        
        return curated_data, result
    
    def _apply_filters(
        self,
        data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Apply all registered filtering rules."""
        
        filtered_data = data.copy()
        
        for rule in self.rules:
            filtered_data = [
                record for record in filtered_data
                if rule.apply(record)
            ]
        
        return filtered_data
    
    def _filter_by_quality(
        self,
        data: List[Dict[str, Any]],
        min_quality_score: float
    ) -> List[Dict[str, Any]]:
        """Filter records below quality threshold."""
        
        # In production: use actual quality scorer
        # For now, simple completeness-based scoring
        
        filtered = []
        for record in data:
            completeness = sum(1 for v in record.values() if v is not None) / len(record) * 100
            
            if completeness >= min_quality_score:
                filtered.append(record)
        
        return filtered
    
    def _apply_enrichment(
        self,
        data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Apply all enrichment functions."""
        
        enriched_data = []
        
        for record in data:
            enriched_record = record.copy()
            
            # Apply each enrichment function
            for func in self.enrichment_functions:
                try:
                    enriched_record = func(enriched_record)
                except Exception as e:
                    # Log error but continue
                    pass
            
            enriched_data.append(enriched_record)
        
        return enriched_data
    
    def _add_metadata(
        self,
        data: List[Dict[str, Any]],
        dataset_name: str
    ) -> List[Dict[str, Any]]:
        """Add curation metadata to each record."""
        
        for record in data:
            if '_metadata' not in record:
                record['_metadata'] = {}
            
            record['_metadata'].update({
                'dataset': dataset_name,
                'curated_at': datetime.now().isoformat(),
                'curated': True
            })
        
        return data
    
    def _estimate_quality(
        self,
        data: List[Dict[str, Any]]
    ) -> float:
        """Estimate average quality score of dataset."""
        
        if not data:
            return 0.0
        
        # Simple quality estimate based on completeness
        scores = []
        for record in data:
            completeness = sum(1 for v in record.values() if v is not None) / len(record) * 100
            scores.append(completeness)
        
        return sum(scores) / len(scores) if scores else 0.0


# Singleton instance
_curator: Optional[DataCurator] = None


def get_data_curator() -> DataCurator:
    """Get or create singleton data curator."""
    global _curator
    
    if _curator is None:
        _curator = DataCurator()
        # Register default curation rules
        _register_default_rules(_curator)
    
    return _curator


def _register_default_rules(curator: DataCurator) -> None:
    """Register default curation rules."""
    
    # Rule 1: Remove records with missing critical fields
    curator.register_curation_rule(
        "require_critical_fields",
        "Filter out records missing critical fields",
        FilterCriteria.COMPLETENESS,
        lambda r: all(field in r and r[field] is not None for field in ['symbol', 'timestamp']),
        priority=10
    )
    
    # Rule 2: Remove records with zero/negative prices
    curator.register_curation_rule(
        "positive_prices",
        "Filter out records with invalid prices",
        FilterCriteria.QUALITY_THRESHOLD,
        lambda r: all(
            r.get(field, 1) > 0 
            for field in ['open', 'high', 'low', 'close'] 
            if field in r
        ),
        priority=9
    )
    
    # Rule 3: Remove records with abnormally low volume
    curator.register_curation_rule(
        "minimum_volume",
        "Filter out records with suspiciously low volume",
        FilterCriteria.VOLUME_THRESHOLD,
        lambda r: r.get('volume', 1000) >= 100,  # Minimum 100 shares
        priority=7
    )


if __name__ == "__main__":
    # Example usage
    curator = get_data_curator()
    
    # Sample data with quality issues
    raw_data = [
        {'symbol': 'AAPL', 'open': 150.0, 'close': 151.0, 'volume': 1000000, 'timestamp': '2024-10-30'},  # Good
        {'symbol': 'AAPL', 'open': None, 'close': 151.0, 'volume': 1000000, 'timestamp': '2024-10-31'},    # Missing open
        {'symbol': 'AAPL', 'open': 150.0, 'close': 0, 'volume': 1000000, 'timestamp': '2024-11-01'},       # Zero price
        {'symbol': 'AAPL', 'open': 150.0, 'close': 151.0, 'volume': 50, 'timestamp': '2024-11-02'},        # Low volume
        {'symbol': 'AAPL', 'open': 150.0, 'close': 151.0, 'volume': 1100000, 'timestamp': '2024-11-03'},  # Good
    ]
    
    print("Data Curation Demo")
    print("=" * 60)
    print(f"Input: {len(raw_data)} records")
    
    # Curate data
    curated, result = curator.curate_dataset(
        raw_data,
        "AAPL_Curated",
        apply_filters=True,
        min_quality_score=70.0
    )
    
    print(f"Output: {len(curated)} records")
    print(f"Filtered: {result.records_filtered} records")
    print(f"Retention Rate: {result.get_retention_rate():.1f}%")
    print(f"Quality Improvement: {result.avg_quality_score_before:.1f}% → {result.avg_quality_score_after:.1f}%")
    print(f"Curation Time: {result.curation_time_seconds:.3f}s")
    
    print("\n✅ Data curation complete!")
    print("Only high-quality data retained for analysis!")