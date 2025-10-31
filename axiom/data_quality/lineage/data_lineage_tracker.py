"""
Data Lineage Tracking System - Audit Trail & Compliance

Tracks complete data journey from source to model/output.
Critical for:
- Regulatory compliance (audit trails)
- Debugging data issues
- Impact analysis
- Reproducibility

This provides complete transparency for data legitimacy verification.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hashlib


class LineageNodeType(Enum):
    """Types of nodes in data lineage graph."""
    SOURCE = "source"              # Data source (API, database, file)
    TRANSFORMATION = "transformation"  # Data transformation
    VALIDATION = "validation"      # Quality check
    FEATURE = "feature"            # Feature computation
    MODEL = "model"                # ML model
    OUTPUT = "output"              # Final output


@dataclass
class LineageNode:
    """
    Single node in data lineage graph.
    
    Represents a step in data journey:
    - Where data came from
    - What was done to it
    - Where it went
    """
    
    node_id: str
    node_type: LineageNodeType
    name: str
    description: str
    
    # Inputs to this node
    inputs: List[str] = field(default_factory=list)  # List of node_ids
    
    # Outputs from this node
    outputs: List[str] = field(default_factory=list)  # List of node_ids
    
    # Transformation details
    transformation_code: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Quality metrics
    quality_score: Optional[float] = None
    validation_passed: bool = True
    
    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    execution_time_ms: Optional[float] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            "node_id": self.node_id,
            "type": self.node_type.value,
            "name": self.name,
            "description": self.description,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "parameters": self.parameters,
            "quality_score": self.quality_score,
            "validation_passed": self.validation_passed,
            "created_at": self.created_at.isoformat(),
            "execution_time_ms": self.execution_time_ms,
            "version": self.version
        }


@dataclass
class DataLineage:
    """
    Complete data lineage for a dataset or feature.
    
    Tracks full data journey from source to consumption.
    """
    
    lineage_id: str
    entity_name: str  # Dataset/feature name
    
    # Lineage graph
    nodes: Dict[str, LineageNode] = field(default_factory=dict)
    
    # Graph structure
    root_nodes: List[str] = field(default_factory=list)  # Source nodes
    leaf_nodes: List[str] = field(default_factory=list)  # Output nodes
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def add_node(self, node: LineageNode) -> None:
        """Add node to lineage graph."""
        self.nodes[node.node_id] = node
        
        # Update root/leaf tracking
        if not node.inputs:
            self.root_nodes.append(node.node_id)
        if not node.outputs:
            self.leaf_nodes.append(node.node_id)
        
        self.last_updated = datetime.now()
    
    def add_edge(self, from_node_id: str, to_node_id: str) -> None:
        """Add edge between nodes (data flow)."""
        if from_node_id in self.nodes:
            self.nodes[from_node_id].outputs.append(to_node_id)
        
        if to_node_id in self.nodes:
            self.nodes[to_node_id].inputs.append(from_node_id)
        
        self.last_updated = datetime.now()
    
    def get_lineage_path(self, node_id: str) -> List[LineageNode]:
        """Get complete lineage path from source to node."""
        path = []
        
        def trace_back(current_id: str):
            if current_id not in self.nodes:
                return
            
            node = self.nodes[current_id]
            path.append(node)
            
            # Trace back through inputs (DFS)
            for input_id in node.inputs:
                trace_back(input_id)
        
        trace_back(node_id)
        return path[::-1]  # Reverse to get source-to-destination order
    
    def get_downstream_impact(self, node_id: str) -> List[LineageNode]:
        """Get all downstream nodes affected by this node."""
        affected = []
        
        def trace_forward(current_id: str):
            if current_id not in self.nodes:
                return
            
            node = self.nodes[current_id]
            affected.append(node)
            
            for output_id in node.outputs:
                trace_forward(output_id)
        
        trace_forward(node_id)
        return affected
    
    def to_dict(self) -> Dict:
        """Convert lineage to dictionary."""
        return {
            "lineage_id": self.lineage_id,
            "entity_name": self.entity_name,
            "nodes": {
                node_id: node.to_dict() 
                for node_id, node in self.nodes.items()
            },
            "root_nodes": self.root_nodes,
            "leaf_nodes": self.leaf_nodes,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat()
        }


class DataLineageTracker:
    """
    Data lineage tracking system.
    
    Maintains complete audit trail of data transformations.
    
    Features:
    - Track data sources
    - Record all transformations
    - Capture validation steps
    - Monitor feature derivations
    - Trace model inputs
    
    Benefits:
    - Regulatory compliance (audit trail)
    - Debugging (trace data issues)
    - Impact analysis (what breaks if X changes)
    - Reproducibility (recreate results)
    """
    
    def __init__(self):
        """Initialize lineage tracker."""
        # Lineage graphs by entity
        self.lineages: Dict[str, DataLineage] = {}
        
        # Global node registry
        self.all_nodes: Dict[str, LineageNode] = {}
    
    def create_lineage(self, entity_name: str) -> DataLineage:
        """Create new lineage graph for entity."""
        lineage_id = self._generate_lineage_id(entity_name)
        
        lineage = DataLineage(
            lineage_id=lineage_id,
            entity_name=entity_name
        )
        
        self.lineages[entity_name] = lineage
        return lineage
    
    def record_source(
        self,
        entity_name: str,
        source_name: str,
        source_type: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Record data source.
        
        Args:
            entity_name: Entity this source belongs to
            source_name: Name of source (e.g., 'yahoo_finance', 'bloomberg')
            source_type: Type of source (e.g., 'api', 'database', 'file')
            metadata: Additional metadata
        
        Returns:
            Node ID of created source node
        """
        if entity_name not in self.lineages:
            self.create_lineage(entity_name)
        
        node_id = f"source_{source_name}_{int(time.time())}"
        
        node = LineageNode(
            node_id=node_id,
            node_type=LineageNodeType.SOURCE,
            name=source_name,
            description=f"Data source: {source_type}",
            metadata=metadata or {"source_type": source_type}
        )
        
        self.lineages[entity_name].add_node(node)
        self.all_nodes[node_id] = node
        
        return node_id
    
    def record_transformation(
        self,
        entity_name: str,
        transformation_name: str,
        input_node_ids: List[str],
        transformation_code: Optional[str] = None,
        parameters: Optional[Dict] = None
    ) -> str:
        """
        Record data transformation.
        
        Args:
            entity_name: Entity being transformed
            transformation_name: Name of transformation
            input_node_ids: IDs of input nodes
            transformation_code: Code/function used
            parameters: Transformation parameters
        
        Returns:
            Node ID of transformation
        """
        if entity_name not in self.lineages:
            self.create_lineage(entity_name)
        
        node_id = f"transform_{transformation_name}_{int(time.time())}"
        
        node = LineageNode(
            node_id=node_id,
            node_type=LineageNodeType.TRANSFORMATION,
            name=transformation_name,
            description=f"Transformation: {transformation_name}",
            inputs=input_node_ids,
            transformation_code=transformation_code,
            parameters=parameters or {}
        )
        
        self.lineages[entity_name].add_node(node)
        
        # Add edges
        for input_id in input_node_ids:
            self.lineages[entity_name].add_edge(input_id, node_id)
        
        self.all_nodes[node_id] = node
        
        return node_id
    
    def record_validation(
        self,
        entity_name: str,
        input_node_id: str,
        validation_results: Any,
        quality_score: float
    ) -> str:
        """Record validation step."""
        node_id = f"validation_{int(time.time())}"
        
        node = LineageNode(
            node_id=node_id,
            node_type=LineageNodeType.VALIDATION,
            name="Data Quality Validation",
            description="Validation and quality checks",
            inputs=[input_node_id],
            quality_score=quality_score,
            validation_passed=quality_score >= 70
        )
        
        if entity_name not in self.lineages:
            self.create_lineage(entity_name)
        
        self.lineages[entity_name].add_node(node)
        self.lineages[entity_name].add_edge(input_node_id, node_id)
        
        self.all_nodes[node_id] = node
        
        return node_id
    
    def get_lineage(self, entity_name: str) -> Optional[DataLineage]:
        """Get lineage graph for entity."""
        return self.lineages.get(entity_name)
    
    def export_lineage(self, entity_name: str, format: str = "dict") -> Any:
        """
        Export lineage for visualization or storage.
        
        Args:
            entity_name: Entity to export
            format: Export format ('dict', 'json', 'graphviz')
        
        Returns:
            Lineage in requested format
        """
        lineage = self.get_lineage(entity_name)
        if not lineage:
            return None
        
        if format == "dict":
            return lineage.to_dict()
        elif format == "json":
            import json
            return json.dumps(lineage.to_dict(), indent=2)
        else:
            return lineage.to_dict()
    
    def _generate_lineage_id(self, entity_name: str) -> str:
        """Generate unique lineage ID."""
        content = f"{entity_name}_{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:16]


# Singleton instance
_lineage_tracker: Optional[DataLineageTracker] = None


def get_lineage_tracker() -> DataLineageTracker:
    """Get or create singleton lineage tracker."""
    global _lineage_tracker
    
    if _lineage_tracker is None:
        _lineage_tracker = DataLineageTracker()
    
    return _lineage_tracker


if __name__ == "__main__":
    # Example usage
    import time
    
    tracker = get_lineage_tracker()
    
    # Track data journey
    print("Data Lineage Tracking Example")
    print("=" * 60)
    
    # 1. Record source
    source_id = tracker.record_source(
        "AAPL_Price",
        "yahoo_finance",
        "api",
        {"endpoint": "quote", "frequency": "daily"}
    )
    print(f"✅ Recorded source: {source_id}")
    
    # 2. Record transformation
    transform_id = tracker.record_transformation(
        "AAPL_Price",
        "clean_and_normalize",
        [source_id],
        transformation_code="normalize_prices()",
        parameters={"method": "z-score"}
    )
    print(f"✅ Recorded transformation: {transform_id}")
    
    # 3. Record validation
    validation_id = tracker.record_validation(
        "AAPL_Price",
        transform_id,
        validation_results=None,
        quality_score=92.5
    )
    print(f"✅ Recorded validation: {validation_id}")
    
    # Get lineage
    lineage = tracker.get_lineage("AAPL_Price")
    print(f"\nLineage for AAPL_Price:")
    print(f"  Nodes: {len(lineage.nodes)}")
    print(f"  Root nodes: {lineage.root_nodes}")
    print(f"  Leaf nodes: {lineage.leaf_nodes}")
    
    # Get lineage path
    path = lineage.get_lineage_path(validation_id)
    print(f"\nLineage path to validation:")
    for node in path:
        print(f"  {node.node_type.value}: {node.name}")
    
    print("\n✅ Data lineage tracking complete!")
    print("Complete audit trail available for compliance!")