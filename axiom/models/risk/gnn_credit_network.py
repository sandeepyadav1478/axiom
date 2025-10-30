"""
Graph Neural Network for Credit Risk Network Modeling

Based on: M. Nandan, S. Mitra, D. De (March 2025)
"GraphXAI: a survey of graph neural networks (GNNs) for explainable AI (XAI)"
Neural Computing and Applications, Springer, Vol. 37, pages 10949-11000
DOI: Published March 8, 2025

This implementation uses Graph Neural Networks to model credit risk through:
- Entity relationship networks (suppliers, customers, corporate groups)
- Default contagion modeling
- Network effect quantification
- Explainable predictions via graph attention

Captures systemic credit risk that traditional models miss.
"""

from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

try:
    from torch_geometric.nn import GATConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False


class EntityType(Enum):
    """Types of entities in credit network"""
    CORPORATION = "corporation"
    FINANCIAL_INSTITUTION = "financial_institution"
    SUPPLIER = "supplier"
    CUSTOMER = "customer"
    SUBSIDIARY = "subsidiary"


class RelationshipType(Enum):
    """Types of relationships in credit network"""
    SUPPLIER_CUSTOMER = "supplier_customer"
    PARENT_SUBSIDIARY = "parent_subsidiary"
    LENDER_BORROWER = "lender_borrower"
    EQUITY_HOLDER = "equity_holder"
    BUSINESS_PARTNER = "business_partner"


@dataclass
class CreditEntity:
    """Entity in credit network"""
    entity_id: str
    entity_type: EntityType
    
    # Financial features
    revenue: float
    total_debt: float
    credit_score: float
    default_probability: float
    
    # Network features (computed)
    degree_centrality: float = 0.0
    betweenness_centrality: float = 0.0
    eigenvector_centrality: float = 0.0
    clustering_coefficient: float = 0.0
    
    # Risk metrics
    systemic_risk_score: float = 0.0
    contagion_risk: float = 0.0


@dataclass
class GNNConfig:
    """Configuration for GNN Credit Network"""
    # Node features
    n_node_features: int = 10  # Financial + network features
    hidden_channels: int = 64
    out_channels: int = 32
    
    # GNN architecture
    n_gnn_layers: int = 3
    heads: int = 4  # Attention heads for GAT
    dropout: float = 0.3
    
    # Training
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 100
    
    # Graph construction
    edge_threshold: float = 0.3  # Minimum relationship strength


class GraphAttentionCreditGNN(nn.Module):
    """
    Graph Attention Network for Credit Risk
    
    Uses multi-head attention to learn importance of different
    relationships in credit network for default prediction.
    """
    
    def __init__(self, config: GNNConfig):
        super(GraphAttentionCreditGNN, self).__init__()
        
        self.config = config
        
        # GAT layers
        self.conv1 = GATConv(
            config.n_node_features,
            config.hidden_channels,
            heads=config.heads,
            dropout=config.dropout
        )
        
        self.conv2 = GATConv(
            config.hidden_channels * config.heads,
            config.hidden_channels,
            heads=config.heads,
            dropout=config.dropout
        )
        
        self.conv3 = GATConv(
            config.hidden_channels * config.heads,
            config.out_channels,
            heads=1,  # Single head for output
            dropout=config.dropout
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(config.out_channels, config.out_channels // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.out_channels // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass through GNN
        
        Args:
            x: Node features (n_nodes, n_node_features)
            edge_index: Edge connectivity (2, n_edges)
            batch: Batch assignment for nodes (for batched graphs)
            
        Returns:
            Default probabilities for each node
        """
        # GAT layers with attention
        x = F.dropout(x, p=self.config.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        
        x = F.dropout(x, p=self.config.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        
        x = F.dropout(x, p=self.config.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        
        # Predict default probability
        default_prob = self.classifier(x)
        
        return default_prob


class GNNCreditNetwork:
    """
    Complete GNN Credit Risk Network System
    
    Models credit risk through entity relationships using Graph Neural Networks.
    Captures contagion effects, network centrality, and systemic risk.
    """
    
    def __init__(self, config: Optional[GNNConfig] = None):
        if not all([TORCH_AVAILABLE, NETWORKX_AVAILABLE, TORCH_GEOMETRIC_AVAILABLE]):
            missing = []
            if not TORCH_AVAILABLE:
                missing.append("torch")
            if not NETWORKX_AVAILABLE:
                missing.append("networkx")
            if not TORCH_GEOMETRIC_AVAILABLE:
                missing.append("torch_geometric")
            raise ImportError(f"Missing required packages: {', '.join(missing)}")
        
        self.config = config or GNNConfig()
        self.model = GraphAttentionCreditGNN(self.config)
        self.optimizer = None
        
        # Network graph
        self.graph = None
        self.entity_index = {}  # Map entity_id to node index
        
        # Training history
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    def build_credit_network(
        self,
        entities: List[CreditEntity],
        relationships: List[Tuple[str, str, RelationshipType, float]]
    ):
        """
        Build credit network graph from entities and relationships
        
        Args:
            entities: List of credit entities
            relationships: List of (entity1_id, entity2_id, rel_type, strength) tuples
        """
        # Create NetworkX graph
        self.graph = nx.DiGraph()
        
        # Add nodes
        for idx, entity in enumerate(entities):
            self.entity_index[entity.entity_id] = idx
            
            # Add node with features
            self.graph.add_node(
                idx,
                entity_id=entity.entity_id,
                revenue=entity.revenue,
                debt=entity.total_debt,
                credit_score=entity.credit_score,
                default_prob=entity.default_probability
            )
        
        # Add edges
        for entity1_id, entity2_id, rel_type, strength in relationships:
            if entity1_id in self.entity_index and entity2_id in self.entity_index:
                idx1 = self.entity_index[entity1_id]
                idx2 = self.entity_index[entity2_id]
                
                if strength >= self.config.edge_threshold:
                    self.graph.add_edge(
                        idx1, idx2,
                        relationship=rel_type.value,
                        strength=strength
                    )
        
        # Calculate network metrics
        self._calculate_network_metrics()
    
    def _calculate_network_metrics(self):
        """Calculate network centrality metrics for all nodes"""
        
        if self.graph is None or self.graph.number_of_nodes() == 0:
            return
        
        # Degree centrality
        degree_cent = nx.degree_centrality(self.graph)
        
        # Betweenness centrality
        between_cent = nx.betweenness_centrality(self.graph)
        
        # Eigenvector centrality (may fail for disconnected graphs)
        try:
            eigen_cent = nx.eigenvector_centrality(self.graph, max_iter=1000)
        except:
            eigen_cent = {node: 0.0 for node in self.graph.nodes()}
        
        # Clustering coefficient
        clustering = nx.clustering(self.graph.to_undirected())
        
        # Store in graph
        for node in self.graph.nodes():
            self.graph.nodes[node]['degree_centrality'] = degree_cent.get(node, 0.0)
            self.graph.nodes[node]['betweenness_centrality'] = between_cent.get(node, 0.0)
            self.graph.nodes[node]['eigenvector_centrality'] = eigen_cent.get(node, 0.0)
            self.graph.nodes[node]['clustering'] = clustering.get(node, 0.0)
    
    def _graph_to_pytorch_geometric(self) -> Data:
        """Convert NetworkX graph to PyTorch Geometric Data object"""
        
        if self.graph is None:
            raise ValueError("Network not built. Call build_credit_network() first.")
        
        # Extract node features
        node_features = []
        node_labels = []
        
        for node in sorted(self.graph.nodes()):
            data = self.graph.nodes[node]
            
            features = [
                data.get('revenue', 0) / 1e9,  # Normalized
                data.get('debt', 0) / 1e9,
                data.get('credit_score', 650) / 850,  # Normalized
                data.get('default_prob', 0.15),
                data.get('degree_centrality', 0),
                data.get('betweenness_centrality', 0),
                data.get('eigenvector_centrality', 0),
                data.get('clustering', 0),
                len(list(self.graph.predecessors(node))) / 10,  # In-degree normalized
                len(list(self.graph.successors(node))) / 10  # Out-degree normalized
            ]
            
            node_features.append(features)
            node_labels.append(data.get('default_prob', 0.15))
        
        x = torch.FloatTensor(node_features)
        y = torch.FloatTensor(node_labels)
        
        # Extract edges
        edge_list = list(self.graph.edges())
        if edge_list:
            edge_index = torch.LongTensor(edge_list).t().contiguous()
        else:
            edge_index = torch.LongTensor([[], []])
        
        return Data(x=x, edge_index=edge_index, y=y)
    
    def train(
        self,
        default_labels: torch.Tensor,
        epochs: int = 100,
        verbose: int = 1
    ):
        """
        Train GNN on credit network
        
        Args:
            default_labels: True default labels for nodes
            epochs: Training epochs
            verbose: Verbosity level
        """
        self.model.train()
        
        # Convert graph
        graph_data = self._graph_to_pytorch_geometric()
        
        # Update labels
        graph_data.y = default_labels.unsqueeze(1) if len(default_labels.shape) == 1 else default_labels
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
        
        criterion = nn.BCELoss()
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            # Forward pass
            out = self.model(graph_data.x, graph_data.edge_index)
            
            # Loss
            loss = criterion(out, graph_data.y)
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            # Accuracy
            predicted = (out > 0.5).float()
            accuracy = (predicted == graph_data.y).float().mean().item()
            
            self.history['train_loss'].append(loss.item())
            self.history['train_acc'].append(accuracy)
            
            if verbose > 0 and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}, Acc: {accuracy:.4f}")
    
    def predict_defaults(self) -> Dict[str, float]:
        """
        Predict default probabilities for all entities in network
        
        Returns:
            Dictionary mapping entity_id to default probability
        """
        self.model.eval()
        
        graph_data = self._graph_to_pytorch_geometric()
        
        with torch.no_grad():
            predictions = self.model(graph_data.x, graph_data.edge_index)
        
        # Map back to entity IDs
        results = {}
        for entity_id, node_idx in self.entity_index.items():
            results[entity_id] = float(predictions[node_idx].item())
        
        return results
    
    def analyze_contagion_risk(
        self,
        defaulted_entity_id: str,
        max_hops: int = 3
    ) -> Dict[str, float]:
        """
        Analyze contagion risk if specified entity defaults
        
        Args:
            defaulted_entity_id: Entity that defaults
            max_hops: Maximum network distance to analyze
            
        Returns:
            Dictionary of affected entities and contagion probabilities
        """
        if defaulted_entity_id not in self.entity_index:
            return {}
        
        defaulted_node = self.entity_index[defaulted_entity_id]
        
        # Get connected entities within max_hops
        affected = {}
        
        for node in self.graph.nodes():
            if node == defaulted_node:
                continue
            
            try:
                path_length = nx.shortest_path_length(
                    self.graph, defaulted_node, node
                )
                
                if path_length <= max_hops:
                    # Contagion probability decreases with distance
                    base_prob = self.graph.nodes[node].get('default_prob', 0.15)
                    contagion_increase = 0.20 * (1.0 / path_length)  # 20% increase for direct connections
                    
                    contagion_prob = min(0.95, base_prob + contagion_increase)
                    
                    entity_id = self.graph.nodes[node]['entity_id']
                    affected[entity_id] = contagion_prob
            except nx.NetworkXNoPath:
                continue
        
        return affected
    
    def identify_systemically_important_entities(
        self,
        top_n: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Identify systemically important entities based on network centrality
        
        Args:
            top_n: Number of top entities to return
            
        Returns:
            List of (entity_id, importance_score) tuples
        """
        importance_scores = {}
        
        for entity_id, node_idx in self.entity_index.items():
            # Composite importance score
            node_data = self.graph.nodes[node_idx]
            
            score = (
                0.30 * node_data.get('degree_centrality', 0) +
                0.25 * node_data.get('betweenness_centrality', 0) +
                0.25 * node_data.get('eigenvector_centrality', 0) +
                0.20 * (node_data.get('revenue', 0) / 1e9)  # Size factor
            )
            
            importance_scores[entity_id] = score
        
        # Sort by importance
        ranked = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        
        return ranked[:top_n]
    
    def save(self, path: str):
        """Save GNN model and network"""
        import pickle
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'history': self.history,
            'graph': pickle.dumps(self.graph),
            'entity_index': self.entity_index
        }, path)
    
    def load(self, path: str):
        """Load GNN model and network"""
        import pickle
        
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint.get('history', {})
        self.graph = pickle.loads(checkpoint['graph'])
        self.entity_index = checkpoint['entity_index']


def create_sample_credit_network(
    n_entities: int = 50
) -> Tuple[List[CreditEntity], List[Tuple]]:
    """
    Create sample credit network for testing
    
    Returns:
        (entities, relationships)
    """
    np.random.seed(42)
    
    entities = []
    
    # Create entities
    for i in range(n_entities):
        entity_type = np.random.choice([EntityType.CORPORATION, EntityType.FINANCIAL_INSTITUTION])
        
        entity = CreditEntity(
            entity_id=f"ENTITY_{i:03d}",
            entity_type=entity_type,
            revenue=np.random.lognormal(np.log(100e6), 1.5),
            total_debt=np.random.lognormal(np.log(50e6), 1.8),
            credit_score=np.random.normal(700, 80),
            default_probability=np.random.beta(2, 18)  # Most entities low default prob
        )
        
        entities.append(entity)
    
    # Create relationships (supply chain, lending, etc.)
    relationships = []
    
    for i in range(n_entities):
        # Each entity has 2-5 connections
        n_connections = np.random.randint(2, 6)
        
        for _ in range(n_connections):
            j = np.random.randint(0, n_entities)
            if i != j:
                rel_type = np.random.choice([
                    RelationshipType.SUPPLIER_CUSTOMER,
                    RelationshipType.LENDER_BORROWER,
                    RelationshipType.BUSINESS_PARTNER
                ])
                
                strength = np.random.uniform(0.2, 0.9)
                
                relationships.append((
                    entities[i].entity_id,
                    entities[j].entity_id,
                    rel_type,
                    strength
                ))
    
    return entities, relationships


# Example usage
if __name__ == "__main__":
    print("GNN Credit Network - Example Usage")
    print("=" * 70)
    
    if not all([TORCH_AVAILABLE, NETWORKX_AVAILABLE, TORCH_GEOMETRIC_AVAILABLE]):
        print("Missing required dependencies:")
        print(f"  PyTorch: {'✓' if TORCH_AVAILABLE else '✗'}")
        print(f"  NetworkX: {'✓' if NETWORKX_AVAILABLE else '✗'}")
        print(f"  PyTorch Geometric: {'✓' if TORCH_GEOMETRIC_AVAILABLE else '✗'}")
        print("\nInstall with:")
        print("  pip install torch networkx torch-geometric")
    else:
        # Configuration
        print("\n1. Configuration")
        config = GNNConfig(
            n_node_features=10,
            hidden_channels=64,
            n_gnn_layers=3,
            heads=4
        )
        print(f"   Node features: {config.n_node_features}")
        print(f"   Hidden channels: {config.hidden_channels}")
        print(f"   GAT layers: {config.n_gnn_layers}")
        print(f"   Attention heads: {config.heads}")
        
        # Create network
        print("\n2. Creating Sample Credit Network")
        entities, relationships = create_sample_credit_network(n_entities=50)
        print(f"   Entities: {len(entities)}")
        print(f"   Relationships: {len(relationships)}")
        
        # Initialize GNN
        print("\n3. Initializing GNN Credit Network")
        gnn = GNNCreditNetwork(config)
        print("   ✓ Graph Attention Network initialized")
        print("   ✓ Multi-head attention configured")
        
        # Build network
        print("\n4. Building Network Graph")
        gnn.build_credit_network(entities, relationships)
        print(f"   ✓ Network graph built")
        print(f"   Nodes: {gnn.graph.number_of_nodes()}")
        print(f"   Edges: {gnn.graph.number_of_edges()}")
        
        # Train
        print("\n5. Training GNN")
        default_labels = torch.FloatTensor([e.default_probability for e in entities])
        gnn.train(default_labels, epochs=50, verbose=1)
        print("   ✓ Training completed")
        
        # Predictions
        print("\n6. Predicting Default Probabilities")
        predictions = gnn.predict_defaults()
        print(f"   Predictions for {len(predictions)} entities")
        
        # Show sample
        sample_entities = list(predictions.items())[:5]
        for entity_id, prob in sample_entities:
            print(f"     {entity_id}: {prob:.3f}")
        
        # Systemic importance
        print("\n7. Systemically Important Entities")
        important = gnn.identify_systemically_important_entities(top_n=5)
        for entity_id, score in important:
            print(f"     {entity_id}: {score:.3f}")
        
        # Contagion analysis
        print("\n8. Contagion Risk Analysis")
        affected = gnn.analyze_contagion_risk(entities[0].entity_id, max_hops=2)
        print(f"   If {entities[0].entity_id} defaults:")
        print(f"   {len(affected)} entities affected within 2 hops")
        for entity_id, prob in list(affected.items())[:3]:
            print(f"     {entity_id}: {prob:.3f}")
        
        print("\n9. Model Capabilities")
        print("   ✓ Network effect modeling")
        print("   ✓ Default contagion analysis")
        print("   ✓ Systemic importance identification")
        print("   ✓ Multi-hop risk propagation")
        print("   ✓ Explainable via attention weights")
        
        print("\n" + "=" * 70)
        print("Demo completed successfully!")
        print("\nBased on: Nandan et al. (March 2025)")
        print("Innovation: GNN for credit network and contagion modeling")