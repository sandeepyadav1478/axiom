"""
Systemic Risk Analyzer for Financial Networks

Analyzes interconnected risks across financial system.
Uses network analysis to identify:
- Systemically important institutions
- Contagion pathways
- Network vulnerabilities
- Cascade effects
"""

import numpy as np
import networkx as nx

class SystemicRiskAnalyzer:
    """Analyze systemic risk in financial networks"""
    
    def __init__(self):
        self.network = nx.DiGraph()
    
    def build_network(self, exposures: dict):
        """Build financial network from exposures"""
        for (node1, node2), amount in exposures.items():
            self.network.add_edge(node1, node2, weight=amount)
    
    def identify_systemically_important(self, top_n: int = 10) -> list:
        """Identify SIFI (Systemically Important Financial Institutions)"""
        centrality = nx.eigenvector_centrality(self.network, max_iter=1000)
        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        return sorted_nodes[:top_n]
    
    def simulate_default_cascade(self, initial_default: str, threshold: float = 0.3) -> list:
        """Simulate cascade if institution defaults"""
        defaulted = {initial_default}
        new_defaults = True
        
        while new_defaults:
            new_defaults = False
            for node in list(self.network.nodes()):
                if node in defaulted:
                    continue
                
                # Check if exposures exceed threshold
                total_exposure = sum(
                    self.network[source][node]['weight']
                    for source in self.network.predecessors(node)
                    if source in defaulted
                )
                
                node_capital = self.network.nodes[node].get('capital', 1.0)
                
                if total_exposure / node_capital > threshold:
                    defaulted.add(node)
                    new_defaults = True
        
        return list(defaulted)

if __name__ == "__main__":
    analyzer = SystemicRiskAnalyzer()
    exposures = {
        ('Bank_A', 'Bank_B'): 100,
        ('Bank_B', 'Bank_C'): 50
    }
    analyzer.build_network(exposures)
    sifi = analyzer.identify_systemically_important()
    print(f"SIFI: {sifi}")