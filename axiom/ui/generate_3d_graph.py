"""
Generate 3D Interactive Graph Visualization from Neo4j
Uses PyVis - lightweight library (5MB)
Output: Interactive HTML file you can open in any browser
"""

from pyvis.network import Network
from neo4j import GraphDatabase
import os

# Neo4j connection
neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
neo4j_user = os.getenv('NEO4J_USER', 'neo4j')
neo4j_password = os.getenv('NEO4J_PASSWORD', 'axiom_neo4j')

print("Connecting to Neo4j...")
driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

# Create network
net = Network(
    height='900px',
    width='100%',
    bgcolor='#1a1a1a',
    font_color='white',
    notebook=False
)

# Physics for 3D-like effect
net.barnes_hut(
    gravity=-80000,
    central_gravity=0.3,
    spring_length=200,
    spring_strength=0.001,
    damping=0.09
)

print("Fetching graph from Neo4j...")
with driver.session() as session:
    # Fetch all nodes (with or without relationships)
    result = session.run("""
        MATCH (n)
        OPTIONAL MATCH (n)-[r]-(m)
        RETURN n, r, m
        LIMIT 500
    """)
    
    nodes_added = set()
    edges_count = 0
    
    for record in result:
        # Source node
        n = record['n']
        n_id = n.element_id
        n_props = dict(n)
        n_label = list(n.labels)[0] if n.labels else 'Node'
        n_name = n_props.get('name') or n_props.get('symbol') or n_id
        
        if n_id not in nodes_added:
            # Color by type
            color = {
                'Company': '#4CAF50',
                'Sector': '#2196F3',
                'Stock': '#FF9800',
                'MarketEvent': '#F44336',
                'Person': '#9C27B0',
                'Product': '#00BCD4'
            }.get(n_label, '#999999')
            
            # Size by importance (market cap for companies)
            size = 25
            if n_label == 'Company' and 'market_cap' in n_props:
                # Scale size by market cap
                size = min(50, max(15, n_props['market_cap'] / 1e10))
            
            net.add_node(
                n_id,
                label=n_name,
                title=f"{n_label}: {n_name}",
                color=color,
                size=size
            )
            nodes_added.add(n_id)
        
        # Target node and relationship (may be None if node has no relationships)
        m = record.get('m')
        r = record.get('r')
        
        if m is not None and r is not None:
            m_id = m.element_id
            m_props = dict(m)
            m_label = list(m.labels)[0] if m.labels else 'Node'
            m_name = m_props.get('name') or m_props.get('symbol') or m_id
            
            if m_id not in nodes_added:
                color = {
                    'Company': '#4CAF50',
                    'Sector': '#2196F3',
                    'Stock': '#FF9800',
                    'MarketEvent': '#F44336'
                }.get(m_label, '#999999')
                
                size = 25
                if m_label == 'Company' and 'market_cap' in m_props:
                    size = min(50, max(15, m_props['market_cap'] / 1e10))
                
                net.add_node(
                    m_id,
                    label=m_name,
                    title=f"{m_label}: {m_name}",
                    color=color,
                    size=size
                )
                nodes_added.add(m_id)
            
            # Relationship
            r_props = dict(r)
            
            # Edge label
            edge_title = f"{r.type}"
            if 'intensity' in r_props:
                edge_title += f" (intensity: {r_props['intensity']:.2f})"
            elif 'coefficient' in r_props:
                edge_title += f" (coef: {r_props['coefficient']:.2f})"
            
            net.add_edge(n_id, m_id, title=edge_title)
            edges_count += 1

print(f"✅ Graph loaded: {len(nodes_added)} nodes, {edges_count} edges")

# Generate HTML (use save_graph instead of show to avoid notebook issues)
output_file = 'axiom/ui/knowledge_graph_3d.html'
net.save_graph(output_file)

print(f"\n🎉 3D Graph visualization created!")
print(f"📁 File: {output_file}")
print(f"🌐 Open in browser: file://{os.path.abspath(output_file)}")
print(f"\n💡 The graph is interactive:")
print(f"   - Drag nodes to reposition")
print(f"   - Scroll to zoom")
print(f"   - Click nodes for details")
print(f"   - Physics simulation animates the network")