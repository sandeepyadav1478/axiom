"""
Populate Neo4j with sample data for testing visualization
Creates companies, sectors, and relationships
"""

from neo4j import GraphDatabase
import os

# Connect to Neo4j
driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "axiom_neo4j")
)

print("Populating Neo4j with sample data...")

with driver.session() as session:
    # Create tech companies
    session.run("""
        // Create Companies
        MERGE (aapl:Company {symbol: 'AAPL', name: 'Apple Inc.', sector: 'Technology', market_cap: 2800000000000})
        MERGE (msft:Company {symbol: 'MSFT', name: 'Microsoft Corporation', sector: 'Technology', market_cap: 2500000000000})
        MERGE (googl:Company {symbol: 'GOOGL', name: 'Alphabet Inc.', sector: 'Technology', market_cap: 1700000000000})
        MERGE (amzn:Company {symbol: 'AMZN', name: 'Amazon.com Inc.', sector: 'Technology', market_cap: 1400000000000})
        MERGE (meta:Company {symbol: 'META', name: 'Meta Platforms Inc.', sector: 'Technology', market_cap: 800000000000})
        MERGE (tsla:Company {symbol: 'TSLA', name: 'Tesla Inc.', sector: 'Automotive', market_cap: 700000000000})
        MERGE (nvda:Company {symbol: 'NVDA', name: 'NVIDIA Corporation', sector: 'Technology', market_cap: 1200000000000})
        
        // Create Sectors
        MERGE (tech:Sector {name: 'Technology'})
        MERGE (auto:Sector {name: 'Automotive'})
        
        // BELONGS_TO relationships
        MERGE (aapl)-[:BELONGS_TO]->(tech)
        MERGE (msft)-[:BELONGS_TO]->(tech)
        MERGE (googl)-[:BELONGS_TO]->(tech)
        MERGE (amzn)-[:BELONGS_TO]->(tech)
        MERGE (meta)-[:BELONGS_TO]->(tech)
        MERGE (nvda)-[:BELONGS_TO]->(tech)
        MERGE (tsla)-[:BELONGS_TO]->(auto)
        
        // COMPETES_WITH relationships
        MERGE (aapl)-[:COMPETES_WITH {intensity: 0.85, market: 'smartphones'}]-(googl)
        MERGE (aapl)-[:COMPETES_WITH {intensity: 0.70, market: 'cloud services'}]-(msft)
        MERGE (msft)-[:COMPETES_WITH {intensity: 0.90, market: 'cloud services'}]-(amzn)
        MERGE (msft)-[:COMPETES_WITH {intensity: 0.75, market: 'productivity software'}]-(googl)
        MERGE (meta)-[:COMPETES_WITH {intensity: 0.60, market: 'digital advertising'}]-(googl)
        MERGE (nvda)-[:COMPETES_WITH {intensity: 0.40, market: 'AI chips'}]-(aapl)
        
        // CORRELATED_WITH relationships
        MERGE (aapl)-[:CORRELATED_WITH {coefficient: 0.82, period_days: 30, explanation: 'Both large-cap tech with similar market exposure'}]-(msft)
        MERGE (aapl)-[:CORRELATED_WITH {coefficient: 0.75, period_days: 30, explanation: 'Technology sector correlation'}]-(googl)
        MERGE (msft)-[:CORRELATED_WITH {coefficient: 0.88, period_days: 30, explanation: 'Cloud computing business overlap'}]-(amzn)
        MERGE (meta)-[:CORRELATED_WITH {coefficient: 0.70, period_days: 30, explanation: 'Digital advertising sector'}]-(googl)
        
        // Create some market events
        CREATE (e1:MarketEvent {type: 'earnings', date: date('2025-11-01'), title: 'AAPL Q4 Earnings Beat', relevance: 0.9})
        CREATE (e2:MarketEvent {type: 'product_launch', date: date('2025-11-10'), title: 'NVDA New AI Chip', relevance: 0.85})
        
        // Link events to companies
        MERGE (aapl)-[:AFFECTED_BY {impact_score: 0.9}]->(e1)
        MERGE (nvda)-[:AFFECTED_BY {impact_score: 0.95}]->(e2)
        MERGE (msft)-[:AFFECTED_BY {impact_score: 0.3}]->(e2)  // Competitor affected
    """)
    
    print("✅ Sample data created!")
    
    # Verify
    result = session.run("MATCH (n) RETURN labels(n)[0] as type, count(n) as count ORDER BY count DESC")
    print("\n📊 Graph Contents:")
    for record in result:
        print(f"  {record['type']}: {record['count']} nodes")
    
    result = session.run("MATCH ()-[r]->() RETURN type(r) as rel, count(r) as count ORDER BY count DESC")
    print("\n🔗 Relationships:")
    for record in result:
        print(f"  {record['rel']}: {record['count']} edges")

driver.close()
print("\n🎉 Neo4j populated with rich sample data!")
print("📝 Run: python axiom/ui/generate_3d_graph.py")
print("📂 Then open: axiom/ui/knowledge_graph_3d.html")