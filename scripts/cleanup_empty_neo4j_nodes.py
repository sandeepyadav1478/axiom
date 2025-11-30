"""
Neo4j Empty Node Cleanup Script
Removes 28K+ empty unlabeled nodes that are corrupting the graph

Issue Discovered: 2025-11-28
- 28,252 nodes with no labels and no properties
- 84% of all nodes are empty placeholders
- Created by DAG bug (likely MERGE without setting labels)

Strategy:
1. Identify all empty nodes (no labels, no properties)
2. Check which are connected to real data
3. Delete isolated orphans safely
4. Investigate remaining connected empties
"""

import os
import sys
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

def analyze_empty_nodes():
    """Analyze the empty node situation."""
    
    uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    user = os.getenv('NEO4J_USER', 'neo4j')
    password = os.getenv('NEO4J_PASSWORD', 'axiom_neo4j')
    
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    with driver.session() as session:
        # Count empty nodes
        result = session.run("""
            MATCH (n)
            WHERE labels(n) = [] AND size(keys(n)) = 0
            RETURN count(n) as total_empty
        """)
        total_empty = result.single()['total_empty']
        print(f"\n📊 Analysis Results:")
        print(f"   Total empty nodes: {total_empty:,}")
        
        # Count isolated empty nodes (no relationships at all)
        result = session.run("""
            MATCH (n)
            WHERE labels(n) = [] AND size(keys(n)) = 0
            AND NOT (n)-[]-()
            RETURN count(n) as isolated_empty
        """)
        isolated = result.single()['isolated_empty']
        print(f"   Isolated (no relationships): {isolated:,}")
        
        # Count connected to real nodes
        result = session.run("""
            MATCH (empty)-[r]-(real)
            WHERE labels(empty) = [] AND size(keys(empty)) = 0
            AND size(labels(real)) > 0
            RETURN type(r) as rel_type, labels(real)[0] as target_type, count(*) as count
            ORDER BY count DESC
        """)
        
        print(f"\n   Connected to real nodes:")
        connected_count = 0
        for record in result:
            print(f"     • {record['rel_type']} → {record['target_type']}: {record['count']:,}")
            connected_count += record['count']
        
        print(f"   Total connected to real data: {connected_count:,}")
        print(f"   Empty-to-empty connections: {total_empty - isolated - connected_count:,}")
        
        # Check relationships involving empty nodes
        result = session.run("""
            MATCH (n)-[r]-()
            WHERE labels(n) = [] AND size(keys(n)) = 0
            RETURN type(r) as rel_type, count(*) as count
            ORDER BY count DESC
        """)
        
        print(f"\n   Relationship types on empty nodes:")
        for record in result:
            print(f"     • {record['rel_type']}: {record['count']:,}")
    
    driver.close()

def delete_isolated_empty_nodes(dry_run=True):
    """Delete completely isolated empty nodes."""
    
    uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    user = os.getenv('NEO4J_USER', 'neo4j')
    password = os.getenv('NEO4J_PASSWORD', 'axiom_neo4j')
    
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    with driver.session() as session:
        if dry_run:
            # Count what would be deleted
            result = session.run("""
                MATCH (n)
                WHERE labels(n) = [] AND size(keys(n)) = 0
                AND NOT (n)-[]-()
                RETURN count(n) as would_delete
            """)
            count = result.single()['would_delete']
            print(f"\n🔍 DRY RUN: Would delete {count:,} isolated empty nodes")
        else:
            # Actually delete
            result = session.run("""
                MATCH (n)
                WHERE labels(n) = [] AND size(keys(n)) = 0
                AND NOT (n)-[]-()
                DELETE n
                RETURN count(n) as deleted
            """)
            count = result.single()['deleted'] if result.single() else 0
            print(f"\n✅ DELETED {count:,} isolated empty nodes")
    
    driver.close()
    return count

def delete_empty_to_empty_chains(dry_run=True):
    """Delete chains of empty nodes connecting to each other."""
    
    uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    user = os.getenv('NEO4J_USER', 'neo4j')
    password = os.getenv('NEO4J_PASSWORD', 'axiom_neo4j')
    
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    with driver.session() as session:
        if dry_run:
            # Count chains
            result = session.run("""
                MATCH (e1)-[r]-(e2)
                WHERE labels(e1) = [] AND size(keys(e1)) = 0
                AND labels(e2) = [] AND size(keys(e2)) = 0
                RETURN count(DISTINCT e1) + count(DISTINCT e2) as chain_nodes,
                       count(r) as chain_relationships
            """)
            record = result.single()
            print(f"\n🔍 DRY RUN: Would delete empty-to-empty chains:")
            print(f"   Nodes: {record['chain_nodes']:,}")
            print(f"   Relationships: {record['chain_relationships']:,}")
        else:
            # Delete empty-to-empty relationships first
            result = session.run("""
                MATCH (e1)-[r]-(e2)
                WHERE labels(e1) = [] AND size(keys(e1)) = 0
                AND labels(e2) = [] AND size(keys(e2)) = 0
                DELETE r
                RETURN count(r) as deleted_rels
            """)
            rels_deleted = result.single()['deleted_rels'] if result.single() else 0
            
            # Then delete now-isolated empty nodes
            result = session.run("""
                MATCH (n)
                WHERE labels(n) = [] AND size(keys(n)) = 0
                AND NOT (n)-[]-()
                DELETE n
                RETURN count(n) as deleted_nodes
            """)
            nodes_deleted = result.single()['deleted_nodes'] if result.single() else 0
            
            print(f"\n✅ DELETED empty-to-empty chains:")
            print(f"   Relationships: {rels_deleted:,}")
            print(f"   Nodes: {nodes_deleted:,}")
    
    driver.close()

def investigate_empty_connected_to_companies():
    """Investigate why empty nodes connect to Company nodes."""
    
    uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    user = os.getenv('NEO4J_USER', 'neo4j')
    password = os.getenv('NEO4J_PASSWORD', 'axiom_neo4j')
    
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    with driver.session() as session:
        # Sample companies with empty node connections
        result = session.run("""
            MATCH (empty)-[r]-(c:Company)
            WHERE labels(empty) = [] AND size(keys(empty)) = 0
            RETURN c.symbol as company_symbol, 
                   c.name as company_name,
                   type(r) as relationship,
                   count(empty) as empty_count
            ORDER BY empty_count DESC
            LIMIT 10
        """)
        
        print(f"\n🔍 Companies with empty node connections:")
        for record in result:
            print(f"   {record['company_symbol']} ({record['company_name']}): "
                  f"{record['empty_count']} empty nodes via {record['relationship']}")
    
    driver.close()

def generate_cleanup_report():
    """Generate final cleanup report."""
    
    uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    user = os.getenv('NEO4J_USER', 'neo4j')
    password = os.getenv('NEO4J_PASSWORD', 'axiom_neo4j')
    
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    with driver.session() as session:
        # Get current counts
        result = session.run("""
            MATCH (n)
            RETURN 
                count(CASE WHEN labels(n) = [] AND size(keys(n)) = 0 THEN 1 END) as empty_nodes,
                count(CASE WHEN size(labels(n)) > 0 THEN 1 END) as labeled_nodes,
                count(n) as total_nodes
        """)
        
        record = result.single()
        empty = record['empty_nodes']
        labeled = record['labeled_nodes']
        total = record['total_nodes']
        
        print(f"\n📊 Neo4j Node Status:")
        print(f"   Total nodes: {total:,}")
        print(f"   Labeled (good): {labeled:,} ({labeled/total*100:.1f}%)")
        print(f"   Empty (bad): {empty:,} ({empty/total*100:.1f}%)")
        
        # Get relationship counts
        result = session.run("""
            MATCH ()-[r]->()
            RETURN count(r) as total_relationships
        """)
        rels = result.single()['total_relationships']
        print(f"   Total relationships: {rels:,}")
    
    driver.close()

if __name__ == '__main__':
    print("=" * 70)
    print("NEO4J EMPTY NODE CLEANUP SCRIPT")
    print("=" * 70)
    
    # Step 1: Analyze
    print("\nStep 1: Analyzing empty nodes...")
    analyze_empty_nodes()
    
    # Step 2: Investigate Company connections
    print("\nStep 2: Investigating Company connections...")
    investigate_empty_connected_to_companies()
    
    # Step 3: Dry run cleanup
    print("\nStep 3: Dry run cleanup...")
    delete_isolated_empty_nodes(dry_run=True)
    delete_empty_to_empty_chains(dry_run=True)
    
    # Step 4: Confirm before actual deletion
    print("\n" + "=" * 70)
    response = input("\n⚠️  Ready to DELETE empty nodes? (yes/no): ")
    
    if response.lower() == 'yes':
        print("\n🗑️  Deleting empty nodes...")
        
        # Delete isolated first
        delete_isolated_empty_nodes(dry_run=False)
        
        # Delete empty-to-empty chains
        delete_empty_to_empty_chains(dry_run=False)
        
        print("\n✅ Cleanup complete!")
        
        # Final report
        generate_cleanup_report()
    else:
        print("\n❌ Cleanup cancelled. No changes made.")
        print("   Run again with 'yes' when ready.")
    
    print("\n" + "=" * 70)