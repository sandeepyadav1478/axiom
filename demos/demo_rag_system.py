"""
Production RAG System Demo for M&A Intelligence.

Demonstrates:
1. Document ingestion (PDF, DOCX, TXT)
2. Vector storage with ChromaDB
3. Graph enhancement with Neo4j
4. Hybrid retrieval (vector + graph)
5. Claude-powered generation
6. Complete RAG pipeline

This showcases advanced AI architecture combining:
- Vector search for semantic similarity
- Graph traversal for relationship context
- LLM generation for natural language responses
"""

import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from axiom.models.rag import (
    RAGPipeline,
    RAGConfig,
    DocumentProcessor,
    EmbeddingService,
    HybridRetriever,
    GraphEnhancer
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_documents():
    """Create sample M&A documents for testing."""
    
    sample_docs = {
        "microsoft_linkedin.txt": """
Microsoft Acquires LinkedIn for $26.2 Billion

Date: June 13, 2016

Microsoft Corporation (NASDAQ: MSFT) and LinkedIn Corporation (NYSE: LNKD) 
announced that Microsoft will acquire LinkedIn for $196 per share in an 
all-cash transaction valued at $26.2 billion, inclusive of LinkedIn's net cash.

Strategic Rationale:
- Enhance Microsoft's productivity and business processes
- Integrate LinkedIn's network with Microsoft Office and Dynamics
- Accelerate growth in cloud services

Deal Structure:
- All-cash transaction
- $196 per share
- Total value: $26.2 billion
- Expected closing: End of 2016

Companies Involved:
- Acquirer: Microsoft Corporation
- Target: LinkedIn Corporation
- Advisors: Morgan Stanley, Qatalyst Partners

Impact:
- Microsoft expands into professional networking
- LinkedIn gains resources for growth
- Creates largest professional network integrated with productivity tools
""",
        
        "disney_fox.txt": """
The Walt Disney Company Acquires 21st Century Fox Assets

Date: March 20, 2019

The Walt Disney Company (NYSE: DIS) successfully completed the acquisition 
of 21st Century Fox Inc. (NASDAQ: TFCF) for approximately $71.3 billion.

Transaction Details:
- Total value: $71.3 billion
- Structure: Stock and cash
- Closing date: March 20, 2019

Assets Acquired:
- 20th Century Fox Film Studio
- Fox Television Studios
- FX Networks
- National Geographic Partners
- Star India
- Hulu stake (increased to 60%)

Strategic Benefits:
- Content library expansion
- International growth
- Direct-to-consumer capabilities
- Production capacity increase

Regulatory Approval:
- Department of Justice approval obtained
- International regulatory clearances completed
- Divestitures: Regional Sports Networks sold

Companies:
- Acquirer: The Walt Disney Company
- Seller: 21st Century Fox Inc.
- Key executives: Bob Iger (Disney CEO), Rupert Murdoch (Fox)

Market Impact:
- Media consolidation continues
- Streaming wars intensify
- Content is king strategy validated
""",
        
        "amazon_whole_foods.txt": """
Amazon Acquires Whole Foods Market

Date: August 28, 2017

Amazon.com, Inc. (NASDAQ: AMZN) completed the acquisition of 
Whole Foods Market, Inc. (NASDAQ: WFM) for $13.7 billion.

Transaction Overview:
- Purchase price: $42 per share
- Total enterprise value: $13.7 billion
- All-cash transaction
- Premium: 27% over closing price

Strategic Objectives:
- Enter physical retail market
- Leverage Amazon's technology and distribution
- Accelerate grocery delivery
- Integrate Prime membership

Immediate Changes:
- Price reductions on key items
- Amazon Prime integration
- Delivery partnerships
- Technology infrastructure upgrades

Companies Involved:
- Acquirer: Amazon.com, Inc.
- Target: Whole Foods Market, Inc.
- Whole Foods CEO: John Mackey (retained)

Industry Impact:
- Traditional grocery disruption
- Omnichannel retail evolution
- Technology-enabled commerce
- Prime ecosystem expansion

Financial Metrics:
- Deal value: $13.7 billion
- Whole Foods locations: 460+ stores
- Amazon's retail footprint: Significant expansion
"""
    }
    
    # Create temp directory
    temp_dir = Path("/tmp/rag_demo_docs")
    temp_dir.mkdir(exist_ok=True)
    
    file_paths = []
    for filename, content in sample_docs.items():
        file_path = temp_dir / filename
        file_path.write_text(content)
        file_paths.append(str(file_path))
        logger.info(f"Created sample document: {filename}")
    
    return file_paths


def populate_graph_database(graph_enhancer: GraphEnhancer):
    """Populate Neo4j with M&A relationships."""
    
    if not graph_enhancer.graph:
        logger.warning("Graph database not available, skipping population")
        return
    
    logger.info("Populating graph database with M&A relationships...")
    
    try:
        # Create companies
        companies = [
            ("MSFT", "Microsoft Corporation", "Technology", "Software"),
            ("LNKD", "LinkedIn Corporation", "Technology", "Social Media"),
            ("DIS", "The Walt Disney Company", "Media", "Entertainment"),
            ("FOX", "21st Century Fox", "Media", "Entertainment"),
            ("AMZN", "Amazon.com Inc.", "Technology", "E-commerce"),
            ("WFM", "Whole Foods Market", "Retail", "Grocery")
        ]
        
        for symbol, name, sector, industry in companies:
            graph_enhancer.graph.create_company(
                symbol=symbol,
                name=name,
                sector=sector,
                industry=industry
            )
        
        # Create acquisitions
        from datetime import datetime
        
        acquisitions = [
            ("MSFT", "LNKD", 26.2e9, datetime(2016, 6, 13), datetime(2016, 12, 31), "acquisition"),
            ("DIS", "FOX", 71.3e9, datetime(2017, 12, 14), datetime(2019, 3, 20), "acquisition"),
            ("AMZN", "WFM", 13.7e9, datetime(2017, 6, 16), datetime(2017, 8, 28), "acquisition")
        ]
        
        for acquirer, target, value, announce, complete, deal_type in acquisitions:
            graph_enhancer.graph.create_acquisition(
                acquirer_symbol=acquirer,
                target_symbol=target,
                deal_value=value,
                announcement_date=announce,
                completion_date=complete,
                deal_type=deal_type
            )
        
        logger.info("✅ Graph database populated with 6 companies and 3 acquisitions")
        
        # Verify
        stats = graph_enhancer.graph.get_statistics()
        logger.info(f"Graph stats: {stats}")
        
    except Exception as e:
        logger.error(f"Failed to populate graph: {e}")


def demo_rag_pipeline():
    """Demonstrate complete RAG pipeline."""
    
    print("\n" + "="*80)
    print("🚀 PRODUCTION RAG SYSTEM FOR M&A INTELLIGENCE")
    print("="*80 + "\n")
    
    # Step 1: Initialize components
    print("📦 Step 1: Initializing RAG Pipeline Components")
    print("-" * 80)
    
    config = RAGConfig(
        chunk_size=1000,
        chunk_overlap=200,
        top_k=5,
        similarity_threshold=0.7,
        enable_graph=True,
        enable_reranking=True,
        model="claude-3-5-sonnet-20241022",
        temperature=0.1
    )
    
    pipeline = RAGPipeline(config=config, collection_name="ma_intelligence_demo")
    logger.info("✅ RAG pipeline initialized")
    
    # Step 2: Create and ingest sample documents
    print("\n📄 Step 2: Creating Sample M&A Documents")
    print("-" * 80)
    
    doc_paths = create_sample_documents()
    print(f"✅ Created {len(doc_paths)} sample documents")
    
    # Step 3: Populate graph database
    print("\n🔗 Step 3: Populating Knowledge Graph")
    print("-" * 80)
    
    if pipeline.retriever.graph_enhancer:
        populate_graph_database(pipeline.retriever.graph_enhancer)
    else:
        print("⚠️  Graph database not available (optional)")
    
    # Step 4: Ingest documents
    print("\n💾 Step 4: Ingesting Documents into Vector Store")
    print("-" * 80)
    
    chunks_indexed = pipeline.ingest_documents(
        file_paths=doc_paths,
        metadata={"source": "demo", "category": "M&A_deals"}
    )
    print(f"✅ Indexed {chunks_indexed} chunks in ChromaDB")
    
    # Step 5: Query the system
    print("\n🔍 Step 5: Querying RAG System")
    print("-" * 80)
    
    queries = [
        "What was the largest M&A deal by value?",
        "Which technology companies were involved in acquisitions?",
        "What were the strategic reasons for Disney's acquisition of Fox?",
        "How did Amazon's acquisition of Whole Foods impact the retail industry?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'='*80}")
        print(f"Query {i}: {query}")
        print('='*80)
        
        try:
            response = pipeline.query(query=query, top_k=3)
            
            print(f"\n📊 Retrieval Performance:")
            print(f"  • Total time: {response.total_time_ms:.1f}ms")
            print(f"  • Retrieval: {response.retrieval_time_ms:.1f}ms")
            print(f"  • Generation: {response.generation_time_ms:.1f}ms")
            print(f"  • Sources found: {len(response.sources)}")
            print(f"  • Confidence: {response.confidence:.1%}")
            
            print(f"\n💡 Answer:")
            print("-" * 80)
            print(response.answer)
            
            if response.sources:
                print(f"\n📚 Sources ({len(response.sources)}):")
                for j, source in enumerate(response.sources, 1):
                    print(f"  {j}. {source['document_name']} "
                          f"(score: {source['relevance_score']:.2f})")
                    if source.get('companies'):
                        print(f"     Companies: {', '.join(source['companies'])}")
            
            print()
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            print(f"❌ Error: {e}")
    
    # Step 6: Show statistics
    print("\n📈 Step 6: System Statistics")
    print("-" * 80)
    
    stats = pipeline.get_statistics()
    
    print(f"\n🔧 Pipeline Configuration:")
    for key, value in stats['pipeline'].items():
        print(f"  • {key}: {value}")
    
    print(f"\n🔍 Retriever Stats:")
    retriever_stats = stats['retriever']
    print(f"  • Vector store: {retriever_stats['vector_store_type']}")
    print(f"  • Collections: {retriever_stats['total_collections']}")
    print(f"  • Graph enabled: {retriever_stats['graph_enabled']}")
    print(f"  • Reranking enabled: {retriever_stats['reranking_enabled']}")
    
    print(f"\n🎯 Embedding Service:")
    embedding_stats = stats['embedding']
    print(f"  • Model: {embedding_stats['model']}")
    print(f"  • Dimension: {embedding_stats['dimension']}")
    print(f"  • Cached embeddings: {embedding_stats['cached_embeddings']}")
    
    # Final summary
    print("\n" + "="*80)
    print("✅ RAG SYSTEM DEMO COMPLETE")
    print("="*80)
    print("""
Key Features Demonstrated:
✓ Document ingestion with intelligent chunking
✓ Vector embeddings with OpenAI
✓ ChromaDB for semantic search
✓ Neo4j graph enhancement (775K+ relationships capable)
✓ Hybrid retrieval (vector + graph)
✓ DSPy-optimized retrieval chains
✓ Claude API for high-quality generation
✓ Source attribution and confidence scoring
✓ Production-ready architecture

Architecture Highlights:
• Multi-modal retrieval (semantic + structural)
• Graph-enhanced context (M&A relationships)
• Advanced AI orchestration (DSPy)
• Enterprise-grade generation (Claude)
• Containerized deployment (Docker)
• Monitoring and observability (Prometheus + Grafana)
    """)


if __name__ == "__main__":
    try:
        demo_rag_pipeline()
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        print(f"\n❌ Demo failed: {e}")
        sys.exit(1)