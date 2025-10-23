"""
Comprehensive Database Integration Demo.

Demonstrates:
- PostgreSQL integration for financial data
- Vector DB for semantic search
- VaR and portfolio optimization storage
- Market data management
- RAG with document embeddings
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
import numpy as np
import pandas as pd

# Database imports
from axiom.database import (
    get_db,
    SessionManager,
    get_migration_manager,
    VectorStore,
    get_vector_store,
)
from axiom.database.integrations import (
    VaRIntegration,
    PortfolioIntegration,
    MarketDataIntegration,
    VectorIntegration,
)
from axiom.database.models import TradeType, OrderType

# Model imports
from axiom.models.risk.var_models import VaRCalculator, VaRMethod
from axiom.models.portfolio.optimization import PortfolioOptimizer, OptimizationMethod


def demo_database_setup():
    """Demo: Database initialization and migration."""
    print("=" * 80)
    print("DATABASE SETUP DEMO")
    print("=" * 80)
    
    # Get database connection
    db = get_db()
    print(f"✓ Connected to database")
    
    # Check connection health
    if db.health_check():
        print(f"✓ Database health check passed")
    
    # Get pool status
    pool_status = db.get_pool_status()
    print(f"✓ Connection pool: {pool_status['pool_size']} connections")
    
    # Initialize migration manager
    migration_manager = get_migration_manager()
    
    # Initialize schema
    migration_manager.init_schema()
    print(f"✓ Database schema initialized")
    
    # Check migration status
    status = migration_manager.status()
    print(f"✓ Migration status: {status['applied_migrations']} applied, "
          f"{status['pending_migrations']} pending")
    
    print()


def demo_var_integration():
    """Demo: VaR calculation storage."""
    print("=" * 80)
    print("VAR INTEGRATION DEMO")
    print("=" * 80)
    
    # Generate sample returns
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 252)  # 1 year daily returns
    
    # Calculate VaR
    calculator = VaRCalculator(default_confidence=0.95)
    portfolio_value = 1_000_000
    
    var_result = calculator.calculate_var(
        portfolio_value=portfolio_value,
        returns=returns,
        method=VaRMethod.HISTORICAL,
        time_horizon_days=1
    )
    
    print(f"✓ Calculated VaR: ${var_result.var_amount:,.2f}")
    print(f"  Confidence: {var_result.confidence_level*100:.0f}%")
    print(f"  Method: {var_result.method.value}")
    
    # Store in database
    var_integration = VaRIntegration()
    stored_var = var_integration.store_var_result(
        portfolio_id="demo_portfolio",
        var_result=var_result
    )
    
    print(f"✓ Stored VaR calculation (ID: {stored_var.id})")
    
    # Retrieve history
    history = var_integration.get_var_history("demo_portfolio", days=30)
    print(f"✓ Retrieved {len(history)} VaR calculations from history")
    
    # Get latest VaR
    latest = var_integration.get_latest_var("demo_portfolio")
    if latest:
        print(f"✓ Latest VaR: ${latest.var_amount:,.2f}")
    
    print()


def demo_portfolio_integration():
    """Demo: Portfolio optimization and performance tracking."""
    print("=" * 80)
    print("PORTFOLIO INTEGRATION DEMO")
    print("=" * 80)
    
    # Generate sample returns for multiple assets
    np.random.seed(42)
    n_days = 252
    assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
    
    returns_data = {}
    for asset in assets:
        returns_data[asset] = np.random.normal(0.001, 0.02, n_days)
    
    returns_df = pd.DataFrame(returns_data)
    
    # Run portfolio optimization
    optimizer = PortfolioOptimizer(risk_free_rate=0.02)
    opt_result = optimizer.optimize(
        returns=returns_df,
        method=OptimizationMethod.MAX_SHARPE
    )
    
    print(f"✓ Optimized portfolio:")
    print(f"  Expected Return: {opt_result.metrics.expected_return*100:.2f}%")
    print(f"  Volatility: {opt_result.metrics.volatility*100:.2f}%")
    print(f"  Sharpe Ratio: {opt_result.metrics.sharpe_ratio:.3f}")
    
    print(f"\n  Optimal Weights:")
    for asset, weight in opt_result.get_weights_dict().items():
        if weight > 0.01:
            print(f"    {asset}: {weight*100:.2f}%")
    
    # Store optimization result
    portfolio_integration = PortfolioIntegration()
    stored_opt = portfolio_integration.store_optimization_result(
        portfolio_id="demo_portfolio",
        opt_result=opt_result
    )
    
    print(f"\n✓ Stored optimization result (ID: {stored_opt.id})")
    
    # Store performance metrics
    perf_metric = portfolio_integration.store_performance_metrics(
        portfolio_id="demo_portfolio",
        metrics=opt_result.metrics,
        portfolio_value=Decimal('1000000'),
        num_positions=len(assets)
    )
    
    print(f"✓ Stored performance metrics (ID: {perf_metric.id})")
    
    # Store positions
    for asset, weight in opt_result.get_weights_dict().items():
        if weight > 0.01:
            position = portfolio_integration.store_position(
                portfolio_id="demo_portfolio",
                symbol=asset,
                quantity=Decimal(str(weight * 1000)),  # Example quantity
                avg_cost=Decimal('150.00'),
                current_price=Decimal('155.00')
            )
            print(f"✓ Stored position: {asset} (ID: {position.id})")
    
    # Store a sample trade
    trade = portfolio_integration.store_trade(
        portfolio_id="demo_portfolio",
        symbol='AAPL',
        trade_type=TradeType.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal('100'),
        price=Decimal('150.00'),
        commission=Decimal('1.00')
    )
    
    print(f"✓ Stored trade: BUY 100 AAPL @ $150.00 (ID: {trade.id})")
    
    print()


def demo_market_data_integration():
    """Demo: Market data storage."""
    print("=" * 80)
    print("MARKET DATA INTEGRATION DEMO")
    print("=" * 80)
    
    # Generate sample price data
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    
    price_data = {
        'Open': np.random.uniform(145, 155, 30),
        'High': np.random.uniform(150, 160, 30),
        'Low': np.random.uniform(140, 150, 30),
        'Close': np.random.uniform(145, 155, 30),
        'Volume': np.random.uniform(1e6, 5e6, 30),
    }
    
    df = pd.DataFrame(price_data, index=dates)
    df['Adj Close'] = df['Close']
    
    # Store price data
    market_data = MarketDataIntegration()
    count = market_data.store_price_data(
        df=df,
        symbol='AAPL',
        source='demo'
    )
    
    print(f"✓ Stored {count} price records for AAPL")
    
    # Store fundamental data
    fundamental_data = {
        'company_name': 'Apple Inc.',
        'sector': 'Technology',
        'industry': 'Consumer Electronics',
        'market_cap': 3_000_000_000_000,
        'revenue': 394_328_000_000,
        'net_income': 99_803_000_000,
        'eps': 6.13,
        'pe_ratio': 28.5,
    }
    
    fundamental = market_data.store_fundamentals(
        symbol='AAPL',
        data=fundamental_data,
        report_date=datetime.now(),
        source='demo'
    )
    
    print(f"✓ Stored fundamental data for AAPL (ID: {fundamental.id})")
    print(f"  Market Cap: ${fundamental.market_cap:,.0f}")
    print(f"  Revenue: ${fundamental.revenue:,.0f}")
    print(f"  P/E Ratio: {fundamental.pe_ratio}")
    
    print()


def demo_vector_integration():
    """Demo: Vector database and semantic search."""
    print("=" * 80)
    print("VECTOR DATABASE INTEGRATION DEMO")
    print("=" * 80)
    
    # Initialize vector integration
    vector_integration = VectorIntegration(vector_store_type='chroma')
    
    # Sample documents (SEC filings, research reports, etc.)
    documents = [
        {
            'id': 'sec_filing_aapl_2023q4',
            'type': 'sec_filing',
            'symbol': 'AAPL',
            'content': 'Apple Inc. reported strong quarterly results with revenue growth of 8% year-over-year. iPhone sales remained the primary revenue driver, contributing 52% of total revenue. Services segment showed remarkable growth at 16% YoY.',
            'metadata': {
                'title': 'Apple Inc. Q4 2023 10-K Filing',
                'source_url': 'https://sec.gov/example',
                'keywords': ['revenue', 'growth', 'iphone', 'services']
            }
        },
        {
            'id': 'research_aapl_valuation',
            'type': 'research',
            'symbol': 'AAPL',
            'content': 'Our DCF valuation model suggests Apple is fairly valued at current levels. The company\'s strong cash generation and ecosystem lock-in provide competitive advantages. Price target: $185.',
            'metadata': {
                'title': 'Apple Inc. Valuation Analysis',
                'source_url': 'https://research.example.com',
                'keywords': ['valuation', 'dcf', 'target price']
            }
        },
        {
            'id': 'news_aapl_ai',
            'type': 'news',
            'symbol': 'AAPL',
            'content': 'Apple announces major AI initiative, integrating advanced machine learning capabilities into its product ecosystem. The move positions Apple as a serious competitor in the AI space.',
            'metadata': {
                'title': 'Apple Unveils AI Strategy',
                'source_url': 'https://news.example.com',
                'keywords': ['ai', 'machine learning', 'innovation']
            }
        }
    ]
    
    # Generate simple embeddings (in production, use proper embedding models)
    np.random.seed(42)
    embedding_dim = 384  # Common dimension for sentence transformers
    
    for doc in documents:
        # Generate random embedding (replace with actual model in production)
        embedding = np.random.randn(embedding_dim)
        embedding = embedding / np.linalg.norm(embedding)  # Normalize
        
        # Store document with embedding
        stored_doc = vector_integration.store_document_embedding(
            document_id=doc['id'],
            document_type=doc['type'],
            content=doc['content'],
            embedding=embedding,
            embedding_model='demo-model-v1',
            symbol=doc.get('symbol'),
            metadata=doc.get('metadata')
        )
        
        print(f"✓ Stored document: {doc['id']} (ID: {stored_doc.id})")
    
    # Perform semantic search
    query_embedding = np.random.randn(embedding_dim)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    
    print(f"\n✓ Searching for similar documents...")
    results = vector_integration.search_documents(
        query_embedding=query_embedding,
        document_type='sec_filing',
        top_k=3,
        symbol='AAPL'
    )
    
    print(f"✓ Found {len(results)} similar documents:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result['id']} (score: {result['score']:.4f})")
    
    print()


def demo_query_performance():
    """Demo: Query optimization and performance."""
    print("=" * 80)
    print("QUERY PERFORMANCE DEMO")
    print("=" * 80)
    
    from time import time
    from axiom.database.models import PriceData
    
    session = SessionManager()
    
    # Test indexed query (fast)
    start = time()
    result = session.query(PriceData).filter(
        PriceData.symbol == 'AAPL'
    ).order_by(PriceData.timestamp.desc()).limit(100).all()
    elapsed = time() - start
    
    print(f"✓ Indexed query (symbol): {len(result)} records in {elapsed*1000:.2f}ms")
    
    # Test date range query (optimized with index)
    start = time()
    cutoff = datetime.now() - timedelta(days=30)
    result = session.query(PriceData).filter(
        PriceData.symbol == 'AAPL',
        PriceData.timestamp >= cutoff
    ).all()
    elapsed = time() - start
    
    print(f"✓ Date range query: {len(result)} records in {elapsed*1000:.2f}ms")
    
    # Connection pool status
    db = get_db()
    pool_status = db.get_pool_status()
    print(f"\n✓ Connection Pool Status:")
    print(f"  Active connections: {pool_status['checked_out']}")
    print(f"  Available: {pool_status['pool_size'] - pool_status['checked_out']}")
    
    session.close()
    print()


def main():
    """Run all demos."""
    print("\n" + "=" * 80)
    print("AXIOM DATABASE INTEGRATION - COMPREHENSIVE DEMO")
    print("=" * 80 + "\n")
    
    try:
        # 1. Database setup
        demo_database_setup()
        
        # 2. VaR integration
        demo_var_integration()
        
        # 3. Portfolio integration
        demo_portfolio_integration()
        
        # 4. Market data integration
        demo_market_data_integration()
        
        # 5. Vector database integration
        demo_vector_integration()
        
        # 6. Query performance
        demo_query_performance()
        
        print("=" * 80)
        print("✓ ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nKey Features Demonstrated:")
        print("  • PostgreSQL connection with pooling")
        print("  • VaR calculation storage and retrieval")
        print("  • Portfolio optimization tracking")
        print("  • Market data management")
        print("  • Vector database for semantic search")
        print("  • Query optimization and performance")
        print("\n" + "=" * 80 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        from axiom.database import close_db
        close_db()
        print("✓ Database connection closed")


if __name__ == "__main__":
    main()