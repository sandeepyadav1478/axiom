# Axiom Database Infrastructure

Institutional-grade data management for quantitative finance and AI-powered analytics.

## Overview

This module provides a comprehensive database infrastructure designed for high-performance financial data management and AI-powered semantic search. It integrates seamlessly with existing VaR models, portfolio optimization, and M&A workflows.

## Features

### 🗄️ PostgreSQL Integration
- **Connection Pooling**: High-performance connection management with automatic recycling
- **Transaction Support**: ACID-compliant transactions with automatic rollback
- **Schema Management**: Version-controlled migrations with rollback support
- **Query Optimization**: Indexed queries for sub-millisecond performance

### 🔍 Vector Database Support
- **Multiple Backends**: Pinecone (production), Weaviate (self-hosted), ChromaDB (local)
- **Semantic Search**: Fast similarity search for documents and company profiles
- **RAG Support**: Retrieval Augmented Generation for AI-powered insights
- **Automatic Sync**: PostgreSQL ↔ Vector DB synchronization

### 📊 Data Models
- **Price Data**: OHLCV with volume, adjusted prices, VWAP
- **Portfolio Management**: Positions, trades, P&L tracking
- **Risk Metrics**: VaR calculations, performance metrics
- **Fundamentals**: Company financials, ratios, growth metrics
- **Document Embeddings**: SEC filings, research reports, news

### 🔄 Integrations
- **VaR Models**: Automatic storage of VaR calculations
- **Portfolio Optimization**: Track optimization results and rebalancing
- **Market Data**: Bulk import from multiple data providers
- **Vector Search**: Semantic search for M&A target identification

## Quick Start

### 1. Setup Database

```bash
# Start PostgreSQL and vector DB with Docker Compose
cd axiom/database
docker-compose up -d

# For production with Weaviate
docker-compose --profile vector-db up -d

# For local development with ChromaDB
docker-compose --profile vector-db-light up -d
```

### 2. Configure Environment

```bash
# Copy example configuration
cp .env.example .env

# Edit database credentials
# POSTGRES_USER=axiom
# POSTGRES_PASSWORD=your_secure_password
# POSTGRES_DB=axiom_finance
```

### 3. Initialize Schema

```python
from axiom.database import get_db, get_migration_manager

# Connect to database
db = get_db()

# Initialize schema
migration_manager = get_migration_manager()
migration_manager.init_schema()

print("✓ Database initialized!")
```

## Usage Examples

### Store VaR Calculations

```python
from axiom.database.integrations import VaRIntegration
from axiom.models.risk.var_models import VaRCalculator, VaRMethod
import numpy as np

# Calculate VaR
calculator = VaRCalculator()
returns = np.random.normal(0.001, 0.02, 252)
var_result = calculator.calculate_var(
    portfolio_value=1_000_000,
    returns=returns,
    method=VaRMethod.HISTORICAL
)

# Store in database
var_integration = VaRIntegration()
stored_var = var_integration.store_var_result(
    portfolio_id="my_portfolio",
    var_result=var_result
)

print(f"VaR stored: ${stored_var.var_amount:,.2f}")
```

### Portfolio Optimization Tracking

```python
from axiom.database.integrations import PortfolioIntegration
from axiom.models.portfolio.optimization import PortfolioOptimizer
import pandas as pd

# Optimize portfolio
optimizer = PortfolioOptimizer()
returns_df = pd.DataFrame(...)  # Your returns data

opt_result = optimizer.optimize(
    returns=returns_df,
    method="max_sharpe"
)

# Store optimization
portfolio_integration = PortfolioIntegration()
stored_opt = portfolio_integration.store_optimization_result(
    portfolio_id="my_portfolio",
    opt_result=opt_result
)

print(f"Sharpe Ratio: {opt_result.metrics.sharpe_ratio:.3f}")
```

### Semantic Document Search

```python
from axiom.database.integrations import VectorIntegration
import numpy as np

# Initialize vector integration
vector_integration = VectorIntegration()

# Store document
embedding = model.encode("SEC filing content...")  # Use actual embedding model
stored_doc = vector_integration.store_document_embedding(
    document_id="sec_10k_aapl_2023",
    document_type="sec_filing",
    content="Full document text...",
    embedding=embedding,
    embedding_model="sentence-transformers",
    symbol="AAPL"
)

# Search similar documents
query_embedding = model.encode("revenue growth analysis")
results = vector_integration.search_documents(
    query_embedding=query_embedding,
    document_type="sec_filing",
    top_k=10,
    symbol="AAPL"
)

for result in results:
    print(f"{result['id']}: {result['score']:.4f}")
```

### Market Data Storage

```python
from axiom.database.integrations import MarketDataIntegration
import yfinance as yf

# Download market data
ticker = yf.Ticker("AAPL")
hist = ticker.history(period="1y")

# Store in database
market_data = MarketDataIntegration()
count = market_data.store_price_data(
    df=hist,
    symbol="AAPL",
    source="yahoo_finance"
)

print(f"Stored {count} price records")
```

### Transaction Management

```python
from axiom.database import session_scope

# Automatic transaction management
with session_scope() as session:
    # All operations in this block are atomic
    session.add(position1)
    session.add(position2)
    session.add(trade)
    # Automatic commit on success, rollback on error
```

## Architecture

### Database Schema

```
price_data          - OHLCV price data with volume
├─ symbol (indexed)
├─ timestamp (indexed)
├─ timeframe
└─ OHLCV + volume

portfolio_positions - Current holdings
├─ portfolio_id (indexed)
├─ symbol (indexed)
├─ quantity, avg_cost
└─ P&L tracking

trades              - Trade execution history
├─ portfolio_id (indexed)
├─ symbol, type, quantity
└─ execution details

var_calculations    - VaR history
├─ portfolio_id (indexed)
├─ method, confidence
└─ var_amount, expected_shortfall

performance_metrics - Portfolio performance
├─ portfolio_id (indexed)
├─ returns, volatility
└─ sharpe, sortino, max_drawdown

portfolio_optimizations - Optimization results
├─ portfolio_id (indexed)
├─ method, weights
└─ expected_return, volatility

company_fundamentals - Company financials
├─ symbol (indexed)
├─ financials, ratios
└─ growth metrics

document_embeddings - Vector embeddings
├─ document_id (indexed)
├─ type, symbol (indexed)
└─ vector_db sync status
```

### Connection Pooling

```python
# Automatic connection pooling
DatabaseConnection(
    pool_size=10,         # Base connections
    max_overflow=20,      # Additional connections
    pool_timeout=30,      # Get connection timeout
    pool_recycle=3600     # Recycle after 1 hour
)
```

### Performance Optimizations

1. **Indexes**: All foreign keys and frequently queried columns
2. **Constraints**: Check constraints for data integrity
3. **Connection Pooling**: Reuse connections across requests
4. **Batch Operations**: Bulk insert/update for high throughput
5. **Query Optimization**: Indexed queries with proper joins

## Vector Database Comparison

| Feature | Pinecone | Weaviate | ChromaDB |
|---------|----------|----------|----------|
| **Deployment** | Cloud | Self-hosted | Local |
| **Performance** | Excellent | Very Good | Good |
| **Scalability** | Unlimited | High | Limited |
| **Cost** | $$ | $ | Free |
| **Best For** | Production | Enterprise | Development |

### Choosing a Vector DB

**Pinecone** - Production deployments
- ✓ Managed service, no infrastructure
- ✓ Automatic scaling
- ✓ Best performance
- ✗ Requires API key and subscription

**Weaviate** - Self-hosted production
- ✓ Full control over data
- ✓ GraphQL queries
- ✓ On-premise deployment
- ✗ Requires infrastructure management

**ChromaDB** - Local development
- ✓ No setup required
- ✓ Fast local queries
- ✓ Perfect for testing
- ✗ Not for production scale

## Migration Management

### Create Migration

```python
from axiom.database import get_migration_manager

migration_manager = get_migration_manager()

def upgrade(session):
    session.execute("""
        ALTER TABLE price_data 
        ADD COLUMN extended_hours BOOLEAN DEFAULT FALSE
    """)

def downgrade(session):
    session.execute("""
        ALTER TABLE price_data 
        DROP COLUMN extended_hours
    """)

migration_manager.register_migration(
    version=1,
    name="add_extended_hours",
    up=upgrade,
    down=downgrade
)
```

### Run Migrations

```python
# Migrate to latest
migration_manager.migrate()

# Migrate to specific version
migration_manager.migrate(target_version=5)

# Rollback
migration_manager.rollback(target_version=4)

# Check status
status = migration_manager.status()
print(f"Current: v{status['current_version']}, Latest: v{status['latest_version']}")
```

## Performance Monitoring

### Query Performance

```python
from axiom.database import get_db

db = get_db()

# Check pool status
pool_status = db.get_pool_status()
print(f"Active: {pool_status['checked_out']}/{pool_status['pool_size']}")

# Health check
if db.health_check():
    print("✓ Database healthy")
```

### Slow Query Logging

```bash
# Enable in .env
DB_QUERY_LOGGING=true
DB_SLOW_QUERY_THRESHOLD=1.0
```

## Production Deployment

### Docker Compose Production

```yaml
# docker-compose.prod.yml
services:
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}  # Use secrets
    volumes:
      - /var/lib/postgresql/data  # Persistent volume
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
```

### Backup Strategy

```bash
# Automated backups
docker exec axiom_postgres pg_dump -U axiom axiom_finance > backup.sql

# Restore
docker exec -i axiom_postgres psql -U axiom axiom_finance < backup.sql
```

### Monitoring

```python
# Health endpoint
from axiom.database import get_db

def health_check():
    db = get_db()
    return {
        "database": "healthy" if db.health_check() else "unhealthy",
        "pool": db.get_pool_status()
    }
```

## Security Best Practices

1. **Never commit credentials**: Use environment variables
2. **Use SSL/TLS**: Enable encrypted connections
3. **Rotate passwords**: Regular password rotation
4. **Principle of least privilege**: Limited database permissions
5. **Audit logs**: Track all data access

## Troubleshooting

### Connection Issues

```python
# Test connection
from axiom.database import get_db

try:
    db = get_db()
    if db.health_check():
        print("✓ Connected")
except Exception as e:
    print(f"✗ Connection failed: {e}")
```

### Pool Exhaustion

```bash
# Increase pool size in .env
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=40
```

### Slow Queries

```sql
-- Analyze query performance
EXPLAIN ANALYZE 
SELECT * FROM price_data 
WHERE symbol = 'AAPL' AND timestamp > NOW() - INTERVAL '30 days';
```

## API Reference

See [API Documentation](./API.md) for detailed API reference.

## Contributing

When adding new models:
1. Define model in `models.py`
2. Create migration in `migrations.py`
3. Add integration in `integrations.py`
4. Update tests
5. Update documentation

## License

Part of the Axiom Investment Banking Analytics Platform.