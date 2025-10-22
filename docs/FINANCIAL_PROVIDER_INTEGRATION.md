# Financial Data Provider Integration Guide

## Overview

Axiom's M&A analytics platform now integrates multiple financial data providers to deliver comprehensive, high-quality financial data for investment banking workflows. This integration combines **Tavily** (web intelligence), **Financial Modeling Prep (FMP)**, **Finnhub**, and **Alpha Vantage** to provide robust, multi-source financial analytics.

## Architecture

### Financial Data Aggregator

The [`FinancialDataAggregator`](../axiom/integrations/data_sources/finance/financial_data_aggregator.py) serves as the central hub for all financial data operations:

```python
from axiom.integrations.data_sources.finance.financial_data_aggregator import get_financial_aggregator

# Get singleton aggregator instance
aggregator = get_financial_aggregator()

# Get company fundamentals with multi-provider consensus
fundamentals = await aggregator.get_company_fundamentals(
    company_identifier="MSFT",
    use_consensus=True  # Aggregate data from multiple providers
)

# Get comparable companies
comparables = await aggregator.get_comparable_companies(
    target_company="PLTR",
    use_consensus=True
)

# Get real-time market data
market_data = await aggregator.get_market_data(
    symbols=["MSFT", "GOOGL", "AMZN"]
)
```

### Provider Capabilities

| Provider | Free Tier | Premium Cost | Key Features |
|----------|-----------|--------------|--------------|
| **Tavily** | Yes | $0/month | Web intelligence, M&A news, market research |
| **Alpha Vantage** | 500 calls/day | $49/month | Fundamentals, real-time data, technical indicators |
| **FMP** | 250 calls/day | $14-99/month | Comprehensive financials, DCF models, screener |
| **Finnhub** | 60 calls/min | $7.99/month | Real-time quotes, insider trading, global coverage |

## Integration Points

### 1. Valuation Workflows

The valuation engine uses financial providers for:

- **DCF Analysis**: Historical financial data and projections
- **Comparable Analysis**: Automated comparable company identification
- **Market Data**: Real-time trading multiples and valuations

```python
from axiom.core.analysis_engines.valuation import MAValuationWorkflow

workflow = MAValuationWorkflow()

# Automatically uses financial providers for data gathering
valuation = await workflow.execute_comprehensive_valuation(
    target_company="Palantir Technologies",
    target_metrics={"revenue": 2_200_000_000}
)
```

**Integration Benefits:**
- ✅ Accurate historical financials from professional sources
- ✅ Multi-provider consensus for higher confidence
- ✅ Automatic fallback if a provider fails
- ✅ Comprehensive ratio and metric coverage

### 2. Due Diligence Workflows

The due diligence engine uses financial providers for:

- **Financial DD**: Revenue, profitability, and balance sheet analysis
- **Comprehensive Metrics**: Financial ratios, liquidity, leverage
- **Data Quality**: Professional-grade financial data

```python
from axiom.core.analysis_engines.due_diligence import MADueDiligenceWorkflow

workflow = MADueDiligenceWorkflow()

# Automatically enriches analysis with provider data
dd_result = await workflow.execute_comprehensive_dd(
    target_company="Snowflake Inc"
)
```

**Integration Benefits:**
- ✅ Comprehensive financial statement analysis
- ✅ Automated ratio calculations
- ✅ Higher confidence in financial assessments
- ✅ Structured data for AI analysis

### 3. Target Screening Workflows

The target screening engine uses financial providers for:

- **Financial Enrichment**: Automatic target profile enhancement
- **Screening Validation**: Verify financial criteria compliance
- **Data Accuracy**: Replace web scraping with API data

```python
from axiom.core.analysis_engines.target_screening import (
    MATargetScreeningWorkflow,
    TargetCriteria
)

workflow = MATargetScreeningWorkflow()
criteria = TargetCriteria(
    industry_sectors=["Enterprise Software"],
    min_revenue=100_000_000,
    min_ebitda_margin=0.15
)

# Automatically enhances targets with financial provider data
screening = await workflow.execute(criteria)
```

**Integration Benefits:**
- ✅ Accurate financial metrics for screening
- ✅ Higher quality target profiles
- ✅ Faster screening with API data
- ✅ Better confidence scores

### 4. Market Intelligence

The market intelligence engine uses financial providers for:

- **Competitor Analysis**: Real market data for competitor profiles
- **Market Data**: Current pricing and valuation metrics
- **Trend Analysis**: Financial performance trends

## Configuration

### Environment Variables

Configure providers in your `.env` file:

```bash
# Tavily (Web Intelligence)
TAVILY_API_KEY=tvly-your-key-here

# Alpha Vantage (FREE: 500 calls/day)
ALPHA_VANTAGE_API_KEY=key1,key2,key3  # Multiple keys for rotation

# Financial Modeling Prep (FREE: 250 calls/day)
FMP_API_KEY=your-fmp-key-here

# Finnhub (FREE: 60 calls/minute)
FINNHUB_API_KEY=your-finnhub-key-here

# API Rotation (optional)
FINANCIAL_API_ROTATION_ENABLED=true
ALPHA_VANTAGE_API_ROTATION_ENABLED=true
```

### Provider Initialization

The aggregator automatically initializes all configured providers on first use:

```python
from axiom.integrations.data_sources.finance.financial_data_aggregator import get_financial_aggregator

# Singleton instance - initialized once
aggregator = get_financial_aggregator()

# Check available providers
providers = aggregator.get_available_providers()
# Returns: ['alpha_vantage', 'fmp', 'finnhub']

# Check provider health
health = await aggregator.health_check()
# Returns: {'alpha_vantage': True, 'fmp': True, 'finnhub': True}
```

## Features

### Multi-Provider Consensus

When `use_consensus=True`, the aggregator queries multiple providers and builds a consensus response:

```python
# Query all providers for consensus
fundamentals = await aggregator.get_company_fundamentals(
    company_identifier="AAPL",
    use_consensus=True  # Queries all available providers
)

# Result includes consensus metadata
consensus = fundamentals.data_payload.get("consensus_data")
# {
#   "provider_count": 3,
#   "providers_used": ["alpha_vantage", "fmp", "finnhub"],
#   "individual_results": {...}
# }

# Confidence boosted by multiple sources
print(fundamentals.confidence)  # e.g., 0.92 (higher than single provider)
```

### Intelligent Fallback

If a provider fails, the aggregator automatically falls back to alternatives:

```python
# Even if FMP fails, will try Finnhub then Alpha Vantage
fundamentals = await aggregator.get_company_fundamentals(
    company_identifier="TSLA",
    use_consensus=False  # Single provider with fallback
)

# Logs show fallback chain:
# "Trying fmp for fundamentals"
# "fmp failed: Rate limit exceeded"
# "Trying finnhub for fundamentals"
# "Successfully got fundamentals from finnhub"
```

### Data Quality Scoring

The aggregator provides confidence scores based on:
- Provider data quality
- Number of sources used (consensus)
- Data freshness and completeness

```python
# Single provider: confidence ~0.85-0.90
# Consensus (2 providers): confidence ~0.90-0.95
# Consensus (3+ providers): confidence ~0.95-0.98
```

## Usage Examples

### Example 1: Company Analysis

```python
import asyncio
from axiom.integrations.data_sources.finance.financial_data_aggregator import get_financial_aggregator

async def analyze_company(symbol: str):
    aggregator = get_financial_aggregator()
    
    # Get fundamentals
    fundamentals = await aggregator.get_company_fundamentals(
        company_identifier=symbol,
        use_consensus=True
    )
    
    payload = fundamentals.data_payload
    
    print(f"Company: {payload.get('company_name')}")
    print(f"Revenue: ${payload.get('annual_revenue'):,.0f}")
    print(f"Market Cap: ${payload.get('market_cap'):,.0f}")
    print(f"EBITDA Margin: {payload.get('ebitda_margin'):.1%}")
    print(f"Data Quality: {fundamentals.confidence:.2f}")
    
    return fundamentals

asyncio.run(analyze_company("MSFT"))
```

### Example 2: Comparable Companies

```python
async def find_comparables(target: str):
    aggregator = get_financial_aggregator()
    
    # Get comparables from multiple sources
    comparables = await aggregator.get_comparable_companies(
        target_company=target,
        industry_sector="Enterprise Software",
        use_consensus=True
    )
    
    comps = comparables.data_payload.get("comparables", [])
    
    print(f"Found {len(comps)} comparables for {target}")
    for comp in comps[:5]:
        print(f"  - {comp.get('name')}: "
              f"${comp.get('market_cap'):,.0f} market cap, "
              f"{comp.get('similarity_score'):.2f} similarity")
    
    return comparables

asyncio.run(find_comparables("PLTR"))
```

### Example 3: M&A Valuation with Provider Data

```python
from axiom.core.analysis_engines.valuation import run_comprehensive_valuation

async def value_target(target: str, metrics: dict):
    # Automatically uses financial providers
    valuation = await run_comprehensive_valuation(
        target_company=target,
        target_metrics=metrics
    )
    
    print(f"Valuation Range: "
          f"${valuation.valuation_low/1e9:.2f}B - "
          f"${valuation.valuation_high/1e9:.2f}B")
    print(f"Comparable Count: {valuation.comparable_analysis.comp_count}")
    print(f"Data Confidence: {valuation.valuation_confidence:.2f}")
    
    return valuation

asyncio.run(value_target(
    "Databricks",
    {"revenue": 1_500_000_000, "ebitda": 300_000_000}
))
```

## Error Handling

The integration includes comprehensive error handling:

```python
from axiom.integrations.data_sources.finance.base_financial_provider import (
    FinancialProviderError
)

try:
    fundamentals = await aggregator.get_company_fundamentals("INVALID")
except FinancialProviderError as e:
    print(f"Provider error: {e.provider}")
    print(f"Message: {str(e)}")
    # Logs automatically capture full context
```

### Logging Integration

All provider operations are logged using [`AxiomLogger`](../axiom/core/logging/axiom_logger.py):

```python
# Automatic logging of:
# - Provider initialization
# - API calls and responses
# - Errors and fallbacks
# - Performance metrics
# - Data quality scores

# View logs for troubleshooting:
# "Initializing FMP provider with subscription level: free"
# "Fetching FMP fundamentals for MSFT"
# "Successfully retrieved FMP fundamentals for MSFT"
# "Retrieved historical financial data from Financial Modeling Prep"
```

## Performance Optimization

### API Call Minimization

The aggregator optimizes API usage through:

1. **Strategic provider selection** based on query type
2. **Batch operations** where supported
3. **Timeout management** (30s per provider)
4. **Parallel queries** for consensus building

### Cost Optimization

All providers support generous free tiers:

- **Total Free Capacity**: 500 (Alpha Vantage) + 250 (FMP) + 3,600/hr (Finnhub) = **4,350+ free calls/day**
- **Cost-effective premium**: Starting at $7.99/month (Finnhub)
- **API rotation**: Multiple free accounts maximize free tier usage

## Testing

Run the comprehensive integration demo:

```bash
# Test all providers and workflows
python demos/demo_financial_provider_integration.py
```

Expected output:
```
🚀 AXIOM FINANCIAL PROVIDER INTEGRATION DEMO
================================================================================

🔧 Environment Check:
   Configured Providers: Tavily, FMP, Finnhub, Alpha Vantage

🏦 FINANCIAL DATA AGGREGATOR DEMO
   ✅ Available Providers: alpha_vantage, fmp, finnhub

📈 COMPANY FUNDAMENTALS DEMO - MSFT
   ✅ Retrieved from: Aggregator (3 sources)
   Confidence Score: 0.95

🏢 COMPARABLE COMPANIES DEMO - PLTR
   ✅ Found 12 Comparable Companies

💰 VALUATION WORKFLOW DEMO - Palantir Technologies
   ✅ Valuation Analysis Completed
   Base Case: $8.50B
   Range: $7.22B - $10.20B

✅ DEMO COMPLETED SUCCESSFULLY
```

## Troubleshooting

### No Providers Available

If you see "No financial data providers available":

1. Check `.env` file has API keys configured
2. Verify environment variables are loaded
3. Run health check: `await aggregator.health_check()`

### Provider Failures

If a specific provider fails:

1. Check API key validity
2. Verify rate limits not exceeded
3. Check provider's service status
4. Review logs for detailed error messages

### Low Confidence Scores

If confidence scores are unexpectedly low:

1. Enable consensus mode: `use_consensus=True`
2. Check number of providers available
3. Verify API keys for all providers
4. Review data payload for completeness

## Best Practices

### 1. Use Consensus for Critical Decisions

```python
# For M&A valuations, due diligence - use consensus
fundamentals = await aggregator.get_company_fundamentals(
    company_identifier=symbol,
    use_consensus=True  # Higher confidence with multiple sources
)
```

### 2. Single Provider for Screening

```python
# For target screening, single provider is sufficient
fundamentals = await aggregator.get_company_fundamentals(
    company_identifier=symbol,
    use_consensus=False  # Faster, conserves API calls
)
```

### 3. Monitor API Usage

```python
# Check provider info for usage tracking
provider_info = aggregator.get_provider_info()

for name, info in provider_info.items():
    print(f"{name}: {info.get('subscription_level')}")
```

### 4. Handle Provider Unavailability

```python
# Always handle potential provider failures
try:
    data = await aggregator.get_company_fundamentals(symbol)
except FinancialProviderError as e:
    logger.error(f"Provider failed: {e}")
    # Fallback to alternative data source
```

## Workflow Integration Summary

### Valuation Workflow (`axiom/core/analysis_engines/valuation.py`)

**Integrated Methods:**
- [`_gather_projection_data()`](../axiom/core/analysis_engines/valuation.py:893) - Uses providers for historical financials
- [`_identify_comparable_companies()`](../axiom/core/analysis_engines/valuation.py:691) - Uses providers for comp screening

**Benefits:**
- More accurate DCF inputs
- Better comparable company selection
- Higher valuation confidence

### Due Diligence Workflow (`axiom/core/analysis_engines/due_diligence.py`)

**Integrated Methods:**
- [`_gather_financial_information()`](../axiom/core/analysis_engines/due_diligence.py:478) - Comprehensive financial data from providers

**Benefits:**
- Deeper financial analysis
- More evidence sources
- Better risk assessment

### Target Screening Workflow (`axiom/core/analysis_engines/target_screening.py`)

**Integrated Methods:**
- [`_enhance_financial_data()`](../axiom/core/analysis_engines/target_screening.py:672) - Enriches target profiles with provider data

**Benefits:**
- Accurate screening metrics
- Higher quality target profiles
- Better prioritization

### Market Intelligence Workflow (`axiom/core/analysis_engines/market_intelligence.py`)

**Integrated Methods:**
- [`_parse_competitors_from_intelligence()`](../axiom/core/analysis_engines/market_intelligence.py:554) - Enriches competitor profiles with market data

**Benefits:**
- Real-time competitor metrics
- Better competitive analysis
- Market-based valuations

## Migration Guide

### Updating Existing Code

If you have existing workflows, update them to use the aggregator:

```python
# Old approach (web search only)
search_results = await tavily_client.search(
    query=f"{company} financial data"
)

# New approach (financial providers + web search)
from axiom.integrations.data_sources.finance.financial_data_aggregator import get_financial_aggregator

aggregator = get_financial_aggregator()
fundamentals = await aggregator.get_company_fundamentals(company)

# Fallback to web search if providers fail
if not fundamentals:
    search_results = await tavily_client.search(...)
```

### Adding New Workflows

When creating new M&A workflows:

```python
from axiom.integrations.data_sources.finance.financial_data_aggregator import get_financial_aggregator
from axiom.core.logging.axiom_logger import AxiomLogger

logger = AxiomLogger("my_new_workflow")

class MyMAWorkflow:
    def __init__(self):
        self.financial_aggregator = get_financial_aggregator()
        logger.info("Initialized MyMAWorkflow with financial aggregator")
    
    async def analyze(self, company: str):
        # Use aggregator for financial data
        fundamentals = await self.financial_aggregator.get_company_fundamentals(
            company_identifier=company,
            use_consensus=True
        )
        
        logger.info("Retrieved financial data", 
                   company=company,
                   provider=fundamentals.provider,
                   confidence=fundamentals.confidence)
        
        # Continue with analysis...
```

## API Reference

### FinancialDataAggregator

#### Methods

##### `get_company_fundamentals(company_identifier, metrics, use_consensus, **kwargs)`

Get company fundamental data with optional multi-provider consensus.

**Parameters:**
- `company_identifier` (str): Stock ticker or company name
- `metrics` (List[str], optional): Specific metrics to retrieve
- `use_consensus` (bool): Whether to aggregate from multiple providers
- `**kwargs`: Provider-specific parameters

**Returns:** [`FinancialDataResponse`](../axiom/integrations/data_sources/finance/base_financial_provider.py:14)

##### `get_comparable_companies(target_company, industry_sector, size_criteria, use_consensus, **kwargs)`

Get comparable companies with multi-source aggregation.

**Parameters:**
- `target_company` (str): Target company identifier
- `industry_sector` (str, optional): Industry filter
- `size_criteria` (Dict, optional): Size filtering criteria
- `use_consensus` (bool): Whether to aggregate from multiple sources

**Returns:** [`FinancialDataResponse`](../axiom/integrations/data_sources/finance/base_financial_provider.py:14) with aggregated comparables

##### `get_market_data(symbols, data_fields, **kwargs)`

Get real-time market data with provider fallback.

**Parameters:**
- `symbols` (List[str]): List of stock symbols
- `data_fields` (List[str], optional): Specific fields to retrieve

**Returns:** [`FinancialDataResponse`](../axiom/integrations/data_sources/finance/base_financial_provider.py:14)

##### `get_available_providers()`

Get list of initialized providers.

**Returns:** List[str] - Provider names

##### `get_provider_info()`

Get detailed information about all providers.

**Returns:** Dict[str, Dict] - Provider information and capabilities

##### `health_check()`

Check health status of all providers.

**Returns:** Dict[str, bool] - Provider health status

## Advanced Usage

### Custom Provider Configuration

```python
from axiom.integrations.data_sources.finance.financial_data_aggregator import FinancialDataAggregator

# Custom configuration
config = {
    "timeout": 60,  # Custom timeout
    "retry_count": 3,  # Retry failed calls
}

aggregator = FinancialDataAggregator(config=config)
```

### Provider-Specific Queries

```python
# Access specific provider directly
aggregator = get_financial_aggregator()

# Get FMP provider instance
fmp_provider = aggregator.providers.get("fmp")

if fmp_provider:
    # Use FMP-specific features
    dcf = fmp_provider.get_dcf_valuation("AAPL")
    transcripts = fmp_provider.get_earnings_transcripts("AAPL", year=2024, quarter=1)
```

## Support and Resources

- **Documentation**: See [`/docs`](../docs/) for full system documentation
- **Examples**: See [`/demos`](../demos/) for working examples
- **Issues**: Report integration issues via GitHub
- **API Keys**: Get free API keys:
  - Alpha Vantage: https://www.alphavantage.co/support/#api-key
  - FMP: https://financialmodelingprep.com/developer
  - Finnhub: https://finnhub.io/register

## Future Enhancements

Planned improvements for financial provider integration:

1. **Additional Providers**: Bloomberg Terminal, FactSet integration
2. **Caching Layer**: Redis caching for frequently accessed data
3. **Smart Routing**: ML-based provider selection optimization
4. **Cost Tracking**: Detailed API cost monitoring and optimization
5. **Data Validation**: Cross-provider data validation and anomaly detection

## Related Documentation

- [M&A Workflow Architecture](../axiom/workflows/MA_WORKFLOW_ARCHITECTURE.md)
- [Financial MCP Servers Guide](../guides/FINANCIAL_MCP_SERVERS_GUIDE.md)
- [System Setup Guide](./SETUP_GUIDE.md)
- [M&A System Overview](./ma-workflows/M&A_SYSTEM_OVERVIEW.md)

---

Last Updated: 2025-01-22
Version: 1.0.0