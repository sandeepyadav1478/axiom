"""
Demo: Financial Provider Integration for M&A Analytics

This demo shows how Tavily, FMP, Finnhub, and Alpha Vantage providers
are integrated into Axiom M&A workflows for enhanced financial analysis.
"""

import asyncio
import os
from datetime import datetime

from axiom.integrations.data_sources.finance.financial_data_aggregator import get_financial_aggregator
from axiom.core.analysis_engines.valuation import MAValuationWorkflow
from axiom.core.analysis_engines.due_diligence import MADueDiligenceWorkflow
from axiom.core.analysis_engines.target_screening import MATargetScreeningWorkflow, TargetCriteria
from axiom.core.logging.axiom_logger import AxiomLogger

logger = AxiomLogger("financial_provider_demo")


async def demo_financial_aggregator():
    """Demonstrate the financial data aggregator with multiple providers."""
    
    print("\n" + "="*80)
    print("🏦 FINANCIAL DATA AGGREGATOR DEMO")
    print("="*80 + "\n")
    
    # Initialize aggregator
    aggregator = get_financial_aggregator()
    
    # Show available providers
    available_providers = aggregator.get_available_providers()
    print(f"✅ Available Providers: {', '.join(available_providers)}\n")
    
    # Show provider info
    provider_info = aggregator.get_provider_info()
    for name, info in provider_info.items():
        print(f"📊 {name.upper()} Provider:")
        print(f"   - Available: {info.get('available')}")
        print(f"   - Subscription: {info.get('subscription_level')}")
        capabilities = info.get('capabilities', {})
        key_capabilities = [k for k, v in capabilities.items() if v and k in [
            'fundamental_analysis', 'real_time_market_data', 'free_tier', 
            'affordable_premium', 'global_coverage'
        ]]
        print(f"   - Key Capabilities: {', '.join(key_capabilities)}")
        print()
    
    # Health check
    print("🏥 Running provider health checks...")
    health_status = await aggregator.health_check()
    for provider, is_healthy in health_status.items():
        status = "✅ Healthy" if is_healthy else "❌ Unavailable"
        print(f"   {provider}: {status}")
    print()
    
    return aggregator


async def demo_company_fundamentals(aggregator, symbol="MSFT"):
    """Demonstrate getting company fundamentals with consensus from multiple providers."""
    
    print("\n" + "="*80)
    print(f"📈 COMPANY FUNDAMENTALS DEMO - {symbol}")
    print("="*80 + "\n")
    
    try:
        # Get fundamentals with consensus from multiple providers
        print(f"Fetching fundamentals for {symbol} (using consensus from multiple providers)...")
        
        fundamentals = await aggregator.get_company_fundamentals(
            company_identifier=symbol,
            use_consensus=True
        )
        
        if fundamentals:
            print(f"\n✅ Retrieved from: {fundamentals.provider}")
            print(f"   Confidence Score: {fundamentals.confidence:.2f}")
            print(f"   Data Type: {fundamentals.data_type}")
            
            payload = fundamentals.data_payload
            
            # Display key metrics
            print(f"\n📊 Key Metrics:")
            if payload.get("company_name"):
                print(f"   Company: {payload.get('company_name')}")
            if payload.get("annual_revenue"):
                print(f"   Revenue: ${payload.get('annual_revenue'):,.0f}")
            if payload.get("market_cap"):
                print(f"   Market Cap: ${payload.get('market_cap'):,.0f}")
            if payload.get('ebitda_margin'):
                print(f"   EBITDA Margin: {payload.get('ebitda_margin'):.1%}")
            
            # Show consensus info if available
            consensus_data = payload.get("consensus_data", {})
            if consensus_data:
                print(f"\n🤝 Consensus Information:")
                print(f"   Providers Used: {', '.join(consensus_data.get('providers_used', []))}")
                print(f"   Provider Count: {consensus_data.get('provider_count')}")
                
            print("\n" + "-"*80)
    
    except Exception as e:
        logger.error(f"Failed to get fundamentals for {symbol}", error=str(e))
        print(f"❌ Error: {str(e)}")


async def demo_comparable_companies(aggregator, target="PLTR"):
    """Demonstrate finding comparable companies with multi-provider aggregation."""
    
    print("\n" + "="*80)
    print(f"🏢 COMPARABLE COMPANIES DEMO - {target}")
    print("="*80 + "\n")
    
    try:
        print(f"Finding comparable companies for {target}...")
        
        comparables = await aggregator.get_comparable_companies(
            target_company=target,
            use_consensus=True
        )
        
        if comparables:
            print(f"\n✅ Retrieved from: {comparables.provider}")
            print(f"   Confidence Score: {comparables.confidence:.2f}")
            
            payload = comparables.data_payload
            comp_list = payload.get("comparables", [])
            
            print(f"\n🏢 Found {len(comp_list)} Comparable Companies:")
            for i, comp in enumerate(comp_list[:5], 1):
                print(f"\n   {i}. {comp.get('name', comp.get('company_name', 'Unknown'))}")
                if comp.get('symbol'):
                    print(f"      Symbol: {comp.get('symbol')}")
                if comp.get('market_cap'):
                    print(f"      Market Cap: ${comp.get('market_cap'):,.0f}")
                if comp.get('similarity_score'):
                    print(f"      Similarity: {comp.get('similarity_score'):.2f}")
                if comp.get('source_provider'):
                    print(f"      Source: {comp.get('source_provider')}")
            
            if len(comp_list) > 5:
                print(f"\n   ... and {len(comp_list) - 5} more comparables")
            
            print("\n" + "-"*80)
    
    except Exception as e:
        logger.error(f"Failed to get comparables for {target}", error=str(e))
        print(f"❌ Error: {str(e)}")


async def demo_market_data(aggregator, symbols=["MSFT", "GOOGL", "AMZN"]):
    """Demonstrate getting real-time market data."""
    
    print("\n" + "="*80)
    print(f"💹 MARKET DATA DEMO - {', '.join(symbols)}")
    print("="*80 + "\n")
    
    try:
        print(f"Fetching real-time market data for {len(symbols)} symbols...")
        
        market_data = await aggregator.get_market_data(
            symbols=symbols
        )
        
        if market_data:
            print(f"\n✅ Retrieved from: {market_data.provider}")
            print(f"   Confidence Score: {market_data.confidence:.2f}")
            
            payload = market_data.data_payload
            data_map = payload.get("market_data", {})
            
            print(f"\n📊 Market Data for {payload.get('symbols_retrieved')} symbols:")
            for symbol, data in data_map.items():
                print(f"\n   {symbol}:")
                if data.get('current_price'):
                    print(f"      Price: ${data.get('current_price'):.2f}")
                if data.get('change_percent'):
                    change = data.get('change_percent')
                    arrow = "📈" if change > 0 else "📉"
                    print(f"      Change: {arrow} {change:.2f}%")
                if data.get('volume'):
                    print(f"      Volume: {data.get('volume'):,.0f}")
                if data.get('market_cap'):
                    print(f"      Market Cap: ${data.get('market_cap'):,.0f}")
            
            print("\n" + "-"*80)
    
    except Exception as e:
        logger.error(f"Failed to get market data", error=str(e))
        print(f"❌ Error: {str(e)}")


async def demo_valuation_workflow(target="Palantir"):
    """Demonstrate valuation workflow with integrated financial providers."""
    
    print("\n" + "="*80)
    print(f"💰 VALUATION WORKFLOW DEMO - {target}")
    print("="*80 + "\n")
    
    try:
        workflow = MAValuationWorkflow()
        
        print(f"Running comprehensive valuation for {target}...")
        print("This will use integrated financial providers for:")
        print("  - Historical financial data and projections")
        print("  - Comparable company identification")
        print("  - Trading multiples calculation")
        print()
        
        # Create sample target metrics
        target_metrics = {
            "revenue": 2_200_000_000,  # $2.2B
            "ebitda": 440_000_000,      # $440M
            "employees": 4_500,
            "geography": "Global (US, EU, Asia)"
        }
        
        valuation = await workflow.execute_comprehensive_valuation(
            target_company=target,
            target_metrics=target_metrics
        )
        
        print(f"\n✅ Valuation Analysis Completed in {valuation.analysis_duration:.1f}s")
        print(f"\n📊 Valuation Results:")
        print(f"   Base Case: ${valuation.valuation_base/1e9:.2f}B" if valuation.valuation_base else "   Base Case: Not available")
        print(f"   Range: ${valuation.valuation_low/1e9:.2f}B - ${valuation.valuation_high/1e9:.2f}B" if valuation.valuation_low else "   Range: Not available")
        print(f"   Confidence: {valuation.valuation_confidence:.2f}")
        
        print(f"\n🎯 Deal Structure:")
        print(f"   Cash: {valuation.cash_percentage:.0%}")
        print(f"   Stock: {valuation.stock_percentage:.0%}")
        if valuation.earnout_amount > 0:
            print(f"   Earnout: ${valuation.earnout_amount/1e6:.1f}M")
        
        print(f"\n📈 Comparable Analysis:")
        print(f"   Companies: {valuation.comparable_analysis.comp_count}")
        print(f"   Quality Score: {valuation.comparable_analysis.comparability_score:.2f}")
        
        print("\n" + "-"*80)
        
    except Exception as e:
        logger.error(f"Valuation workflow failed for {target}", error=str(e))
        print(f"❌ Error: {str(e)}")


async def demo_due_diligence_workflow(target="Snowflake"):
    """Demonstrate due diligence workflow with integrated financial providers."""
    
    print("\n" + "="*80)
    print(f"🔍 DUE DILIGENCE WORKFLOW DEMO - {target}")
    print("="*80 + "\n")
    
    try:
        workflow = MADueDiligenceWorkflow()
        
        print(f"Running comprehensive due diligence for {target}...")
        print("This will use integrated financial providers for:")
        print("  - Financial statement analysis")
        print("  - Profitability and margin analysis")
        print("  - Balance sheet and liquidity assessment")
        print()
        
        dd_result = await workflow.execute_comprehensive_dd(
            target_company=target,
            analysis_scope="full"
        )
        
        print(f"\n✅ Due Diligence Completed in {dd_result.total_analysis_time:.1f}s")
        print(f"\n📊 Overall Assessment:")
        print(f"   Risk Rating: {dd_result.overall_risk_rating.upper()}")
        print(f"   Recommendation: {dd_result.investment_recommendation.upper()}")
        print(f"   Confidence: {dd_result.overall_confidence:.2f}")
        
        print(f"\n💰 Financial DD:")
        print(f"   Revenue Quality: {dd_result.financial_dd.revenue_quality_score:.2f}")
        print(f"   EBITDA Quality: {dd_result.financial_dd.ebitda_quality_score:.2f}")
        print(f"   Balance Sheet: {dd_result.financial_dd.balance_sheet_strength:.2f}")
        print(f"   Evidence Sources: {len(dd_result.financial_dd.evidence)}")
        
        print(f"\n🏢 Commercial DD:")
        print(f"   Market Position: {dd_result.commercial_dd.market_position_strength:.2f}")
        print(f"   Customer Diversification: {dd_result.commercial_dd.customer_diversification:.2f}")
        
        print("\n" + "-"*80)
        
    except Exception as e:
        logger.error(f"Due diligence workflow failed for {target}", error=str(e))
        print(f"❌ Error: {str(e)}")


async def demo_target_screening_workflow():
    """Demonstrate target screening workflow with integrated financial providers."""
    
    print("\n" + "="*80)
    print("🎯 TARGET SCREENING WORKFLOW DEMO")
    print("="*80 + "\n")
    
    try:
        workflow = MATargetScreeningWorkflow()
        
        # Define screening criteria
        criteria = TargetCriteria(
            industry_sectors=["Enterprise Software", "Cloud Computing"],
            geographic_regions=["US", "EU"],
            strategic_rationale="Expand AI/ML capabilities and customer base",
            min_revenue=100_000_000,        # $100M minimum
            max_revenue=5_000_000_000,      # $5B maximum
            min_ebitda_margin=0.15,         # 15% minimum margin
            min_growth_rate=0.10,           # 10% minimum growth
            max_valuation=3_000_000_000     # $3B maximum
        )
        
        print("Screening Criteria:")
        print(f"  Industries: {', '.join(criteria.industry_sectors)}")
        print(f"  Regions: {', '.join(criteria.geographic_regions)}")
        print(f"  Revenue Range: ${criteria.min_revenue/1e6:.0f}M - ${criteria.max_revenue/1e9:.1f}B")
        print(f"  Min EBITDA Margin: {criteria.min_ebitda_margin:.0%}")
        print(f"  Min Growth: {criteria.min_growth_rate:.0%}")
        print()
        
        print("Running target screening with financial provider integration...")
        print("This will use integrated financial providers for:")
        print("  - Target financial data enrichment")
        print("  - Real-time market metrics")
        print("  - Financial screening validation")
        print()
        
        screening_result = await workflow.execute(criteria)
        
        print(f"\n✅ Screening Completed in {screening_result.execution_time:.1f}s")
        print(f"\n📊 Screening Results:")
        print(f"   Targets Screened: {screening_result.targets_screened}")
        print(f"   Qualified Targets: {screening_result.targets_qualified}")
        print(f"   Overall Confidence: {screening_result.confidence_level:.2f}")
        
        if screening_result.targets_identified:
            print(f"\n🎯 Top 3 Qualified Targets:")
            for i, target in enumerate(screening_result.targets_identified[:3], 1):
                print(f"\n   {i}. {target.company_name}")
                print(f"      Industry: {target.industry}")
                print(f"      Strategic Fit: {target.strategic_fit_score:.2f}")
                print(f"      Financial Score: {target.financial_attractiveness:.2f}")
                print(f"      Acquisition Probability: {target.acquisition_probability:.2f}")
                if target.annual_revenue:
                    print(f"      Revenue: ${target.annual_revenue:,.0f}")
                if target.data_sources:
                    print(f"      Data Sources: {', '.join(target.data_sources)}")
        
        print("\n" + "-"*80)
        
    except Exception as e:
        logger.error("Target screening workflow failed", error=str(e))
        print(f"❌ Error: {str(e)}")


async def main():
    """Run all financial provider integration demos."""
    
    print("\n" + "="*80)
    print("🚀 AXIOM FINANCIAL PROVIDER INTEGRATION DEMO")
    print("="*80)
    print(f"\nTimestamp: {datetime.now().isoformat()}")
    print(f"Demonstrating integration of Tavily, FMP, Finnhub, and Alpha Vantage")
    print("\n" + "="*80)
    
    # Check environment
    print("\n🔧 Environment Check:")
    providers_configured = []
    if os.getenv("TAVILY_API_KEY"):
        providers_configured.append("Tavily")
    if os.getenv("FMP_API_KEY") or os.getenv("FINANCIAL_MODELING_PREP_API_KEY"):
        providers_configured.append("FMP")
    if os.getenv("FINNHUB_API_KEY"):
        providers_configured.append("Finnhub")
    if os.getenv("ALPHA_VANTAGE_API_KEY"):
        providers_configured.append("Alpha Vantage")
    
    print(f"   Configured Providers: {', '.join(providers_configured) if providers_configured else 'None'}")
    
    if not providers_configured:
        print("\n⚠️  WARNING: No financial providers configured!")
        print("   Please add API keys to .env file to test provider integration.")
        print("   See .env.example for configuration details.")
        return
    
    # Demo 1: Financial Aggregator
    aggregator = await demo_financial_aggregator()
    
    # Demo 2: Company Fundamentals (if providers available)
    if aggregator.get_available_providers():
        await demo_company_fundamentals(aggregator, "MSFT")
        await demo_comparable_companies(aggregator, "PLTR")
        await demo_market_data(aggregator, ["MSFT", "PLTR", "SNOW"])
    
    # Demo 3: M&A Workflows with Financial Integration
    await demo_valuation_workflow("Palantir Technologies")
    await demo_due_diligence_workflow("Snowflake Inc")
    await demo_target_screening_workflow()
    
    print("\n" + "="*80)
    print("✅ DEMO COMPLETED SUCCESSFULLY")
    print("="*80)
    print("\nKey Integration Points Demonstrated:")
    print("  ✅ Financial Data Aggregator with multi-provider consensus")
    print("  ✅ Valuation workflow using provider fundamentals and comparables")
    print("  ✅ Due diligence workflow with comprehensive financial data")
    print("  ✅ Target screening with automated financial enrichment")
    print("  ✅ Fallback mechanisms when providers unavailable")
    print("  ✅ Logging integration for observability")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())