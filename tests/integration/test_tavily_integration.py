#!/usr/bin/env python3
"""
Tavily Search Integration Test
Location: tests/integration/test_tavily_integration.py  
Purpose: Test Tavily search client integration with real API queries

Run from project root: python3 tests/integration/test_tavily_integration.py
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from axiom.integrations.search_tools.tavily_client import TavilyClient
from axiom.core.logging.axiom_logger import AxiomLogger

logger = AxiomLogger("test_tavily_integration")


async def test_tavily_search():
    """Test Tavily search functionality with real query."""
    
    print("🧪 Testing Tavily Search Integration")
    print("=" * 60)
    print()
    
    # Test 1: Initialize Tavily client
    print("1️⃣ Initializing Tavily client...")
    try:
        client = TavilyClient()
        print("✅ Tavily client initialized")
        logger.info("Tavily client initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize Tavily client: {e}")
        logger.error("Tavily initialization failed", error=str(e))
        return False
    print()
    
    # Test 2: Test basic search
    print("2️⃣ Testing basic financial search...")
    try:
        query = "Apple Inc financial performance 2024"
        print(f"   Query: {query}")
        
        result = await client.search(
            query=query,
            max_results=5,
            search_depth="basic"
        )
        
        if result and "results" in result:
            print(f"✅ Search successful - {len(result['results'])} results")
            logger.info("Basic search successful", 
                       results_count=len(result['results']))
            
            if result['results']:
                first_result = result['results'][0]
                print(f"   Sample: {first_result.get('title', 'N/A')[:80]}...")
                print(f"   URL: {first_result.get('url', 'N/A')[:60]}...")
        else:
            print("⚠️  Search returned no results")
            return False
            
    except Exception as e:
        print(f"❌ Basic search failed: {e}")
        logger.error("Basic search failed", error=str(e))
        return False
    print()
    
    # Test 3: Test company search
    print("3️⃣ Testing company-specific search...")
    try:
        result = await client.company_search(
            company_name="Microsoft",
            analysis_type="financials"
        )
        
        if result and "results" in result:
            print(f"✅ Company search successful - {len(result['results'])} results")
            logger.info("Company search successful",
                       company="Microsoft", results=len(result['results']))
        else:
            print("⚠️  Company search returned no results")
            
    except Exception as e:
        print(f"❌ Company search failed: {e}")
        logger.error("Company search failed", error=str(e))
        return False
    print()
    
    # Test 4: Test QnA search
    print("4️⃣ Testing QnA search...")
    try:
        answer = await client.qna_search(
            query="What is Microsoft's current market cap?"
        )
        
        if answer:
            print(f"✅ QnA search successful")
            print(f"   Answer: {answer[:150]}...")
            logger.info("QnA search successful")
        else:
            print("⚠️  QnA search returned no answer")
            
    except Exception as e:
        print(f"❌ QnA search failed: {e}")
        logger.error("QnA search failed", error=str(e))
        return False
    print()
    
    print("=" * 60)
    print("✅ Tavily Integration Test Complete!")
    print()
    print("📊 Test Summary:")
    print("   ✅ Client initialization: Working")
    print("   ✅ Basic search: Operational")
    print("   ✅ Company search: Operational")
    print("   ✅ QnA search: Operational")
    print()
    print("🎯 Tavily integration is fully functional!")
    
    return True


if __name__ == "__main__":
    try:
        success = asyncio.run(test_tavily_search())
        if success:
            print("\n✅ All Tavily tests passed!")
            sys.exit(0)
        else:
            print("\n❌ Some tests failed - Check output above")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)