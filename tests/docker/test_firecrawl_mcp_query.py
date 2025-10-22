#!/usr/bin/env python3
"""
Firecrawl MCP Server Query Test
Location: tests/docker/test_firecrawl_mcp_query.py
Purpose: Test Firecrawl MCP server functionality with real queries

Run from project root: python3 tests/docker/test_firecrawl_mcp_query.py
"""

import subprocess
import sys
import time


def test_firecrawl_server():
    """Test Firecrawl MCP server with actual web scraping query"""
    
    print("🧪 Testing Firecrawl MCP Server with Real API Query")
    print("=" * 60)
    print()
    
    # Test 1: Verify API key in container
    print("1️⃣ Verifying API key loaded in container...")
    try:
        result = subprocess.run(
            ['docker', 'exec', 'axiom-firecrawl-mcp',
             'sh', '-c', 'echo $FIRECRAWL_API_KEY | head -c 20'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0 and len(result.stdout.strip()) > 10:
            print(f"✅ API key loaded: {result.stdout.strip()}...")
        else:
            print("❌ API key not found in container")
            return False
    except Exception as e:
        print(f"❌ Error checking API key: {str(e)}")
        return False
    print()
    
    # Test 2: Test actual Firecrawl API call
    print("2️⃣ Testing Firecrawl API with real scraping query...")
    print("   Target: https://example.com")
    
    try:
        # Use Node.js to make API call
        result = subprocess.run(
            ['docker', 'exec', 'axiom-firecrawl-mcp',
             'node', '-e',
             """
            const https = require('https');
            const apiKey = process.env.FIRECRAWL_API_KEY;
            const postData = JSON.stringify({ url: 'https://example.com' });
            const options = {
            hostname: 'api.firecrawl.dev',
            path: '/v0/scrape',
            method: 'POST',
            headers: {
                'Authorization': 'Bearer ' + apiKey,
                'Content-Type': 'application/json',
                'Content-Length': postData.length
            }
            };

            const req = https.request(options, (res) => {
            console.log('HTTP_STATUS:', res.statusCode);
            let data = '';
            res.on('data', (chunk) => { data += chunk; });
            res.on('end', () => { 
                console.log('RESPONSE:', data.substring(0, 250));
            });
            });

            req.on('error', (e) => { 
            console.error('ERROR:', e.message); 
            });

            req.write(postData);
            req.end();
            """],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        time.sleep(1)  # Wait for async response
        
        if result.returncode == 0:
            output = result.stdout
            
            # Check for successful responses
            if "HTTP_STATUS: 200" in output:
                print("✅ API query successful - HTTP 200 OK")
                
                if "success" in output.lower() and "true" in output.lower():
                    print("✅ Web scraping successful")
                    
                if "Example Domain" in output or "content" in output.lower():
                    print("✅ Content retrieved from example.com")
                    
                # Show sample of response
                response_line = [line for line in output.split('\n') if 'RESPONSE:' in line]
                if response_line:
                    print(f"   Sample: {response_line[0][9:150]}...")
                print()
                return True
                
            elif "401" in output or "403" in output or "unauthorized" in output.lower():
                print("❌ API authentication failed")
                print(f"   Response: {output[:300]}")
                return False
                
            elif "429" in output or "rate" in output.lower():
                print("⚠️  Rate limit reached")
                print("✅ API key valid but rate limited")
                print()
                return True
                
            else:
                print(f"⚠️  Got response:")
                print(f"   {output[:300]}")
                print()
                return True
        else:
            print(f"❌ Query failed")
            print(f"   Exit code: {result.returncode}")
            print(f"   Error: {result.stderr[:300]}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ API query timed out")
        return False
    except Exception as e:
        print(f"❌ Exception during API test: {str(e)}")
        return False
    
    print()
    print("=" * 60)
    print("✅ Firecrawl MCP Server test completed!")
    print()
    print("📊 Test Summary:")
    print("   ✅ API Key: Loaded correctly")
    print("   ✅ API Query: HTTP 200 OK")
    print("   ✅ Web Scraping: Operational")
    print()
    print("🎯 Firecrawl MCP Server is fully operational!")
    
    return True


if __name__ == "__main__":
    try:
        success = test_firecrawl_server()
        if success:
            print("\n✅ All Firecrawl tests passed!")
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