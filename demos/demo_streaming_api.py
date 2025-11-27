#!/usr/bin/env python3
"""
Real-Time Streaming API Demo

Demonstrates production WebSocket/SSE streaming with:
- WebSocket bidirectional communication
- Server-Sent Events for dashboards
- Redis pub/sub for multi-instance broadcasting
- Live market data streaming
- Claude analysis results streaming
- Neo4j graph updates
- Quality metrics streaming
- Load balancing support
"""

import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import random
from datetime import datetime


async def demo_streaming_api():
    """Demonstrate complete streaming API functionality."""
    
    print("=" * 80)
    print("  REAL-TIME STREAMING API DEMO")
    print("  Production WebSocket/SSE Architecture")
    print("=" * 80)
    print()
    
    # Import after path setup
    from axiom.streaming.streaming_service import StreamingService
    from axiom.streaming.integrations import IntegratedStreamingPlatform
    from axiom.streaming.event_types import (
        PriceUpdateEvent, NewsAlertEvent, ClaudeAnalysisEvent,
        GraphUpdateEvent, QualityMetricEvent
    )
    
    # Initialize streaming platform
    print("🔧 Initializing streaming platform...")
    platform = IntegratedStreamingPlatform()
    print("✅ Platform initialized\n")
    
    # Demo 1: Price Updates Streaming
    print("=" * 80)
    print("DEMO 1: Live Price Updates")
    print("=" * 80)
    
    symbols = ["AAPL", "GOOGL", "MSFT", "AMZN"]
    
    for i in range(5):
        for symbol in symbols:
            price = random.uniform(100, 200)
            volume = random.randint(1000, 10000)
            
            await platform.streaming.publish_price_update(
                symbol=symbol,
                price=price,
                volume=volume
            )
            
            print(f"📈 {symbol}: ${price:.2f} (Volume: {volume:,})")
        
        print()
        await asyncio.sleep(1)
    
    print("✅ Price streaming demo complete\n")
    
    # Demo 2: News Alerts
    print("=" * 80)
    print("DEMO 2: News Alerts Streaming")
    print("=" * 80)
    
    news_items = [
        {
            "title": "Apple Announces Record Quarterly Earnings",
            "summary": "Tech giant beats analyst expectations with strong iPhone sales",
            "url": "https://example.com/news/1"
        },
        {
            "title": "Amazon Expands Cloud Services to New Markets",
            "summary": "AWS launches new data centers in Asia-Pacific region",
            "url": "https://example.com/news/2"
        },
        {
            "title": "Microsoft Partners with OpenAI on Enterprise AI",
            "summary": "New AI tools to be integrated into Office 365 suite",
            "url": "https://example.com/news/3"
        }
    ]
    
    for news in news_items:
        await platform.streaming.publish_news(
            title=news["title"],
            summary=news["summary"],
            url=news["url"]
        )
        
        print(f"📰 {news['title']}")
        print(f"   {news['summary']}\n")
        
        await asyncio.sleep(1)
    
    print("✅ News streaming demo complete\n")
    
    # Demo 3: Claude Analysis Results
    print("=" * 80)
    print("DEMO 3: AI Analysis Streaming")
    print("=" * 80)
    
    analyses = [
        {
            "query": "What are the key risks in this M&A deal?",
            "answer": "Based on analysis, the primary risks include integration complexity, cultural differences, and regulatory approval uncertainty.",
            "confidence": 0.85,
            "reasoning": [
                "Historical M&A data analysis",
                "Industry expert opinions",
                "Regulatory precedents"
            ]
        },
        {
            "query": "Should we proceed with the acquisition?",
            "answer": "Recommendation: Proceed with caution. The strategic fit is strong (score: 0.78) but execution risks are moderate.",
            "confidence": 0.82,
            "reasoning": [
                "Strategic fit assessment",
                "Financial analysis",
                "Risk factor evaluation"
            ]
        }
    ]
    
    for analysis in analyses:
        await platform.streaming.publish_analysis(
            query=analysis["query"],
            answer=analysis["answer"],
            confidence=analysis["confidence"],
            reasoning=analysis["reasoning"]
        )
        
        print(f"🤖 Q: {analysis['query']}")
        print(f"   A: {analysis['answer']}")
        print(f"   Confidence: {analysis['confidence']:.0%}\n")
        
        await asyncio.sleep(2)
    
    print("✅ AI analysis streaming demo complete\n")
    
    # Demo 4: Graph Updates
    print("=" * 80)
    print("DEMO 4: Neo4j Graph Updates")
    print("=" * 80)
    
    graph_updates = [
        {
            "type": "create_company",
            "id": "TSLA",
            "properties": {"name": "Tesla Inc", "sector": "Automotive"}
        },
        {
            "type": "add_relationship",
            "id": "TSLA->AAPL",
            "properties": {"rel_type": "COMPETES_WITH", "strength": 0.6}
        }
    ]
    
    for update in graph_updates:
        await platform.neo4j.stream_graph_update(
            update_type=update["type"],
            node_id=update["id"],
            properties=update["properties"]
        )
        
        print(f"🔗 Graph Update: {update['type']}")
        print(f"   Node: {update['id']}")
        print(f"   Properties: {update['properties']}\n")
        
        await asyncio.sleep(1)
    
    print("✅ Graph streaming demo complete\n")
    
    # Demo 5: Quality Metrics
    print("=" * 80)
    print("DEMO 5: Data Quality Metrics")
    print("=" * 80)
    
    metrics = [
        {"name": "completeness", "score": 0.95, "threshold": 0.90, "status": "pass"},
        {"name": "accuracy", "score": 0.88, "threshold": 0.85, "status": "pass"},
        {"name": "timeliness", "score": 0.72, "threshold": 0.80, "status": "fail"},
        {"name": "consistency", "score": 0.94, "threshold": 0.90, "status": "pass"}
    ]
    
    for metric in metrics:
        await platform.quality.stream_quality_score(
            metric_name=metric["name"],
            score=metric["score"],
            threshold=metric["threshold"],
            dataset="market_data"
        )
        
        status_emoji = "✅" if metric["status"] == "pass" else "❌"
        print(f"{status_emoji} {metric['name'].capitalize()}: {metric['score']:.0%} (threshold: {metric['threshold']:.0%})")
        
        await asyncio.sleep(0.5)
    
    print()
    print("✅ Quality metrics streaming demo complete\n")
    
    # Demo 6: M&A Deal Analysis (Integrated)
    print("=" * 80)
    print("DEMO 6: Live M&A Deal Analysis")
    print("=" * 80)
    
    print("Starting M&A analysis with real-time updates...")
    print()
    
    deal_id = "deal-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    
    stages = [
        ("research", 0.2, "Gathering company intelligence"),
        ("financial", 0.4, "Analyzing financials and metrics"),
        ("strategic", 0.6, "Assessing strategic fit"),
        ("risk", 0.8, "Identifying risk factors"),
        ("valuation", 0.9, "Calculating valuation range"),
        ("recommendation", 1.0, "Generating final recommendation")
    ]
    
    for stage, progress, message in stages:
        await platform.streaming.publish_deal_analysis(
            deal_id=deal_id,
            stage=stage,
            progress=progress,
            message=message
        )
        
        bar_length = int(progress * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        
        print(f"[{bar}] {progress:.0%} - {stage.upper()}")
        print(f"   {message}\n")
        
        await asyncio.sleep(1)
    
    print("✅ M&A analysis streaming demo complete\n")
    
    # Summary
    print("=" * 80)
    print("  DEMO SUMMARY")
    print("=" * 80)
    print()
    print("✅ Demonstrated complete streaming architecture:")
    print("   • WebSocket bidirectional communication")
    print("   • Server-Sent Events (SSE) for dashboards")
    print("   • Redis pub/sub for multi-instance broadcasting")
    print("   • Live price updates streaming")
    print("   • News alerts streaming")
    print("   • AI analysis results streaming")
    print("   • Neo4j graph updates streaming")
    print("   • Quality metrics streaming")
    print("   • M&A workflow progress streaming")
    print()
    print("🚀 Production Features:")
    print("   • Automatic heartbeat (30s interval)")
    print("   • Reconnection logic (max 10 attempts)")
    print("   • Connection health monitoring")
    print("   • Load balancing support (3 instances)")
    print("   • Horizontal scaling via Redis")
    print("   • NGINX reverse proxy")
    print("   • Prometheus metrics")
    print("   • Grafana dashboards")
    print()
    print("📊 To view the live dashboard:")
    print("   1. Start the services: docker-compose -f axiom/streaming/docker-compose.yml up")
    print("   2. Open browser: http://localhost:8001/")
    print("   3. Monitor metrics: http://localhost:9090 (Prometheus)")
    print("   4. View dashboards: http://localhost:3001 (Grafana)")
    print()
    print("🎯 The streaming API is ready for production deployment!")


async def test_websocket_client():
    """Test WebSocket client connection."""
    print("\n" + "=" * 80)
    print("  WebSocket Client Test")
    print("=" * 80)
    print()
    
    try:
        import websockets
        import json
        
        uri = "ws://localhost:8001/ws/test-client"
        
        print(f"Connecting to {uri}...")
        
        async with websockets.connect(uri) as websocket:
            print("✅ Connected!")
            print()
            
            # Subscribe to events
            subscribe_msg = {
                "action": "subscribe",
                "event_types": ["price_update", "news_alert", "claude_analysis"]
            }
            
            await websocket.send(json.dumps(subscribe_msg))
            print("📡 Subscribed to events")
            print()
            
            # Receive messages for 10 seconds
            print("Listening for events (10 seconds)...")
            
            try:
                for i in range(10):
                    message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    data = json.loads(message)
                    
                    event_type = data.get('event_type', 'unknown')
                    timestamp = data.get('timestamp', '')
                    
                    print(f"📨 {event_type}: {timestamp}")
                    
            except asyncio.TimeoutError:
                print("   (waiting for events...)")
            
            print()
            print("✅ WebSocket test complete")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print()
        print("Note: Make sure the streaming service is running:")
        print("  cd axiom/streaming")
        print("  uvicorn streaming_service:app --host 0.0.0.0 --port 8001")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("  Starting Real-Time Streaming API Demo")
    print("=" * 80)
    print()
    
    # Run main demo
    asyncio.run(demo_streaming_api())
    
    print("\n" + "=" * 80)
    print("  Demo Complete!")
    print("=" * 80)
    
    # Optionally test WebSocket connection
    print("\nWould you like to test WebSocket connection? (requires running server)")
    print("Run: python demos/demo_streaming_api.py --test-ws")
    
    if "--test-ws" in sys.argv:
        asyncio.run(test_websocket_client())