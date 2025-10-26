"""Week 3 MCP Server Integration Examples.

Demonstrates usage of all 5 advanced MCP servers:
- AWS MCP Server
- GCP MCP Server  
- Notification MCP Server
- Vector DB MCP Server
- Kubernetes MCP Server
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Optional

# Note: This is a demo/example file
# Actual imports would be from axiom.integrations.mcp_servers


async def demo_aws_operations():
    """Demonstrate AWS MCP server operations."""
    print("\n" + "="*60)
    print("AWS MCP Server Demo")
    print("="*60)
    
    from axiom.integrations.mcp_servers.manager import mcp_manager
    
    # 1. S3 Operations
    print("\n1. S3 Upload...")
    portfolio_data = {
        "date": "2024-01-24",
        "positions": [
            {"symbol": "AAPL", "quantity": 100, "value": 15000},
            {"symbol": "GOOGL", "quantity": 50, "value": 7500},
        ],
        "total_value": 22500,
    }
    
    result = await mcp_manager.call_tool(
        server_name="aws",
        tool_name="s3_upload",
        bucket="axiom-portfolios",
        key="snapshots/2024-01-24/portfolio.json",
        data=json.dumps(portfolio_data),
        metadata={"date": "2024-01-24", "strategy": "momentum"}
    )
    print(f"✓ Upload: {result.get('url')}")
    
    # 2. List S3 objects
    print("\n2. S3 List...")
    result = await mcp_manager.call_tool(
        server_name="aws",
        tool_name="s3_list",
        bucket="axiom-portfolios",
        prefix="snapshots/",
        max_keys=10
    )
    print(f"✓ Found {result.get('count')} objects")
    
    # 3. EC2 Operations
    print("\n3. EC2 List Instances...")
    result = await mcp_manager.call_tool(
        server_name="aws",
        tool_name="ec2_list_instances",
        filters=[{"Name": "instance-state-name", "Values": ["running"]}]
    )
    print(f"✓ Found {result.get('count')} running instances")
    
    # 4. Lambda Invocation
    print("\n4. Lambda Invoke...")
    result = await mcp_manager.call_tool(
        server_name="aws",
        tool_name="lambda_invoke",
        function_name="process-market-data",
        payload={"symbol": "AAPL", "action": "analyze"}
    )
    print(f"✓ Status: {result.get('status_code')}")
    
    # 5. CloudWatch Metrics
    print("\n5. CloudWatch Metrics...")
    result = await mcp_manager.call_tool(
        server_name="aws",
        tool_name="cloudwatch_get_metrics",
        namespace="AWS/Lambda",
        metric_name="Duration",
        statistic="Average",
        period=3600
    )
    print(f"✓ Retrieved {result.get('count')} datapoints")


async def demo_gcp_operations():
    """Demonstrate GCP MCP server operations."""
    print("\n" + "="*60)
    print("GCP MCP Server Demo")
    print("="*60)
    
    from axiom.integrations.mcp_servers.manager import mcp_manager
    
    # 1. Cloud Storage Upload
    print("\n1. Cloud Storage Upload...")
    result = await mcp_manager.call_tool(
        server_name="gcp",
        tool_name="storage_upload",
        bucket_name="axiom-data-eu",
        blob_name="backups/portfolio.json",
        data='{"data": "backup"}',
        content_type="application/json"
    )
    print(f"✓ Uploaded to {result.get('public_url')}")
    
    # 2. BigQuery Query
    print("\n2. BigQuery Analytics...")
    result = await mcp_manager.call_tool(
        server_name="gcp",
        tool_name="bigquery_query",
        query="""
            SELECT 
                symbol,
                AVG(price) as avg_price,
                COUNT(*) as trade_count
            FROM `axiom-prod.trading.executions`
            WHERE date = CURRENT_DATE()
            GROUP BY symbol
            LIMIT 10
        """
    )
    print(f"✓ Query returned {result.get('row_count')} rows")
    print(f"  Bytes processed: {result.get('bytes_processed')}")
    
    # 3. Compute Engine
    print("\n3. Compute Engine List...")
    result = await mcp_manager.call_tool(
        server_name="gcp",
        tool_name="compute_list_instances",
        zone="us-central1-a"
    )
    print(f"✓ Found {result.get('count')} instances")
    
    # 4. Cloud Functions
    print("\n4. Cloud Function Invoke...")
    result = await mcp_manager.call_tool(
        server_name="gcp",
        tool_name="function_invoke",
        function_name="process-trades",
        region="us-central1",
        data={"symbol": "TSLA", "quantity": 100}
    )
    print(f"✓ Function executed: {result.get('execution_id')}")


async def demo_notification_operations():
    """Demonstrate Notification MCP server operations."""
    print("\n" + "="*60)
    print("Notification MCP Server Demo")
    print("="*60)
    
    from axiom.integrations.mcp_servers.manager import mcp_manager
    
    # 1. Simple Email
    print("\n1. Send Email...")
    result = await mcp_manager.call_tool(
        server_name="notification",
        tool_name="send_email",
        to="trader@example.com",
        subject="Trade Execution Confirmation",
        body="Your order for 100 shares of AAPL has been executed at $150.25"
    )
    print(f"✓ Email sent: {result.get('success')}")
    
    # 2. HTML Email
    print("\n2. Send HTML Email...")
    html_content = """
    <html>
    <body>
        <h2>Trade Summary</h2>
        <table>
            <tr><td>Symbol:</td><td>AAPL</td></tr>
            <tr><td>Quantity:</td><td>100</td></tr>
            <tr><td>Price:</td><td>$150.25</td></tr>
        </table>
    </body>
    </html>
    """
    
    result = await mcp_manager.call_tool(
        server_name="notification",
        tool_name="send_html_email",
        to="trader@example.com",
        subject="Trade Confirmation",
        html_body=html_content
    )
    print(f"✓ HTML email sent: {result.get('success')}")
    
    # 3. Daily Report
    print("\n3. Send Daily Report...")
    result = await mcp_manager.call_tool(
        server_name="notification",
        tool_name="send_daily_report",
        to=["trader@example.com", "risk@example.com"],
        report_title="Daily P&L Summary",
        report_data={
            "total_pnl": "$12,450.75",
            "winning_trades": 45,
            "losing_trades": 12,
            "win_rate": "78.9%",
            "sharpe_ratio": "2.34",
            "max_drawdown": "-$2,100.00",
            "best_trade": "NVDA +$1,250",
            "worst_trade": "TSLA -$450"
        }
    )
    print(f"✓ Daily report sent: {result.get('success')}")
    
    # 4. Critical Alert
    print("\n4. Send Critical Alert...")
    result = await mcp_manager.call_tool(
        server_name="notification",
        tool_name="send_alert",
        recipients={
            "email": ["risk@example.com", "cto@example.com"],
            "phone": "+1234567890"
        },
        severity="critical",
        title="VaR Limit Breach",
        message="Portfolio VaR exceeded 2.0% limit, currently at 3.2%",
        metadata={
            "current_var": "3.2%",
            "limit": "2.0%",
            "portfolio_value": "$10,000,000",
            "action_required": "Reduce exposure immediately"
        }
    )
    print(f"✓ Alert sent via: {result.get('channels')}")


async def demo_vector_db_operations():
    """Demonstrate Vector DB MCP server operations."""
    print("\n" + "="*60)
    print("Vector DB MCP Server Demo")
    print("="*60)
    
    from axiom.integrations.mcp_servers.manager import mcp_manager
    
    # 1. Create Collection
    print("\n1. Create Collection...")
    result = await mcp_manager.call_tool(
        server_name="vector_db",
        tool_name="create_collection",
        collection="companies",
        dimension=1536
    )
    print(f"✓ Collection created: {result.get('collection')}")
    
    # 2. Add Documents
    print("\n2. Add Documents...")
    companies = [
        {
            "id": "AAPL",
            "text": "Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide. Known for innovation in consumer electronics and software.",
            "metadata": {"sector": "Technology", "market_cap": "3T"}
        },
        {
            "id": "MSFT",
            "text": "Microsoft Corporation develops, licenses, and supports software, services, devices, and solutions worldwide. Leader in cloud computing, productivity software, and AI.",
            "metadata": {"sector": "Technology", "market_cap": "2.8T"}
        },
        {
            "id": "NVDA",
            "text": "NVIDIA Corporation provides graphics and compute and networking solutions globally. Dominant in GPU computing for AI, gaming, and data centers.",
            "metadata": {"sector": "Technology", "market_cap": "1.2T"}
        },
    ]
    
    # Generate mock embeddings (in production, use OpenAI API)
    def mock_embedding(text: str) -> list[float]:
        """Generate mock embedding for demo."""
        import hashlib
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        return [(hash_val % 1000) / 1000.0] * 1536
    
    for company in companies:
        embedding = mock_embedding(company["text"])
        result = await mcp_manager.call_tool(
            server_name="vector_db",
            tool_name="add_document",
            collection="companies",
            document_id=company["id"],
            text=company["text"],
            embedding=embedding,
            metadata=company["metadata"]
        )
        print(f"✓ Added {company['id']}: {result.get('success')}")
    
    # 3. Semantic Search
    print("\n3. Semantic Search...")
    query = "GPU computing and AI infrastructure company"
    query_embedding = mock_embedding(query)
    
    result = await mcp_manager.call_tool(
        server_name="vector_db",
        tool_name="search_similar",
        collection="companies",
        query_embedding=query_embedding,
        limit=3
    )
    
    print(f"✓ Found {result.get('count')} similar companies:")
    for match in result.get("results", []):
        print(f"  - {match['id']}: score={match['score']:.3f}")
    
    # 4. Filter Search
    print("\n4. Filter Search...")
    result = await mcp_manager.call_tool(
        server_name="vector_db",
        tool_name="filter_search",
        collection="companies",
        query_embedding=query_embedding,
        filters={"sector": "Technology"},
        limit=5
    )
    print(f"✓ Filtered results: {result.get('count')}")


async def demo_kubernetes_operations():
    """Demonstrate Kubernetes MCP server operations."""
    print("\n" + "="*60)
    print("Kubernetes MCP Server Demo")
    print("="*60)
    
    from axiom.integrations.mcp_servers.manager import mcp_manager
    
    # 1. Create Deployment
    print("\n1. Create Deployment...")
    result = await mcp_manager.call_tool(
        server_name="kubernetes",
        tool_name="create_deployment",
        name="axiom-api",
        image="axiom/api:latest",
        replicas=3,
        namespace="staging",
        port=8000,
        env_vars={
            "ENV": "staging",
            "LOG_LEVEL": "INFO"
        },
        resources={
            "requests": {"cpu": "500m", "memory": "512Mi"},
            "limits": {"cpu": "1000m", "memory": "1Gi"}
        }
    )
    print(f"✓ Deployment created: {result.get('name')}")
    
    # 2. Expose as Service
    print("\n2. Expose Service...")
    result = await mcp_manager.call_tool(
        server_name="kubernetes",
        tool_name="expose_service",
        name="axiom-api-service",
        deployment_name="axiom-api",
        port=80,
        service_type="LoadBalancer",
        namespace="staging"
    )
    print(f"✓ Service exposed: {result.get('name')}")
    
    # 3. List Pods
    print("\n3. List Pods...")
    result = await mcp_manager.call_tool(
        server_name="kubernetes",
        tool_name="list_pods",
        namespace="staging",
        label_selector="app=axiom-api"
    )
    print(f"✓ Found {result.get('count')} pods:")
    for pod in result.get("pods", []):
        print(f"  - {pod['name']}: {pod['status']}")
    
    # 4. Get Resource Usage
    print("\n4. Get Resource Usage...")
    result = await mcp_manager.call_tool(
        server_name="kubernetes",
        tool_name="get_resource_usage",
        namespace="staging"
    )
    print(f"✓ CPU Requests: {result.get('cpu_requests')}")
    print(f"✓ Memory Requests: {result.get('memory_requests')}")
    
    # 5. Get Cluster Info
    print("\n5. Get Cluster Info...")
    result = await mcp_manager.call_tool(
        server_name="kubernetes",
        tool_name="get_cluster_info"
    )
    print(f"✓ Nodes: {result.get('node_count')}")
    print(f"✓ Namespaces: {result.get('namespace_count')}")


async def demo_complete_workflow():
    """Demonstrate complete end-to-end workflow using multiple MCP servers."""
    print("\n" + "="*60)
    print("Complete Workflow: Trading Alert System")
    print("="*60)
    
    from axiom.integrations.mcp_servers.manager import mcp_manager
    
    # Scenario: Stock price moved significantly
    symbol = "AAPL"
    price_change = 5.2  # 5.2% increase
    current_price = 150.25
    
    print(f"\n📊 Detected: {symbol} moved {price_change:+.1f}% to ${current_price}")
    
    # Step 1: Store alert data in S3
    print("\n1. Storing alert data in S3...")
    alert_data = {
        "symbol": symbol,
        "price_change": price_change,
        "current_price": current_price,
        "timestamp": datetime.utcnow().isoformat(),
        "alert_type": "price_movement"
    }
    
    s3_result = await mcp_manager.call_tool(
        server_name="aws",
        tool_name="s3_upload",
        bucket="axiom-alerts",
        key=f"alerts/{symbol}/{datetime.utcnow().date()}.json",
        data=json.dumps(alert_data)
    )
    print(f"✓ Stored in S3: {s3_result.get('url')}")
    
    # Step 2: Add to vector DB for pattern analysis
    print("\n2. Adding to vector DB for pattern analysis...")
    
    # Mock embedding generation (use OpenAI in production)
    def generate_mock_embedding(text: str) -> list[float]:
        import hashlib
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        return [(hash_val % 1000) / 1000.0] * 1536
    
    alert_text = f"{symbol} price changed by {price_change}% to ${current_price}"
    embedding = generate_mock_embedding(alert_text)
    
    vector_result = await mcp_manager.call_tool(
        server_name="vector_db",
        tool_name="add_document",
        collection="price_alerts",
        document_id=f"{symbol}_{int(datetime.utcnow().timestamp())}",
        text=alert_text,
        embedding=embedding,
        metadata={
            "symbol": symbol,
            "change_pct": price_change,
            "price": current_price,
            "date": datetime.utcnow().date().isoformat()
        }
    )
    print(f"✓ Added to vector DB: {vector_result.get('document_id')}")
    
    # Step 3: Find similar historical alerts
    print("\n3. Finding similar historical alerts...")
    similar_result = await mcp_manager.call_tool(
        server_name="vector_db",
        tool_name="search_similar",
        collection="price_alerts",
        query_embedding=embedding,
        limit=5,
        filter={"symbol": symbol}
    )
    
    print(f"✓ Found {similar_result.get('count')} similar alerts")
    for match in similar_result.get("results", [])[:3]:
        print(f"  - Score: {match['score']:.3f}, Date: {match['metadata'].get('date')}")
    
    # Step 4: Send multi-channel notification
    print("\n4. Sending alert notification...")
    severity = "critical" if abs(price_change) > 10 else "warning"
    
    notification_result = await mcp_manager.call_tool(
        server_name="notification",
        tool_name="send_alert",
        recipients={
            "email": "traders@example.com",
            "phone": "+1234567890" if severity == "critical" else None
        },
        severity=severity,
        title=f"Price Alert: {symbol}",
        message=f"{symbol} moved {price_change:+.1f}% to ${current_price}",
        metadata=alert_data
    )
    print(f"✓ Alert sent via: {notification_result.get('channels')}")
    
    # Step 5: Log to GCP BigQuery for analytics
    print("\n5. Logging to BigQuery...")
    bq_result = await mcp_manager.call_tool(
        server_name="gcp",
        tool_name="bigquery_load",
        dataset_id="trading",
        table_id="price_alerts",
        data=[{
            "symbol": symbol,
            "price_change": price_change,
            "current_price": current_price,
            "timestamp": alert_data["timestamp"],
            "severity": severity
        }]
    )
    print(f"✓ Logged to BigQuery: {bq_result.get('rows_loaded')} rows")
    
    # Step 6: Update Kubernetes deployment if needed (scale based on activity)
    print("\n6. Checking deployment scale...")
    pods_result = await mcp_manager.call_tool(
        server_name="kubernetes",
        tool_name="list_pods",
        namespace="production",
        label_selector="app=axiom-api"
    )
    
    active_pods = pods_result.get("count", 0)
    print(f"✓ Active pods: {active_pods}")
    
    # Auto-scale if high activity
    if active_pods < 5 and severity == "critical":
        scale_result = await mcp_manager.call_tool(
            server_name="kubernetes",
            tool_name="scale_deployment",
            name="axiom-api",
            replicas=5,
            namespace="production"
        )
        print(f"✓ Scaled to {scale_result.get('new_replicas')} replicas")
    
    print("\n✅ Workflow completed successfully!")


async def demo_vector_search_use_case():
    """Demonstrate practical vector search for M&A target identification."""
    print("\n" + "="*60)
    print("Vector Search: M&A Target Identification")
    print("="*60)
    
    from axiom.integrations.mcp_servers.manager import mcp_manager
    
    # Mock embedding function
    def generate_mock_embedding(text: str) -> list[float]:
        import hashlib
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        return [(hash_val % 1000) / 1000.0] * 1536
    
    # Step 1: Create M&A targets collection
    print("\n1. Creating M&A targets collection...")
    await mcp_manager.call_tool(
        server_name="vector_db",
        tool_name="create_collection",
        collection="ma_targets",
        dimension=1536
    )
    
    # Step 2: Add company profiles
    print("\n2. Adding company profiles...")
    companies = [
        {
            "ticker": "SMALL_AI_1",
            "description": "AI-powered financial analytics platform with 50 employees, $10M revenue, specializing in portfolio optimization",
            "metadata": {"employees": 50, "revenue": "10M", "sector": "FinTech"}
        },
        {
            "ticker": "SMALL_AI_2",
            "description": "Machine learning company focused on algorithmic trading, 30 employees, $5M revenue, strong IP portfolio",
            "metadata": {"employees": 30, "revenue": "5M", "sector": "FinTech"}
        },
        {
            "ticker": "SMALL_DATA_1",
            "description": "Real-time market data provider with proprietary data feeds, 100 employees, $25M revenue",
            "metadata": {"employees": 100, "revenue": "25M", "sector": "Data"}
        },
    ]
    
    for company in companies:
        embedding = generate_mock_embedding(company["description"])
        await mcp_manager.call_tool(
            server_name="vector_db",
            tool_name="add_document",
            collection="ma_targets",
            document_id=company["ticker"],
            text=company["description"],
            embedding=embedding,
            metadata=company["metadata"]
        )
        print(f"✓ Added {company['ticker']}")
    
    # Step 3: Search for acquisition targets
    print("\n3. Searching for acquisition targets...")
    search_criteria = "AI and machine learning company for algorithmic trading with strong technology"
    search_embedding = generate_mock_embedding(search_criteria)
    
    result = await mcp_manager.call_tool(
        server_name="vector_db",
        tool_name="filter_search",
        collection="ma_targets",
        query_embedding=search_embedding,
        filters={"sector": "FinTech"},
        limit=5
    )
    
    print(f"\n✓ Found {result.get('count')} potential targets:")
    for match in result.get("results", []):
        print(f"\n  {match['id']}:")
        print(f"    Score: {match['score']:.3f}")
        print(f"    Employees: {match['metadata']['employees']}")
        print(f"    Revenue: {match['metadata']['revenue']}")
    
    # Step 4: Send findings to acquisition team
    print("\n4. Sending findings to acquisition team...")
    
    targets_summary = "\n".join([
        f"- {m['id']}: {m['metadata']['employees']} employees, {m['metadata']['revenue']} revenue (Score: {m['score']:.3f})"
        for m in result.get("results", [])
    ])
    
    await mcp_manager.call_tool(
        server_name="notification",
        tool_name="send_email",
        to="ma-team@example.com",
        subject="M&A Target Analysis Results",
        body=f"""
M&A Target Search Results

Search Criteria: {search_criteria}

Top Matches:
{targets_summary}

These companies match your acquisition criteria based on semantic similarity analysis.
Please review and provide feedback.
        """
    )
    print("✓ Results sent to M&A team")


async def demo_production_deployment_pipeline():
    """Demonstrate production deployment pipeline using multiple servers."""
    print("\n" + "="*60)
    print("Production Deployment Pipeline")
    print("="*60)
    
    from axiom.integrations.mcp_servers.manager import mcp_manager
    
    version = "v2.1.0"
    
    # Step 1: Build notification
    print(f"\n1. Deploying version {version}...")
    await mcp_manager.call_tool(
        server_name="notification",
        tool_name="send_email",
        to="devops@example.com",
        subject=f"Deployment Started: {version}",
        body=f"Starting deployment of {version} to production cluster"
    )
    
    # Step 2: Create Kubernetes deployment
    print("\n2. Creating Kubernetes deployment...")
    deploy_result = await mcp_manager.call_tool(
        server_name="kubernetes",
        tool_name="create_deployment",
        name=f"axiom-api",
        image=f"axiom/api:{version}",
        replicas=1,  # Start with 1 for testing
        namespace="production",
        port=8000,
        env_vars={"VERSION": version}
    )
    
    if not deploy_result.get("success"):
        print("❌ Deployment failed!")
        await mcp_manager.call_tool(
            server_name="notification",
            tool_name="send_alert",
            recipients={"email": "devops@example.com"},
            severity="error",
            title="Deployment Failed",
            message=f"Failed to deploy {version}: {deploy_result.get('error')}"
        )
        return False
    
    print(f"✓ Deployment created: {deploy_result.get('uid')}")
    
    # Step 3: Wait for pod to be ready
    print("\n3. Waiting for pod to be ready...")
    await asyncio.sleep(5)  # In production, implement proper health check polling
    
    pods_result = await mcp_manager.call_tool(
        server_name="kubernetes",
        tool_name="list_pods",
        namespace="production",
        label_selector="app=axiom-api"
    )
    
    all_ready = all(pod["status"] == "Running" for pod in pods_result.get("pods", []))
    
    if all_ready:
        print("✓ Pods are healthy")
        
        # Step 4: Scale to full capacity
        print("\n4. Scaling to full capacity...")
        await mcp_manager.call_tool(
            server_name="kubernetes",
            tool_name="scale_deployment",
            name="axiom-api",
            replicas=3,
            namespace="production"
        )
        print("✓ Scaled to 3 replicas")
        
        # Step 5: Backup deployment config to S3
        print("\n5. Backing up deployment config...")
        config_backup = {
            "version": version,
            "replicas": 3,
            "deployed_at": datetime.utcnow().isoformat(),
            "deployment_uid": deploy_result.get("uid")
        }
        
        await mcp_manager.call_tool(
            server_name="aws",
            tool_name="s3_upload",
            bucket="axiom-deployments",
            key=f"configs/{version}.json",
            data=json.dumps(config_backup)
        )
        print("✓ Config backed up to S3")
        
        # Step 6: Send success notification
        print("\n6. Sending success notification...")
        await mcp_manager.call_tool(
            server_name="notification",
            tool_name="send_email",
            to=["devops@example.com", "cto@example.com"],
            subject=f"✅ Deployment Success: {version}",
            body=f"""
Deployment Successful

Version: {version}
Replicas: 3
Namespace: production
Deployed at: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC

All health checks passed.
            """
        )
        print("✓ Success notification sent")
        
        print("\n✅ Deployment pipeline completed successfully!")
        return True
    
    else:
        print("❌ Health checks failed!")
        
        # Rollback
        print("\n4. Rolling back deployment...")
        await mcp_manager.call_tool(
            server_name="kubernetes",
            tool_name="delete_deployment",
            name=f"axiom-api",
            namespace="production"
        )
        
        # Send failure notification
        await mcp_manager.call_tool(
            server_name="notification",
            tool_name="send_alert",
            recipients={
                "email": "devops@example.com",
                "phone": "+1234567890"
            },
            severity="critical",
            title="Deployment Failed",
            message=f"Version {version} failed health checks and was rolled back"
        )
        
        return False


async def main():
    """Run all demos."""
    print("="*60)
    print("Week 3 MCP Server Integration Demos")
    print("="*60)
    
    demos = [
        ("AWS Operations", demo_aws_operations),
        ("GCP Operations", demo_gcp_operations),
        ("Notification Operations", demo_notification_operations),
        ("Vector DB Operations", demo_vector_db_operations),
        ("Kubernetes Operations", demo_kubernetes_operations),
        ("Complete Workflow", demo_complete_workflow),
        ("Vector Search Use Case", demo_vector_search_use_case),
        ("Production Pipeline", demo_production_deployment_pipeline),
    ]
    
    for name, demo_func in demos:
        try:
            await demo_func()
        except Exception as e:
            print(f"\n❌ {name} demo failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("All demos completed!")
    print("="*60)


if __name__ == "__main__":
    # Run demos
    print("""
    Week 3 MCP Integration Examples
    ================================
    
    This demo showcases all 5 advanced MCP servers:
    1. AWS MCP Server (S3, EC2, Lambda, CloudWatch)
    2. GCP MCP Server (Storage, Compute, BigQuery, Functions)
    3. Notification MCP Server (Email, SMS, Multi-channel)
    4. Vector DB MCP Server (Semantic search, Collections)
    5. Kubernetes MCP Server (Deployments, Services, Pods)
    
    Note: Some demos require actual service credentials and running infrastructure.
    Set up environment variables and Docker services before running.
    """)
    
    asyncio.run(main())