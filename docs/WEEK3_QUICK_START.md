# Week 3 MCP Servers - Quick Start Guide

Get started with Week 3 MCP servers in 5 minutes!

## Prerequisites

- Python 3.11+
- Docker and Docker Compose
- AWS/GCP accounts (optional for testing)

---

## 1. Install Dependencies

```bash
# Install all Week 3 dependencies
pip install boto3 botocore google-cloud-storage google-cloud-compute \
    google-cloud-bigquery sendgrid twilio chromadb qdrant-client \
    pinecone-client kubernetes

# Or install from requirements.txt
pip install -r requirements.txt
```

---

## 2. Start Docker Services

```bash
# Create network
docker network create axiom-network

# Start Week 3 services
docker-compose -f docker/week3-services.yml up -d

# Verify services
docker-compose -f docker/week3-services.yml ps
```

You should see:
- ✅ ChromaDB running on port 8000
- ✅ Qdrant running on port 6333
- ✅ Weaviate running on port 8080
- ✅ LocalStack running on port 4566
- ✅ MailHog running on ports 1025/8025

---

## 3. Configure Credentials

Copy and update `.env.example` to `.env`:

```bash
cp .env.example .env
```

**Minimal Configuration** (for local testing):

```bash
# Vector DB (using local ChromaDB)
VECTOR_DB_MCP_PROVIDER=chromadb
VECTOR_DB_MCP_CHROMADB_HOST=localhost
VECTOR_DB_MCP_CHROMADB_PORT=8000

# Email (using MailHog for testing)
NOTIFICATION_MCP_SMTP_SERVER=localhost
NOTIFICATION_MCP_SMTP_PORT=1025
NOTIFICATION_MCP_SMTP_USER=test@example.com
NOTIFICATION_MCP_SMTP_PASSWORD=password
```

**Production Configuration**:

```bash
# AWS
AWS_MCP_REGION=us-east-1
AWS_MCP_PROFILE=default  # Or use access keys

# GCP
GCP_MCP_PROJECT_ID=your-project-id
GCP_MCP_CREDENTIALS_PATH=/path/to/credentials.json

# Real Email (Gmail example)
NOTIFICATION_MCP_SMTP_SERVER=smtp.gmail.com
NOTIFICATION_MCP_SMTP_PORT=587
NOTIFICATION_MCP_SMTP_USER=your-email@gmail.com
NOTIFICATION_MCP_SMTP_PASSWORD=your-app-password

# SMS (Twilio)
NOTIFICATION_MCP_TWILIO_ACCOUNT_SID=ACxxx
NOTIFICATION_MCP_TWILIO_AUTH_TOKEN=xxx
NOTIFICATION_MCP_TWILIO_FROM_NUMBER=+1234567890
```

---

## 4. Test Installation

Run the demo script:

```bash
python demos/demo_week3_mcp_integration.py
```

Or run specific tests:

```bash
# Test all servers
pytest tests/test_mcp_week3_servers.py -v

# Test specific server
pytest tests/test_mcp_week3_servers.py::TestAWSMCPServer -v
pytest tests/test_mcp_week3_servers.py::TestVectorDBMCPServer -v
```

---

## 5. First Usage Example

```python
import asyncio
from axiom.integrations.mcp_servers.manager import mcp_manager

async def hello_week3():
    # 1. Send an email
    result = await mcp_manager.call_tool(
        server_name="notification",
        tool_name="send_email",
        to="test@example.com",
        subject="Hello from Week 3 MCP!",
        body="Successfully configured Week 3 MCP servers!"
    )
    print(f"Email sent: {result['success']}")
    
    # 2. Create a vector collection
    result = await mcp_manager.call_tool(
        server_name="vector_db",
        tool_name="create_collection",
        collection="test_collection",
        dimension=1536
    )
    print(f"Collection created: {result['success']}")
    
    # 3. List collections
    result = await mcp_manager.call_tool(
        server_name="vector_db",
        tool_name="list_collections"
    )
    print(f"Collections: {result['collections']}")

asyncio.run(hello_week3())
```

---

## Common Issues & Solutions

### Issue: "boto3 not found"
**Solution**: Install AWS SDK
```bash
pip install boto3 botocore
```

### Issue: "Cannot connect to ChromaDB"
**Solution**: Start Docker services
```bash
docker-compose -f docker/week3-services.yml up -d chromadb
```

### Issue: "SMTP authentication failed"
**Solution**: Use app-specific password for Gmail
1. Go to Google Account → Security → 2-Step Verification
2. App passwords → Generate new password
3. Use generated password in `.env`

### Issue: "Kubernetes config not found"
**Solution**: Set up kubeconfig
```bash
# For GKE
gcloud container clusters get-credentials CLUSTER_NAME

# For EKS
aws eks update-kubeconfig --name CLUSTER_NAME

# For minikube
minikube start
```

---

## Next Steps

1. **Explore Examples**: Review [`demo_week3_mcp_integration.py`](../demos/demo_week3_mcp_integration.py)
2. **Read Documentation**: See [`WEEK3_CLOUD_MCP_GUIDE.md`](WEEK3_CLOUD_MCP_GUIDE.md)
3. **Configure Production**: Set up real cloud credentials
4. **Deploy to K8s**: Follow production deployment guide
5. **Monitor Performance**: Check CloudWatch/GCP metrics

---

## Quick Reference

### Available Servers

| Server | Category | Tools | Key Features |
|--------|----------|-------|--------------|
| **aws** | cloud | 12 | S3, EC2, Lambda, CloudWatch |
| **gcp** | cloud | 10 | Storage, Compute, BigQuery, Functions |
| **notification** | communication | 12 | Email, SMS, Multi-channel |
| **vector_db** | storage | 10 | Semantic search, 4 providers |
| **kubernetes** | devops | 15 | Deployments, Services, Pods |

### Service URLs (Docker)

- ChromaDB: http://localhost:8000
- Qdrant: http://localhost:6333
- Weaviate: http://localhost:8080
- LocalStack: http://localhost:4566
- MailHog Web UI: http://localhost:8025

### Documentation

- [`README_WEEK3.md`](../axiom/integrations/mcp_servers/README_WEEK3.md) - Server reference
- [`WEEK3_CLOUD_MCP_GUIDE.md`](WEEK3_CLOUD_MCP_GUIDE.md) - Integration guide
- [`WEEK3_EXECUTION_SUMMARY.md`](WEEK3_EXECUTION_SUMMARY.md) - Implementation summary

---

**Ready to use Week 3 MCP servers!** 🚀