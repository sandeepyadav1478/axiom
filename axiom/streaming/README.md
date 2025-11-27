# Axiom Real-Time Streaming API

Production-grade real-time streaming infrastructure for live market intelligence, AI analysis results, and data quality metrics.

## 🚀 Features

### Core Capabilities
- **WebSocket Endpoints**: Bidirectional real-time communication
- **Server-Sent Events (SSE)**: One-way streaming for dashboards
- **Redis Pub/Sub**: Multi-instance message broadcasting
- **Connection Management**: Automatic heartbeat, health monitoring, reconnection
- **Load Balancing**: NGINX reverse proxy with 3 API instances
- **Horizontal Scaling**: Redis-powered event distribution
- **Production Ready**: Docker containerization, monitoring, metrics

### Event Types
- **Price Updates**: Live market data streaming
- **News Alerts**: Breaking news and market events
- **Claude Analysis**: AI-powered insights and recommendations
- **Graph Updates**: Neo4j relationship and node changes
- **Quality Metrics**: Data validation and anomaly detection
- **M&A Workflow**: Deal analysis progress tracking

## 📋 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Client Applications                      │
│  (Dashboard, Mobile App, Trading System, Analytics)         │
└────────┬────────────────────────────────────────────────┬───┘
         │ WebSocket/SSE                                   │
         ↓                                                 ↓
┌────────────────────────────────────────────────────────────┐
│              NGINX Load Balancer (Port 8001)               │
│  • Least Connection Load Balancing                         │
│  • WebSocket Upgrade Support                               │
│  • SSE Connection Handling                                 │
└────────┬───────────────┬───────────────┬──────────────────┘
         │               │               │
         ↓               ↓               ↓
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ Streaming   │  │ Streaming   │  │ Streaming   │
│ API #1      │  │ API #2      │  │ API #3      │
│ (FastAPI)   │  │ (FastAPI)   │  │ (FastAPI)   │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       │                │                │
       └────────────────┼────────────────┘
                        ↓
                ┌───────────────┐
                │  Redis Pub/Sub │
                │  (Port 6379)   │
                └───────────────┘
                        ↑
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ↓               ↓               ↓
┌──────────────┐ ┌─────────────┐ ┌──────────────┐
│  LangGraph   │ │   Neo4j     │ │   Quality    │
│  M&A Agent   │ │   Graph DB  │ │   Framework  │
└──────────────┘ └─────────────┘ └──────────────┘
```

## 🔧 Installation

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- Redis 7+

### Quick Start

```bash
# Clone repository
cd axiom/streaming

# Install dependencies
pip install -r requirements.txt

# Start with Docker Compose (recommended)
docker-compose up -d

# Or run locally
uvicorn axiom.streaming.streaming_service:app --host 0.0.0.0 --port 8001
```

### Docker Deployment

```bash
# Start all services (3 API instances + Redis + NGINX + Monitoring)
docker-compose -f axiom/streaming/docker-compose.yml up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f streaming-api-1

# Scale API instances
docker-compose up -d --scale streaming-api=5

# Stop services
docker-compose down
```

## 📡 API Endpoints

### WebSocket

```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8001/ws/client-id-123');

// Subscribe to events
ws.send(JSON.stringify({
    action: 'subscribe',
    event_types: ['price_update', 'news_alert', 'claude_analysis']
}));

// Receive events
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Event:', data.event_type, data.data);
};

// Handle connection
ws.onopen = () => console.log('Connected');
ws.onclose = () => console.log('Disconnected');
```

### Server-Sent Events (SSE)

```javascript
// Connect to SSE
const eventSource = new EventSource('http://localhost:8001/sse/client-id-123');

// Listen for specific event types
eventSource.addEventListener('price_update', (e) => {
    const data = JSON.parse(e.data);
    console.log('Price Update:', data);
});

eventSource.addEventListener('news_alert', (e) => {
    const data = JSON.parse(e.data);
    console.log('News Alert:', data);
});
```

### REST API

#### Publish Events

```bash
# Publish price update
curl -X POST http://localhost:8001/publish/price \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "price": 150.25,
    "volume": 5000
  }'

# Publish news alert
curl -X POST http://localhost:8001/publish/news \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Breaking News",
    "summary": "Market update...",
    "url": "https://example.com"
  }'

# Publish Claude analysis
curl -X POST http://localhost:8001/publish/analysis \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Should we invest?",
    "answer": "Based on analysis...",
    "confidence": 0.85,
    "reasoning": ["Data point 1", "Data point 2"]
  }'
```

#### Health & Stats

```bash
# Health check
curl http://localhost:8001/health

# Statistics
curl http://localhost:8001/stats
```

## 🎯 Usage Examples

### Python Client

```python
import asyncio
from axiom.streaming.integrations import IntegratedStreamingPlatform

async def main():
    # Initialize platform
    platform = IntegratedStreamingPlatform()
    
    # Publish price update
    await platform.market.stream_price_updates(['AAPL', 'GOOGL'])
    
    # Publish analysis
    await platform.streaming.publish_analysis(
        query="What are the risks?",
        answer="Primary risks include...",
        confidence=0.82,
        reasoning=["Historical data", "Expert opinions"]
    )
    
    # Monitor quality
    data = {"price": 100.50, "volume": 5000}
    await platform.quality.monitor_data_quality(data, "market_data")

asyncio.run(main())
```

### JavaScript Dashboard

```javascript
class StreamingDashboard {
    constructor(clientId) {
        this.ws = new WebSocket(`ws://localhost:8001/ws/${clientId}`);
        this.setupHandlers();
    }
    
    setupHandlers() {
        this.ws.onopen = () => {
            console.log('Connected');
            this.subscribe(['price_update', 'news_alert']);
        };
        
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleEvent(data);
        };
        
        this.ws.onclose = () => {
            console.log('Disconnected, reconnecting...');
            setTimeout(() => this.reconnect(), 5000);
        };
    }
    
    subscribe(eventTypes) {
        this.ws.send(JSON.stringify({
            action: 'subscribe',
            event_types: eventTypes
        }));
    }
    
    handleEvent(data) {
        switch(data.event_type) {
            case 'price_update':
                this.updatePrice(data.data);
                break;
            case 'news_alert':
                this.showNews(data.data);
                break;
            case 'claude_analysis':
                this.displayAnalysis(data.data);
                break;
        }
    }
}

// Usage
const dashboard = new StreamingDashboard('dashboard-123');
```

## 🔐 Production Configuration

### Environment Variables

```bash
# Redis Configuration
REDIS_URL=redis://redis:6379

# API Configuration
LOG_LEVEL=INFO
PYTHONUNBUFFERED=1

# Connection Settings
HEARTBEAT_INTERVAL=30  # seconds
CONNECTION_TIMEOUT=90  # seconds
MAX_RECONNECT_ATTEMPTS=10
```

### NGINX Configuration

Key settings in [`nginx.conf`](./nginx.conf):
- Load balancing algorithm: `least_conn`
- WebSocket upgrade support
- SSE connection handling
- Long-lived connection timeouts (7 days)
- Buffering disabled for real-time streaming

### Redis Configuration

Optimized for pub/sub:
- Max memory: 256MB
- Eviction policy: `allkeys-lru`
- Persistence: AOF (Append-Only File)

## 📊 Monitoring

### Prometheus Metrics

Access at: `http://localhost:9090`

Available metrics:
- Connection count
- Message throughput
- Event latency
- Error rates
- Redis pub/sub stats

### Grafana Dashboards

Access at: `http://localhost:3001`
- Default credentials: `admin/admin`
- Pre-configured streaming dashboards
- Real-time connection monitoring
- Performance analytics

## 🧪 Testing

### Run Demo

```bash
# Comprehensive demo
python demos/demo_streaming_api.py

# With WebSocket testing
python demos/demo_streaming_api.py --test-ws
```

### Load Testing

```bash
# Install locust
pip install locust

# Run load test (1000 concurrent connections)
locust -f tests/load_test_streaming.py --host ws://localhost:8001
```

## 🚨 Troubleshooting

### Connection Issues

```bash
# Check if services are running
docker-compose ps

# View logs
docker-compose logs -f streaming-api-1

# Test connectivity
curl http://localhost:8001/health
```

### Redis Issues

```bash
# Check Redis connection
docker-compose exec redis redis-cli ping

# View Redis info
docker-compose exec redis redis-cli info
```

### Performance Issues

```bash
# Check resource usage
docker stats

# Scale API instances
docker-compose up -d --scale streaming-api=5

# Monitor metrics
curl http://localhost:8001/stats
```

## 🎓 Best Practices

### Connection Management
1. Always implement reconnection logic
2. Use exponential backoff for reconnects
3. Set appropriate timeouts
4. Handle connection errors gracefully

### Event Publishing
1. Use appropriate event types
2. Include correlation IDs for tracking
3. Add metadata for debugging
4. Validate data before publishing

### Scaling
1. Use Redis for multi-instance coordination
2. Monitor connection distribution
3. Scale horizontally with Docker Compose
4. Use load balancer health checks

### Security
1. Implement authentication (JWT tokens)
2. Use WSS (WebSocket Secure) in production
3. Rate limit connections per client
4. Validate all incoming messages

## 📚 Additional Resources

- [FastAPI WebSocket Documentation](https://fastapi.tiangolo.com/advanced/websockets/)
- [Redis Pub/Sub Guide](https://redis.io/docs/manual/pubsub/)
- [NGINX Load Balancing](https://docs.nginx.com/nginx/admin-guide/load-balancer/)
- [Server-Sent Events Specification](https://html.spec.whatwg.org/multipage/server-sent-events.html)

## 🤝 Contributing

Contributions welcome! Please:
1. Follow existing code style
2. Add tests for new features
3. Update documentation
4. Submit pull requests

## 📄 License

Part of the Axiom Analytics Platform. See main repository LICENSE file.

## 🆘 Support

For issues, questions, or feature requests:
- Open an issue on GitHub
- Check existing documentation
- Review demo examples

---

**Built with ❤️ for production real-time streaming**

Last Updated: November 2025
Version: 1.0.0