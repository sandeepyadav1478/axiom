# Axiom Unified Financial Data Services

Consolidated MCP Servers and Provider Containers for comprehensive financial data access in the Axiom Investment Banking Analytics platform.

## 🎯 Overview

This directory contains a **unified Docker Compose system** that consolidates:

1. **MCP Servers** (Model Context Protocol) - For AI agent integration via stdio protocol
2. **Provider Containers** (REST APIs) - For direct HTTP access with health checks

Both systems share the same environment configuration from the project root [`/.env`](../../../.env) file for simplified management.

## 📋 Architecture

```
axiom/integrations/data_sources/finance/
├── docker-compose.yml              # Unified compose for all services
├── manage-financial-services.sh    # Unified management script
├── README.md                        # This file
├── mcp_servers/                     # Legacy - will be deprecated
└── provider_containers/             # Legacy - will be deprecated
    ├── tavily/
    ├── fmp/
    ├── finnhub/
    └── alpha_vantage/
```

## 🚀 Quick Start

### Prerequisites

1. **Docker Desktop** installed and running
2. **Project root .env file** configured with API keys:
   ```bash
   cd /Users/sandeep.yadav/work/axiom
   cp .env.example .env
   # Edit .env and add your API keys
   ```

### Start All Services

```bash
cd axiom/integrations/data_sources/finance
./manage-financial-services.sh start
```

### Check Status

```bash
./manage-financial-services.sh status
./manage-financial-services.sh health
```

## 📊 Available Services

### MCP Servers (Profile: `mcp`)

AI-integrated services using stdio protocol:

| Service | Profile | Description | Cost |
|---------|---------|-------------|------|
| **polygon-io-server** | `polygon` | Real-time market data | FREE: 5 calls/min<br>Premium: $25/mo |
| **yahoo-finance-professional** | `yahoo-pro` | 27 professional tools | 100% FREE |
| **yahoo-finance-comprehensive** | `yahoo-comp` | Fundamental analysis | 100% FREE |
| **firecrawl-server** | `firecrawl` | Web scraping | FREE tier available |

### Provider Containers (Profile: `providers`)

REST API services with health monitoring:

| Service | Port | Profile | Description | Cost |
|---------|------|---------|-------------|------|
| **tavily-provider** | 8001 | `tavily` | Search & M&A intelligence | Paid service |
| **fmp-provider** | 8002 | `fmp` | Comprehensive financial data | FREE: 250/day<br>Premium: $14/mo |
| **finnhub-provider** | 8003 | `finnhub` | Real-time market data | FREE: 60/min<br>Premium: $7.99/mo |
| **alpha-vantage-provider** | 8004 | `alpha-vantage` | Financial data | FREE: 500/day<br>Premium: $49/mo |

## 🛠️ Management Commands

### Basic Commands

```bash
# Start all services
./manage-financial-services.sh start

# Start specific category
./manage-financial-services.sh start mcp        # All MCP servers
./manage-financial-services.sh start providers  # All providers

# Start specific services
./manage-financial-services.sh start polygon fmp tavily

# Stop services
./manage-financial-services.sh stop

# Restart services
./manage-financial-services.sh restart

# View logs
./manage-financial-services.sh logs
./manage-financial-services.sh logs fmp        # Specific service
```

### Status & Health

```bash
# Check status of all services
./manage-financial-services.sh status

# Health check for provider containers
./manage-financial-services.sh health

# Show API key configuration
./manage-financial-services.sh keys

# Show detailed service information
./manage-financial-services.sh info
```

### Maintenance

```bash
# Rebuild containers
./manage-financial-services.sh rebuild
./manage-financial-services.sh rebuild fmp     # Specific service

# Clean up (removes all containers and volumes)
./manage-financial-services.sh clean
```

## 🔑 Configuration

All services use environment variables from the project root [`.env`](../../../.env) file.

### Required API Keys

**For MCP Servers:**
```bash
POLYGON_API_KEY=your_polygon_key          # Get FREE key at polygon.io
FIRECRAWL_API_KEY=your_firecrawl_key      # Get FREE key at firecrawl.dev
```

**For Provider Containers:**
```bash
TAVILY_API_KEY=your_tavily_key                    # Get key at tavily.com
FINANCIAL_MODELING_PREP_API_KEY=your_fmp_key      # Get FREE key at financialmodelingprep.com
FINNHUB_API_KEY=your_finnhub_key                  # Get FREE key at finnhub.io
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key      # Get FREE key at alphavantage.co
```

### Optional Configuration

```bash
# API Key Rotation
FINANCIAL_API_ROTATION_ENABLED=true
POLYGON_API_ROTATION_ENABLED=false
TAVILY_ROTATION_ENABLED=false

# Multiple keys for rotation (comma-separated)
TAVILY_API_KEYS=key1,key2,key3

# Data Configuration
FINANCIAL_DATA_DEPTH=comprehensive   # or 'basic'
DEBUG=false

# Port Configuration (optional overrides)
TAVILY_PROVIDER_PORT=8001
FMP_PROVIDER_PORT=8002
FINNHUB_PROVIDER_PORT=8003
ALPHA_VANTAGE_PROVIDER_PORT=8004
```

## 📡 API Access

### Provider Container REST APIs

Each provider container exposes REST endpoints:

```bash
# Health checks
curl http://localhost:8001/health  # Tavily
curl http://localhost:8002/health  # FMP
curl http://localhost:8003/health  # Finnhub
curl http://localhost:8004/health  # Alpha Vantage

# API documentation (when running)
http://localhost:8001/docs  # Tavily
http://localhost:8002/docs  # FMP
http://localhost:8003/docs  # Finnhub
http://localhost:8004/docs  # Alpha Vantage
```

### MCP Server Integration

MCP servers use stdio protocol and integrate with AI tools:
- Configure in Roo/Claude MCP settings
- Automatically detected by AI agents
- Access via natural language queries

## 🎯 Usage Examples

### Start Everything

```bash
./manage-financial-services.sh start
```

### Start Only Free Services

```bash
# Free MCP servers (Yahoo Finance)
./manage-financial-services.sh start yahoo-pro yahoo-comp

# Free provider containers
./manage-financial-services.sh start fmp finnhub alpha-vantage
```

### Development Setup

```bash
# Start core services for development
./manage-financial-services.sh start polygon fmp

# Watch logs
./manage-financial-services.sh logs polygon fmp
```

### Production Setup

```bash
# Start all services
./manage-financial-services.sh start

# Enable automatic restart
# (already configured with restart: unless-stopped in docker-compose.yml)

# Monitor health
watch -n 30 './manage-financial-services.sh health'
```

## 🔍 Troubleshooting

### Services Won't Start

1. **Check Docker is running:**
   ```bash
   docker info
   ```

2. **Verify API keys in .env:**
   ```bash
   ./manage-financial-services.sh keys
   ```

3. **Check logs:**
   ```bash
   ./manage-financial-services.sh logs [service-name]
   ```

4. **Rebuild containers:**
   ```bash
   ./manage-financial-services.sh rebuild
   ```

### Port Already in Use

```bash
# Check what's using the port
lsof -i :8001  # or other port

# Kill the process or change port in .env
```

### API Keys Not Working

1. Verify keys are correctly formatted in `.env`
2. Restart services:
   ```bash
   ./manage-financial-services.sh restart
   ```
3. Check logs for specific errors

### Container Health Issues

```bash
# Check container logs
docker logs axiom-fmp-provider

# Check network connectivity
docker network inspect axiom-financial-data-unified

# Recreate containers
./manage-financial-services.sh clean
./manage-financial-services.sh start
```

## 💰 Cost Analysis

### FREE Tier Capabilities

| Service | Free Limit | Cost |
|---------|-----------|------|
| Yahoo Finance Pro | Unlimited | $0 |
| Yahoo Finance Comprehensive | Unlimited | $0 |
| FMP | 250 calls/day | $0 |
| Finnhub | 60 calls/minute | $0 |
| Alpha Vantage | 500 calls/day | $0 |
| Polygon.io | 5 calls/minute | $0 |
| Firecrawl | Limited | $0 |
| **Total FREE** | **~750+ calls/day** | **$0/month** |

### Premium Costs

| Service | Plan | Cost |
|---------|------|------|
| FMP | 10K calls | $14/month |
| Finnhub | Unlimited | $7.99/month |
| Alpha Vantage | Unlimited | $49/month |
| Polygon.io | Unlimited | $25/month |
| Tavily | Search API | Variable |
| **Total Premium** | **All unlimited** | **~$71/month** |

**Compare to Bloomberg Terminal:** $2,000+/month  
**Cost Savings:** 96%+ 🎉

## 🐳 Docker Compose Commands

Alternative to using the management script:

```bash
# Start all services
docker-compose up -d

# Start with profiles
docker-compose --profile mcp up -d
docker-compose --profile providers up -d
docker-compose --profile polygon --profile fmp up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f

# Rebuild
docker-compose build --no-cache
docker-compose up -d --build

# Check status
docker-compose ps
```

## 📚 Related Documentation

- [Project Setup Guide](../../../SETUP_GUIDE.md)
- [Financial Provider Integration](../../../FINANCIAL_PROVIDER_INTEGRATION_SUMMARY.md)
- [.env.example Configuration](../../../.env.example)
- [Installation Guide](../../../guides/INSTALLATION_GUIDE.md)

## 🔄 Migration from Legacy System

If you're migrating from the old separate systems:

1. **Stop old services:**
   ```bash
   cd axiom/integrations/data_sources/finance/mcp_servers
   docker-compose down
   
   cd ../provider_containers
   docker-compose down
   ```

2. **Use new unified system:**
   ```bash
   cd axiom/integrations/data_sources/finance
   ./manage-financial-services.sh start
   ```

3. **Same .env file works** - no configuration changes needed!

## 🤝 Contributing

When adding new services:

1. Add service definition to [`docker-compose.yml`](docker-compose.yml)
2. Use appropriate profile (`mcp` or `providers`)
3. Reference root `.env` file: `env_file: - ../../../.env`
4. Add to [`manage-financial-services.sh`](manage-financial-services.sh) if needed
5. Update this README with service details

## 📄 License

Part of the Axiom Investment Banking Analytics platform.

## 🆘 Support

For issues or questions:
1. Check logs: `./manage-financial-services.sh logs`
2. Verify configuration: `./manage-financial-services.sh keys`
3. Review troubleshooting section above
4. Check Docker status: `docker ps`

---

**Last Updated:** 2025-01-22  
**Unified System Version:** 1.0.0