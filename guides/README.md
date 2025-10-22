# 📚 Axiom Guides & Setup Instructions

This directory contains setup guides and instructions for configuring various components of the Axiom investment banking analytics platform.

## 📁 Available Guides

### MCP Server Setup
- **[`FINANCIAL_MCP_SERVERS_GUIDE.md`](FINANCIAL_MCP_SERVERS_GUIDE.md)** - Comprehensive guide to financial MCP servers
  - Polygon.io MCP Server setup
  - Yahoo Finance Professional MCP Server
  - Yahoo Finance Comprehensive MCP Server
  - Cost structures and API key rotation strategies
  
- **[`INSTALLATION_GUIDE.md`](INSTALLATION_GUIDE.md)** - Installation instructions for MCP servers
  - Official uvx installation methods
  - Project-level MCP configuration
  - Docker external reference setup

## 🚀 Quick Links

### Getting Started
1. Review the **[Project README](../README.md)** for overview
2. Follow the **[Setup Guide](../docs/SETUP_GUIDE.md)** for initial setup
3. Check the **[Quick Start](../docs/QUICKSTART.md)** for basic usage
4. Explore **[MCP Server Installation](INSTALLATION_GUIDE.md)** for enhanced capabilities

### Documentation Structure
```
axiom/
├── demos/              # Demo files and examples
├── guides/             # Setup and configuration guides (you are here)
├── docs/               # Core documentation
│   ├── ma-workflows/   # M&A workflow documentation
│   ├── deployment/     # Deployment guides
│   └── architecture/   # Architecture documentation
└── axiom/              # Source code
```

## 🔧 Configuration Files

### Environment Configuration
- **[`../.env`](../.env)** - Main environment configuration file
- **[`../.env.example`](../.env.example)** - Example environment configuration

### Docker Compose
- **[MCP Docker Compose](../axiom/integrations/data_sources/finance/mcp_servers/docker-compose.yml)** - MCP server container configuration

## 💡 Key Features Covered

### Financial MCP Servers
- **Free Tier Optimization**: Using multiple free API keys with rotation
- **Professional Data Sources**: Polygon.io, Yahoo Finance, and more
- **Cost Savings**: 98%+ savings vs Bloomberg Terminal
- **40+ Financial Tools**: Comprehensive market data and analysis

### API Key Management
- Centralized configuration through `.env` file
- API key rotation for free tier maximization
- Support for multiple providers

## 🎯 Next Steps

1. **Set up your API keys**: Follow the [Installation Guide](INSTALLATION_GUIDE.md)
2. **Configure MCP servers**: See [Financial MCP Servers Guide](FINANCIAL_MCP_SERVERS_GUIDE.md)
3. **Run demos**: Try examples in the [`../demos/`](../demos/) folder
4. **Explore workflows**: Check [`../docs/ma-workflows/`](../docs/ma-workflows/) for M&A workflows

## 📞 Support

For more information:
- Review the main [README](../README.md)
- Check [documentation](../docs/)
- Explore [demo files](../demos/)