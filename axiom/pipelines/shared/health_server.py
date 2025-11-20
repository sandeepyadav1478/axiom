"""
Health Check HTTP Server for Pipeline Monitoring
Exposes /health and /metrics endpoints for observability
"""
import asyncio
from aiohttp import web
import logging
from typing import Dict, Any
import json

logger = logging.getLogger(__name__)


class HealthCheckServer:
    """
    HTTP server for health checks and metrics
    Runs alongside pipeline to expose monitoring endpoints
    """
    
    def __init__(self, port: int = 8080, metrics_collector=None):
        self.port = port
        self.metrics_collector = metrics_collector
        self.app = web.Application()
        self.runner = None
        
        # Setup routes
        self.app.router.add_get('/health', self.health_handler)
        self.app.router.add_get('/metrics', self.metrics_handler)
        self.app.router.add_get('/ready', self.readiness_handler)
        self.app.router.add_get('/live', self.liveness_handler)
        
    async def health_handler(self, request):
        """Health check endpoint - detailed status"""
        if self.metrics_collector:
            health_status = self.metrics_collector.get_health_status()
            status_code = 200 if health_status['status'] == 'healthy' else 503
            return web.json_response(health_status, status=status_code)
        return web.json_response({'status': 'unknown'}, status=503)
        
    async def metrics_handler(self, request):
        """Prometheus metrics endpoint"""
        if self.metrics_collector:
            metrics_text = self.metrics_collector.to_prometheus()
            return web.Response(text=metrics_text, content_type='text/plain')
        return web.Response(text='# No metrics available', content_type='text/plain')
        
    async def readiness_handler(self, request):
        """Readiness probe - can accept traffic"""
        if self.metrics_collector:
            health = self.metrics_collector.get_health_status()
            if health['status'] in ['healthy', 'degraded']:
                return web.json_response({'ready': True}, status=200)
        return web.json_response({'ready': False}, status=503)
        
    async def liveness_handler(self, request):
        """Liveness probe - is running"""
        return web.json_response({'alive': True}, status=200)
        
    async def start(self):
        """Start health check server"""
        try:
            self.runner = web.AppRunner(self.app)
            await self.runner.setup()
            site = web.TCPSite(self.runner, '0.0.0.0', self.port)
            await site.start()
            logger.info(f"✅ Health check server started on port {self.port}")
            logger.info(f"   Health: http://localhost:{self.port}/health")
            logger.info(f"   Metrics: http://localhost:{self.port}/metrics")
        except Exception as e:
            logger.error(f"Failed to start health server: {e}")
            
    async def stop(self):
        """Stop health check server"""
        if self.runner:
            await self.runner.cleanup()
            logger.info("Health check server stopped")