"""
Integrations with LangGraph, Neo4j, and Quality Metrics.

Connects streaming service with existing platform components.
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from datetime import datetime

from .streaming_service import StreamingService
from .event_types import (
    DealAnalysisEvent, GraphUpdateEvent, QualityMetricEvent,
    NewsAlertEvent, ClaudeAnalysisEvent
)

logger = logging.getLogger(__name__)


class LangGraphIntegration:
    """
    Integration with LangGraph M&A orchestrator.
    
    Streams real-time updates during deal analysis.
    """
    
    def __init__(self, streaming_service: StreamingService):
        self.streaming = streaming_service
        self.active_analyses: Dict[str, Dict] = {}
    
    async def start_deal_analysis(
        self,
        deal_id: str,
        target_company: str,
        acquirer_company: Optional[str] = None
    ):
        """
        Start M&A deal analysis with real-time streaming.
        
        Args:
            deal_id: Unique deal identifier
            target_company: Target company symbol
            acquirer_company: Optional acquirer symbol
        """
        try:
            from axiom.ai_layer.langgraph_ma_orchestrator import MAOrchestrator
            
            # Initialize orchestrator
            orchestrator = MAOrchestrator()
            
            # Track analysis
            self.active_analyses[deal_id] = {
                "target": target_company,
                "acquirer": acquirer_company,
                "start_time": datetime.now(),
                "status": "running"
            }
            
            # Publish start event
            await self.streaming.publish_deal_analysis(
                deal_id=deal_id,
                stage="research",
                progress=0.0,
                message=f"Starting analysis for {target_company}"
            )
            
            # Run analysis with progress updates
            result = await self._run_with_streaming(
                orchestrator,
                deal_id,
                target_company,
                acquirer_company
            )
            
            # Publish completion
            await self.streaming.publish_deal_analysis(
                deal_id=deal_id,
                stage="complete",
                progress=1.0,
                message=f"Analysis complete: {result.get('recommendation', 'unknown')}"
            )
            
            # Mark as complete
            self.active_analyses[deal_id]["status"] = "complete"
            
            return result
            
        except Exception as e:
            logger.error(f"Error in deal analysis: {e}")
            
            # Publish error
            await self.streaming.publish_deal_analysis(
                deal_id=deal_id,
                stage="error",
                progress=0.0,
                message=f"Analysis failed: {str(e)}"
            )
            
            if deal_id in self.active_analyses:
                self.active_analyses[deal_id]["status"] = "error"
            
            raise
    
    async def _run_with_streaming(
        self,
        orchestrator,
        deal_id: str,
        target: str,
        acquirer: Optional[str]
    ) -> Dict[str, Any]:
        """Run analysis with progress streaming."""
        
        # Define stages with progress percentages
        stages = [
            ("research", 0.2, "Gathering company intelligence"),
            ("financial", 0.4, "Analyzing financials"),
            ("strategic", 0.6, "Assessing strategic fit"),
            ("risk", 0.8, "Identifying risks"),
            ("valuation", 0.9, "Calculating valuation"),
            ("recommendation", 1.0, "Generating recommendation")
        ]
        
        # Stream progress updates
        for stage, progress, message in stages:
            await self.streaming.publish_deal_analysis(
                deal_id=deal_id,
                stage=stage,
                progress=progress,
                message=message
            )
            await asyncio.sleep(0.5)  # Small delay for demo
        
        # Run actual analysis
        result = orchestrator.analyze_deal(
            target=target,
            acquirer=acquirer,
            analysis_type='acquisition_target'
        )
        
        return result


class Neo4jIntegration:
    """
    Integration with Neo4j graph database.
    
    Streams graph updates in real-time.
    """
    
    def __init__(self, streaming_service: StreamingService):
        self.streaming = streaming_service
    
    async def stream_graph_update(
        self,
        update_type: str,
        node_id: str,
        properties: Dict[str, Any]
    ):
        """
        Stream a graph update event.
        
        Args:
            update_type: Type of update (create, update, delete)
            node_id: Node identifier
            properties: Node properties
        """
        await self.streaming.publish_graph_update(
            update_type=update_type,
            node_id=node_id,
            properties=properties
        )
    
    async def stream_company_creation(self, symbol: str, name: str, sector: str):
        """Stream company node creation."""
        await self.stream_graph_update(
            update_type="create_company",
            node_id=symbol,
            properties={
                "name": name,
                "sector": sector,
                "created_at": datetime.now().isoformat()
            }
        )
    
    async def stream_relationship_added(
        self,
        from_id: str,
        to_id: str,
        rel_type: str,
        properties: Optional[Dict] = None
    ):
        """Stream relationship creation."""
        await self.streaming.publish_graph_update(
            update_type="add_relationship",
            node_id=f"{from_id}->{to_id}",
            properties={
                "relationship_type": rel_type,
                "from": from_id,
                "to": to_id,
                **(properties or {})
            }
        )


class QualityMetricsIntegration:
    """
    Integration with data quality framework.
    
    Streams quality metrics and anomalies.
    """
    
    def __init__(self, streaming_service: StreamingService):
        self.streaming = streaming_service
    
    async def stream_quality_score(
        self,
        metric_name: str,
        score: float,
        threshold: float,
        dataset: str
    ):
        """
        Stream a quality metric.
        
        Args:
            metric_name: Name of the metric
            score: Metric score (0-1)
            threshold: Acceptable threshold
            dataset: Dataset being measured
        """
        status = "pass" if score >= threshold else "fail"
        
        await self.streaming.publish_quality_metric(
            metric_name=metric_name,
            score=score,
            threshold=threshold,
            status=status
        )
    
    async def monitor_data_quality(self, data: Dict[str, Any], dataset_name: str):
        """
        Monitor data quality and stream metrics.
        
        Args:
            data: Data to validate
            dataset_name: Name of the dataset
        """
        try:
            from axiom.data_quality import get_validation_engine
            from axiom.data_quality.profiling.anomaly_detector import get_anomaly_detector
            
            # Get validation engine
            validation_engine = get_validation_engine()
            
            # Run validation
            results = validation_engine.validate_data(
                data,
                dataset_name,
                raise_on_critical=False
            )
            
            # Stream validation results
            for result in results:
                score = 1.0 if result.passed else 0.0
                
                await self.stream_quality_score(
                    metric_name=result.rule_name,
                    score=score,
                    threshold=0.5,
                    dataset=dataset_name
                )
            
            # Check for anomalies
            anomaly_detector = get_anomaly_detector()
            anomalies = anomaly_detector.detect_anomalies([data], dataset_name)
            
            if anomalies:
                await self.stream_quality_score(
                    metric_name="anomaly_detection",
                    score=0.0,
                    threshold=0.5,
                    dataset=dataset_name
                )
            
        except Exception as e:
            logger.error(f"Error monitoring data quality: {e}")


class MarketDataIntegration:
    """
    Integration with market data sources.
    
    Streams live price updates and news.
    """
    
    def __init__(self, streaming_service: StreamingService):
        self.streaming = streaming_service
        self._price_tasks: Dict[str, asyncio.Task] = {}
    
    async def stream_price_updates(self, symbols: list[str], interval: float = 1.0):
        """
        Stream live price updates for symbols.
        
        Args:
            symbols: List of stock symbols
            interval: Update interval in seconds
        """
        for symbol in symbols:
            if symbol not in self._price_tasks:
                task = asyncio.create_task(
                    self._price_update_loop(symbol, interval)
                )
                self._price_tasks[symbol] = task
    
    async def _price_update_loop(self, symbol: str, interval: float):
        """Background loop for price updates."""
        try:
            import yfinance as yf
            
            ticker = yf.Ticker(symbol)
            
            while True:
                try:
                    # Get current price
                    info = ticker.info
                    current_price = info.get('currentPrice') or info.get('regularMarketPrice')
                    volume = info.get('volume', 0)
                    
                    if current_price:
                        # Stream price update
                        await self.streaming.publish_price_update(
                            symbol=symbol,
                            price=float(current_price),
                            volume=int(volume)
                        )
                    
                except Exception as e:
                    logger.warning(f"Error fetching price for {symbol}: {e}")
                
                await asyncio.sleep(interval)
                
        except asyncio.CancelledError:
            logger.info(f"Price streaming stopped for {symbol}")
        except Exception as e:
            logger.error(f"Fatal error in price streaming for {symbol}: {e}")
    
    def stop_price_updates(self, symbol: Optional[str] = None):
        """Stop price update streaming."""
        if symbol:
            if symbol in self._price_tasks:
                self._price_tasks[symbol].cancel()
                del self._price_tasks[symbol]
        else:
            # Stop all
            for task in self._price_tasks.values():
                task.cancel()
            self._price_tasks.clear()
    
    async def stream_news_alert(self, title: str, summary: str, url: str, sentiment: str = "neutral"):
        """Stream a news alert."""
        await self.streaming.publish_news(
            title=title,
            summary=summary,
            url=url
        )


class RAGIntegration:
    """
    Integration with RAG system.
    
    Streams RAG query results and analysis.
    """
    
    def __init__(self, streaming_service: StreamingService):
        self.streaming = streaming_service
    
    async def stream_rag_query(self, query: str, answer: str, confidence: float, sources: list):
        """
        Stream RAG query result.
        
        Args:
            query: User query
            answer: Generated answer
            confidence: Confidence score
            sources: List of source documents
        """
        await self.streaming.publish_analysis(
            query=query,
            answer=answer,
            confidence=confidence,
            reasoning=[f"Source: {s.get('title', 'Unknown')}" for s in sources[:3]]
        )


class IntegratedStreamingPlatform:
    """
    Complete integrated streaming platform.
    
    Combines all integrations into a single interface.
    """
    
    def __init__(self, streaming_service: Optional[StreamingService] = None):
        """Initialize integrated platform."""
        self.streaming = streaming_service or StreamingService()
        
        # Initialize integrations
        self.langgraph = LangGraphIntegration(self.streaming)
        self.neo4j = Neo4jIntegration(self.streaming)
        self.quality = QualityMetricsIntegration(self.streaming)
        self.market = MarketDataIntegration(self.streaming)
        self.rag = RAGIntegration(self.streaming)
        
        logger.info("Integrated streaming platform initialized")
    
    async def start_full_monitoring(self, symbols: list[str]):
        """
        Start comprehensive monitoring.
        
        Args:
            symbols: Stock symbols to monitor
        """
        # Start price streaming
        await self.market.stream_price_updates(symbols, interval=5.0)
        
        logger.info(f"Full monitoring started for: {symbols}")
    
    def stop_monitoring(self):
        """Stop all monitoring."""
        self.market.stop_price_updates()
        logger.info("Monitoring stopped")


__all__ = [
    "LangGraphIntegration",
    "Neo4jIntegration",
    "QualityMetricsIntegration",
    "MarketDataIntegration",
    "RAGIntegration",
    "IntegratedStreamingPlatform"
]