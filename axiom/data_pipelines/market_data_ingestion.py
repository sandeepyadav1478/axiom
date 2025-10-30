"""
Market Data Ingestion Pipeline

Real-time and batch data ingestion for:
- Price data (stocks, options, bonds)
- Market data (volumes, spreads)
- News feeds
- Alternative data sources

Feeds into Feature Store and ML models.
"""

from typing import Dict, List, Optional
from datetime import datetime
import asyncio

try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class MarketDataPipeline:
    """
    Real-time market data ingestion pipeline
    
    Ingests data from multiple sources and feeds models.
    """
    
    def __init__(self):
        self.data_sources = []
        self.subscribers = []
    
    async def ingest_real_time_prices(self, symbol: str):
        """
        Ingest real-time price data
        
        In production, connects to:
        - Bloomberg Terminal
        - Interactive Brokers
        - Polygon.io
        - Alpha Vantage
        """
        # Simulated real-time data
        while True:
            price_data = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'price': np.random.uniform(99, 101),
                'volume': np.random.randint(1000, 10000),
                'bid': np.random.uniform(99, 100),
                'ask': np.random.uniform(100, 101)
            }
            
            # Notify subscribers (ML models, dashboards, etc.)
            await self._notify_subscribers(price_data)
            
            await asyncio.sleep(1)  # 1-second updates
    
    async def _notify_subscribers(self, data: Dict):
        """Notify all subscribers of new data"""
        for subscriber in self.subscribers:
            try:
                await subscriber(data)
            except Exception as e:
                print(f"Subscriber error: {e}")
    
    def subscribe(self, callback):
        """Subscribe to data updates"""
        self.subscribers.append(callback)
    
    def ingest_batch_historical(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Batch ingest historical data
        
        For backtesting and model training.
        """
        if not PANDAS_AVAILABLE:
            return None
        
        # Would fetch from data source
        # For now, generate sample data
        dates = pd.date_range(start_date, end_date, freq='D')
        
        data = {}
        for symbol in symbols:
            prices = 100 * np.exp(np.cumsum(np.random.randn(len(dates)) * 0.02))
            data[symbol] = prices
        
        df = pd.DataFrame(data, index=dates)
        
        return df
    
    def process_news_feed(self, news_item: Dict):
        """
        Process incoming news for sentiment analysis
        
        Feeds M&A sentiment models, credit models, etc.
        """
        # Extract relevant info
        processed = {
            'timestamp': news_item.get('timestamp', datetime.now()),
            'company': news_item.get('company', ''),
            'headline': news_item.get('headline', ''),
            'sentiment': self._quick_sentiment(news_item.get('headline', '')),
            'relevance': self._calculate_relevance(news_item)
        }
        
        return processed
    
    def _quick_sentiment(self, text: str) -> float:
        """Quick sentiment scoring"""
        positive_words = ['growth', 'profit', 'success', 'strong']
        negative_words = ['loss', 'decline', 'risk', 'weak']
        
        text_lower = text.lower()
        pos_count = sum(word in text_lower for word in positive_words)
        neg_count = sum(word in text_lower for word in negative_words)
        
        if pos_count + neg_count > 0:
            return (pos_count - neg_count) / (pos_count + neg_count)
        return 0.0
    
    def _calculate_relevance(self, news_item: Dict) -> float:
        """Calculate relevance score"""
        # Based on source, company, keywords
        return 0.8  # Placeholder


if __name__ == "__main__":
    print("Market Data Ingestion Pipeline")
    
    pipeline = MarketDataPipeline()
    
    # Batch historical
    if PANDAS_AVAILABLE:
        historical = pipeline.ingest_batch_historical(
            ['AAPL', 'MSFT'],
            '2024-01-01',
            '2024-10-29'
        )
        print(f"Historical data: {historical.shape}")
    
    # Process news
    news = {
        'company': 'AAPL',
        'headline': 'Apple reports strong quarterly growth',
        'timestamp': datetime.now()
    }
    
    processed = pipeline.process_news_feed(news)
    print(f"News sentiment: {processed['sentiment']:.2f}")
    
    print("\n✓ Data pipeline ready to feed 60 ML models")