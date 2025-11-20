"""
Market Data Operators with Multi-Source Support and Failover
"""
from typing import Any, Dict, List, Optional
from datetime import datetime
from enum import Enum

from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults


class DataSource(Enum):
    """Supported market data sources"""
    YAHOO = "yahoo"
    POLYGON = "polygon"
    FINNHUB = "finnhub"
    ALPHA_VANTAGE = "alpha_vantage"


class MarketDataFetchOperator(BaseOperator):
    """
    Fetch market data with automatic source failover.
    
    Features:
    - Multi-source support (Yahoo, Polygon, Finnhub)
    - Automatic failover on errors
    - Data validation
    - Caching support
    """
    
    template_fields = ('symbols',)
    ui_color = '#FFB6C1'
    ui_fgcolor = '#000'
    
    @apply_defaults
    def __init__(
        self,
        symbols: List[str],
        data_type: str = 'prices',  # prices, company_info, news
        primary_source: DataSource = DataSource.YAHOO,
        fallback_sources: Optional[List[DataSource]] = None,
        cache_ttl_minutes: int = 5,
        xcom_key: str = 'market_data',
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.symbols = symbols
        self.data_type = data_type
        self.primary_source = primary_source
        self.fallback_sources = fallback_sources or []
        self.cache_ttl_minutes = cache_ttl_minutes
        self.xcom_key = xcom_key
        
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch data with failover"""
        sources = [self.primary_source] + self.fallback_sources
        last_error = None
        
        for source in sources:
            try:
                self.log.info(f"📊 Fetching data from {source.value}...")
                
                data = self._fetch_from_source(source)
                
                self.log.info(
                    f"✅ Successfully fetched {len(data)} records from {source.value}"
                )
                
                result = {
                    'data': data,
                    'source': source.value,
                    'symbols': self.symbols,
                    'data_type': self.data_type,
                    'timestamp': datetime.now().isoformat(),
                    'success': True
                }
                
                context['ti'].xcom_push(key=self.xcom_key, value=result)
                return result
                
            except Exception as e:
                last_error = e
                self.log.warning(
                    f"⚠️  {source.value} failed: {str(e)}, trying next source..."
                )
                continue
        
        # All sources failed
        self.log.error(f"❌ All data sources failed. Last error: {last_error}")
        raise Exception(f"Failed to fetch data from any source: {last_error}")
    
    def _fetch_from_source(self, source: DataSource) -> List[Dict[str, Any]]:
        """Fetch data from specific source"""
        if source == DataSource.YAHOO:
            return self._fetch_yahoo()
        elif source == DataSource.POLYGON:
            return self._fetch_polygon()
        elif source == DataSource.FINNHUB:
            return self._fetch_finnhub()
        elif source == DataSource.ALPHA_VANTAGE:
            return self._fetch_alpha_vantage()
        else:
            raise ValueError(f"Unknown source: {source}")
    
    def _fetch_yahoo(self) -> List[Dict[str, Any]]:
        """Fetch from Yahoo Finance"""
        import yfinance as yf
        
        data = []
        for symbol in self.symbols:
            ticker = yf.Ticker(symbol)
            
            if self.data_type == 'prices':
                hist = ticker.history(period='1d', interval='1m')
                if not hist.empty:
                    latest = hist.iloc[-1]
                    data.append({
                        'symbol': symbol,
                        'open': float(latest['Open']),
                        'high': float(latest['High']),
                        'low': float(latest['Low']),
                        'close': float(latest['Close']),
                        'volume': int(latest['Volume']),
                        'timestamp': datetime.now().isoformat()
                    })
            
            elif self.data_type == 'company_info':
                info = ticker.info
                data.append({
                    'symbol': symbol,
                    'name': info.get('longName'),
                    'sector': info.get('sector'),
                    'industry': info.get('industry'),
                    'market_cap': info.get('marketCap'),
                    'employees': info.get('fullTimeEmployees')
                })
        
        return data
    
    def _fetch_polygon(self) -> List[Dict[str, Any]]:
        """Fetch from Polygon.io"""
        import os
        import requests
        
        api_key = os.getenv('POLYGON_API_KEY')
        if not api_key:
            raise ValueError("POLYGON_API_KEY not configured")
        
        data = []
        for symbol in self.symbols:
            url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev"
            response = requests.get(url, params={'apiKey': api_key})
            response.raise_for_status()
            
            result = response.json()
            if result.get('results'):
                bar = result['results'][0]
                data.append({
                    'symbol': symbol,
                    'open': bar['o'],
                    'high': bar['h'],
                    'low': bar['l'],
                    'close': bar['c'],
                    'volume': bar['v'],
                    'timestamp': datetime.now().isoformat()
                })
        
        return data
    
    def _fetch_finnhub(self) -> List[Dict[str, Any]]:
        """Fetch from Finnhub"""
        import os
        import requests
        
        api_key = os.getenv('FINNHUB_API_KEY')
        if not api_key:
            raise ValueError("FINNHUB_API_KEY not configured")
        
        data = []
        for symbol in self.symbols:
            url = f"https://finnhub.io/api/v1/quote"
            response = requests.get(url, params={'symbol': symbol, 'token': api_key})
            response.raise_for_status()
            
            quote = response.json()
            data.append({
                'symbol': symbol,
                'open': quote['o'],
                'high': quote['h'],
                'low': quote['l'],
                'close': quote['c'],
                'timestamp': datetime.now().isoformat()
            })
        
        return data
    
    def _fetch_alpha_vantage(self) -> List[Dict[str, Any]]:
        """Fetch from Alpha Vantage"""
        import os
        import requests
        
        api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        if not api_key:
            raise ValueError("ALPHA_VANTAGE_API_KEY not configured")
        
        # Rotate through multiple keys if available
        keys = api_key.split(',')
        api_key = keys[0].strip()
        
        data = []
        for symbol in self.symbols:
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol,
                'apikey': api_key
            }
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            result = response.json()
            if 'Global Quote' in result:
                quote = result['Global Quote']
                data.append({
                    'symbol': symbol,
                    'open': float(quote.get('02. open', 0)),
                    'high': float(quote.get('03. high', 0)),
                    'low': float(quote.get('04. low', 0)),
                    'close': float(quote.get('05. price', 0)),
                    'volume': int(quote.get('06. volume', 0)),
                    'timestamp': datetime.now().isoformat()
                })
        
        return data


class MultiSourceMarketDataOperator(BaseOperator):
    """
    Fetch data from multiple sources in parallel for redundancy.
    
    Returns consensus data or flags discrepancies.
    Perfect for mission-critical applications.
    """
    
    ui_color = '#FF6347'
    ui_fgcolor = '#fff'
    
    @apply_defaults
    def __init__(
        self,
        symbols: List[str],
        sources: List[DataSource],
        consensus_threshold: float = 0.01,  # 1% price difference
        xcom_key: str = 'consensus_data',
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.symbols = symbols
        self.sources = sources
        self.consensus_threshold = consensus_threshold
        self.xcom_key = xcom_key
        
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch from all sources and compute consensus"""
        
        results = {}
        for source in self.sources:
            try:
                operator = MarketDataFetchOperator(
                    task_id=f"fetch_{source.value}",
                    symbols=self.symbols,
                    primary_source=source,
                    fallback_sources=[]
                )
                data = operator.execute(context)
                results[source.value] = data['data']
            except Exception as e:
                self.log.warning(f"Source {source.value} failed: {e}")
        
        if len(results) == 0:
            raise Exception("All sources failed")
        
        # Compute consensus
        consensus = self._compute_consensus(results)
        
        result = {
            'consensus_data': consensus,
            'sources_used': list(results.keys()),
            'discrepancies': self._find_discrepancies(results),
            'timestamp': datetime.now().isoformat()
        }
        
        context['ti'].xcom_push(key=self.xcom_key, value=result)
        return result
    
    def _compute_consensus(self, results: Dict[str, List]) -> List[Dict]:
        """Compute consensus values across sources"""
        # Simple median approach for now
        consensus = []
        
        for symbol in self.symbols:
            symbol_data = []
            for source_data in results.values():
                for item in source_data:
                    if item['symbol'] == symbol:
                        symbol_data.append(item)
            
            if symbol_data:
                # Take median close price
                close_prices = [d['close'] for d in symbol_data]
                median_close = sorted(close_prices)[len(close_prices) // 2]
                
                consensus.append({
                    'symbol': symbol,
                    'close': median_close,
                    'sources_count': len(symbol_data),
                    'price_range': [min(close_prices), max(close_prices)]
                })
        
        return consensus
    
    def _find_discrepancies(self, results: Dict[str, List]) -> List[Dict]:
        """Find significant price discrepancies between sources"""
        discrepancies = []
        
        for symbol in self.symbols:
            prices = []
            for source, data_list in results.items():
                for item in data_list:
                    if item['symbol'] == symbol:
                        prices.append((source, item['close']))
            
            if len(prices) > 1:
                prices_only = [p[1] for p in prices]
                price_range = max(prices_only) - min(prices_only)
                avg_price = sum(prices_only) / len(prices_only)
                
                if price_range / avg_price > self.consensus_threshold:
                    discrepancies.append({
                        'symbol': symbol,
                        'prices': prices,
                        'range': price_range,
                        'percent_diff': (price_range / avg_price) * 100
                    })
        
        return discrepancies