"""
Market Data Domain Value Objects

Immutable value objects for market data domain.
Following DDD principles - these capture quotes, chains, and NBBO.

Key concepts:
- Value objects are immutable (frozen dataclasses)
- Self-validating (fail fast on invalid data)
- Rich behavior (NBBO calculation, spread analysis, quality checks)
- Type-safe (using Decimal for prices, Enum for types)

These represent market data as first-class domain concepts.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from datetime import datetime
from typing import List, Optional, Dict, Tuple
from enum import Enum


class DataSource(str, Enum):
    """Market data sources"""
    OPRA = "opra"
    CBOE = "cboe"
    ISE = "ise"
    POLYGON = "polygon"
    IEX = "iex"
    CACHE = "cache"


class DataQuality(str, Enum):
    """Data quality level"""
    EXCELLENT = "excellent"  # <1ms latency, validated
    GOOD = "good"  # <10ms, validated
    ACCEPTABLE = "acceptable"  # <100ms
    POOR = "poor"  # >100ms or stale


@dataclass(frozen=True)
class OptionQuote:
    """
    Immutable option quote
    
    Single option quote with validation
    """
    symbol: str
    underlying: str
    strike: Decimal
    expiry: datetime
    option_type: str  # 'call' or 'put'
    
    # Pricing
    bid: Decimal
    ask: Decimal
    last: Decimal
    
    # Volume
    volume: int
    open_interest: int
    
    # Greeks (from exchange if available)
    implied_vol: Optional[Decimal] = None
    delta: Optional[Decimal] = None
    gamma: Optional[Decimal] = None
    theta: Optional[Decimal] = None
    vega: Optional[Decimal] = None
    
    # Metadata
    source: DataSource = DataSource.OPRA
    timestamp: datetime = field(default_factory=datetime.utcnow)
    latency_microseconds: Decimal = Decimal('0')
    
    def __post_init__(self):
        """Validate quote"""
        if self.option_type not in ['call', 'put']:
            raise ValueError(f"Invalid option type: {self.option_type}")
        
        if self.strike <= Decimal('0'):
            raise ValueError("Strike must be positive")
        
        if self.bid < Decimal('0') or self.ask < Decimal('0'):
            raise ValueError("Prices must be non-negative")
        
        if self.bid > self.ask:
            raise ValueError(f"Bid ({self.bid}) cannot exceed ask ({self.ask})")
        
        if self.volume < 0 or self.open_interest < 0:
            raise ValueError("Volume and open interest must be non-negative")
    
    def get_mid_price(self) -> Decimal:
        """Calculate mid price"""
        return (self.bid + self.ask) / Decimal('2')
    
    def get_spread(self) -> Decimal:
        """Calculate bid-ask spread"""
        return self.ask - self.bid
    
    def get_spread_bps(self) -> Decimal:
        """Calculate spread in basis points"""
        mid = self.get_mid_price()
        if mid > Decimal('0'):
            return (self.get_spread() / mid) * Decimal('10000')
        return Decimal('0')
    
    def is_tight_market(self, max_spread_bps: Decimal = Decimal('50')) -> bool:
        """Check if market has tight spread"""
        return self.get_spread_bps() <= max_spread_bps
    
    def has_liquidity(self, min_volume: int = 100) -> bool:
        """Check if option has sufficient liquidity"""
        return self.volume >= min_volume or self.open_interest >= min_volume * 10
    
    def is_fresh(self, max_age_seconds: int = 1) -> bool:
        """Check if quote is fresh"""
        age = (datetime.utcnow() - self.timestamp).total_seconds()
        return age <= max_age_seconds


@dataclass(frozen=True)
class OptionsChain:
    """
    Complete options chain
    
    Immutable collection of all options for an underlying
    """
    underlying: str
    spot_price: Decimal
    quotes: Tuple[OptionQuote, ...]
    
    # Metadata
    source: DataSource
    timestamp: datetime = field(default_factory=datetime.utcnow)
    retrieval_time_ms: Decimal = Decimal('0')
    
    def __post_init__(self):
        """Validate chain"""
        if len(self.quotes) == 0:
            raise ValueError("Options chain cannot be empty")
        
        if self.spot_price <= Decimal('0'):
            raise ValueError("Spot price must be positive")
        
        # All quotes should be for same underlying
        underlyings = set(q.underlying for q in self.quotes)
        if len(underlyings) > 1:
            raise ValueError(f"Mixed underlyings in chain: {underlyings}")
    
    def get_quote_count(self) -> int:
        """Get total number of quotes"""
        return len(self.quotes)
    
    def get_expiries(self) -> List[datetime]:
        """Get all unique expiry dates"""
        return sorted(list(set(q.expiry for q in self.quotes)))
    
    def get_strikes(self) -> List[Decimal]:
        """Get all unique strike prices"""
        return sorted(list(set(q.strike for q in self.quotes)))
    
    def get_calls(self) -> Tuple[OptionQuote, ...]:
        """Get all call options"""
        return tuple(q for q in self.quotes if q.option_type == 'call')
    
    def get_puts(self) -> Tuple[OptionQuote, ...]:
        """Get all put options"""
        return tuple(q for q in self.quotes if q.option_type == 'put')
    
    def get_atm_strike(self) -> Decimal:
        """Get at-the-money strike"""
        strikes = self.get_strikes()
        # Find strike closest to spot
        return min(strikes, key=lambda s: abs(s - self.spot_price))
    
    def get_total_volume(self) -> int:
        """Get total volume across chain"""
        return sum(q.volume for q in self.quotes)
    
    def get_total_open_interest(self) -> int:
        """Get total open interest"""
        return sum(q.open_interest for q in self.quotes)


@dataclass(frozen=True)
class NBBO:
    """
    National Best Bid and Offer
    
    Immutable NBBO calculation (regulatory requirement)
    """
    symbol: str
    
    # Best prices
    best_bid: Decimal
    best_ask: Decimal
    best_bid_size: int
    best_ask_size: int
    
    # Source venues
    bid_venue: DataSource
    ask_venue: DataSource
    
    # Quotes used (for audit)
    venue_quotes: Tuple[OptionQuote, ...]
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    calculation_time_microseconds: Decimal = Decimal('0')
    
    def __post_init__(self):
        """Validate NBBO"""
        if self.best_bid < Decimal('0') or self.best_ask < Decimal('0'):
            raise ValueError("NBBO prices must be non-negative")
        
        if self.best_bid > self.best_ask:
            raise ValueError(f"NBBO bid ({self.best_bid}) cannot exceed ask ({self.best_ask})")
        
        if len(self.venue_quotes) == 0:
            raise ValueError("NBBO must be calculated from at least one quote")
    
    def get_mid_price(self) -> Decimal:
        """Calculate NBBO mid price"""
        return (self.best_bid + self.best_ask) / Decimal('2')
    
    def get_spread(self) -> Decimal:
        """Calculate NBBO spread"""
        return self.best_ask - self.best_bid
    
    def get_spread_bps(self) -> Decimal:
        """Calculate NBBO spread in basis points"""
        mid = self.get_mid_price()
        if mid > Decimal('0'):
            return (self.get_spread() / mid) * Decimal('10000')
        return Decimal('0')
    
    def is_locked_or_crossed(self) -> bool:
        """Check if market is locked (bid = ask) or crossed (bid > ask)"""
        return self.best_bid >= self.best_ask
    
    def has_sufficient_size(self, min_size: int = 10) -> bool:
        """Check if NBBO has sufficient size"""
        return self.best_bid_size >= min_size and self.best_ask_size >= min_size


@dataclass(frozen=True)
class MarketDataStatistics:
    """
    Aggregated market data statistics
    
    Immutable performance metrics for data feeds
    """
    total_requests: int
    successful_requests: int
    failed_requests: int
    
    # Cache performance
    cache_hits: int
    cache_misses: int
    cache_hit_rate: Decimal
    
    # Latency
    average_latency_ms: Decimal
    p50_latency_ms: Decimal
    p95_latency_ms: Decimal
    p99_latency_ms: Decimal
    
    # Data quality
    average_quality_score: Decimal
    stale_data_count: int
    validation_failures: int
    
    # Source distribution
    source_distribution: Dict[DataSource, int]
    failover_count: int
    
    # Time period
    start_time: datetime
    end_time: datetime
    
    def __post_init__(self):
        """Validate statistics"""
        if self.total_requests < 0:
            raise ValueError("Total requests must be non-negative")
        
        if not (Decimal('0') <= self.cache_hit_rate <= Decimal('1')):
            raise ValueError("Cache hit rate must be between 0 and 1")
    
    def get_success_rate(self) -> Decimal:
        """Calculate request success rate"""
        if self.total_requests > 0:
            return Decimal(str(self.successful_requests)) / Decimal(str(self.total_requests))
        return Decimal('0')
    
    def is_high_performance(
        self,
        min_cache_hit: Decimal = Decimal('0.80'),
        max_p95_latency: Decimal = Decimal('10.0')
    ) -> bool:
        """Check if data feed is high performance"""
        return (
            self.cache_hit_rate >= min_cache_hit and
            self.p95_latency_ms <= max_p95_latency
        )


# Example usage
if __name__ == "__main__":
    from axiom.ai_layer.infrastructure.observability import Logger
    
    logger = Logger("market_data_domain_test")
    
    logger.info("test_starting", test="MARKET DATA DOMAIN VALUE OBJECTS")
    
    # Create option quote
    logger.info("creating_option_quote")
    
    quote = OptionQuote(
        symbol="SPY241115C00450000",
        underlying="SPY",
        strike=Decimal('450.00'),
        expiry=datetime(2024, 11, 15),
        option_type='call',
        bid=Decimal('5.48'),
        ask=Decimal('5.52'),
        last=Decimal('5.50'),
        volume=1250,
        open_interest=5800,
        implied_vol=Decimal('0.18'),
        delta=Decimal('0.55'),
        source=DataSource.OPRA,
        latency_microseconds=Decimal('85')
    )
    
    logger.info(
        "quote_created",
        symbol=quote.symbol,
        mid_price=float(quote.get_mid_price()),
        spread_bps=float(quote.get_spread_bps()),
        tight_market=quote.is_tight_market(),
        has_liquidity=quote.has_liquidity()
    )
    
    # Create options chain
    logger.info("creating_options_chain")
    
    chain = OptionsChain(
        underlying="SPY",
        spot_price=Decimal('450.00'),
        quotes=(quote,),  # Would have many more in production
        source=DataSource.OPRA,
        retrieval_time_ms=Decimal('4.2')
    )
    
    logger.info(
        "chain_created",
        underlying=chain.underlying,
        quote_count=chain.get_quote_count(),
        total_volume=chain.get_total_volume(),
        total_oi=chain.get_total_open_interest()
    )
    
    # Create NBBO
    logger.info("creating_nbbo")
    
    nbbo = NBBO(
        symbol="SPY241115C00450000",
        best_bid=Decimal('5.48'),
        best_ask=Decimal('5.52'),
        best_bid_size=150,
        best_ask_size=200,
        bid_venue=DataSource.CBOE,
        ask_venue=DataSource.ISE,
        venue_quotes=(quote,),
        calculation_time_microseconds=Decimal('12')
    )
    
    logger.info(
        "nbbo_created",
        best_bid=float(nbbo.best_bid),
        best_ask=float(nbbo.best_ask),
        mid=float(nbbo.get_mid_price()),
        spread_bps=float(nbbo.get_spread_bps()),
        locked_or_crossed=nbbo.is_locked_or_crossed()
    )
    
    # Create statistics
    logger.info("creating_statistics")
    
    stats = MarketDataStatistics(
        total_requests=10000,
        successful_requests=9985,
        failed_requests=15,
        cache_hits=7800,
        cache_misses=2200,
        cache_hit_rate=Decimal('0.78'),
        average_latency_ms=Decimal('1.2'),
        p50_latency_ms=Decimal('0.8'),
        p95_latency_ms=Decimal('3.5'),
        p99_latency_ms=Decimal('8.2'),
        average_quality_score=Decimal('0.95'),
        stale_data_count=5,
        validation_failures=2,
        source_distribution={DataSource.OPRA: 8500, DataSource.POLYGON: 1200, DataSource.IEX: 285},
        failover_count=12,
        start_time=datetime(2024, 10, 1),
        end_time=datetime(2024, 10, 30)
    )
    
    logger.info(
        "statistics_created",
        success_rate=float(stats.get_success_rate()),
        cache_hit_rate=float(stats.cache_hit_rate),
        high_performance=stats.is_high_performance()
    )
    
    logger.info(
        "test_complete",
        artifacts_created=[
            "Immutable market data objects",
            "Self-validating quotes",
            "Rich domain behavior",
            "Type-safe with Decimal",
            "NBBO calculation",
            "Quality tracking",
            "Proper logging (no print)"
        ]
    )