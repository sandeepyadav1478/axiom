"""
Market Data API endpoints.
"""

from typing import List, Dict
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from axiom.api.auth import get_current_user, User, get_optional_user
from axiom.api.rate_limit import limiter, high_rate_limit

router = APIRouter(prefix="/market-data", tags=["Market Data"])


class QuoteResponse(BaseModel):
    """Market quote response."""
    symbol: str
    price: float
    bid: float
    ask: float
    volume: int
    timestamp: float


class HistoricalDataResponse(BaseModel):
    """Historical data response."""
    symbol: str
    data: List[Dict[str, float]]
    period: str


@router.get("/quote/{symbol}", response_model=QuoteResponse)
@high_rate_limit
async def get_quote(
    symbol: str,
    current_user: User = Depends(get_optional_user),
):
    """Get real-time quote for symbol."""
    return QuoteResponse(
        symbol=symbol,
        price=100.0,
        bid=99.95,
        ask=100.05,
        volume=1000000,
        timestamp=1234567890.0,
    )


@router.get("/historical/{symbol}", response_model=HistoricalDataResponse)
@high_rate_limit
async def get_historical_data(
    symbol: str,
    period: str = "1y",
    current_user: User = Depends(get_current_user),
):
    """Get historical price data."""
    return HistoricalDataResponse(
        symbol=symbol,
        data=[{"date": "2024-01-01", "price": 100.0}],
        period=period,
    )