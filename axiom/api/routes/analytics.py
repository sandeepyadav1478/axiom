"""
Analytics API endpoints.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from axiom.api.auth import get_current_user, User

router = APIRouter(prefix="/analytics", tags=["Analytics"])


class AnalyticsResponse(BaseModel):
    """Generic analytics response."""
    result: dict
    execution_time_ms: float


@router.get("/summary")
async def get_analytics_summary(
    current_user: User = Depends(get_current_user),
):
    """Get analytics summary."""
    return {"message": "Analytics summary endpoint"}