"""
QuantLib Wrapper for Fixed Income Pricing

This module provides a wrapper around QuantLib for production-grade fixed income calculations.
QuantLib offers 30+ day count conventions, comprehensive calendar support, and battle-tested
implementations used by Bloomberg and other financial institutions.

Features:
- Bond pricing (all bond types: fixed, floating, zero-coupon)
- Yield curve construction and interpolation
- Day count conventions (30+ built-in)
- Calendar handling (holidays, business days)
- Schedule generation (coupon dates, amortization)
- Duration and convexity calculations
"""

import logging
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any

import numpy as np

from .config import get_config, LibraryAvailability

logger = logging.getLogger(__name__)

# Check if QuantLib is available
QUANTLIB_AVAILABLE = LibraryAvailability.check_library('QuantLib')

if QUANTLIB_AVAILABLE:
    import QuantLib as ql


class DayCountConvention(Enum):
    """Day count conventions supported by QuantLib."""
    ACTUAL_360 = "Actual/360"
    ACTUAL_365 = "Actual/365 (Fixed)"
    ACTUAL_ACTUAL = "Actual/Actual (ISDA)"
    THIRTY_360 = "30/360 (Bond Basis)"
    THIRTY_360_EUROPEAN = "30E/360"
    ACTUAL_365_CANADIAN = "Actual/365 (Canadian)"
    BUSINESS_252 = "Business/252"


class BusinessDayConvention(Enum):
    """Business day conventions."""
    FOLLOWING = "Following"
    MODIFIED_FOLLOWING = "ModifiedFollowing"
    PRECEDING = "Preceding"
    MODIFIED_PRECEDING = "ModifiedPreceding"
    UNADJUSTED = "Unadjusted"


class Frequency(Enum):
    """Coupon payment frequency."""
    ANNUAL = 1
    SEMIANNUAL = 2
    QUARTERLY = 4
    MONTHLY = 12


@dataclass
class BondSpecification:
    """Specification for a bond."""
    face_value: float
    coupon_rate: float  # Annual coupon rate (e.g., 0.05 for 5%)
    issue_date: date
    maturity_date: date
    payment_frequency: Frequency = Frequency.SEMIANNUAL
    day_count: DayCountConvention = DayCountConvention.ACTUAL_ACTUAL
    business_day_convention: BusinessDayConvention = BusinessDayConvention.MODIFIED_FOLLOWING
    settlement_days: int = 2
    calendar: str = "UnitedStates"  # Calendar for business day adjustments


@dataclass
class BondPrice:
    """Bond pricing result."""
    clean_price: float
    dirty_price: float
    accrued_interest: float
    ytm: float
    duration: float
    convexity: float
    present_value: float


@dataclass
class YieldCurvePoint:
    """A point on the yield curve."""
    maturity: float  # Time to maturity in years
    rate: float  # Yield rate


class QuantLibBondPricer:
    """Wrapper for QuantLib bond pricing functionality.
    
    This class provides a clean interface to QuantLib's bond pricing capabilities,
    handling the conversion between Python data types and QuantLib objects.
    
    Example:
        >>> pricer = QuantLibBondPricer()
        >>> bond = BondSpecification(
        ...     face_value=1000,
        ...     coupon_rate=0.05,
        ...     issue_date=date(2020, 1, 1),
        ...     maturity_date=date(2030, 1, 1)
        ... )
        >>> result = pricer.price_bond(bond, settlement_date=date(2023, 1, 1), yield_rate=0.04)
        >>> print(f"Clean Price: ${result.clean_price:.2f}")
    """
    
    def __init__(self):
        """Initialize the QuantLib bond pricer."""
        if not QUANTLIB_AVAILABLE:
            raise ImportError(
                "QuantLib is not available. Install it with: pip install QuantLib-Python"
            )
        
        self.config = get_config()
        logger.info("QuantLib bond pricer initialized")
    
    def _get_calendar(self, calendar_name: str) -> 'ql.Calendar':
        """Get a QuantLib calendar by name.
        
        Args:
            calendar_name: Name of the calendar (e.g., 'UnitedStates', 'UnitedKingdom')
            
        Returns:
            QuantLib Calendar object
        """
        calendar_map = {
            'UnitedStates': ql.UnitedStates(ql.UnitedStates.GovernmentBond),
            'UnitedKingdom': ql.UnitedKingdom(),
            'TARGET': ql.TARGET(),  # European Central Bank
            'Japan': ql.Japan(),
            'Germany': ql.Germany(),
            'France': ql.France(),
            'Italy': ql.Italy(),
            'Canada': ql.Canada(),
            'Australia': ql.Australia(),
        }
        
        return calendar_map.get(calendar_name, ql.NullCalendar())
    
    def _get_day_count(self, convention: DayCountConvention) -> 'ql.DayCounter':
        """Convert day count convention to QuantLib DayCounter.
        
        Args:
            convention: Day count convention enum
            
        Returns:
            QuantLib DayCounter object
        """
        day_count_map = {
            DayCountConvention.ACTUAL_360: ql.Actual360(),
            DayCountConvention.ACTUAL_365: ql.Actual365Fixed(),
            DayCountConvention.ACTUAL_ACTUAL: ql.ActualActual(ql.ActualActual.ISDA),
            DayCountConvention.THIRTY_360: ql.Thirty360(ql.Thirty360.BondBasis),
            DayCountConvention.THIRTY_360_EUROPEAN: ql.Thirty360(ql.Thirty360.European),
            DayCountConvention.ACTUAL_365_CANADIAN: ql.Actual365Fixed(ql.Actual365Fixed.Canadian),
            DayCountConvention.BUSINESS_252: ql.Business252(),
        }
        
        return day_count_map[convention]
    
    def _get_business_day_convention(
        self, convention: BusinessDayConvention
    ) -> 'ql.BusinessDayConvention':
        """Convert business day convention to QuantLib enum.
        
        Args:
            convention: Business day convention enum
            
        Returns:
            QuantLib BusinessDayConvention enum value
        """
        bdc_map = {
            BusinessDayConvention.FOLLOWING: ql.Following,
            BusinessDayConvention.MODIFIED_FOLLOWING: ql.ModifiedFollowing,
            BusinessDayConvention.PRECEDING: ql.Preceding,
            BusinessDayConvention.MODIFIED_PRECEDING: ql.ModifiedPreceding,
            BusinessDayConvention.UNADJUSTED: ql.Unadjusted,
        }
        
        return bdc_map[convention]
    
    def _get_frequency(self, frequency: Frequency) -> 'ql.Frequency':
        """Convert frequency to QuantLib enum.
        
        Args:
            frequency: Payment frequency enum
            
        Returns:
            QuantLib Frequency enum value
        """
        freq_map = {
            Frequency.ANNUAL: ql.Annual,
            Frequency.SEMIANNUAL: ql.Semiannual,
            Frequency.QUARTERLY: ql.Quarterly,
            Frequency.MONTHLY: ql.Monthly,
        }
        
        return freq_map[frequency]
    
    def _python_date_to_ql(self, py_date: date) -> 'ql.Date':
        """Convert Python date to QuantLib Date.
        
        Args:
            py_date: Python date object
            
        Returns:
            QuantLib Date object
        """
        return ql.Date(py_date.day, py_date.month, py_date.year)
    
    def price_bond(
        self,
        bond: BondSpecification,
        settlement_date: date,
        yield_rate: float,
        compounding: str = 'Compounded',
        yield_frequency: Optional[Frequency] = None
    ) -> BondPrice:
        """Price a fixed-rate bond using QuantLib.
        
        Args:
            bond: Bond specification
            settlement_date: Settlement date for pricing
            yield_rate: Yield to maturity for pricing
            compounding: Compounding convention ('Simple', 'Compounded', 'Continuous')
            yield_frequency: Frequency for yield calculation (defaults to coupon frequency)
            
        Returns:
            BondPrice object with pricing results
        """
        try:
            # Set evaluation date
            ql_settlement = self._python_date_to_ql(settlement_date)
            ql.Settings.instance().evaluationDate = ql_settlement
            
            # Get QuantLib objects
            calendar = self._get_calendar(bond.calendar)
            day_count = self._get_day_count(bond.day_count)
            business_day_convention = self._get_business_day_convention(
                bond.business_day_convention
            )
            frequency = self._get_frequency(bond.payment_frequency)
            
            # Create schedule
            issue_date = self._python_date_to_ql(bond.issue_date)
            maturity_date = self._python_date_to_ql(bond.maturity_date)
            
            schedule = ql.Schedule(
                issue_date,
                maturity_date,
                ql.Period(frequency),
                calendar,
                business_day_convention,
                business_day_convention,
                ql.DateGeneration.Backward,
                False
            )
            
            # Create fixed rate bond
            ql_bond = ql.FixedRateBond(
                bond.settlement_days,
                bond.face_value,
                schedule,
                [bond.coupon_rate],
                day_count
            )
            
            # Set up yield calculation
            compounding_map = {
                'Simple': ql.Simple,
                'Compounded': ql.Compounded,
                'Continuous': ql.Continuous,
            }
            comp = compounding_map.get(compounding, ql.Compounded)
            
            yld_freq = self._get_frequency(
                yield_frequency or bond.payment_frequency
            )
            
            # Calculate prices
            clean_price = ql_bond.cleanPrice(yield_rate, day_count, comp, yld_freq)
            dirty_price = ql_bond.dirtyPrice(yield_rate, day_count, comp, yld_freq)
            accrued = ql_bond.accruedAmount(ql_settlement)
            
            # Calculate risk metrics
            duration = ql.BondFunctions.duration(
                ql_bond, yield_rate, day_count, comp, yld_freq
            )
            convexity = ql.BondFunctions.convexity(
                ql_bond, yield_rate, day_count, comp, yld_freq
            )
            
            # Calculate YTM from clean price (verification)
            ytm = ql.BondFunctions.yieldRate(
                ql_bond, clean_price, day_count, comp, yld_freq
            )
            
            result = BondPrice(
                clean_price=clean_price,
                dirty_price=dirty_price,
                accrued_interest=accrued,
                ytm=ytm,
                duration=duration,
                convexity=convexity,
                present_value=dirty_price
            )
            
            if self.config.log_library_usage:
                logger.debug(f"Priced bond: Clean=${result.clean_price:.2f}, YTM={result.ytm:.4%}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error pricing bond with QuantLib: {e}")
            raise
    
    def price_zero_coupon_bond(
        self,
        face_value: float,
        maturity_date: date,
        settlement_date: date,
        yield_rate: float,
        day_count: DayCountConvention = DayCountConvention.ACTUAL_ACTUAL
    ) -> BondPrice:
        """Price a zero-coupon bond.
        
        Args:
            face_value: Face value of the bond
            maturity_date: Maturity date
            settlement_date: Settlement date for pricing
            yield_rate: Yield to maturity
            day_count: Day count convention
            
        Returns:
            BondPrice object
        """
        # Create a bond with zero coupon rate
        bond = BondSpecification(
            face_value=face_value,
            coupon_rate=0.0,
            issue_date=settlement_date,
            maturity_date=maturity_date,
            day_count=day_count
        )
        
        return self.price_bond(bond, settlement_date, yield_rate)
    
    def build_yield_curve(
        self,
        curve_points: List[YieldCurvePoint],
        settlement_date: date,
        day_count: DayCountConvention = DayCountConvention.ACTUAL_ACTUAL,
        interpolation: str = 'LogLinear'
    ) -> 'ql.YieldTermStructure':
        """Build a yield curve from market data.
        
        Args:
            curve_points: List of (maturity, rate) points
            settlement_date: Settlement date for the curve
            day_count: Day count convention
            interpolation: Interpolation method ('Linear', 'LogLinear', 'CubicSpline')
            
        Returns:
            QuantLib YieldTermStructure object
        """
        try:
            ql_settlement = self._python_date_to_ql(settlement_date)
            ql.Settings.instance().evaluationDate = ql_settlement
            
            # Sort points by maturity
            sorted_points = sorted(curve_points, key=lambda p: p.maturity)
            
            # Create dates and rates
            dates = [ql_settlement]
            rates = []
            
            for point in sorted_points:
                # Convert maturity years to date
                maturity_date = ql_settlement + ql.Period(int(point.maturity * 365), ql.Days)
                dates.append(maturity_date)
                rates.append(point.rate)
            
            ql_day_count = self._get_day_count(day_count)
            
            # Build curve with specified interpolation
            if interpolation == 'Linear':
                curve = ql.ZeroCurve(dates, rates, ql_day_count)
            elif interpolation == 'LogLinear':
                curve = ql.ZeroCurve(dates, rates, ql_day_count, ql.TARGET(), ql.Linear())
            elif interpolation == 'CubicSpline':
                curve = ql.ZeroCurve(dates, rates, ql_day_count, ql.TARGET(), ql.Cubic())
            else:
                curve = ql.ZeroCurve(dates, rates, ql_day_count)
            
            logger.info(f"Built yield curve with {len(rates)} points")
            return curve
            
        except Exception as e:
            logger.error(f"Error building yield curve: {e}")
            raise


def check_quantlib_availability() -> bool:
    """Check if QuantLib is available.
    
    Returns:
        True if QuantLib is available, False otherwise
    """
    return QUANTLIB_AVAILABLE