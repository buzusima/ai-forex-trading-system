"""
Institutional-grade Forex AI Trading System
Position Manager Module

This module provides intelligent position management including exposure calculation,
correlation analysis, dynamic hedging, and risk-adjusted position sizing.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import threading
from collections import defaultdict
import json
import asyncio
from abc import ABC, abstractmethod


class PositionStatus(Enum):
    """Position status enumeration"""

    OPEN = "OPEN"
    CLOSED = "CLOSED"
    PENDING = "PENDING"
    PARTIAL = "PARTIAL"
    CANCELLED = "CANCELLED"


class PositionType(Enum):
    """Position type enumeration"""

    LONG = "LONG"
    SHORT = "SHORT"
    HEDGE = "HEDGE"


@dataclass
class Position:
    """Position data class"""

    position_id: str
    symbol: str
    position_type: PositionType
    volume: float
    entry_price: float
    current_price: float
    open_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    profit: float = 0.0
    unrealized_profit: float = 0.0
    commission: float = 0.0
    swap: float = 0.0
    status: PositionStatus = PositionStatus.OPEN
    close_time: Optional[datetime] = None
    close_price: Optional[float] = None
    magic_number: Optional[int] = None
    comment: str = ""
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        if isinstance(self.position_type, str):
            self.position_type = PositionType(self.position_type)
        if isinstance(self.status, str):
            self.status = PositionStatus(self.status)
        if isinstance(self.open_time, str):
            self.open_time = datetime.fromisoformat(self.open_time)
        if isinstance(self.close_time, str) and self.close_time:
            self.close_time = datetime.fromisoformat(self.close_time)

    def update_current_price(self, price: float):
        """Update current price and calculate unrealized profit"""
        self.current_price = price
        self.unrealized_profit = self.calculate_unrealized_profit()

    def calculate_unrealized_profit(self) -> float:
        """Calculate unrealized profit/loss"""
        if self.status != PositionStatus.OPEN:
            return 0.0

        if self.position_type == PositionType.LONG:
            return (
                (self.current_price - self.entry_price) * self.volume * 100000
            )  # Standard lot
        else:  # SHORT
            return (self.entry_price - self.current_price) * self.volume * 100000

    def calculate_total_profit(self) -> float:
        """Calculate total profit including commissions and swaps"""
        return self.unrealized_profit - self.commission - self.swap

    def get_duration(self) -> timedelta:
        """Get position duration"""
        end_time = self.close_time if self.close_time else datetime.now(timezone.utc)
        return end_time - self.open_time

    def is_profitable(self) -> bool:
        """Check if position is profitable"""
        return self.calculate_total_profit() > 0

    def get_risk_reward_ratio(self) -> Optional[float]:
        """Calculate risk-reward ratio"""
        if not self.stop_loss or not self.take_profit:
            return None

        if self.position_type == PositionType.LONG:
            risk = abs(self.entry_price - self.stop_loss)
            reward = abs(self.take_profit - self.entry_price)
        else:
            risk = abs(self.stop_loss - self.entry_price)
            reward = abs(self.entry_price - self.take_profit)

        return reward / risk if risk > 0 else None

    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary"""
        data = asdict(self)
        data["position_type"] = self.position_type.value
        data["status"] = self.status.value
        data["open_time"] = self.open_time.isoformat()
        if self.close_time:
            data["close_time"] = self.close_time.isoformat()
        return data


@dataclass
class ExposureInfo:
    """Exposure information for a symbol or currency"""

    symbol: str
    long_volume: float = 0.0
    short_volume: float = 0.0
    net_volume: float = 0.0
    long_positions: int = 0
    short_positions: int = 0
    total_unrealized_profit: float = 0.0
    average_entry_price: float = 0.0

    @property
    def is_net_long(self) -> bool:
        return self.net_volume > 0

    @property
    def is_net_short(self) -> bool:
        return self.net_volume < 0

    @property
    def is_neutral(self) -> bool:
        return abs(self.net_volume) < 0.001  # Consider small differences as neutral


@dataclass
class CorrelationData:
    """Correlation data between symbols"""

    symbol_pair: Tuple[str, str]
    correlation: float
    lookback_period: int
    last_updated: datetime
    confidence_level: float


class PositionAnalyzer:
    """Position analysis and metrics calculator"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def calculate_portfolio_exposure(
        self, positions: List[Position]
    ) -> Dict[str, ExposureInfo]:
        """Calculate exposure for each symbol"""
        exposure_data = defaultdict(lambda: ExposureInfo(""))

        for position in positions:
            if position.status != PositionStatus.OPEN:
                continue

            exposure = exposure_data[position.symbol]
            exposure.symbol = position.symbol

            if position.position_type == PositionType.LONG:
                exposure.long_volume += position.volume
                exposure.long_positions += 1
            else:
                exposure.short_volume += position.volume
                exposure.short_positions += 1

            exposure.total_unrealized_profit += position.unrealized_profit

        # Calculate net exposure and average prices
        for symbol, exposure in exposure_data.items():
            exposure.net_volume = exposure.long_volume - exposure.short_volume

            # Calculate weighted average entry price
            total_volume = exposure.long_volume + exposure.short_volume
            if total_volume > 0:
                relevant_positions = [
                    p
                    for p in positions
                    if p.symbol == symbol and p.status == PositionStatus.OPEN
                ]
                if relevant_positions:
                    weighted_sum = sum(
                        p.entry_price * p.volume for p in relevant_positions
                    )
                    exposure.average_entry_price = weighted_sum / total_volume

        return dict(exposure_data)

    def calculate_currency_exposure(
        self, positions: List[Position]
    ) -> Dict[str, float]:
        """Calculate exposure by currency"""
        currency_exposure = defaultdict(float)

        for position in positions:
            if position.status != PositionStatus.OPEN:
                continue

            # Extract base and quote currencies (e.g., EURUSD -> EUR, USD)
            symbol = position.symbol
            if len(symbol) == 6:
                base_currency = symbol[:3]
                quote_currency = symbol[3:]

                volume = position.volume
                if position.position_type == PositionType.SHORT:
                    volume = -volume

                currency_exposure[base_currency] += volume
                currency_exposure[quote_currency] -= volume

        return dict(currency_exposure)

    def calculate_correlation_matrix(
        self, price_data: Dict[str, pd.Series], lookback_period: int = 30
    ) -> Dict[Tuple[str, str], CorrelationData]:
        """Calculate correlation matrix between symbols"""
        correlations = {}
        symbols = list(price_data.keys())

        for i, symbol1 in enumerate(symbols):
            for symbol2 in symbols[i + 1 :]:
                if (
                    len(price_data[symbol1]) >= lookback_period
                    and len(price_data[symbol2]) >= lookback_period
                ):
                    # Get returns for correlation calculation
                    returns1 = (
                        price_data[symbol1].pct_change().dropna().tail(lookback_period)
                    )
                    returns2 = (
                        price_data[symbol2].pct_change().dropna().tail(lookback_period)
                    )

                    # Align data
                    aligned_data = pd.concat([returns1, returns2], axis=1).dropna()

                    if len(aligned_data) >= 10:  # Minimum data points
                        correlation = aligned_data.iloc[:, 0].corr(
                            aligned_data.iloc[:, 1]
                        )
                        confidence = min(len(aligned_data) / lookback_period, 1.0)

                        correlations[(symbol1, symbol2)] = CorrelationData(
                            symbol_pair=(symbol1, symbol2),
                            correlation=(
                                correlation if not np.isnan(correlation) else 0.0
                            ),
                            lookback_period=lookback_period,
                            last_updated=datetime.now(timezone.utc),
                            confidence_level=confidence,
                        )

        return correlations

    def calculate_portfolio_risk_metrics(
        self, positions: List[Position]
    ) -> Dict[str, float]:
        """Calculate portfolio risk metrics"""
        if not positions:
            return {}

        open_positions = [p for p in positions if p.status == PositionStatus.OPEN]
        if not open_positions:
            return {}

        # Basic metrics
        total_volume = sum(p.volume for p in open_positions)
        total_unrealized_profit = sum(p.unrealized_profit for p in open_positions)
        total_positions = len(open_positions)

        # Profit distribution
        profits = [p.unrealized_profit for p in open_positions]
        profitable_positions = sum(1 for p in profits if p > 0)

        # Risk metrics
        position_sizes = [p.volume for p in open_positions]
        largest_position = max(position_sizes) if position_sizes else 0

        return {
            "total_positions": total_positions,
            "total_volume": total_volume,
            "total_unrealized_profit": total_unrealized_profit,
            "average_position_size": np.mean(position_sizes),
            "largest_position_size": largest_position,
            "position_concentration": (
                largest_position / total_volume if total_volume > 0 else 0
            ),
            "win_rate": (
                profitable_positions / total_positions if total_positions > 0 else 0
            ),
            "average_profit_per_position": (
                total_unrealized_profit / total_positions if total_positions > 0 else 0
            ),
            "profit_std": np.std(profits) if len(profits) > 1 else 0,
            "sharpe_ratio": (
                np.mean(profits) / np.std(profits)
                if len(profits) > 1 and np.std(profits) > 0
                else 0
            ),
        }


class HedgingStrategy(ABC):
    """Abstract base class for hedging strategies"""

    @abstractmethod
    def calculate_hedge_requirements(
        self, positions: List[Position], exposure_data: Dict[str, ExposureInfo]
    ) -> List[Dict[str, Any]]:
        """Calculate hedge requirements"""
        pass


class DynamicHedgeStrategy(HedgingStrategy):
    """Dynamic hedging strategy based on exposure and correlation"""

    def __init__(
        self, hedge_threshold: float = 0.5, correlation_threshold: float = 0.7
    ):
        self.hedge_threshold = hedge_threshold
        self.correlation_threshold = correlation_threshold
        self.logger = logging.getLogger(__name__)

    def calculate_hedge_requirements(
        self, positions: List[Position], exposure_data: Dict[str, ExposureInfo]
    ) -> List[Dict[str, Any]]:
        """Calculate hedge requirements based on exposure"""
        hedge_orders = []

        for symbol, exposure in exposure_data.items():
            if abs(exposure.net_volume) > self.hedge_threshold:
                # Calculate hedge volume
                hedge_volume = abs(exposure.net_volume) * 0.8  # Partial hedge
                hedge_type = (
                    PositionType.SHORT if exposure.is_net_long else PositionType.LONG
                )

                hedge_orders.append(
                    {
                        "symbol": symbol,
                        "action": "SELL" if hedge_type == PositionType.SHORT else "BUY",
                        "volume": hedge_volume,
                        "order_type": "MARKET",
                        "reason": f"Hedge for {exposure.net_volume:.3f} net exposure",
                        "hedge_flag": True,
                    }
                )

        return hedge_orders


class PositionSizer:
    """Dynamic position sizing based on risk parameters"""

    def __init__(
        self, max_risk_per_trade: float = 0.02, max_portfolio_risk: float = 0.1
    ):
        self.max_risk_per_trade = max_risk_per_trade  # 2% risk per trade
        self.max_portfolio_risk = max_portfolio_risk  # 10% total portfolio risk
        self.logger = logging.getLogger(__name__)

    def calculate_position_size(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss: float,
        symbol: str,
        current_exposure: Dict[str, ExposureInfo],
    ) -> float:
        """Calculate optimal position size"""

        # Calculate risk per unit
        risk_per_unit = abs(entry_price - stop_loss)

        # Calculate max risk amount
        max_risk_amount = account_balance * self.max_risk_per_trade

        # Basic position size calculation
        base_position_size = max_risk_amount / (risk_per_unit * 100000)  # Standard lot

        # Adjust for existing exposure
        if symbol in current_exposure:
            exposure = current_exposure[symbol]
            # Reduce position size if already heavily exposed
            exposure_factor = max(0.1, 1.0 - abs(exposure.net_volume) / 10.0)
            base_position_size *= exposure_factor

        # Ensure minimum and maximum limits
        min_size = 0.01
        max_size = min(10.0, account_balance / 10000)  # Max 10 lots or based on balance

        final_size = max(min_size, min(max_size, base_position_size))

        self.logger.info(
            f"Position size calculation for {symbol}: "
            f"Risk per unit: {risk_per_unit:.5f}, "
            f"Base size: {base_position_size:.3f}, "
            f"Final size: {final_size:.3f}"
        )

        return round(final_size, 2)


class PositionManager:
    """Main position manager class"""

    def __init__(self, hedging_strategy: HedgingStrategy = None):
        self.positions: Dict[str, Position] = {}
        self.analyzer = PositionAnalyzer()
        self.hedging_strategy = hedging_strategy or DynamicHedgeStrategy()
        self.position_sizer = PositionSizer()
        self.logger = logging.getLogger(__name__)
        self.lock = threading.Lock()

        # Performance tracking
        self.daily_stats = {}
        self.correlation_cache = {}
        self.last_correlation_update = None

    def add_position(self, position: Position) -> bool:
        """Add new position"""
        with self.lock:
            try:
                if position.position_id in self.positions:
                    self.logger.warning(
                        f"Position {position.position_id} already exists"
                    )
                    return False

                self.positions[position.position_id] = position
                self.logger.info(
                    f"Added position {position.position_id}: "
                    f"{position.symbol} {position.position_type.value} {position.volume}"
                )
                return True

            except Exception as e:
                self.logger.error(f"Error adding position: {str(e)}")
                return False

    def update_position(self, position_id: str, **kwargs) -> bool:
        """Update existing position"""
        with self.lock:
            try:
                if position_id not in self.positions:
                    self.logger.warning(f"Position {position_id} not found")
                    return False

                position = self.positions[position_id]

                for key, value in kwargs.items():
                    if hasattr(position, key):
                        setattr(position, key, value)

                # Recalculate unrealized profit if price updated
                if "current_price" in kwargs:
                    position.update_current_price(kwargs["current_price"])

                return True

            except Exception as e:
                self.logger.error(f"Error updating position {position_id}: {str(e)}")
                return False

    def close_position(
        self, position_id: str, close_price: float, close_time: datetime = None
    ) -> bool:
        """Close position"""
        with self.lock:
            try:
                if position_id not in self.positions:
                    self.logger.warning(f"Position {position_id} not found")
                    return False

                position = self.positions[position_id]
                position.status = PositionStatus.CLOSED
                position.close_price = close_price
                position.close_time = close_time or datetime.now(timezone.utc)

                # Calculate final profit
                if position.position_type == PositionType.LONG:
                    position.profit = (
                        (close_price - position.entry_price) * position.volume * 100000
                    )
                else:
                    position.profit = (
                        (position.entry_price - close_price) * position.volume * 100000
                    )

                position.profit -= position.commission + position.swap

                self.logger.info(
                    f"Closed position {position_id}: " f"Profit: {position.profit:.2f}"
                )
                return True

            except Exception as e:
                self.logger.error(f"Error closing position {position_id}: {str(e)}")
                return False

    def get_open_positions(self) -> List[Position]:
        """Get all open positions"""
        with self.lock:
            return [
                p for p in self.positions.values() if p.status == PositionStatus.OPEN
            ]

    def get_positions_by_symbol(self, symbol: str) -> List[Position]:
        """Get positions for specific symbol"""
        with self.lock:
            return [p for p in self.positions.values() if p.symbol == symbol]

    def update_prices(self, price_data: Dict[str, float]):
        """Update current prices for all positions"""
        with self.lock:
            for position in self.positions.values():
                if (
                    position.status == PositionStatus.OPEN
                    and position.symbol in price_data
                ):
                    position.update_current_price(price_data[position.symbol])

    def get_portfolio_exposure(self) -> Dict[str, ExposureInfo]:
        """Get current portfolio exposure"""
        open_positions = self.get_open_positions()
        return self.analyzer.calculate_portfolio_exposure(open_positions)

    def get_currency_exposure(self) -> Dict[str, float]:
        """Get currency exposure"""
        open_positions = self.get_open_positions()
        return self.analyzer.calculate_currency_exposure(open_positions)

    def get_risk_metrics(self) -> Dict[str, float]:
        """Get portfolio risk metrics"""
        open_positions = self.get_open_positions()
        return self.analyzer.calculate_portfolio_risk_metrics(open_positions)

    def calculate_optimal_position_size(
        self, account_balance: float, entry_price: float, stop_loss: float, symbol: str
    ) -> float:
        """Calculate optimal position size"""
        current_exposure = self.get_portfolio_exposure()
        return self.position_sizer.calculate_position_size(
            account_balance, entry_price, stop_loss, symbol, current_exposure
        )

    def get_hedge_recommendations(self) -> List[Dict[str, Any]]:
        """Get hedge recommendations"""
        exposure_data = self.get_portfolio_exposure()
        open_positions = self.get_open_positions()
        return self.hedging_strategy.calculate_hedge_requirements(
            open_positions, exposure_data
        )

    def get_position_performance(self, days: int = 30) -> Dict[str, Any]:
        """Get position performance metrics"""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        recent_positions = [
            p
            for p in self.positions.values()
            if p.open_time >= cutoff_date
            or (p.close_time and p.close_time >= cutoff_date)
        ]

        if not recent_positions:
            return {}

        closed_positions = [
            p for p in recent_positions if p.status == PositionStatus.CLOSED
        ]

        if not closed_positions:
            return {"message": "No closed positions in the specified period"}

        profits = [p.profit for p in closed_positions]
        durations = [
            p.get_duration().total_seconds() / 3600 for p in closed_positions
        ]  # Hours

        return {
            "total_trades": len(closed_positions),
            "winning_trades": sum(1 for p in profits if p > 0),
            "losing_trades": sum(1 for p in profits if p <= 0),
            "win_rate": sum(1 for p in profits if p > 0) / len(profits),
            "total_profit": sum(profits),
            "average_profit": np.mean(profits),
            "profit_factor": (
                sum(p for p in profits if p > 0)
                / abs(sum(p for p in profits if p <= 0))
                if any(p <= 0 for p in profits)
                else float("inf")
            ),
            "largest_win": max(profits),
            "largest_loss": min(profits),
            "average_duration_hours": np.mean(durations),
            "sharpe_ratio": (
                np.mean(profits) / np.std(profits)
                if len(profits) > 1 and np.std(profits) > 0
                else 0
            ),
        }

    def export_positions(self, include_closed: bool = False) -> List[Dict[str, Any]]:
        """Export positions to dictionary format"""
        positions_to_export = list(self.positions.values())
        if not include_closed:
            positions_to_export = [
                p for p in positions_to_export if p.status == PositionStatus.OPEN
            ]

        return [position.to_dict() for position in positions_to_export]

    def import_positions(self, positions_data: List[Dict[str, Any]]) -> int:
        """Import positions from dictionary format"""
        imported_count = 0

        for position_data in positions_data:
            try:
                position = Position(**position_data)
                if self.add_position(position):
                    imported_count += 1
            except Exception as e:
                self.logger.error(f"Error importing position: {str(e)}")

        return imported_count

    def cleanup_old_positions(self, days: int = 90):
        """Remove old closed positions"""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)

        with self.lock:
            old_positions = [
                pid
                for pid, p in self.positions.items()
                if p.status == PositionStatus.CLOSED
                and p.close_time
                and p.close_time < cutoff_date
            ]

            for pid in old_positions:
                del self.positions[pid]

            self.logger.info(f"Cleaned up {len(old_positions)} old positions")
            return len(old_positions)


# Example usage and testing
if __name__ == "__main__":
    # Initialize position manager
    position_manager = PositionManager()

    # Create test positions
    test_positions = [
        Position(
            position_id="POS_001",
            symbol="EURUSD",
            position_type=PositionType.LONG,
            volume=0.1,
            entry_price=1.1234,
            current_price=1.1250,
            open_time=datetime.now(timezone.utc),
            stop_loss=1.1200,
            take_profit=1.1300,
        ),
        Position(
            position_id="POS_002",
            symbol="GBPUSD",
            position_type=PositionType.SHORT,
            volume=0.2,
            entry_price=1.2750,
            current_price=1.2730,
            open_time=datetime.now(timezone.utc),
            stop_loss=1.2800,
            take_profit=1.2650,
        ),
    ]

    # Add positions
    for position in test_positions:
        position_manager.add_position(position)

    # Test functionality
    print("Portfolio Exposure:")
    exposure = position_manager.get_portfolio_exposure()
    for symbol, exp in exposure.items():
        print(f"  {symbol}: Net {exp.net_volume:.3f} lots")

    print("\nCurrency Exposure:")
    currency_exp = position_manager.get_currency_exposure()
    for currency, exposure in currency_exp.items():
        print(f"  {currency}: {exposure:.3f}")

    print("\nRisk Metrics:")
    risk_metrics = position_manager.get_risk_metrics()
    for metric, value in risk_metrics.items():
        print(f"  {metric}: {value:.4f}")

    print("\nHedge Recommendations:")
    hedge_recs = position_manager.get_hedge_recommendations()
    for rec in hedge_recs:
        print(
            f"  {rec['symbol']}: {rec['action']} {rec['volume']:.3f} lots - {rec['reason']}"
        )

    # Test position sizing
    optimal_size = position_manager.calculate_optimal_position_size(
        account_balance=10000, entry_price=1.1234, stop_loss=1.1200, symbol="EURUSD"
    )
    print(f"\nOptimal position size for EURUSD: {optimal_size:.3f} lots")

    print("Position manager test completed successfully!")
