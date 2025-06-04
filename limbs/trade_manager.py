"""
limbs/trade_manager.py
Advanced Trade Management System
Institutional-grade TP/SL/Trailing Stop management for Forex trading
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import warnings

warnings.filterwarnings("ignore")


class TradeStatus(Enum):
    """Trade status enumeration"""

    PENDING = "pending"
    ACTIVE = "active"
    CLOSED_PROFIT = "closed_profit"
    CLOSED_LOSS = "closed_loss"
    CLOSED_MANUAL = "closed_manual"
    CANCELLED = "cancelled"
    ERROR = "error"


class TradeType(Enum):
    """Trade type enumeration"""

    BUY = "buy"
    SELL = "sell"


class StopType(Enum):
    """Stop loss type enumeration"""

    FIXED = "fixed"
    TRAILING = "trailing"
    ATR_BASED = "atr_based"
    PERCENTAGE = "percentage"
    SUPPORT_RESISTANCE = "support_resistance"
    DYNAMIC = "dynamic"


class ExitReason(Enum):
    """Exit reason enumeration"""

    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"
    TRAILING_STOP = "trailing_stop"
    TIME_EXIT = "time_exit"
    MANUAL_EXIT = "manual_exit"
    RISK_MANAGEMENT = "risk_management"
    SIGNAL_REVERSAL = "signal_reversal"


@dataclass
class TradePosition:
    """Trade position structure"""

    trade_id: str
    symbol: str
    trade_type: TradeType
    entry_price: float
    volume: float
    entry_time: datetime
    status: TradeStatus = TradeStatus.PENDING

    # Stop Loss Management
    stop_loss_price: Optional[float] = None
    stop_loss_type: StopType = StopType.FIXED
    initial_stop_loss: Optional[float] = None

    # Take Profit Management
    take_profit_price: Optional[float] = None
    take_profit_levels: List[Tuple[float, float]] = field(
        default_factory=list
    )  # (price, partial_close_pct)

    # Trailing Stop Management
    trailing_stop_distance: Optional[float] = None
    trailing_stop_activated: bool = False
    highest_price: Optional[float] = None  # For buy trades
    lowest_price: Optional[float] = None  # For sell trades

    # Dynamic Management
    breakeven_triggered: bool = False
    partial_closes: List[Dict] = field(default_factory=list)

    # Current Market Data
    current_price: Optional[float] = None
    current_profit: float = 0.0
    current_profit_pct: float = 0.0
    max_profit: float = 0.0
    max_drawdown: float = 0.0

    # Exit Information
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: Optional[ExitReason] = None

    # Metadata
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert position to dictionary"""
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "trade_type": self.trade_type.value,
            "entry_price": self.entry_price,
            "volume": self.volume,
            "entry_time": self.entry_time,
            "status": self.status.value,
            "stop_loss_price": self.stop_loss_price,
            "stop_loss_type": self.stop_loss_type.value,
            "take_profit_price": self.take_profit_price,
            "current_price": self.current_price,
            "current_profit": self.current_profit,
            "current_profit_pct": self.current_profit_pct,
            "max_profit": self.max_profit,
            "max_drawdown": self.max_drawdown,
            "exit_price": self.exit_price,
            "exit_time": self.exit_time,
            "exit_reason": self.exit_reason.value if self.exit_reason else None,
            "metadata": self.metadata,
        }


@dataclass
class TradeManagerConfig:
    """Configuration for trade management"""

    # Default Risk Management
    default_stop_loss_pct: float = 0.02  # 2%
    default_take_profit_pct: float = 0.04  # 4%
    default_risk_reward_ratio: float = 2.0

    # Trailing Stop Configuration
    trailing_stop_distance_pct: float = 0.01  # 1%
    trailing_stop_activation_pct: float = 0.005  # 0.5% profit to activate
    trailing_stop_step_pct: float = 0.002  # 0.2% step size

    # ATR-based stops
    atr_stop_multiplier: float = 2.0
    atr_trailing_multiplier: float = 1.5

    # Breakeven Management
    breakeven_trigger_pct: float = 0.01  # 1% profit to trigger breakeven
    breakeven_buffer_pct: float = 0.002  # 0.2% buffer above breakeven

    # Partial Close Management
    partial_close_enabled: bool = True
    partial_close_levels: List[Tuple[float, float]] = field(
        default_factory=lambda: [(0.02, 0.5), (0.04, 0.3)]  # (profit_pct, close_pct)
    )

    # Time-based Management
    max_trade_duration_hours: int = 24
    time_exit_enabled: bool = True

    # Dynamic Management
    dynamic_stops_enabled: bool = True
    support_resistance_stops: bool = True

    # Risk Limits
    max_drawdown_pct: float = 0.05  # 5% max drawdown to force close
    correlation_exit_enabled: bool = True

    # Update Frequency
    update_interval_seconds: float = 1.0
    price_precision: int = 5


class TradeManager:
    """
    Advanced Trade Management System
    Handles TP/SL/Trailing stops with institutional-grade features
    """

    def __init__(
        self, config: TradeManagerConfig = None, mt5_connector=None, risk_manager=None
    ):
        self.logger = logging.getLogger(__name__)
        self.config = config or TradeManagerConfig()
        self.mt5_connector = mt5_connector
        self.risk_manager = risk_manager

        # Active positions
        self.active_positions: Dict[str, TradePosition] = {}
        self.closed_positions: List[TradePosition] = []

        # Management thread
        self.management_thread = None
        self.is_running = False
        self.stop_event = threading.Event()

        # Price update callbacks
        self.price_callbacks: List[Callable] = []

        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_profit = 0.0

        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)

        self.logger.info("TradeManager initialized successfully")

    def start_management(self):
        """Start the trade management thread"""
        try:
            if self.is_running:
                self.logger.warning("Trade management already running")
                return

            self.is_running = True
            self.stop_event.clear()

            self.management_thread = threading.Thread(
                target=self._management_loop, daemon=True
            )
            self.management_thread.start()

            self.logger.info("Trade management started")

        except Exception as e:
            self.logger.error(f"Error starting trade management: {e}")
            raise

    def stop_management(self):
        """Stop the trade management thread"""
        try:
            if not self.is_running:
                return

            self.is_running = False
            self.stop_event.set()

            if self.management_thread and self.management_thread.is_alive():
                self.management_thread.join(timeout=5.0)

            self.executor.shutdown(wait=True)

            self.logger.info("Trade management stopped")

        except Exception as e:
            self.logger.error(f"Error stopping trade management: {e}")

    def add_position(self, position: TradePosition) -> bool:
        """Add a new position to management"""
        try:
            if position.trade_id in self.active_positions:
                self.logger.warning(f"Position {position.trade_id} already exists")
                return False

            # Set initial values
            position.status = TradeStatus.ACTIVE
            position.highest_price = position.entry_price
            position.lowest_price = position.entry_price
            position.initial_stop_loss = position.stop_loss_price

            # Add to active positions
            self.active_positions[position.trade_id] = position
            self.total_trades += 1

            self.logger.info(
                f"Added position {position.trade_id} for {position.symbol}"
            )
            return True

        except Exception as e:
            self.logger.error(f"Error adding position: {e}")
            return False

    def remove_position(
        self, trade_id: str, exit_reason: ExitReason = ExitReason.MANUAL_EXIT
    ) -> bool:
        """Remove position from management"""
        try:
            if trade_id not in self.active_positions:
                self.logger.warning(f"Position {trade_id} not found")
                return False

            position = self.active_positions[trade_id]
            position.exit_time = datetime.now()
            position.exit_reason = exit_reason
            position.status = TradeStatus.CLOSED_MANUAL

            # Move to closed positions
            self.closed_positions.append(position)
            del self.active_positions[trade_id]

            self.logger.info(f"Removed position {trade_id}")
            return True

        except Exception as e:
            self.logger.error(f"Error removing position: {e}")
            return False

    def update_price(self, symbol: str, bid: float, ask: float):
        """Update market prices for positions"""
        try:
            updated_positions = []

            for trade_id, position in self.active_positions.items():
                if position.symbol == symbol:
                    # Use appropriate price based on trade type
                    current_price = ask if position.trade_type == TradeType.BUY else bid

                    # Update position
                    self._update_position_metrics(position, current_price)
                    updated_positions.append(position)

            # Check for exits after all positions updated
            for position in updated_positions:
                self._check_exit_conditions(position)

        except Exception as e:
            self.logger.error(f"Error updating prices: {e}")

    def _management_loop(self):
        """Main management loop"""
        self.logger.info("Trade management loop started")

        while self.is_running and not self.stop_event.is_set():
            try:
                start_time = time.time()

                # Update all positions
                self._update_all_positions()

                # Check exit conditions
                self._check_all_exit_conditions()

                # Update trailing stops
                self._update_trailing_stops()

                # Check time-based exits
                self._check_time_exits()

                # Update breakeven stops
                self._update_breakeven_stops()

                # Check partial closes
                self._check_partial_closes()

                # Sleep for remaining time
                elapsed = time.time() - start_time
                sleep_time = max(0, self.config.update_interval_seconds - elapsed)

                if sleep_time > 0:
                    self.stop_event.wait(sleep_time)

            except Exception as e:
                self.logger.error(f"Error in management loop: {e}")
                time.sleep(1.0)  # Prevent tight error loop

        self.logger.info("Trade management loop stopped")

    def _update_position_metrics(self, position: TradePosition, current_price: float):
        """Update position metrics with current price"""
        try:
            position.current_price = current_price

            # Calculate profit
            if position.trade_type == TradeType.BUY:
                position.current_profit = (
                    current_price - position.entry_price
                ) * position.volume
                position.current_profit_pct = (
                    current_price - position.entry_price
                ) / position.entry_price

                # Update highest price for trailing stops
                if (
                    position.highest_price is None
                    or current_price > position.highest_price
                ):
                    position.highest_price = current_price

            else:  # SELL
                position.current_profit = (
                    position.entry_price - current_price
                ) * position.volume
                position.current_profit_pct = (
                    position.entry_price - current_price
                ) / position.entry_price

                # Update lowest price for trailing stops
                if (
                    position.lowest_price is None
                    or current_price < position.lowest_price
                ):
                    position.lowest_price = current_price

            # Update max profit and drawdown
            if position.current_profit > position.max_profit:
                position.max_profit = position.current_profit

            drawdown = position.max_profit - position.current_profit
            if drawdown > position.max_drawdown:
                position.max_drawdown = drawdown

        except Exception as e:
            self.logger.error(f"Error updating position metrics: {e}")

    def _check_exit_conditions(self, position: TradePosition):
        """Check if position should be exited"""
        try:
            if position.current_price is None:
                return

            # Check take profit
            if self._check_take_profit(position):
                return

            # Check stop loss
            if self._check_stop_loss(position):
                return

            # Check max drawdown
            if self._check_max_drawdown(position):
                return

        except Exception as e:
            self.logger.error(f"Error checking exit conditions: {e}")

    def _check_take_profit(self, position: TradePosition) -> bool:
        """Check take profit conditions"""
        try:
            if position.take_profit_price is None:
                return False

            should_exit = False

            if position.trade_type == TradeType.BUY:
                should_exit = position.current_price >= position.take_profit_price
            else:  # SELL
                should_exit = position.current_price <= position.take_profit_price

            if should_exit:
                self._close_position(position, ExitReason.TAKE_PROFIT)
                return True

            return False

        except Exception as e:
            self.logger.error(f"Error checking take profit: {e}")
            return False

    def _check_stop_loss(self, position: TradePosition) -> bool:
        """Check stop loss conditions"""
        try:
            if position.stop_loss_price is None:
                return False

            should_exit = False

            if position.trade_type == TradeType.BUY:
                should_exit = position.current_price <= position.stop_loss_price
            else:  # SELL
                should_exit = position.current_price >= position.stop_loss_price

            if should_exit:
                exit_reason = (
                    ExitReason.TRAILING_STOP
                    if position.stop_loss_type == StopType.TRAILING
                    else ExitReason.STOP_LOSS
                )
                self._close_position(position, exit_reason)
                return True

            return False

        except Exception as e:
            self.logger.error(f"Error checking stop loss: {e}")
            return False

    def _check_max_drawdown(self, position: TradePosition) -> bool:
        """Check maximum drawdown limit"""
        try:
            if position.max_profit <= 0:
                return False

            drawdown_pct = position.max_drawdown / (
                position.entry_price * position.volume
            )

            if drawdown_pct > self.config.max_drawdown_pct:
                self._close_position(position, ExitReason.RISK_MANAGEMENT)
                return True

            return False

        except Exception as e:
            self.logger.error(f"Error checking max drawdown: {e}")
            return False

    def _update_trailing_stops(self):
        """Update trailing stop levels for all positions"""
        try:
            for position in self.active_positions.values():
                if position.stop_loss_type == StopType.TRAILING:
                    self._update_trailing_stop(position)

        except Exception as e:
            self.logger.error(f"Error updating trailing stops: {e}")

    def _update_trailing_stop(self, position: TradePosition):
        """Update trailing stop for a specific position"""
        try:
            if (
                position.current_price is None
                or position.trailing_stop_distance is None
            ):
                return

            # Check if trailing stop should be activated
            if not position.trailing_stop_activated:
                profit_pct = abs(position.current_profit_pct)
                if profit_pct >= self.config.trailing_stop_activation_pct:
                    position.trailing_stop_activated = True
                    self.logger.info(f"Trailing stop activated for {position.trade_id}")

            if not position.trailing_stop_activated:
                return

            # Calculate new trailing stop level
            new_stop = None

            if position.trade_type == TradeType.BUY:
                # For buy trades, trail below the highest price
                new_stop = position.highest_price - position.trailing_stop_distance

                # Only move stop loss up (never down)
                if (
                    position.stop_loss_price is None
                    or new_stop > position.stop_loss_price
                ):
                    position.stop_loss_price = round(
                        new_stop, self.config.price_precision
                    )

            else:  # SELL
                # For sell trades, trail above the lowest price
                new_stop = position.lowest_price + position.trailing_stop_distance

                # Only move stop loss down (never up)
                if (
                    position.stop_loss_price is None
                    or new_stop < position.stop_loss_price
                ):
                    position.stop_loss_price = round(
                        new_stop, self.config.price_precision
                    )

        except Exception as e:
            self.logger.error(f"Error updating trailing stop: {e}")

    def _update_breakeven_stops(self):
        """Update breakeven stops for profitable positions"""
        try:
            for position in self.active_positions.values():
                if not position.breakeven_triggered and position.current_profit_pct > 0:
                    self._check_breakeven_trigger(position)

        except Exception as e:
            self.logger.error(f"Error updating breakeven stops: {e}")

    def _check_breakeven_trigger(self, position: TradePosition):
        """Check if breakeven should be triggered"""
        try:
            profit_pct = abs(position.current_profit_pct)

            if profit_pct >= self.config.breakeven_trigger_pct:
                # Move stop loss to breakeven + buffer
                buffer = position.entry_price * self.config.breakeven_buffer_pct

                if position.trade_type == TradeType.BUY:
                    new_stop = position.entry_price + buffer
                    if (
                        position.stop_loss_price is None
                        or new_stop > position.stop_loss_price
                    ):
                        position.stop_loss_price = round(
                            new_stop, self.config.price_precision
                        )
                        position.breakeven_triggered = True

                else:  # SELL
                    new_stop = position.entry_price - buffer
                    if (
                        position.stop_loss_price is None
                        or new_stop < position.stop_loss_price
                    ):
                        position.stop_loss_price = round(
                            new_stop, self.config.price_precision
                        )
                        position.breakeven_triggered = True

                if position.breakeven_triggered:
                    self.logger.info(f"Breakeven triggered for {position.trade_id}")

        except Exception as e:
            self.logger.error(f"Error checking breakeven trigger: {e}")

    def _check_partial_closes(self):
        """Check for partial close opportunities"""
        try:
            if not self.config.partial_close_enabled:
                return

            for position in self.active_positions.values():
                self._check_position_partial_close(position)

        except Exception as e:
            self.logger.error(f"Error checking partial closes: {e}")

    def _check_position_partial_close(self, position: TradePosition):
        """Check partial close for a specific position"""
        try:
            if not position.current_profit_pct > 0:
                return

            profit_pct = abs(position.current_profit_pct)

            for target_profit, close_pct in self.config.partial_close_levels:
                # Check if this level hasn't been triggered yet
                level_key = f"partial_{target_profit}_{close_pct}"
                if level_key in position.metadata:
                    continue

                if profit_pct >= target_profit:
                    # Execute partial close
                    close_volume = position.volume * close_pct

                    if self._execute_partial_close(position, close_volume):
                        position.metadata[level_key] = {
                            "triggered_at": datetime.now(),
                            "price": position.current_price,
                            "profit_pct": profit_pct,
                            "volume_closed": close_volume,
                        }

                        # Record partial close
                        position.partial_closes.append(
                            {
                                "timestamp": datetime.now(),
                                "price": position.current_price,
                                "volume": close_volume,
                                "profit_pct": profit_pct,
                            }
                        )

                        self.logger.info(
                            f"Partial close executed for {position.trade_id}: {close_volume} lots at {profit_pct:.2%} profit"
                        )

        except Exception as e:
            self.logger.error(f"Error checking position partial close: {e}")

    def _execute_partial_close(self, position: TradePosition, volume: float) -> bool:
        """Execute partial close of position"""
        try:
            # Update position volume
            position.volume -= volume

            # If volume becomes too small, close completely
            if position.volume < 0.01:  # Minimum lot size
                self._close_position(position, ExitReason.TAKE_PROFIT)
                return True

            # Here you would integrate with MT5 to actually close the partial position
            # For now, we just update the internal position

            return True

        except Exception as e:
            self.logger.error(f"Error executing partial close: {e}")
            return False

    def _check_time_exits(self):
        """Check for time-based exits"""
        try:
            if not self.config.time_exit_enabled:
                return

            current_time = datetime.now()

            for position in list(self.active_positions.values()):
                time_in_trade = current_time - position.entry_time

                if time_in_trade.total_seconds() > (
                    self.config.max_trade_duration_hours * 3600
                ):
                    self._close_position(position, ExitReason.TIME_EXIT)

        except Exception as e:
            self.logger.error(f"Error checking time exits: {e}")

    def _close_position(self, position: TradePosition, exit_reason: ExitReason):
        """Close a position"""
        try:
            position.exit_price = position.current_price
            position.exit_time = datetime.now()
            position.exit_reason = exit_reason

            # Determine final status
            if position.current_profit > 0:
                position.status = TradeStatus.CLOSED_PROFIT
                self.winning_trades += 1
            else:
                position.status = TradeStatus.CLOSED_LOSS

            # Update total profit
            self.total_profit += position.current_profit

            # Move to closed positions
            self.closed_positions.append(position)
            del self.active_positions[position.trade_id]

            # Here you would integrate with MT5 to actually close the position

            self.logger.info(
                f"Position {position.trade_id} closed: {exit_reason.value}, Profit: {position.current_profit:.2f}"
            )

        except Exception as e:
            self.logger.error(f"Error closing position: {e}")

    def _update_all_positions(self):
        """Update all positions with current market data"""
        try:
            if not self.mt5_connector:
                return

            # Get unique symbols
            symbols = list(set(pos.symbol for pos in self.active_positions.values()))

            for symbol in symbols:
                # Get current prices (this would use mt5_connector in real implementation)
                # For now, simulate price updates
                pass

        except Exception as e:
            self.logger.error(f"Error updating all positions: {e}")

    def _check_all_exit_conditions(self):
        """Check exit conditions for all positions"""
        try:
            for position in list(self.active_positions.values()):
                self._check_exit_conditions(position)

        except Exception as e:
            self.logger.error(f"Error checking all exit conditions: {e}")

    def set_trailing_stop(self, trade_id: str, distance_pct: float = None) -> bool:
        """Set trailing stop for a position"""
        try:
            if trade_id not in self.active_positions:
                self.logger.error(f"Position {trade_id} not found")
                return False

            position = self.active_positions[trade_id]
            position.stop_loss_type = StopType.TRAILING

            # Set trailing distance
            if distance_pct is None:
                distance_pct = self.config.trailing_stop_distance_pct

            position.trailing_stop_distance = position.entry_price * distance_pct

            self.logger.info(f"Trailing stop set for {trade_id}: {distance_pct:.2%}")
            return True

        except Exception as e:
            self.logger.error(f"Error setting trailing stop: {e}")
            return False

    def modify_stop_loss(self, trade_id: str, new_stop_loss: float) -> bool:
        """Modify stop loss for a position"""
        try:
            if trade_id not in self.active_positions:
                self.logger.error(f"Position {trade_id} not found")
                return False

            position = self.active_positions[trade_id]
            position.stop_loss_price = round(new_stop_loss, self.config.price_precision)

            self.logger.info(f"Stop loss modified for {trade_id}: {new_stop_loss}")
            return True

        except Exception as e:
            self.logger.error(f"Error modifying stop loss: {e}")
            return False

    def modify_take_profit(self, trade_id: str, new_take_profit: float) -> bool:
        """Modify take profit for a position"""
        try:
            if trade_id not in self.active_positions:
                self.logger.error(f"Position {trade_id} not found")
                return False

            position = self.active_positions[trade_id]
            position.take_profit_price = round(
                new_take_profit, self.config.price_precision
            )

            self.logger.info(f"Take profit modified for {trade_id}: {new_take_profit}")
            return True

        except Exception as e:
            self.logger.error(f"Error modifying take profit: {e}")
            return False

    def get_position_info(self, trade_id: str) -> Optional[Dict]:
        """Get detailed position information"""
        try:
            if trade_id in self.active_positions:
                return self.active_positions[trade_id].to_dict()

            # Check closed positions
            for position in self.closed_positions:
                if position.trade_id == trade_id:
                    return position.to_dict()

            return None

        except Exception as e:
            self.logger.error(f"Error getting position info: {e}")
            return None

    def get_all_positions(self, include_closed: bool = False) -> Dict[str, Dict]:
        """Get all positions"""
        try:
            result = {}

            # Active positions
            for trade_id, position in self.active_positions.items():
                result[trade_id] = position.to_dict()

            # Closed positions if requested
            if include_closed:
                for position in self.closed_positions:
                    result[position.trade_id] = position.to_dict()

            return result

        except Exception as e:
            self.logger.error(f"Error getting all positions: {e}")
            return {}

    def get_performance_summary(self) -> Dict[str, any]:
        """Get performance summary"""
        try:
            active_count = len(self.active_positions)
            closed_count = len(self.closed_positions)

            win_rate = (
                (self.winning_trades / self.total_trades * 100)
                if self.total_trades > 0
                else 0
            )

            # Calculate average metrics from closed positions
            if closed_count > 0:
                avg_profit = (
                    sum(pos.current_profit for pos in self.closed_positions)
                    / closed_count
                )
                avg_duration = (
                    sum(
                        (pos.exit_time - pos.entry_time).total_seconds()
                        for pos in self.closed_positions
                        if pos.exit_time
                    )
                    / closed_count
                    / 3600
                )  # Convert to hours
            else:
                avg_profit = 0
                avg_duration = 0

            summary = {
                "total_trades": self.total_trades,
                "active_positions": active_count,
                "closed_positions": closed_count,
                "winning_trades": self.winning_trades,
                "win_rate_pct": win_rate,
                "total_profit": self.total_profit,
                "average_profit": avg_profit,
                "average_duration_hours": avg_duration,
                "is_running": self.is_running,
            }

            return summary

        except Exception as e:
            self.logger.error(f"Error generating performance summary: {e}")
            return {}

    def export_trade_history(self) -> pd.DataFrame:
        """Export trade history to DataFrame"""
        try:
            if not self.closed_positions:
                return pd.DataFrame()

            data = [pos.to_dict() for pos in self.closed_positions]
            df = pd.DataFrame(data)

            return df

        except Exception as e:
            self.logger.error(f"Error exporting trade history: {e}")
            return pd.DataFrame()


# Helper functions for creating positions
def create_position_from_signal(
    signal, trade_id: str, volume: float = 0.1
) -> TradePosition:
    """Create TradePosition from Signal"""
    try:
        trade_type = TradeType.BUY if signal.signal_type.value > 0 else TradeType.SELL

        position = TradePosition(
            trade_id=trade_id,
            symbol=signal.symbol,
            trade_type=trade_type,
            entry_price=signal.entry_price,
            volume=volume,
            entry_time=signal.timestamp,
            stop_loss_price=signal.stop_loss,
            take_profit_price=signal.take_profit,
            metadata={
                "signal_source": signal.source.value,
                "signal_strength": signal.strength,
                "signal_confidence": signal.confidence,
                "risk_reward_ratio": signal.risk_reward_ratio,
            },
        )

        return position

    except Exception as e:
        logging.getLogger(__name__).error(f"Error creating position from signal: {e}")
        return None


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create trade manager
    config = TradeManagerConfig()
    manager = TradeManager(config)

    # Create sample position
    position = TradePosition(
        trade_id="TEST_001",
        symbol="EURUSD",
        trade_type=TradeType.BUY,
        entry_price=1.1000,
        volume=0.1,
        entry_time=datetime.now(),
        stop_loss_price=1.0900,
        take_profit_price=1.1200,
    )

    # Add position
    manager.add_position(position)

    # Start management
    manager.start_management()

    try:
        # Simulate price updates
        for i in range(10):
            price = 1.1000 + (i * 0.0010)  # Simulate upward movement
            manager.update_price("EURUSD", price - 0.0001, price + 0.0001)
            time.sleep(1)

        # Get performance summary
        summary = manager.get_performance_summary()
        print("\nPerformance Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")

        # Get position info
        position_info = manager.get_position_info("TEST_001")
        if position_info:
            print(f"\nPosition Info:")
            for key, value in position_info.items():
                print(f"  {key}: {value}")

    finally:
        # Stop management
        manager.stop_management()
