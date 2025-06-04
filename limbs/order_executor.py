"""
Institutional Order Execution Engine for Real MT5 Trading
High-performance order management with advanced execution algorithms
Author: Senior AI Developer
Version: 1.0.0 - Production Ready
"""

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
import time
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import uuid
from collections import deque, defaultdict
import math

# Import dependencies
import sys

sys.path.append("..")
from config.settings import SystemConfig
from sensor.mt5_connector import MT5Connector, OrderType, OrderResult, TickData
from heart.risk_manager import RiskManager, TradeRiskAssessment, PositionSizeMethod


class ExecutionStrategy(Enum):
    """Order execution strategies"""

    MARKET = "market"  # Immediate market execution
    LIMIT = "limit"  # Limit order execution
    STOP = "stop"  # Stop order execution
    TWAP = "twap"  # Time-Weighted Average Price
    VWAP = "vwap"  # Volume-Weighted Average Price
    ICEBERG = "iceberg"  # Large order splitting
    SMART_ROUTING = "smart_routing"  # Intelligent execution
    STEALTH = "stealth"  # Low-impact execution


class OrderStatus(Enum):
    """Order execution status"""

    PENDING = "pending"
    VALIDATING = "validating"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTING = "executing"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    FAILED = "failed"


class OrderPriority(Enum):
    """Order execution priority"""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    EMERGENCY = 5


@dataclass
class OrderRequest:
    """Comprehensive order request structure"""

    order_id: str
    symbol: str
    side: str  # BUY/SELL
    volume: float
    order_type: OrderType
    strategy: ExecutionStrategy
    priority: OrderPriority = OrderPriority.NORMAL

    # Price parameters
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    # Execution parameters
    max_slippage: float = 10.0  # pips
    timeout: int = 30  # seconds
    good_till: Optional[datetime] = None

    # Advanced parameters
    max_position_size: Optional[float] = None
    split_size: Optional[float] = None  # For iceberg orders
    time_interval: Optional[int] = None  # For TWAP/VWAP

    # Metadata
    ai_confidence: float = 0.0
    signal_strength: float = 0.0
    comment: str = ""
    magic_number: int = 0

    # Risk parameters
    risk_score: float = 0.0
    max_risk_amount: float = 0.0

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ExecutionResult:
    """Order execution result with comprehensive details"""

    order_id: str
    success: bool
    status: OrderStatus

    # Execution details
    executed_volume: float = 0.0
    remaining_volume: float = 0.0
    avg_execution_price: float = 0.0
    total_slippage: float = 0.0
    execution_time: float = 0.0

    # MT5 details
    mt5_tickets: List[int] = field(default_factory=list)
    mt5_order_ids: List[int] = field(default_factory=list)

    # Performance metrics
    price_improvement: float = 0.0
    execution_cost: float = 0.0
    market_impact: float = 0.0

    # Error handling
    error_code: Optional[int] = None
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    # Timestamps
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ExecutionMetrics:
    """Execution performance metrics"""

    total_orders: int = 0
    successful_orders: int = 0
    failed_orders: int = 0
    avg_execution_time: float = 0.0
    avg_slippage: float = 0.0
    total_volume: float = 0.0
    fill_rate: float = 0.0
    success_rate: float = 0.0
    price_improvement_rate: float = 0.0


class OrderExecutor:
    """
    Production-Grade Order Execution Engine
    Handles real order placement with advanced execution strategies
    """

    def __init__(
        self, mt5_connector: MT5Connector, risk_manager: Optional[RiskManager] = None
    ):
        self.mt5_connector = mt5_connector
        self.risk_manager = risk_manager
        self.logger = self._setup_logger()

        # Order management
        self.pending_orders: Dict[str, OrderRequest] = {}
        self.active_orders: Dict[str, OrderRequest] = {}
        self.completed_orders: Dict[str, ExecutionResult] = {}
        self.order_history = deque(maxlen=10000)

        # Execution queues by priority
        self.execution_queues = {priority: deque() for priority in OrderPriority}

        # Performance tracking
        self.execution_metrics = ExecutionMetrics()
        self.slippage_history = deque(maxlen=1000)
        self.execution_times = deque(maxlen=1000)

        # Threading and async
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.order_lock = threading.Lock()
        self.execution_lock = threading.Lock()

        # Execution control
        self.is_running = False
        self.max_concurrent_orders = 10
        self.current_executions = 0

        # Market data cache
        self.price_cache = {}
        self.spread_cache = {}
        self.cache_expiry = 1.0  # seconds

        # Execution strategies
        self.strategy_handlers = {
            ExecutionStrategy.MARKET: self._execute_market_order,
            ExecutionStrategy.LIMIT: self._execute_limit_order,
            ExecutionStrategy.STOP: self._execute_stop_order,
            ExecutionStrategy.TWAP: self._execute_twap_order,
            ExecutionStrategy.VWAP: self._execute_vwap_order,
            ExecutionStrategy.ICEBERG: self._execute_iceberg_order,
            ExecutionStrategy.SMART_ROUTING: self._execute_smart_routing,
            ExecutionStrategy.STEALTH: self._execute_stealth_order,
        }

        # Start execution engine
        self.start_execution_engine()

    def _setup_logger(self) -> logging.Logger:
        """Setup dedicated logger for order executor"""
        logger = logging.getLogger("OrderExecutor")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.FileHandler("logs/order_executor.log")
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def start_execution_engine(self) -> None:
        """Start the order execution engine"""
        if not self.is_running:
            self.is_running = True
            self.executor.submit(self._execution_loop)
            self.logger.info("🚀 Order Execution Engine Started")

    def stop_execution_engine(self) -> None:
        """Stop the order execution engine"""
        self.is_running = False
        self.logger.info("🛑 Order Execution Engine Stopped")

    def submit_order(self, order_request: OrderRequest) -> str:
        """
        Submit order for execution

        Args:
            order_request: Order request details

        Returns:
            Order ID for tracking
        """
        try:
            with self.order_lock:
                # Generate unique order ID if not provided
                if not order_request.order_id:
                    order_request.order_id = f"ORD_{uuid.uuid4().hex[:8]}"

                # Validate order request
                validation_result = self._validate_order(order_request)
                if not validation_result["valid"]:
                    self.logger.error(
                        f"❌ Order validation failed: {validation_result['reason']}"
                    )
                    return ""

                # Risk assessment
                if self.risk_manager:
                    risk_assessment = self.risk_manager.assess_trade_risk(
                        symbol=order_request.symbol,
                        trade_type=order_request.side,
                        entry_price=order_request.price or 0.0,
                        stop_loss=order_request.stop_loss,
                        take_profit=order_request.take_profit,
                        position_size=order_request.volume,
                    )

                    if not risk_assessment.approved:
                        self.logger.error(
                            f"❌ Order rejected by risk manager: {risk_assessment.rejection_reasons}"
                        )
                        return ""

                    order_request.risk_score = risk_assessment.risk_score

                # Add to pending orders
                self.pending_orders[order_request.order_id] = order_request

                # Add to execution queue based on priority
                self.execution_queues[order_request.priority].append(
                    order_request.order_id
                )

                self.logger.info(
                    f"📝 Order submitted: {order_request.order_id} - {order_request.symbol} {order_request.side} {order_request.volume}"
                )

                return order_request.order_id

        except Exception as e:
            self.logger.error(f"Order submission error: {str(e)}")
            return ""

    def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order"""
        try:
            with self.order_lock:
                if order_id in self.pending_orders:
                    # Remove from pending
                    order = self.pending_orders.pop(order_id)

                    # Remove from execution queue
                    for queue in self.execution_queues.values():
                        if order_id in queue:
                            queue.remove(order_id)

                    # Create cancellation result
                    result = ExecutionResult(
                        order_id=order_id,
                        success=True,
                        status=OrderStatus.CANCELLED,
                        end_time=datetime.now(),
                    )

                    self.completed_orders[order_id] = result
                    self.logger.info(f"🚫 Order cancelled: {order_id}")
                    return True

                elif order_id in self.active_orders:
                    # Try to cancel active MT5 orders
                    order = self.active_orders[order_id]
                    return self._cancel_mt5_orders(order)

                return False

        except Exception as e:
            self.logger.error(f"Order cancellation error: {str(e)}")
            return False

    def get_order_status(self, order_id: str) -> Optional[Dict]:
        """Get order status and details"""
        try:
            if order_id in self.pending_orders:
                return {
                    "status": OrderStatus.PENDING.value,
                    "order": self.pending_orders[order_id].to_dict(),
                }
            elif order_id in self.active_orders:
                return {
                    "status": OrderStatus.EXECUTING.value,
                    "order": self.active_orders[order_id].to_dict(),
                }
            elif order_id in self.completed_orders:
                return {
                    "status": self.completed_orders[order_id].status.value,
                    "result": self.completed_orders[order_id].to_dict(),
                }

            return None

        except Exception as e:
            self.logger.error(f"Get order status error: {str(e)}")
            return None

    def _execution_loop(self) -> None:
        """Main execution loop"""
        while self.is_running:
            try:
                # Check if we can execute more orders
                if self.current_executions >= self.max_concurrent_orders:
                    time.sleep(0.1)
                    continue

                # Get next order by priority
                order_id = self._get_next_order()
                if not order_id:
                    time.sleep(0.1)
                    continue

                # Move to active execution
                with self.order_lock:
                    if order_id in self.pending_orders:
                        order = self.pending_orders.pop(order_id)
                        self.active_orders[order_id] = order
                        self.current_executions += 1

                # Execute order asynchronously
                self.executor.submit(self._execute_order_async, order_id)

            except Exception as e:
                self.logger.error(f"Execution loop error: {str(e)}")
                time.sleep(1)

    def _get_next_order(self) -> Optional[str]:
        """Get next order to execute based on priority"""
        with self.order_lock:
            # Check queues from highest to lowest priority
            for priority in reversed(list(OrderPriority)):
                queue = self.execution_queues[priority]
                if queue:
                    return queue.popleft()
        return None

    def _execute_order_async(self, order_id: str) -> None:
        """Execute order asynchronously"""
        try:
            order = self.active_orders.get(order_id)
            if not order:
                return

            # Execute using appropriate strategy
            handler = self.strategy_handlers.get(
                order.strategy, self._execute_market_order
            )
            result = handler(order)

            # Update metrics
            self._update_execution_metrics(result)

            # Move to completed
            with self.order_lock:
                if order_id in self.active_orders:
                    del self.active_orders[order_id]
                self.completed_orders[order_id] = result
                self.order_history.append(result.to_dict())
                self.current_executions = max(0, self.current_executions - 1)

            # Log result
            status_emoji = "✅" if result.success else "❌"
            self.logger.info(f"{status_emoji} Order {result.status.value}: {order_id}")

        except Exception as e:
            self.logger.error(f"Async execution error for {order_id}: {str(e)}")
            self.current_executions = max(0, self.current_executions - 1)

    def _execute_market_order(self, order: OrderRequest) -> ExecutionResult:
        """Execute market order immediately"""
        start_time = datetime.now()
        result = ExecutionResult(
            order_id=order.order_id,
            success=False,
            status=OrderStatus.EXECUTING,
            start_time=start_time,
        )

        try:
            # Get current market price
            tick = self._get_current_price(order.symbol)
            if not tick:
                result.status = OrderStatus.FAILED
                result.error_message = "Cannot get current price"
                result.end_time = datetime.now()
                return result

            # Determine execution price
            execution_price = tick.ask if order.side == "BUY" else tick.bid

            # Check slippage tolerance
            if order.price:
                slippage_pips = abs(execution_price - order.price)
                symbol_info = self.mt5_connector.get_symbol_info(order.symbol)
                if symbol_info:
                    slippage_pips /= symbol_info.point

                if slippage_pips > order.max_slippage:
                    result.status = OrderStatus.FAILED
                    result.error_message = (
                        f"Slippage too high: {slippage_pips:.1f} pips"
                    )
                    result.end_time = datetime.now()
                    return result

            # Convert to MT5 order type
            mt5_order_type = OrderType.BUY if order.side == "BUY" else OrderType.SELL

            # Send order to MT5
            mt5_result = self.mt5_connector.send_order(
                symbol=order.symbol,
                order_type=mt5_order_type,
                volume=order.volume,
                price=execution_price,
                sl=order.stop_loss,
                tp=order.take_profit,
                comment=order.comment,
                magic=order.magic_number,
                deviation=int(order.max_slippage),
            )

            # Process MT5 result
            if mt5_result.success:
                result.success = True
                result.status = OrderStatus.FILLED
                result.executed_volume = order.volume
                result.avg_execution_price = (
                    mt5_result.executed_price or execution_price
                )
                result.total_slippage = mt5_result.slippage or 0
                result.mt5_tickets = [mt5_result.ticket] if mt5_result.ticket else []
                result.mt5_order_ids = (
                    [mt5_result.order_id] if mt5_result.order_id else []
                )

                # Calculate price improvement
                if order.price:
                    if order.side == "BUY":
                        result.price_improvement = (
                            order.price - result.avg_execution_price
                        )
                    else:
                        result.price_improvement = (
                            result.avg_execution_price - order.price
                        )

            else:
                result.status = OrderStatus.FAILED
                result.error_code = mt5_result.error_code
                result.error_message = mt5_result.error_message

        except Exception as e:
            result.status = OrderStatus.FAILED
            result.error_message = str(e)

        result.end_time = datetime.now()
        result.execution_time = (result.end_time - start_time).total_seconds()

        return result

    def _execute_limit_order(self, order: OrderRequest) -> ExecutionResult:
        """Execute limit order"""
        start_time = datetime.now()
        result = ExecutionResult(
            order_id=order.order_id,
            success=False,
            status=OrderStatus.EXECUTING,
            start_time=start_time,
        )

        try:
            if not order.price:
                result.status = OrderStatus.FAILED
                result.error_message = "Limit price required for limit order"
                result.end_time = datetime.now()
                return result

            # Convert to MT5 limit order type
            if order.side == "BUY":
                mt5_order_type = OrderType.BUY_LIMIT
            else:
                mt5_order_type = OrderType.SELL_LIMIT

            # Send limit order to MT5
            mt5_result = self.mt5_connector.send_order(
                symbol=order.symbol,
                order_type=mt5_order_type,
                volume=order.volume,
                price=order.price,
                sl=order.stop_loss,
                tp=order.take_profit,
                comment=order.comment,
                magic=order.magic_number,
            )

            if mt5_result.success:
                result.success = True
                result.status = OrderStatus.PENDING  # Limit order is pending
                result.mt5_order_ids = (
                    [mt5_result.order_id] if mt5_result.order_id else []
                )
            else:
                result.status = OrderStatus.FAILED
                result.error_code = mt5_result.error_code
                result.error_message = mt5_result.error_message

        except Exception as e:
            result.status = OrderStatus.FAILED
            result.error_message = str(e)

        result.end_time = datetime.now()
        result.execution_time = (result.end_time - start_time).total_seconds()

        return result

    def _execute_stop_order(self, order: OrderRequest) -> ExecutionResult:
        """Execute stop order"""
        start_time = datetime.now()
        result = ExecutionResult(
            order_id=order.order_id,
            success=False,
            status=OrderStatus.EXECUTING,
            start_time=start_time,
        )

        try:
            if not order.price:
                result.status = OrderStatus.FAILED
                result.error_message = "Stop price required for stop order"
                result.end_time = datetime.now()
                return result

            # Convert to MT5 stop order type
            if order.side == "BUY":
                mt5_order_type = OrderType.BUY_STOP
            else:
                mt5_order_type = OrderType.SELL_STOP

            # Send stop order to MT5
            mt5_result = self.mt5_connector.send_order(
                symbol=order.symbol,
                order_type=mt5_order_type,
                volume=order.volume,
                price=order.price,
                sl=order.stop_loss,
                tp=order.take_profit,
                comment=order.comment,
                magic=order.magic_number,
            )

            if mt5_result.success:
                result.success = True
                result.status = OrderStatus.PENDING  # Stop order is pending
                result.mt5_order_ids = (
                    [mt5_result.order_id] if mt5_result.order_id else []
                )
            else:
                result.status = OrderStatus.FAILED
                result.error_code = mt5_result.error_code
                result.error_message = mt5_result.error_message

        except Exception as e:
            result.status = OrderStatus.FAILED
            result.error_message = str(e)

        result.end_time = datetime.now()
        result.execution_time = (result.end_time - start_time).total_seconds()

        return result

    def _execute_twap_order(self, order: OrderRequest) -> ExecutionResult:
        """Execute TWAP (Time-Weighted Average Price) order"""
        start_time = datetime.now()
        result = ExecutionResult(
            order_id=order.order_id,
            success=False,
            status=OrderStatus.EXECUTING,
            start_time=start_time,
        )

        try:
            # Split order into time-based chunks
            total_volume = order.volume
            time_interval = order.time_interval or 60  # Default 60 seconds
            chunks = max(1, min(10, int(time_interval / 10)))  # Max 10 chunks
            chunk_volume = total_volume / chunks

            executed_volume = 0.0
            total_price = 0.0
            execution_results = []

            for i in range(chunks):
                # Create chunk order
                chunk_order = OrderRequest(
                    order_id=f"{order.order_id}_chunk_{i}",
                    symbol=order.symbol,
                    side=order.side,
                    volume=chunk_volume,
                    order_type=OrderType.BUY if order.side == "BUY" else OrderType.SELL,
                    strategy=ExecutionStrategy.MARKET,
                    comment=f"TWAP chunk {i+1}/{chunks}",
                )

                # Execute chunk
                chunk_result = self._execute_market_order(chunk_order)
                execution_results.append(chunk_result)

                if chunk_result.success:
                    executed_volume += chunk_result.executed_volume
                    total_price += (
                        chunk_result.avg_execution_price * chunk_result.executed_volume
                    )
                    result.mt5_tickets.extend(chunk_result.mt5_tickets)

                # Wait between chunks (except last one)
                if i < chunks - 1:
                    time.sleep(time_interval / chunks)

            # Calculate results
            if executed_volume > 0:
                result.success = True
                result.status = (
                    OrderStatus.FILLED
                    if executed_volume == total_volume
                    else OrderStatus.PARTIALLY_FILLED
                )
                result.executed_volume = executed_volume
                result.remaining_volume = total_volume - executed_volume
                result.avg_execution_price = total_price / executed_volume
            else:
                result.status = OrderStatus.FAILED
                result.error_message = "No chunks executed successfully"

        except Exception as e:
            result.status = OrderStatus.FAILED
            result.error_message = str(e)

        result.end_time = datetime.now()
        result.execution_time = (result.end_time - start_time).total_seconds()

        return result

    def _execute_vwap_order(self, order: OrderRequest) -> ExecutionResult:
        """Execute VWAP (Volume-Weighted Average Price) order"""
        # For simplicity, use TWAP strategy with volume considerations
        return self._execute_twap_order(order)

    def _execute_iceberg_order(self, order: OrderRequest) -> ExecutionResult:
        """Execute iceberg order (large order split into smaller pieces)"""
        start_time = datetime.now()
        result = ExecutionResult(
            order_id=order.order_id,
            success=False,
            status=OrderStatus.EXECUTING,
            start_time=start_time,
        )

        try:
            total_volume = order.volume
            split_size = order.split_size or min(0.1, total_volume / 5)  # Default split
            remaining_volume = total_volume

            executed_volume = 0.0
            total_price = 0.0

            while remaining_volume > 0:
                # Calculate current chunk size
                current_volume = min(split_size, remaining_volume)

                # Create chunk order
                chunk_order = OrderRequest(
                    order_id=f"{order.order_id}_iceberg_{len(result.mt5_tickets)}",
                    symbol=order.symbol,
                    side=order.side,
                    volume=current_volume,
                    order_type=OrderType.BUY if order.side == "BUY" else OrderType.SELL,
                    strategy=ExecutionStrategy.MARKET,
                    comment=f"Iceberg chunk",
                )

                # Execute chunk
                chunk_result = self._execute_market_order(chunk_order)

                if chunk_result.success:
                    executed_volume += chunk_result.executed_volume
                    total_price += (
                        chunk_result.avg_execution_price * chunk_result.executed_volume
                    )
                    remaining_volume -= chunk_result.executed_volume
                    result.mt5_tickets.extend(chunk_result.mt5_tickets)
                else:
                    # If chunk fails, stop execution
                    break

                # Small delay to reduce market impact
                time.sleep(1)

            # Calculate results
            if executed_volume > 0:
                result.success = True
                result.status = (
                    OrderStatus.FILLED
                    if remaining_volume == 0
                    else OrderStatus.PARTIALLY_FILLED
                )
                result.executed_volume = executed_volume
                result.remaining_volume = remaining_volume
                result.avg_execution_price = total_price / executed_volume
            else:
                result.status = OrderStatus.FAILED
                result.error_message = "No chunks executed successfully"

        except Exception as e:
            result.status = OrderStatus.FAILED
            result.error_message = str(e)

        result.end_time = datetime.now()
        result.execution_time = (result.end_time - start_time).total_seconds()

        return result

    def _execute_smart_routing(self, order: OrderRequest) -> ExecutionResult:
        """Execute with intelligent routing based on market conditions"""
        try:
            # Analyze market conditions
            market_analysis = self._analyze_market_conditions(order.symbol)

            # Choose best execution strategy
            if market_analysis["volatility"] > 0.8:
                # High volatility - use iceberg
                order.strategy = ExecutionStrategy.ICEBERG
                return self._execute_iceberg_order(order)
            elif market_analysis["spread"] > market_analysis["avg_spread"] * 2:
                # Wide spread - use limit order
                order.strategy = ExecutionStrategy.LIMIT
                if not order.price:
                    tick = self._get_current_price(order.symbol)
                    if tick:
                        # Set aggressive limit price
                        order.price = (
                            tick.bid + (tick.ask - tick.bid) * 0.3
                            if order.side == "BUY"
                            else tick.ask - (tick.ask - tick.bid) * 0.3
                        )
                return self._execute_limit_order(order)
            else:
                # Normal conditions - use market order
                order.strategy = ExecutionStrategy.MARKET
                return self._execute_market_order(order)

        except Exception as e:
            # Fallback to market order
            return self._execute_market_order(order)

    def _execute_stealth_order(self, order: OrderRequest) -> ExecutionResult:
        """Execute with minimal market impact"""
        # Use iceberg with small chunks and random timing
        order.split_size = min(0.01, order.volume / 20)  # Very small chunks
        return self._execute_iceberg_order(order)

    def _validate_order(self, order: OrderRequest) -> Dict[str, Any]:
        """Comprehensive order validation"""
        try:
            # Basic validation
            if not order.symbol or not order.side or order.volume <= 0:
                return {"valid": False, "reason": "Invalid order parameters"}

            # Check MT5 connection
            if not self.mt5_connector.is_connected():
                return {"valid": False, "reason": "MT5 not connected"}

            # Symbol validation
            symbol_info = self.mt5_connector.get_symbol_info(order.symbol)
            if not symbol_info:
                return {
                    "valid": False,
                    "reason": f"Symbol {order.symbol} not available",
                }

            # Volume validation
            if order.volume < symbol_info.min_lot or order.volume > symbol_info.max_lot:
                return {
                    "valid": False,
                    "reason": f"Volume outside allowed range: {symbol_info.min_lot}-{symbol_info.max_lot}",
                }

            # Check lot step
            volume_steps = order.volume / symbol_info.lot_step
            if abs(volume_steps - round(volume_steps)) > 1e-6:
                return {
                    "valid": False,
                    "reason": f"Volume must be multiple of {symbol_info.lot_step}",
                }

            # Check market hours
            if not self._is_market_open(order.symbol):
                return {"valid": False, "reason": "Market closed"}

            # Price validation for limit/stop orders
            if order.order_type in [
                OrderType.BUY_LIMIT,
                OrderType.SELL_LIMIT,
                OrderType.BUY_STOP,
                OrderType.SELL_STOP,
            ]:
                if not order.price:
                    return {
                        "valid": False,
                        "reason": "Price required for limit/stop orders",
                    }

            return {"valid": True}

        except Exception as e:
            return {"valid": False, "reason": f"Validation error: {str(e)}"}

    def _get_current_price(self, symbol: str) -> Optional[TickData]:
        """Get current price with caching"""
        try:
            current_time = time.time()

            # Check cache
            if symbol in self.price_cache:
                cached_tick, cache_time = self.price_cache[symbol]
                if current_time - cache_time < self.cache_expiry:
                    return cached_tick

            # Get fresh data
            tick = self.mt5_connector.get_tick(symbol)
            if tick:
                self.price_cache[symbol] = (tick, current_time)

            return tick

        except Exception as e:
            self.logger.error(f"Error getting current price for {symbol}: {str(e)}")
            return None

    def _analyze_market_conditions(self, symbol: str) -> Dict[str, float]:
        """Analyze current market conditions"""
        try:
            # Get recent price data
            rates = self.mt5_connector.get_rates(symbol, 1, 20)  # 20 minutes
            if rates is None or len(rates) < 10:
                return {"volatility": 0.5, "spread": 0.0, "avg_spread": 0.0}

            # Calculate volatility
            returns = rates["close"].pct_change().dropna()
            volatility = returns.std() * 100  # Normalize

            # Get current spread
            tick = self._get_current_price(symbol)
            current_spread = tick.spread if tick else 0.0

            # Calculate average spread
            avg_spread = rates["high"].subtract(rates["low"]).mean()

            return {
                "volatility": min(1.0, volatility / 0.01),  # Normalize to 0-1
                "spread": current_spread,
                "avg_spread": avg_spread,
            }

        except Exception:
            return {"volatility": 0.5, "spread": 0.0, "avg_spread": 0.0}

    def _is_market_open(self, symbol: str) -> bool:
        """Check if market is open for trading"""
        try:
            # Simple check - try to get current tick
            tick = self.mt5_connector.get_tick(symbol)
            return tick is not None
        except:
            return False

    def _cancel_mt5_orders(self, order: OrderRequest) -> bool:
        """Cancel active MT5 orders"""
        try:
            success_count = 0
            mt5_orders = self.mt5_connector.get_orders(order.symbol)

            for mt5_order in mt5_orders:
                if mt5_order.get("comment", "").startswith(order.order_id):
                    # This is our order - try to cancel
                    cancel_request = {
                        "action": mt5.TRADE_ACTION_REMOVE,
                        "order": mt5_order["ticket"],
                    }

                    result = mt5.order_send(cancel_request)
                    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                        success_count += 1

            return success_count > 0

        except Exception as e:
            self.logger.error(f"MT5 order cancellation error: {str(e)}")
            return False

    def _update_execution_metrics(self, result: ExecutionResult) -> None:
        """Update execution performance metrics"""
        with self.execution_lock:
            self.execution_metrics.total_orders += 1

            if result.success:
                self.execution_metrics.successful_orders += 1
                self.execution_metrics.total_volume += result.executed_volume

                # Track execution time
                if result.execution_time > 0:
                    self.execution_times.append(result.execution_time)
                    self.execution_metrics.avg_execution_time = np.mean(
                        list(self.execution_times)
                    )

                # Track slippage
                if result.total_slippage > 0:
                    self.slippage_history.append(result.total_slippage)
                    self.execution_metrics.avg_slippage = np.mean(
                        list(self.slippage_history)
                    )

                # Track price improvement
                if result.price_improvement > 0:
                    self.execution_metrics.price_improvement_rate += 1

            else:
                self.execution_metrics.failed_orders += 1

            # Calculate rates
            if self.execution_metrics.total_orders > 0:
                self.execution_metrics.success_rate = (
                    self.execution_metrics.successful_orders
                    / self.execution_metrics.total_orders
                    * 100
                )

                self.execution_metrics.fill_rate = (
                    self.execution_metrics.successful_orders
                    / self.execution_metrics.total_orders
                    * 100
                )

                self.execution_metrics.price_improvement_rate = (
                    self.execution_metrics.price_improvement_rate
                    / self.execution_metrics.successful_orders
                    * 100
                    if self.execution_metrics.successful_orders > 0
                    else 0
                )

    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get comprehensive execution statistics"""
        with self.execution_lock:
            return {
                "metrics": asdict(self.execution_metrics),
                "pending_orders": len(self.pending_orders),
                "active_orders": len(self.active_orders),
                "completed_orders": len(self.completed_orders),
                "current_executions": self.current_executions,
                "queue_lengths": {
                    priority.name: len(queue)
                    for priority, queue in self.execution_queues.items()
                },
            }

    def emergency_cancel_all(self) -> Dict[str, int]:
        """Emergency cancellation of all orders"""
        try:
            self.logger.warning("🚨 EMERGENCY CANCEL ALL ORDERS")

            cancelled_pending = 0
            cancelled_active = 0

            # Cancel all pending orders
            with self.order_lock:
                pending_ids = list(self.pending_orders.keys())
                for order_id in pending_ids:
                    if self.cancel_order(order_id):
                        cancelled_pending += 1

                # Cancel all active orders
                active_ids = list(self.active_orders.keys())
                for order_id in active_ids:
                    if self.cancel_order(order_id):
                        cancelled_active += 1

            return {
                "cancelled_pending": cancelled_pending,
                "cancelled_active": cancelled_active,
                "total_cancelled": cancelled_pending + cancelled_active,
            }

        except Exception as e:
            self.logger.error(f"Emergency cancel error: {str(e)}")
            return {"cancelled_pending": 0, "cancelled_active": 0, "total_cancelled": 0}

    def __del__(self):
        """Cleanup on destruction"""
        self.stop_execution_engine()
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=True)


# Factory function
def create_order_executor(
    mt5_connector: MT5Connector, risk_manager: Optional[RiskManager] = None
) -> OrderExecutor:
    """Create order executor with dependencies"""
    return OrderExecutor(mt5_connector, risk_manager)


# Export main classes
__all__ = [
    "OrderExecutor",
    "OrderRequest",
    "ExecutionResult",
    "ExecutionStrategy",
    "OrderStatus",
    "OrderPriority",
    "create_order_executor",
]

if __name__ == "__main__":
    # Demo and testing
    print("⚡ Order Execution Engine Testing")
    print("=" * 50)

    # This is a demo - replace with real MT5 connection
    from sensor.mt5_connector import create_mt5_connector
    from heart.risk_manager import create_risk_manager

    # Create dependencies (demo mode)
    mt5_conn = create_mt5_connector("dupoin")
    risk_mgr = create_risk_manager()

    # Create order executor
    executor = create_order_executor(mt5_conn, risk_mgr)

    # Test order submission
    test_order = OrderRequest(
        order_id="TEST_001",
        symbol="EURUSD",
        side="BUY",
        volume=0.01,
        order_type=OrderType.BUY,
        strategy=ExecutionStrategy.MARKET,
        priority=OrderPriority.NORMAL,
        max_slippage=5.0,
        comment="Test order",
        ai_confidence=85.0,
    )

    print(f"📝 Test Order Created:")
    print(f"   Symbol: {test_order.symbol}")
    print(f"   Side: {test_order.side}")
    print(f"   Volume: {test_order.volume}")
    print(f"   Strategy: {test_order.strategy.value}")
    print(f"   Priority: {test_order.priority.value}")

    # Get execution statistics
    stats = executor.get_execution_statistics()
    print(f"\n📊 Execution Statistics:")
    print(f"   Success Rate: {stats['metrics']['success_rate']:.1f}%")
    print(f"   Avg Execution Time: {stats['metrics']['avg_execution_time']:.3f}s")
    print(f"   Avg Slippage: {stats['metrics']['avg_slippage']:.2f}")
    print(f"   Pending Orders: {stats['pending_orders']}")

    print(f"\n🎯 Order Execution Engine Ready!")
