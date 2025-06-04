"""
MT5 Real Account Connector for Institutional Forex Trading System
Universal connector supporting all MT5 brokers (Dupoin, Monaxa, etc.)
Author: Senior AI Developer
Version: 1.0.0 - Production Ready
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, asdict
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import asyncio
from enum import Enum

# Import configurations
import sys

sys.path.append("..")
from config.mt5_config import MT5Config, get_broker_config


class ConnectionStatus(Enum):
    """MT5 Connection Status"""
    
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    RECONNECTING = "reconnecting"


class OrderType(Enum):
    """MT5 Order Types"""

    BUY = mt5.ORDER_TYPE_BUY
    SELL = mt5.ORDER_TYPE_SELL
    BUY_LIMIT = mt5.ORDER_TYPE_BUY_LIMIT
    SELL_LIMIT = mt5.ORDER_TYPE_SELL_LIMIT
    BUY_STOP = mt5.ORDER_TYPE_BUY_STOP
    SELL_STOP = mt5.ORDER_TYPE_SELL_STOP


@dataclass
class TickData:
    """Real-time tick data structure"""

    symbol: str
    time: datetime
    bid: float
    ask: float
    last: float
    volume: int
    spread: float

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class MarketInfo:
    """Market information structure"""

    symbol: str
    digits: int
    spread: float
    point: float
    min_lot: float
    max_lot: float
    lot_step: float
    margin_required: float
    tick_value: float
    tick_size: float
    swap_long: float
    swap_short: float
    session_open: str
    session_close: str


@dataclass
class OrderResult:
    """Order execution result"""

    success: bool
    order_id: Optional[int] = None
    ticket: Optional[int] = None
    error_code: Optional[int] = None
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    executed_price: Optional[float] = None
    slippage: Optional[float] = None


class MT5Connector:
    """
    Universal MT5 Connector for Real Account Trading
    Supports all MT5 brokers with production-grade reliability
    """

    def __init__(self, config: Optional[MT5Config] = None):
        self.config = config or MT5Config()
        self.broker_config = get_broker_config(self.config.broker_name)
        self.logger = self._setup_logger()

        # Connection state
        self.status = ConnectionStatus.DISCONNECTED
        self.account_info = None
        self.symbols_info = {}
        self.last_tick_time = {}

        # Threading and async support
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.connection_lock = threading.Lock()
        self.data_lock = threading.Lock()

        # Reconnection handling
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 5

        # Performance monitoring
        self.connection_start_time = None
        self.total_requests = 0
        self.failed_requests = 0
        self.avg_response_time = 0.0

        # Data caching
        self.tick_cache = {}
        self.rate_cache = {}
        self.cache_expiry = 1  # seconds

    def _setup_logger(self) -> logging.Logger:
        """Setup dedicated logger for MT5 connector"""
        logger = logging.getLogger(f"MT5Connector_{self.config.broker_name}")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.FileHandler(
                f"logs/mt5_connector_{self.config.broker_name}.log"
            )
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def connect(self) -> bool:
        """
        Connect to MT5 terminal with broker-specific settings
        Returns True if connection successful
        """
        try:
            with self.connection_lock:
                self.status = ConnectionStatus.CONNECTING
                self.logger.info(
                    f"Connecting to MT5 - Broker: {self.config.broker_name}"
                )

                # Initialize MT5 connection
                if not mt5.initialize(
                    path=self.broker_config.terminal_path,
                    login=self.config.login,
                    password=self.config.password,
                    server=self.broker_config.server,
                    timeout=self.config.timeout,
                    portable=self.config.portable,
                ):
                    error = mt5.last_error()
                    self.logger.error(f"MT5 initialization failed: {error}")
                    self.status = ConnectionStatus.ERROR
                    return False

                # Verify connection
                account_info = mt5.account_info()
                if account_info is None:
                    self.logger.error("Failed to get account info")
                    self.status = ConnectionStatus.ERROR
                    return False

                self.account_info = account_info._asdict()
                self.status = ConnectionStatus.CONNECTED
                self.connection_start_time = datetime.now()
                self.reconnect_attempts = 0

                self.logger.info(
                    f"✅ Connected successfully to {self.config.broker_name}"
                )
                self.logger.info(f"Account: {self.account_info['login']}")
                self.logger.info(f"Balance: {self.account_info['balance']:.2f}")
                self.logger.info(f"Equity: {self.account_info['equity']:.2f}")

                # Initialize symbols
                self._initialize_symbols()

                return True

        except Exception as e:
            self.logger.error(f"Connection error: {str(e)}")
            self.status = ConnectionStatus.ERROR
            return False

    def disconnect(self) -> bool:
        """Disconnect from MT5 terminal"""
        try:
            with self.connection_lock:
                if self.status == ConnectionStatus.CONNECTED:
                    mt5.shutdown()
                    self.status = ConnectionStatus.DISCONNECTED
                    self.logger.info("✅ Disconnected from MT5")
                return True
        except Exception as e:
            self.logger.error(f"Disconnect error: {str(e)}")
            return False

    def _initialize_symbols(self) -> None:
        """Initialize trading symbols information"""
        try:
            for symbol in self.config.symbols:
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info is not None:
                    # Select symbol for trading
                    if not mt5.symbol_select(symbol, True):
                        self.logger.warning(f"Failed to select symbol: {symbol}")
                        continue

                    # Store symbol information
                    self.symbols_info[symbol] = MarketInfo(
                        symbol=symbol,
                        digits=symbol_info.digits,
                        spread=symbol_info.spread,
                        point=symbol_info.point,
                        min_lot=symbol_info.volume_min,
                        max_lot=symbol_info.volume_max,
                        lot_step=symbol_info.volume_step,
                        margin_required=symbol_info.margin_initial,
                        tick_value=symbol_info.trade_tick_value,
                        tick_size=symbol_info.trade_tick_size,
                        swap_long=symbol_info.swap_long,
                        swap_short=symbol_info.swap_short,
                        session_open=str(
                            symbol_info.sessions_aw[0][0]
                            if symbol_info.sessions_aw
                            else "00:00"
                        ),
                        session_close=str(
                            symbol_info.sessions_aw[0][1]
                            if symbol_info.sessions_aw
                            else "23:59"
                        ),
                    )

                    self.logger.info(f"✅ Symbol initialized: {symbol}")
                else:
                    self.logger.warning(f"❌ Symbol not found: {symbol}")

        except Exception as e:
            self.logger.error(f"Symbol initialization error: {str(e)}")

    def is_connected(self) -> bool:
        """Check if connection is active"""
        if self.status != ConnectionStatus.CONNECTED:
            return False

        try:
            # Test connection with account info request
            account = mt5.account_info()
            return account is not None
        except:
            self.status = ConnectionStatus.ERROR
            return False

    def auto_reconnect(self) -> bool:
        """Automatic reconnection with exponential backoff"""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            self.logger.error("Max reconnection attempts reached")
            return False

        self.reconnect_attempts += 1
        self.status = ConnectionStatus.RECONNECTING

        # Exponential backoff
        delay = self.reconnect_delay * (2 ** (self.reconnect_attempts - 1))
        self.logger.info(
            f"Reconnection attempt {self.reconnect_attempts}/{self.max_reconnect_attempts} in {delay}s"
        )

        time.sleep(delay)

        # Disconnect first
        try:
            mt5.shutdown()
        except:
            pass

        # Attempt reconnection
        return self.connect()

    def get_tick(self, symbol: str) -> Optional[TickData]:
        """Get real-time tick data for symbol"""
        if not self.is_connected():
            if not self.auto_reconnect():
                return None

        try:
            # Check cache first
            cache_key = f"tick_{symbol}"
            current_time = time.time()

            if cache_key in self.tick_cache:
                cached_data, cache_time = self.tick_cache[cache_key]
                if current_time - cache_time < self.cache_expiry:
                    return cached_data

            # Get fresh tick data
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                self.logger.warning(f"No tick data for {symbol}")
                return None

            tick_data = TickData(
                symbol=symbol,
                time=datetime.fromtimestamp(tick.time),
                bid=tick.bid,
                ask=tick.ask,
                last=tick.last,
                volume=tick.volume,
                spread=(
                    (tick.ask - tick.bid) / self.symbols_info[symbol].point
                    if symbol in self.symbols_info
                    else 0
                ),
            )

            # Cache the data
            self.tick_cache[cache_key] = (tick_data, current_time)
            self.last_tick_time[symbol] = tick_data.time

            return tick_data

        except Exception as e:
            self.logger.error(f"Error getting tick for {symbol}: {str(e)}")
            self.failed_requests += 1
            return None

    def get_rates(
        self, symbol: str, timeframe: int, count: int = 100
    ) -> Optional[pd.DataFrame]:
        """
        Get historical rate data

        Args:
            symbol: Trading symbol
            timeframe: MT5 timeframe constant
            count: Number of bars to retrieve

        Returns:
            DataFrame with OHLCV data
        """
        if not self.is_connected():
            if not self.auto_reconnect():
                return None

        try:
            start_time = time.time()

            # Get rates from MT5
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)

            if rates is None or len(rates) == 0:
                self.logger.warning(f"No rate data for {symbol}")
                return None

            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df.set_index("time", inplace=True)

            # Add spread information if available
            if symbol in self.symbols_info:
                df["spread"] = self.symbols_info[symbol].spread

            # Performance tracking
            execution_time = time.time() - start_time
            self.total_requests += 1
            self.avg_response_time = (
                self.avg_response_time * (self.total_requests - 1) + execution_time
            ) / self.total_requests

            return df

        except Exception as e:
            self.logger.error(f"Error getting rates for {symbol}: {str(e)}")
            self.failed_requests += 1
            return None

    def send_order(
        self,
        symbol: str,
        order_type: OrderType,
        volume: float,
        price: Optional[float] = None,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        comment: str = "",
        magic: int = 0,
        deviation: int = 10,
    ) -> OrderResult:
        """
        Send trading order to MT5

        Args:
            symbol: Trading symbol
            order_type: Order type (BUY, SELL, etc.)
            volume: Trading volume in lots
            price: Order price (for pending orders)
            sl: Stop loss price
            tp: Take profit price
            comment: Order comment
            magic: Magic number
            deviation: Price deviation in points

        Returns:
            OrderResult with execution details
        """
        if not self.is_connected():
            if not self.auto_reconnect():
                return OrderResult(False, error_message="Not connected to MT5")

        try:
            start_time = time.time()

            # Get current prices for market orders
            if order_type in [OrderType.BUY, OrderType.SELL] and price is None:
                tick = self.get_tick(symbol)
                if tick is None:
                    return OrderResult(False, error_message="Cannot get current price")
                price = tick.ask if order_type == OrderType.BUY else tick.bid

            # Validate volume
            if symbol in self.symbols_info:
                symbol_info = self.symbols_info[symbol]
                volume = max(symbol_info.min_lot, min(symbol_info.max_lot, volume))
                # Round to lot step
                volume = round(volume / symbol_info.lot_step) * symbol_info.lot_step

            # Create order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": order_type.value,
                "price": price,
                "deviation": deviation,
                "magic": magic,
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            # Add SL/TP if provided
            if sl is not None:
                request["sl"] = sl
            if tp is not None:
                request["tp"] = tp

            # Send order
            result = mt5.order_send(request)
            execution_time = time.time() - start_time

            if result is None:
                return OrderResult(False, error_message="Order send failed - no result")

            if result.retcode != mt5.TRADE_RETCODE_DONE:
                error_msg = f"Order failed: {result.retcode} - {result.comment}"
                self.logger.error(error_msg)
                return OrderResult(
                    False,
                    error_code=result.retcode,
                    error_message=error_msg,
                    execution_time=execution_time,
                )

            # Calculate slippage
            slippage = 0
            if result.price and price:
                slippage = abs(result.price - price)

            self.logger.info(
                f"✅ Order executed: {symbol} {order_type.name} {volume} @ {result.price}"
            )

            return OrderResult(
                True,
                order_id=result.order,
                ticket=result.deal,
                execution_time=execution_time,
                executed_price=result.price,
                slippage=slippage,
            )

        except Exception as e:
            self.logger.error(f"Order execution error: {str(e)}")
            return OrderResult(False, error_message=str(e))

    def close_position(
        self, ticket: int, volume: Optional[float] = None
    ) -> OrderResult:
        """Close existing position"""
        try:
            # Get position info
            position = mt5.positions_get(ticket=ticket)
            if not position:
                return OrderResult(False, error_message="Position not found")

            pos = position[0]
            symbol = pos.symbol
            pos_volume = volume or pos.volume

            # Determine opposite order type
            order_type = (
                OrderType.SELL if pos.type == mt5.POSITION_TYPE_BUY else OrderType.BUY
            )

            # Close position
            return self.send_order(
                symbol=symbol,
                order_type=order_type,
                volume=pos_volume,
                comment=f"Close position {ticket}",
            )

        except Exception as e:
            self.logger.error(f"Close position error: {str(e)}")
            return OrderResult(False, error_message=str(e))

    def get_positions(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get open positions"""
        if not self.is_connected():
            return []

        try:
            if symbol:
                positions = mt5.positions_get(symbol=symbol)
            else:
                positions = mt5.positions_get()

            if positions is None:
                return []

            return [pos._asdict() for pos in positions]

        except Exception as e:
            self.logger.error(f"Error getting positions: {str(e)}")
            return []

    def get_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get pending orders"""
        if not self.is_connected():
            return []

        try:
            if symbol:
                orders = mt5.orders_get(symbol=symbol)
            else:
                orders = mt5.orders_get()

            if orders is None:
                return []

            return [order._asdict() for order in orders]

        except Exception as e:
            self.logger.error(f"Error getting orders: {str(e)}")
            return []

    def get_account_info(self) -> Optional[Dict]:
        """Get account information"""
        if not self.is_connected():
            return None

        try:
            account = mt5.account_info()
            return account._asdict() if account else None
        except Exception as e:
            self.logger.error(f"Error getting account info: {str(e)}")
            return None

    def get_symbol_info(self, symbol: str) -> Optional[MarketInfo]:
        """Get symbol information"""
        return self.symbols_info.get(symbol)

    def get_connection_stats(self) -> Dict:
        """Get connection performance statistics"""
        uptime = 0
        if self.connection_start_time:
            uptime = (datetime.now() - self.connection_start_time).total_seconds()

        success_rate = 0
        if self.total_requests > 0:
            success_rate = (
                (self.total_requests - self.failed_requests) / self.total_requests
            ) * 100

        return {
            "status": self.status.value,
            "broker": self.config.broker_name,
            "uptime_seconds": uptime,
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "success_rate": success_rate,
            "avg_response_time": self.avg_response_time,
            "reconnect_attempts": self.reconnect_attempts,
        }

    def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check"""
        health = {
            "status": "healthy",
            "connection": self.is_connected(),
            "account_accessible": False,
            "symbols_accessible": 0,
            "timestamp": datetime.now().isoformat(),
        }

        try:
            # Test account access
            account = self.get_account_info()
            health["account_accessible"] = account is not None

            # Test symbol access
            accessible_symbols = 0
            for symbol in self.config.symbols:
                if self.get_tick(symbol) is not None:
                    accessible_symbols += 1

            health["symbols_accessible"] = accessible_symbols

            # Overall health assessment
            if not health["connection"] or not health["account_accessible"]:
                health["status"] = "unhealthy"
            elif accessible_symbols < len(self.config.symbols) * 0.8:  # 80% threshold
                health["status"] = "degraded"

        except Exception as e:
            health["status"] = "error"
            health["error"] = str(e)

        return health

    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()
        if self.executor:
            self.executor.shutdown(wait=True)


# Factory function for easy instantiation
def create_mt5_connector(broker_name: str = "dupoin", **kwargs) -> MT5Connector:
    """
    Factory function to create MT5 connector with broker-specific configuration

    Args:
        broker_name: Broker name (dupoin, monaxa, etc.)
        **kwargs: Additional configuration parameters

    Returns:
        Configured MT5Connector instance
    """
    config = MT5Config(broker_name=broker_name, **kwargs)
    return MT5Connector(config)


# Export main connector class
__all__ = [
    "MT5Connector",
    "TickData",
    "MarketInfo",
    "OrderResult",
    "OrderType",
    "ConnectionStatus",
    "create_mt5_connector",
]

if __name__ == "__main__":
    # Demo and testing
    print("🔌 MT5 Connector Testing")
    print("=" * 50)

    # Test connection (using demo credentials)
    connector = create_mt5_connector(
        broker_name="dupoin",
        login=12345,  # Replace with real credentials
        password="password",
        symbols=["EURUSD", "GBPUSD"],
    )

    try:
        with connector:
            if connector.is_connected():
                print("✅ Connection successful!")

                # Test tick data
                tick = connector.get_tick("EURUSD")
                if tick:
                    print(f"📊 EURUSD Tick: {tick.bid}/{tick.ask}")

                # Test historical data
                rates = connector.get_rates("EURUSD", mt5.TIMEFRAME_M1, 10)
                if rates is not None:
                    print(f"📈 Retrieved {len(rates)} M1 bars")

                # Health check
                health = connector.health_check()
                print(f"🏥 Health Status: {health['status']}")

            else:
                print("❌ Connection failed")

    except Exception as e:
        print(f"❌ Error: {str(e)}")

    print("\n🎯 MT5 Connector Ready for Production!")
