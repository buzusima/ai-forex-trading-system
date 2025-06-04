"""
Institutional-grade Forex AI Trading System
Market Data Feed Module

This module provides real-time market data streaming from multiple sources
with failover, data normalization, and low-latency processing capabilities.
"""

import asyncio
import websockets
import json
import threading
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Callable, Any, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import deque, defaultdict
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from queue import Queue, Empty
import ssl
import aiohttp
import MetaTrader5 as mt5
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import pickle
import gzip
import hashlib


class DataSourceType(Enum):
    """Data source types"""

    MT5 = "MT5"
    WEBSOCKET = "WEBSOCKET"
    REST_API = "REST_API"
    FILE = "FILE"
    REDIS = "REDIS"


class DataType(Enum):
    """Market data types"""

    TICK = "TICK"
    OHLCV = "OHLCV"
    DEPTH = "DEPTH"  # Market depth/order book
    NEWS = "NEWS"
    ECONOMIC = "ECONOMIC"


class DataQuality(Enum):
    """Data quality levels"""

    EXCELLENT = "EXCELLENT"
    GOOD = "GOOD"
    FAIR = "FAIR"
    POOR = "POOR"
    INVALID = "INVALID"


@dataclass
class MarketTick:
    """Market tick data structure"""

    symbol: str
    bid: float
    ask: float
    last: float
    volume: float
    timestamp: datetime
    source: str
    spread: float = 0.0
    bid_volume: float = 0.0
    ask_volume: float = 0.0
    flags: int = 0

    def __post_init__(self):
        self.spread = self.ask - self.bid
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(
                self.timestamp.replace("Z", "+00:00")
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data

    def get_mid_price(self) -> float:
        """Get mid price"""
        return (self.bid + self.ask) / 2


@dataclass
class OHLCV:
    """OHLCV bar data structure"""

    symbol: str
    timeframe: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: datetime
    source: str
    tick_count: int = 0
    spread: float = 0.0

    def __post_init__(self):
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(
                self.timestamp.replace("Z", "+00:00")
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data

    def is_valid(self) -> bool:
        """Check if OHLCV data is valid"""
        return (
            self.high >= max(self.open, self.close, self.low)
            and self.low <= min(self.open, self.close, self.high)
            and all(x > 0 for x in [self.open, self.high, self.low, self.close])
            and self.volume >= 0
        )


@dataclass
class DataFeedConfig:
    """Configuration for data feed"""

    symbols: List[str] = field(default_factory=list)
    timeframes: List[str] = field(default_factory=lambda: ["M1", "M5", "H1"])
    data_types: List[DataType] = field(
        default_factory=lambda: [DataType.TICK, DataType.OHLCV]
    )

    # Connection settings
    max_reconnect_attempts: int = 5
    reconnect_delay: float = 1.0
    connection_timeout: float = 10.0
    heartbeat_interval: float = 30.0

    # Data settings
    max_buffer_size: int = 10000
    data_retention_hours: int = 24
    enable_compression: bool = True
    enable_encryption: bool = False

    # Quality settings
    min_data_quality: DataQuality = DataQuality.FAIR
    max_latency_ms: float = 100.0
    enable_data_validation: bool = True

    # Storage settings
    enable_historical_storage: bool = True
    storage_path: str = "data/market_data"
    max_storage_size_gb: float = 10.0


class DataSource(ABC):
    """Abstract base class for data sources"""

    def __init__(self, source_type: DataSourceType, config: Dict[str, Any]):
        self.source_type = source_type
        self.config = config
        self.is_connected = False
        self.last_data_time = None
        self.data_quality = DataQuality.POOR
        self.error_count = 0
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to data source"""
        pass

    @abstractmethod
    async def disconnect(self):
        """Disconnect from data source"""
        pass

    @abstractmethod
    async def subscribe_symbols(self, symbols: List[str]):
        """Subscribe to symbols"""
        pass

    @abstractmethod
    async def get_tick_data(self, symbol: str) -> Optional[MarketTick]:
        """Get latest tick data"""
        pass

    @abstractmethod
    async def get_ohlcv_data(
        self, symbol: str, timeframe: str, start: datetime = None, end: datetime = None
    ) -> List[OHLCV]:
        """Get OHLCV data"""
        pass

    def update_quality(self, latency_ms: float, data_freshness_ms: float):
        """Update data quality based on metrics"""
        if latency_ms < 50 and data_freshness_ms < 1000:
            self.data_quality = DataQuality.EXCELLENT
        elif latency_ms < 100 and data_freshness_ms < 2000:
            self.data_quality = DataQuality.GOOD
        elif latency_ms < 200 and data_freshness_ms < 5000:
            self.data_quality = DataQuality.FAIR
        else:
            self.data_quality = DataQuality.POOR


class MT5DataSource(DataSource):
    """MetaTrader 5 data source"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(DataSourceType.MT5, config)
        self.account = config.get("account")
        self.password = config.get("password")
        self.server = config.get("server")
        self.subscribed_symbols = set()

    async def connect(self) -> bool:
        """Connect to MT5"""
        try:
            if not mt5.initialize():
                self.logger.error(f"MT5 initialization failed: {mt5.last_error()}")
                return False

            if self.account and self.password and self.server:
                if not mt5.login(self.account, self.password, self.server):
                    self.logger.error(f"MT5 login failed: {mt5.last_error()}")
                    return False

            self.is_connected = True
            self.logger.info("Connected to MT5")
            return True

        except Exception as e:
            self.logger.error(f"MT5 connection error: {str(e)}")
            return False

    async def disconnect(self):
        """Disconnect from MT5"""
        mt5.shutdown()
        self.is_connected = False
        self.logger.info("Disconnected from MT5")

    async def subscribe_symbols(self, symbols: List[str]):
        """Subscribe to symbols"""
        for symbol in symbols:
            if mt5.symbol_select(symbol, True):
                self.subscribed_symbols.add(symbol)
                self.logger.info(f"Subscribed to {symbol}")
            else:
                self.logger.warning(f"Failed to subscribe to {symbol}")

    async def get_tick_data(self, symbol: str) -> Optional[MarketTick]:
        """Get latest tick data from MT5"""
        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return None

            return MarketTick(
                symbol=symbol,
                bid=tick.bid,
                ask=tick.ask,
                last=tick.last,
                volume=tick.volume,
                timestamp=datetime.fromtimestamp(tick.time, tz=timezone.utc),
                source="MT5",
                flags=tick.flags,
            )

        except Exception as e:
            self.logger.error(f"Error getting tick data for {symbol}: {str(e)}")
            return None

    async def get_ohlcv_data(
        self, symbol: str, timeframe: str, start: datetime = None, end: datetime = None
    ) -> List[OHLCV]:
        """Get OHLCV data from MT5"""
        try:
            # Convert timeframe
            mt5_timeframe = self._convert_timeframe(timeframe)
            if mt5_timeframe is None:
                return []

            # Set default time range if not provided
            if end is None:
                end = datetime.now(timezone.utc)
            if start is None:
                start = end - timedelta(hours=24)

            # Get rates
            rates = mt5.copy_rates_range(symbol, mt5_timeframe, start, end)
            if rates is None:
                return []

            ohlcv_list = []
            for rate in rates:
                ohlcv = OHLCV(
                    symbol=symbol,
                    timeframe=timeframe,
                    open=rate["open"],
                    high=rate["high"],
                    low=rate["low"],
                    close=rate["close"],
                    volume=rate["tick_volume"],
                    timestamp=datetime.fromtimestamp(rate["time"], tz=timezone.utc),
                    source="MT5",
                )
                if ohlcv.is_valid():
                    ohlcv_list.append(ohlcv)

            return ohlcv_list

        except Exception as e:
            self.logger.error(f"Error getting OHLCV data for {symbol}: {str(e)}")
            return []

    def _convert_timeframe(self, timeframe: str) -> Optional[int]:
        """Convert timeframe string to MT5 constant"""
        timeframe_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
            "W1": mt5.TIMEFRAME_W1,
            "MN1": mt5.TIMEFRAME_MN1,
        }
        return timeframe_map.get(timeframe)


class WebSocketDataSource(DataSource):
    """WebSocket data source"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(DataSourceType.WEBSOCKET, config)
        self.url = config.get("url")
        self.headers = config.get("headers", {})
        self.websocket = None
        self.message_queue = Queue()
        self.is_streaming = False

    async def connect(self) -> bool:
        """Connect to WebSocket"""
        try:
            ssl_context = ssl.create_default_context()
            if self.config.get("verify_ssl", True) == False:
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE

            self.websocket = await websockets.connect(
                self.url,
                extra_headers=self.headers,
                ssl=ssl_context,
                ping_interval=30,
                ping_timeout=10,
            )

            self.is_connected = True
            self.is_streaming = True

            # Start message handling
            asyncio.create_task(self._handle_messages())

            self.logger.info(f"Connected to WebSocket: {self.url}")
            return True

        except Exception as e:
            self.logger.error(f"WebSocket connection error: {str(e)}")
            return False

    async def disconnect(self):
        """Disconnect from WebSocket"""
        self.is_streaming = False
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        self.is_connected = False
        self.logger.info("Disconnected from WebSocket")

    async def subscribe_symbols(self, symbols: List[str]):
        """Subscribe to symbols via WebSocket"""
        if not self.websocket:
            return

        subscribe_message = {
            "action": "subscribe",
            "symbols": symbols,
            "types": ["tick", "ohlcv"],
        }

        await self.websocket.send(json.dumps(subscribe_message))
        self.logger.info(f"Subscribed to symbols: {symbols}")

    async def _handle_messages(self):
        """Handle incoming WebSocket messages"""
        try:
            async for message in self.websocket:
                if not self.is_streaming:
                    break

                try:
                    data = json.loads(message)
                    self.message_queue.put(data)
                    self.last_data_time = datetime.now(timezone.utc)

                except json.JSONDecodeError as e:
                    self.logger.warning(f"Invalid JSON message: {str(e)}")

        except websockets.exceptions.ConnectionClosed:
            self.logger.warning("WebSocket connection closed")
            self.is_connected = False
        except Exception as e:
            self.logger.error(f"WebSocket message handling error: {str(e)}")
            self.error_count += 1

    async def get_tick_data(self, symbol: str) -> Optional[MarketTick]:
        """Get latest tick data from message queue"""
        try:
            while not self.message_queue.empty():
                data = self.message_queue.get_nowait()

                if data.get("type") == "tick" and data.get("symbol") == symbol:
                    return MarketTick(
                        symbol=symbol,
                        bid=data["bid"],
                        ask=data["ask"],
                        last=data.get("last", (data["bid"] + data["ask"]) / 2),
                        volume=data.get("volume", 0),
                        timestamp=datetime.fromisoformat(data["timestamp"]),
                        source="WebSocket",
                    )

            return None

        except Empty:
            return None
        except Exception as e:
            self.logger.error(f"Error processing tick data: {str(e)}")
            return None

    async def get_ohlcv_data(
        self, symbol: str, timeframe: str, start: datetime = None, end: datetime = None
    ) -> List[OHLCV]:
        """Get OHLCV data from WebSocket (limited historical data)"""
        # WebSocket typically provides real-time data, not historical
        # This method would need to be implemented based on specific API
        return []


class DataBuffer:
    """High-performance data buffer with compression"""

    def __init__(self, max_size: int = 10000, enable_compression: bool = True):
        self.max_size = max_size
        self.enable_compression = enable_compression
        self.tick_buffer = deque(maxlen=max_size)
        self.ohlcv_buffer = defaultdict(lambda: deque(maxlen=max_size))
        self.lock = threading.Lock()
        self.stats = {
            "total_ticks": 0,
            "total_bars": 0,
            "buffer_hits": 0,
            "compression_ratio": 0.0,
        }

    def add_tick(self, tick: MarketTick):
        """Add tick to buffer"""
        with self.lock:
            if self.enable_compression:
                compressed_tick = self._compress_data(tick.to_dict())
                self.tick_buffer.append(compressed_tick)
            else:
                self.tick_buffer.append(tick)

            self.stats["total_ticks"] += 1

    def add_ohlcv(self, ohlcv: OHLCV):
        """Add OHLCV to buffer"""
        with self.lock:
            key = f"{ohlcv.symbol}_{ohlcv.timeframe}"

            if self.enable_compression:
                compressed_ohlcv = self._compress_data(ohlcv.to_dict())
                self.ohlcv_buffer[key].append(compressed_ohlcv)
            else:
                self.ohlcv_buffer[key].append(ohlcv)

            self.stats["total_bars"] += 1

    def get_latest_tick(self, symbol: str) -> Optional[MarketTick]:
        """Get latest tick for symbol"""
        with self.lock:
            for item in reversed(self.tick_buffer):
                tick = self._decompress_data(item) if self.enable_compression else item
                if isinstance(tick, dict):
                    tick = MarketTick(**tick)

                if tick.symbol == symbol:
                    self.stats["buffer_hits"] += 1
                    return tick

            return None

    def get_ohlcv_history(
        self, symbol: str, timeframe: str, count: int = 100
    ) -> List[OHLCV]:
        """Get OHLCV history for symbol and timeframe"""
        with self.lock:
            key = f"{symbol}_{timeframe}"

            if key not in self.ohlcv_buffer:
                return []

            result = []
            buffer_data = list(self.ohlcv_buffer[key])[-count:]

            for item in buffer_data:
                ohlcv = self._decompress_data(item) if self.enable_compression else item
                if isinstance(ohlcv, dict):
                    ohlcv = OHLCV(**ohlcv)
                result.append(ohlcv)

            self.stats["buffer_hits"] += 1
            return result

    def _compress_data(self, data: Dict[str, Any]) -> bytes:
        """Compress data using gzip"""
        try:
            pickled_data = pickle.dumps(data)
            compressed_data = gzip.compress(pickled_data)

            # Update compression ratio
            original_size = len(pickled_data)
            compressed_size = len(compressed_data)
            ratio = compressed_size / original_size if original_size > 0 else 1.0
            self.stats["compression_ratio"] = (
                self.stats["compression_ratio"] * 0.9 + ratio * 0.1
            )

            return compressed_data

        except Exception:
            return pickle.dumps(data)  # Fallback to uncompressed

    def _decompress_data(self, data: bytes) -> Dict[str, Any]:
        """Decompress data"""
        try:
            if isinstance(data, bytes) and data[:2] == b"\x1f\x8b":  # gzip magic number
                decompressed_data = gzip.decompress(data)
                return pickle.loads(decompressed_data)
            else:
                return pickle.loads(data)
        except Exception:
            return data


class MarketDataFeed:
    """Main market data feed orchestrator"""

    def __init__(self, config: DataFeedConfig):
        self.config = config
        self.data_sources: List[DataSource] = []
        self.primary_source: Optional[DataSource] = None
        self.data_buffer = DataBuffer(config.max_buffer_size, config.enable_compression)
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.is_running = False
        self.logger = logging.getLogger(__name__)

        # Threading
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.event_loop = None

        # Performance metrics
        self.metrics = {
            "total_ticks_processed": 0,
            "total_bars_processed": 0,
            "avg_latency_ms": 0.0,
            "error_count": 0,
            "uptime_start": None,
        }

        # Historical data storage
        if config.enable_historical_storage:
            self._init_storage()

    def _init_storage(self):
        """Initialize historical data storage"""
        try:
            import os

            os.makedirs(self.config.storage_path, exist_ok=True)

            self.db_path = f"{self.config.storage_path}/market_data.db"
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)

            # Create tables
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ticks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    bid REAL NOT NULL,
                    ask REAL NOT NULL,
                    last_price REAL NOT NULL,
                    volume REAL NOT NULL,
                    timestamp DATETIME NOT NULL,
                    source TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ohlcv (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    open_price REAL NOT NULL,
                    high_price REAL NOT NULL,
                    low_price REAL NOT NULL,
                    close_price REAL NOT NULL,
                    volume REAL NOT NULL,
                    timestamp DATETIME NOT NULL,
                    source TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timeframe, timestamp)
                )
            """
            )

            # Create indexes
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ticks_symbol_time ON ticks(symbol, timestamp)"
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_timeframe_time ON ohlcv(symbol, timeframe, timestamp)"
            )

            self.conn.commit()
            self.logger.info("Historical data storage initialized")

        except Exception as e:
            self.logger.error(f"Storage initialization error: {str(e)}")

    def add_data_source(self, source: DataSource, is_primary: bool = False):
        """Add data source"""
        self.data_sources.append(source)
        if is_primary or self.primary_source is None:
            self.primary_source = source

        self.logger.info(f"Added data source: {source.source_type.value}")

    def subscribe_to_ticks(self, symbol: str, callback: Callable[[MarketTick], None]):
        """Subscribe to tick data"""
        self.subscribers[f"tick_{symbol}"].append(callback)
        self.logger.info(f"Subscribed to tick data for {symbol}")

    def subscribe_to_ohlcv(
        self, symbol: str, timeframe: str, callback: Callable[[OHLCV], None]
    ):
        """Subscribe to OHLCV data"""
        self.subscribers[f"ohlcv_{symbol}_{timeframe}"].append(callback)
        self.logger.info(f"Subscribed to OHLCV data for {symbol} {timeframe}")

    async def start(self):
        """Start data feed"""
        if self.is_running:
            return

        self.is_running = True
        self.metrics["uptime_start"] = datetime.now(timezone.utc)

        # Connect to all data sources
        for source in self.data_sources:
            try:
                await source.connect()
                await source.subscribe_symbols(self.config.symbols)
            except Exception as e:
                self.logger.error(
                    f"Error connecting to {source.source_type.value}: {str(e)}"
                )

        # Start data processing tasks
        asyncio.create_task(self._process_tick_data())
        asyncio.create_task(self._process_ohlcv_data())
        asyncio.create_task(self._monitor_health())

        self.logger.info("Market data feed started")

    async def stop(self):
        """Stop data feed"""
        self.is_running = False

        # Disconnect from all data sources
        for source in self.data_sources:
            try:
                await source.disconnect()
            except Exception as e:
                self.logger.error(
                    f"Error disconnecting from {source.source_type.value}: {str(e)}"
                )

        # Close storage connection
        if hasattr(self, "conn"):
            self.conn.close()

        self.logger.info("Market data feed stopped")

    async def _process_tick_data(self):
        """Process tick data from all sources"""
        while self.is_running:
            try:
                for symbol in self.config.symbols:
                    # Try primary source first
                    tick = None
                    if self.primary_source and self.primary_source.is_connected:
                        tick = await self.primary_source.get_tick_data(symbol)

                    # Fallback to other sources
                    if tick is None:
                        for source in self.data_sources:
                            if source != self.primary_source and source.is_connected:
                                tick = await source.get_tick_data(symbol)
                                if tick:
                                    break

                    if tick:
                        # Add to buffer
                        self.data_buffer.add_tick(tick)

                        # Store in database
                        if self.config.enable_historical_storage:
                            self._store_tick(tick)

                        # Notify subscribers
                        await self._notify_tick_subscribers(tick)

                        self.metrics["total_ticks_processed"] += 1

                await asyncio.sleep(0.01)  # Small delay to prevent excessive CPU usage

            except Exception as e:
                self.logger.error(f"Error processing tick data: {str(e)}")
                self.metrics["error_count"] += 1
                await asyncio.sleep(1)

    async def _process_ohlcv_data(self):
        """Process OHLCV data from all sources"""
        while self.is_running:
            try:
                for symbol in self.config.symbols:
                    for timeframe in self.config.timeframes:
                        # Get OHLCV data
                        ohlcv_data = []
                        if self.primary_source and self.primary_source.is_connected:
                            ohlcv_data = await self.primary_source.get_ohlcv_data(
                                symbol, timeframe
                            )

                        # Process each bar
                        for ohlcv in ohlcv_data:
                            if ohlcv.is_valid():
                                # Add to buffer
                                self.data_buffer.add_ohlcv(ohlcv)

                                # Store in database
                                if self.config.enable_historical_storage:
                                    self._store_ohlcv(ohlcv)

                                # Notify subscribers
                                await self._notify_ohlcv_subscribers(ohlcv)

                                self.metrics["total_bars_processed"] += 1

                await asyncio.sleep(60)  # Update OHLCV data every minute

            except Exception as e:
                self.logger.error(f"Error processing OHLCV data: {str(e)}")
                self.metrics["error_count"] += 1
                await asyncio.sleep(60)

    async def _notify_tick_subscribers(self, tick: MarketTick):
        """Notify tick subscribers"""
        key = f"tick_{tick.symbol}"
        for callback in self.subscribers[key]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(tick)
                else:
                    callback(tick)
            except Exception as e:
                self.logger.error(f"Error in tick callback: {str(e)}")

    async def _notify_ohlcv_subscribers(self, ohlcv: OHLCV):
        """Notify OHLCV subscribers"""
        key = f"ohlcv_{ohlcv.symbol}_{ohlcv.timeframe}"
        for callback in self.subscribers[key]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(ohlcv)
                else:
                    callback(ohlcv)
            except Exception as e:
                self.logger.error(f"Error in OHLCV callback: {str(e)}")

    async def _monitor_health(self):
        """Monitor data feed health"""
        while self.is_running:
            try:
                # Check source connectivity
                for source in self.data_sources:
                    if not source.is_connected:
                        self.logger.warning(
                            f"Data source {source.source_type.value} is disconnected"
                        )
                        # Attempt reconnection
                        try:
                            await source.connect()
                            await source.subscribe_symbols(self.config.symbols)
                        except Exception as e:
                            self.logger.error(
                                f"Reconnection failed for {source.source_type.value}: {str(e)}"
                            )

                # Log performance metrics
                uptime = datetime.now(timezone.utc) - self.metrics["uptime_start"]
                self.logger.info(
                    f"Data feed metrics - "
                    f"Uptime: {uptime.total_seconds():.0f}s, "
                    f"Ticks: {self.metrics['total_ticks_processed']}, "
                    f"Bars: {self.metrics['total_bars_processed']}, "
                    f"Errors: {self.metrics['error_count']}"
                )

                await asyncio.sleep(self.config.heartbeat_interval)

            except Exception as e:
                self.logger.error(f"Error in health monitoring: {str(e)}")
                await asyncio.sleep(30)

    def _store_tick(self, tick: MarketTick):
        """Store tick data in database"""
        try:
            self.conn.execute(
                """
                INSERT INTO ticks (symbol, bid, ask, last_price, volume, timestamp, source)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    tick.symbol,
                    tick.bid,
                    tick.ask,
                    tick.last,
                    tick.volume,
                    tick.timestamp.isoformat(),
                    tick.source,
                ),
            )
            self.conn.commit()
        except Exception as e:
            self.logger.error(f"Error storing tick data: {str(e)}")

    def _store_ohlcv(self, ohlcv: OHLCV):
        """Store OHLCV data in database"""
        try:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO ohlcv 
                (symbol, timeframe, open_price, high_price, low_price, close_price, volume, timestamp, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    ohlcv.symbol,
                    ohlcv.timeframe,
                    ohlcv.open,
                    ohlcv.high,
                    ohlcv.low,
                    ohlcv.close,
                    ohlcv.volume,
                    ohlcv.timestamp.isoformat(),
                    ohlcv.source,
                ),
            )
            self.conn.commit()
        except Exception as e:
            self.logger.error(f"Error storing OHLCV data: {str(e)}")

    def get_latest_tick(self, symbol: str) -> Optional[MarketTick]:
        """Get latest tick data"""
        return self.data_buffer.get_latest_tick(symbol)

    def get_ohlcv_history(
        self, symbol: str, timeframe: str, count: int = 100
    ) -> List[OHLCV]:
        """Get OHLCV history"""
        return self.data_buffer.get_ohlcv_history(symbol, timeframe, count)

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        uptime = (
            datetime.now(timezone.utc) - self.metrics["uptime_start"]
            if self.metrics["uptime_start"]
            else timedelta(0)
        )

        return {
            **self.metrics,
            "uptime_seconds": uptime.total_seconds(),
            "buffer_stats": self.data_buffer.stats,
            "connected_sources": sum(1 for s in self.data_sources if s.is_connected),
            "total_sources": len(self.data_sources),
        }


# Example usage and testing
if __name__ == "__main__":

    async def test_market_data_feed():
        # Create configuration
        config = DataFeedConfig(
            symbols=["EURUSD", "GBPUSD", "USDJPY"],
            timeframes=["M1", "M5", "H1"],
            max_buffer_size=1000,
            enable_compression=True,
            enable_historical_storage=True,
        )

        # Create data feed
        feed = MarketDataFeed(config)

        # Add MT5 data source
        mt5_config = {
            "account": 123456,  # Your MT5 account
            "password": "password",  # Your MT5 password
            "server": "server_name",  # Your MT5 server
        }
        mt5_source = MT5DataSource(mt5_config)
        feed.add_data_source(mt5_source, is_primary=True)

        # Add WebSocket data source as backup
        ws_config = {
            "url": "wss://api.example.com/v1/stream",
            "headers": {"Authorization": "Bearer token"},
        }
        ws_source = WebSocketDataSource(ws_config)
        feed.add_data_source(ws_source)

        # Subscribe to data
        def on_tick(tick: MarketTick):
            print(f"Tick: {tick.symbol} {tick.bid}/{tick.ask} at {tick.timestamp}")

        def on_ohlcv(ohlcv: OHLCV):
            print(
                f"OHLCV: {ohlcv.symbol} {ohlcv.timeframe} OHLC: {ohlcv.open}/{ohlcv.high}/{ohlcv.low}/{ohlcv.close}"
            )

        feed.subscribe_to_ticks("EURUSD", on_tick)
        feed.subscribe_to_ohlcv("EURUSD", "M1", on_ohlcv)

        # Start feed
        await feed.start()

        # Run for a short time
        await asyncio.sleep(10)

        # Get metrics
        metrics = feed.get_metrics()
        print(f"Metrics: {metrics}")

        # Stop feed
        await feed.stop()

    # Run test
    asyncio.run(test_market_data_feed())
    print("Market data feed test completed!")
