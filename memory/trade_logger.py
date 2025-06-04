"""
Institutional-Grade Trade Logger & Analytics System
สำหรับ AI Trading System ที่เชื่อมต่อ MT5 บัญชีจริง

Features:
- Real-time trade logging with complete audit trail
- Performance analytics and statistics
- Risk metrics calculation
- P&L tracking and reporting
- Trade journal with technical analysis
- Export capabilities (CSV, JSON, Excel)
- Database storage with backup system
- Thread-safe concurrent logging
"""

import logging
import sqlite3
import pandas as pd
import numpy as np
import json
import csv
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from threading import Lock
import os
from pathlib import Path
import pickle
import gzip
from enum import Enum

# Custom imports (จะถูกสร้างในไฟล์ถัดไป)
from config.settings import TRADING_SETTINGS, RISK_SETTINGS
from utils.logger_config import setup_logger


class TradeType(Enum):
    """ประเภท Order"""

    BUY = "BUY"
    SELL = "SELL"
    BUY_LIMIT = "BUY_LIMIT"
    SELL_LIMIT = "SELL_LIMIT"
    BUY_STOP = "BUY_STOP"
    SELL_STOP = "SELL_STOP"


class TradeStatus(Enum):
    """สถานะของ Trade"""

    OPENED = "OPENED"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"
    PARTIAL = "PARTIAL"
    PENDING = "PENDING"


@dataclass
class TradeRecord:
    """Complete Trade Record Structure"""

    # Basic Trade Info
    ticket: int
    symbol: str
    trade_type: TradeType
    volume: float
    open_price: float
    close_price: Optional[float] = None

    # Timing
    open_time: datetime = None
    close_time: Optional[datetime] = None
    duration_minutes: Optional[int] = None

    # P&L Information
    profit: Optional[float] = None
    pips: Optional[float] = None
    commission: float = 0.0
    swap: float = 0.0
    net_profit: Optional[float] = None

    # Risk Management
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_reward_ratio: Optional[float] = None

    # AI Model Information
    model_name: Optional[str] = None
    confidence_score: Optional[float] = None
    predicted_direction: Optional[str] = None
    signal_strength: Optional[float] = None

    # Technical Analysis
    entry_reason: Optional[str] = None
    exit_reason: Optional[str] = None
    market_condition: Optional[str] = None

    # Status
    status: TradeStatus = TradeStatus.OPENED

    # Metadata
    comment: Optional[str] = None
    magic_number: Optional[int] = None

    def __post_init__(self):
        if self.open_time is None:
            self.open_time = datetime.now()

    def calculate_metrics(self):
        """คำนวณ metrics ต่างๆ เมื่อ close trade"""
        if self.close_price is not None and self.open_price is not None:
            # คำนวณ pips
            if self.symbol.endswith("JPY"):
                pip_value = 0.01
            else:
                pip_value = 0.0001

            if self.trade_type in [
                TradeType.BUY,
                TradeType.BUY_LIMIT,
                TradeType.BUY_STOP,
            ]:
                self.pips = (self.close_price - self.open_price) / pip_value
            else:
                self.pips = (self.open_price - self.close_price) / pip_value

            # คำนวณ net profit
            if self.profit is not None:
                self.net_profit = self.profit - self.commission - self.swap

            # คำนวณ duration
            if self.close_time and self.open_time:
                duration = self.close_time - self.open_time
                self.duration_minutes = int(duration.total_seconds() / 60)

            # คำนวณ Risk/Reward Ratio
            if self.stop_loss and self.take_profit:
                if self.trade_type in [
                    TradeType.BUY,
                    TradeType.BUY_LIMIT,
                    TradeType.BUY_STOP,
                ]:
                    risk = abs(self.open_price - self.stop_loss)
                    reward = abs(self.take_profit - self.open_price)
                else:
                    risk = abs(self.stop_loss - self.open_price)
                    reward = abs(self.open_price - self.take_profit)

                if risk > 0:
                    self.risk_reward_ratio = reward / risk


class TradeLogger:
    """
    Institutional-Grade Trade Logger
    Thread-safe logging system with comprehensive analytics
    """

    def __init__(self, db_path: str = "data/trades.db", backup_enabled: bool = True):
        """
        Initialize Trade Logger

        Args:
            db_path: Path to SQLite database
            backup_enabled: Enable automatic backup
        """
        self.db_path = db_path
        self.backup_enabled = backup_enabled
        self.lock = Lock()

        # Create data directory
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Setup logger
        self.logger = setup_logger("TradeLogger", "logs/trade_logger.log")

        # Initialize database
        self._init_database()

        # In-memory cache for fast access
        self.cache = {}
        self.cache_size = 1000

        # Performance tracking
        self.session_stats = {
            "trades_logged": 0,
            "total_profit": 0.0,
            "winning_trades": 0,
            "losing_trades": 0,
            "start_time": datetime.now(),
        }

        self.logger.info("TradeLogger initialized successfully")

    def _init_database(self):
        """Initialize SQLite database with proper schema"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Main trades table
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        ticket INTEGER UNIQUE NOT NULL,
                        symbol TEXT NOT NULL,
                        trade_type TEXT NOT NULL,
                        volume REAL NOT NULL,
                        open_price REAL NOT NULL,
                        close_price REAL,
                        open_time TIMESTAMP NOT NULL,
                        close_time TIMESTAMP,
                        duration_minutes INTEGER,
                        profit REAL,
                        pips REAL,
                        commission REAL DEFAULT 0,
                        swap REAL DEFAULT 0,
                        net_profit REAL,
                        stop_loss REAL,
                        take_profit REAL,
                        risk_reward_ratio REAL,
                        model_name TEXT,
                        confidence_score REAL,
                        predicted_direction TEXT,
                        signal_strength REAL,
                        entry_reason TEXT,
                        exit_reason TEXT,
                        market_condition TEXT,
                        status TEXT NOT NULL,
                        comment TEXT,
                        magic_number INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """
                )

                # Performance summary table
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS performance_summary (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date DATE NOT NULL,
                        symbol TEXT,
                        total_trades INTEGER DEFAULT 0,
                        winning_trades INTEGER DEFAULT 0,
                        losing_trades INTEGER DEFAULT 0,
                        win_rate REAL DEFAULT 0,
                        total_profit REAL DEFAULT 0,
                        total_pips REAL DEFAULT 0,
                        avg_profit_per_trade REAL DEFAULT 0,
                        avg_win REAL DEFAULT 0,
                        avg_loss REAL DEFAULT 0,
                        profit_factor REAL DEFAULT 0,
                        max_drawdown REAL DEFAULT 0,
                        sharpe_ratio REAL DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(date, symbol)
                    )
                """
                )

                # Create indexes for better performance
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_trades_open_time ON trades(open_time)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_trades_ticket ON trades(ticket)"
                )

                conn.commit()
                self.logger.info("Database initialized successfully")

        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
            raise

    def log_trade_open(self, trade: TradeRecord) -> bool:
        """
        Log trade opening

        Args:
            trade: TradeRecord object

        Returns:
            bool: Success status
        """
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()

                    # Insert trade record
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO trades (
                            ticket, symbol, trade_type, volume, open_price,
                            open_time, stop_loss, take_profit, model_name,
                            confidence_score, predicted_direction, signal_strength,
                            entry_reason, market_condition, status, comment,
                            magic_number
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            trade.ticket,
                            trade.symbol,
                            trade.trade_type.value,
                            trade.volume,
                            trade.open_price,
                            trade.open_time,
                            trade.stop_loss,
                            trade.take_profit,
                            trade.model_name,
                            trade.confidence_score,
                            trade.predicted_direction,
                            trade.signal_strength,
                            trade.entry_reason,
                            trade.market_condition,
                            trade.status.value,
                            trade.comment,
                            trade.magic_number,
                        ),
                    )

                    conn.commit()

                    # Update cache
                    self.cache[trade.ticket] = trade

                    # Update session stats
                    self.session_stats["trades_logged"] += 1

                    self.logger.info(
                        f"Trade opened logged: {trade.ticket} {trade.symbol} {trade.trade_type.value}"
                    )
                    return True

        except Exception as e:
            self.logger.error(f"Failed to log trade open: {e}")
            return False

    def log_trade_close(
        self,
        ticket: int,
        close_price: float,
        close_time: datetime = None,
        profit: float = None,
        commission: float = 0.0,
        swap: float = 0.0,
        exit_reason: str = None,
    ) -> bool:
        """
        Log trade closing

        Args:
            ticket: Trade ticket number
            close_price: Closing price
            close_time: Closing time
            profit: Trade profit
            commission: Commission paid
            swap: Swap charges
            exit_reason: Reason for exit

        Returns:
            bool: Success status
        """
        try:
            with self.lock:
                # Get existing trade
                trade = self.get_trade(ticket)
                if not trade:
                    self.logger.warning(f"Trade {ticket} not found for closing")
                    return False

                # Update trade record
                trade.close_price = close_price
                trade.close_time = close_time or datetime.now()
                trade.profit = profit
                trade.commission = commission
                trade.swap = swap
                trade.exit_reason = exit_reason
                trade.status = TradeStatus.CLOSED

                # Calculate metrics
                trade.calculate_metrics()

                # Update database
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()

                    cursor.execute(
                        """
                        UPDATE trades SET
                            close_price = ?, close_time = ?, duration_minutes = ?,
                            profit = ?, pips = ?, commission = ?, swap = ?,
                            net_profit = ?, risk_reward_ratio = ?, exit_reason = ?,
                            status = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE ticket = ?
                    """,
                        (
                            trade.close_price,
                            trade.close_time,
                            trade.duration_minutes,
                            trade.profit,
                            trade.pips,
                            trade.commission,
                            trade.swap,
                            trade.net_profit,
                            trade.risk_reward_ratio,
                            trade.exit_reason,
                            trade.status.value,
                            ticket,
                        ),
                    )

                    conn.commit()

                # Update cache
                self.cache[ticket] = trade

                # Update session stats
                if trade.net_profit:
                    self.session_stats["total_profit"] += trade.net_profit
                    if trade.net_profit > 0:
                        self.session_stats["winning_trades"] += 1
                    else:
                        self.session_stats["losing_trades"] += 1

                self.logger.info(
                    f"Trade closed logged: {ticket} P&L: {trade.net_profit}"
                )

                # Update daily performance summary
                self._update_daily_summary(trade)

                return True

        except Exception as e:
            self.logger.error(f"Failed to log trade close: {e}")
            return False

    def get_trade(self, ticket: int) -> Optional[TradeRecord]:
        """
        Get trade record by ticket

        Args:
            ticket: Trade ticket number

        Returns:
            TradeRecord or None
        """
        try:
            # Check cache first
            if ticket in self.cache:
                return self.cache[ticket]

            # Query database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM trades WHERE ticket = ?", (ticket,))
                row = cursor.fetchone()

                if row:
                    # Convert row to TradeRecord
                    trade = self._row_to_trade_record(row)
                    self.cache[ticket] = trade
                    return trade

            return None

        except Exception as e:
            self.logger.error(f"Failed to get trade {ticket}: {e}")
            return None

    def get_trades_by_symbol(
        self, symbol: str, start_date: datetime = None, end_date: datetime = None
    ) -> List[TradeRecord]:
        """
        Get trades by symbol and date range

        Args:
            symbol: Trading symbol
            start_date: Start date filter
            end_date: End date filter

        Returns:
            List of TradeRecord
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = "SELECT * FROM trades WHERE symbol = ?"
                params = [symbol]

                if start_date:
                    query += " AND open_time >= ?"
                    params.append(start_date)

                if end_date:
                    query += " AND open_time <= ?"
                    params.append(end_date)

                query += " ORDER BY open_time DESC"

                cursor = conn.cursor()
                cursor.execute(query, params)
                rows = cursor.fetchall()

                return [self._row_to_trade_record(row) for row in rows]

        except Exception as e:
            self.logger.error(f"Failed to get trades by symbol: {e}")
            return []

    def get_performance_stats(
        self, start_date: datetime = None, end_date: datetime = None, symbol: str = None
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive performance statistics

        Args:
            start_date: Start date filter
            end_date: End date filter
            symbol: Symbol filter

        Returns:
            Dictionary with performance metrics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT * FROM trades 
                    WHERE status = 'CLOSED'
                """
                params = []

                if start_date:
                    query += " AND close_time >= ?"
                    params.append(start_date)

                if end_date:
                    query += " AND close_time <= ?"
                    params.append(end_date)

                if symbol:
                    query += " AND symbol = ?"
                    params.append(symbol)

                df = pd.read_sql_query(query, conn, params=params)

                if df.empty:
                    return self._empty_stats()

                # Calculate basic metrics
                total_trades = len(df)
                winning_trades = len(df[df["net_profit"] > 0])
                losing_trades = len(df[df["net_profit"] < 0])
                win_rate = (
                    (winning_trades / total_trades) * 100 if total_trades > 0 else 0
                )

                # P&L metrics
                total_profit = df["net_profit"].sum()
                avg_profit = df["net_profit"].mean()
                avg_win = (
                    df[df["net_profit"] > 0]["net_profit"].mean()
                    if winning_trades > 0
                    else 0
                )
                avg_loss = (
                    df[df["net_profit"] < 0]["net_profit"].mean()
                    if losing_trades > 0
                    else 0
                )

                # Profit factor
                gross_profit = df[df["net_profit"] > 0]["net_profit"].sum()
                gross_loss = abs(df[df["net_profit"] < 0]["net_profit"].sum())
                profit_factor = (
                    gross_profit / gross_loss if gross_loss > 0 else float("inf")
                )

                # Risk metrics
                max_drawdown = self._calculate_max_drawdown(df)
                sharpe_ratio = self._calculate_sharpe_ratio(df)

                # Pips metrics
                total_pips = df["pips"].sum() if "pips" in df.columns else 0
                avg_pips = df["pips"].mean() if "pips" in df.columns else 0

                return {
                    "total_trades": total_trades,
                    "winning_trades": winning_trades,
                    "losing_trades": losing_trades,
                    "win_rate": round(win_rate, 2),
                    "total_profit": round(total_profit, 2),
                    "avg_profit_per_trade": round(avg_profit, 2),
                    "avg_win": round(avg_win, 2),
                    "avg_loss": round(avg_loss, 2),
                    "profit_factor": round(profit_factor, 2),
                    "max_drawdown": round(max_drawdown, 2),
                    "sharpe_ratio": round(sharpe_ratio, 4),
                    "total_pips": round(total_pips, 1),
                    "avg_pips_per_trade": round(avg_pips, 1),
                    "gross_profit": round(gross_profit, 2),
                    "gross_loss": round(gross_loss, 2),
                    "largest_win": round(df["net_profit"].max(), 2),
                    "largest_loss": round(df["net_profit"].min(), 2),
                    "avg_trade_duration": (
                        round(df["duration_minutes"].mean(), 1)
                        if "duration_minutes" in df.columns
                        else 0
                    ),
                }

        except Exception as e:
            self.logger.error(f"Failed to calculate performance stats: {e}")
            return self._empty_stats()

    def export_trades_csv(
        self, filepath: str, start_date: datetime = None, end_date: datetime = None
    ) -> bool:
        """
        Export trades to CSV file

        Args:
            filepath: Output file path
            start_date: Start date filter
            end_date: End date filter

        Returns:
            bool: Success status
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = "SELECT * FROM trades"
                params = []

                if start_date or end_date:
                    query += " WHERE"
                    conditions = []

                    if start_date:
                        conditions.append(" open_time >= ?")
                        params.append(start_date)

                    if end_date:
                        conditions.append(" open_time <= ?")
                        params.append(end_date)

                    query += " AND".join(conditions)

                query += " ORDER BY open_time DESC"

                df = pd.read_sql_query(query, conn, params=params)
                df.to_csv(filepath, index=False)

                self.logger.info(f"Trades exported to {filepath}")
                return True

        except Exception as e:
            self.logger.error(f"Failed to export trades: {e}")
            return False

    def backup_database(self, backup_path: str = None) -> bool:
        """
        Create compressed backup of database

        Args:
            backup_path: Custom backup path

        Returns:
            bool: Success status
        """
        try:
            if not backup_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = f"backups/trades_backup_{timestamp}.db.gz"

            # Create backup directory
            Path(backup_path).parent.mkdir(parents=True, exist_ok=True)

            # Create compressed backup
            with open(self.db_path, "rb") as f_in:
                with gzip.open(backup_path, "wb") as f_out:
                    f_out.writelines(f_in)

            self.logger.info(f"Database backed up to {backup_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to backup database: {e}")
            return False

    def get_session_summary(self) -> Dict[str, Any]:
        """Get current session summary"""
        duration = datetime.now() - self.session_stats["start_time"]

        return {
            "session_duration_hours": round(duration.total_seconds() / 3600, 2),
            "trades_logged": self.session_stats["trades_logged"],
            "total_profit": round(self.session_stats["total_profit"], 2),
            "winning_trades": self.session_stats["winning_trades"],
            "losing_trades": self.session_stats["losing_trades"],
            "win_rate": round(
                (
                    self.session_stats["winning_trades"]
                    / max(
                        1,
                        self.session_stats["winning_trades"]
                        + self.session_stats["losing_trades"],
                    )
                )
                * 100,
                2,
            ),
        }

    def _row_to_trade_record(self, row) -> TradeRecord:
        """Convert database row to TradeRecord"""
        # Database column mapping
        columns = [
            "id",
            "ticket",
            "symbol",
            "trade_type",
            "volume",
            "open_price",
            "close_price",
            "open_time",
            "close_time",
            "duration_minutes",
            "profit",
            "pips",
            "commission",
            "swap",
            "net_profit",
            "stop_loss",
            "take_profit",
            "risk_reward_ratio",
            "model_name",
            "confidence_score",
            "predicted_direction",
            "signal_strength",
            "entry_reason",
            "exit_reason",
            "market_condition",
            "status",
            "comment",
            "magic_number",
            "created_at",
            "updated_at",
        ]

        data = dict(zip(columns, row))

        return TradeRecord(
            ticket=data["ticket"],
            symbol=data["symbol"],
            trade_type=TradeType(data["trade_type"]),
            volume=data["volume"],
            open_price=data["open_price"],
            close_price=data["close_price"],
            open_time=(
                datetime.fromisoformat(data["open_time"]) if data["open_time"] else None
            ),
            close_time=(
                datetime.fromisoformat(data["close_time"])
                if data["close_time"]
                else None
            ),
            duration_minutes=data["duration_minutes"],
            profit=data["profit"],
            pips=data["pips"],
            commission=data["commission"] or 0.0,
            swap=data["swap"] or 0.0,
            net_profit=data["net_profit"],
            stop_loss=data["stop_loss"],
            take_profit=data["take_profit"],
            risk_reward_ratio=data["risk_reward_ratio"],
            model_name=data["model_name"],
            confidence_score=data["confidence_score"],
            predicted_direction=data["predicted_direction"],
            signal_strength=data["signal_strength"],
            entry_reason=data["entry_reason"],
            exit_reason=data["exit_reason"],
            market_condition=data["market_condition"],
            status=TradeStatus(data["status"]),
            comment=data["comment"],
            magic_number=data["magic_number"],
        )

    def _update_daily_summary(self, trade: TradeRecord):
        """Update daily performance summary"""
        try:
            if not trade.close_time:
                return

            date = trade.close_time.date()

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Get existing summary
                cursor.execute(
                    """
                    SELECT * FROM performance_summary 
                    WHERE date = ? AND symbol = ?
                """,
                    (date, trade.symbol),
                )

                row = cursor.fetchone()

                if row:
                    # Update existing summary
                    total_trades = row[3] + 1
                    winning_trades = row[4] + (
                        1 if trade.net_profit and trade.net_profit > 0 else 0
                    )
                    losing_trades = row[5] + (
                        1 if trade.net_profit and trade.net_profit < 0 else 0
                    )
                    total_profit = (row[7] or 0) + (trade.net_profit or 0)
                    total_pips = (row[8] or 0) + (trade.pips or 0)

                    cursor.execute(
                        """
                        UPDATE performance_summary SET
                            total_trades = ?, winning_trades = ?, losing_trades = ?,
                            win_rate = ?, total_profit = ?, total_pips = ?,
                            avg_profit_per_trade = ?
                        WHERE date = ? AND symbol = ?
                    """,
                        (
                            total_trades,
                            winning_trades,
                            losing_trades,
                            (
                                (winning_trades / total_trades * 100)
                                if total_trades > 0
                                else 0
                            ),
                            total_profit,
                            total_pips,
                            total_profit / total_trades if total_trades > 0 else 0,
                            date,
                            trade.symbol,
                        ),
                    )
                else:
                    # Create new summary
                    winning_trades = (
                        1 if trade.net_profit and trade.net_profit > 0 else 0
                    )
                    losing_trades = (
                        1 if trade.net_profit and trade.net_profit < 0 else 0
                    )

                    cursor.execute(
                        """
                        INSERT INTO performance_summary (
                            date, symbol, total_trades, winning_trades, losing_trades,
                            win_rate, total_profit, total_pips, avg_profit_per_trade
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            date,
                            trade.symbol,
                            1,
                            winning_trades,
                            losing_trades,
                            winning_trades * 100,
                            trade.net_profit or 0,
                            trade.pips or 0,
                            trade.net_profit or 0,
                        ),
                    )

                conn.commit()

        except Exception as e:
            self.logger.error(f"Failed to update daily summary: {e}")

    def _calculate_max_drawdown(self, df: pd.DataFrame) -> float:
        """Calculate maximum drawdown"""
        if df.empty or "net_profit" not in df.columns:
            return 0.0

        try:
            # Calculate cumulative P&L
            df_sorted = df.sort_values("close_time")
            cumulative_pl = df_sorted["net_profit"].cumsum()

            # Calculate running maximum
            running_max = cumulative_pl.cummax()

            # Calculate drawdown
            drawdown = cumulative_pl - running_max

            return abs(drawdown.min())

        except Exception:
            return 0.0

    def _calculate_sharpe_ratio(
        self, df: pd.DataFrame, risk_free_rate: float = 0.02
    ) -> float:
        """Calculate Sharpe ratio"""
        if df.empty or "net_profit" not in df.columns:
            return 0.0

        try:
            returns = df["net_profit"]
            if len(returns) < 2:
                return 0.0

            mean_return = returns.mean()
            std_return = returns.std()

            if std_return == 0:
                return 0.0

            # Annualized Sharpe ratio (assuming daily returns)
            sharpe = (mean_return * 252 - risk_free_rate) / (std_return * np.sqrt(252))
            return sharpe

        except Exception:
            return 0.0

    def _empty_stats(self) -> Dict[str, Any]:
        """Return empty statistics structure"""
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "total_profit": 0.0,
            "avg_profit_per_trade": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "total_pips": 0.0,
            "avg_pips_per_trade": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "largest_win": 0.0,
            "largest_loss": 0.0,
            "avg_trade_duration": 0.0,
        }

    def __del__(self):
        """Cleanup on destruction"""
        try:
            if self.backup_enabled:
                self.backup_database()
        except:
            pass


# Usage Example
if __name__ == "__main__":
    # Initialize logger
    logger = TradeLogger()

    # Example trade
    trade = TradeRecord(
        ticket=123456,
        symbol="EURUSD",
        trade_type=TradeType.BUY,
        volume=0.1,
        open_price=1.1000,
        stop_loss=1.0950,
        take_profit=1.1100,
        model_name="LSTM_v1",
        confidence_score=0.85,
        entry_reason="Strong bullish signal",
    )

    # Log trade opening
    logger.log_trade_open(trade)

    # Log trade closing
    logger.log_trade_close(
        ticket=123456, close_price=1.1080, profit=80.0, exit_reason="Take profit hit"
    )

    # Get performance stats
    stats = logger.get_performance_stats()
    print("Performance Stats:", stats)

    # Export trades
    logger.export_trades_csv("trades_export.csv")
