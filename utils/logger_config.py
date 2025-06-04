"""
Institutional-grade Forex AI Trading System
Logger Configuration Module

This module provides comprehensive logging configuration for the trading system
with multiple handlers, formatters, and monitoring capabilities.
"""

import os
import sys
import json
import logging
import logging.handlers
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import threading
from queue import Queue, Empty
import time


@dataclass
class LogConfig:
    """Logging configuration dataclass"""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    file_max_bytes: int = 10 * 1024 * 1024  # 10MB
    file_backup_count: int = 10
    console_enabled: bool = True
    file_enabled: bool = True
    json_enabled: bool = True
    trade_log_enabled: bool = True
    error_log_enabled: bool = True
    performance_log_enabled: bool = True


class TradingFormatter(logging.Formatter):
    """Custom formatter for trading logs"""

    def __init__(self, include_trade_info: bool = False):
        super().__init__()
        self.include_trade_info = include_trade_info

    def format(self, record):
        # Add trading-specific information
        if hasattr(record, "symbol"):
            record.trade_symbol = record.symbol
        if hasattr(record, "action"):
            record.trade_action = record.action
        if hasattr(record, "volume"):
            record.trade_volume = record.volume
        if hasattr(record, "price"):
            record.trade_price = record.price
        if hasattr(record, "profit"):
            record.trade_profit = record.profit

        # Standard formatting
        formatted = super().format(record)

        # Add trade info if enabled
        if self.include_trade_info and hasattr(record, "trade_symbol"):
            trade_info = f" [Symbol: {getattr(record, 'trade_symbol', 'N/A')}]"
            if hasattr(record, "trade_action"):
                trade_info += f" [Action: {record.trade_action}]"
            if hasattr(record, "trade_volume"):
                trade_info += f" [Volume: {record.trade_volume}]"
            if hasattr(record, "trade_price"):
                trade_info += f" [Price: {record.trade_price}]"
            formatted += trade_info

        return formatted


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""

    def format(self, record):
        log_entry = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.thread,
            "thread_name": record.threadName,
            "process": record.process,
        }

        # Add trading-specific fields
        trading_fields = [
            "symbol",
            "action",
            "volume",
            "price",
            "profit",
            "order_id",
            "position_id",
            "account_id",
        ]
        for field in trading_fields:
            if hasattr(record, field):
                log_entry[f"trade_{field}"] = getattr(record, field)

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False)


class PerformanceFilter(logging.Filter):
    """Filter for performance-related logs"""

    def filter(self, record):
        return (
            hasattr(record, "performance")
            or "performance" in record.getMessage().lower()
        )


class ErrorFilter(logging.Filter):
    """Filter for error-related logs"""

    def filter(self, record):
        return record.levelno >= logging.ERROR


class TradeFilter(logging.Filter):
    """Filter for trade-related logs"""

    def filter(self, record):
        return hasattr(record, "symbol") or any(
            keyword in record.getMessage().lower()
            for keyword in ["trade", "order", "position", "buy", "sell"]
        )


class LogMonitor:
    """Real-time log monitoring and alerting"""

    def __init__(self, alert_callback=None):
        self.alert_callback = alert_callback
        self.error_count = 0
        self.warning_count = 0
        self.critical_count = 0
        self.last_reset = datetime.now()
        self.lock = threading.Lock()

    def handle_log(self, record):
        """Handle incoming log records"""
        with self.lock:
            if record.levelno >= logging.CRITICAL:
                self.critical_count += 1
                if self.alert_callback:
                    self.alert_callback("CRITICAL", record)
            elif record.levelno >= logging.ERROR:
                self.error_count += 1
                if self.alert_callback:
                    self.alert_callback("ERROR", record)
            elif record.levelno >= logging.WARNING:
                self.warning_count += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics"""
        with self.lock:
            return {
                "error_count": self.error_count,
                "warning_count": self.warning_count,
                "critical_count": self.critical_count,
                "last_reset": self.last_reset.isoformat(),
                "uptime_minutes": (datetime.now() - self.last_reset).total_seconds()
                / 60,
            }

    def reset_stats(self):
        """Reset monitoring statistics"""
        with self.lock:
            self.error_count = 0
            self.warning_count = 0
            self.critical_count = 0
            self.last_reset = datetime.now()


class MonitoringHandler(logging.Handler):
    """Custom handler for log monitoring"""

    def __init__(self, monitor: LogMonitor):
        super().__init__()
        self.monitor = monitor

    def emit(self, record):
        self.monitor.handle_log(record)


class LoggerConfig:
    """Main logger configuration class"""

    def __init__(self, config: LogConfig = None, log_dir: str = "logs"):
        self.config = config or LogConfig()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.monitor = LogMonitor()
        self.loggers: Dict[str, logging.Logger] = {}
        self.handlers: Dict[str, logging.Handler] = {}

        self._setup_logging()

    def _setup_logging(self):
        """Setup logging configuration"""
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.config.level))

        # Clear existing handlers
        root_logger.handlers.clear()

        # Setup handlers
        self._setup_console_handler()
        self._setup_file_handlers()
        self._setup_monitoring_handler()

    def _setup_console_handler(self):
        """Setup console handler"""
        if not self.config.console_enabled:
            return

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        formatter = TradingFormatter()
        formatter.datefmt = self.config.date_format
        console_handler.setFormatter(formatter)

        logging.getLogger().addHandler(console_handler)
        self.handlers["console"] = console_handler

    def _setup_file_handlers(self):
        """Setup file handlers"""
        if self.config.file_enabled:
            self._setup_main_file_handler()

        if self.config.json_enabled:
            self._setup_json_handler()

        if self.config.trade_log_enabled:
            self._setup_trade_handler()

        if self.config.error_log_enabled:
            self._setup_error_handler()

        if self.config.performance_log_enabled:
            self._setup_performance_handler()

    def _setup_main_file_handler(self):
        """Setup main file handler"""
        log_file = self.log_dir / "trading_system.log"

        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=self.config.file_max_bytes,
            backupCount=self.config.file_backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(getattr(logging, self.config.level))

        formatter = TradingFormatter(include_trade_info=True)
        formatter.datefmt = self.config.date_format
        file_handler.setFormatter(formatter)

        logging.getLogger().addHandler(file_handler)
        self.handlers["main_file"] = file_handler

    def _setup_json_handler(self):
        """Setup JSON handler for structured logging"""
        log_file = self.log_dir / "trading_system.json"

        json_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=self.config.file_max_bytes,
            backupCount=self.config.file_backup_count,
            encoding="utf-8",
        )
        json_handler.setLevel(logging.DEBUG)
        json_handler.setFormatter(JSONFormatter())

        logging.getLogger().addHandler(json_handler)
        self.handlers["json"] = json_handler

    def _setup_trade_handler(self):
        """Setup trade-specific handler"""
        log_file = self.log_dir / "trades.log"

        trade_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=self.config.file_max_bytes,
            backupCount=self.config.file_backup_count,
            encoding="utf-8",
        )
        trade_handler.setLevel(logging.INFO)
        trade_handler.addFilter(TradeFilter())

        formatter = TradingFormatter(include_trade_info=True)
        formatter.datefmt = self.config.date_format
        trade_handler.setFormatter(formatter)

        logging.getLogger().addHandler(trade_handler)
        self.handlers["trade"] = trade_handler

    def _setup_error_handler(self):
        """Setup error handler"""
        log_file = self.log_dir / "errors.log"

        error_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=self.config.file_max_bytes,
            backupCount=self.config.file_backup_count,
            encoding="utf-8",
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.addFilter(ErrorFilter())

        formatter = TradingFormatter(include_trade_info=True)
        formatter.datefmt = self.config.date_format
        error_handler.setFormatter(formatter)

        logging.getLogger().addHandler(error_handler)
        self.handlers["error"] = error_handler

    def _setup_performance_handler(self):
        """Setup performance handler"""
        log_file = self.log_dir / "performance.log"

        perf_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=self.config.file_max_bytes,
            backupCount=self.config.file_backup_count,
            encoding="utf-8",
        )
        perf_handler.setLevel(logging.INFO)
        perf_handler.addFilter(PerformanceFilter())

        formatter = JSONFormatter()
        perf_handler.setFormatter(formatter)

        logging.getLogger().addHandler(perf_handler)
        self.handlers["performance"] = perf_handler

    def _setup_monitoring_handler(self):
        """Setup monitoring handler"""
        monitoring_handler = MonitoringHandler(self.monitor)
        monitoring_handler.setLevel(logging.WARNING)

        logging.getLogger().addHandler(monitoring_handler)
        self.handlers["monitoring"] = monitoring_handler

    def get_logger(self, name: str, level: str = None) -> logging.Logger:
        """Get or create a logger with specific name"""
        if name in self.loggers:
            return self.loggers[name]

        logger = logging.getLogger(name)
        if level:
            logger.setLevel(getattr(logging, level.upper()))

        self.loggers[name] = logger
        return logger

    def log_trade(self, logger_name: str, message: str, level: str = "INFO", **kwargs):
        """Log trade-specific information"""
        logger = self.get_logger(logger_name)
        log_level = getattr(logging, level.upper())

        # Create log record with extra fields
        extra = {k: v for k, v in kwargs.items() if v is not None}
        logger.log(log_level, message, extra=extra)

    def log_performance(self, logger_name: str, metrics: Dict[str, Any]):
        """Log performance metrics"""
        logger = self.get_logger(logger_name)

        # Add performance flag
        extra = {"performance": True}
        extra.update(metrics)

        message = f"Performance metrics: {json.dumps(metrics, default=str)}"
        logger.info(message, extra=extra)

    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics"""
        return self.monitor.get_stats()

    def reset_monitoring(self):
        """Reset monitoring statistics"""
        self.monitor.reset_stats()

    def update_config(self, new_config: LogConfig):
        """Update logging configuration"""
        self.config = new_config
        self._setup_logging()

    def close_all_handlers(self):
        """Close all file handlers"""
        for handler in self.handlers.values():
            if hasattr(handler, "close"):
                handler.close()


# Global logger configuration instance
_logger_config: Optional[LoggerConfig] = None


def setup_logging(config: LogConfig = None, log_dir: str = "logs") -> LoggerConfig:
    """Setup global logging configuration"""
    global _logger_config
    _logger_config = LoggerConfig(config, log_dir)
    return _logger_config


def get_logger(name: str, level: str = None) -> logging.Logger:
    """Get logger instance"""
    if _logger_config is None:
        setup_logging()
    return _logger_config.get_logger(name, level)


def log_trade(logger_name: str, message: str, level: str = "INFO", **kwargs):
    """Log trade information"""
    if _logger_config is None:
        setup_logging()
    _logger_config.log_trade(logger_name, message, level, **kwargs)


def log_performance(logger_name: str, metrics: Dict[str, Any]):
    """Log performance metrics"""
    if _logger_config is None:
        setup_logging()
    _logger_config.log_performance(logger_name, metrics)


def get_monitoring_stats() -> Dict[str, Any]:
    """Get monitoring statistics"""
    if _logger_config is None:
        setup_logging()
    return _logger_config.get_monitoring_stats()

# เพิ่มบรรทัดนี้ที่ท้ายไฟล์ utils/logger_config.py (หลังฟังก์ชัน get_monitoring_stats)


def setup_logger(
    name: str, log_file: str = None, level: str = "INFO"
) -> logging.Logger:
    """
    Setup and return a logger instance (alias for backward compatibility)

    Args:
        name: Logger name
        log_file: Optional log file path (not used in current implementation)
        level: Logging level

    Returns:
        logging.Logger: Configured logger instance
    """
    # Initialize global config if not exists
    if _logger_config is None:
        setup_logging()

    # Get logger using existing function
    logger = get_logger(name, level)

    return logger


# Export for easy import
__all__ = [
    "LogConfig",
    "TradingFormatter",
    "JSONFormatter",
    "LoggerConfig",
    "setup_logging",
    "setup_logger",  # เพิ่มนี้
    "get_logger",
    "log_trade",
    "log_performance",
    "get_monitoring_stats",
]

# Example usage and testing
if __name__ == "__main__":
    # Test logging configuration
    config = LogConfig(
        level="DEBUG",
        console_enabled=True,
        file_enabled=True,
        json_enabled=True,
        trade_log_enabled=True,
        error_log_enabled=True,
        performance_log_enabled=True,
    )

    logger_config = setup_logging(config)

    # Test different types of logging
    main_logger = get_logger("main")
    trade_logger = get_logger("trade")

    # Test regular logging
    main_logger.info("System started")
    main_logger.warning("This is a warning")
    main_logger.error("This is an error")

    # Test trade logging
    log_trade(
        "trade",
        "Buy order executed",
        level="INFO",
        symbol="EURUSD",
        action="BUY",
        volume=0.1,
        price=1.1234,
        order_id=12345,
    )

    # Test performance logging
    log_performance(
        "performance",
        {
            "total_trades": 100,
            "win_rate": 0.65,
            "profit_factor": 1.8,
            "sharpe_ratio": 1.2,
            "max_drawdown": 0.05,
        },
    )

    # Test monitoring
    print("Monitoring stats:", get_monitoring_stats())

    print("Logging configuration test completed successfully!")
