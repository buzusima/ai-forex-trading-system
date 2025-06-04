#!/usr/bin/env python3
"""
⚙️ SYSTEM SETTINGS - Central Configuration
All system-wide configurations and constants
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Optional
from pathlib import Path


@dataclass
class SystemConfig:
    """
    🏗️ Main System Configuration
    Central hub for all system settings
    """

    # === SYSTEM PATHS ===
    BASE_DIR: str = Path(__file__).parent.parent.absolute()
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    MODELS_DIR: str = os.path.join(DATA_DIR, "models")
    LOGS_DIR: str = os.path.join(DATA_DIR, "logs")
    BACKTEST_DIR: str = os.path.join(DATA_DIR, "backtest")
    TEMP_DIR: str = os.path.join(DATA_DIR, "temp")

    # === TRADING CONFIGURATION ===
    trading_symbols: List[str] = None
    analysis_timeframes: List[str] = None
    max_concurrent_trades: int = 5
    max_daily_trades: int = 20
    trading_hours: Dict[str, str] = None
    weekend_trading: bool = False

    # === RISK MANAGEMENT ===
    max_risk_per_trade: float = 0.02  # 2% per trade
    max_daily_risk: float = 0.06  # 6% per day
    max_drawdown: float = 0.15  # 15% max drawdown
    minimum_balance: float = 1000.0  # Minimum account balance
    emergency_stop_loss: float = 0.10  # 10% emergency stop

    # === AI/ML CONFIGURATION ===
    min_prediction_confidence: float = 0.65  # 65% minimum confidence
    ensemble_min_models: int = 3  # Minimum models for ensemble
    retrain_frequency: int = 100  # Retrain after N trades
    lookback_periods: int = 500  # Historical data periods
    feature_selection_threshold: float = 0.05  # Feature importance threshold

    # === SYSTEM TIMING ===
    loop_interval: float = 5.0  # Main loop interval (seconds)
    health_check_interval: int = 60  # Health check every N seconds
    error_retry_interval: float = 10.0  # Wait time after error
    connection_timeout: int = 30  # MT5 connection timeout

    # === NOTIFICATIONS ===
    telegram_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    discord_webhook: Optional[str] = None
    email_notifications: bool = False
    notification_levels: List[str] = None

    # === LOGGING ===
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_rotation: str = "midnight"
    log_retention: int = 30  # days
    detailed_logging: bool = True

    # === DATABASE ===
    database_type: str = "sqlite"  # sqlite, postgresql, mysql
    database_url: Optional[str] = None
    database_pool_size: int = 10
    database_timeout: int = 30

    # === PERFORMANCE ===
    max_memory_usage: float = 0.8  # 80% max memory
    max_cpu_usage: float = 0.7  # 70% max CPU
    cleanup_interval: int = 3600  # Cleanup every hour
    cache_size: int = 1000  # Cache size for features

    # === SECURITY ===
    api_rate_limit: int = 100  # API calls per minute
    max_login_attempts: int = 3  # Max MT5 login attempts
    session_timeout: int = 3600  # Session timeout seconds
    encrypt_logs: bool = True  # Encrypt sensitive logs

    def __post_init__(self):
        """Initialize default values and validate configuration"""

        # Set default trading symbols
        if self.trading_symbols is None:
            self.trading_symbols = [
                "EURUSD",
                "GBPUSD",
                "USDJPY",
                "USDCHF",
                "AUDUSD",
                "USDCAD",
                "NZDUSD",
                "EURJPY",
                "GBPJPY",
                "EURGBP",
                "XAUUSD",
                "USDZAR",
            ]

        # Set default analysis timeframes
        if self.analysis_timeframes is None:
            self.analysis_timeframes = ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]

        # Set default trading hours (24/5 for Forex)
        if self.trading_hours is None:
            self.trading_hours = {
                "monday": "00:00-23:59",
                "tuesday": "00:00-23:59",
                "wednesday": "00:00-23:59",
                "thursday": "00:00-23:59",
                "friday": "00:00-22:00",  # Close before weekend
                "saturday": "closed",
                "sunday": "22:00-23:59",  # Sunday open
            }

        # Set default notification levels
        if self.notification_levels is None:
            self.notification_levels = [
                "CRITICAL",
                "ERROR",
                "WARNING",
                "TRADE_EXECUTED",
                "DAILY_SUMMARY",
                "SYSTEM_STATUS",
            ]

        # Load environment variables
        self._load_environment_variables()

        # Validate configuration
        self._validate_configuration()

        # Create directories
        self._create_directories()

    def _load_environment_variables(self):
        """Load configuration from environment variables"""

        # Telegram configuration
        self.telegram_token = os.getenv("TELEGRAM_BOT_TOKEN", self.telegram_token)
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID", self.telegram_chat_id)

        # Database configuration
        self.database_url = os.getenv("DATABASE_URL", self.database_url)

        # Risk management from env (for production flexibility)
        self.max_risk_per_trade = float(
            os.getenv("MAX_RISK_PER_TRADE", self.max_risk_per_trade)
        )
        self.max_daily_risk = float(os.getenv("MAX_DAILY_RISK", self.max_daily_risk))
        self.max_drawdown = float(os.getenv("MAX_DRAWDOWN", self.max_drawdown))

        # Trading configuration
        self.min_prediction_confidence = float(
            os.getenv("MIN_PREDICTION_CONFIDENCE", self.min_prediction_confidence)
        )

        # Logging level
        self.log_level = os.getenv("LOG_LEVEL", self.log_level).upper()

        # Security settings
        self.encrypt_logs = os.getenv("ENCRYPT_LOGS", "true").lower() == "true"

    def _validate_configuration(self):
        """Validate configuration values"""

        # Risk management validation
        assert 0 < self.max_risk_per_trade <= 0.1, "Max risk per trade must be 0-10%"
        assert 0 < self.max_daily_risk <= 0.2, "Max daily risk must be 0-20%"
        assert 0 < self.max_drawdown <= 0.5, "Max drawdown must be 0-50%"
        assert self.minimum_balance > 0, "Minimum balance must be positive"

        # AI/ML validation
        assert (
            0.5 <= self.min_prediction_confidence <= 1.0
        ), "Confidence must be 50-100%"
        assert self.ensemble_min_models >= 2, "Need at least 2 models for ensemble"
        assert self.lookback_periods >= 100, "Need at least 100 periods for training"

        # Trading validation
        assert self.max_concurrent_trades > 0, "Max concurrent trades must be positive"
        assert self.max_daily_trades > 0, "Max daily trades must be positive"
        assert len(self.trading_symbols) > 0, "Must have at least one trading symbol"

        # Timing validation
        assert self.loop_interval >= 1.0, "Loop interval must be at least 1 second"
        assert self.health_check_interval >= 30, "Health check interval too frequent"

        # Performance validation
        assert 0 < self.max_memory_usage <= 1.0, "Memory usage must be 0-100%"
        assert 0 < self.max_cpu_usage <= 1.0, "CPU usage must be 0-100%"

        # Validate timeframes
        valid_timeframes = ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1"]
        for tf in self.analysis_timeframes:
            assert tf in valid_timeframes, f"Invalid timeframe: {tf}"

    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            self.DATA_DIR,
            self.MODELS_DIR,
            self.LOGS_DIR,
            self.BACKTEST_DIR,
            self.TEMP_DIR,
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def get_symbol_config(self, symbol: str) -> Dict:
        """Get symbol-specific configuration"""

        # Default configuration for all symbols
        default_config = {
            "spread_limit": 3.0,  # Max spread in pips
            "min_volume": 0.01,  # Minimum lot size
            "max_volume": 10.0,  # Maximum lot size
            "volume_step": 0.01,  # Volume step
            "stop_level": 10,  # Minimum stop level in points
            "freeze_level": 5,  # Freeze level in points
            "tick_value": 1.0,  # Tick value
            "tick_size": 0.00001,  # Tick size
            "digits": 5,  # Price digits
        }

        # Symbol-specific configurations
        symbol_configs = {
            "EURUSD": {"spread_limit": 2.0, "max_volume": 50.0},
            "GBPUSD": {"spread_limit": 2.5, "max_volume": 30.0},
            "USDJPY": {"tick_size": 0.001, "digits": 3, "spread_limit": 2.0},
            "USDCHF": {"spread_limit": 3.0, "max_volume": 20.0},
            "AUDUSD": {"spread_limit": 2.5, "max_volume": 25.0},
            "USDCAD": {"spread_limit": 2.5, "max_volume": 20.0},
            "NZDUSD": {"spread_limit": 3.0, "max_volume": 15.0},
            "EURJPY": {"tick_size": 0.001, "digits": 3, "spread_limit": 3.0},
            "GBPJPY": {"tick_size": 0.001, "digits": 3, "spread_limit": 4.0},
            "EURGBP": {"spread_limit": 2.5, "max_volume": 20.0},
            "XAUUSD": {
                "spread_limit": 50.0,
                "tick_size": 0.01,
                "digits": 2,
                "max_volume": 5.0,
            },
            "USDZAR": {"spread_limit": 100.0, "max_volume": 10.0},
        }

        # Merge default with symbol-specific config
        config = default_config.copy()
        if symbol in symbol_configs:
            config.update(symbol_configs[symbol])

        return config

    def get_timeframe_config(self, timeframe: str) -> Dict:
        """Get timeframe-specific configuration"""

        timeframe_configs = {
            "M1": {
                "weight": 0.1,  # Lower weight for shorter timeframes
                "noise_threshold": 0.8,  # Higher noise threshold
                "min_candles": 100,  # Minimum candles needed
                "max_age_hours": 2,  # Maximum age of data in hours
            },
            "M5": {
                "weight": 0.15,
                "noise_threshold": 0.7,
                "min_candles": 200,
                "max_age_hours": 6,
            },
            "M15": {
                "weight": 0.2,
                "noise_threshold": 0.6,
                "min_candles": 300,
                "max_age_hours": 12,
            },
            "M30": {
                "weight": 0.25,
                "noise_threshold": 0.5,
                "min_candles": 400,
                "max_age_hours": 24,
            },
            "H1": {
                "weight": 0.3,
                "noise_threshold": 0.4,
                "min_candles": 500,
                "max_age_hours": 48,
            },
            "H4": {
                "weight": 0.4,
                "noise_threshold": 0.3,
                "min_candles": 300,
                "max_age_hours": 168,  # 1 week
            },
            "D1": {
                "weight": 0.5,  # Highest weight for daily
                "noise_threshold": 0.2,  # Lowest noise threshold
                "min_candles": 100,
                "max_age_hours": 720,  # 1 month
            },
        }

        return timeframe_configs.get(timeframe, timeframe_configs["H1"])

    def is_trading_allowed(self, current_time=None) -> bool:
        """Check if trading is allowed at current time"""
        from datetime import datetime

        if current_time is None:
            current_time = datetime.now()

        weekday = current_time.strftime("%A").lower()
        current_hour = current_time.hour

        if weekday not in self.trading_hours:
            return False

        schedule = self.trading_hours[weekday]

        if schedule == "closed":
            return False

        if schedule == "24h" or schedule == "00:00-23:59":
            return True

        # Parse time range
        try:
            start_time, end_time = schedule.split("-")
            start_hour = int(start_time.split(":")[0])
            end_hour = int(end_time.split(":")[0])

            if start_hour <= end_hour:
                return start_hour <= current_hour <= end_hour
            else:  # Overnight schedule
                return current_hour >= start_hour or current_hour <= end_hour

        except (ValueError, IndexError):
            return True  # Default to allow trading if parsing fails

    def update_config(self, **kwargs):
        """Update configuration at runtime"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # Re-validate after update
        self._validate_configuration()

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary"""
        return {
            key: getattr(self, key)
            for key in dir(self)
            if not key.startswith("_") and not callable(getattr(self, key))
        }

    def save_to_file(self, filepath: str):
        """Save configuration to file"""
        import json

        config_dict = self.to_dict()

        # Convert Path objects to strings
        for key, value in config_dict.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)

        with open(filepath, "w") as f:
            json.dump(config_dict, f, indent=2, default=str)

    @classmethod
    def load_from_file(cls, filepath: str):
        """Load configuration from file"""
        import json

        with open(filepath, "r") as f:
            config_dict = json.load(f)

        return cls(**config_dict)


# === GLOBAL CONSTANTS ===

# Market sessions (UTC times)
MARKET_SESSIONS = {
    "SYDNEY": {"open": "21:00", "close": "06:00"},
    "TOKYO": {"open": "00:00", "close": "09:00"},
    "LONDON": {"open": "08:00", "close": "17:00"},
    "NEW_YORK": {"open": "13:00", "close": "22:00"},
}

# Major economic events to avoid trading
HIGH_IMPACT_EVENTS = [
    "NFP",  # Non-Farm Payrolls
    "FOMC",  # Federal Open Market Committee
    "ECB",  # European Central Bank
    "BOE",  # Bank of England
    "BOJ",  # Bank of Japan
    "GDP",  # Gross Domestic Product
    "CPI",  # Consumer Price Index
    "PPI",  # Producer Price Index
    "UNEMPLOYMENT",  # Unemployment Rate
    "INTEREST_RATE",  # Interest Rate Decisions
]

# Common technical indicators
TECHNICAL_INDICATORS = [
    "SMA",
    "EMA",
    "RSI",
    "MACD",
    "BOLLINGER",
    "STOCH",
    "ADX",
    "CCI",
    "WILLIAMS",
    "MOMENTUM",
    "ATR",
    "VWAP",
    "FIBONACCI",
    "PIVOT",
    "SUPPORT",
    "RESISTANCE",
]

# Risk management constants
RISK_CONSTANTS = {
    "MIN_RR_RATIO": 1.5,  # Minimum Risk/Reward ratio
    "MAX_CORRELATION": 0.7,  # Maximum correlation between trades
    "VOLATILITY_THRESHOLD": 0.03,  # 3% volatility threshold
    "LIQUIDITY_THRESHOLD": 1000,  # Minimum daily volume
    "NEWS_BUFFER_MINUTES": 30,  # Minutes to avoid before/after news
}

# Performance benchmarks
PERFORMANCE_BENCHMARKS = {
    "MIN_WIN_RATE": 0.4,  # 40% minimum win rate
    "MAX_CONSECUTIVE_LOSSES": 5,  # Maximum consecutive losses
    "MIN_PROFIT_FACTOR": 1.2,  # Minimum profit factor
    "MAX_MONTHLY_DD": 0.1,  # 10% maximum monthly drawdown
    "MIN_SHARPE_RATIO": 0.8,  # Minimum Sharpe ratio
}

# Default configuration instance
DEFAULT_CONFIG = SystemConfig()

# === TRADING SETTINGS ALIAS ===
# สำหรับ backward compatibility กับ ensemble_predictor.py
TRADING_SETTINGS = DEFAULT_CONFIG
RISK_SETTINGS = RISK_CONSTANTS
# Export สำหรับ easy import
__all__ = [
    "SystemConfig",
    "DEFAULT_CONFIG",
    "TRADING_SETTINGS",
    "RISK_SETTINGS",
    "MARKET_SESSIONS",
    "HIGH_IMPACT_EVENTS",
    "TECHNICAL_INDICATORS",
    "RISK_CONSTANTS",
    "PERFORMANCE_BENCHMARKS",
]
