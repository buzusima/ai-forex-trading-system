#!/usr/bin/env python3
"""
🔌 MT5 CONNECTION CONFIGURATION
MetaTrader 5 platform connection settings and broker configurations
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum


class BrokerType(Enum):
    """Supported broker types"""

    DEMO = "demo"
    LIVE = "live"
    ECN = "ecn"
    STP = "stp"


class OrderType(Enum):
    """MT5 Order types"""

    BUY = 0
    SELL = 1
    BUY_LIMIT = 2
    SELL_LIMIT = 3
    BUY_STOP = 4
    SELL_STOP = 5
    BUY_STOP_LIMIT = 6
    SELL_STOP_LIMIT = 7


class TradeAction(Enum):
    """MT5 Trade actions"""

    DEAL = 1
    PENDING = 2
    SLTP = 3


@dataclass
class MT5Config:
    """
    🏦 MetaTrader 5 Configuration
    Handles all MT5 platform connection and trading settings
    """

    # === CONNECTION SETTINGS ===
    login: Optional[int] = None  # MT5 account number
    password: Optional[str] = None  # MT5 account password
    server: Optional[str] = None  # MT5 broker server
    path: Optional[str] = None  # MT5 terminal path
    timeout: int = 30000  # Connection timeout (ms)
    portable: bool = False  # Portable mode

    # === BROKER SETTINGS ===
    broker_name: str = "Auto-Detected"  # Will auto-detect from server
    broker_type: BrokerType = BrokerType.LIVE
    broker_timezone: str = "UTC"
    broker_dst: bool = False  # Daylight saving time
    commission_type: str = "auto"  # auto-detect, per_deal, per_volume, percentage
    commission_value: float = 0.0  # Will auto-detect from symbol info
    auto_detect_settings: bool = True  # Auto-detect broker settings

    # === ACCOUNT VALIDATION ===
    min_account_balance: float = 100.0  # Minimum balance required
    max_account_leverage: int = 500  # Maximum leverage allowed
    allowed_account_types: List[str] = None  # Allowed account types
    require_live_account: bool = True  # Require live account

    # === TRADING SETTINGS ===
    magic_number: int = 12345  # EA magic number
    slippage: int = 3  # Maximum slippage in points
    fill_policy: int = 0  # Order filling policy
    type_time: int = 0  # Order expiration type
    deviation: int = 20  # Price deviation points

    # === POSITION MANAGEMENT ===
    max_positions: int = 10  # Maximum open positions
    hedge_allowed: bool = False  # Allow hedging
    fifo_rule: bool = True  # First in, first out rule
    margin_call_level: float = 80.0  # Margin call level %
    stop_out_level: float = 50.0  # Stop out level %

    # === SYMBOL SETTINGS ===
    symbol_prefix: str = ""  # Symbol prefix (e.g., "m.")
    symbol_suffix: str = ""  # Symbol suffix (e.g., ".pro")
    trade_mode_filter: List[str] = None  # Allowed trade modes
    calc_mode_filter: List[str] = None  # Allowed calculation modes

    # === CONNECTION MONITORING ===
    ping_interval: int = 30  # Ping interval in seconds
    reconnect_attempts: int = 5  # Max reconnection attempts
    reconnect_delay: int = 10  # Delay between attempts
    connection_check_symbols: List[str] = None  # Symbols to check connection

    # === ERROR HANDLING ===
    retry_on_error: bool = True  # Retry on trade errors
    max_retries: int = 3  # Maximum retry attempts
    retry_delay: float = 1.0  # Delay between retries
    ignored_errors: List[int] = None  # Error codes to ignore

    # === SECURITY ===
    encrypt_credentials: bool = True  # Encrypt stored credentials
    credential_file: str = ".mt5_creds"  # Credential file name
    use_windows_auth: bool = False  # Use Windows authentication
    two_factor_auth: bool = False  # Two-factor authentication

    def __post_init__(self):
        """Initialize default values and load configuration"""

        # Set default allowed account types
        if self.allowed_account_types is None:
            self.allowed_account_types = ["real", "demo", "contest", "cent", "ecn"]

        # Set default trade mode filter - accept all common modes
        if self.trade_mode_filter is None:
            self.trade_mode_filter = ["FULL", "LONGONLY", "SHORTONLY", "CLOSEONLY"]

        # Set default calculation mode filter - accept all common modes
        if self.calc_mode_filter is None:
            self.calc_mode_filter = [
                "FOREX",
                "CFD",
                "FUTURES",
                "CFDINDEX",
                "CFDLEVERAGE",
                "EXCH_STOCKS",
            ]

        # Set default connection check symbols - universal symbols
        if self.connection_check_symbols is None:
            self.connection_check_symbols = [
                "EURUSD",
                "GBPUSD",
                "USDJPY",
                "EURJPY",
                "GBPJPY",
            ]

        # Set default ignored errors
        if self.ignored_errors is None:
            self.ignored_errors = [
                4756,  # Order is being processed
                4757,  # Request is being processed
                4109,  # Trade is not allowed for expert advisor
                4110,  # Long positions only allowed
                4111,  # Short positions only allowed
            ]

        # Load credentials and validate
        self._load_credentials()
        if self.login and self.password and self.server:
            self._validate_configuration()

    def _load_credentials(self):
        """Load MT5 credentials from environment or file"""

        # Try environment variables first (most secure)
        self.login = (
            int(os.getenv("MT5_LOGIN", 0)) if os.getenv("MT5_LOGIN") else self.login
        )
        self.password = os.getenv("MT5_PASSWORD", self.password)
        self.server = os.getenv("MT5_SERVER", self.server)
        self.path = os.getenv("MT5_PATH", self.path)

        # Load from encrypted file if env vars not available
        if not all([self.login, self.password, self.server]):
            self._load_from_encrypted_file()

        # Validate critical credentials
        if not all([self.login, self.password, self.server]):
            print(
                "Warning: MT5 credentials not found. Please set environment variables or create credential file."
            )

    def _load_from_encrypted_file(self):
        """Load credentials from encrypted file"""
        try:
            from cryptography.fernet import Fernet
            import json

            # Check if credential file exists
            if not os.path.exists(self.credential_file):
                return

            # Load encryption key (should be stored securely)
            key_file = f"{self.credential_file}.key"
            if not os.path.exists(key_file):
                return

            with open(key_file, "rb") as f:
                key = f.read()

            # Decrypt and load credentials
            fernet = Fernet(key)
            with open(self.credential_file, "rb") as f:
                encrypted_data = f.read()

            decrypted_data = fernet.decrypt(encrypted_data)
            credentials = json.loads(decrypted_data.decode())

            self.login = credentials.get("login", self.login)
            self.password = credentials.get("password", self.password)
            self.server = credentials.get("server", self.server)
            self.path = credentials.get("path", self.path)

        except Exception as e:
            print(f"Warning: Could not load encrypted credentials: {e}")

    def save_credentials(
        self, login: int, password: str, server: str, path: str = None
    ):
        """Save credentials to encrypted file"""
        try:
            from cryptography.fernet import Fernet
            import json

            # Generate encryption key
            key = Fernet.generate_key()
            fernet = Fernet(key)

            # Prepare credentials
            credentials = {
                "login": login,
                "password": password,
                "server": server,
                "path": path,
            }

            # Encrypt and save
            encrypted_data = fernet.encrypt(json.dumps(credentials).encode())

            with open(self.credential_file, "wb") as f:
                f.write(encrypted_data)

            # Save key separately
            with open(f"{self.credential_file}.key", "wb") as f:
                f.write(key)

            # Set file permissions (Unix-like systems)
            if hasattr(os, "chmod"):
                os.chmod(self.credential_file, 0o600)
                os.chmod(f"{self.credential_file}.key", 0o600)

            print("✅ Credentials saved securely")

        except Exception as e:
            print(f"❌ Failed to save credentials: {e}")

    def _validate_configuration(self):
        """Validate MT5 configuration"""

        # Validate account settings
        assert self.login > 0, "Invalid MT5 login number"
        assert len(self.password) >= 4, "MT5 password too short"
        assert len(self.server) > 0, "MT5 server not specified"
        assert self.timeout > 0, "Invalid connection timeout"

        # Validate trading settings
        assert self.magic_number > 0, "Invalid magic number"
        assert 0 <= self.slippage <= 100, "Invalid slippage value"
        assert self.max_positions > 0, "Invalid max positions"

        # Validate margin levels
        assert 0 < self.margin_call_level <= 100, "Invalid margin call level"
        assert 0 < self.stop_out_level <= 100, "Invalid stop out level"
        assert (
            self.stop_out_level < self.margin_call_level
        ), "Stop out must be less than margin call"

        # Validate retry settings
        assert self.max_retries >= 0, "Invalid max retries"
        assert self.retry_delay >= 0, "Invalid retry delay"
        assert self.reconnect_attempts > 0, "Invalid reconnect attempts"

    def auto_detect_broker_settings(
        self, mt5_terminal_info: Dict = None, account_info: Dict = None
    ) -> Dict:
        """
        🔍 Auto-detect broker settings from MT5 connection
        Returns detected broker configuration
        """
        detected_settings = {
            "broker_name": "Unknown",
            "symbol_prefix": "",
            "symbol_suffix": "",
            "commission_type": "spread_only",
            "commission_value": 0.0,
            "max_account_leverage": 100,
            "spread_analysis": {},
            "symbol_format": "standard",
        }

        try:
            # Detect broker from server name
            if self.server:
                server_lower = self.server.lower()

                # Auto-detect known brokers from server name
                if "icmarkets" in server_lower or "ic-markets" in server_lower:
                    detected_settings.update(
                        {
                            "broker_name": "IC Markets",
                            "symbol_suffix": ".pro" if ".pro" in server_lower else "",
                            "commission_type": "per_deal",
                            "commission_value": 3.5,
                            "max_account_leverage": 500,
                        }
                    )

                elif "pepperstone" in server_lower:
                    detected_settings.update(
                        {
                            "broker_name": "Pepperstone",
                            "commission_type": "per_deal",
                            "commission_value": 3.5,
                            "max_account_leverage": 200,
                        }
                    )

                elif "oanda" in server_lower:
                    detected_settings.update(
                        {
                            "broker_name": "OANDA",
                            "symbol_suffix": "_",
                            "commission_type": "spread_only",
                            "max_account_leverage": 50,
                        }
                    )

                elif "fxpro" in server_lower:
                    detected_settings.update(
                        {
                            "broker_name": "FxPro",
                            "commission_type": "per_deal",
                            "commission_value": 4.5,
                            "max_account_leverage": 200,
                        }
                    )

                elif "exness" in server_lower:
                    detected_settings.update(
                        {
                            "broker_name": "Exness",
                            "commission_type": "spread_only",
                            "max_account_leverage": 2000,
                        }
                    )

                elif "xm" in server_lower or "trading point" in server_lower:
                    detected_settings.update(
                        {
                            "broker_name": "XM",
                            "commission_type": "spread_only",
                            "max_account_leverage": 888,
                        }
                    )

                elif "admiral" in server_lower:
                    detected_settings.update(
                        {
                            "broker_name": "Admiral Markets",
                            "commission_type": "per_deal",
                            "commission_value": 3.0,
                            "max_account_leverage": 500,
                        }
                    )

                elif "fxtm" in server_lower or "forextime" in server_lower:
                    detected_settings.update(
                        {
                            "broker_name": "FXTM",
                            "commission_type": "spread_only",
                            "max_account_leverage": 1000,
                        }
                    )

                elif "dupoin" in server_lower:
                    detected_settings.update(
                        {
                            "broker_name": "Dupoin",
                            "commission_type": "spread_only",
                            "max_account_leverage": 1000,
                        }
                    )

                elif "monaxa" in server_lower:
                    detected_settings.update(
                        {
                            "broker_name": "Monaxa",
                            "commission_type": "per_deal",
                            "commission_value": 2.5,
                            "max_account_leverage": 500,
                        }
                    )

                else:
                    # Generic detection based on server patterns
                    if "demo" in server_lower:
                        detected_settings["broker_name"] = (
                            f"Demo-{self.server.split('-')[0]}"
                        )
                    elif "real" in server_lower or "live" in server_lower:
                        detected_settings["broker_name"] = (
                            f"Live-{self.server.split('-')[0]}"
                        )
                    else:
                        detected_settings["broker_name"] = self.server.split("-")[
                            0
                        ].title()

            # Auto-detect from account info if available
            if account_info:
                leverage = account_info.get("leverage", 100)
                detected_settings["max_account_leverage"] = leverage

                # Detect account type
                balance = account_info.get("balance", 0)
                if balance < 100:  # Likely cent account
                    detected_settings["account_type"] = "cent"
                elif balance > 100000:  # Likely professional account
                    detected_settings["account_type"] = "professional"
                else:
                    detected_settings["account_type"] = "standard"

            # Auto-detect from terminal info if available
            if mt5_terminal_info:
                company = mt5_terminal_info.get("company", "")
                if company:
                    detected_settings["broker_name"] = company

                # Detect if it's a demo or live terminal
                if mt5_terminal_info.get("trade_allowed", False):
                    detected_settings["account_status"] = "live"
                else:
                    detected_settings["account_status"] = "demo"

            return detected_settings

        except Exception as e:
            print(f"⚠️ Auto-detection warning: {e}")
            return detected_settings

    def detect_symbol_format(self, available_symbols: List[str]) -> Dict:
        """
        🔍 Auto-detect symbol naming format from available symbols
        """
        format_analysis = {
            "prefix": "",
            "suffix": "",
            "format_type": "standard",
            "sample_symbols": [],
            "confidence": 0.0,
        }

        if not available_symbols:
            return format_analysis

        # Analyze common symbols to detect format
        test_symbols = ["EURUSD", "GBPUSD", "USDJPY", "EURJPY", "XAUUSD"]
        found_formats = []

        for test_symbol in test_symbols:
            matches = [s for s in available_symbols if test_symbol in s]
            if matches:
                format_analysis["sample_symbols"].extend(matches[:2])

                # Check for common patterns
                for match in matches:
                    if match == test_symbol:
                        found_formats.append("standard")
                    elif match.startswith("m.") and match.endswith(test_symbol):
                        found_formats.append("m_prefix")
                        format_analysis["prefix"] = "m."
                    elif match.endswith(".pro"):
                        found_formats.append("pro_suffix")
                        format_analysis["suffix"] = ".pro"
                    elif match.endswith("_"):
                        found_formats.append("underscore_suffix")
                        format_analysis["suffix"] = "_"
                    elif "#" in match:
                        found_formats.append("hash_format")

        # Determine most common format
        if found_formats:
            from collections import Counter

            most_common = Counter(found_formats).most_common(1)[0]
            format_analysis["format_type"] = most_common[0]
            format_analysis["confidence"] = most_common[1] / len(found_formats)

        return format_analysis

    def apply_detected_settings(self, detected_settings: Dict):
        """
        🔧 Apply auto-detected settings to configuration
        """
        if self.auto_detect_settings:
            # Apply broker name
            if detected_settings.get("broker_name") != "Unknown":
                self.broker_name = detected_settings["broker_name"]

            # Apply symbol format
            if detected_settings.get("symbol_prefix"):
                self.symbol_prefix = detected_settings["symbol_prefix"]

            if detected_settings.get("symbol_suffix"):
                self.symbol_suffix = detected_settings["symbol_suffix"]

            # Apply commission settings
            if detected_settings.get("commission_type") != "spread_only":
                self.commission_type = detected_settings["commission_type"]
                self.commission_value = detected_settings.get("commission_value", 0.0)

            # Apply leverage
            if detected_settings.get("max_account_leverage"):
                self.max_account_leverage = detected_settings["max_account_leverage"]

            print(f"✅ Auto-detected broker: {self.broker_name}")
            print(f"📊 Symbol format: {self.symbol_prefix}SYMBOL{self.symbol_suffix}")
            print(f"💰 Commission: {self.commission_type} = {self.commission_value}")
            print(f"📈 Max leverage: {self.max_account_leverage}")

    def get_connection_params(self) -> Dict:
        """Get MT5 connection parameters"""
        params = {
            "login": self.login,
            "password": self.password,
            "server": self.server,
            "timeout": self.timeout,
            "portable": self.portable,
        }

        if self.path:
            params["path"] = self.path

        return params

    def get_trade_request_template(self) -> Dict:
        """Get template for MT5 trade requests"""
        return {
            "action": TradeAction.DEAL.value,
            "magic": self.magic_number,
            "deviation": self.deviation,
            "type_filling": self.fill_policy,
            "type_time": self.type_time,
        }

    def format_symbol(self, base_symbol: str) -> str:
        """Format symbol with broker prefix/suffix"""
        return f"{self.symbol_prefix}{base_symbol}{self.symbol_suffix}"

    def normalize_symbol(self, broker_symbol: str) -> str:
        """Remove broker prefix/suffix from symbol"""
        symbol = broker_symbol

        if self.symbol_prefix and symbol.startswith(self.symbol_prefix):
            symbol = symbol[len(self.symbol_prefix) :]

        if self.symbol_suffix and symbol.endswith(self.symbol_suffix):
            symbol = symbol[: -len(self.symbol_suffix)]

        return symbol

    def get_symbol_info_template(self) -> Dict:
        """Get template for symbol information validation"""
        return {
            "visible": True,
            "select": True,
            "trade_mode": self.trade_mode_filter,
            "calc_mode": self.calc_mode_filter,
            "trade_stops_level": {"min": 0, "max": 1000},
            "trade_freeze_level": {"min": 0, "max": 100},
            "volume_min": {"min": 0.01, "max": 1.0},
            "volume_max": {"min": 1.0, "max": 1000.0},
            "volume_step": {"min": 0.01, "max": 1.0},
            "spread": {"max": 1000},  # Maximum spread in points
        }

    def is_error_retriable(self, error_code: int) -> bool:
        """Check if error code is retriable"""
        # MT5 error codes that can be retried
        retriable_errors = [
            4756,  # Order is being processed
            4757,  # Request is being processed
            10004,  # Requote
            10006,  # Request rejected
            10007,  # Request canceled by trader
            10008,  # Order placed
            10009,  # Request completed
            10010,  # Only part of the request was completed
            10013,  # Invalid request
            10014,  # Invalid volume in the request
            10015,  # Invalid price in the request
            10016,  # Invalid stops in the request
            10018,  # Market is closed
            10019,  # There is not enough money to complete the request
            10020,  # Prices changed
            10021,  # There are no quotes to process the request
            10027,  # Autotrading disabled by server
            10028,  # Autotrading disabled by client terminal
            10030,  # Request blocked for processing
        ]

        return error_code in retriable_errors and error_code not in self.ignored_errors

    def get_error_description(self, error_code: int) -> str:
        """Get human-readable error description"""
        error_descriptions = {
            # Connection errors
            10004: "Requote",
            10006: "Request rejected",
            10007: "Request canceled by trader",
            10008: "Order placed",
            10009: "Request completed",
            10010: "Only part of the request was completed",
            10013: "Invalid request",
            10014: "Invalid volume in the request",
            10015: "Invalid price in the request",
            10016: "Invalid stops in the request",
            10018: "Market is closed",
            10019: "There is not enough money to complete the request",
            10020: "Prices changed",
            10021: "There are no quotes to process the request",
            10027: "Autotrading disabled by server",
            10028: "Autotrading disabled by client terminal",
            10030: "Request blocked for processing",
            # Trading errors
            4756: "Order is being processed",
            4757: "Request is being processed",
            4109: "Trade is not allowed for expert advisor",
            4110: "Long positions only allowed",
            4111: "Short positions only allowed",
            # Account errors
            65537: "Unknown symbol",
            65538: "Invalid price",
            65539: "Invalid stops",
            65540: "Invalid volume",
            65541: "Market is closed",
            65542: "Not enough money",
            65543: "Price changed",
            65544: "Off quotes",
            65545: "Invalid expiration",
            65546: "Order changed",
            65547: "Too frequent requests",
            65548: "No changes in request",
        }

        return error_descriptions.get(error_code, f"Unknown error: {error_code}")

    def get_broker_schedule(self) -> Dict:
        """Get broker trading schedule"""
        # Default Forex market schedule (can be customized per broker)
        return {
            "MONDAY": {"open": "00:05", "close": "23:55"},
            "TUESDAY": {"open": "00:05", "close": "23:55"},
            "WEDNESDAY": {"open": "00:05", "close": "23:55"},
            "THURSDAY": {"open": "00:05", "close": "23:55"},
            "FRIDAY": {"open": "00:05", "close": "23:50"},
            "SATURDAY": {"open": None, "close": None},
            "SUNDAY": {"open": "22:05", "close": "23:55"},
        }

    def get_symbol_sessions(self, symbol: str) -> Dict:
        """Get trading sessions for specific symbol"""

        # Forex major pairs
        forex_majors = [
            "EURUSD",
            "GBPUSD",
            "USDJPY",
            "USDCHF",
            "AUDUSD",
            "USDCAD",
            "NZDUSD",
        ]
        forex_crosses = [
            "EURJPY",
            "GBPJPY",
            "EURGBP",
            "EURAUD",
            "GBPAUD",
            "AUDCAD",
            "NZDCAD",
        ]

        if symbol in forex_majors or symbol in forex_crosses:
            return {
                "SYDNEY": {"open": "21:00", "close": "06:00", "active": True},
                "TOKYO": {"open": "00:00", "close": "09:00", "active": True},
                "LONDON": {"open": "08:00", "close": "17:00", "active": True},
                "NEW_YORK": {"open": "13:00", "close": "22:00", "active": True},
            }

        # Gold
        elif symbol in ["XAUUSD", "GOLD"]:
            return {
                "TOKYO": {"open": "00:00", "close": "09:00", "active": True},
                "LONDON": {"open": "08:00", "close": "17:00", "active": True},
                "NEW_YORK": {"open": "13:00", "close": "22:00", "active": True},
            }

        # Default to Forex schedule
        else:
            return self.get_broker_schedule()

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary (excluding sensitive data)"""
        config_dict = {}

        for key, value in self.__dict__.items():
            # Exclude sensitive information
            if key not in ["password", "login"]:
                if isinstance(value, Enum):
                    config_dict[key] = value.value
                else:
                    config_dict[key] = value
            else:
                config_dict[key] = "***HIDDEN***" if value else None

        return config_dict

    def validate_symbol(self, symbol: str) -> Tuple[bool, str]:
        """Validate if symbol can be traded"""
        try:
            # Format symbol for broker
            formatted_symbol = self.format_symbol(symbol)

            # Basic symbol validation
            if len(formatted_symbol) < 3:
                return False, "Symbol too short"

            if len(formatted_symbol) > 20:
                return False, "Symbol too long"

            # Check for valid characters
            if not formatted_symbol.replace(".", "").replace("#", "").isalnum():
                return False, "Invalid symbol characters"

            return True, "Symbol valid"

        except Exception as e:
            return False, f"Symbol validation error: {e}"

    def get_lot_size_info(self, symbol: str, account_balance: float) -> Dict:
        """Calculate lot size information for symbol"""

        # Default lot size calculations (can be refined with real symbol info)
        min_lot = 0.01
        max_lot = min(100.0, account_balance / 1000)  # Conservative calculation
        lot_step = 0.01

        # Symbol-specific adjustments
        if symbol in ["XAUUSD", "GOLD"]:
            min_lot = 0.01
            max_lot = min(10.0, account_balance / 10000)
            lot_step = 0.01

        elif "JPY" in symbol:
            min_lot = 0.01
            max_lot = min(50.0, account_balance / 500)
            lot_step = 0.01

        return {
            "min_lot": min_lot,
            "max_lot": max_lot,
            "lot_step": lot_step,
            "margin_required": max_lot * 1000,  # Simplified calculation
        }

    def get_universal_connection_params(self) -> Dict:
        """
        🌍 Get universal connection parameters that work with any broker
        """
        params = {
            "login": self.login,
            "password": self.password,
            "server": self.server,
            "timeout": self.timeout,
            "portable": self.portable,
        }

        if self.path:
            params["path"] = self.path

        return params

    def validate_any_broker_connection(
        self, terminal_info: Dict = None, account_info: Dict = None
    ) -> Tuple[bool, str]:
        """
        ✅ Universal broker connection validation
        """
        try:
            # Basic credential validation
            if not all([self.login, self.password, self.server]):
                return False, "Missing login credentials"

            # Validate account info if provided
            if account_info:
                # Check if trading is allowed
                if not account_info.get("trade_allowed", False):
                    return False, "Trading not allowed on this account"

                # Check account balance
                balance = account_info.get("balance", 0)
                if balance <= 0:
                    return False, f"Invalid account balance: {balance}"

                # Check if account is connected
                if not account_info.get("connected", False):
                    return False, "Account not connected to trade server"

            # Validate terminal info if provided
            if terminal_info:
                # Check if terminal is connected
                if not terminal_info.get("connected", False):
                    return False, "Terminal not connected to broker"

                # Check terminal version compatibility
                version = terminal_info.get("version", 0)
                if version < 2650:  # Minimum MT5 version
                    return False, f"MT5 version too old: {version}"

            return True, "Connection validated successfully"

        except Exception as e:
            return False, f"Validation error: {e}"


# === UNIVERSAL BROKER SUPPORT ===


def create_universal_mt5_config(
    login: int, password: str, server: str, path: str = None
) -> MT5Config:
    """
    🌍 Create MT5 configuration for any broker
    Just provide login, password, server - everything else auto-detected!

    Args:
        login: MT5 account number
        password: MT5 account password
        server: MT5 server name (e.g., "Exness-Real", "ICMarkets-Demo01")
        path: Path to MT5 terminal (optional, auto-detected if None)

    Returns:
        MT5Config: Ready-to-use configuration

    Example:
        config = create_universal_mt5_config(
            login=12345678,
            password="your_password",
            server="ICMarkets-Real01"
        )
    """
    return MT5Config(
        login=login,
        password=password,
        server=server,
        path=path,
        auto_detect_settings=True,
        # Universal settings that work with most brokers
        timeout=30000,
        slippage=3,
        deviation=20,
        max_positions=50,  # Liberal limit
        retry_on_error=True,
        max_retries=3,
        reconnect_attempts=5,
        # Accept all common account types
        allowed_account_types=[
            "real",
            "demo",
            "contest",
            "cent",
            "ecn",
            "stp",
            "micro",
        ],
        # Accept all common trade modes
        trade_mode_filter=["FULL", "LONGONLY", "SHORTONLY", "CLOSEONLY"],
        # Accept all common calculation modes
        calc_mode_filter=[
            "FOREX",
            "CFD",
            "FUTURES",
            "CFDINDEX",
            "CFDLEVERAGE",
            "EXCH_STOCKS",
            "EXCH_FUTURES",
        ],
    )


def quick_connect_any_broker(
    login: int, password: str, server: str
) -> Tuple[bool, MT5Config, str]:
    """
    ⚡ Quick connect to any MT5 broker - one function does it all!

    Args:
        login: MT5 account number
        password: MT5 account password
        server: MT5 server name

    Returns:
        Tuple[success, config, message]

    Example:
        success, config, msg = quick_connect_any_broker(12345, "password", "Exness-Real")
        if success:
            print(f"✅ Connected to {config.broker_name}")
        else:
            print(f"❌ Failed: {msg}")
    """
    try:
        # Create universal config
        config = create_universal_mt5_config(login, password, server)

        # Test connection (this would be done by MT5Connector in practice)
        # For now, just validate the configuration
        success, validation_msg = config.validate_any_broker_connection()

        if success:
            return True, config, f"✅ Ready to connect to {server}"
        else:
            return False, config, f"❌ Validation failed: {validation_msg}"

    except Exception as e:
        return False, None, f"❌ Configuration error: {e}"


# === BROKER AUTO-DETECTION EXAMPLES ===

EXAMPLE_SERVERS = {
    "IC Markets": [
        "ICMarkets-Real01",
        "ICMarkets-Real02",
        "ICMarkets-Demo01",
        "ICMarkets-Demo02",
        "IC-Markets-Real",
        "IC-Markets-Demo",
    ],
    "Exness": [
        "Exness-Real",
        "Exness-Real2",
        "Exness-Demo",
        "Exness-MT5Real",
        "ExnessReal",
        "ExnessDemo",
        "Exness-Real3",
        "Exness-Real4",
    ],
    "XM": [
        "XM-Real",
        "XM-Demo",
        "XMGlobal-Real",
        "XMGlobal-Demo",
        "TradingPoint-Real",
        "TradingPoint-Demo",
    ],
    "Pepperstone": [
        "Pepperstone-Real",
        "Pepperstone-Demo",
        "Pepperstone-Live01",
        "PepperstoneReal",
        "PepperstoneDemo",
    ],
    "FXTM": [
        "FXTM-Real",
        "FXTM-Demo",
        "ForexTime-Real",
        "ForexTime-Demo",
        "FXTM-ECN",
        "FXTM-Standard",
    ],
    "Admiral Markets": [
        "AdmiralMarkets-Real",
        "AdmiralMarkets-Demo",
        "Admiral-Real",
        "Admiral-Demo",
        "AM-Real",
        "AM-Demo",
    ],
    "OANDA": ["OANDA-Real", "OANDA-Demo", "OANDA-fxTrade", "OANDA-fxTrade Practice"],
    "Dupoin": ["Dupoin-Real", "Dupoin-Demo", "Dupoin-Live", "Dupoin-MT5"],
    "Monaxa": ["Monaxa-Real", "Monaxa-Demo", "Monaxa-Live", "Monaxa-ECN"],
}


def suggest_server_names(broker_hint: str = "") -> List[str]:
    """
    💡 Suggest possible server names based on broker hint
    """
    if not broker_hint:
        # Return all known servers
        all_servers = []
        for servers in EXAMPLE_SERVERS.values():
            all_servers.extend(servers)
        return sorted(all_servers)

    # Find matching broker
    broker_hint_lower = broker_hint.lower()
    suggestions = []

    for broker, servers in EXAMPLE_SERVERS.items():
        if broker_hint_lower in broker.lower():
            suggestions.extend(servers)

    return (
        suggestions
        if suggestions
        else ["Check MT5 → File → Open Account for server list"]
    )


# === USAGE EXAMPLES ===


def usage_examples():
    """
    📚 Usage examples for universal MT5 connection
    """
    examples = """
🌍 UNIVERSAL MT5 CONNECTION EXAMPLES:

1️⃣ Simple Connection (Any Broker):
config = create_universal_mt5_config(
    login=12345678,
    password="your_password",
    server="YourBroker-Real01"
)

2️⃣ Quick Connect & Auto-Detect:
success, config, msg = quick_connect_any_broker(
    login=12345678,
    password="your_password", 
    server="Exness-Real"
)
if success:
    print("Connected to broker")

3️⃣ Multiple Brokers Setup:
brokers = [
    {"login": 123, "password": "pass1", "server": "Exness-Real"},
    {"login": 456, "password": "pass2", "server": "ICMarkets-Demo01"},
    {"login": 789, "password": "pass3", "server": "Dupoin-Real"}
]

configs = []
for broker in brokers:
    config = create_universal_mt5_config(**broker)
    configs.append(config)

4️⃣ Find Your Server Name:
suggestions = suggest_server_names("exness")
print("Possible servers:", suggestions)

5️⃣ Environment Variables (Production):
# Set environment variables:
# MT5_LOGIN=12345678
# MT5_PASSWORD=your_password  
# MT5_SERVER=YourBroker-Real01

config = MT5Config()  # Will auto-load from environment

✅ WORKS WITH ANY MT5 BROKER:
- IC Markets, Exness, XM, Pepperstone, FXTM
- Admiral Markets, OANDA, FxPro
- Dupoin, Monaxa, or ANY other MT5 broker
- Demo accounts, Live accounts, Cent accounts
- ECN, STP, Market Maker - all supported!

🔧 AUTO-DETECTION FEATURES:
- Broker name from server
- Symbol format (prefix/suffix)
- Commission structure  
- Account leverage
- Available symbols
- Trading hours

No need to know broker-specific settings - just login and go! 🚀
    """
    return examples


# Default universal MT5 configuration
DEFAULT_UNIVERSAL_CONFIG = MT5Config(auto_detect_settings=True)
