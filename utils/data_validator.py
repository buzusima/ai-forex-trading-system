"""
Institutional-grade Forex AI Trading System
Data Validator Module

This module provides comprehensive data validation for all system components
including market data, trading signals, model inputs, and system parameters.
"""

import re
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from decimal import Decimal, InvalidOperation
import json
import math


class ValidationSeverity(Enum):
    """Validation severity levels"""

    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ValidationResult:
    """Validation result container"""

    def __init__(
        self,
        is_valid: bool = True,
        severity: ValidationSeverity = ValidationSeverity.INFO,
        message: str = "",
        field: str = "",
        value: Any = None,
    ):
        self.is_valid = is_valid
        self.severity = severity
        self.message = message
        self.field = field
        self.value = value
        self.timestamp = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "severity": self.severity.value,
            "message": self.message,
            "field": self.field,
            "value": str(self.value) if self.value is not None else None,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ValidationConfig:
    """Configuration for data validation"""

    # Market data validation
    max_price_change_percent: float = 10.0  # Maximum price change in single tick
    min_spread_points: float = 0.1  # Minimum spread in points
    max_spread_points: float = 100.0  # Maximum spread in points
    max_volume: float = 1000.0  # Maximum volume per tick
    price_precision: int = 5  # Decimal precision for prices

    # Trading validation
    min_lot_size: float = 0.01  # Minimum lot size
    max_lot_size: float = 100.0  # Maximum lot size
    max_open_positions: int = 50  # Maximum open positions
    min_stop_loss_points: float = 5.0  # Minimum SL in points
    max_stop_loss_points: float = 1000.0  # Maximum SL in points
    min_take_profit_points: float = 5.0  # Minimum TP in points
    max_take_profit_points: float = 1000.0  # Maximum TP in points

    # Model validation
    min_prediction_confidence: float = 0.6  # Minimum prediction confidence
    max_model_latency_ms: float = 1000.0  # Maximum model inference time
    required_features: List[str] = field(default_factory=list)
    feature_value_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    # Time validation
    market_open_hours: Dict[str, List[Tuple[int, int]]] = field(default_factory=dict)
    weekend_trading_allowed: bool = False
    max_data_age_minutes: int = 5  # Maximum age for real-time data


class MarketDataValidator:
    """Validator for market data (OHLCV, spreads, etc.)"""

    def __init__(self, config: ValidationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.previous_prices = {}  # Store previous prices for change validation

    def validate_ohlcv(self, data: Dict[str, Any]) -> List[ValidationResult]:
        """Validate OHLCV data"""
        results = []

        # Required fields check
        required_fields = [
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "timestamp",
        ]
        for field in required_fields:
            if field not in data:
                results.append(
                    ValidationResult(
                        False,
                        ValidationSeverity.ERROR,
                        f"Missing required field: {field}",
                        field,
                    )
                )
                return results

        symbol = data["symbol"]

        # Price validation
        prices = [data["open"], data["high"], data["low"], data["close"]]

        # Check for valid price values
        for i, price_name in enumerate(["open", "high", "low", "close"]):
            price = prices[i]
            result = self._validate_price(price, price_name, symbol)
            if result:
                results.append(result)

        # OHLC relationship validation
        if all(isinstance(p, (int, float)) and not math.isnan(p) for p in prices):
            open_price, high_price, low_price, close_price = prices

            # High should be >= all other prices
            if high_price < max(open_price, low_price, close_price):
                results.append(
                    ValidationResult(
                        False,
                        ValidationSeverity.ERROR,
                        f"High price {high_price} is not the highest",
                        "high",
                        high_price,
                    )
                )

            # Low should be <= all other prices
            if low_price > min(open_price, high_price, close_price):
                results.append(
                    ValidationResult(
                        False,
                        ValidationSeverity.ERROR,
                        f"Low price {low_price} is not the lowest",
                        "low",
                        low_price,
                    )
                )

            # Price change validation
            if symbol in self.previous_prices:
                prev_close = self.previous_prices[symbol]
                change_percent = abs(close_price - prev_close) / prev_close * 100

                if change_percent > self.config.max_price_change_percent:
                    results.append(
                        ValidationResult(
                            False,
                            ValidationSeverity.WARNING,
                            f"Large price change: {change_percent:.2f}%",
                            "close",
                            close_price,
                        )
                    )

            self.previous_prices[symbol] = close_price

        # Volume validation
        volume = data["volume"]
        if not isinstance(volume, (int, float)) or volume < 0:
            results.append(
                ValidationResult(
                    False,
                    ValidationSeverity.ERROR,
                    f"Invalid volume: {volume}",
                    "volume",
                    volume,
                )
            )
        elif volume > self.config.max_volume:
            results.append(
                ValidationResult(
                    False,
                    ValidationSeverity.WARNING,
                    f"High volume detected: {volume}",
                    "volume",
                    volume,
                )
            )

        # Timestamp validation
        timestamp_result = self._validate_timestamp(data["timestamp"])
        if timestamp_result:
            results.append(timestamp_result)

        return results

    def validate_tick_data(self, data: Dict[str, Any]) -> List[ValidationResult]:
        """Validate tick data"""
        results = []

        required_fields = ["symbol", "bid", "ask", "timestamp"]
        for field in required_fields:
            if field not in data:
                results.append(
                    ValidationResult(
                        False,
                        ValidationSeverity.ERROR,
                        f"Missing required field: {field}",
                        field,
                    )
                )
                return results

        # Price validation
        bid = data["bid"]
        ask = data["ask"]

        bid_result = self._validate_price(bid, "bid", data["symbol"])
        if bid_result:
            results.append(bid_result)

        ask_result = self._validate_price(ask, "ask", data["symbol"])
        if ask_result:
            results.append(ask_result)

        # Spread validation
        if isinstance(bid, (int, float)) and isinstance(ask, (int, float)):
            if ask <= bid:
                results.append(
                    ValidationResult(
                        False,
                        ValidationSeverity.ERROR,
                        f"Ask {ask} <= Bid {bid}",
                        "spread",
                    )
                )
            else:
                spread_points = (ask - bid) * 10000  # Convert to points

                if spread_points < self.config.min_spread_points:
                    results.append(
                        ValidationResult(
                            False,
                            ValidationSeverity.WARNING,
                            f"Very tight spread: {spread_points:.1f} points",
                            "spread",
                        )
                    )
                elif spread_points > self.config.max_spread_points:
                    results.append(
                        ValidationResult(
                            False,
                            ValidationSeverity.WARNING,
                            f"Very wide spread: {spread_points:.1f} points",
                            "spread",
                        )
                    )

        # Timestamp validation
        timestamp_result = self._validate_timestamp(data["timestamp"])
        if timestamp_result:
            results.append(timestamp_result)

        return results

    def _validate_price(
        self, price: Any, field_name: str, symbol: str
    ) -> Optional[ValidationResult]:
        """Validate individual price value"""
        if not isinstance(price, (int, float)):
            return ValidationResult(
                False,
                ValidationSeverity.ERROR,
                f"Price must be numeric: {price}",
                field_name,
                price,
            )

        if math.isnan(price) or math.isinf(price):
            return ValidationResult(
                False,
                ValidationSeverity.ERROR,
                f"Invalid price value: {price}",
                field_name,
                price,
            )

        if price <= 0:
            return ValidationResult(
                False,
                ValidationSeverity.ERROR,
                f"Price must be positive: {price}",
                field_name,
                price,
            )

        # Check decimal precision
        decimal_places = len(str(price).split(".")[-1]) if "." in str(price) else 0
        if decimal_places > self.config.price_precision:
            return ValidationResult(
                False,
                ValidationSeverity.WARNING,
                f"Price has too many decimal places: {decimal_places}",
                field_name,
                price,
            )

        return None

    def _validate_timestamp(self, timestamp: Any) -> Optional[ValidationResult]:
        """Validate timestamp"""
        try:
            if isinstance(timestamp, str):
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            elif isinstance(timestamp, (int, float)):
                dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
            elif isinstance(timestamp, datetime):
                dt = timestamp
            else:
                return ValidationResult(
                    False,
                    ValidationSeverity.ERROR,
                    f"Invalid timestamp format: {timestamp}",
                    "timestamp",
                    timestamp,
                )

            # Check if timestamp is too old
            age_minutes = (datetime.now(timezone.utc) - dt).total_seconds() / 60
            if age_minutes > self.config.max_data_age_minutes:
                return ValidationResult(
                    False,
                    ValidationSeverity.WARNING,
                    f"Data is {age_minutes:.1f} minutes old",
                    "timestamp",
                    timestamp,
                )

            # Check if timestamp is in the future
            if dt > datetime.now(timezone.utc) + timedelta(minutes=1):
                return ValidationResult(
                    False,
                    ValidationSeverity.ERROR,
                    "Timestamp is in the future",
                    "timestamp",
                    timestamp,
                )

        except Exception as e:
            return ValidationResult(
                False,
                ValidationSeverity.ERROR,
                f"Timestamp parsing error: {str(e)}",
                "timestamp",
                timestamp,
            )

        return None


class TradingValidator:
    """Validator for trading operations and parameters"""

    def __init__(self, config: ValidationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def validate_order(self, order: Dict[str, Any]) -> List[ValidationResult]:
        """Validate trading order"""
        results = []

        # Required fields
        required_fields = ["symbol", "action", "volume", "order_type"]
        for field in required_fields:
            if field not in order:
                results.append(
                    ValidationResult(
                        False,
                        ValidationSeverity.ERROR,
                        f"Missing required field: {field}",
                        field,
                    )
                )
                return results

        # Symbol validation
        symbol_result = self._validate_symbol(order["symbol"])
        if symbol_result:
            results.append(symbol_result)

        # Action validation
        if order["action"] not in ["BUY", "SELL"]:
            results.append(
                ValidationResult(
                    False,
                    ValidationSeverity.ERROR,
                    f"Invalid action: {order['action']}",
                    "action",
                    order["action"],
                )
            )

        # Volume validation
        volume_result = self._validate_volume(order["volume"])
        if volume_result:
            results.append(volume_result)

        # Order type validation
        valid_types = ["MARKET", "LIMIT", "STOP", "STOP_LIMIT"]
        if order["order_type"] not in valid_types:
            results.append(
                ValidationResult(
                    False,
                    ValidationSeverity.ERROR,
                    f"Invalid order type: {order['order_type']}",
                    "order_type",
                )
            )

        # Price validation for limit orders
        if order["order_type"] in ["LIMIT", "STOP_LIMIT"] and "price" in order:
            price_result = self._validate_price(order["price"])
            if price_result:
                results.append(price_result)

        # Stop loss validation
        if "stop_loss" in order and order["stop_loss"] is not None:
            sl_result = self._validate_stop_loss(order["stop_loss"], order.get("price"))
            if sl_result:
                results.append(sl_result)

        # Take profit validation
        if "take_profit" in order and order["take_profit"] is not None:
            tp_result = self._validate_take_profit(
                order["take_profit"], order.get("price")
            )
            if tp_result:
                results.append(tp_result)

        return results

    def validate_position(self, position: Dict[str, Any]) -> List[ValidationResult]:
        """Validate trading position"""
        results = []

        required_fields = ["symbol", "volume", "open_price", "open_time", "profit"]
        for field in required_fields:
            if field not in position:
                results.append(
                    ValidationResult(
                        False,
                        ValidationSeverity.ERROR,
                        f"Missing required field: {field}",
                        field,
                    )
                )
                return results

        # Volume validation
        volume_result = self._validate_volume(position["volume"])
        if volume_result:
            results.append(volume_result)

        # Price validation
        price_result = self._validate_price(position["open_price"])
        if price_result:
            results.append(price_result)

        # Profit validation
        profit = position["profit"]
        if not isinstance(profit, (int, float)):
            results.append(
                ValidationResult(
                    False,
                    ValidationSeverity.ERROR,
                    f"Invalid profit value: {profit}",
                    "profit",
                    profit,
                )
            )

        return results

    def validate_account_info(self, account: Dict[str, Any]) -> List[ValidationResult]:
        """Validate account information"""
        results = []

        required_fields = ["balance", "equity", "margin", "free_margin"]
        for field in required_fields:
            if field not in account:
                results.append(
                    ValidationResult(
                        False,
                        ValidationSeverity.ERROR,
                        f"Missing required field: {field}",
                        field,
                    )
                )
                continue

            value = account[field]
            if not isinstance(value, (int, float)) or value < 0:
                results.append(
                    ValidationResult(
                        False,
                        ValidationSeverity.ERROR,
                        f"Invalid {field}: {value}",
                        field,
                        value,
                    )
                )

        # Validate relationships
        if all(field in account for field in required_fields):
            balance = account["balance"]
            equity = account["equity"]
            margin = account["margin"]
            free_margin = account["free_margin"]

            # Equity should be close to balance + floating profit/loss
            if abs(equity - balance) > balance * 0.1:  # Allow 10% difference
                results.append(
                    ValidationResult(
                        False,
                        ValidationSeverity.WARNING,
                        f"Large difference between balance ({balance}) and equity ({equity})",
                        "equity",
                        equity,
                    )
                )

            # Free margin should be positive for new trades
            if free_margin <= 0:
                results.append(
                    ValidationResult(
                        False,
                        ValidationSeverity.WARNING,
                        f"Low free margin: {free_margin}",
                        "free_margin",
                        free_margin,
                    )
                )

        return results

    def _validate_symbol(self, symbol: str) -> Optional[ValidationResult]:
        """Validate trading symbol"""
        if not isinstance(symbol, str):
            return ValidationResult(
                False,
                ValidationSeverity.ERROR,
                f"Symbol must be string: {symbol}",
                "symbol",
                symbol,
            )

        # Basic forex symbol pattern (e.g., EURUSD, GBPJPY)
        if not re.match(r"^[A-Z]{6}$", symbol):
            return ValidationResult(
                False,
                ValidationSeverity.WARNING,
                f"Unusual symbol format: {symbol}",
                "symbol",
                symbol,
            )

        return None

    def _validate_volume(self, volume: Any) -> Optional[ValidationResult]:
        """Validate trading volume"""
        if not isinstance(volume, (int, float)):
            return ValidationResult(
                False,
                ValidationSeverity.ERROR,
                f"Volume must be numeric: {volume}",
                "volume",
                volume,
            )

        if volume <= 0:
            return ValidationResult(
                False,
                ValidationSeverity.ERROR,
                f"Volume must be positive: {volume}",
                "volume",
                volume,
            )

        if volume < self.config.min_lot_size:
            return ValidationResult(
                False,
                ValidationSeverity.ERROR,
                f"Volume below minimum: {volume} < {self.config.min_lot_size}",
                "volume",
                volume,
            )

        if volume > self.config.max_lot_size:
            return ValidationResult(
                False,
                ValidationSeverity.WARNING,
                f"Large volume: {volume}",
                "volume",
                volume,
            )

        return None

    def _validate_price(self, price: Any) -> Optional[ValidationResult]:
        """Validate price value"""
        if not isinstance(price, (int, float)):
            return ValidationResult(
                False,
                ValidationSeverity.ERROR,
                f"Price must be numeric: {price}",
                "price",
                price,
            )

        if price <= 0:
            return ValidationResult(
                False,
                ValidationSeverity.ERROR,
                f"Price must be positive: {price}",
                "price",
                price,
            )

        return None

    def _validate_stop_loss(
        self, stop_loss: float, entry_price: Optional[float]
    ) -> Optional[ValidationResult]:
        """Validate stop loss level"""
        if not isinstance(stop_loss, (int, float)):
            return ValidationResult(
                False,
                ValidationSeverity.ERROR,
                f"Stop loss must be numeric: {stop_loss}",
                "stop_loss",
                stop_loss,
            )

        if entry_price:
            sl_distance = abs(entry_price - stop_loss) * 10000  # Convert to points

            if sl_distance < self.config.min_stop_loss_points:
                return ValidationResult(
                    False,
                    ValidationSeverity.WARNING,
                    f"Stop loss too close: {sl_distance:.1f} points",
                    "stop_loss",
                )

            if sl_distance > self.config.max_stop_loss_points:
                return ValidationResult(
                    False,
                    ValidationSeverity.WARNING,
                    f"Stop loss too far: {sl_distance:.1f} points",
                    "stop_loss",
                )

        return None

    def _validate_take_profit(
        self, take_profit: float, entry_price: Optional[float]
    ) -> Optional[ValidationResult]:
        """Validate take profit level"""
        if not isinstance(take_profit, (int, float)):
            return ValidationResult(
                False,
                ValidationSeverity.ERROR,
                f"Take profit must be numeric: {take_profit}",
                "take_profit",
                take_profit,
            )

        if entry_price:
            tp_distance = abs(entry_price - take_profit) * 10000  # Convert to points

            if tp_distance < self.config.min_take_profit_points:
                return ValidationResult(
                    False,
                    ValidationSeverity.WARNING,
                    f"Take profit too close: {tp_distance:.1f} points",
                    "take_profit",
                )

            if tp_distance > self.config.max_take_profit_points:
                return ValidationResult(
                    False,
                    ValidationSeverity.WARNING,
                    f"Take profit too far: {tp_distance:.1f} points",
                    "take_profit",
                )

        return None


class ModelValidator:
    """Validator for ML model inputs and outputs"""

    def __init__(self, config: ValidationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def validate_features(
        self, features: Union[np.ndarray, pd.DataFrame, Dict]
    ) -> List[ValidationResult]:
        """Validate model input features"""
        results = []

        # Convert to appropriate format
        if isinstance(features, dict):
            feature_data = features
        elif isinstance(features, pd.DataFrame):
            feature_data = features.to_dict("records")[0] if len(features) > 0 else {}
        elif isinstance(features, np.ndarray):
            if len(self.config.required_features) == len(features):
                feature_data = dict(zip(self.config.required_features, features))
            else:
                results.append(
                    ValidationResult(
                        False,
                        ValidationSeverity.ERROR,
                        f"Feature array length mismatch: {len(features)} vs {len(self.config.required_features)}",
                        "features",
                    )
                )
                return results
        else:
            results.append(
                ValidationResult(
                    False,
                    ValidationSeverity.ERROR,
                    f"Invalid feature format: {type(features)}",
                    "features",
                )
            )
            return results

        # Check required features
        for feature in self.config.required_features:
            if feature not in feature_data:
                results.append(
                    ValidationResult(
                        False,
                        ValidationSeverity.ERROR,
                        f"Missing required feature: {feature}",
                        feature,
                    )
                )
                continue

            value = feature_data[feature]

            # Check for valid numeric value
            if not isinstance(value, (int, float, np.number)):
                results.append(
                    ValidationResult(
                        False,
                        ValidationSeverity.ERROR,
                        f"Feature must be numeric: {feature} = {value}",
                        feature,
                        value,
                    )
                )
                continue

            # Check for NaN or infinite values
            if math.isnan(value) or math.isinf(value):
                results.append(
                    ValidationResult(
                        False,
                        ValidationSeverity.ERROR,
                        f"Invalid feature value: {feature} = {value}",
                        feature,
                        value,
                    )
                )
                continue

            # Check value ranges
            if feature in self.config.feature_value_ranges:
                min_val, max_val = self.config.feature_value_ranges[feature]
                if value < min_val or value > max_val:
                    results.append(
                        ValidationResult(
                            False,
                            ValidationSeverity.WARNING,
                            f"Feature out of range: {feature} = {value} (expected {min_val}-{max_val})",
                            feature,
                            value,
                        )
                    )

        return results

    def validate_prediction(self, prediction: Dict[str, Any]) -> List[ValidationResult]:
        """Validate model prediction output"""
        results = []

        required_fields = ["signal", "confidence", "probability"]
        for field in required_fields:
            if field not in prediction:
                results.append(
                    ValidationResult(
                        False,
                        ValidationSeverity.ERROR,
                        f"Missing required field: {field}",
                        field,
                    )
                )
                continue

        # Signal validation
        signal = prediction.get("signal")
        if signal not in ["BUY", "SELL", "HOLD"]:
            results.append(
                ValidationResult(
                    False,
                    ValidationSeverity.ERROR,
                    f"Invalid signal: {signal}",
                    "signal",
                    signal,
                )
            )

        # Confidence validation
        confidence = prediction.get("confidence", 0)
        if not isinstance(confidence, (int, float)) or not (0 <= confidence <= 1):
            results.append(
                ValidationResult(
                    False,
                    ValidationSeverity.ERROR,
                    f"Invalid confidence: {confidence}",
                    "confidence",
                    confidence,
                )
            )
        elif confidence < self.config.min_prediction_confidence:
            results.append(
                ValidationResult(
                    False,
                    ValidationSeverity.WARNING,
                    f"Low confidence: {confidence}",
                    "confidence",
                    confidence,
                )
            )

        # Probability validation
        probability = prediction.get("probability", {})
        if isinstance(probability, dict):
            prob_sum = sum(probability.values())
            if abs(prob_sum - 1.0) > 0.01:  # Allow small floating point errors
                results.append(
                    ValidationResult(
                        False,
                        ValidationSeverity.WARNING,
                        f"Probabilities don't sum to 1.0: {prob_sum}",
                        "probability",
                    )
                )

        # Latency validation
        if "inference_time_ms" in prediction:
            latency = prediction["inference_time_ms"]
            if latency > self.config.max_model_latency_ms:
                results.append(
                    ValidationResult(
                        False,
                        ValidationSeverity.WARNING,
                        f"High model latency: {latency}ms",
                        "inference_time_ms",
                        latency,
                    )
                )

        return results


class DataValidator:
    """Main data validator class"""

    def __init__(self, config: ValidationConfig = None):
        self.config = config or ValidationConfig()
        self.market_validator = MarketDataValidator(self.config)
        self.trading_validator = TradingValidator(self.config)
        self.model_validator = ModelValidator(self.config)
        self.logger = logging.getLogger(__name__)

    def validate_market_data(
        self, data: Dict[str, Any], data_type: str = "ohlcv"
    ) -> List[ValidationResult]:
        """Validate market data"""
        if data_type == "ohlcv":
            return self.market_validator.validate_ohlcv(data)
        elif data_type == "tick":
            return self.market_validator.validate_tick_data(data)
        else:
            return [
                ValidationResult(
                    False,
                    ValidationSeverity.ERROR,
                    f"Unknown market data type: {data_type}",
                    "data_type",
                )
            ]

    def validate_trading_data(
        self, data: Dict[str, Any], data_type: str
    ) -> List[ValidationResult]:
        """Validate trading data"""
        if data_type == "order":
            return self.trading_validator.validate_order(data)
        elif data_type == "position":
            return self.trading_validator.validate_position(data)
        elif data_type == "account":
            return self.trading_validator.validate_account_info(data)
        else:
            return [
                ValidationResult(
                    False,
                    ValidationSeverity.ERROR,
                    f"Unknown trading data type: {data_type}",
                    "data_type",
                )
            ]

    def validate_model_data(self, data: Any, data_type: str) -> List[ValidationResult]:
        """Validate model data"""
        if data_type == "features":
            return self.model_validator.validate_features(data)
        elif data_type == "prediction":
            return self.model_validator.validate_prediction(data)
        else:
            return [
                ValidationResult(
                    False,
                    ValidationSeverity.ERROR,
                    f"Unknown model data type: {data_type}",
                    "data_type",
                )
            ]

    def validate_batch(
        self, data_batch: List[Dict[str, Any]], data_type: str, category: str = "market"
    ) -> Dict[str, List[ValidationResult]]:
        """Validate batch of data"""
        results = {}

        for i, data in enumerate(data_batch):
            key = f"{category}_{data_type}_{i}"

            if category == "market":
                results[key] = self.validate_market_data(data, data_type)
            elif category == "trading":
                results[key] = self.validate_trading_data(data, data_type)
            elif category == "model":
                results[key] = self.validate_model_data(data, data_type)
            else:
                results[key] = [
                    ValidationResult(
                        False,
                        ValidationSeverity.ERROR,
                        f"Unknown category: {category}",
                        "category",
                    )
                ]

        return results

    def get_validation_summary(
        self, results: Union[List[ValidationResult], Dict[str, List[ValidationResult]]]
    ) -> Dict[str, Any]:
        """Get validation summary statistics"""
        if isinstance(results, list):
            all_results = results
        else:
            all_results = []
            for result_list in results.values():
                all_results.extend(result_list)

        summary = {
            "total_validations": len(all_results),
            "valid_count": sum(1 for r in all_results if r.is_valid),
            "invalid_count": sum(1 for r in all_results if not r.is_valid),
            "severity_counts": {
                "INFO": sum(
                    1 for r in all_results if r.severity == ValidationSeverity.INFO
                ),
                "WARNING": sum(
                    1 for r in all_results if r.severity == ValidationSeverity.WARNING
                ),
                "ERROR": sum(
                    1 for r in all_results if r.severity == ValidationSeverity.ERROR
                ),
                "CRITICAL": sum(
                    1 for r in all_results if r.severity == ValidationSeverity.CRITICAL
                ),
            },
            "validation_rate": (
                sum(1 for r in all_results if r.is_valid) / len(all_results)
                if all_results
                else 1.0
            ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return summary

    def log_validation_results(
        self, results: List[ValidationResult], context: str = ""
    ):
        """Log validation results"""
        for result in results:
            level = logging.INFO if result.is_valid else logging.ERROR
            if result.severity == ValidationSeverity.WARNING:
                level = logging.WARNING
            elif result.severity == ValidationSeverity.CRITICAL:
                level = logging.CRITICAL

            message = f"{context} - {result.message}"
            if result.field:
                message += f" (Field: {result.field})"

            self.logger.log(
                level,
                message,
                extra={
                    "validation_field": result.field,
                    "validation_value": result.value,
                    "validation_severity": result.severity.value,
                },
            )


# Example usage and testing
if __name__ == "__main__":
    # Test configuration
    config = ValidationConfig(
        max_price_change_percent=5.0,
        min_lot_size=0.01,
        max_lot_size=10.0,
        required_features=["rsi", "macd", "sma_20", "volume"],
        feature_value_ranges={
            "rsi": (0, 100),
            "macd": (-1, 1),
            "sma_20": (0.5, 2.0),
            "volume": (0, 1000000),
        },
    )

    validator = DataValidator(config)

    # Test market data validation
    ohlcv_data = {
        "symbol": "EURUSD",
        "open": 1.1234,
        "high": 1.1250,
        "low": 1.1220,
        "close": 1.1245,
        "volume": 1000,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    market_results = validator.validate_market_data(ohlcv_data, "ohlcv")
    print("Market Data Validation:")
    for result in market_results:
        print(f"  {result.severity.value}: {result.message}")

    # Test trading data validation
    order_data = {
        "symbol": "EURUSD",
        "action": "BUY",
        "volume": 0.1,
        "order_type": "MARKET",
        "stop_loss": 1.1200,
        "take_profit": 1.1300,
    }

    trading_results = validator.validate_trading_data(order_data, "order")
    print("\nTrading Data Validation:")
    for result in trading_results:
        print(f"  {result.severity.value}: {result.message}")

    # Test model data validation
    features = {"rsi": 65.5, "macd": 0.0012, "sma_20": 1.1240, "volume": 50000}

    model_results = validator.validate_model_data(features, "features")
    print("\nModel Data Validation:")
    for result in model_results:
        print(f"  {result.severity.value}: {result.message}")

    # Test validation summary
    all_results = market_results + trading_results + model_results
    summary = validator.get_validation_summary(all_results)
    print(f"\nValidation Summary:")
    print(f"  Total: {summary['total_validations']}")
    print(f"  Valid: {summary['valid_count']}")
    print(f"  Invalid: {summary['invalid_count']}")
    print(f"  Success Rate: {summary['validation_rate']:.2%}")

    print("Data validation test completed successfully!")
