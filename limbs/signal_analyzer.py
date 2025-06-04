"""
limbs/signal_analyzer.py
Trading Signal Generation System
Institutional-grade signal analysis and generation for Forex trading
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import warnings

warnings.filterwarnings("ignore")


class SignalType(Enum):
    """Signal types"""

    BUY = 1
    SELL = -1
    HOLD = 0
    STRONG_BUY = 2
    STRONG_SELL = -2


class SignalSource(Enum):
    """Signal sources"""

    TREND = "trend"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    PATTERN = "pattern"
    VOLUME = "volume"
    CUSTOM = "custom"
    ENSEMBLE = "ensemble"


@dataclass
class Signal:
    """Trading signal structure"""

    timestamp: datetime
    symbol: str
    signal_type: SignalType
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    source: SignalSource
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert signal to dictionary"""
        return {
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "signal_type": self.signal_type.value,
            "strength": self.strength,
            "confidence": self.confidence,
            "source": self.source.value,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "risk_reward_ratio": self.risk_reward_ratio,
            "metadata": self.metadata,
        }


@dataclass
class SignalConfig:
    """Configuration for signal generation"""

    # Trend signals
    trend_enabled: bool = True
    trend_ma_periods: List[int] = field(default_factory=lambda: [20, 50])
    trend_strength_threshold: float = 0.6

    # Momentum signals
    momentum_enabled: bool = True
    rsi_oversold: float = 30
    rsi_overbought: float = 70
    macd_threshold: float = 0.0001

    # Volatility signals
    volatility_enabled: bool = True
    bb_threshold: float = 0.8  # BB position threshold
    atr_multiplier: float = 2.0

    # Pattern signals
    pattern_enabled: bool = True
    pattern_min_strength: int = 100  # Minimum pattern strength

    # Volume signals
    volume_enabled: bool = True
    volume_threshold: float = 1.5  # Volume ratio threshold

    # Signal filtering
    min_confidence: float = 0.6
    min_strength: float = 0.5
    max_signals_per_hour: int = 5

    # Risk management
    default_stop_loss_pct: float = 0.02  # 2%
    default_take_profit_pct: float = 0.04  # 4%
    min_risk_reward_ratio: float = 1.5


class SignalAnalyzer:
    """
    Advanced Trading Signal Analyzer
    Generates institutional-grade trading signals from technical analysis
    """

    def __init__(self, config: SignalConfig = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or SignalConfig()
        self.signal_history = []
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.is_fitted = False

        # Signal generators
        self.signal_generators = {
            SignalSource.TREND: self._generate_trend_signals,
            SignalSource.MOMENTUM: self._generate_momentum_signals,
            SignalSource.VOLATILITY: self._generate_volatility_signals,
            SignalSource.PATTERN: self._generate_pattern_signals,
            SignalSource.VOLUME: self._generate_volume_signals,
            SignalSource.CUSTOM: self._generate_custom_signals,
        }

        # Signal weights for ensemble
        self.signal_weights = {
            SignalSource.TREND: 0.25,
            SignalSource.MOMENTUM: 0.25,
            SignalSource.VOLATILITY: 0.15,
            SignalSource.PATTERN: 0.15,
            SignalSource.VOLUME: 0.10,
            SignalSource.CUSTOM: 0.10,
        }

        self.logger.info("SignalAnalyzer initialized successfully")

    def analyze_signals(
        self, features: pd.DataFrame, symbol: str, current_price: float = None
    ) -> List[Signal]:
        """
        Analyze features and generate trading signals

        Args:
            features: DataFrame with calculated technical indicators
            symbol: Trading symbol
            current_price: Current market price

        Returns:
            List of generated signals
        """
        try:
            if len(features) < 50:
                self.logger.warning("Insufficient data for signal generation")
                return []

            # Use latest row for signal generation
            latest_data = features.iloc[-1]
            current_price = current_price or latest_data.get("close", 0)

            if current_price <= 0:
                self.logger.error("Invalid current price")
                return []

            # Generate signals from different sources
            all_signals = []

            for source, generator in self.signal_generators.items():
                try:
                    if self._is_source_enabled(source):
                        signals = generator(features, symbol, current_price)
                        all_signals.extend(signals)
                        self.logger.debug(
                            f"Generated {len(signals)} signals from {source.value}"
                        )
                except Exception as e:
                    self.logger.error(f"Error generating {source.value} signals: {e}")
                    continue

            # Generate ensemble signals
            if len(all_signals) > 1:
                ensemble_signals = self._generate_ensemble_signals(
                    all_signals, features, symbol, current_price
                )
                all_signals.extend(ensemble_signals)

            # Filter signals
            filtered_signals = self._filter_signals(all_signals, features)

            # Calculate risk management parameters
            for signal in filtered_signals:
                self._calculate_risk_parameters(signal, features)

            # Store in history
            self.signal_history.extend(filtered_signals)
            self._cleanup_history()

            self.logger.info(
                f"Generated {len(filtered_signals)} filtered signals for {symbol}"
            )
            return filtered_signals

        except Exception as e:
            self.logger.error(f"Error in analyze_signals: {e}")
            return []

    def _generate_trend_signals(
        self, features: pd.DataFrame, symbol: str, current_price: float
    ) -> List[Signal]:
        """Generate trend-based signals"""
        signals = []

        try:
            latest = features.iloc[-1]

            # MA crossover signals
            if "MA_20" in features.columns and "MA_50" in features.columns:
                ma_20_slope = features["MA_20"].diff(5).iloc[-1]
                ma_50_slope = features["MA_50"].diff(5).iloc[-1]

                # Bullish trend
                if (
                    latest["MA_20"] > latest["MA_50"]
                    and ma_20_slope > 0
                    and ma_50_slope > 0
                ):

                    strength = min(abs(ma_20_slope) * 10000, 1.0)
                    confidence = self._calculate_trend_confidence(features)

                    if strength >= self.config.trend_strength_threshold:
                        signal = Signal(
                            timestamp=datetime.now(),
                            symbol=symbol,
                            signal_type=SignalType.BUY,
                            strength=strength,
                            confidence=confidence,
                            source=SignalSource.TREND,
                            entry_price=current_price,
                            metadata={
                                "ma_20": latest["MA_20"],
                                "ma_50": latest["MA_50"],
                                "ma_20_slope": ma_20_slope,
                                "ma_50_slope": ma_50_slope,
                            },
                        )
                        signals.append(signal)

                # Bearish trend
                elif (
                    latest["MA_20"] < latest["MA_50"]
                    and ma_20_slope < 0
                    and ma_50_slope < 0
                ):

                    strength = min(abs(ma_20_slope) * 10000, 1.0)
                    confidence = self._calculate_trend_confidence(features)

                    if strength >= self.config.trend_strength_threshold:
                        signal = Signal(
                            timestamp=datetime.now(),
                            symbol=symbol,
                            signal_type=SignalType.SELL,
                            strength=strength,
                            confidence=confidence,
                            source=SignalSource.TREND,
                            entry_price=current_price,
                            metadata={
                                "ma_20": latest["MA_20"],
                                "ma_50": latest["MA_50"],
                                "ma_20_slope": ma_20_slope,
                                "ma_50_slope": ma_50_slope,
                            },
                        )
                        signals.append(signal)

            # EMA trend signals
            if "EMA_21" in features.columns and "EMA_55" in features.columns:
                ema_cross = self._detect_ema_crossover(features)
                if ema_cross != 0:
                    strength = 0.7
                    confidence = 0.8

                    signal = Signal(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        signal_type=(
                            SignalType.BUY if ema_cross > 0 else SignalType.SELL
                        ),
                        strength=strength,
                        confidence=confidence,
                        source=SignalSource.TREND,
                        entry_price=current_price,
                        metadata={
                            "ema_crossover": ema_cross,
                            "ema_21": latest.get("EMA_21"),
                            "ema_55": latest.get("EMA_55"),
                        },
                    )
                    signals.append(signal)

            return signals

        except Exception as e:
            self.logger.error(f"Error generating trend signals: {e}")
            return []

    def _generate_momentum_signals(
        self, features: pd.DataFrame, symbol: str, current_price: float
    ) -> List[Signal]:
        """Generate momentum-based signals"""
        signals = []

        try:
            latest = features.iloc[-1]

            # RSI signals
            if "RSI" in features.columns:
                rsi = latest["RSI"]
                rsi_slope = features["RSI"].diff(3).iloc[-1]

                # RSI oversold
                if rsi < self.config.rsi_oversold and rsi_slope > 0:
                    strength = (
                        self.config.rsi_oversold - rsi
                    ) / self.config.rsi_oversold
                    confidence = 0.7 + (strength * 0.3)

                    signal = Signal(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        signal_type=SignalType.BUY,
                        strength=min(strength, 1.0),
                        confidence=min(confidence, 1.0),
                        source=SignalSource.MOMENTUM,
                        entry_price=current_price,
                        metadata={
                            "rsi": rsi,
                            "rsi_slope": rsi_slope,
                            "signal_reason": "rsi_oversold_reversal",
                        },
                    )
                    signals.append(signal)

                # RSI overbought
                elif rsi > self.config.rsi_overbought and rsi_slope < 0:
                    strength = (rsi - self.config.rsi_overbought) / (
                        100 - self.config.rsi_overbought
                    )
                    confidence = 0.7 + (strength * 0.3)

                    signal = Signal(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        signal_type=SignalType.SELL,
                        strength=min(strength, 1.0),
                        confidence=min(confidence, 1.0),
                        source=SignalSource.MOMENTUM,
                        entry_price=current_price,
                        metadata={
                            "rsi": rsi,
                            "rsi_slope": rsi_slope,
                            "signal_reason": "rsi_overbought_reversal",
                        },
                    )
                    signals.append(signal)

            # MACD signals
            if all(
                col in features.columns
                for col in ["MACD", "MACD_signal", "MACD_histogram"]
            ):
                macd_cross = self._detect_macd_crossover(features)
                if abs(macd_cross) > self.config.macd_threshold:

                    macd_histogram = latest["MACD_histogram"]
                    strength = min(abs(macd_histogram) * 10000, 1.0)
                    confidence = 0.75

                    signal = Signal(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        signal_type=(
                            SignalType.BUY if macd_cross > 0 else SignalType.SELL
                        ),
                        strength=strength,
                        confidence=confidence,
                        source=SignalSource.MOMENTUM,
                        entry_price=current_price,
                        metadata={
                            "macd": latest["MACD"],
                            "macd_signal": latest["MACD_signal"],
                            "macd_histogram": macd_histogram,
                            "macd_crossover": macd_cross,
                        },
                    )
                    signals.append(signal)

            # Stochastic signals
            if "STOCH_K" in features.columns and "STOCH_D" in features.columns:
                stoch_signal = self._analyze_stochastic(features)
                if stoch_signal != 0:

                    signal = Signal(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        signal_type=(
                            SignalType.BUY if stoch_signal > 0 else SignalType.SELL
                        ),
                        strength=0.6,
                        confidence=0.7,
                        source=SignalSource.MOMENTUM,
                        entry_price=current_price,
                        metadata={
                            "stoch_k": latest["STOCH_K"],
                            "stoch_d": latest["STOCH_D"],
                            "stoch_signal": stoch_signal,
                        },
                    )
                    signals.append(signal)

            return signals

        except Exception as e:
            self.logger.error(f"Error generating momentum signals: {e}")
            return []

    def _generate_volatility_signals(
        self, features: pd.DataFrame, symbol: str, current_price: float
    ) -> List[Signal]:
        """Generate volatility-based signals"""
        signals = []

        try:
            latest = features.iloc[-1]

            # Bollinger Bands signals
            if all(
                col in features.columns
                for col in ["BB_upper", "BB_lower", "BB_position"]
            ):
                bb_position = latest["BB_position"]
                bb_width = latest.get("BB_width", 0)

                # BB oversold (near lower band)
                if bb_position < (1 - self.config.bb_threshold):
                    strength = (1 - self.config.bb_threshold - bb_position) / (
                        1 - self.config.bb_threshold
                    )
                    confidence = 0.6 + (
                        bb_width * 0.4
                    )  # Higher confidence in wider bands

                    signal = Signal(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        signal_type=SignalType.BUY,
                        strength=min(strength, 1.0),
                        confidence=min(confidence, 1.0),
                        source=SignalSource.VOLATILITY,
                        entry_price=current_price,
                        metadata={
                            "bb_position": bb_position,
                            "bb_width": bb_width,
                            "bb_upper": latest["BB_upper"],
                            "bb_lower": latest["BB_lower"],
                        },
                    )
                    signals.append(signal)

                # BB overbought (near upper band)
                elif bb_position > self.config.bb_threshold:
                    strength = (bb_position - self.config.bb_threshold) / (
                        1 - self.config.bb_threshold
                    )
                    confidence = 0.6 + (bb_width * 0.4)

                    signal = Signal(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        signal_type=SignalType.SELL,
                        strength=min(strength, 1.0),
                        confidence=min(confidence, 1.0),
                        source=SignalSource.VOLATILITY,
                        entry_price=current_price,
                        metadata={
                            "bb_position": bb_position,
                            "bb_width": bb_width,
                            "bb_upper": latest["BB_upper"],
                            "bb_lower": latest["BB_lower"],
                        },
                    )
                    signals.append(signal)

            # ATR-based breakout signals
            if "ATR" in features.columns:
                atr_signal = self._detect_atr_breakout(features, current_price)
                if atr_signal != 0:

                    signal = Signal(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        signal_type=(
                            SignalType.BUY if atr_signal > 0 else SignalType.SELL
                        ),
                        strength=0.7,
                        confidence=0.6,
                        source=SignalSource.VOLATILITY,
                        entry_price=current_price,
                        metadata={"atr": latest["ATR"], "atr_breakout": atr_signal},
                    )
                    signals.append(signal)

            # Volatility squeeze signals
            if "BB_squeeze" in features.columns:
                squeeze_signal = self._detect_volatility_squeeze(features)
                if squeeze_signal != 0:

                    signal = Signal(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        signal_type=(
                            SignalType.BUY if squeeze_signal > 0 else SignalType.SELL
                        ),
                        strength=0.8,
                        confidence=0.7,
                        source=SignalSource.VOLATILITY,
                        entry_price=current_price,
                        metadata={
                            "volatility_squeeze": squeeze_signal,
                            "bb_squeeze": latest.get("BB_squeeze", 0),
                        },
                    )
                    signals.append(signal)

            return signals

        except Exception as e:
            self.logger.error(f"Error generating volatility signals: {e}")
            return []

    def _generate_pattern_signals(
        self, features: pd.DataFrame, symbol: str, current_price: float
    ) -> List[Signal]:
        """Generate pattern-based signals"""
        signals = []

        try:
            latest = features.iloc[-1]

            # Candlestick pattern signals
            pattern_columns = [col for col in features.columns if col.startswith("CDL")]

            if pattern_columns:
                pattern_strength = latest.get("pattern_strength", 0)
                bullish_patterns = latest.get("bullish_patterns", 0)
                bearish_patterns = latest.get("bearish_patterns", 0)

                # Strong bullish patterns
                if (
                    bullish_patterns > 0
                    and abs(pattern_strength) >= self.config.pattern_min_strength
                ):

                    strength = min(
                        bullish_patterns / 3, 1.0
                    )  # Normalize by max expected patterns
                    confidence = 0.6 + (strength * 0.3)

                    signal = Signal(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        signal_type=SignalType.BUY,
                        strength=strength,
                        confidence=confidence,
                        source=SignalSource.PATTERN,
                        entry_price=current_price,
                        metadata={
                            "pattern_strength": pattern_strength,
                            "bullish_patterns": bullish_patterns,
                            "active_patterns": [
                                col for col in pattern_columns if latest[col] > 0
                            ],
                        },
                    )
                    signals.append(signal)

                # Strong bearish patterns
                elif (
                    bearish_patterns < 0
                    and abs(pattern_strength) >= self.config.pattern_min_strength
                ):

                    strength = min(abs(bearish_patterns) / 3, 1.0)
                    confidence = 0.6 + (strength * 0.3)

                    signal = Signal(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        signal_type=SignalType.SELL,
                        strength=strength,
                        confidence=confidence,
                        source=SignalSource.PATTERN,
                        entry_price=current_price,
                        metadata={
                            "pattern_strength": pattern_strength,
                            "bearish_patterns": bearish_patterns,
                            "active_patterns": [
                                col for col in pattern_columns if latest[col] < 0
                            ],
                        },
                    )
                    signals.append(signal)

            # Price action patterns
            if "inside_bar" in features.columns and latest["inside_bar"]:
                # Inside bar breakout potential
                strength = 0.6
                confidence = 0.5

                # Determine direction based on trend
                signal_type = self._determine_breakout_direction(features)

                if signal_type != SignalType.HOLD:
                    signal = Signal(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        signal_type=signal_type,
                        strength=strength,
                        confidence=confidence,
                        source=SignalSource.PATTERN,
                        entry_price=current_price,
                        metadata={
                            "pattern_type": "inside_bar_breakout",
                            "inside_bar": True,
                        },
                    )
                    signals.append(signal)

            return signals

        except Exception as e:
            self.logger.error(f"Error generating pattern signals: {e}")
            return []

    def _generate_volume_signals(
        self, features: pd.DataFrame, symbol: str, current_price: float
    ) -> List[Signal]:
        """Generate volume-based signals"""
        signals = []

        try:
            latest = features.iloc[-1]

            # Volume confirmation signals
            if "volume_ratio" in features.columns:
                volume_ratio = latest["volume_ratio"]

                # High volume with price movement
                if volume_ratio > self.config.volume_threshold:
                    # Determine direction from price action
                    price_change = features["close"].pct_change().iloc[-1]

                    if abs(price_change) > 0.001:  # Significant price movement
                        strength = min(volume_ratio / 3, 1.0)
                        confidence = 0.7

                        signal_type = (
                            SignalType.BUY if price_change > 0 else SignalType.SELL
                        )

                        signal = Signal(
                            timestamp=datetime.now(),
                            symbol=symbol,
                            signal_type=signal_type,
                            strength=strength,
                            confidence=confidence,
                            source=SignalSource.VOLUME,
                            entry_price=current_price,
                            metadata={
                                "volume_ratio": volume_ratio,
                                "price_change": price_change,
                                "volume_confirmation": True,
                            },
                        )
                        signals.append(signal)

            # OBV divergence signals
            if "OBV" in features.columns:
                obv_signal = self._detect_obv_divergence(features)
                if obv_signal != 0:

                    signal = Signal(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        signal_type=(
                            SignalType.BUY if obv_signal > 0 else SignalType.SELL
                        ),
                        strength=0.6,
                        confidence=0.65,
                        source=SignalSource.VOLUME,
                        entry_price=current_price,
                        metadata={"obv_divergence": obv_signal, "obv": latest["OBV"]},
                    )
                    signals.append(signal)

            return signals

        except Exception as e:
            self.logger.error(f"Error generating volume signals: {e}")
            return []

    def _generate_custom_signals(
        self, features: pd.DataFrame, symbol: str, current_price: float
    ) -> List[Signal]:
        """Generate custom signals"""
        signals = []

        try:
            latest = features.iloc[-1]

            # Support/Resistance breakout
            if all(col in features.columns for col in ["support", "resistance"]):
                sr_signal = self._detect_sr_breakout(features, current_price)
                if sr_signal != 0:

                    signal = Signal(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        signal_type=(
                            SignalType.BUY if sr_signal > 0 else SignalType.SELL
                        ),
                        strength=0.8,
                        confidence=0.7,
                        source=SignalSource.CUSTOM,
                        entry_price=current_price,
                        metadata={
                            "sr_breakout": sr_signal,
                            "support": latest["support"],
                            "resistance": latest["resistance"],
                        },
                    )
                    signals.append(signal)

            # Market structure signals
            if "market_structure" in features.columns:
                structure_signal = self._analyze_market_structure(features)
                if structure_signal != 0:

                    signal = Signal(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        signal_type=(
                            SignalType.BUY if structure_signal > 0 else SignalType.SELL
                        ),
                        strength=0.7,
                        confidence=0.6,
                        source=SignalSource.CUSTOM,
                        entry_price=current_price,
                        metadata={
                            "market_structure": latest["market_structure"],
                            "structure_signal": structure_signal,
                        },
                    )
                    signals.append(signal)

            # Multi-timeframe confirmation
            mtf_signal = self._multi_timeframe_confirmation(features)
            if mtf_signal != 0:

                signal = Signal(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    signal_type=SignalType.BUY if mtf_signal > 0 else SignalType.SELL,
                    strength=0.9,
                    confidence=0.8,
                    source=SignalSource.CUSTOM,
                    entry_price=current_price,
                    metadata={
                        "mtf_confirmation": mtf_signal,
                        "signal_type": "multi_timeframe",
                    },
                )
                signals.append(signal)

            return signals

        except Exception as e:
            self.logger.error(f"Error generating custom signals: {e}")
            return []

    def _generate_ensemble_signals(
        self,
        individual_signals: List[Signal],
        features: pd.DataFrame,
        symbol: str,
        current_price: float,
    ) -> List[Signal]:
        """Generate ensemble signals by combining individual signals"""
        ensemble_signals = []

        try:
            if len(individual_signals) < 2:
                return ensemble_signals

            # Group signals by type
            buy_signals = [
                s for s in individual_signals if s.signal_type == SignalType.BUY
            ]
            sell_signals = [
                s for s in individual_signals if s.signal_type == SignalType.SELL
            ]

            # Calculate ensemble buy signal
            if len(buy_signals) >= 2:
                weighted_strength = sum(
                    signal.strength * self.signal_weights.get(signal.source, 0.1)
                    for signal in buy_signals
                )
                weighted_confidence = sum(
                    signal.confidence * self.signal_weights.get(signal.source, 0.1)
                    for signal in buy_signals
                )

                # Normalize by total weight
                total_weight = sum(
                    self.signal_weights.get(s.source, 0.1) for s in buy_signals
                )
                if total_weight > 0:
                    weighted_strength /= total_weight
                    weighted_confidence /= total_weight

                    # Boost confidence for multiple confirmations
                    confidence_boost = min(len(buy_signals) * 0.1, 0.3)
                    weighted_confidence = min(
                        weighted_confidence + confidence_boost, 1.0
                    )

                    ensemble_signal = Signal(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        signal_type=(
                            SignalType.STRONG_BUY
                            if len(buy_signals) >= 3
                            else SignalType.BUY
                        ),
                        strength=min(weighted_strength, 1.0),
                        confidence=weighted_confidence,
                        source=SignalSource.ENSEMBLE,
                        entry_price=current_price,
                        metadata={
                            "component_signals": len(buy_signals),
                            "sources": [s.source.value for s in buy_signals],
                            "ensemble_type": "buy_consensus",
                        },
                    )
                    ensemble_signals.append(ensemble_signal)

            # Calculate ensemble sell signal
            if len(sell_signals) >= 2:
                weighted_strength = sum(
                    signal.strength * self.signal_weights.get(signal.source, 0.1)
                    for signal in sell_signals
                )
                weighted_confidence = sum(
                    signal.confidence * self.signal_weights.get(signal.source, 0.1)
                    for signal in sell_signals
                )

                total_weight = sum(
                    self.signal_weights.get(s.source, 0.1) for s in sell_signals
                )
                if total_weight > 0:
                    weighted_strength /= total_weight
                    weighted_confidence /= total_weight

                    confidence_boost = min(len(sell_signals) * 0.1, 0.3)
                    weighted_confidence = min(
                        weighted_confidence + confidence_boost, 1.0
                    )

                    ensemble_signal = Signal(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        signal_type=(
                            SignalType.STRONG_SELL
                            if len(sell_signals) >= 3
                            else SignalType.SELL
                        ),
                        strength=min(weighted_strength, 1.0),
                        confidence=weighted_confidence,
                        source=SignalSource.ENSEMBLE,
                        entry_price=current_price,
                        metadata={
                            "component_signals": len(sell_signals),
                            "sources": [s.source.value for s in sell_signals],
                            "ensemble_type": "sell_consensus",
                        },
                    )
                    ensemble_signals.append(ensemble_signal)

            return ensemble_signals

        except Exception as e:
            self.logger.error(f"Error generating ensemble signals: {e}")
            return []

    def _filter_signals(
        self, signals: List[Signal], features: pd.DataFrame
    ) -> List[Signal]:
        """Filter signals based on quality criteria"""
        filtered_signals = []

        try:
            for signal in signals:
                # Basic quality filters
                if (
                    signal.confidence < self.config.min_confidence
                    or signal.strength < self.config.min_strength
                ):
                    continue

                # Check for recent similar signals (avoid spam)
                if self._has_recent_similar_signal(signal):
                    continue

                # Market condition filters
                if not self._is_market_condition_suitable(signal, features):
                    continue

                # Anomaly detection
                if self._is_signal_anomalous(signal, features):
                    continue

                filtered_signals.append(signal)

            # Limit signals per hour
            filtered_signals = self._limit_signals_per_hour(filtered_signals)

            return filtered_signals

        except Exception as e:
            self.logger.error(f"Error filtering signals: {e}")
            return signals  # Return original if filtering fails

    def _calculate_risk_parameters(self, signal: Signal, features: pd.DataFrame):
        """Calculate stop loss and take profit levels"""
        try:
            latest = features.iloc[-1]
            atr = latest.get("ATR", 0.001)

            # Calculate stop loss
            if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                # For buy signals, stop loss below entry
                stop_distance = max(
                    atr * self.config.atr_multiplier,
                    signal.entry_price * self.config.default_stop_loss_pct,
                )
                signal.stop_loss = signal.entry_price - stop_distance
                signal.take_profit = signal.entry_price + (
                    stop_distance * self.config.min_risk_reward_ratio
                )

            elif signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
                # For sell signals, stop loss above entry
                stop_distance = max(
                    atr * self.config.atr_multiplier,
                    signal.entry_price * self.config.default_stop_loss_pct,
                )
                signal.stop_loss = signal.entry_price + stop_distance
                signal.take_profit = signal.entry_price - (
                    stop_distance * self.config.min_risk_reward_ratio
                )

            # Calculate risk-reward ratio
            if signal.stop_loss and signal.take_profit:
                risk = abs(signal.entry_price - signal.stop_loss)
                reward = abs(signal.take_profit - signal.entry_price)
                signal.risk_reward_ratio = reward / risk if risk > 0 else 0

        except Exception as e:
            self.logger.error(f"Error calculating risk parameters: {e}")

    # Helper methods for signal detection
    def _calculate_trend_confidence(self, features: pd.DataFrame) -> float:
        """Calculate trend confidence score"""
        try:
            # Use multiple MA alignments
            confidence_score = 0.5

            if "MA_20" in features.columns and "MA_50" in features.columns:
                latest = features.iloc[-1]
                if latest["MA_20"] > latest["MA_50"]:
                    confidence_score += 0.2
                else:
                    confidence_score -= 0.2

            # Add ADX if available
            if "trend_strength" in features.columns:
                trend_strength = features["trend_strength"].iloc[-1]
                confidence_score += trend_strength * 0.3

            return max(0.0, min(1.0, confidence_score))

        except Exception as e:
            self.logger.error(f"Error calculating trend confidence: {e}")
            return 0.5

    def _detect_ema_crossover(self, features: pd.DataFrame) -> int:
        """Detect EMA crossover"""
        try:
            if "EMA_21" not in features.columns or "EMA_55" not in features.columns:
                return 0

            current = features[["EMA_21", "EMA_55"]].iloc[-1]
            previous = features[["EMA_21", "EMA_55"]].iloc[-2]

            # Bullish crossover
            if (
                current["EMA_21"] > current["EMA_55"]
                and previous["EMA_21"] <= previous["EMA_55"]
            ):
                return 1
            # Bearish crossover
            elif (
                current["EMA_21"] < current["EMA_55"]
                and previous["EMA_21"] >= previous["EMA_55"]
            ):
                return -1

            return 0

        except Exception as e:
            self.logger.error(f"Error detecting EMA crossover: {e}")
            return 0

    def _detect_macd_crossover(self, features: pd.DataFrame) -> float:
        """Detect MACD signal line crossover"""
        try:
            current = features[["MACD", "MACD_signal"]].iloc[-1]
            previous = features[["MACD", "MACD_signal"]].iloc[-2]

            current_diff = current["MACD"] - current["MACD_signal"]
            previous_diff = previous["MACD"] - previous["MACD_signal"]

            # Crossover occurred
            if current_diff * previous_diff < 0:
                return current_diff

            return 0

        except Exception as e:
            self.logger.error(f"Error detecting MACD crossover: {e}")
            return 0

    def _analyze_stochastic(self, features: pd.DataFrame) -> int:
        """Analyze stochastic for signals"""
        try:
            latest = features[["STOCH_K", "STOCH_D"]].iloc[-1]

            # Oversold reversal
            if latest["STOCH_K"] < 20 and latest["STOCH_K"] > latest["STOCH_D"]:
                return 1
            # Overbought reversal
            elif latest["STOCH_K"] > 80 and latest["STOCH_K"] < latest["STOCH_D"]:
                return -1

            return 0

        except Exception as e:
            self.logger.error(f"Error analyzing stochastic: {e}")
            return 0

    def _detect_atr_breakout(self, features: pd.DataFrame, current_price: float) -> int:
        """Detect ATR-based breakout"""
        try:
            if len(features) < 5:
                return 0

            recent_high = features["high"].iloc[-5:].max()
            recent_low = features["low"].iloc[-5:].min()
            atr = features["ATR"].iloc[-1]

            # Upward breakout
            if current_price > recent_high + (atr * 0.5):
                return 1
            # Downward breakout
            elif current_price < recent_low - (atr * 0.5):
                return -1

            return 0

        except Exception as e:
            self.logger.error(f"Error detecting ATR breakout: {e}")
            return 0

    def _detect_volatility_squeeze(self, features: pd.DataFrame) -> int:
        """Detect volatility squeeze breakout"""
        try:
            if "BB_squeeze" not in features.columns:
                return 0

            # Check if squeeze is ending
            recent_squeeze = features["BB_squeeze"].iloc[-5:].sum()
            current_squeeze = features["BB_squeeze"].iloc[-1]

            if recent_squeeze >= 3 and not current_squeeze:
                # Determine breakout direction
                if "MACD_histogram" in features.columns:
                    macd_hist = features["MACD_histogram"].iloc[-1]
                    return 1 if macd_hist > 0 else -1

            return 0

        except Exception as e:
            self.logger.error(f"Error detecting volatility squeeze: {e}")
            return 0

    def _determine_breakout_direction(self, features: pd.DataFrame) -> SignalType:
        """Determine breakout direction for inside bars"""
        try:
            # Use trend indicators to determine direction
            if "MA_20" in features.columns and "MA_50" in features.columns:
                latest = features.iloc[-1]
                if latest["MA_20"] > latest["MA_50"]:
                    return SignalType.BUY
                elif latest["MA_20"] < latest["MA_50"]:
                    return SignalType.SELL

            return SignalType.HOLD

        except Exception as e:
            self.logger.error(f"Error determining breakout direction: {e}")
            return SignalType.HOLD

    def _detect_obv_divergence(self, features: pd.DataFrame) -> int:
        """Detect OBV divergence"""
        try:
            if len(features) < 20:
                return 0

            # Simple divergence detection
            price_trend = features["close"].iloc[-10:].diff().sum()
            obv_trend = features["OBV"].iloc[-10:].diff().sum()

            # Bullish divergence (price down, OBV up)
            if price_trend < 0 and obv_trend > 0:
                return 1
            # Bearish divergence (price up, OBV down)
            elif price_trend > 0 and obv_trend < 0:
                return -1

            return 0

        except Exception as e:
            self.logger.error(f"Error detecting OBV divergence: {e}")
            return 0

    def _detect_sr_breakout(self, features: pd.DataFrame, current_price: float) -> int:
        """Detect support/resistance breakout"""
        try:
            latest = features.iloc[-1]
            resistance = latest["resistance"]
            support = latest["support"]

            # Resistance breakout
            if current_price > resistance * 1.001:  # 0.1% buffer
                return 1
            # Support breakdown
            elif current_price < support * 0.999:  # 0.1% buffer
                return -1

            return 0

        except Exception as e:
            self.logger.error(f"Error detecting S/R breakout: {e}")
            return 0

    def _analyze_market_structure(self, features: pd.DataFrame) -> int:
        """Analyze market structure for signals"""
        try:
            if "market_structure" not in features.columns:
                return 0

            # Trend following in trending markets
            structure = features["market_structure"].iloc[-1]

            if structure == 1:  # Trending market
                # Follow the trend direction
                if "MA_20" in features.columns and "MA_50" in features.columns:
                    latest = features.iloc[-1]
                    if latest["MA_20"] > latest["MA_50"]:
                        return 1
                    else:
                        return -1

            return 0

        except Exception as e:
            self.logger.error(f"Error analyzing market structure: {e}")
            return 0

    def _multi_timeframe_confirmation(self, features: pd.DataFrame) -> int:
        """Multi-timeframe confirmation (simplified)"""
        try:
            # Use multiple MA periods as proxy for different timeframes
            ma_signals = []

            for period in [20, 50, 100]:
                if f"MA_{period}" in features.columns:
                    ma_slope = features[f"MA_{period}"].diff(5).iloc[-1]
                    ma_signals.append(1 if ma_slope > 0 else -1)

            if len(ma_signals) >= 2:
                # Require majority agreement
                buy_count = sum(1 for s in ma_signals if s > 0)
                sell_count = sum(1 for s in ma_signals if s < 0)

                if buy_count >= len(ma_signals) * 0.7:
                    return 1
                elif sell_count >= len(ma_signals) * 0.7:
                    return -1

            return 0

        except Exception as e:
            self.logger.error(f"Error in multi-timeframe confirmation: {e}")
            return 0

    def _is_source_enabled(self, source: SignalSource) -> bool:
        """Check if signal source is enabled"""
        source_config_map = {
            SignalSource.TREND: self.config.trend_enabled,
            SignalSource.MOMENTUM: self.config.momentum_enabled,
            SignalSource.VOLATILITY: self.config.volatility_enabled,
            SignalSource.PATTERN: self.config.pattern_enabled,
            SignalSource.VOLUME: self.config.volume_enabled,
            SignalSource.CUSTOM: True,  # Always enabled
        }
        return source_config_map.get(source, True)

    def _has_recent_similar_signal(self, signal: Signal) -> bool:
        """Check if there's a recent similar signal"""
        try:
            cutoff_time = datetime.now() - timedelta(minutes=15)

            for hist_signal in self.signal_history:
                if (
                    hist_signal.timestamp > cutoff_time
                    and hist_signal.symbol == signal.symbol
                    and hist_signal.signal_type == signal.signal_type
                    and hist_signal.source == signal.source
                ):
                    return True

            return False

        except Exception as e:
            self.logger.error(f"Error checking recent signals: {e}")
            return False

    def _is_market_condition_suitable(
        self, signal: Signal, features: pd.DataFrame
    ) -> bool:
        """Check if market conditions are suitable for the signal"""
        try:
            # Avoid trading in extremely low volatility
            if "ATR_ratio" in features.columns:
                atr_ratio = features["ATR_ratio"].iloc[-1]
                if atr_ratio < 0.0001:  # Very low volatility
                    return False

            # Avoid trading near major S/R levels for reversal signals
            if signal.source == SignalSource.MOMENTUM:
                if (
                    "dist_to_resistance" in features.columns
                    and features["dist_to_resistance"].iloc[-1] < 0.001
                ):
                    return False
                if (
                    "dist_to_support" in features.columns
                    and features["dist_to_support"].iloc[-1] < 0.001
                ):
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Error checking market conditions: {e}")
            return True

    def _is_signal_anomalous(self, signal: Signal, features: pd.DataFrame) -> bool:
        """Check if signal is anomalous using isolation forest"""
        try:
            if not self.is_fitted and len(self.signal_history) >= 50:
                # Fit anomaly detector
                signal_features = self._extract_signal_features(
                    self.signal_history[-50:]
                )
                if len(signal_features) > 0:
                    self.anomaly_detector.fit(signal_features)
                    self.is_fitted = True

            if self.is_fitted:
                current_features = self._extract_signal_features([signal])
                if len(current_features) > 0:
                    anomaly_score = self.anomaly_detector.decision_function(
                        current_features
                    )[0]
                    return anomaly_score < -0.1  # Threshold for anomaly

            return False

        except Exception as e:
            self.logger.error(f"Error detecting signal anomaly: {e}")
            return False

    def _extract_signal_features(self, signals: List[Signal]) -> np.ndarray:
        """Extract features from signals for anomaly detection"""
        try:
            features = []
            for signal in signals:
                feature_vector = [
                    signal.strength,
                    signal.confidence,
                    signal.signal_type.value,
                    len(signal.metadata),
                ]
                features.append(feature_vector)

            return np.array(features)

        except Exception as e:
            self.logger.error(f"Error extracting signal features: {e}")
            return np.array([])

    def _limit_signals_per_hour(self, signals: List[Signal]) -> List[Signal]:
        """Limit number of signals per hour"""
        try:
            # Sort by confidence and strength
            sorted_signals = sorted(
                signals, key=lambda s: (s.confidence * s.strength), reverse=True
            )

            return sorted_signals[: self.config.max_signals_per_hour]

        except Exception as e:
            self.logger.error(f"Error limiting signals: {e}")
            return signals

    def _cleanup_history(self):
        """Clean up old signal history"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.signal_history = [
                s for s in self.signal_history if s.timestamp > cutoff_time
            ]
        except Exception as e:
            self.logger.error(f"Error cleaning up history: {e}")

    def get_signal_summary(self) -> Dict[str, any]:
        """Get summary of recent signals"""
        try:
            recent_signals = [
                s
                for s in self.signal_history
                if s.timestamp > datetime.now() - timedelta(hours=1)
            ]

            summary = {
                "total_signals_last_hour": len(recent_signals),
                "buy_signals": len(
                    [s for s in recent_signals if s.signal_type.value > 0]
                ),
                "sell_signals": len(
                    [s for s in recent_signals if s.signal_type.value < 0]
                ),
                "avg_confidence": (
                    np.mean([s.confidence for s in recent_signals])
                    if recent_signals
                    else 0
                ),
                "avg_strength": (
                    np.mean([s.strength for s in recent_signals])
                    if recent_signals
                    else 0
                ),
                "sources": list(set([s.source.value for s in recent_signals])),
                "last_signal_time": (
                    max([s.timestamp for s in recent_signals])
                    if recent_signals
                    else None
                ),
            }

            return summary

        except Exception as e:
            self.logger.error(f"Error generating signal summary: {e}")
            return {}


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create sample features data (normally from FeatureEngineer)
    dates = pd.date_range(start="2023-01-01", periods=100, freq="1H")
    np.random.seed(42)

    # Sample features
    sample_data = {
        "close": 100 + np.cumsum(np.random.randn(100) * 0.01),
        "MA_20": 100 + np.cumsum(np.random.randn(100) * 0.005),
        "MA_50": 100 + np.cumsum(np.random.randn(100) * 0.003),
        "RSI": 30 + np.random.randn(100) * 20,
        "MACD": np.random.randn(100) * 0.001,
        "MACD_signal": np.random.randn(100) * 0.001,
        "MACD_histogram": np.random.randn(100) * 0.0005,
        "BB_position": np.random.rand(100),
        "BB_width": 0.02 + np.random.rand(100) * 0.01,
        "ATR": 0.001 + np.random.rand(100) * 0.0005,
        "volume_ratio": 0.5 + np.random.rand(100),
        "pattern_strength": np.random.randint(-200, 200, 100),
        "support": 99 + np.random.randn(100) * 0.5,
        "resistance": 101 + np.random.randn(100) * 0.5,
        "market_structure": np.random.choice([0, 1], 100),
    }

    features_df = pd.DataFrame(sample_data, index=dates)

    # Initialize signal analyzer
    config = SignalConfig()
    analyzer = SignalAnalyzer(config)

    # Generate signals
    print("Analyzing signals...")
    signals = analyzer.analyze_signals(features_df, "EURUSD", 1.1234)

    # Display results
    print(f"\nSignal analysis completed!")
    print(f"Generated {len(signals)} signals")

    for signal in signals:
        print(f"\n{signal.signal_type.name} signal:")
        print(f"  Source: {signal.source.value}")
        print(f"  Strength: {signal.strength:.2f}")
        print(f"  Confidence: {signal.confidence:.2f}")
        print(f"  Entry: {signal.entry_price:.5f}")
        print(
            f"  Stop Loss: {signal.stop_loss:.5f}"
            if signal.stop_loss
            else "  Stop Loss: None"
        )
        print(
            f"  Take Profit: {signal.take_profit:.5f}"
            if signal.take_profit
            else "  Take Profit: None"
        )
        print(
            f"  R/R Ratio: {signal.risk_reward_ratio:.2f}"
            if signal.risk_reward_ratio
            else "  R/R Ratio: None"
        )

    # Signal summary
    summary = analyzer.get_signal_summary()
    print(f"\nSignal Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
