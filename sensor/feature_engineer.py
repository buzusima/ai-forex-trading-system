"""
sensor/feature_engineer.py
Technical Indicators Calculation Engine
Institutional-grade feature engineering for Forex trading
"""

import numpy as np
import pandas as pd
import talib
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

warnings.filterwarnings("ignore")


@dataclass
class IndicatorConfig:
    """Configuration for technical indicators"""

    # Moving Averages
    ma_periods: List[int] = None
    ema_periods: List[int] = None

    # Momentum Indicators
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    stoch_k: int = 14
    stoch_d: int = 3
    williams_r_period: int = 14

    # Volatility Indicators
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14

    # Volume Indicators
    obv_enabled: bool = True
    ad_enabled: bool = True

    # Custom Indicators
    fractal_enabled: bool = True
    support_resistance_enabled: bool = True

    def __post_init__(self):
        if self.ma_periods is None:
            self.ma_periods = [5, 10, 20, 50, 100, 200]
        if self.ema_periods is None:
            self.ema_periods = [8, 13, 21, 34, 55, 89]


class FeatureEngineer:
    """
    Advanced Feature Engineering for Forex Trading
    Calculates institutional-grade technical indicators
    """

    def __init__(self, config: IndicatorConfig = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or IndicatorConfig()
        self.cache = {}
        self.last_update = None

        # Initialize calculation methods
        self.indicator_methods = {
            "basic": self._calculate_basic_indicators,
            "momentum": self._calculate_momentum_indicators,
            "volatility": self._calculate_volatility_indicators,
            "volume": self._calculate_volume_indicators,
            "pattern": self._calculate_pattern_indicators,
            "custom": self._calculate_custom_indicators,
            "statistical": self._calculate_statistical_indicators,
        }

        self.logger.info("FeatureEngineer initialized successfully")

    def calculate_all_features(
        self, ohlcv_data: pd.DataFrame, symbol: str = None, use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Calculate all technical indicators and features

        Args:
            ohlcv_data: DataFrame with OHLCV data
            symbol: Trading symbol
            use_cache: Whether to use cached calculations

        Returns:
            DataFrame with all calculated features
        """
        try:
            # Validate input data
            if not self._validate_data(ohlcv_data):
                raise ValueError("Invalid OHLCV data provided")

            # Check cache
            cache_key = (
                f"{symbol}_{len(ohlcv_data)}_{hash(str(ohlcv_data.iloc[-1].values))}"
            )
            if use_cache and cache_key in self.cache:
                self.logger.debug(f"Using cached features for {symbol}")
                return self.cache[cache_key]

            # Copy data to avoid modifying original
            df = ohlcv_data.copy()

            # Calculate features in parallel
            feature_groups = []
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {}

                for group_name, method in self.indicator_methods.items():
                    future = executor.submit(method, df.copy())
                    futures[future] = group_name

                for future in as_completed(futures):
                    group_name = futures[future]
                    try:
                        features = future.result()
                        feature_groups.append(features)
                        self.logger.debug(f"Calculated {group_name} indicators")
                    except Exception as e:
                        self.logger.error(f"Error calculating {group_name}: {e}")
                        continue

            # Combine all features
            result_df = df.copy()
            for features in feature_groups:
                result_df = pd.concat([result_df, features], axis=1)

            # Remove duplicates and NaN
            result_df = result_df.loc[:, ~result_df.columns.duplicated()]
            result_df = result_df.fillna(method="ffill").fillna(0)

            # Cache result
            if use_cache:
                self.cache[cache_key] = result_df.copy()
                self._clean_cache()

            self.last_update = datetime.now()
            self.logger.info(
                f"Calculated {len(result_df.columns)} features for {symbol}"
            )

            return result_df

        except Exception as e:
            self.logger.error(f"Error in calculate_all_features: {e}")
            raise

    def _calculate_basic_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic indicators (MA, EMA)"""
        features = pd.DataFrame(index=df.index)

        try:
            # Simple Moving Averages
            for period in self.config.ma_periods:
                features[f"MA_{period}"] = talib.SMA(df["close"], timeperiod=period)
                features[f"MA_{period}_slope"] = features[f"MA_{period}"].diff(5)
                features[f"price_to_MA_{period}"] = (
                    df["close"] / features[f"MA_{period}"] - 1
                )

            # Exponential Moving Averages
            for period in self.config.ema_periods:
                features[f"EMA_{period}"] = talib.EMA(df["close"], timeperiod=period)
                features[f"EMA_{period}_slope"] = features[f"EMA_{period}"].diff(5)
                features[f"price_to_EMA_{period}"] = (
                    df["close"] / features[f"EMA_{period}"] - 1
                )

            # MA Crossovers
            features["MA_5_20_cross"] = np.where(
                features["MA_5"] > features["MA_20"], 1, -1
            )
            features["EMA_8_21_cross"] = np.where(
                features["EMA_8"] > features["EMA_21"], 1, -1
            )

            # Price position relative to MAs
            features["above_MA_count"] = sum(
                [
                    (df["close"] > features[f"MA_{p}"]).astype(int)
                    for p in self.config.ma_periods
                ]
            )

            return features

        except Exception as e:
            self.logger.error(f"Error calculating basic indicators: {e}")
            return pd.DataFrame(index=df.index)

    def _calculate_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum indicators"""
        features = pd.DataFrame(index=df.index)

        try:
            # RSI
            features["RSI"] = talib.RSI(df["close"], timeperiod=self.config.rsi_period)
            features["RSI_oversold"] = (features["RSI"] < 30).astype(int)
            features["RSI_overbought"] = (features["RSI"] > 70).astype(int)
            features["RSI_slope"] = features["RSI"].diff(3)

            # MACD
            macd, macd_signal, macd_hist = talib.MACD(
                df["close"],
                fastperiod=self.config.macd_fast,
                slowperiod=self.config.macd_slow,
                signalperiod=self.config.macd_signal,
            )
            features["MACD"] = macd
            features["MACD_signal"] = macd_signal
            features["MACD_histogram"] = macd_hist
            features["MACD_cross"] = np.where(macd > macd_signal, 1, -1)
            features["MACD_divergence"] = macd - macd_signal

            # Stochastic
            stoch_k, stoch_d = talib.STOCH(
                df["high"],
                df["low"],
                df["close"],
                fastk_period=self.config.stoch_k,
                slowk_period=self.config.stoch_d,
            )
            features["STOCH_K"] = stoch_k
            features["STOCH_D"] = stoch_d
            features["STOCH_cross"] = np.where(stoch_k > stoch_d, 1, -1)

            # Williams %R
            features["WILLIAMS_R"] = talib.WILLR(
                df["high"],
                df["low"],
                df["close"],
                timeperiod=self.config.williams_r_period,
            )

            # Rate of Change
            features["ROC_5"] = talib.ROC(df["close"], timeperiod=5)
            features["ROC_10"] = talib.ROC(df["close"], timeperiod=10)

            # Momentum
            features["MOM_10"] = talib.MOM(df["close"], timeperiod=10)

            # Commodity Channel Index
            features["CCI"] = talib.CCI(df["high"], df["low"], df["close"])

            return features

        except Exception as e:
            self.logger.error(f"Error calculating momentum indicators: {e}")
            return pd.DataFrame(index=df.index)

    def _calculate_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility indicators"""
        features = pd.DataFrame(index=df.index)

        try:
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(
                df["close"],
                timeperiod=self.config.bb_period,
                nbdevup=self.config.bb_std,
                nbdevdn=self.config.bb_std,
            )
            features["BB_upper"] = bb_upper
            features["BB_middle"] = bb_middle
            features["BB_lower"] = bb_lower
            features["BB_width"] = (bb_upper - bb_lower) / bb_middle
            features["BB_position"] = (df["close"] - bb_lower) / (bb_upper - bb_lower)
            features["BB_squeeze"] = (
                features["BB_width"] < features["BB_width"].rolling(20).mean() * 0.8
            ).astype(int)

            # Average True Range
            features["ATR"] = talib.ATR(
                df["high"], df["low"], df["close"], timeperiod=self.config.atr_period
            )
            features["ATR_ratio"] = features["ATR"] / df["close"]

            # True Range
            features["TR"] = talib.TRANGE(df["high"], df["low"], df["close"])

            # Volatility measures
            features["price_volatility"] = df["close"].rolling(20).std()
            features["high_low_ratio"] = df["high"] / df["low"] - 1
            features["close_to_high"] = df["close"] / df["high"]
            features["close_to_low"] = df["close"] / df["low"]

            # Keltner Channels
            kc_middle = talib.EMA(df["close"], timeperiod=20)
            kc_upper = kc_middle + 2 * features["ATR"]
            kc_lower = kc_middle - 2 * features["ATR"]
            features["KC_upper"] = kc_upper
            features["KC_lower"] = kc_lower
            features["KC_position"] = (df["close"] - kc_lower) / (kc_upper - kc_lower)

            return features

        except Exception as e:
            self.logger.error(f"Error calculating volatility indicators: {e}")
            return pd.DataFrame(index=df.index)

    def _calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume indicators"""
        features = pd.DataFrame(index=df.index)

        try:
            if "volume" not in df.columns:
                # For Forex, create synthetic volume based on price movement
                df["volume"] = abs(df["close"] - df["open"]) * 1000000

            # On Balance Volume
            if self.config.obv_enabled:
                features["OBV"] = talib.OBV(df["close"], df["volume"])
                features["OBV_MA"] = talib.SMA(features["OBV"], timeperiod=10)

            # Accumulation/Distribution Line
            if self.config.ad_enabled:
                features["AD"] = talib.AD(
                    df["high"], df["low"], df["close"], df["volume"]
                )

            # Volume Moving Average
            features["volume_MA"] = talib.SMA(df["volume"], timeperiod=20)
            features["volume_ratio"] = df["volume"] / features["volume_MA"]

            # Money Flow Index
            features["MFI"] = talib.MFI(
                df["high"], df["low"], df["close"], df["volume"]
            )

            # Volume Rate of Change
            features["volume_ROC"] = talib.ROC(df["volume"], timeperiod=10)

            return features

        except Exception as e:
            self.logger.error(f"Error calculating volume indicators: {e}")
            return pd.DataFrame(index=df.index)

    def _calculate_pattern_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate pattern recognition indicators"""
        features = pd.DataFrame(index=df.index)

        try:
            # Candlestick patterns
            patterns = [
                "CDL2CROWS",
                "CDL3BLACKCROWS",
                "CDL3INSIDE",
                "CDL3LINESTRIKE",
                "CDL3OUTSIDE",
                "CDL3STARSINSOUTH",
                "CDL3WHITESOLDIERS",
                "CDLABANDONEDBABY",
                "CDLADVANCEBLOCK",
                "CDLBELTHOLD",
                "CDLBREAKAWAY",
                "CDLCLOSINGMARUBOZU",
                "CDLCONCEALBABYSWALL",
                "CDLCOUNTERATTACK",
                "CDLDARKCLOUDCOVER",
                "CDLDOJI",
                "CDLDOJISTAR",
                "CDLDRAGONFLYDOJI",
                "CDLENGULFING",
                "CDLEVENINGDOJISTAR",
                "CDLEVENINGSTAR",
                "CDLGAPSIDESIDEWHITE",
                "CDLGRAVESTONEDOJI",
                "CDLHAMMER",
                "CDLHANGINGMAN",
                "CDLHARAMI",
                "CDLHARAMICROSS",
                "CDLHIGHWAVE",
                "CDLHIKKAKE",
                "CDLHIKKAKEMOD",
                "CDLHOMINGPIGEON",
                "CDLIDENTICAL3CROWS",
                "CDLINNECK",
                "CDLINVERTEDHAMMER",
                "CDLKICKING",
                "CDLKICKINGBYLENGTH",
                "CDLLADDERBOTTOM",
                "CDLLONGLEGGEDDOJI",
                "CDLMARUBOZU",
                "CDLMATCHINGLOW",
                "CDLMORNINGDOJISTAR",
                "CDLMORNINGSTAR",
                "CDLONNECK",
                "CDLPIERCING",
                "CDLRICKSHAWMAN",
                "CDLRISEFALL3METHODS",
                "CDLSEPARATINGLINES",
                "CDLSHOOTINGSTAR",
                "CDLSHORTLINE",
                "CDLSPINNINGTOP",
                "CDLSTALLEDPATTERN",
                "CDLSTICKSANDWICH",
                "CDLTAKURI",
                "CDLTASUKIGAP",
                "CDLTHRUSTING",
                "CDLTRISTAR",
                "CDLUNIQUE3RIVER",
                "CDLUPSIDEGAP2CROWS",
                "CDLXSIDEGAP3METHODS",
            ]

            # Calculate selected important patterns
            important_patterns = [
                "CDLDOJI",
                "CDLHAMMER",
                "CDLENGULFING",
                "CDLMORNINGSTAR",
                "CDLEVENINGSTAR",
                "CDLSHOOTINGSTAR",
                "CDLDRAGONFLYDOJI",
                "CDLPIERCING",
                "CDLDARKCLOUDCOVER",
                "CDLHARAMI",
            ]

            for pattern in important_patterns:
                try:
                    func = getattr(talib, pattern)
                    features[pattern] = func(
                        df["open"], df["high"], df["low"], df["close"]
                    )
                except:
                    continue

            # Pattern strength (sum of all patterns)
            pattern_cols = [col for col in features.columns if col.startswith("CDL")]
            if pattern_cols:
                features["pattern_strength"] = features[pattern_cols].sum(axis=1)
                features["bullish_patterns"] = (
                    features[pattern_cols]
                    .where(features[pattern_cols] > 0, 0)
                    .sum(axis=1)
                )
                features["bearish_patterns"] = (
                    features[pattern_cols]
                    .where(features[pattern_cols] < 0, 0)
                    .sum(axis=1)
                )

            return features

        except Exception as e:
            self.logger.error(f"Error calculating pattern indicators: {e}")
            return pd.DataFrame(index=df.index)

    def _calculate_custom_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate custom indicators"""
        features = pd.DataFrame(index=df.index)

        try:
            # Fractals
            if self.config.fractal_enabled:
                fractals = self._calculate_fractals(df)
                features = pd.concat([features, fractals], axis=1)

            # Support/Resistance levels
            if self.config.support_resistance_enabled:
                sr_levels = self._calculate_support_resistance(df)
                features = pd.concat([features, sr_levels], axis=1)

            # Market structure
            features["market_structure"] = self._calculate_market_structure(df)

            # Price action signals
            features["inside_bar"] = self._detect_inside_bars(df)
            features["outside_bar"] = self._detect_outside_bars(df)

            # Gap detection
            features["gap_up"] = (df["open"] > df["high"].shift(1)).astype(int)
            features["gap_down"] = (df["open"] < df["low"].shift(1)).astype(int)

            # Trend strength
            features["trend_strength"] = self._calculate_trend_strength(df)

            return features

        except Exception as e:
            self.logger.error(f"Error calculating custom indicators: {e}")
            return pd.DataFrame(index=df.index)

    def _calculate_statistical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate statistical indicators"""
        features = pd.DataFrame(index=df.index)

        try:
            # Returns
            features["returns"] = df["close"].pct_change()
            features["log_returns"] = np.log(df["close"]).diff()

            # Rolling statistics
            for window in [5, 10, 20]:
                features[f"returns_mean_{window}"] = (
                    features["returns"].rolling(window).mean()
                )
                features[f"returns_std_{window}"] = (
                    features["returns"].rolling(window).std()
                )
                features[f"returns_skew_{window}"] = (
                    features["returns"].rolling(window).skew()
                )
                features[f"returns_kurt_{window}"] = (
                    features["returns"].rolling(window).kurt()
                )

            # Z-score
            features["price_zscore"] = (
                df["close"] - df["close"].rolling(20).mean()
            ) / df["close"].rolling(20).std()

            # Percentile rank
            features["price_percentile"] = df["close"].rolling(50).rank(pct=True)

            # Distance from extremes
            features["dist_from_high"] = (
                df["high"].rolling(50).max() - df["close"]
            ) / df["close"]
            features["dist_from_low"] = (
                df["close"] - df["low"].rolling(50).min()
            ) / df["close"]

            return features

        except Exception as e:
            self.logger.error(f"Error calculating statistical indicators: {e}")
            return pd.DataFrame(index=df.index)

    def _calculate_fractals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate fractal levels"""
        features = pd.DataFrame(index=df.index)

        try:
            # Fractal highs and lows
            features["fractal_high"] = 0.0
            features["fractal_low"] = 0.0

            for i in range(2, len(df) - 2):
                # Fractal high
                if (
                    df["high"].iloc[i] > df["high"].iloc[i - 2 : i].max()
                    and df["high"].iloc[i] > df["high"].iloc[i + 1 : i + 3].max()
                    and df["high"].iloc[i] > df["high"].iloc[i - 1]
                    and df["high"].iloc[i] > df["high"].iloc[i + 1]
                ):
                    features["fractal_high"].iloc[i] = df["high"].iloc[i]

                # Fractal low
                if (
                    df["low"].iloc[i] < df["low"].iloc[i - 2 : i].min()
                    and df["low"].iloc[i] < df["low"].iloc[i + 1 : i + 3].min()
                    and df["low"].iloc[i] < df["low"].iloc[i - 1]
                    and df["low"].iloc[i] < df["low"].iloc[i + 1]
                ):
                    features["fractal_low"].iloc[i] = df["low"].iloc[i]

            return features

        except Exception as e:
            self.logger.error(f"Error calculating fractals: {e}")
            return pd.DataFrame(index=df.index)

    def _calculate_support_resistance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate support and resistance levels"""
        features = pd.DataFrame(index=df.index)

        try:
            # Simple support/resistance based on recent highs/lows
            features["resistance"] = df["high"].rolling(20).max()
            features["support"] = df["low"].rolling(20).min()

            # Distance to S/R levels
            features["dist_to_resistance"] = (
                features["resistance"] - df["close"]
            ) / df["close"]
            features["dist_to_support"] = (df["close"] - features["support"]) / df[
                "close"
            ]

            return features

        except Exception as e:
            self.logger.error(f"Error calculating support/resistance: {e}")
            return pd.DataFrame(index=df.index)

    def _calculate_market_structure(self, df: pd.DataFrame) -> pd.Series:
        """Calculate market structure (trending/ranging)"""
        try:
            # ADX for trend strength
            adx = talib.ADX(df["high"], df["low"], df["close"])

            # Define market structure
            structure = pd.Series(index=df.index, dtype=float)
            structure[adx > 25] = 1  # Trending
            structure[adx <= 25] = 0  # Ranging

            return structure.fillna(0)

        except Exception as e:
            self.logger.error(f"Error calculating market structure: {e}")
            return pd.Series(0, index=df.index)

    def _detect_inside_bars(self, df: pd.DataFrame) -> pd.Series:
        """Detect inside bars"""
        try:
            inside_bars = (
                (df["high"] <= df["high"].shift(1)) & (df["low"] >= df["low"].shift(1))
            ).astype(int)

            return inside_bars

        except Exception as e:
            self.logger.error(f"Error detecting inside bars: {e}")
            return pd.Series(0, index=df.index)

    def _detect_outside_bars(self, df: pd.DataFrame) -> pd.Series:
        """Detect outside bars"""
        try:
            outside_bars = (
                (df["high"] > df["high"].shift(1)) & (df["low"] < df["low"].shift(1))
            ).astype(int)

            return outside_bars

        except Exception as e:
            self.logger.error(f"Error detecting outside bars: {e}")
            return pd.Series(0, index=df.index)

    def _calculate_trend_strength(self, df: pd.DataFrame) -> pd.Series:
        """Calculate trend strength"""
        try:
            # Using ADX
            adx = talib.ADX(df["high"], df["low"], df["close"])

            # Normalize to 0-1 scale
            trend_strength = adx / 100

            return trend_strength.fillna(0)

        except Exception as e:
            self.logger.error(f"Error calculating trend strength: {e}")
            return pd.Series(0, index=df.index)

    def _validate_data(self, df: pd.DataFrame) -> bool:
        """Validate input OHLCV data"""
        try:
            required_columns = ["open", "high", "low", "close"]

            if not all(col in df.columns for col in required_columns):
                self.logger.error("Missing required OHLCV columns")
                return False

            if len(df) < 50:
                self.logger.warning("Insufficient data for reliable calculations")
                return False

            if df[required_columns].isnull().any().any():
                self.logger.warning("Data contains NaN values")

            return True

        except Exception as e:
            self.logger.error(f"Error validating data: {e}")
            return False

    def _clean_cache(self):
        """Clean old cache entries"""
        try:
            if len(self.cache) > 100:  # Keep only latest 100 entries
                # Remove oldest entries
                keys_to_remove = list(self.cache.keys())[:-50]
                for key in keys_to_remove:
                    del self.cache[key]

            self.logger.debug(f"Cache cleaned, {len(self.cache)} entries remaining")

        except Exception as e:
            self.logger.error(f"Error cleaning cache: {e}")

    def get_feature_importance(self, features: pd.DataFrame) -> Dict[str, float]:
        """Calculate feature importance based on correlation with returns"""
        try:
            if "returns" not in features.columns:
                return {}

            # Calculate correlation with returns
            correlations = features.corr()["returns"].abs().sort_values(ascending=False)

            # Remove returns itself and NaN values
            correlations = correlations.drop("returns", errors="ignore").dropna()

            return correlations.head(20).to_dict()

        except Exception as e:
            self.logger.error(f"Error calculating feature importance: {e}")
            return {}

    def get_feature_summary(self, features: pd.DataFrame) -> Dict[str, any]:
        """Get summary statistics of calculated features"""
        try:
            summary = {
                "total_features": len(features.columns),
                "data_points": len(features),
                "null_percentage": features.isnull().sum().sum()
                / (len(features) * len(features.columns))
                * 100,
                "feature_groups": {
                    "basic": len(
                        [
                            c
                            for c in features.columns
                            if any(x in c for x in ["MA", "EMA"])
                        ]
                    ),
                    "momentum": len(
                        [
                            c
                            for c in features.columns
                            if any(x in c for x in ["RSI", "MACD", "STOCH"])
                        ]
                    ),
                    "volatility": len(
                        [
                            c
                            for c in features.columns
                            if any(x in c for x in ["BB", "ATR", "KC"])
                        ]
                    ),
                    "volume": len(
                        [
                            c
                            for c in features.columns
                            if any(x in c for x in ["OBV", "AD", "MFI"])
                        ]
                    ),
                    "pattern": len(
                        [c for c in features.columns if c.startswith("CDL")]
                    ),
                    "custom": len(
                        [
                            c
                            for c in features.columns
                            if any(x in c for x in ["fractal", "support", "resistance"])
                        ]
                    ),
                },
                "last_update": self.last_update,
            }

            return summary

        except Exception as e:
            self.logger.error(f"Error generating feature summary: {e}")
            return {}


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create sample data
    dates = pd.date_range(start="2023-01-01", periods=1000, freq="1H")
    np.random.seed(42)

    price_data = {
        "open": 100 + np.cumsum(np.random.randn(1000) * 0.01),
        "high": 100
        + np.cumsum(np.random.randn(1000) * 0.01)
        + abs(np.random.randn(1000) * 0.5),
        "low": 100
        + np.cumsum(np.random.randn(1000) * 0.01)
        - abs(np.random.randn(1000) * 0.5),
        "close": 100 + np.cumsum(np.random.randn(1000) * 0.01),
        "volume": np.random.randint(1000, 10000, 1000),
    }

    df = pd.DataFrame(price_data, index=dates)
    df["high"] = np.maximum(df["high"], df[["open", "close"]].max(axis=1))
    df["low"] = np.minimum(df["low"], df[["open", "close"]].min(axis=1))

    # Initialize feature engineer
    config = IndicatorConfig()
    fe = FeatureEngineer(config)

    # Calculate features
    print("Calculating features...")
    features = fe.calculate_all_features(df, symbol="EURUSD")

    # Display results
    print(f"\nFeature calculation completed!")
    print(f"Total features: {len(features.columns)}")
    print(f"Data points: {len(features)}")

    # Feature summary
    summary = fe.get_feature_summary(features)
    print(f"\nFeature Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")

    # Feature importance
    importance = fe.get_feature_importance(features)
    print(f"\nTop 10 Most Important Features:")
    for feature, score in list(importance.items())[:10]:
        print(f"{feature}: {score:.4f}")
