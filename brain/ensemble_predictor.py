"""
Institutional-Grade AI Ensemble Predictor
ระบบ AI หลักที่รวม predictions จากหลาย models เข้าด้วยกัน

Features:
- Multi-model ensemble predictions (LSTM, Transformer, Random Forest, XGBoost)
- Dynamic model weighting based on recent performance
- Confidence scoring and uncertainty quantification
- Real-time prediction with feature engineering
- Active learning integration for model improvement
- Risk-adjusted signal generation
- Market regime detection and adaptation
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import joblib
import json
import warnings

warnings.filterwarnings("ignore")

# Scientific computing
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb

# Deep Learning (จะใช้ placeholders ถ้าไม่มี actual models)
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Custom imports
from config.model_config import MODEL_CONFIG, ENSEMBLE_CONFIG
from config.settings import TRADING_SETTINGS
from memory.trade_logger import TradeLogger
from utils.logger_config import setup_logger


class MarketRegime(Enum):
    """Market Regime Types"""

    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    BREAKOUT = "BREAKOUT"


class SignalStrength(Enum):
    """Signal Strength Levels"""

    VERY_WEAK = 1
    WEAK = 2
    MODERATE = 3
    STRONG = 4
    VERY_STRONG = 5


@dataclass
class PredictionResult:
    """Structure for prediction results"""

    direction: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float  # 0.0 to 1.0
    strength: SignalStrength
    price_target: Optional[float]
    probability_up: float
    probability_down: float
    expected_return: float
    risk_score: float
    market_regime: MarketRegime
    model_predictions: Dict[str, float]
    model_weights: Dict[str, float]
    features_used: List[str]
    timestamp: datetime


@dataclass
class ModelPerformance:
    """Track individual model performance"""

    name: str
    accuracy: float
    sharpe_ratio: float
    max_drawdown: float
    recent_predictions: List[float]
    recent_actuals: List[float]
    weight: float
    last_updated: datetime


class EnsemblePredictor:
    """
    Advanced Ensemble Prediction System
    ใช้หลาย AI models ร่วมกันเพื่อความแม่นยำสูง
    """

    def __init__(self, symbol: str = "EURUSD"):
        """
        Initialize Ensemble Predictor

        Args:
            symbol: Trading symbol
        """
        self.symbol = symbol
        self.logger = setup_logger("EnsemblePredictor", f"logs/ensemble_{symbol}.log")

        # Model storage
        self.models = {}
        self.scalers = {}
        self.model_performances = {}

        # Prediction history
        self.prediction_history = []
        self.performance_window = ENSEMBLE_CONFIG.get("performance_window", 100)

        # Market regime detection
        self.current_regime = MarketRegime.RANGING
        self.regime_history = []

        # Feature engineering
        self.feature_columns = []
        self.feature_scaler = RobustScaler()

        # Active learning
        self.uncertainty_threshold = ENSEMBLE_CONFIG.get("uncertainty_threshold", 0.3)
        self.retrain_frequency = ENSEMBLE_CONFIG.get("retrain_frequency", 1000)
        self.prediction_count = 0

        # Load or initialize models
        self._initialize_models()

        self.logger.info(f"EnsemblePredictor initialized for {symbol}")

    def _initialize_models(self):
        """Initialize all ensemble models"""
        try:
            # Traditional ML Models
            self._init_random_forest()
            self._init_xgboost()

            # Deep Learning Models (placeholders)
            if TF_AVAILABLE:
                self._init_lstm_model()
                self._init_transformer_model()

            # Initialize model performances
            for model_name in self.models.keys():
                self.model_performances[model_name] = ModelPerformance(
                    name=model_name,
                    accuracy=0.5,
                    sharpe_ratio=0.0,
                    max_drawdown=0.0,
                    recent_predictions=[],
                    recent_actuals=[],
                    weight=1.0 / len(self.models),
                    last_updated=datetime.now(),
                )

            self.logger.info(f"Initialized {len(self.models)} models")

        except Exception as e:
            self.logger.error(f"Failed to initialize models: {e}")
            raise

    def _init_random_forest(self):
        """Initialize Random Forest model"""
        try:
            rf_config = MODEL_CONFIG["random_forest"]
            self.models["random_forest"] = RandomForestRegressor(
                n_estimators=rf_config["n_estimators"],
                max_depth=rf_config["max_depth"],
                min_samples_split=rf_config["min_samples_split"],
                min_samples_leaf=rf_config["min_samples_leaf"],
                random_state=42,
                n_jobs=-1,
            )
            self.scalers["random_forest"] = StandardScaler()
            self.logger.info("Random Forest model initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize Random Forest: {e}")

    def _init_xgboost(self):
        """Initialize XGBoost model"""
        try:
            xgb_config = MODEL_CONFIG["xgboost"]
            self.models["xgboost"] = xgb.XGBRegressor(
                n_estimators=xgb_config["n_estimators"],
                max_depth=xgb_config["max_depth"],
                learning_rate=xgb_config["learning_rate"],
                subsample=xgb_config["subsample"],
                colsample_bytree=xgb_config["colsample_bytree"],
                random_state=42,
                n_jobs=-1,
            )
            self.scalers["xgboost"] = StandardScaler()
            self.logger.info("XGBoost model initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize XGBoost: {e}")

    def _init_lstm_model(self):
        """Initialize LSTM model (placeholder)"""
        try:
            # This will be implemented in lstm_model.py
            # For now, create a simple placeholder
            self.models["lstm"] = None  # Will be loaded from saved model
            self.scalers["lstm"] = StandardScaler()
            self.logger.info("LSTM model placeholder initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize LSTM: {e}")

    def _init_transformer_model(self):
        """Initialize Transformer model (placeholder)"""
        try:
            # This will be implemented in transformer_model.py
            # For now, create a simple placeholder
            self.models["transformer"] = None  # Will be loaded from saved model
            self.scalers["transformer"] = StandardScaler()
            self.logger.info("Transformer model placeholder initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize Transformer: {e}")

    def predict(
        self, features: pd.DataFrame, market_data: pd.DataFrame = None
    ) -> PredictionResult:
        """
        Generate ensemble prediction

        Args:
            features: Engineered features DataFrame
            market_data: Raw market data for regime detection

        Returns:
            PredictionResult with comprehensive prediction info
        """
        try:
            start_time = datetime.now()

            # Detect market regime
            if market_data is not None:
                self.current_regime = self._detect_market_regime(market_data)

            # Get individual model predictions
            model_predictions = self._get_model_predictions(features)

            # Calculate ensemble prediction
            ensemble_pred = self._calculate_ensemble_prediction(model_predictions)

            # Generate trading signal
            signal = self._generate_trading_signal(
                ensemble_pred, model_predictions, features
            )

            # Update prediction count for active learning
            self.prediction_count += 1

            # Log prediction
            self.logger.debug(f"Prediction generated in {datetime.now() - start_time}")

            return signal

        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return self._generate_default_prediction()

    def _get_model_predictions(self, features: pd.DataFrame) -> Dict[str, float]:
        """Get predictions from all models"""
        predictions = {}

        try:
            # Prepare features
            if len(features) == 0:
                return predictions

            feature_array = (
                features.values.reshape(1, -1)
                if len(features.shape) == 1
                else features.values
            )

            # Get predictions from each model
            for model_name, model in self.models.items():
                if model is None:
                    continue

                try:
                    # Scale features
                    scaler = self.scalers.get(model_name)
                    if scaler is not None:
                        scaled_features = scaler.transform(feature_array)
                    else:
                        scaled_features = feature_array

                    # Get prediction
                    if model_name in ["lstm", "transformer"]:
                        # Deep learning models (placeholder)
                        pred = self._predict_deep_model(model_name, scaled_features)
                    else:
                        # Traditional ML models
                        pred = model.predict(scaled_features)[0]

                    predictions[model_name] = float(pred)

                except Exception as e:
                    self.logger.warning(f"Model {model_name} prediction failed: {e}")
                    predictions[model_name] = 0.0

            return predictions

        except Exception as e:
            self.logger.error(f"Failed to get model predictions: {e}")
            return {}

    def _predict_deep_model(self, model_name: str, features: np.ndarray) -> float:
        """Predict using deep learning models (placeholder)"""
        # This is a placeholder - actual implementation will be in separate model files
        # For now, return a simple prediction based on feature mean
        try:
            if len(features) > 0:
                # Simple heuristic based on feature values
                feature_mean = np.mean(features)
                return float(np.tanh(feature_mean))  # Bounded between -1 and 1
            return 0.0
        except:
            return 0.0

    def _calculate_ensemble_prediction(
        self, model_predictions: Dict[str, float]
    ) -> float:
        """Calculate weighted ensemble prediction"""
        try:
            if not model_predictions:
                return 0.0

            # Get current model weights
            total_weighted_pred = 0.0
            total_weight = 0.0

            for model_name, prediction in model_predictions.items():
                if model_name in self.model_performances:
                    weight = self.model_performances[model_name].weight
                    total_weighted_pred += prediction * weight
                    total_weight += weight

            if total_weight > 0:
                ensemble_pred = total_weighted_pred / total_weight
            else:
                ensemble_pred = np.mean(list(model_predictions.values()))

            return float(ensemble_pred)

        except Exception as e:
            self.logger.error(f"Failed to calculate ensemble prediction: {e}")
            return 0.0

    def _generate_trading_signal(
        self,
        ensemble_pred: float,
        model_predictions: Dict[str, float],
        features: pd.DataFrame,
    ) -> PredictionResult:
        """Generate comprehensive trading signal"""
        try:
            # Calculate prediction statistics
            pred_values = list(model_predictions.values())
            pred_std = np.std(pred_values) if len(pred_values) > 1 else 0.0
            pred_mean = np.mean(pred_values) if pred_values else 0.0

            # Calculate confidence based on model agreement
            confidence = self._calculate_confidence(model_predictions, pred_std)

            # Determine direction
            direction_threshold = ENSEMBLE_CONFIG.get("direction_threshold", 0.1)
            if ensemble_pred > direction_threshold:
                direction = "BUY"
                probability_up = 0.5 + min(0.5, abs(ensemble_pred))
                probability_down = 1.0 - probability_up
            elif ensemble_pred < -direction_threshold:
                direction = "SELL"
                probability_down = 0.5 + min(0.5, abs(ensemble_pred))
                probability_up = 1.0 - probability_down
            else:
                direction = "HOLD"
                probability_up = 0.5
                probability_down = 0.5

            # Calculate signal strength
            strength = self._calculate_signal_strength(abs(ensemble_pred), confidence)

            # Calculate risk score
            risk_score = self._calculate_risk_score(
                pred_std, confidence, self.current_regime
            )

            # Expected return (simplified)
            expected_return = ensemble_pred * confidence

            # Get model weights
            model_weights = {
                name: perf.weight for name, perf in self.model_performances.items()
            }

            # Create prediction result
            result = PredictionResult(
                direction=direction,
                confidence=confidence,
                strength=strength,
                price_target=None,  # Will be calculated by signal analyzer
                probability_up=probability_up,
                probability_down=probability_down,
                expected_return=expected_return,
                risk_score=risk_score,
                market_regime=self.current_regime,
                model_predictions=model_predictions,
                model_weights=model_weights,
                features_used=(
                    list(features.columns) if hasattr(features, "columns") else []
                ),
                timestamp=datetime.now(),
            )

            # Store prediction history
            self._store_prediction_history(result)

            return result

        except Exception as e:
            self.logger.error(f"Failed to generate trading signal: {e}")
            return self._generate_default_prediction()

    def _calculate_confidence(
        self, model_predictions: Dict[str, float], pred_std: float
    ) -> float:
        """Calculate prediction confidence"""
        try:
            if not model_predictions:
                return 0.0

            # Base confidence on model agreement (lower std = higher confidence)
            agreement_confidence = 1.0 / (1.0 + pred_std)

            # Weight by model performance
            weighted_performance = 0.0
            total_weight = 0.0

            for model_name in model_predictions.keys():
                if model_name in self.model_performances:
                    perf = self.model_performances[model_name]
                    weighted_performance += perf.accuracy * perf.weight
                    total_weight += perf.weight

            if total_weight > 0:
                avg_performance = weighted_performance / total_weight
            else:
                avg_performance = 0.5

            # Combine agreement and performance
            confidence = agreement_confidence * 0.6 + avg_performance * 0.4

            # Apply market regime adjustment
            regime_multiplier = self._get_regime_confidence_multiplier()
            confidence *= regime_multiplier

            return min(1.0, max(0.0, confidence))

        except Exception as e:
            self.logger.error(f"Failed to calculate confidence: {e}")
            return 0.5

    def _calculate_signal_strength(
        self, pred_magnitude: float, confidence: float
    ) -> SignalStrength:
        """Calculate signal strength enum"""
        try:
            # Combine prediction magnitude and confidence
            strength_score = pred_magnitude * confidence

            if strength_score >= 0.8:
                return SignalStrength.VERY_STRONG
            elif strength_score >= 0.6:
                return SignalStrength.STRONG
            elif strength_score >= 0.4:
                return SignalStrength.MODERATE
            elif strength_score >= 0.2:
                return SignalStrength.WEAK
            else:
                return SignalStrength.VERY_WEAK

        except:
            return SignalStrength.WEAK

    def _calculate_risk_score(
        self, pred_std: float, confidence: float, regime: MarketRegime
    ) -> float:
        """Calculate risk score (0.0 = low risk, 1.0 = high risk)"""
        try:
            # Base risk on prediction uncertainty
            uncertainty_risk = pred_std

            # Confidence risk (low confidence = high risk)
            confidence_risk = 1.0 - confidence

            # Market regime risk
            regime_risk = self._get_regime_risk_multiplier(regime)

            # Combine risk factors
            total_risk = (
                uncertainty_risk * 0.4 + confidence_risk * 0.4 + regime_risk * 0.2
            )

            return min(1.0, max(0.0, total_risk))

        except:
            return 0.5

    def _detect_market_regime(self, market_data: pd.DataFrame) -> MarketRegime:
        """Detect current market regime"""
        try:
            if len(market_data) < 20:
                return MarketRegime.RANGING

            # Calculate indicators for regime detection
            close_prices = market_data["close"].tail(20)

            # Trend detection
            sma_short = close_prices.tail(5).mean()
            sma_long = close_prices.tail(20).mean()
            trend_strength = abs(sma_short - sma_long) / sma_long

            # Volatility detection
            returns = close_prices.pct_change().dropna()
            volatility = returns.std()
            high_vol_threshold = 0.02  # 2% daily volatility

            # Price range
            price_range = (
                close_prices.max() - close_prices.min()
            ) / close_prices.mean()

            # Determine regime
            if volatility > high_vol_threshold:
                if trend_strength > 0.01:
                    if sma_short > sma_long:
                        return MarketRegime.TRENDING_UP
                    else:
                        return MarketRegime.TRENDING_DOWN
                else:
                    return MarketRegime.HIGH_VOLATILITY
            elif trend_strength > 0.005:
                if sma_short > sma_long:
                    return MarketRegime.TRENDING_UP
                else:
                    return MarketRegime.TRENDING_DOWN
            elif price_range < 0.01:
                return MarketRegime.LOW_VOLATILITY
            else:
                return MarketRegime.RANGING

        except Exception as e:
            self.logger.error(f"Failed to detect market regime: {e}")
            return MarketRegime.RANGING

    def _get_regime_confidence_multiplier(self) -> float:
        """Get confidence multiplier based on market regime"""
        multipliers = {
            MarketRegime.TRENDING_UP: 1.2,
            MarketRegime.TRENDING_DOWN: 1.2,
            MarketRegime.RANGING: 0.8,
            MarketRegime.HIGH_VOLATILITY: 0.6,
            MarketRegime.LOW_VOLATILITY: 1.1,
            MarketRegime.BREAKOUT: 1.3,
        }
        return multipliers.get(self.current_regime, 1.0)

    def _get_regime_risk_multiplier(self, regime: MarketRegime) -> float:
        """Get risk multiplier based on market regime"""
        multipliers = {
            MarketRegime.TRENDING_UP: 0.3,
            MarketRegime.TRENDING_DOWN: 0.3,
            MarketRegime.RANGING: 0.5,
            MarketRegime.HIGH_VOLATILITY: 0.9,
            MarketRegime.LOW_VOLATILITY: 0.2,
            MarketRegime.BREAKOUT: 0.8,
        }
        return multipliers.get(regime, 0.5)

    def _store_prediction_history(self, prediction: PredictionResult):
        """Store prediction in history for performance tracking"""
        try:
            self.prediction_history.append(prediction)

            # Keep only recent predictions
            max_history = ENSEMBLE_CONFIG.get("max_prediction_history", 10000)
            if len(self.prediction_history) > max_history:
                self.prediction_history = self.prediction_history[-max_history:]

        except Exception as e:
            self.logger.error(f"Failed to store prediction history: {e}")

    def update_model_performance(self, ticket: int, actual_result: float):
        """
        Update model performances based on actual trade results

        Args:
            ticket: Trade ticket number
            actual_result: Actual trade result (profit/loss)
        """
        try:
            # Find corresponding prediction
            prediction = None
            for pred in reversed(self.prediction_history):
                if (
                    abs((pred.timestamp - datetime.now()).total_seconds()) < 3600
                ):  # Within 1 hour
                    prediction = pred
                    break

            if not prediction:
                return

            # Update each model's performance
            for model_name, model_pred in prediction.model_predictions.items():
                if model_name in self.model_performances:
                    perf = self.model_performances[model_name]

                    # Add to recent predictions
                    perf.recent_predictions.append(model_pred)
                    perf.recent_actuals.append(actual_result)

                    # Keep only recent window
                    if len(perf.recent_predictions) > self.performance_window:
                        perf.recent_predictions = perf.recent_predictions[
                            -self.performance_window :
                        ]
                        perf.recent_actuals = perf.recent_actuals[
                            -self.performance_window :
                        ]

                    # Recalculate performance metrics
                    if len(perf.recent_predictions) >= 10:
                        self._update_model_metrics(model_name)

            # Rebalance model weights
            self._rebalance_model_weights()

            self.logger.info(f"Updated model performances for ticket {ticket}")

        except Exception as e:
            self.logger.error(f"Failed to update model performance: {e}")

    def _update_model_metrics(self, model_name: str):
        """Update performance metrics for specific model"""
        try:
            perf = self.model_performances[model_name]

            if len(perf.recent_predictions) < 10:
                return

            predictions = np.array(perf.recent_predictions)
            actuals = np.array(perf.recent_actuals)

            # Calculate accuracy (correlation)
            correlation = np.corrcoef(predictions, actuals)[0, 1]
            perf.accuracy = max(0.0, correlation) if not np.isnan(correlation) else 0.0

            # Calculate Sharpe ratio
            if len(actuals) > 1:
                sharpe = np.mean(actuals) / (np.std(actuals) + 1e-8)
                perf.sharpe_ratio = sharpe

            # Calculate max drawdown
            cumulative = np.cumsum(actuals)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = cumulative - running_max
            perf.max_drawdown = abs(np.min(drawdown))

            perf.last_updated = datetime.now()

        except Exception as e:
            self.logger.error(f"Failed to update metrics for {model_name}: {e}")

    def _rebalance_model_weights(self):
        """Rebalance model weights based on performance"""
        try:
            # Calculate new weights based on performance
            total_score = 0.0
            scores = {}

            for model_name, perf in self.model_performances.items():
                # Combine accuracy and Sharpe ratio
                score = perf.accuracy * 0.7 + max(0, perf.sharpe_ratio) * 0.3
                scores[model_name] = max(0.1, score)  # Minimum weight
                total_score += scores[model_name]

            # Normalize weights
            if total_score > 0:
                for model_name in scores:
                    self.model_performances[model_name].weight = (
                        scores[model_name] / total_score
                    )

            self.logger.debug("Model weights rebalanced")

        except Exception as e:
            self.logger.error(f"Failed to rebalance model weights: {e}")

    def _generate_default_prediction(self) -> PredictionResult:
        """Generate default prediction when errors occur"""
        return PredictionResult(
            direction="HOLD",
            confidence=0.0,
            strength=SignalStrength.VERY_WEAK,
            price_target=None,
            probability_up=0.5,
            probability_down=0.5,
            expected_return=0.0,
            risk_score=1.0,
            market_regime=MarketRegime.RANGING,
            model_predictions={},
            model_weights={},
            features_used=[],
            timestamp=datetime.now(),
        )

    def get_model_status(self) -> Dict[str, Any]:
        """Get current status of all models"""
        try:
            status = {
                "symbol": self.symbol,
                "current_regime": self.current_regime.value,
                "prediction_count": self.prediction_count,
                "models": {},
            }

            for model_name, perf in self.model_performances.items():
                status["models"][model_name] = {
                    "accuracy": round(perf.accuracy, 4),
                    "sharpe_ratio": round(perf.sharpe_ratio, 4),
                    "weight": round(perf.weight, 4),
                    "predictions_count": len(perf.recent_predictions),
                    "last_updated": perf.last_updated.isoformat(),
                }

            return status

        except Exception as e:
            self.logger.error(f"Failed to get model status: {e}")
            return {}

    def save_ensemble_state(self, filepath: str):
        """Save ensemble state for recovery"""
        try:
            state = {
                "symbol": self.symbol,
                "model_performances": {
                    name: {
                        "accuracy": perf.accuracy,
                        "sharpe_ratio": perf.sharpe_ratio,
                        "max_drawdown": perf.max_drawdown,
                        "weight": perf.weight,
                        "recent_predictions": perf.recent_predictions[-50:],  # Last 50
                        "recent_actuals": perf.recent_actuals[-50:],
                        "last_updated": perf.last_updated.isoformat(),
                    }
                    for name, perf in self.model_performances.items()
                },
                "current_regime": self.current_regime.value,
                "prediction_count": self.prediction_count,
            }

            with open(filepath, "w") as f:
                json.dump(state, f, indent=2)

            self.logger.info(f"Ensemble state saved to {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to save ensemble state: {e}")

    def load_ensemble_state(self, filepath: str):
        """Load ensemble state from file"""
        try:
            with open(filepath, "r") as f:
                state = json.load(f)

            # Restore model performances
            for name, perf_data in state.get("model_performances", {}).items():
                if name in self.model_performances:
                    perf = self.model_performances[name]
                    perf.accuracy = perf_data["accuracy"]
                    perf.sharpe_ratio = perf_data["sharpe_ratio"]
                    perf.max_drawdown = perf_data["max_drawdown"]
                    perf.weight = perf_data["weight"]
                    perf.recent_predictions = perf_data["recent_predictions"]
                    perf.recent_actuals = perf_data["recent_actuals"]
                    perf.last_updated = datetime.fromisoformat(
                        perf_data["last_updated"]
                    )

            # Restore other state
            self.current_regime = MarketRegime(state.get("current_regime", "RANGING"))
            self.prediction_count = state.get("prediction_count", 0)

            self.logger.info(f"Ensemble state loaded from {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to load ensemble state: {e}")


# Usage Example
if __name__ == "__main__":
    # Initialize ensemble predictor
    predictor = EnsemblePredictor("EURUSD")

    # Example features (normally from feature_engineer.py)
    features = pd.DataFrame(
        {
            "rsi": [65.5],
            "macd": [0.002],
            "bb_position": [0.7],
            "volume_ratio": [1.2],
            "atr": [0.0015],
        }
    )

    # Example market data
    market_data = pd.DataFrame(
        {
            "close": [1.1000, 1.1010, 1.1020, 1.1015, 1.1025] * 4,
            "volume": [1000, 1200, 900, 1100, 1300] * 4,
        }
    )

    # Generate prediction
    prediction = predictor.predict(features, market_data)

    print(f"Direction: {prediction.direction}")
    print(f"Confidence: {prediction.confidence:.3f}")
    print(f"Strength: {prediction.strength}")
    print(f"Market Regime: {prediction.market_regime}")
    print(f"Model Predictions: {prediction.model_predictions}")

    # Get model status
    status = predictor.get_model_status()
    print(f"Model Status: {status}")
