"""
LSTM Time Series Prediction Model
สำหรับ AI Trading System ที่เชื่อมต่อ MT5 บัญชีจริง

Features:
- Multi-layered LSTM architecture for time series prediction
- Bidirectional LSTM for pattern recognition
- Attention mechanism integration
- Price movement and volatility prediction
- Technical indicators integration
- Dropout and regularization for overfitting prevention
- Real-time training and inference
- Model checkpointing and recovery
- GPU acceleration support
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import pickle
import json
import warnings

warnings.filterwarnings("ignore")

# Deep Learning imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model, Sequential, load_model
    from tensorflow.keras.layers import (
        LSTM,
        Dense,
        Dropout,
        BatchNormalization,
        Input,
        Bidirectional,
        Attention,
        MultiHeadAttention,
        LayerNormalization,
        Add,
        Concatenate,
    )
    from tensorflow.keras.optimizers import Adam, RMSprop
    from tensorflow.keras.callbacks import (
        EarlyStopping,
        ModelCheckpoint,
        ReduceLROnPlateau,
        TensorBoard,
    )
    from tensorflow.keras.regularizers import l1_l2
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available. LSTM model will use simplified implementation.")

# Custom imports
from config.model_config import MODEL_CONFIG
from utils.logger_config import setup_logger


class LSTMTimeSeriesPredictor:
    """
    Advanced LSTM Model for Forex Time Series Prediction
    รองรับการพยากรณ์ราคาและความผันผวน
    """

    def __init__(
        self,
        symbol: str = "EURUSD",
        sequence_length: int = 60,
        prediction_horizon: int = 1,
    ):
        """
        Initialize LSTM Model

        Args:
            symbol: Trading symbol
            sequence_length: Length of input sequences
            prediction_horizon: Number of steps to predict ahead
        """
        self.symbol = symbol
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon

        # Setup logging
        self.logger = setup_logger("LSTMModel", f"logs/lstm_{symbol}.log")

        # Model configuration
        if TF_AVAILABLE:
            self.config = MODEL_CONFIG.get(
                "lstm",
                {
                    "units": [128, 64, 32],
                    "dropout": 0.2,
                    "recurrent_dropout": 0.1,
                    "learning_rate": 0.001,
                    "batch_size": 32,
                    "epochs": 100,
                    "patience": 15,
                },
            )
        else:
            self.config = {}

        # Model components
        self.model = None
        self.price_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.feature_scaler = StandardScaler()
        self.volatility_scaler = MinMaxScaler(feature_range=(0, 1))

        # Training data storage
        self.training_data = []
        self.validation_data = []

        # Model metadata
        self.is_trained = False
        self.last_training_time = None
        self.training_history = {}
        self.model_version = "1.0"

        # Performance tracking
        self.prediction_accuracy = 0.0
        self.directional_accuracy = 0.0
        self.recent_predictions = []
        self.recent_actuals = []

        if TF_AVAILABLE:
            # Setup TensorFlow
            self._setup_tensorflow()

            # Build model architecture
            self._build_model()
        else:
            # Use simplified prediction model
            self._build_simplified_model()

        self.logger.info(f"LSTM Model initialized for {symbol}")

    def _setup_tensorflow(self):
        """Setup TensorFlow configuration"""
        try:
            # Set memory growth for GPU
            gpus = tf.config.experimental.list_physical_devices("GPU")
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                self.logger.info(f"GPU acceleration enabled: {len(gpus)} GPU(s)")
            else:
                self.logger.info("Using CPU for training")

            # Set random seeds for reproducibility
            tf.random.set_seed(42)
            np.random.seed(42)

        except Exception as e:
            self.logger.warning(f"TensorFlow setup issue: {e}")

    def _build_model(self):
        """Build LSTM model architecture"""
        try:
            if not TF_AVAILABLE:
                return

            # Input layers
            price_input = Input(shape=(self.sequence_length, 1), name="price_input")
            feature_input = Input(
                shape=(self.sequence_length, 10), name="feature_input"
            )  # 10 technical indicators

            # Price processing branch
            price_lstm1 = Bidirectional(
                LSTM(
                    self.config["units"][0],
                    return_sequences=True,
                    dropout=self.config["dropout"],
                    recurrent_dropout=self.config["recurrent_dropout"],
                )
            )(price_input)
            price_lstm1 = BatchNormalization()(price_lstm1)

            price_lstm2 = Bidirectional(
                LSTM(
                    self.config["units"][1],
                    return_sequences=True,
                    dropout=self.config["dropout"],
                    recurrent_dropout=self.config["recurrent_dropout"],
                )
            )(price_lstm1)
            price_lstm2 = BatchNormalization()(price_lstm2)

            # Feature processing branch
            feature_lstm1 = LSTM(
                self.config["units"][0],
                return_sequences=True,
                dropout=self.config["dropout"],
                recurrent_dropout=self.config["recurrent_dropout"],
            )(feature_input)
            feature_lstm1 = BatchNormalization()(feature_lstm1)

            feature_lstm2 = LSTM(
                self.config["units"][1],
                return_sequences=True,
                dropout=self.config["dropout"],
                recurrent_dropout=self.config["recurrent_dropout"],
            )(feature_lstm1)
            feature_lstm2 = BatchNormalization()(feature_lstm2)

            # Attention mechanism
            attention_layer = MultiHeadAttention(
                num_heads=4, key_dim=self.config["units"][1]
            )

            # Apply attention to price features
            price_attention = attention_layer(price_lstm2, price_lstm2)
            price_attention = LayerNormalization()(price_attention)
            price_attention = Add()([price_lstm2, price_attention])

            # Apply attention to technical features
            feature_attention = attention_layer(feature_lstm2, feature_lstm2)
            feature_attention = LayerNormalization()(feature_attention)
            feature_attention = Add()([feature_lstm2, feature_attention])

            # Combine branches
            combined = Concatenate()([price_attention, feature_attention])

            # Final LSTM layer
            final_lstm = LSTM(
                self.config["units"][2],
                return_sequences=False,
                dropout=self.config["dropout"],
                recurrent_dropout=self.config["recurrent_dropout"],
            )(combined)
            final_lstm = BatchNormalization()(final_lstm)

            # Dense layers for prediction
            dense1 = Dense(
                64, activation="relu", kernel_regularizer=l1_l2(l1=0.01, l2=0.01)
            )(final_lstm)
            dense1 = Dropout(self.config["dropout"])(dense1)

            dense2 = Dense(
                32, activation="relu", kernel_regularizer=l1_l2(l1=0.01, l2=0.01)
            )(dense1)
            dense2 = Dropout(self.config["dropout"])(dense2)

            # Output layers
            price_output = Dense(
                self.prediction_horizon, activation="linear", name="price_prediction"
            )(dense2)
            direction_output = Dense(
                1, activation="sigmoid", name="direction_prediction"
            )(dense2)
            volatility_output = Dense(
                1, activation="sigmoid", name="volatility_prediction"
            )(dense2)

            # Create model
            self.model = Model(
                inputs=[price_input, feature_input],
                outputs=[price_output, direction_output, volatility_output],
                name=f"LSTM_{self.symbol}",
            )

            # Compile model
            optimizer = Adam(learning_rate=self.config["learning_rate"])

            self.model.compile(
                optimizer=optimizer,
                loss={
                    "price_prediction": "mse",
                    "direction_prediction": "binary_crossentropy",
                    "volatility_prediction": "mse",
                },
                loss_weights={
                    "price_prediction": 0.6,
                    "direction_prediction": 0.3,
                    "volatility_prediction": 0.1,
                },
                metrics={
                    "price_prediction": ["mae"],
                    "direction_prediction": ["accuracy"],
                    "volatility_prediction": ["mae"],
                },
            )

            self.logger.info(f"LSTM model built successfully")
            self.logger.info(f"Model parameters: {self.model.count_params():,}")

        except Exception as e:
            self.logger.error(f"Failed to build LSTM model: {e}")
            self._build_simplified_model()

    def _build_simplified_model(self):
        """Build simplified model when TensorFlow is not available"""
        self.logger.info("Using simplified LSTM model (TensorFlow not available)")

        # Simple moving average based prediction
        self.simple_model = {
            "window_short": 5,
            "window_long": 20,
            "momentum_window": 10,
        }

        self.is_trained = True  # Mark as "trained" for simplified model

    def prepare_data(
        self, market_data: pd.DataFrame, technical_indicators: pd.DataFrame = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from market data

        Args:
            market_data: OHLCV data
            technical_indicators: Technical indicators data

        Returns:
            Tuple of (X, y) for training
        """
        try:
            if len(market_data) < self.sequence_length + self.prediction_horizon:
                raise ValueError("Insufficient data for sequence preparation")

            # Prepare price data
            prices = market_data["close"].values
            volumes = (
                market_data["volume"].values
                if "volume" in market_data.columns
                else np.ones(len(prices))
            )

            # Calculate returns and volatility
            returns = np.diff(prices) / prices[:-1]
            volatility = pd.Series(returns).rolling(window=20).std().fillna(0).values

            # Prepare technical indicators
            if technical_indicators is not None:
                features = technical_indicators.values
            else:
                features = self._calculate_basic_indicators(market_data)

            # Ensure features have correct shape
            min_length = min(len(prices), len(features))
            prices = prices[:min_length]
            features = features[:min_length]
            volatility = (
                volatility[:min_length]
                if len(volatility) >= min_length
                else np.zeros(min_length)
            )

            if TF_AVAILABLE:
                return self._prepare_sequences_tf(prices, features, volatility)
            else:
                return self._prepare_sequences_simple(prices, features)

        except Exception as e:
            self.logger.error(f"Failed to prepare data: {e}")
            return np.array([]), np.array([])

    def _prepare_sequences_tf(
        self, prices: np.ndarray, features: np.ndarray, volatility: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for TensorFlow model"""
        try:
            # Scale data
            prices_scaled = self.price_scaler.fit_transform(
                prices.reshape(-1, 1)
            ).flatten()
            features_scaled = self.feature_scaler.fit_transform(features)
            volatility_scaled = self.volatility_scaler.fit_transform(
                volatility.reshape(-1, 1)
            ).flatten()

            # Create sequences
            X_price, X_features, y_price, y_direction, y_volatility = [], [], [], [], []

            for i in range(
                len(prices_scaled) - self.sequence_length - self.prediction_horizon + 1
            ):
                # Input sequences
                price_seq = prices_scaled[i : i + self.sequence_length].reshape(-1, 1)
                feature_seq = features_scaled[i : i + self.sequence_length]

                # Ensure feature sequence has correct dimensions
                if feature_seq.shape[1] < 10:
                    # Pad with zeros if not enough features
                    padding = np.zeros(
                        (feature_seq.shape[0], 10 - feature_seq.shape[1])
                    )
                    feature_seq = np.concatenate([feature_seq, padding], axis=1)
                elif feature_seq.shape[1] > 10:
                    # Take first 10 features
                    feature_seq = feature_seq[:, :10]

                X_price.append(price_seq)
                X_features.append(feature_seq)

                # Target values
                future_prices = prices_scaled[
                    i
                    + self.sequence_length : i
                    + self.sequence_length
                    + self.prediction_horizon
                ]
                current_price = prices_scaled[i + self.sequence_length - 1]

                # Price prediction (normalized change)
                price_change = future_prices - current_price
                y_price.append(price_change)

                # Direction prediction (1 if up, 0 if down)
                direction = 1.0 if future_prices[-1] > current_price else 0.0
                y_direction.append(direction)

                # Volatility prediction
                vol_target = volatility_scaled[i + self.sequence_length - 1]
                y_volatility.append(vol_target)

            X = [np.array(X_price), np.array(X_features)]
            y = [np.array(y_price), np.array(y_direction), np.array(y_volatility)]

            return X, y

        except Exception as e:
            self.logger.error(f"Failed to prepare TF sequences: {e}")
            return np.array([]), np.array([])

    def _prepare_sequences_simple(
        self, prices: np.ndarray, features: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for simplified model"""
        try:
            X, y = [], []

            for i in range(
                len(prices) - self.sequence_length - self.prediction_horizon + 1
            ):
                # Simple features: recent prices and basic indicators
                recent_prices = prices[i : i + self.sequence_length]
                price_features = [
                    recent_prices[-1],  # Current price
                    np.mean(recent_prices[-5:]),  # Short MA
                    np.mean(recent_prices),  # Long MA
                    np.std(recent_prices),  # Volatility
                    (recent_prices[-1] - recent_prices[0])
                    / recent_prices[0],  # Total return
                ]

                X.append(price_features)

                # Target: future price change
                future_price = prices[
                    i + self.sequence_length + self.prediction_horizon - 1
                ]
                current_price = prices[i + self.sequence_length - 1]
                change = (future_price - current_price) / current_price

                y.append(change)

            return np.array(X), np.array(y)

        except Exception as e:
            self.logger.error(f"Failed to prepare simple sequences: {e}")
            return np.array([]), np.array([])

    def _calculate_basic_indicators(self, market_data: pd.DataFrame) -> np.ndarray:
        """Calculate basic technical indicators"""
        try:
            df = market_data.copy()

            # Simple Moving Averages
            df["sma_5"] = df["close"].rolling(5).mean()
            df["sma_20"] = df["close"].rolling(20).mean()

            # Exponential Moving Averages
            df["ema_12"] = df["close"].ewm(span=12).mean()
            df["ema_26"] = df["close"].ewm(span=26).mean()

            # MACD
            df["macd"] = df["ema_12"] - df["ema_26"]
            df["macd_signal"] = df["macd"].ewm(span=9).mean()

            # RSI
            delta = df["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df["rsi"] = 100 - (100 / (1 + rs))

            # Bollinger Bands
            df["bb_middle"] = df["close"].rolling(20).mean()
            bb_std = df["close"].rolling(20).std()
            df["bb_upper"] = df["bb_middle"] + (bb_std * 2)
            df["bb_lower"] = df["bb_middle"] - (bb_std * 2)
            df["bb_position"] = (df["close"] - df["bb_lower"]) / (
                df["bb_upper"] - df["bb_lower"]
            )

            # Select features
            feature_columns = [
                "sma_5",
                "sma_20",
                "ema_12",
                "ema_26",
                "macd",
                "macd_signal",
                "rsi",
                "bb_position",
            ]

            # Add volume-based indicators if available
            if "volume" in df.columns:
                df["volume_ma"] = df["volume"].rolling(20).mean()
                df["volume_ratio"] = df["volume"] / df["volume_ma"]
                feature_columns.extend(["volume_ratio"])

            # Fill missing values and return
            features = df[feature_columns].fillna(method="bfill").fillna(0)

            return features.values

        except Exception as e:
            self.logger.error(f"Failed to calculate indicators: {e}")
            return np.zeros((len(market_data), 8))

    def train(
        self,
        market_data: pd.DataFrame,
        technical_indicators: pd.DataFrame = None,
        validation_split: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Train the LSTM model

        Args:
            market_data: Historical market data
            technical_indicators: Technical indicators
            validation_split: Validation data split ratio

        Returns:
            Training history
        """
        try:
            self.logger.info("Starting LSTM model training...")

            # Prepare data
            X, y = self.prepare_data(market_data, technical_indicators)

            if len(X) == 0:
                raise ValueError("No training data prepared")

            if TF_AVAILABLE and self.model is not None:
                return self._train_tf_model(X, y, validation_split)
            else:
                return self._train_simple_model(X, y)

        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            return {}

    def _train_tf_model(
        self, X: List[np.ndarray], y: List[np.ndarray], validation_split: float
    ) -> Dict[str, Any]:
        """Train TensorFlow model"""
        try:
            # Split data
            split_idx = int(len(X[0]) * (1 - validation_split))

            X_train = [x[:split_idx] for x in X]
            X_val = [x[split_idx:] for x in X]
            y_train = [target[:split_idx] for target in y]
            y_val = [target[split_idx:] for target in y]

            self.logger.info(
                f"Training samples: {len(X_train[0])}, Validation samples: {len(X_val[0])}"
            )

            # Callbacks
            callbacks = [
                EarlyStopping(
                    monitor="val_loss",
                    patience=self.config["patience"],
                    restore_best_weights=True,
                    verbose=1,
                ),
                ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=10, min_lr=1e-7, verbose=1
                ),
                ModelCheckpoint(
                    f"models/lstm_{self.symbol}_best.h5",
                    monitor="val_loss",
                    save_best_only=True,
                    verbose=1,
                ),
            ]

            # Train model
            history = self.model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=self.config["epochs"],
                batch_size=self.config["batch_size"],
                callbacks=callbacks,
                verbose=1,
            )

            # Update training status
            self.is_trained = True
            self.last_training_time = datetime.now()
            self.training_history = history.history

            # Calculate final metrics
            val_loss = min(history.history["val_loss"])
            val_price_mae = min(
                history.history.get("val_price_prediction_mae", [float("inf")])
            )
            val_direction_acc = max(
                history.history.get("val_direction_prediction_accuracy", [0])
            )

            self.logger.info(
                f"Training completed - Val Loss: {val_loss:.6f}, "
                f"Price MAE: {val_price_mae:.6f}, Direction Acc: {val_direction_acc:.3f}"
            )

            return {
                "final_val_loss": val_loss,
                "final_price_mae": val_price_mae,
                "final_direction_accuracy": val_direction_acc,
                "epochs_trained": len(history.history["loss"]),
                "training_time": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"TF model training failed: {e}")
            return {}

    def _train_simple_model(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train simplified model"""
        try:
            self.logger.info("Training simplified model...")

            # Store training data for simple prediction
            self.training_data = {
                "X": X,
                "y": y,
                "mean_change": np.mean(y),
                "std_change": np.std(y),
            }

            self.is_trained = True
            self.last_training_time = datetime.now()

            # Calculate simple accuracy
            predictions = self._predict_simple(X)
            mse = mean_squared_error(y, predictions)
            mae = mean_absolute_error(y, predictions)

            self.logger.info(f"Simple model trained - MSE: {mse:.6f}, MAE: {mae:.6f}")

            return {
                "mse": mse,
                "mae": mae,
                "samples": len(X),
                "training_time": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Simple model training failed: {e}")
            return {}

    def predict(
        self, market_data: pd.DataFrame, technical_indicators: pd.DataFrame = None
    ) -> Dict[str, Any]:
        """
        Make prediction using the trained model

        Args:
            market_data: Recent market data
            technical_indicators: Technical indicators

        Returns:
            Prediction results
        """
        try:
            if not self.is_trained:
                raise ValueError("Model not trained yet")

            if TF_AVAILABLE and self.model is not None:
                return self._predict_tf_model(market_data, technical_indicators)
            else:
                return self._predict_simple_model(market_data)

        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return self._get_default_prediction()

    def _predict_tf_model(
        self, market_data: pd.DataFrame, technical_indicators: pd.DataFrame = None
    ) -> Dict[str, Any]:
        """Make prediction using TensorFlow model"""
        try:
            # Prepare input data
            X, _ = self.prepare_data(market_data, technical_indicators)

            if len(X[0]) == 0:
                return self._get_default_prediction()

            # Use only the last sequence for prediction
            X_pred = [x[-1:] for x in X]

            # Make prediction
            predictions = self.model.predict(X_pred, verbose=0)

            price_pred = predictions[0][0]
            direction_pred = predictions[1][0][0]
            volatility_pred = predictions[2][0][0]

            # Convert predictions back to original scale
            current_price = market_data["close"].iloc[-1]
            predicted_price_change = self.price_scaler.inverse_transform(
                price_pred.reshape(-1, 1)
            ).flatten()

            # Calculate final metrics
            predicted_price = current_price + predicted_price_change[0]
            direction = "BUY" if direction_pred > 0.5 else "SELL"
            confidence = abs(direction_pred - 0.5) * 2  # Convert to 0-1 scale

            return {
                "predicted_price": float(predicted_price),
                "price_change": float(predicted_price_change[0]),
                "direction": direction,
                "direction_probability": float(direction_pred),
                "confidence": float(confidence),
                "volatility": float(volatility_pred),
                "model_type": "LSTM_TF",
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"TF prediction failed: {e}")
            return self._get_default_prediction()

    def _predict_simple_model(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Make prediction using simplified model"""
        try:
            if not self.training_data:
                return self._get_default_prediction()

            # Get recent prices
            prices = market_data["close"].tail(self.sequence_length).values

            if len(prices) < self.sequence_length:
                return self._get_default_prediction()

            # Simple prediction based on momentum and moving averages
            current_price = prices[-1]
            short_ma = np.mean(prices[-5:])
            long_ma = np.mean(prices)
            momentum = (current_price - prices[0]) / prices[0]
            volatility = np.std(prices) / np.mean(prices)

            # Predict direction based on moving averages
            if short_ma > long_ma * 1.001:  # 0.1% threshold
                direction = "BUY"
                direction_prob = 0.6 + min(0.3, abs(momentum))
            elif short_ma < long_ma * 0.999:
                direction = "SELL"
                direction_prob = 0.4 - min(0.3, abs(momentum))
            else:
                direction = "HOLD"
                direction_prob = 0.5

            # Simple price prediction
            trend_factor = (short_ma - long_ma) / long_ma
            predicted_change = (
                trend_factor * current_price * 0.1
            )  # Conservative prediction
            predicted_price = current_price + predicted_change

            # Confidence based on trend strength
            confidence = min(1.0, abs(trend_factor) * 10 + 0.3)

            return {
                "predicted_price": float(predicted_price),
                "price_change": float(predicted_change),
                "direction": direction,
                "direction_probability": float(direction_prob),
                "confidence": float(confidence),
                "volatility": float(volatility),
                "model_type": "LSTM_Simple",
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Simple prediction failed: {e}")
            return self._get_default_prediction()

    def _predict_simple(self, X: np.ndarray) -> np.ndarray:
        """Simple prediction for training evaluation"""
        try:
            predictions = []
            for x in X:
                # Simple trend-based prediction
                current_price = x[0]
                short_ma = x[1]
                long_ma = x[2]

                trend = (short_ma - long_ma) / long_ma if long_ma != 0 else 0
                pred = trend * 0.1  # Conservative prediction
                predictions.append(pred)

            return np.array(predictions)

        except:
            return np.zeros(len(X))

    def _get_default_prediction(self) -> Dict[str, Any]:
        """Get default prediction when model fails"""
        return {
            "predicted_price": 0.0,
            "price_change": 0.0,
            "direction": "HOLD",
            "direction_probability": 0.5,
            "confidence": 0.0,
            "volatility": 0.0,
            "model_type": "DEFAULT",
            "timestamp": datetime.now().isoformat(),
        }

    def update_with_actual(self, predicted_change: float, actual_change: float):
        """Update model performance with actual results"""
        try:
            self.recent_predictions.append(predicted_change)
            self.recent_actuals.append(actual_change)

            # Keep only recent history
            max_history = 1000
            if len(self.recent_predictions) > max_history:
                self.recent_predictions = self.recent_predictions[-max_history:]
                self.recent_actuals = self.recent_actuals[-max_history:]

            # Calculate updated accuracy
            if len(self.recent_predictions) >= 10:
                mse = mean_squared_error(self.recent_actuals, self.recent_predictions)
                self.prediction_accuracy = 1.0 / (1.0 + mse)

                # Directional accuracy
                pred_directions = [1 if p > 0 else 0 for p in self.recent_predictions]
                actual_directions = [1 if a > 0 else 0 for a in self.recent_actuals]
                correct_directions = sum(
                    1 for p, a in zip(pred_directions, actual_directions) if p == a
                )
                self.directional_accuracy = correct_directions / len(pred_directions)

            self.logger.debug(
                f"Updated accuracy: {self.prediction_accuracy:.3f}, "
                f"Directional: {self.directional_accuracy:.3f}"
            )

        except Exception as e:
            self.logger.error(f"Failed to update with actual: {e}")

    def save_model(self, filepath: str):
        """Save trained model"""
        try:
            if TF_AVAILABLE and self.model is not None:
                self.model.save(filepath)

                # Save scalers and metadata
                metadata = {
                    "symbol": self.symbol,
                    "sequence_length": self.sequence_length,
                    "prediction_horizon": self.prediction_horizon,
                    "model_version": self.model_version,
                    "is_trained": self.is_trained,
                    "last_training_time": (
                        self.last_training_time.isoformat()
                        if self.last_training_time
                        else None
                    ),
                    "prediction_accuracy": self.prediction_accuracy,
                    "directional_accuracy": self.directional_accuracy,
                }

                with open(filepath.replace(".h5", "_metadata.json"), "w") as f:
                    json.dump(metadata, f, indent=2)

                # Save scalers
                scalers = {
                    "price_scaler": self.price_scaler,
                    "feature_scaler": self.feature_scaler,
                    "volatility_scaler": self.volatility_scaler,
                }

                with open(filepath.replace(".h5", "_scalers.pkl"), "wb") as f:
                    pickle.dump(scalers, f)
            else:
                # Save simplified model
                model_data = {
                    "training_data": self.training_data,
                    "simple_model": self.simple_model,
                    "symbol": self.symbol,
                    "is_trained": self.is_trained,
                    "prediction_accuracy": self.prediction_accuracy,
                }

                with open(filepath.replace(".h5", ".pkl"), "wb") as f:
                    pickle.dump(model_data, f)

            self.logger.info(f"Model saved to {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")

    def load_model(self, filepath: str):
        """Load trained model"""
        try:
            if TF_AVAILABLE and filepath.endswith(".h5"):
                self.model = load_model(filepath)

                # Load metadata
                metadata_path = filepath.replace(".h5", "_metadata.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)

                    self.is_trained = metadata.get("is_trained", False)
                    self.prediction_accuracy = metadata.get("prediction_accuracy", 0.0)
                    self.directional_accuracy = metadata.get(
                        "directional_accuracy", 0.0
                    )

                # Load scalers
                scalers_path = filepath.replace(".h5", "_scalers.pkl")
                if os.path.exists(scalers_path):
                    with open(scalers_path, "rb") as f:
                        scalers = pickle.load(f)

                    self.price_scaler = scalers["price_scaler"]
                    self.feature_scaler = scalers["feature_scaler"]
                    self.volatility_scaler = scalers["volatility_scaler"]
            else:
                # Load simplified model
                pkl_path = filepath.replace(".h5", ".pkl")
                with open(pkl_path, "rb") as f:
                    model_data = pickle.load(f)

                self.training_data = model_data.get("training_data", {})
                self.simple_model = model_data.get("simple_model", {})
                self.is_trained = model_data.get("is_trained", False)
                self.prediction_accuracy = model_data.get("prediction_accuracy", 0.0)

            self.logger.info(f"Model loaded from {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and performance"""
        return {
            "symbol": self.symbol,
            "model_type": "LSTM_TF" if TF_AVAILABLE and self.model else "LSTM_Simple",
            "sequence_length": self.sequence_length,
            "prediction_horizon": self.prediction_horizon,
            "is_trained": self.is_trained,
            "last_training_time": (
                self.last_training_time.isoformat() if self.last_training_time else None
            ),
            "prediction_accuracy": round(self.prediction_accuracy, 4),
            "directional_accuracy": round(self.directional_accuracy, 4),
            "recent_predictions_count": len(self.recent_predictions),
            "model_version": self.model_version,
            "tensorflow_available": TF_AVAILABLE,
        }


# Usage Example
if __name__ == "__main__":
    import os

    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Initialize LSTM model
    lstm = LSTMTimeSeriesPredictor("EURUSD", sequence_length=60)

    # Example market data (normally from MT5)
    dates = pd.date_range(start="2023-01-01", periods=1000, freq="H")
    np.random.seed(42)
    prices = 1.1000 + np.cumsum(np.random.randn(1000) * 0.0001)

    market_data = pd.DataFrame(
        {
            "datetime": dates,
            "close": prices,
            "volume": np.random.randint(1000, 5000, 1000),
        }
    )

    print("LSTM Model Info:", lstm.get_model_info())

    # Train model
    if len(market_data) >= 100:
        training_result = lstm.train(market_data)
        print("Training Result:", training_result)

        # Make prediction
        recent_data = market_data.tail(lstm.sequence_length)
        prediction = lstm.predict(recent_data)
        print("Prediction:", prediction)

        # Save model
        lstm.save_model("models/lstm_eurusd.h5")
