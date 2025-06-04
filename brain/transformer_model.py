"""
🧠 Transformer Model for Forex Trading
Advanced attention-based model for time series prediction

Features:
- Multi-head attention mechanism
- Positional encoding for time series
- Layer normalization and dropout
- Forex-specific feature processing
- Real-time prediction capabilities
- Model checkpointing and recovery
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
import json
import pickle
import warnings

warnings.filterwarnings("ignore")

# Try to import TensorFlow, fallback if not available
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import (
        Input,
        Dense,
        Dropout,
        LayerNormalization,
        MultiHeadAttention,
        GlobalAveragePooling1D,
        Embedding,
        Add,
        Lambda,
    )
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import (
        EarlyStopping,
        ModelCheckpoint,
        ReduceLROnPlateau,
    )
    from tensorflow.keras.regularizers import l2

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Custom imports
from config.model_config import MODEL_CONFIG
from utils.logger_config import setup_logger


@dataclass
class TransformerConfig:
    """Configuration for Transformer model"""

    # Model architecture
    sequence_length: int = 60
    n_features: int = 10
    d_model: int = 64
    num_heads: int = 4
    num_layers: int = 2
    dff: int = 256  # Feed forward dimension

    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    validation_split: float = 0.2

    # Regularization
    dropout_rate: float = 0.1
    l2_reg: float = 0.01

    # Early stopping
    patience: int = 15
    min_delta: float = 0.0001

    # Output
    output_dim: int = 1
    activation: str = "tanh"


class PositionalEncoding(tf.keras.layers.Layer):
    """Positional encoding for transformer"""

    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            np.arange(position)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model,
        )

        # Apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # Apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, x):
        return x + self.pos_encoding[:, : tf.shape(x)[1], :]


class TransformerBlock(tf.keras.layers.Layer):
    """Transformer block with multi-head attention"""

    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate

        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([Dense(dff, activation="relu"), Dense(d_model)])

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, x, training):
        attn_output = self.mha(x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2


class ForexTransformer:
    """
    Transformer model specifically designed for Forex prediction
    """

    def __init__(
        self,
        config: TransformerConfig = None,
        symbol: str = "EURUSD",
        model_dir: str = "models",
    ):
        """
        Initialize Forex Transformer

        Args:
            config: Model configuration
            symbol: Trading symbol
            model_dir: Directory to save models
        """
        self.config = config or TransformerConfig()
        self.symbol = symbol
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.logger = setup_logger(
            f"TransformerModel_{symbol}", f"logs/transformer_{symbol}.log"
        )

        # Model components
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.is_trained = False
        self.training_history = None

        # Performance tracking
        self.training_metrics = {}
        self.prediction_history = []

        # Initialize model if TensorFlow is available
        if TF_AVAILABLE:
            self._build_model()
        else:
            self.logger.warning("TensorFlow not available, using placeholder model")

        self.logger.info(f"ForexTransformer initialized for {symbol}")

    def _build_model(self):
        """Build the transformer model architecture"""
        try:
            # Input layer
            inputs = Input(shape=(self.config.sequence_length, self.config.n_features))

            # Initial projection to d_model dimensions
            x = Dense(self.config.d_model)(inputs)

            # Positional encoding
            pos_encoding = PositionalEncoding(
                self.config.sequence_length, self.config.d_model
            )
            x = pos_encoding(x)

            # Transformer blocks
            for _ in range(self.config.num_layers):
                x = TransformerBlock(
                    d_model=self.config.d_model,
                    num_heads=self.config.num_heads,
                    dff=self.config.dff,
                    dropout_rate=self.config.dropout_rate,
                )(x)

            # Global average pooling
            x = GlobalAveragePooling1D()(x)

            # Final dense layers
            x = Dense(
                128, activation="relu", kernel_regularizer=l2(self.config.l2_reg)
            )(x)
            x = Dropout(self.config.dropout_rate)(x)

            x = Dense(64, activation="relu", kernel_regularizer=l2(self.config.l2_reg))(
                x
            )
            x = Dropout(self.config.dropout_rate)(x)

            # Output layer
            outputs = Dense(self.config.output_dim, activation=self.config.activation)(
                x
            )

            # Create model
            self.model = Model(inputs=inputs, outputs=outputs)

            # Compile model
            self.model.compile(
                optimizer=Adam(learning_rate=self.config.learning_rate),
                loss="mse",
                metrics=["mae", "mape"],
            )

            self.logger.info("Transformer model built successfully")
            self.logger.info(f"Model parameters: {self.model.count_params()}")

        except Exception as e:
            self.logger.error(f"Failed to build transformer model: {e}")
            raise

    def prepare_data(
        self, features: pd.DataFrame, targets: pd.Series = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for transformer training/prediction

        Args:
            features: Feature DataFrame
            targets: Target series (optional)

        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        try:
            # Convert to numpy
            X = features.values
            y = targets.values if targets is not None else None

            # Create sequences
            X_sequences = []
            y_sequences = []

            for i in range(self.config.sequence_length, len(X)):
                X_sequences.append(X[i - self.config.sequence_length : i])

                if y is not None:
                    y_sequences.append(y[i])

            X_sequences = np.array(X_sequences)
            y_sequences = np.array(y_sequences) if y is not None else None

            self.logger.debug(f"Prepared {len(X_sequences)} sequences")

            return X_sequences, y_sequences

        except Exception as e:
            self.logger.error(f"Failed to prepare data: {e}")
            raise

    def train(
        self,
        features: pd.DataFrame,
        targets: pd.Series,
        validation_data: Tuple = None,
        save_model: bool = True,
    ) -> Dict[str, Any]:
        """
        Train the transformer model

        Args:
            features: Training features
            targets: Training targets
            validation_data: Optional validation data
            save_model: Whether to save the trained model

        Returns:
            Training history and metrics
        """
        try:
            if not TF_AVAILABLE:
                self.logger.warning("TensorFlow not available, skipping training")
                return {"status": "skipped", "reason": "TensorFlow not available"}

            self.logger.info("Starting transformer training...")

            # Prepare data
            X_train, y_train = self.prepare_data(features, targets)

            if X_train is None or len(X_train) == 0:
                raise ValueError("No training data available")

            # Prepare validation data
            X_val, y_val = None, None
            if validation_data is not None:
                X_val, y_val = validation_data
            elif self.config.validation_split > 0:
                split_idx = int(len(X_train) * (1 - self.config.validation_split))
                X_val = X_train[split_idx:]
                y_val = y_train[split_idx:]
                X_train = X_train[:split_idx]
                y_train = y_train[:split_idx]

            # Setup callbacks
            callbacks = self._setup_callbacks()

            # Train model
            history = self.model.fit(
                X_train,
                y_train,
                batch_size=self.config.batch_size,
                epochs=self.config.epochs,
                validation_data=(X_val, y_val) if X_val is not None else None,
                callbacks=callbacks,
                verbose=1,
            )

            self.training_history = history.history
            self.is_trained = True

            # Calculate training metrics
            train_loss = min(history.history["loss"])
            val_loss = min(history.history.get("val_loss", [float("inf")]))

            self.training_metrics = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "epochs_trained": len(history.history["loss"]),
                "best_epoch": np.argmin(
                    history.history.get("val_loss", history.history["loss"])
                )
                + 1,
                "training_time": datetime.now().isoformat(),
            }

            # Save model
            if save_model:
                self.save_model()

            self.logger.info(
                f"Training completed. Train loss: {train_loss:.6f}, Val loss: {val_loss:.6f}"
            )

            return {
                "status": "success",
                "metrics": self.training_metrics,
                "history": self.training_history,
            }

        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            return {"status": "failed", "error": str(e)}

    def predict(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Make prediction using the transformer model

        Args:
            features: Feature DataFrame

        Returns:
            Prediction results
        """
        try:
            if not TF_AVAILABLE or self.model is None:
                # Fallback prediction
                return self._fallback_prediction(features)

            # Prepare data
            X_sequences, _ = self.prepare_data(features)

            if X_sequences is None or len(X_sequences) == 0:
                return {
                    "prediction": 0.0,
                    "confidence": 0.0,
                    "error": "No data for prediction",
                }

            # Make prediction
            prediction = self.model.predict(X_sequences[-1:], verbose=0)

            # Get prediction value
            pred_value = float(prediction[0][0])

            # Calculate confidence based on recent performance
            confidence = self._calculate_prediction_confidence()

            # Store prediction history
            self.prediction_history.append(
                {
                    "timestamp": datetime.now(),
                    "prediction": pred_value,
                    "confidence": confidence,
                }
            )

            # Keep only recent predictions
            if len(self.prediction_history) > 1000:
                self.prediction_history = self.prediction_history[-1000:]

            result = {
                "prediction": pred_value,
                "confidence": confidence,
                "model_type": "transformer",
                "timestamp": datetime.now().isoformat(),
            }

            self.logger.debug(
                f"Prediction: {pred_value:.6f}, Confidence: {confidence:.3f}"
            )

            return result

        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return self._fallback_prediction(features)

    def _fallback_prediction(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Fallback prediction when TensorFlow model is not available"""
        try:
            # Simple heuristic based on recent price movements
            if len(features) > 0:
                # Use last few values to create a simple prediction
                last_values = features.iloc[-min(5, len(features)) :].mean()
                prediction = float(
                    np.tanh(last_values.mean()) * 0.1
                )  # Small bounded prediction
            else:
                prediction = 0.0

            return {
                "prediction": prediction,
                "confidence": 0.3,  # Low confidence for fallback
                "model_type": "fallback",
                "timestamp": datetime.now().isoformat(),
            }

        except:
            return {
                "prediction": 0.0,
                "confidence": 0.0,
                "model_type": "fallback",
                "timestamp": datetime.now().isoformat(),
            }

    def _calculate_prediction_confidence(self) -> float:
        """Calculate prediction confidence based on model performance"""
        try:
            if not self.is_trained or not self.training_metrics:
                return 0.5

            # Base confidence on validation loss
            val_loss = self.training_metrics.get("val_loss", 1.0)
            base_confidence = 1.0 / (1.0 + val_loss)

            # Adjust based on recent prediction consistency
            if len(self.prediction_history) > 10:
                recent_preds = [p["prediction"] for p in self.prediction_history[-10:]]
                pred_std = np.std(recent_preds)
                consistency_factor = 1.0 / (1.0 + pred_std)
                confidence = base_confidence * 0.7 + consistency_factor * 0.3
            else:
                confidence = base_confidence

            return min(1.0, max(0.0, confidence))

        except:
            return 0.5

    def _setup_callbacks(self) -> List:
        """Setup training callbacks"""
        callbacks = []

        # Early stopping
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=self.config.patience,
            min_delta=self.config.min_delta,
            restore_best_weights=True,
            verbose=1,
        )
        callbacks.append(early_stopping)

        # Model checkpoint
        checkpoint_path = self.model_dir / f"transformer_{self.symbol}_best.h5"
        model_checkpoint = ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        )
        callbacks.append(model_checkpoint)

        # Reduce learning rate
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=self.config.patience // 2,
            min_lr=self.config.learning_rate * 0.01,
            verbose=1,
        )
        callbacks.append(reduce_lr)

        return callbacks

    def save_model(self, filepath: str = None):
        """Save the trained model"""
        try:
            if not TF_AVAILABLE or self.model is None:
                self.logger.warning("No model to save")
                return

            if filepath is None:
                filepath = self.model_dir / f"transformer_{self.symbol}.h5"

            # Save model
            self.model.save(filepath)

            # Save configuration and metrics
            config_path = str(filepath).replace(".h5", "_config.json")
            config_data = {
                "config": {
                    "sequence_length": self.config.sequence_length,
                    "n_features": self.config.n_features,
                    "d_model": self.config.d_model,
                    "num_heads": self.config.num_heads,
                    "num_layers": self.config.num_layers,
                    "dff": self.config.dff,
                    "dropout_rate": self.config.dropout_rate,
                    "learning_rate": self.config.learning_rate,
                },
                "training_metrics": self.training_metrics,
                "symbol": self.symbol,
                "is_trained": self.is_trained,
                "saved_at": datetime.now().isoformat(),
            }

            with open(config_path, "w") as f:
                json.dump(config_data, f, indent=2)

            self.logger.info(f"Model saved to {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")

    def load_model(self, filepath: str = None):
        """Load a trained model"""
        try:
            if not TF_AVAILABLE:
                self.logger.warning("TensorFlow not available, cannot load model")
                return False

            if filepath is None:
                filepath = self.model_dir / f"transformer_{self.symbol}.h5"

            if not Path(filepath).exists():
                self.logger.warning(f"Model file not found: {filepath}")
                return False

            # Load model
            self.model = tf.keras.models.load_model(
                filepath,
                custom_objects={
                    "TransformerBlock": TransformerBlock,
                    "PositionalEncoding": PositionalEncoding,
                },
            )

            # Load configuration
            config_path = str(filepath).replace(".h5", "_config.json")
            if Path(config_path).exists():
                with open(config_path, "r") as f:
                    config_data = json.load(f)

                self.training_metrics = config_data.get("training_metrics", {})
                self.is_trained = config_data.get("is_trained", False)

            self.logger.info(f"Model loaded from {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and status"""
        info = {
            "symbol": self.symbol,
            "model_type": "transformer",
            "is_trained": self.is_trained,
            "tensorflow_available": TF_AVAILABLE,
            "config": {
                "sequence_length": self.config.sequence_length,
                "n_features": self.config.n_features,
                "d_model": self.config.d_model,
                "num_heads": self.config.num_heads,
                "num_layers": self.config.num_layers,
            },
            "training_metrics": self.training_metrics,
            "predictions_made": len(self.prediction_history),
        }

        if TF_AVAILABLE and self.model is not None:
            info["parameters"] = self.model.count_params()

        return info

    def update_performance(
        self, actual_result: float, prediction_timestamp: datetime = None
    ):
        """Update model performance with actual results"""
        try:
            # Find corresponding prediction
            target_prediction = None

            if prediction_timestamp:
                # Find prediction closest to timestamp
                for pred in reversed(self.prediction_history):
                    pred_time = pred["timestamp"]
                    if isinstance(pred_time, str):
                        pred_time = datetime.fromisoformat(pred_time)

                    if (
                        abs((pred_time - prediction_timestamp).total_seconds()) < 3600
                    ):  # Within 1 hour
                        target_prediction = pred
                        break
            else:
                # Use most recent prediction
                target_prediction = (
                    self.prediction_history[-1] if self.prediction_history else None
                )

            if target_prediction:
                # Calculate error
                prediction_error = abs(target_prediction["prediction"] - actual_result)
                target_prediction["actual_result"] = actual_result
                target_prediction["error"] = prediction_error

                self.logger.debug(
                    f"Updated prediction performance: predicted {target_prediction['prediction']:.6f}, actual {actual_result:.6f}, error {prediction_error:.6f}"
                )

        except Exception as e:
            self.logger.error(f"Failed to update performance: {e}")


# Alias for backward compatibility
TransformerForexModel = ForexTransformer


# Factory function for easy model creation
def create_transformer_model(
    symbol: str = "EURUSD", config: TransformerConfig = None
) -> ForexTransformer:
    """
    Factory function to create a transformer model

    Args:
        symbol: Trading symbol
        config: Model configuration

    Returns:
        ForexTransformer instance
    """
    return ForexTransformer(config=config, symbol=symbol)


# Export all important classes and functions
__all__ = [
    "TransformerConfig",
    "ForexTransformer",
    "TransformerForexModel",  # Alias
    "create_transformer_model",
    "PositionalEncoding",
    "TransformerBlock",
]


# Usage example
if __name__ == "__main__":
    # Test transformer model
    config = TransformerConfig(
        sequence_length=60,
        n_features=10,
        d_model=64,
        num_heads=4,
        num_layers=2,
        epochs=5,  # Small for testing
    )

    # Create model
    transformer = ForexTransformer(config=config, symbol="EURUSD")

    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    features = pd.DataFrame(
        np.random.randn(n_samples, 10), columns=[f"feature_{i}" for i in range(10)]
    )

    # Create synthetic targets (price movements)
    targets = pd.Series(np.random.randn(n_samples) * 0.01)

    # Test training (if TensorFlow available)
    if TF_AVAILABLE:
        print("Testing transformer training...")
        result = transformer.train(features, targets)
        print(f"Training result: {result['status']}")

        if result["status"] == "success":
            print(f"Training metrics: {result['metrics']}")

    # Test prediction
    print("Testing prediction...")
    prediction = transformer.predict(features.tail(100))
    print(f"Prediction: {prediction}")

    # Get model info
    info = transformer.get_model_info()
    print(f"Model info: {info}")

    print("Transformer model test completed!")
