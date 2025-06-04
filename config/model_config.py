"""
AI Model Configuration for Institutional Forex Trading System
Supports Deep Learning + Active Learning Architecture
Author: Senior AI Developer
Version: 1.0.0 - Production Ready
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import torch
import numpy as np


class ModelType(Enum):
    """Supported AI Model Types"""

    LSTM = "lstm"
    TRANSFORMER = "transformer"
    CNN_LSTM = "cnn_lstm"
    ENSEMBLE = "ensemble"
    XGB = "xgboost"
    LIGHTGBM = "lightgbm"


class PredictionMode(Enum):
    """Model Prediction Modes"""

    CLASSIFICATION = "classification"  # Buy/Sell/Hold signals
    REGRESSION = "regression"  # Price prediction
    MULTI_OUTPUT = "multi_output"  # Both price & signal


class TrainingMode(Enum):
    """Model Training Strategies"""

    BATCH = "batch"  # Traditional batch training
    ONLINE = "online"  # Online learning
    ACTIVE = "active"  # Active learning with uncertainty
    INCREMENTAL = "incremental"  # Incremental learning


@dataclass
class ModelArchitecture:
    """Neural Network Architecture Configuration"""

    # LSTM Configuration
    lstm_layers: int = 3
    lstm_units: List[int] = field(default_factory=lambda: [256, 128, 64])
    lstm_dropout: float = 0.2
    lstm_bidirectional: bool = True

    # Transformer Configuration
    transformer_layers: int = 8
    attention_heads: int = 16
    embed_dim: int = 512
    ff_dim: int = 2048
    transformer_dropout: float = 0.1

    # CNN Configuration (for CNN-LSTM hybrid)
    cnn_filters: List[int] = field(default_factory=lambda: [64, 128, 256])
    cnn_kernel_sizes: List[int] = field(default_factory=lambda: [3, 5, 7])
    cnn_pool_size: int = 2

    # Dense Layers
    dense_layers: List[int] = field(default_factory=lambda: [512, 256, 128])
    dense_dropout: float = 0.3

    # Output Configuration
    output_classes: int = 3  # Buy/Sell/Hold
    activation: str = "relu"
    output_activation: str = "softmax"


@dataclass
class TrainingConfig:
    """Model Training Configuration"""

    # Basic Training Parameters
    batch_size: int = 64
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-4

    # Learning Rate Scheduling
    lr_scheduler: str = "cosine"  # cosine, step, exponential
    lr_patience: int = 10
    lr_factor: float = 0.5

    # Early Stopping
    early_stopping: bool = True
    early_stopping_patience: int = 15
    early_stopping_delta: float = 0.001

    # Validation
    validation_split: float = 0.2
    cross_validation: bool = True
    cv_folds: int = 5

    # Data Augmentation
    use_augmentation: bool = True
    noise_factor: float = 0.01
    time_shift_range: int = 5

    # Active Learning
    uncertainty_threshold: float = 0.3
    active_learning_budget: int = 100
    query_strategy: str = "uncertainty"  # uncertainty, diversity, hybrid

    # Model Checkpointing
    save_best_only: bool = True
    save_weights_only: bool = False
    checkpoint_frequency: int = 10


@dataclass
class EnsembleConfig:
    """Ensemble Model Configuration"""

    models: List[ModelType] = field(
        default_factory=lambda: [ModelType.LSTM, ModelType.TRANSFORMER, ModelType.XGB]
    )

    # Voting Strategy
    voting_method: str = "soft"  # hard, soft, weighted
    weights: Optional[List[float]] = None

    # Stacking Configuration
    use_stacking: bool = True
    meta_model: str = "neural_network"  # linear, neural_network, xgb

    # Model Selection
    dynamic_selection: bool = True
    selection_metric: str = "sharpe_ratio"  # accuracy, sharpe_ratio, profit_factor

    # Confidence Thresholds
    min_confidence: float = 0.6
    ensemble_threshold: float = 0.7


@dataclass
class FeatureConfig:
    """Feature Engineering Configuration"""

    # Sequence Length
    sequence_length: int = 60  # lookback period
    prediction_horizon: int = 1  # forecast steps ahead

    # Technical Indicators
    use_technical_indicators: bool = True
    indicators: List[str] = field(
        default_factory=lambda: [
            "sma",
            "ema",
            "rsi",
            "macd",
            "bollinger",
            "stochastic",
            "atr",
            "adx",
            "cci",
            "williams_r",
            "momentum",
            "roc",
        ]
    )

    # Price Features
    price_features: List[str] = field(
        default_factory=lambda: [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "tick_volume",
        ]
    )

    # Derived Features
    use_price_ratios: bool = True
    use_volatility_features: bool = True
    use_volume_features: bool = True
    use_time_features: bool = True

    # Feature Scaling
    scaling_method: str = "robust"  # standard, minmax, robust
    scale_per_symbol: bool = True

    # Feature Selection
    feature_selection: bool = True
    selection_method: str = "mutual_info"  # correlation, mutual_info, pca
    max_features: Optional[int] = None


@dataclass
class ModelConfig:
    """Complete Model Configuration"""

    # Model Identification
    model_name: str = "InstitutionalForexAI"
    version: str = "1.0.0"

    # Primary Model Settings
    primary_model: ModelType = ModelType.ENSEMBLE
    prediction_mode: PredictionMode = PredictionMode.MULTI_OUTPUT
    training_mode: TrainingMode = TrainingMode.ACTIVE

    # Architecture & Training
    architecture: ModelArchitecture = field(default_factory=ModelArchitecture)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)

    # Model Paths
    model_dir: str = "models"
    weights_dir: str = "models/weights"
    logs_dir: str = "models/logs"
    cache_dir: str = "models/cache"

    # Device Configuration
    device: str = "auto"  # auto, cpu, cuda, mps
    mixed_precision: bool = True
    compile_model: bool = True  # PyTorch 2.0 compile

    # Production Settings
    inference_batch_size: int = 1
    max_memory_usage: float = 0.8  # 80% of available GPU memory
    model_warming: bool = True

    # Monitoring & Logging
    log_predictions: bool = True
    log_uncertainties: bool = True
    performance_tracking: bool = True

    def __post_init__(self):
        """Post-initialization validation and setup"""
        # Create directories
        for directory in [
            self.model_dir,
            self.weights_dir,
            self.logs_dir,
            self.cache_dir,
        ]:
            os.makedirs(directory, exist_ok=True)

        # Device auto-detection
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        # Validate ensemble configuration
        if self.primary_model == ModelType.ENSEMBLE:
            if not self.ensemble.models:
                raise ValueError("Ensemble models list cannot be empty")

            if self.ensemble.weights and len(self.ensemble.weights) != len(
                self.ensemble.models
            ):
                raise ValueError("Ensemble weights must match number of models")

        # Validate feature configuration
        if self.features.sequence_length <= 0:
            raise ValueError("Sequence length must be positive")

        if self.features.prediction_horizon <= 0:
            raise ValueError("Prediction horizon must be positive")

    @classmethod
    def from_symbol(cls, symbol: str) -> "ModelConfig":
        """Create symbol-specific configuration"""
        config = cls()
        config.model_name = f"ForexAI_{symbol}"
        config.model_dir = f"models/{symbol}"
        config.weights_dir = f"models/{symbol}/weights"
        config.logs_dir = f"models/{symbol}/logs"
        config.cache_dir = f"models/{symbol}/cache"

        # Symbol-specific optimizations
        major_pairs = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF"]
        if symbol in major_pairs:
            config.features.sequence_length = 120  # Longer history for major pairs
            config.training.batch_size = 128
        else:
            config.features.sequence_length = 60  # Standard for minor pairs
            config.training.batch_size = 64

        return config

    def get_model_path(self, model_type: Optional[ModelType] = None) -> str:
        """Get model file path"""
        model_type = model_type or self.primary_model
        return os.path.join(self.weights_dir, f"{model_type.value}_model.pth")

    def get_scaler_path(self) -> str:
        """Get feature scaler path"""
        return os.path.join(self.cache_dir, "feature_scaler.pkl")

    def get_feature_selector_path(self) -> str:
        """Get feature selector path"""
        return os.path.join(self.cache_dir, "feature_selector.pkl")


# Production-Ready Model Configurations
PRODUCTION_CONFIGS = {
    "conservative": ModelConfig(
        model_name="ConservativeForexAI",
        ensemble=EnsembleConfig(min_confidence=0.8, ensemble_threshold=0.85),
        training=TrainingConfig(early_stopping_patience=20, uncertainty_threshold=0.2),
    ),
    "aggressive": ModelConfig(
        model_name="AggressiveForexAI",
        ensemble=EnsembleConfig(min_confidence=0.6, ensemble_threshold=0.65),
        training=TrainingConfig(learning_rate=0.002, uncertainty_threshold=0.4),
    ),
    "scalping": ModelConfig(
        model_name="ScalpingForexAI",
        features=FeatureConfig(sequence_length=30, prediction_horizon=1),
        training=TrainingConfig(batch_size=128, learning_rate=0.003),
    ),
    "swing": ModelConfig(
        model_name="SwingForexAI",
        features=FeatureConfig(sequence_length=240, prediction_horizon=24),
        training=TrainingConfig(batch_size=32, epochs=200),
    ),
}

# Symbol-Specific Configurations
SYMBOL_CONFIGS = {
    "EURUSD": {"sequence_length": 120, "batch_size": 128, "learning_rate": 0.001},
    "GBPUSD": {"sequence_length": 100, "batch_size": 96, "learning_rate": 0.0012},
    "USDJPY": {"sequence_length": 80, "batch_size": 64, "learning_rate": 0.0015},
}


def get_model_config(
    config_type: str = "default", symbol: Optional[str] = None
) -> ModelConfig:
    """
    Factory function to get model configuration

    Args:
        config_type: Type of configuration (default, conservative, aggressive, scalping, swing)
        symbol: Trading symbol for symbol-specific optimizations

    Returns:
        ModelConfig: Configured model settings
    """
    if config_type in PRODUCTION_CONFIGS:
        config = PRODUCTION_CONFIGS[config_type]
    else:
        config = ModelConfig()

    # Apply symbol-specific optimizations
    if symbol and symbol in SYMBOL_CONFIGS:
        symbol_config = SYMBOL_CONFIGS[symbol]
        config.features.sequence_length = symbol_config["sequence_length"]
        config.training.batch_size = symbol_config["batch_size"]
        config.training.learning_rate = symbol_config["learning_rate"]
        config.model_name = f"{config.model_name}_{symbol}"

    return config


# Export main configuration
DEFAULT_MODEL_CONFIG = ModelConfig()

if __name__ == "__main__":
    # Configuration validation and testing
    print("🧠 AI Model Configuration System")
    print("=" * 50)

    # Test default configuration
    config = ModelConfig()
    print(f"✅ Default Config: {config.model_name}")
    print(f"📱 Device: {config.device}")
    print(f"🎯 Primary Model: {config.primary_model.value}")
    print(f"📊 Prediction Mode: {config.prediction_mode.value}")

    # Test symbol-specific configuration
    eurusd_config = ModelConfig.from_symbol("EURUSD")
    print(f"\n✅ EURUSD Config: {eurusd_config.model_name}")
    print(f"📈 Sequence Length: {eurusd_config.features.sequence_length}")

    # Test production configurations
    print(f"\n🏭 Production Configurations Available:")
    for name, cfg in PRODUCTION_CONFIGS.items():
        print(f"   - {name}: {cfg.model_name}")

    print(f"\n🎯 Configuration System Ready!")
    # เพิ่มโค้ดนี้ต่อท้ายไฟล์ config/model_config.py ของคุณ

# =============================================================================
# EXPORTS FOR BACKWARD COMPATIBILITY
# =============================================================================

# Export default model configuration as MODEL_CONFIG
MODEL_CONFIG = {
    # LSTM Model Configuration
    "lstm": {
        "sequence_length": DEFAULT_MODEL_CONFIG.features.sequence_length,
        "hidden_units": DEFAULT_MODEL_CONFIG.architecture.lstm_units,
        "dropout_rate": DEFAULT_MODEL_CONFIG.architecture.lstm_dropout,
        "learning_rate": DEFAULT_MODEL_CONFIG.training.learning_rate,
        "batch_size": DEFAULT_MODEL_CONFIG.training.batch_size,
        "epochs": DEFAULT_MODEL_CONFIG.training.epochs,
        "validation_split": DEFAULT_MODEL_CONFIG.training.validation_split,
        "bidirectional": DEFAULT_MODEL_CONFIG.architecture.lstm_bidirectional,
    },
    # Transformer Model Configuration
    "transformer": {
        "sequence_length": DEFAULT_MODEL_CONFIG.features.sequence_length,
        "d_model": DEFAULT_MODEL_CONFIG.architecture.embed_dim,
        "num_heads": DEFAULT_MODEL_CONFIG.architecture.attention_heads,
        "num_layers": DEFAULT_MODEL_CONFIG.architecture.transformer_layers,
        "dff": DEFAULT_MODEL_CONFIG.architecture.ff_dim,
        "dropout_rate": DEFAULT_MODEL_CONFIG.architecture.transformer_dropout,
        "learning_rate": DEFAULT_MODEL_CONFIG.training.learning_rate,
        "batch_size": DEFAULT_MODEL_CONFIG.training.batch_size,
        "epochs": DEFAULT_MODEL_CONFIG.training.epochs,
    },
    # CNN-LSTM Model Configuration
    "cnn_lstm": {
        "sequence_length": DEFAULT_MODEL_CONFIG.features.sequence_length,
        "cnn_filters": DEFAULT_MODEL_CONFIG.architecture.cnn_filters,
        "cnn_kernel_sizes": DEFAULT_MODEL_CONFIG.architecture.cnn_kernel_sizes,
        "lstm_units": DEFAULT_MODEL_CONFIG.architecture.lstm_units,
        "dropout_rate": DEFAULT_MODEL_CONFIG.architecture.lstm_dropout,
        "learning_rate": DEFAULT_MODEL_CONFIG.training.learning_rate,
        "batch_size": DEFAULT_MODEL_CONFIG.training.batch_size,
        "epochs": DEFAULT_MODEL_CONFIG.training.epochs,
    },
    # XGBoost Configuration
    "xgboost": {
        "n_estimators": 1000,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
    },
    # LightGBM Configuration
    "lightgbm": {
        "n_estimators": 1000,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
    },
    # General Settings
    "device": DEFAULT_MODEL_CONFIG.device,
    "mixed_precision": DEFAULT_MODEL_CONFIG.mixed_precision,
    "model_dir": DEFAULT_MODEL_CONFIG.model_dir,
    "weights_dir": DEFAULT_MODEL_CONFIG.weights_dir,
    "logs_dir": DEFAULT_MODEL_CONFIG.logs_dir,
}

# Export ensemble configuration as ENSEMBLE_CONFIG
ENSEMBLE_CONFIG = {
    "models": [model.value for model in DEFAULT_MODEL_CONFIG.ensemble.models],
    "voting_method": DEFAULT_MODEL_CONFIG.ensemble.voting_method,
    "weights": DEFAULT_MODEL_CONFIG.ensemble.weights,
    "use_stacking": DEFAULT_MODEL_CONFIG.ensemble.use_stacking,
    "meta_model": DEFAULT_MODEL_CONFIG.ensemble.meta_model,
    "dynamic_selection": DEFAULT_MODEL_CONFIG.ensemble.dynamic_selection,
    "selection_metric": DEFAULT_MODEL_CONFIG.ensemble.selection_metric,
    "min_confidence": DEFAULT_MODEL_CONFIG.ensemble.min_confidence,
    "ensemble_threshold": DEFAULT_MODEL_CONFIG.ensemble.ensemble_threshold,
}

# Feature Engineering Configuration Export
FEATURE_CONFIG = {
    "sequence_length": DEFAULT_MODEL_CONFIG.features.sequence_length,
    "prediction_horizon": DEFAULT_MODEL_CONFIG.features.prediction_horizon,
    "technical_indicators": DEFAULT_MODEL_CONFIG.features.indicators,
    "price_features": DEFAULT_MODEL_CONFIG.features.price_features,
    "scaling_method": DEFAULT_MODEL_CONFIG.features.scaling_method,
    "feature_selection": DEFAULT_MODEL_CONFIG.features.feature_selection,
    "selection_method": DEFAULT_MODEL_CONFIG.features.selection_method,
}

# Training Configuration Export
TRAINING_CONFIG = {
    "batch_size": DEFAULT_MODEL_CONFIG.training.batch_size,
    "epochs": DEFAULT_MODEL_CONFIG.training.epochs,
    "learning_rate": DEFAULT_MODEL_CONFIG.training.learning_rate,
    "early_stopping": DEFAULT_MODEL_CONFIG.training.early_stopping,
    "early_stopping_patience": DEFAULT_MODEL_CONFIG.training.early_stopping_patience,
    "validation_split": DEFAULT_MODEL_CONFIG.training.validation_split,
    "use_augmentation": DEFAULT_MODEL_CONFIG.training.use_augmentation,
    "uncertainty_threshold": DEFAULT_MODEL_CONFIG.training.uncertainty_threshold,
}

# Active Learning Configuration Export
ACTIVE_LEARNING_CONFIG = {
    "uncertainty_threshold": DEFAULT_MODEL_CONFIG.training.uncertainty_threshold,
    "active_learning_budget": DEFAULT_MODEL_CONFIG.training.active_learning_budget,
    "query_strategy": DEFAULT_MODEL_CONFIG.training.query_strategy,
}

# Model Paths Export
MODEL_PATHS = {
    "model_dir": DEFAULT_MODEL_CONFIG.model_dir,
    "weights_dir": DEFAULT_MODEL_CONFIG.weights_dir,
    "logs_dir": DEFAULT_MODEL_CONFIG.logs_dir,
    "cache_dir": DEFAULT_MODEL_CONFIG.cache_dir,
}
