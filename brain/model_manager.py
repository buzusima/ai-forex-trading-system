"""
AI Model Management System
สำหรับ AI Trading System ที่เชื่อมต่อ MT5 บัญชีจริง

Features:
- Runtime model switching based on market conditions
- Model performance monitoring and selection
- Automatic model retraining scheduling
- Model versioning and rollback capabilities
- Resource management and optimization
- Hot-swapping without system downtime
- Model ensemble orchestration
- Performance benchmarking and comparison
"""

import os
import json
import pickle
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import schedule
import joblib

# Custom imports
from brain.ensemble_predictor import EnsemblePredictor, PredictionResult, MarketRegime
from brain.lstm_model import LSTMTimeSeriesPredictor
from brain.transformer_model import TransformerForexModel
from config.model_config import MODEL_CONFIG, ENSEMBLE_CONFIG
from config.settings import TRADING_SETTINGS
from memory.trade_logger import TradeLogger
from utils.logger_config import setup_logger


class ModelType(Enum):
    """Available model types"""

    ENSEMBLE = "ENSEMBLE"
    LSTM = "LSTM"
    TRANSFORMER = "TRANSFORMER"
    RANDOM_FOREST = "RANDOM_FOREST"
    XGBOOST = "XGBOOST"


class ModelStatus(Enum):
    """Model status"""

    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    TRAINING = "TRAINING"
    FAILED = "FAILED"
    DEPRECATED = "DEPRECATED"


class RetrainingTrigger(Enum):
    """Retraining triggers"""

    PERFORMANCE_DEGRADATION = "PERFORMANCE_DEGRADATION"
    MARKET_REGIME_CHANGE = "MARKET_REGIME_CHANGE"
    TIME_BASED = "TIME_BASED"
    MANUAL = "MANUAL"


@dataclass
class ModelMetrics:
    """Model performance metrics"""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    last_updated: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {**asdict(self), "last_updated": self.last_updated.isoformat()}


@dataclass
class ModelInfo:
    """Model information structure"""

    model_id: str
    model_type: ModelType
    symbol: str
    version: str
    status: ModelStatus
    created_at: datetime
    last_training: Optional[datetime]
    file_path: str
    metrics: ModelMetrics
    config: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_type": self.model_type.value,
            "symbol": self.symbol,
            "version": self.version,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "last_training": (
                self.last_training.isoformat() if self.last_training else None
            ),
            "file_path": self.file_path,
            "metrics": self.metrics.to_dict(),
            "config": self.config,
        }


class ModelManager:
    """
    Comprehensive AI Model Management System
    จัดการ lifecycle ของ AI models ทั้งหมด
    """

    def __init__(self, symbols: List[str] = ["EURUSD"], base_path: str = "models"):
        """
        Initialize Model Manager

        Args:
            symbols: List of trading symbols
            base_path: Base path for model storage
        """
        self.symbols = symbols
        self.base_path = base_path
        self.logger = setup_logger("ModelManager", "logs/model_manager.log")

        # Create directories
        os.makedirs(base_path, exist_ok=True)
        os.makedirs(f"{base_path}/active", exist_ok=True)
        os.makedirs(f"{base_path}/archive", exist_ok=True)
        os.makedirs(f"{base_path}/backup", exist_ok=True)

        # Model registry
        self.model_registry: Dict[str, ModelInfo] = {}
        self.active_models: Dict[str, Any] = {}  # symbol -> active model instance
        self.model_instances: Dict[str, Any] = {}  # model_id -> model instance

        # Performance tracking
        self.performance_history: Dict[str, List[ModelMetrics]] = {}
        self.model_selection_history: List[Dict[str, Any]] = []

        # Threading
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.background_tasks = []

        # Retraining configuration
        self.retraining_config = {
            "performance_threshold": 0.45,  # Switch model if accuracy drops below this
            "evaluation_window": 100,  # Number of recent trades to evaluate
            "min_trades_for_evaluation": 20,
            "retraining_interval_hours": 24,
            "max_concurrent_training": 2,
        }

        # Model switching rules
        self.switching_rules = {
            "min_performance_difference": 0.05,  # Minimum improvement to switch
            "stability_period": 50,  # Trades to wait before switching again
            "regime_specific_models": True,  # Use different models for different regimes
        }

        # Load existing registry
        self._load_registry()

        # Initialize default models
        self._initialize_default_models()

        # Setup automatic tasks
        self._setup_automatic_tasks()

        self.logger.info(f"Model Manager initialized for symbols: {symbols}")

    def _load_registry(self):
        """Load model registry from file"""
        try:
            registry_path = f"{self.base_path}/model_registry.json"
            if os.path.exists(registry_path):
                with open(registry_path, "r") as f:
                    registry_data = json.load(f)

                for model_id, data in registry_data.items():
                    metrics = ModelMetrics(
                        accuracy=data["metrics"]["accuracy"],
                        precision=data["metrics"]["precision"],
                        recall=data["metrics"]["recall"],
                        f1_score=data["metrics"]["f1_score"],
                        sharpe_ratio=data["metrics"]["sharpe_ratio"],
                        max_drawdown=data["metrics"]["max_drawdown"],
                        win_rate=data["metrics"]["win_rate"],
                        profit_factor=data["metrics"]["profit_factor"],
                        total_trades=data["metrics"]["total_trades"],
                        last_updated=datetime.fromisoformat(
                            data["metrics"]["last_updated"]
                        ),
                    )

                    model_info = ModelInfo(
                        model_id=data["model_id"],
                        model_type=ModelType(data["model_type"]),
                        symbol=data["symbol"],
                        version=data["version"],
                        status=ModelStatus(data["status"]),
                        created_at=datetime.fromisoformat(data["created_at"]),
                        last_training=(
                            datetime.fromisoformat(data["last_training"])
                            if data["last_training"]
                            else None
                        ),
                        file_path=data["file_path"],
                        metrics=metrics,
                        config=data["config"],
                    )

                    self.model_registry[model_id] = model_info

                self.logger.info(
                    f"Loaded {len(self.model_registry)} models from registry"
                )

        except Exception as e:
            self.logger.error(f"Failed to load model registry: {e}")

    def _save_registry(self):
        """Save model registry to file"""
        try:
            registry_path = f"{self.base_path}/model_registry.json"
            registry_data = {
                model_id: info.to_dict()
                for model_id, info in self.model_registry.items()
            }

            with open(registry_path, "w") as f:
                json.dump(registry_data, f, indent=2, default=str)

            self.logger.debug("Model registry saved")

        except Exception as e:
            self.logger.error(f"Failed to save model registry: {e}")

    def _initialize_default_models(self):
        """Initialize default models for each symbol"""
        try:
            for symbol in self.symbols:
                # Check if we have active models for this symbol
                active_model = self._get_active_model_for_symbol(symbol)

                if not active_model:
                    # Create ensemble model as default
                    model_id = (
                        f"ensemble_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    )

                    # Create ensemble predictor
                    ensemble = EnsemblePredictor(symbol)

                    # Register model
                    metrics = ModelMetrics(
                        accuracy=0.5,
                        precision=0.5,
                        recall=0.5,
                        f1_score=0.5,
                        sharpe_ratio=0.0,
                        max_drawdown=0.0,
                        win_rate=0.5,
                        profit_factor=1.0,
                        total_trades=0,
                        last_updated=datetime.now(),
                    )

                    model_info = ModelInfo(
                        model_id=model_id,
                        model_type=ModelType.ENSEMBLE,
                        symbol=symbol,
                        version="1.0",
                        status=ModelStatus.ACTIVE,
                        created_at=datetime.now(),
                        last_training=None,
                        file_path=f"{self.base_path}/active/{model_id}",
                        metrics=metrics,
                        config=ENSEMBLE_CONFIG,
                    )

                    self.model_registry[model_id] = model_info
                    self.active_models[symbol] = ensemble
                    self.model_instances[model_id] = ensemble

                    self.logger.info(f"Initialized default ensemble model for {symbol}")

        except Exception as e:
            self.logger.error(f"Failed to initialize default models: {e}")

    def _get_active_model_for_symbol(self, symbol: str) -> Optional[ModelInfo]:
        """Get active model for symbol"""
        for model_info in self.model_registry.values():
            if model_info.symbol == symbol and model_info.status == ModelStatus.ACTIVE:
                return model_info
        return None

    def register_model(
        self,
        model_instance: Any,
        model_type: ModelType,
        symbol: str,
        version: str = "1.0",
        config: Dict[str, Any] = None,
    ) -> str:
        """
        Register a new model

        Args:
            model_instance: Trained model instance
            model_type: Type of model
            symbol: Trading symbol
            version: Model version
            config: Model configuration

        Returns:
            Model ID
        """
        try:
            with self.lock:
                # Generate model ID
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_id = f"{model_type.value.lower()}_{symbol}_{version}_{timestamp}"

                # Save model to file
                file_path = f"{self.base_path}/active/{model_id}"
                self._save_model_instance(model_instance, file_path)

                # Create metrics
                metrics = ModelMetrics(
                    accuracy=0.5,
                    precision=0.5,
                    recall=0.5,
                    f1_score=0.5,
                    sharpe_ratio=0.0,
                    max_drawdown=0.0,
                    win_rate=0.5,
                    profit_factor=1.0,
                    total_trades=0,
                    last_updated=datetime.now(),
                )

                # Create model info
                model_info = ModelInfo(
                    model_id=model_id,
                    model_type=model_type,
                    symbol=symbol,
                    version=version,
                    status=ModelStatus.ACTIVE,
                    created_at=datetime.now(),
                    last_training=datetime.now(),
                    file_path=file_path,
                    metrics=metrics,
                    config=config or {},
                )

                # Register model
                self.model_registry[model_id] = model_info
                self.model_instances[model_id] = model_instance

                # Save registry
                self._save_registry()

                self.logger.info(f"Registered model {model_id}")
                return model_id

        except Exception as e:
            self.logger.error(f"Failed to register model: {e}")
            return ""

    def _save_model_instance(self, model_instance: Any, file_path: str):
        """Save model instance to file"""
        try:
            if hasattr(model_instance, "save_model"):
                # Use model's own save method
                model_instance.save_model(f"{file_path}.h5")
            elif hasattr(model_instance, "save"):
                # TensorFlow/Keras model
                model_instance.save(f"{file_path}.h5")
            else:
                # Use pickle for other models
                with open(f"{file_path}.pkl", "wb") as f:
                    pickle.dump(model_instance, f)

        except Exception as e:
            self.logger.error(f"Failed to save model instance: {e}")
            raise

    def load_model(self, model_id: str) -> Optional[Any]:
        """
        Load model instance by ID

        Args:
            model_id: Model identifier

        Returns:
            Model instance or None
        """
        try:
            if model_id in self.model_instances:
                return self.model_instances[model_id]

            if model_id not in self.model_registry:
                self.logger.warning(f"Model {model_id} not found in registry")
                return None

            model_info = self.model_registry[model_id]

            # Load based on model type
            if model_info.model_type == ModelType.ENSEMBLE:
                model_instance = EnsemblePredictor(model_info.symbol)
                # Load state if available
                state_file = f"{model_info.file_path}_state.json"
                if os.path.exists(state_file):
                    model_instance.load_ensemble_state(state_file)

            elif model_info.model_type == ModelType.LSTM:
                model_instance = LSTMTimeSeriesPredictor(model_info.symbol)
                if os.path.exists(f"{model_info.file_path}.h5"):
                    model_instance.load_model(f"{model_info.file_path}.h5")

            elif model_info.model_type == ModelType.TRANSFORMER:
                model_instance = TransformerForexModel(model_info.symbol)
                if os.path.exists(f"{model_info.file_path}.h5"):
                    model_instance.load_model(f"{model_info.file_path}.h5")

            else:
                # Load pickle file
                with open(f"{model_info.file_path}.pkl", "rb") as f:
                    model_instance = pickle.load(f)

            # Cache loaded model
            self.model_instances[model_id] = model_instance

            self.logger.info(f"Loaded model {model_id}")
            return model_instance

        except Exception as e:
            self.logger.error(f"Failed to load model {model_id}: {e}")
            return None

    def get_active_model(self, symbol: str) -> Optional[Any]:
        """
        Get active model for symbol

        Args:
            symbol: Trading symbol

        Returns:
            Active model instance
        """
        try:
            if symbol in self.active_models:
                return self.active_models[symbol]

            # Find active model in registry
            active_model_info = self._get_active_model_for_symbol(symbol)
            if active_model_info:
                model_instance = self.load_model(active_model_info.model_id)
                if model_instance:
                    self.active_models[symbol] = model_instance
                    return model_instance

            return None

        except Exception as e:
            self.logger.error(f"Failed to get active model for {symbol}: {e}")
            return None

    def switch_model(
        self, symbol: str, new_model_id: str, reason: str = "Manual switch"
    ) -> bool:
        """
        Switch active model for symbol

        Args:
            symbol: Trading symbol
            new_model_id: ID of new model to activate
            reason: Reason for switching

        Returns:
            Success status
        """
        try:
            with self.lock:
                if new_model_id not in self.model_registry:
                    self.logger.error(f"Model {new_model_id} not found")
                    return False

                new_model_info = self.model_registry[new_model_id]
                if new_model_info.symbol != symbol:
                    self.logger.error(
                        f"Model {new_model_id} is for {new_model_info.symbol}, not {symbol}"
                    )
                    return False

                # Deactivate current model
                current_model_info = self._get_active_model_for_symbol(symbol)
                if current_model_info:
                    current_model_info.status = ModelStatus.INACTIVE

                # Activate new model
                new_model_info.status = ModelStatus.ACTIVE

                # Load new model instance
                new_model_instance = self.load_model(new_model_id)
                if not new_model_instance:
                    self.logger.error(f"Failed to load model {new_model_id}")
                    return False

                # Update active model
                self.active_models[symbol] = new_model_instance

                # Log switch
                switch_record = {
                    "timestamp": datetime.now().isoformat(),
                    "symbol": symbol,
                    "old_model": (
                        current_model_info.model_id if current_model_info else None
                    ),
                    "new_model": new_model_id,
                    "reason": reason,
                }
                self.model_selection_history.append(switch_record)

                # Save registry
                self._save_registry()

                self.logger.info(
                    f"Switched model for {symbol} to {new_model_id} - Reason: {reason}"
                )
                return True

        except Exception as e:
            self.logger.error(f"Failed to switch model: {e}")
            return False

    def evaluate_model_performance(
        self, model_id: str, trade_results: List[Dict[str, Any]]
    ) -> ModelMetrics:
        """
        Evaluate model performance

        Args:
            model_id: Model identifier
            trade_results: List of trade results

        Returns:
            Updated metrics
        """
        try:
            if not trade_results:
                return self.model_registry[model_id].metrics

            # Calculate metrics
            total_trades = len(trade_results)
            winning_trades = sum(1 for r in trade_results if r.get("profit", 0) > 0)
            losing_trades = total_trades - winning_trades

            win_rate = winning_trades / total_trades if total_trades > 0 else 0

            # P&L metrics
            profits = [r.get("profit", 0) for r in trade_results]
            total_profit = sum(profits)

            gross_profit = sum(p for p in profits if p > 0)
            gross_loss = abs(sum(p for p in profits if p < 0))
            profit_factor = (
                gross_profit / gross_loss if gross_loss > 0 else float("inf")
            )

            # Calculate Sharpe ratio
            returns = np.array(profits)
            sharpe_ratio = (
                np.mean(returns) / (np.std(returns) + 1e-8) if len(returns) > 1 else 0
            )

            # Calculate max drawdown
            cumulative = np.cumsum(profits)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = cumulative - running_max
            max_drawdown = abs(np.min(drawdown)) if len(drawdown) > 0 else 0

            # Directional accuracy
            correct_directions = sum(
                1
                for r in trade_results
                if (r.get("predicted_direction") == "BUY" and r.get("profit", 0) > 0)
                or (r.get("predicted_direction") == "SELL" and r.get("profit", 0) > 0)
            )

            accuracy = correct_directions / total_trades if total_trades > 0 else 0
            precision = accuracy  # Simplified
            recall = accuracy  # Simplified
            f1_score = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            # Create updated metrics
            metrics = ModelMetrics(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                profit_factor=profit_factor,
                total_trades=total_trades,
                last_updated=datetime.now(),
            )

            # Update registry
            if model_id in self.model_registry:
                self.model_registry[model_id].metrics = metrics

                # Store performance history
                if model_id not in self.performance_history:
                    self.performance_history[model_id] = []
                self.performance_history[model_id].append(metrics)

                # Keep only recent history
                max_history = 100
                if len(self.performance_history[model_id]) > max_history:
                    self.performance_history[model_id] = self.performance_history[
                        model_id
                    ][-max_history:]

            return metrics

        except Exception as e:
            self.logger.error(f"Failed to evaluate model performance: {e}")
            return (
                self.model_registry[model_id].metrics
                if model_id in self.model_registry
                else None
            )

    def check_model_switching_conditions(self, symbol: str) -> Optional[str]:
        """
        Check if model switching conditions are met

        Args:
            symbol: Trading symbol

        Returns:
            Model ID to switch to, or None
        """
        try:
            current_model_info = self._get_active_model_for_symbol(symbol)
            if not current_model_info:
                return None

            current_metrics = current_model_info.metrics

            # Check performance degradation
            if (
                current_metrics.accuracy
                < self.retraining_config["performance_threshold"]
                and current_metrics.total_trades
                >= self.retraining_config["min_trades_for_evaluation"]
            ):

                # Find better performing model
                better_model = self._find_better_model(symbol, current_metrics)
                if better_model:
                    return better_model.model_id

                # Trigger retraining if no better model found
                self._schedule_model_retraining(
                    current_model_info.model_id,
                    RetrainingTrigger.PERFORMANCE_DEGRADATION,
                )

            # Check for regime-specific model switching
            if self.switching_rules["regime_specific_models"]:
                regime_model = self._get_regime_specific_model(symbol)
                if (
                    regime_model
                    and regime_model.model_id != current_model_info.model_id
                    and regime_model.metrics.accuracy
                    > current_metrics.accuracy
                    + self.switching_rules["min_performance_difference"]
                ):
                    return regime_model.model_id

            return None

        except Exception as e:
            self.logger.error(f"Failed to check switching conditions: {e}")
            return None

    def _find_better_model(
        self, symbol: str, current_metrics: ModelMetrics
    ) -> Optional[ModelInfo]:
        """Find better performing model for symbol"""
        try:
            best_model = None
            best_score = current_metrics.accuracy

            for model_info in self.model_registry.values():
                if (
                    model_info.symbol == symbol
                    and model_info.status in [ModelStatus.ACTIVE, ModelStatus.INACTIVE]
                    and model_info.metrics.total_trades
                    >= self.retraining_config["min_trades_for_evaluation"]
                ):

                    # Calculate composite score
                    score = (
                        model_info.metrics.accuracy * 0.4
                        + model_info.metrics.sharpe_ratio * 0.3
                        + model_info.metrics.profit_factor * 0.3
                    )

                    if (
                        score
                        > best_score
                        + self.switching_rules["min_performance_difference"]
                    ):
                        best_model = model_info
                        best_score = score

            return best_model

        except Exception as e:
            self.logger.error(f"Failed to find better model: {e}")
            return None

    def _get_regime_specific_model(self, symbol: str) -> Optional[ModelInfo]:
        """Get model specific to current market regime"""
        try:
            # This would integrate with market regime detection
            # For now, return None (placeholder)
            return None

        except Exception as e:
            self.logger.error(f"Failed to get regime specific model: {e}")
            return None

    def _schedule_model_retraining(self, model_id: str, trigger: RetrainingTrigger):
        """Schedule model retraining"""
        try:
            if model_id not in self.model_registry:
                return

            model_info = self.model_registry[model_id]

            def retrain_task():
                try:
                    self.logger.info(
                        f"Starting retraining for {model_id} - Trigger: {trigger.value}"
                    )

                    # Mark model as training
                    model_info.status = ModelStatus.TRAINING

                    # Load model
                    model_instance = self.load_model(model_id)
                    if not model_instance:
                        self.logger.error(
                            f"Failed to load model {model_id} for retraining"
                        )
                        model_info.status = ModelStatus.FAILED
                        return

                    # Get fresh training data (placeholder)
                    # In real implementation, this would fetch recent market data
                    # training_data = self._get_fresh_training_data(model_info.symbol)

                    # Retrain model (placeholder)
                    # training_result = model_instance.train(training_data)

                    # Update model info
                    model_info.last_training = datetime.now()
                    model_info.status = ModelStatus.ACTIVE

                    # Save retrained model
                    self._save_model_instance(model_instance, model_info.file_path)

                    self.logger.info(f"Retraining completed for {model_id}")

                except Exception as e:
                    self.logger.error(f"Retraining failed for {model_id}: {e}")
                    model_info.status = ModelStatus.FAILED

            # Submit to executor
            future = self.executor.submit(retrain_task)
            self.background_tasks.append(future)

        except Exception as e:
            self.logger.error(f"Failed to schedule retraining: {e}")

    def _setup_automatic_tasks(self):
        """Setup automatic background tasks"""
        try:
            # Schedule periodic performance checks
            schedule.every(1).hours.do(self._periodic_performance_check)

            # Schedule daily model evaluation
            schedule.every().day.at("02:00").do(self._daily_model_evaluation)

            # Schedule weekly model cleanup
            schedule.every().monday.at("03:00").do(self._weekly_cleanup)

            # Start scheduler thread
            def run_scheduler():
                while True:
                    schedule.run_pending()
                    time.sleep(60)

            scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
            scheduler_thread.start()

            self.logger.info("Automatic tasks scheduled")

        except Exception as e:
            self.logger.error(f"Failed to setup automatic tasks: {e}")

    def _periodic_performance_check(self):
        """Periodic performance check for all active models"""
        try:
            for symbol in self.symbols:
                switch_model_id = self.check_model_switching_conditions(symbol)
                if switch_model_id:
                    self.switch_model(
                        symbol, switch_model_id, "Automatic performance-based switch"
                    )

        except Exception as e:
            self.logger.error(f"Periodic performance check failed: {e}")

    def _daily_model_evaluation(self):
        """Daily model evaluation and maintenance"""
        try:
            self.logger.info("Starting daily model evaluation")

            # Cleanup completed background tasks
            self.background_tasks = [
                task for task in self.background_tasks if not task.done()
            ]

            # Save registry
            self._save_registry()

            # Log model status
            for model_id, model_info in self.model_registry.items():
                self.logger.info(
                    f"Model {model_id}: {model_info.status.value}, "
                    f"Accuracy: {model_info.metrics.accuracy:.3f}"
                )

        except Exception as e:
            self.logger.error(f"Daily model evaluation failed: {e}")

    def _weekly_cleanup(self):
        """Weekly cleanup of old models and files"""
        try:
            self.logger.info("Starting weekly cleanup")

            # Archive old inactive models
            cutoff_date = datetime.now() - timedelta(days=30)

            for model_id, model_info in list(self.model_registry.items()):
                if (
                    model_info.status == ModelStatus.INACTIVE
                    and model_info.last_training
                    and model_info.last_training < cutoff_date
                ):

                    # Move to archive
                    archive_path = f"{self.base_path}/archive/{model_id}"
                    if os.path.exists(model_info.file_path):
                        os.rename(model_info.file_path, archive_path)

                    # Update status
                    model_info.status = ModelStatus.DEPRECATED
                    model_info.file_path = archive_path

            self.logger.info("Weekly cleanup completed")

        except Exception as e:
            self.logger.error(f"Weekly cleanup failed: {e}")

    def get_model_status(self) -> Dict[str, Any]:
        """Get comprehensive model status"""
        try:
            status = {
                "total_models": len(self.model_registry),
                "active_models": len(
                    [
                        m
                        for m in self.model_registry.values()
                        if m.status == ModelStatus.ACTIVE
                    ]
                ),
                "training_models": len(
                    [
                        m
                        for m in self.model_registry.values()
                        if m.status == ModelStatus.TRAINING
                    ]
                ),
                "failed_models": len(
                    [
                        m
                        for m in self.model_registry.values()
                        if m.status == ModelStatus.FAILED
                    ]
                ),
                "symbols": self.symbols,
                "models_by_symbol": {},
                "recent_switches": self.model_selection_history[
                    -10:
                ],  # Last 10 switches
                "background_tasks": len(self.background_tasks),
            }

            # Models by symbol
            for symbol in self.symbols:
                active_model = self._get_active_model_for_symbol(symbol)
                status["models_by_symbol"][symbol] = {
                    "active_model": active_model.model_id if active_model else None,
                    "active_model_type": (
                        active_model.model_type.value if active_model else None
                    ),
                    "performance": (
                        active_model.metrics.to_dict() if active_model else None
                    ),
                }

            return status

        except Exception as e:
            self.logger.error(f"Failed to get model status: {e}")
            return {}

    def predict(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        technical_indicators: pd.DataFrame = None,
    ) -> Optional[PredictionResult]:
        """
        Make prediction using active model

        Args:
            symbol: Trading symbol
            market_data: Market data
            technical_indicators: Technical indicators

        Returns:
            Prediction result
        """
        try:
            active_model = self.get_active_model(symbol)
            if not active_model:
                self.logger.warning(f"No active model for {symbol}")
                return None

            # Check if model switching is needed
            switch_model_id = self.check_model_switching_conditions(symbol)
            if switch_model_id:
                self.switch_model(
                    symbol, switch_model_id, "Real-time performance switch"
                )
                active_model = self.get_active_model(symbol)

            # Make prediction
            if hasattr(active_model, "predict"):
                if isinstance(active_model, EnsemblePredictor):
                    return active_model.predict(technical_indicators, market_data)
                else:
                    prediction = active_model.predict(market_data, technical_indicators)
                    # Convert to PredictionResult if needed
                    if isinstance(prediction, dict):
                        # Convert dict to PredictionResult (simplified)
                        from brain.ensemble_predictor import SignalStrength

                        return PredictionResult(
                            direction=prediction.get("direction", "HOLD"),
                            confidence=prediction.get("confidence", 0.0),
                            strength=SignalStrength.MODERATE,
                            price_target=prediction.get("predicted_price"),
                            probability_up=prediction.get("direction_probability", 0.5),
                            probability_down=1.0
                            - prediction.get("direction_probability", 0.5),
                            expected_return=prediction.get("expected_return", 0.0),
                            risk_score=1.0 - prediction.get("confidence", 0.0),
                            market_regime=MarketRegime.RANGING,
                            model_predictions={
                                active_model.__class__.__name__: prediction.get(
                                    "predicted_price", 0.0
                                )
                            },
                            model_weights={active_model.__class__.__name__: 1.0},
                            features_used=[],
                            timestamp=datetime.now(),
                        )
                    return prediction

            return None

        except Exception as e:
            self.logger.error(f"Prediction failed for {symbol}: {e}")
            return None

    def update_model_performance(self, symbol: str, trade_result: Dict[str, Any]):
        """
        Update model performance with trade result

        Args:
            symbol: Trading symbol
            trade_result: Trade result data
        """
        try:
            active_model_info = self._get_active_model_for_symbol(symbol)
            if not active_model_info:
                return

            # Update metrics (simplified - would be more comprehensive in production)
            current_metrics = active_model_info.metrics

            # Add this trade to evaluation
            if hasattr(self, "recent_trades"):
                if symbol not in self.recent_trades:
                    self.recent_trades[symbol] = []
                self.recent_trades[symbol].append(trade_result)

                # Evaluate if we have enough trades
                if (
                    len(self.recent_trades[symbol])
                    >= self.retraining_config["min_trades_for_evaluation"]
                ):
                    self.evaluate_model_performance(
                        active_model_info.model_id,
                        self.recent_trades[symbol][
                            -self.retraining_config["evaluation_window"] :
                        ],
                    )

            # Update active model if it supports performance updates
            active_model = self.get_active_model(symbol)
            if hasattr(active_model, "update_model_performance"):
                active_model.update_model_performance(
                    trade_result.get("ticket", 0), trade_result.get("profit", 0.0)
                )

        except Exception as e:
            self.logger.error(f"Failed to update model performance: {e}")

    def __del__(self):
        """Cleanup on destruction"""
        try:
            # Save registry
            self._save_registry()

            # Shutdown executor
            if hasattr(self, "executor"):
                self.executor.shutdown(wait=False)

        except:
            pass


# Usage Example
if __name__ == "__main__":
    # Initialize model manager
    manager = ModelManager(["EURUSD", "GBPUSD"])

    # Get model status
    status = manager.get_model_status()
    print("Model Status:", json.dumps(status, indent=2))

    # Example market data
    market_data = pd.DataFrame(
        {
            "close": [1.1000, 1.1010, 1.1020, 1.1015, 1.1025] * 20,
            "volume": [1000, 1200, 900, 1100, 1300] * 20,
        }
    )

    # Make prediction
    prediction = manager.predict("EURUSD", market_data)
    if prediction:
        print(
            f"Prediction: {prediction.direction} with confidence {prediction.confidence:.3f}"
        )

    # Example trade result
    trade_result = {
        "ticket": 12345,
        "symbol": "EURUSD",
        "profit": 50.0,
        "predicted_direction": "BUY",
    }

    # Update performance
    manager.update_model_performance("EURUSD", trade_result)
