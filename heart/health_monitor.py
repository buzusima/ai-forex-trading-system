"""
heart/health_monitor.py
System Health Monitoring
Comprehensive health monitoring for institutional Forex trading system
"""

import psutil
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import threading
import time
import json
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import warnings
import socket
import requests
from collections import deque, defaultdict

warnings.filterwarnings("ignore")


class HealthStatus(Enum):
    """System health status levels"""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILURE = "failure"
    UNKNOWN = "unknown"


class ComponentType(Enum):
    """System component types"""

    MT5_CONNECTION = "mt5_connection"
    DATABASE = "database"
    MODEL = "model"
    RISK_MANAGER = "risk_manager"
    TRADE_MANAGER = "trade_manager"
    SIGNAL_ANALYZER = "signal_analyzer"
    FEATURE_ENGINEER = "feature_engineer"
    DATA_FEED = "data_feed"
    NETWORK = "network"
    SYSTEM = "system"
    MEMORY = "memory"
    CPU = "cpu"
    DISK = "disk"


@dataclass
class HealthMetric:
    """Health metric structure"""

    name: str
    value: Union[float, int, bool, str]
    unit: str = ""
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    status: HealthStatus = HealthStatus.HEALTHY
    timestamp: datetime = field(default_factory=datetime.now)
    component: ComponentType = ComponentType.SYSTEM
    description: str = ""

    def evaluate_status(self) -> HealthStatus:
        """Evaluate health status based on thresholds"""
        if not isinstance(self.value, (int, float)):
            return HealthStatus.UNKNOWN

        if (
            self.threshold_critical is not None
            and self.value >= self.threshold_critical
        ):
            self.status = HealthStatus.CRITICAL
        elif (
            self.threshold_warning is not None and self.value >= self.threshold_warning
        ):
            self.status = HealthStatus.WARNING
        else:
            self.status = HealthStatus.HEALTHY

        return self.status

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "threshold_warning": self.threshold_warning,
            "threshold_critical": self.threshold_critical,
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "component": self.component.value,
            "description": self.description,
        }


@dataclass
class AlertRule:
    """Alert rule configuration"""

    component: ComponentType
    metric_name: str
    condition: str  # 'greater_than', 'less_than', 'equals', 'not_equals'
    threshold: Union[float, int, bool, str]
    severity: HealthStatus
    cooldown_minutes: int = 5
    last_triggered: Optional[datetime] = None
    enabled: bool = True

    def should_trigger(self, metric: HealthMetric) -> bool:
        """Check if alert should be triggered"""
        if (
            not self.enabled
            or self.component != metric.component
            or self.metric_name != metric.name
        ):
            return False

        # Check cooldown
        if self.last_triggered and datetime.now() - self.last_triggered < timedelta(
            minutes=self.cooldown_minutes
        ):
            return False

        # Evaluate condition
        if self.condition == "greater_than":
            return metric.value > self.threshold
        elif self.condition == "less_than":
            return metric.value < self.threshold
        elif self.condition == "equals":
            return metric.value == self.threshold
        elif self.condition == "not_equals":
            return metric.value != self.threshold

        return False


@dataclass
class SystemSnapshot:
    """System health snapshot"""

    timestamp: datetime
    overall_status: HealthStatus
    metrics: List[HealthMetric]
    alerts: List[str] = field(default_factory=list)
    component_status: Dict[ComponentType, HealthStatus] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "overall_status": self.overall_status.value,
            "metrics": [m.to_dict() for m in self.metrics],
            "alerts": self.alerts,
            "component_status": {
                k.value: v.value for k, v in self.component_status.items()
            },
        }


@dataclass
class HealthMonitorConfig:
    """Health monitor configuration"""

    # Monitoring intervals
    system_check_interval: int = 30  # seconds
    component_check_interval: int = 60  # seconds
    detailed_check_interval: int = 300  # seconds

    # System thresholds
    cpu_warning_pct: float = 75.0
    cpu_critical_pct: float = 90.0
    memory_warning_pct: float = 80.0
    memory_critical_pct: float = 95.0
    disk_warning_pct: float = 85.0
    disk_critical_pct: float = 95.0

    # Network thresholds
    network_timeout_sec: int = 30
    max_latency_ms: float = 1000.0

    # Component thresholds
    mt5_connection_timeout: int = 5
    model_prediction_timeout: int = 10
    max_error_rate_pct: float = 5.0

    # Alert settings
    enable_alerts: bool = True
    alert_cooldown_minutes: int = 5
    max_alerts_per_hour: int = 10

    # Data retention
    metric_retention_hours: int = 24
    snapshot_retention_hours: int = 48
    log_retention_days: int = 7

    # Database settings
    db_path: str = "health_monitor.db"
    enable_persistence: bool = True


class HealthMonitor:
    """
    Comprehensive System Health Monitor
    Monitors all aspects of the trading system
    """

    def __init__(self, config: HealthMonitorConfig = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or HealthMonitorConfig()

        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread = None
        self.stop_event = threading.Event()

        # Health data
        self.current_metrics: Dict[str, HealthMetric] = {}
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.snapshots: deque = deque(maxlen=100)
        self.component_registry: Dict[ComponentType, Dict] = {}

        # Alert management
        self.alert_rules: List[AlertRule] = []
        self.active_alerts: List[str] = []
        self.alert_history: deque = deque(maxlen=500)
        self.alert_callbacks: List[Callable] = []

        # Performance tracking
        self.error_counts: Dict[ComponentType, int] = defaultdict(int)
        self.last_check_times: Dict[ComponentType, datetime] = {}
        self.response_times: Dict[ComponentType, deque] = defaultdict(
            lambda: deque(maxlen=100)
        )

        # Database connection
        self.db_conn = None
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Initialize
        self._initialize_database()
        self._setup_default_alert_rules()

        self.logger.info("HealthMonitor initialized successfully")

    def start_monitoring(self):
        """Start health monitoring"""
        try:
            if self.is_monitoring:
                self.logger.warning("Health monitoring already running")
                return

            self.is_monitoring = True
            self.stop_event.clear()

            self.monitor_thread = threading.Thread(
                target=self._monitoring_loop, daemon=True
            )
            self.monitor_thread.start()

            self.logger.info("Health monitoring started")

        except Exception as e:
            self.logger.error(f"Error starting health monitoring: {e}")
            raise

    def stop_monitoring(self):
        """Stop health monitoring"""
        try:
            if not self.is_monitoring:
                return

            self.is_monitoring = False
            self.stop_event.set()

            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=10.0)

            self.executor.shutdown(wait=True)

            if self.db_conn:
                self.db_conn.close()

            self.logger.info("Health monitoring stopped")

        except Exception as e:
            self.logger.error(f"Error stopping health monitoring: {e}")

    def register_component(
        self,
        component_type: ComponentType,
        component_instance: object,
        health_check_func: Callable = None,
        custom_metrics: Dict[str, Callable] = None,
    ):
        """Register a component for monitoring"""
        try:
            self.component_registry[component_type] = {
                "instance": component_instance,
                "health_check": health_check_func,
                "custom_metrics": custom_metrics or {},
                "registered_at": datetime.now(),
                "last_check": None,
                "status": HealthStatus.UNKNOWN,
            }

            self.logger.info(f"Registered component: {component_type.value}")

        except Exception as e:
            self.logger.error(f"Error registering component: {e}")

    def add_alert_rule(self, rule: AlertRule):
        """Add custom alert rule"""
        try:
            self.alert_rules.append(rule)
            self.logger.info(
                f"Added alert rule for {rule.component.value}.{rule.metric_name}"
            )

        except Exception as e:
            self.logger.error(f"Error adding alert rule: {e}")

    def add_alert_callback(self, callback: Callable):
        """Add alert callback function"""
        try:
            self.alert_callbacks.append(callback)
            self.logger.info("Added alert callback")

        except Exception as e:
            self.logger.error(f"Error adding alert callback: {e}")

    def _monitoring_loop(self):
        """Main monitoring loop"""
        self.logger.info("Health monitoring loop started")

        last_system_check = 0
        last_component_check = 0
        last_detailed_check = 0

        while self.is_monitoring and not self.stop_event.is_set():
            try:
                current_time = time.time()

                # System checks
                if (
                    current_time - last_system_check
                    >= self.config.system_check_interval
                ):
                    self._check_system_health()
                    last_system_check = current_time

                # Component checks
                if (
                    current_time - last_component_check
                    >= self.config.component_check_interval
                ):
                    self._check_component_health()
                    last_component_check = current_time

                # Detailed checks
                if (
                    current_time - last_detailed_check
                    >= self.config.detailed_check_interval
                ):
                    self._perform_detailed_checks()
                    last_detailed_check = current_time

                # Create snapshot
                self._create_health_snapshot()

                # Check alerts
                self._check_alerts()

                # Cleanup old data
                self._cleanup_old_data()

                # Sleep
                self.stop_event.wait(5.0)  # Check every 5 seconds

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5.0)

        self.logger.info("Health monitoring loop stopped")

    def _check_system_health(self):
        """Check basic system health metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self._add_metric(
                HealthMetric(
                    name="cpu_usage",
                    value=cpu_percent,
                    unit="%",
                    threshold_warning=self.config.cpu_warning_pct,
                    threshold_critical=self.config.cpu_critical_pct,
                    component=ComponentType.CPU,
                    description="CPU usage percentage",
                )
            )

            # Memory usage
            memory = psutil.virtual_memory()
            self._add_metric(
                HealthMetric(
                    name="memory_usage",
                    value=memory.percent,
                    unit="%",
                    threshold_warning=self.config.memory_warning_pct,
                    threshold_critical=self.config.memory_critical_pct,
                    component=ComponentType.MEMORY,
                    description="Memory usage percentage",
                )
            )

            self._add_metric(
                HealthMetric(
                    name="memory_available",
                    value=memory.available / (1024**3),  # GB
                    unit="GB",
                    component=ComponentType.MEMORY,
                    description="Available memory in GB",
                )
            )

            # Disk usage
            disk = psutil.disk_usage("/")
            disk_percent = (disk.used / disk.total) * 100
            self._add_metric(
                HealthMetric(
                    name="disk_usage",
                    value=disk_percent,
                    unit="%",
                    threshold_warning=self.config.disk_warning_pct,
                    threshold_critical=self.config.disk_critical_pct,
                    component=ComponentType.DISK,
                    description="Disk usage percentage",
                )
            )

            # Network connectivity
            self._check_network_connectivity()

            # Process count
            process_count = len(psutil.pids())
            self._add_metric(
                HealthMetric(
                    name="process_count",
                    value=process_count,
                    unit="count",
                    component=ComponentType.SYSTEM,
                    description="Number of running processes",
                )
            )

            # Load average (Unix/Linux only)
            try:
                load_avg = os.getloadavg()
                self._add_metric(
                    HealthMetric(
                        name="load_average_1m",
                        value=load_avg[0],
                        unit="",
                        threshold_warning=psutil.cpu_count(),
                        threshold_critical=psutil.cpu_count() * 2,
                        component=ComponentType.CPU,
                        description="1-minute load average",
                    )
                )
            except (OSError, AttributeError):
                pass  # Not available on Windows

        except Exception as e:
            self.logger.error(f"Error checking system health: {e}")

    def _check_component_health(self):
        """Check health of registered components"""
        try:
            for component_type, component_info in self.component_registry.items():
                try:
                    start_time = time.time()

                    # Run health check if available
                    if component_info["health_check"]:
                        health_result = component_info["health_check"]()
                        component_info["status"] = health_result.get(
                            "status", HealthStatus.UNKNOWN
                        )

                        # Add component-specific metrics
                        for metric_name, metric_value in health_result.get(
                            "metrics", {}
                        ).items():
                            self._add_metric(
                                HealthMetric(
                                    name=f"{component_type.value}_{metric_name}",
                                    value=metric_value,
                                    component=component_type,
                                    description=f"{component_type.value} {metric_name}",
                                )
                            )

                    # Run custom metrics
                    for metric_name, metric_func in component_info[
                        "custom_metrics"
                    ].items():
                        try:
                            metric_value = metric_func()
                            self._add_metric(
                                HealthMetric(
                                    name=f"{component_type.value}_{metric_name}",
                                    value=metric_value,
                                    component=component_type,
                                    description=f"{component_type.value} {metric_name}",
                                )
                            )
                        except Exception as e:
                            self.logger.error(
                                f"Error executing custom metric {metric_name}: {e}"
                            )

                    # Track response time
                    response_time = (time.time() - start_time) * 1000  # ms
                    self.response_times[component_type].append(response_time)

                    self._add_metric(
                        HealthMetric(
                            name=f"{component_type.value}_response_time",
                            value=response_time,
                            unit="ms",
                            threshold_warning=1000.0,  # 1 second
                            threshold_critical=5000.0,  # 5 seconds
                            component=component_type,
                            description=f"{component_type.value} response time",
                        )
                    )

                    component_info["last_check"] = datetime.now()

                except Exception as e:
                    self.logger.error(
                        f"Error checking component {component_type.value}: {e}"
                    )
                    self.error_counts[component_type] += 1
                    component_info["status"] = HealthStatus.FAILURE

        except Exception as e:
            self.logger.error(f"Error in component health check: {e}")

    def _perform_detailed_checks(self):
        """Perform detailed system checks"""
        try:
            # Database connectivity
            self._check_database_health()

            # File system checks
            self._check_file_system()

            # Performance metrics
            self._calculate_performance_metrics()

            # Error rate analysis
            self._analyze_error_rates()

        except Exception as e:
            self.logger.error(f"Error in detailed checks: {e}")

    def _check_network_connectivity(self):
        """Check network connectivity"""
        try:
            # Test internet connectivity
            start_time = time.time()
            try:
                response = requests.get(
                    "https://www.google.com", timeout=self.config.network_timeout_sec
                )
                latency = (time.time() - start_time) * 1000  # ms

                if response.status_code == 200:
                    self._add_metric(
                        HealthMetric(
                            name="internet_connectivity",
                            value=True,
                            component=ComponentType.NETWORK,
                            description="Internet connectivity status",
                        )
                    )

                    self._add_metric(
                        HealthMetric(
                            name="internet_latency",
                            value=latency,
                            unit="ms",
                            threshold_warning=self.config.max_latency_ms,
                            threshold_critical=self.config.max_latency_ms * 2,
                            component=ComponentType.NETWORK,
                            description="Internet latency",
                        )
                    )
                else:
                    self._add_metric(
                        HealthMetric(
                            name="internet_connectivity",
                            value=False,
                            component=ComponentType.NETWORK,
                            description="Internet connectivity status",
                        )
                    )

            except requests.RequestException:
                self._add_metric(
                    HealthMetric(
                        name="internet_connectivity",
                        value=False,
                        component=ComponentType.NETWORK,
                        description="Internet connectivity status",
                    )
                )

            # Check network interfaces
            net_io = psutil.net_io_counters()
            self._add_metric(
                HealthMetric(
                    name="network_bytes_sent",
                    value=net_io.bytes_sent,
                    unit="bytes",
                    component=ComponentType.NETWORK,
                    description="Total bytes sent",
                )
            )

            self._add_metric(
                HealthMetric(
                    name="network_bytes_recv",
                    value=net_io.bytes_recv,
                    unit="bytes",
                    component=ComponentType.NETWORK,
                    description="Total bytes received",
                )
            )

        except Exception as e:
            self.logger.error(f"Error checking network connectivity: {e}")

    def _check_database_health(self):
        """Check database health"""
        try:
            if self.config.enable_persistence and self.db_conn:
                start_time = time.time()

                # Simple query to test connection
                cursor = self.db_conn.cursor()
                cursor.execute("SELECT 1")
                result = cursor.fetchone()

                query_time = (time.time() - start_time) * 1000  # ms

                self._add_metric(
                    HealthMetric(
                        name="database_connectivity",
                        value=result[0] == 1,
                        component=ComponentType.DATABASE,
                        description="Database connectivity status",
                    )
                )

                self._add_metric(
                    HealthMetric(
                        name="database_query_time",
                        value=query_time,
                        unit="ms",
                        threshold_warning=100.0,
                        threshold_critical=1000.0,
                        component=ComponentType.DATABASE,
                        description="Database query response time",
                    )
                )

                # Check database size
                db_size = os.path.getsize(self.config.db_path) / (1024**2)  # MB
                self._add_metric(
                    HealthMetric(
                        name="database_size",
                        value=db_size,
                        unit="MB",
                        component=ComponentType.DATABASE,
                        description="Database file size",
                    )
                )

        except Exception as e:
            self.logger.error(f"Error checking database health: {e}")
            self._add_metric(
                HealthMetric(
                    name="database_connectivity",
                    value=False,
                    component=ComponentType.DATABASE,
                    description="Database connectivity status",
                )
            )

    def _check_file_system(self):
        """Check file system health"""
        try:
            # Check important directories
            important_dirs = ["logs", "data", "models", "config"]

            for dir_name in important_dirs:
                if os.path.exists(dir_name):
                    # Check if writable
                    test_file = os.path.join(dir_name, ".health_test")
                    try:
                        with open(test_file, "w") as f:
                            f.write("test")
                        os.remove(test_file)
                        writable = True
                    except:
                        writable = False

                    self._add_metric(
                        HealthMetric(
                            name=f"directory_{dir_name}_writable",
                            value=writable,
                            component=ComponentType.DISK,
                            description=f"Directory {dir_name} write access",
                        )
                    )

            # Check log file sizes
            if os.path.exists("logs"):
                total_log_size = 0
                for root, dirs, files in os.walk("logs"):
                    for file in files:
                        if file.endswith(".log"):
                            total_log_size += os.path.getsize(os.path.join(root, file))

                self._add_metric(
                    HealthMetric(
                        name="log_files_size",
                        value=total_log_size / (1024**2),  # MB
                        unit="MB",
                        threshold_warning=1000.0,  # 1GB
                        threshold_critical=5000.0,  # 5GB
                        component=ComponentType.DISK,
                        description="Total log files size",
                    )
                )

        except Exception as e:
            self.logger.error(f"Error checking file system: {e}")

    def _calculate_performance_metrics(self):
        """Calculate performance metrics"""
        try:
            # Calculate average response times
            for component_type, response_times in self.response_times.items():
                if response_times:
                    avg_response_time = np.mean(list(response_times))
                    self._add_metric(
                        HealthMetric(
                            name=f"{component_type.value}_avg_response_time",
                            value=avg_response_time,
                            unit="ms",
                            component=component_type,
                            description=f"Average response time for {component_type.value}",
                        )
                    )

            # System uptime
            boot_time = psutil.boot_time()
            uptime_seconds = time.time() - boot_time
            uptime_hours = uptime_seconds / 3600

            self._add_metric(
                HealthMetric(
                    name="system_uptime",
                    value=uptime_hours,
                    unit="hours",
                    component=ComponentType.SYSTEM,
                    description="System uptime",
                )
            )

        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")

    def _analyze_error_rates(self):
        """Analyze error rates"""
        try:
            for component_type, error_count in self.error_counts.items():
                # Calculate error rate (errors per hour)
                last_check = self.last_check_times.get(component_type)
                if last_check:
                    hours_since_last = (
                        datetime.now() - last_check
                    ).total_seconds() / 3600
                    error_rate = error_count / max(
                        hours_since_last, 0.1
                    )  # Avoid division by zero
                else:
                    error_rate = 0

                self._add_metric(
                    HealthMetric(
                        name=f"{component_type.value}_error_rate",
                        value=error_rate,
                        unit="errors/hour",
                        threshold_warning=self.config.max_error_rate_pct,
                        threshold_critical=self.config.max_error_rate_pct * 2,
                        component=component_type,
                        description=f"Error rate for {component_type.value}",
                    )
                )

        except Exception as e:
            self.logger.error(f"Error analyzing error rates: {e}")

    def _add_metric(self, metric: HealthMetric):
        """Add metric to current metrics and history"""
        try:
            # Evaluate status
            metric.evaluate_status()

            # Add to current metrics
            metric_key = f"{metric.component.value}_{metric.name}"
            self.current_metrics[metric_key] = metric

            # Add to history
            self.metric_history[metric_key].append(metric)

            # Persist to database
            if self.config.enable_persistence:
                self._persist_metric(metric)

        except Exception as e:
            self.logger.error(f"Error adding metric: {e}")

    def _create_health_snapshot(self):
        """Create current health snapshot"""
        try:
            # Determine overall status
            statuses = [metric.status for metric in self.current_metrics.values()]

            if HealthStatus.FAILURE in statuses or HealthStatus.CRITICAL in statuses:
                overall_status = HealthStatus.CRITICAL
            elif HealthStatus.WARNING in statuses:
                overall_status = HealthStatus.WARNING
            else:
                overall_status = HealthStatus.HEALTHY

            # Get component statuses
            component_status = {}
            for component_type in ComponentType:
                component_metrics = [
                    metric
                    for metric in self.current_metrics.values()
                    if metric.component == component_type
                ]

                if component_metrics:
                    component_statuses = [m.status for m in component_metrics]
                    if (
                        HealthStatus.FAILURE in component_statuses
                        or HealthStatus.CRITICAL in component_statuses
                    ):
                        component_status[component_type] = HealthStatus.CRITICAL
                    elif HealthStatus.WARNING in component_statuses:
                        component_status[component_type] = HealthStatus.WARNING
                    else:
                        component_status[component_type] = HealthStatus.HEALTHY
                else:
                    component_status[component_type] = HealthStatus.UNKNOWN

            # Create snapshot
            snapshot = SystemSnapshot(
                timestamp=datetime.now(),
                overall_status=overall_status,
                metrics=list(self.current_metrics.values()),
                alerts=list(self.active_alerts),
                component_status=component_status,
            )

            self.snapshots.append(snapshot)

            # Persist snapshot
            if self.config.enable_persistence:
                self._persist_snapshot(snapshot)

        except Exception as e:
            self.logger.error(f"Error creating health snapshot: {e}")

    def _check_alerts(self):
        """Check alert rules and trigger alerts"""
        try:
            if not self.config.enable_alerts:
                return

            new_alerts = []

            for rule in self.alert_rules:
                for metric in self.current_metrics.values():
                    if rule.should_trigger(metric):
                        alert_message = f"{rule.severity.value.upper()}: {metric.component.value} {metric.name} = {metric.value}{metric.unit}"
                        new_alerts.append(alert_message)
                        rule.last_triggered = datetime.now()

                        # Add to alert history
                        self.alert_history.append(
                            {
                                "timestamp": datetime.now(),
                                "message": alert_message,
                                "rule": rule,
                                "metric": metric,
                            }
                        )

            # Update active alerts
            self.active_alerts = new_alerts

            # Trigger callbacks
            if new_alerts:
                for callback in self.alert_callbacks:
                    try:
                        callback(new_alerts)
                    except Exception as e:
                        self.logger.error(f"Error in alert callback: {e}")

        except Exception as e:
            self.logger.error(f"Error checking alerts: {e}")

    def _cleanup_old_data(self):
        """Cleanup old data"""
        try:
            cutoff_time = datetime.now() - timedelta(
                hours=self.config.metric_retention_hours
            )

            # Clean metric history
            for metric_key, history in self.metric_history.items():
                while history and history[0].timestamp < cutoff_time:
                    history.popleft()

            # Clean snapshots
            snapshot_cutoff = datetime.now() - timedelta(
                hours=self.config.snapshot_retention_hours
            )
            while self.snapshots and self.snapshots[0].timestamp < snapshot_cutoff:
                self.snapshots.popleft()

            # Clean alert history
            alert_cutoff = datetime.now() - timedelta(days=1)
            while (
                self.alert_history and self.alert_history[0]["timestamp"] < alert_cutoff
            ):
                self.alert_history.popleft()

        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")

    def _initialize_database(self):
        """Initialize SQLite database for persistence"""
        try:
            if not self.config.enable_persistence:
                return

            self.db_conn = sqlite3.connect(self.config.db_path, check_same_thread=False)

            # Create tables
            self.db_conn.execute(
                """
                CREATE TABLE IF NOT EXISTS health_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    component TEXT NOT NULL,
                    name TEXT NOT NULL,
                    value TEXT NOT NULL,
                    unit TEXT,
                    status TEXT NOT NULL,
                    description TEXT
                )
            """
            )

            self.db_conn.execute(
                """
                CREATE TABLE IF NOT EXISTS health_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    overall_status TEXT NOT NULL,
                    data TEXT NOT NULL
                )
            """
            )

            self.db_conn.commit()

        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
            self.db_conn = None

    def _persist_metric(self, metric: HealthMetric):
        """Persist metric to database"""
        try:
            if not self.db_conn:
                return

            self.db_conn.execute(
                """
                INSERT INTO health_metrics 
                (timestamp, component, name, value, unit, status, description)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    metric.timestamp.isoformat(),
                    metric.component.value,
                    metric.name,
                    str(metric.value),
                    metric.unit,
                    metric.status.value,
                    metric.description,
                ),
            )

            self.db_conn.commit()

        except Exception as e:
            self.logger.error(f"Error persisting metric: {e}")

    def _persist_snapshot(self, snapshot: SystemSnapshot):
        """Persist snapshot to database"""
        try:
            if not self.db_conn:
                return

            self.db_conn.execute(
                """
                INSERT INTO health_snapshots 
                (timestamp, overall_status, data)
                VALUES (?, ?, ?)
            """,
                (
                    snapshot.timestamp.isoformat(),
                    snapshot.overall_status.value,
                    json.dumps(snapshot.to_dict()),
                ),
            )

            self.db_conn.commit()

        except Exception as e:
            self.logger.error(f"Error persisting snapshot: {e}")

    def _setup_default_alert_rules(self):
        """Setup default alert rules"""
        try:
            default_rules = [
                AlertRule(
                    component=ComponentType.CPU,
                    metric_name="cpu_usage",
                    condition="greater_than",
                    threshold=self.config.cpu_critical_pct,
                    severity=HealthStatus.CRITICAL,
                ),
                AlertRule(
                    component=ComponentType.MEMORY,
                    metric_name="memory_usage",
                    condition="greater_than",
                    threshold=self.config.memory_critical_pct,
                    severity=HealthStatus.CRITICAL,
                ),
                AlertRule(
                    component=ComponentType.DISK,
                    metric_name="disk_usage",
                    condition="greater_than",
                    threshold=self.config.disk_critical_pct,
                    severity=HealthStatus.CRITICAL,
                ),
                AlertRule(
                    component=ComponentType.NETWORK,
                    metric_name="internet_connectivity",
                    condition="equals",
                    threshold=False,
                    severity=HealthStatus.CRITICAL,
                ),
                AlertRule(
                    component=ComponentType.DATABASE,
                    metric_name="database_connectivity",
                    condition="equals",
                    threshold=False,
                    severity=HealthStatus.CRITICAL,
                ),
            ]

            self.alert_rules.extend(default_rules)

        except Exception as e:
            self.logger.error(f"Error setting up default alert rules: {e}")

    def get_current_status(self) -> Dict[str, any]:
        """Get current system status"""
        try:
            if not self.snapshots:
                return {"status": "unknown", "message": "No health data available"}

            latest_snapshot = self.snapshots[-1]

            return {
                "overall_status": latest_snapshot.overall_status.value,
                "timestamp": latest_snapshot.timestamp.isoformat(),
                "component_count": len(latest_snapshot.component_status),
                "metric_count": len(latest_snapshot.metrics),
                "active_alerts": len(latest_snapshot.alerts),
                "is_monitoring": self.is_monitoring,
            }

        except Exception as e:
            self.logger.error(f"Error getting current status: {e}")
            return {"status": "error", "message": str(e)}

    def get_component_status(self, component_type: ComponentType) -> Dict[str, any]:
        """Get status for specific component"""
        try:
            component_metrics = [
                metric.to_dict()
                for metric in self.current_metrics.values()
                if metric.component == component_type
            ]

            if not component_metrics:
                return {"status": "unknown", "metrics": []}

            # Determine component status
            statuses = [HealthStatus(m["status"]) for m in component_metrics]
            if HealthStatus.CRITICAL in statuses:
                status = HealthStatus.CRITICAL
            elif HealthStatus.WARNING in statuses:
                status = HealthStatus.WARNING
            else:
                status = HealthStatus.HEALTHY

            return {
                "status": status.value,
                "metrics": component_metrics,
                "error_count": self.error_counts.get(component_type, 0),
                "last_check": self.last_check_times.get(component_type),
            }

        except Exception as e:
            self.logger.error(f"Error getting component status: {e}")
            return {"status": "error", "message": str(e)}

    def get_health_summary(self) -> Dict[str, any]:
        """Get comprehensive health summary"""
        try:
            if not self.snapshots:
                return {}

            latest_snapshot = self.snapshots[-1]

            # Calculate component health distribution
            component_health = {}
            for component, status in latest_snapshot.component_status.items():
                if status.value not in component_health:
                    component_health[status.value] = 0
                component_health[status.value] += 1

            # Get recent alerts
            recent_alerts = [
                alert
                for alert in self.alert_history
                if alert["timestamp"] > datetime.now() - timedelta(hours=1)
            ]

            return {
                "overall_status": latest_snapshot.overall_status.value,
                "last_updated": latest_snapshot.timestamp.isoformat(),
                "total_metrics": len(latest_snapshot.metrics),
                "component_health": component_health,
                "active_alerts": len(latest_snapshot.alerts),
                "recent_alerts_1h": len(recent_alerts),
                "system_uptime": self.current_metrics.get(
                    "system_system_uptime", {}
                ).get("value", 0),
                "monitoring_active": self.is_monitoring,
                "registered_components": len(self.component_registry),
            }

        except Exception as e:
            self.logger.error(f"Error getting health summary: {e}")
            return {}


# Example usage and helper functions
def create_mt5_health_check(mt5_connector) -> Callable:
    """Create health check function for MT5 connector"""

    def health_check():
        try:
            if hasattr(mt5_connector, "is_connected") and mt5_connector.is_connected():
                return {
                    "status": HealthStatus.HEALTHY,
                    "metrics": {
                        "connection_status": True,
                        "last_tick_age": 0,  # You would implement this
                    },
                }
            else:
                return {
                    "status": HealthStatus.CRITICAL,
                    "metrics": {"connection_status": False},
                }
        except Exception as e:
            return {
                "status": HealthStatus.FAILURE,
                "metrics": {"connection_status": False, "error": str(e)},
            }

    return health_check


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create health monitor
    config = HealthMonitorConfig()
    monitor = HealthMonitor(config)

    # Add alert callback
    def alert_callback(alerts):
        for alert in alerts:
            print(f"ALERT: {alert}")

    monitor.add_alert_callback(alert_callback)

    # Start monitoring
    monitor.start_monitoring()

    try:
        # Let it run for a bit
        time.sleep(30)

        # Get status
        status = monitor.get_current_status()
        print("\nCurrent Status:")
        for key, value in status.items():
            print(f"  {key}: {value}")

        # Get health summary
        summary = monitor.get_health_summary()
        print("\nHealth Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")

    finally:
        # Stop monitoring
        monitor.stop_monitoring()
