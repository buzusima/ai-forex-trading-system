"""
utils/telegram_notifier.py
Telegram Alert & Notification System
Comprehensive notification system for institutional Forex trading
"""

import asyncio
import aiohttp
import requests
import json
import time
import threading
from typing import Dict, List, Optional, Union, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import os
from urllib.parse import urlencode
from concurrent.futures import ThreadPoolExecutor
import queue
import sqlite3
from collections import deque, defaultdict
import hashlib
import warnings

warnings.filterwarnings("ignore")


class NotificationLevel(Enum):
    """Notification severity levels"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    SUCCESS = "success"
    TRADE = "trade"
    SIGNAL = "signal"
    SYSTEM = "system"


class MessageFormat(Enum):
    """Message format types"""

    TEXT = "text"
    HTML = "html"
    MARKDOWN = "markdown"


@dataclass
class TelegramConfig:
    """Telegram bot configuration"""

    bot_token: str
    default_chat_id: Union[str, int]
    admin_chat_ids: List[Union[str, int]] = field(default_factory=list)

    # API settings
    api_base_url: str = "https://api.telegram.org"
    request_timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0

    # Rate limiting
    max_messages_per_minute: int = 20
    max_messages_per_hour: int = 100

    # Message settings
    default_format: MessageFormat = MessageFormat.HTML
    max_message_length: int = 4096
    enable_notifications: bool = True
    enable_web_page_preview: bool = False

    # Alert settings
    enable_alert_grouping: bool = True
    alert_grouping_window_minutes: int = 5
    max_grouped_alerts: int = 10

    # Persistence
    enable_persistence: bool = True
    db_path: str = "telegram_notifications.db"
    message_retention_days: int = 30


@dataclass
class NotificationMessage:
    """Notification message structure"""

    level: NotificationLevel
    title: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    chat_id: Optional[Union[str, int]] = None
    format_type: MessageFormat = MessageFormat.HTML

    # Optional fields
    disable_notification: bool = False
    reply_to_message_id: Optional[int] = None
    parse_mode: Optional[str] = None

    # Metadata
    source: str = "system"
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    data: Dict = field(default_factory=dict)

    # Message ID for tracking
    message_id: Optional[str] = None
    telegram_message_id: Optional[int] = None

    # Status
    sent: bool = False
    sent_at: Optional[datetime] = None
    error: Optional[str] = None
    retry_count: int = 0

    def __post_init__(self):
        if self.message_id is None:
            self.message_id = self._generate_message_id()

    def _generate_message_id(self) -> str:
        """Generate unique message ID"""
        content = f"{self.timestamp.isoformat()}{self.title}{self.message}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def format_message(self) -> str:
        """Format message for Telegram"""
        if self.format_type == MessageFormat.HTML:
            return self._format_html()
        elif self.format_type == MessageFormat.MARKDOWN:
            return self._format_markdown()
        else:
            return self._format_text()

    def _format_html(self) -> str:
        """Format message as HTML"""
        emoji = self._get_level_emoji()
        level_text = self.level.value.upper()

        formatted = f"🤖 <b>{emoji} {level_text}</b>\n\n"
        formatted += f"<b>📋 {self.title}</b>\n"
        formatted += f"{self.message}\n\n"
        formatted += f"🕐 <i>{self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</i>"

        if self.source != "system":
            formatted += f"\n📌 <i>Source: {self.source}</i>"

        return formatted

    def _format_markdown(self) -> str:
        """Format message as Markdown"""
        emoji = self._get_level_emoji()
        level_text = self.level.value.upper()

        formatted = f"🤖 *{emoji} {level_text}*\n\n"
        formatted += f"*📋 {self.title}*\n"
        formatted += f"{self.message}\n\n"
        formatted += f"🕐 _{self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}_"

        if self.source != "system":
            formatted += f"\n📌 _Source: {self.source}_"

        return formatted

    def _format_text(self) -> str:
        """Format message as plain text"""
        emoji = self._get_level_emoji()
        level_text = self.level.value.upper()

        formatted = f"🤖 {emoji} {level_text}\n\n"
        formatted += f"📋 {self.title}\n"
        formatted += f"{self.message}\n\n"
        formatted += f"🕐 {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"

        if self.source != "system":
            formatted += f"\n📌 Source: {self.source}"

        return formatted

    def _get_level_emoji(self) -> str:
        """Get emoji for notification level"""
        emoji_map = {
            NotificationLevel.INFO: "ℹ️",
            NotificationLevel.WARNING: "⚠️",
            NotificationLevel.ERROR: "❌",
            NotificationLevel.CRITICAL: "🚨",
            NotificationLevel.SUCCESS: "✅",
            NotificationLevel.TRADE: "💰",
            NotificationLevel.SIGNAL: "📊",
            NotificationLevel.SYSTEM: "⚙️",
        }
        return emoji_map.get(self.level, "📢")

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "message_id": self.message_id,
            "level": self.level.value,
            "title": self.title,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "chat_id": self.chat_id,
            "format_type": self.format_type.value,
            "source": self.source,
            "category": self.category,
            "tags": self.tags,
            "data": self.data,
            "sent": self.sent,
            "sent_at": self.sent_at.isoformat() if self.sent_at else None,
            "error": self.error,
            "retry_count": self.retry_count,
            "telegram_message_id": self.telegram_message_id,
        }


class TelegramNotifier:
    """
    Advanced Telegram Notification System
    Handles alerts, trade notifications, and system messages
    """

    def __init__(self, config: TelegramConfig):
        self.logger = logging.getLogger(__name__)
        self.config = config

        # Validate configuration
        if not self.config.bot_token:
            raise ValueError("Bot token is required")
        if not self.config.default_chat_id:
            raise ValueError("Default chat ID is required")

        # Queue management
        self.message_queue = queue.Queue()
        self.failed_messages = queue.Queue()

        # Rate limiting
        self.rate_limiter = {
            "minute": deque(maxlen=self.config.max_messages_per_minute),
            "hour": deque(maxlen=self.config.max_messages_per_hour),
        }

        # Alert grouping
        self.grouped_alerts = defaultdict(list)
        self.last_group_send = defaultdict(datetime)

        # Worker thread
        self.worker_thread = None
        self.is_running = False
        self.stop_event = threading.Event()

        # Database connection
        self.db_conn = None

        # Message templates
        self.templates = self._load_message_templates()

        # Async session
        self.session = None

        # Statistics
        self.stats = {
            "total_sent": 0,
            "total_failed": 0,
            "messages_by_level": defaultdict(int),
            "last_reset": datetime.now(),
        }

        # Initialize
        self._initialize_database()
        self._test_bot_connection()

        self.logger.info("TelegramNotifier initialized successfully")

    def start(self):
        """Start the notification service"""
        try:
            if self.is_running:
                self.logger.warning("Telegram notifier already running")
                return

            self.is_running = True
            self.stop_event.clear()

            # Start worker thread
            self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.worker_thread.start()

            self.logger.info("Telegram notifier started")

        except Exception as e:
            self.logger.error(f"Error starting Telegram notifier: {e}")
            raise

    def stop(self):
        """Stop the notification service"""
        try:
            if not self.is_running:
                return

            self.is_running = False
            self.stop_event.set()

            # Wait for worker thread
            if self.worker_thread and self.worker_thread.is_alive():
                self.worker_thread.join(timeout=10.0)

            # Close database connection
            if self.db_conn:
                self.db_conn.close()

            # Close async session
            if self.session:
                asyncio.run(self.session.close())

            self.logger.info("Telegram notifier stopped")

        except Exception as e:
            self.logger.error(f"Error stopping Telegram notifier: {e}")

    def send_message(
        self,
        level: NotificationLevel,
        title: str,
        message: str,
        chat_id: Optional[Union[str, int]] = None,
        **kwargs,
    ) -> str:
        """Send notification message"""
        try:
            if not self.config.enable_notifications:
                self.logger.debug("Notifications disabled, skipping message")
                return ""

            # Create notification message
            notification = NotificationMessage(
                level=level,
                title=title,
                message=message,
                chat_id=chat_id or self.config.default_chat_id,
                format_type=self.config.default_format,
                **kwargs,
            )

            # Check if should group alerts
            if self.config.enable_alert_grouping and level in [
                NotificationLevel.WARNING,
                NotificationLevel.ERROR,
                NotificationLevel.CRITICAL,
            ]:
                self._handle_alert_grouping(notification)
            else:
                # Add to queue
                self.message_queue.put(notification)

            return notification.message_id

        except Exception as e:
            self.logger.error(f"Error sending message: {e}")
            return ""

    def send_trade_notification(
        self,
        trade_action: str,
        symbol: str,
        entry_price: float,
        volume: float,
        profit: Optional[float] = None,
        **kwargs,
    ) -> str:
        """Send trade notification"""
        try:
            # Format trade message
            if trade_action.lower() in ["buy", "sell"]:
                title = f"🔥 Trade Opened: {trade_action.upper()} {symbol}"
                message = f"💰 <b>Entry Price:</b> {entry_price:.5f}\n"
                message += f"📊 <b>Volume:</b> {volume} lots"
            elif trade_action.lower() in ["close", "exit"]:
                title = f"💼 Trade Closed: {symbol}"
                message = f"💰 <b>Exit Price:</b> {entry_price:.5f}\n"
                message += f"📊 <b>Volume:</b> {volume} lots"
                if profit is not None:
                    profit_emoji = "💚" if profit > 0 else "❤️"
                    message += f"\n{profit_emoji} <b>Profit:</b> ${profit:.2f}"
            else:
                title = f"📈 Trade Update: {symbol}"
                message = f"🔄 <b>Action:</b> {trade_action}\n"
                message += f"💰 <b>Price:</b> {entry_price:.5f}\n"
                message += f"📊 <b>Volume:</b> {volume} lots"

            # Add additional data
            if "stop_loss" in kwargs:
                message += f"\n🛑 <b>Stop Loss:</b> {kwargs['stop_loss']:.5f}"
            if "take_profit" in kwargs:
                message += f"\n🎯 <b>Take Profit:</b> {kwargs['take_profit']:.5f}"
            if "signal_source" in kwargs:
                message += f"\n📡 <b>Signal:</b> {kwargs['signal_source']}"

            return self.send_message(
                NotificationLevel.TRADE,
                title,
                message,
                source="trade_manager",
                category="trading",
                data=kwargs,
            )

        except Exception as e:
            self.logger.error(f"Error sending trade notification: {e}")
            return ""

    def send_signal_notification(
        self,
        signal_type: str,
        symbol: str,
        strength: float,
        confidence: float,
        entry_price: float,
        **kwargs,
    ) -> str:
        """Send trading signal notification"""
        try:
            signal_emoji = "📈" if signal_type.upper() == "BUY" else "📉"
            title = f"{signal_emoji} Trading Signal: {signal_type.upper()} {symbol}"

            message = f"🎯 <b>Entry Price:</b> {entry_price:.5f}\n"
            message += f"💪 <b>Strength:</b> {strength:.1%}\n"
            message += f"🎯 <b>Confidence:</b> {confidence:.1%}"

            # Add optional fields
            if "stop_loss" in kwargs:
                message += f"\n🛑 <b>Stop Loss:</b> {kwargs['stop_loss']:.5f}"
            if "take_profit" in kwargs:
                message += f"\n🎯 <b>Take Profit:</b> {kwargs['take_profit']:.5f}"
            if "risk_reward" in kwargs:
                message += f"\n⚖️ <b>R/R Ratio:</b> {kwargs['risk_reward']:.1f}"
            if "source" in kwargs:
                message += f"\n📡 <b>Source:</b> {kwargs['source']}"

            return self.send_message(
                NotificationLevel.SIGNAL,
                title,
                message,
                source="signal_analyzer",
                category="signals",
                data=kwargs,
            )

        except Exception as e:
            self.logger.error(f"Error sending signal notification: {e}")
            return ""

    def send_system_alert(
        self,
        alert_type: str,
        component: str,
        description: str,
        level: NotificationLevel = NotificationLevel.WARNING,
        **kwargs,
    ) -> str:
        """Send system alert notification"""
        try:
            title = f"🔧 System Alert: {component}"
            message = f"⚠️ <b>Type:</b> {alert_type}\n"
            message += f"📝 <b>Description:</b> {description}"

            # Add technical details if available
            if "metric_value" in kwargs:
                message += f"\n📊 <b>Current Value:</b> {kwargs['metric_value']}"
            if "threshold" in kwargs:
                message += f"\n🚨 <b>Threshold:</b> {kwargs['threshold']}"
            if "action_required" in kwargs:
                message += f"\n🔧 <b>Action Required:</b> {kwargs['action_required']}"

            # Send to admin chats for critical alerts
            chat_ids = [self.config.default_chat_id]
            if level == NotificationLevel.CRITICAL and self.config.admin_chat_ids:
                chat_ids.extend(self.config.admin_chat_ids)

            message_ids = []
            for chat_id in chat_ids:
                message_id = self.send_message(
                    level,
                    title,
                    message,
                    chat_id=chat_id,
                    source="health_monitor",
                    category="system",
                    data=kwargs,
                )
                if message_id:
                    message_ids.append(message_id)

            return ",".join(message_ids)

        except Exception as e:
            self.logger.error(f"Error sending system alert: {e}")
            return ""

    def send_performance_report(self, report_data: Dict) -> str:
        """Send performance report"""
        try:
            title = "📊 Performance Report"

            message = f"📈 <b>Trading Performance Summary</b>\n\n"

            if "total_trades" in report_data:
                message += f"🔢 <b>Total Trades:</b> {report_data['total_trades']}\n"
            if "win_rate" in report_data:
                message += f"🎯 <b>Win Rate:</b> {report_data['win_rate']:.1%}\n"
            if "total_profit" in report_data:
                profit_emoji = "💚" if report_data["total_profit"] > 0 else "❤️"
                message += f"{profit_emoji} <b>Total Profit:</b> ${report_data['total_profit']:.2f}\n"
            if "sharpe_ratio" in report_data:
                message += (
                    f"📐 <b>Sharpe Ratio:</b> {report_data['sharpe_ratio']:.2f}\n"
                )
            if "max_drawdown" in report_data:
                message += (
                    f"📉 <b>Max Drawdown:</b> {report_data['max_drawdown']:.1%}\n"
                )

            # System performance
            if "system_uptime" in report_data:
                message += (
                    f"\n⏱️ <b>System Uptime:</b> {report_data['system_uptime']:.1f}h\n"
                )
            if "signals_generated" in report_data:
                message += (
                    f"📡 <b>Signals Generated:</b> {report_data['signals_generated']}\n"
                )

            return self.send_message(
                NotificationLevel.INFO,
                title,
                message,
                source="performance_analyzer",
                category="reports",
                data=report_data,
            )

        except Exception as e:
            self.logger.error(f"Error sending performance report: {e}")
            return ""

    def _worker_loop(self):
        """Main worker loop for processing messages"""
        self.logger.info("Telegram worker loop started")

        while self.is_running and not self.stop_event.is_set():
            try:
                # Process grouped alerts
                self._process_grouped_alerts()

                # Process message queue
                try:
                    message = self.message_queue.get(timeout=1.0)
                    self._process_message(message)
                    self.message_queue.task_done()
                except queue.Empty:
                    continue

                # Process failed messages
                self._retry_failed_messages()

            except Exception as e:
                self.logger.error(f"Error in worker loop: {e}")
                time.sleep(1.0)

        self.logger.info("Telegram worker loop stopped")

    def _process_message(self, message: NotificationMessage):
        """Process individual message"""
        try:
            # Check rate limits
            if not self._check_rate_limits():
                self.logger.warning("Rate limit exceeded, queuing message for later")
                self.failed_messages.put(message)
                return

            # Send message
            success = self._send_telegram_message(message)

            if success:
                message.sent = True
                message.sent_at = datetime.now()
                self.stats["total_sent"] += 1
                self.stats["messages_by_level"][message.level.value] += 1

                # Update rate limiter
                now = datetime.now()
                self.rate_limiter["minute"].append(now)
                self.rate_limiter["hour"].append(now)

                # Persist to database
                if self.config.enable_persistence:
                    self._persist_message(message)

                self.logger.debug(f"Message sent successfully: {message.message_id}")

            else:
                message.retry_count += 1
                if message.retry_count < self.config.max_retries:
                    self.failed_messages.put(message)
                    self.logger.warning(
                        f"Message failed, will retry: {message.message_id}"
                    )
                else:
                    self.stats["total_failed"] += 1
                    self.logger.error(
                        f"Message failed permanently: {message.message_id}"
                    )

        except Exception as e:
            self.logger.error(f"Error processing message: {e}")

    def _send_telegram_message(self, message: NotificationMessage) -> bool:
        """Send message to Telegram API"""
        try:
            url = f"{self.config.api_base_url}/bot{self.config.bot_token}/sendMessage"

            # Prepare message text
            text = message.format_message()

            # Truncate if too long
            if len(text) > self.config.max_message_length:
                text = (
                    text[: self.config.max_message_length - 100] + "\n\n... (truncated)"
                )

            # Prepare payload
            payload = {
                "chat_id": message.chat_id,
                "text": text,
                "disable_web_page_preview": not self.config.enable_web_page_preview,
                "disable_notification": message.disable_notification,
            }

            # Set parse mode
            if message.format_type == MessageFormat.HTML:
                payload["parse_mode"] = "HTML"
            elif message.format_type == MessageFormat.MARKDOWN:
                payload["parse_mode"] = "MarkdownV2"

            # Add reply-to if specified
            if message.reply_to_message_id:
                payload["reply_to_message_id"] = message.reply_to_message_id

            # Send request
            response = requests.post(
                url, json=payload, timeout=self.config.request_timeout
            )

            if response.status_code == 200:
                result = response.json()
                if result.get("ok"):
                    message.telegram_message_id = result.get("result", {}).get(
                        "message_id"
                    )
                    return True
                else:
                    message.error = result.get("description", "Unknown error")
                    self.logger.error(f"Telegram API error: {message.error}")
                    return False
            else:
                message.error = f"HTTP {response.status_code}: {response.text}"
                self.logger.error(f"HTTP error: {message.error}")
                return False

        except Exception as e:
            message.error = str(e)
            self.logger.error(f"Exception sending message: {e}")
            return False

    def _handle_alert_grouping(self, message: NotificationMessage):
        """Handle alert grouping logic"""
        try:
            group_key = f"{message.level.value}_{message.source}_{message.category}"

            # Add to group
            self.grouped_alerts[group_key].append(message)

            # Check if should send group
            should_send = False

            # Send if group is full
            if len(self.grouped_alerts[group_key]) >= self.config.max_grouped_alerts:
                should_send = True

            # Send if window time elapsed
            last_send = self.last_group_send.get(group_key)
            if last_send and datetime.now() - last_send >= timedelta(
                minutes=self.config.alert_grouping_window_minutes
            ):
                should_send = True

            # Send if first message in group
            if not last_send and len(self.grouped_alerts[group_key]) == 1:
                # Wait a bit to see if more messages come
                threading.Timer(
                    self.config.alert_grouping_window_minutes * 60,
                    lambda: self._send_grouped_alerts(group_key),
                ).start()

            if should_send:
                self._send_grouped_alerts(group_key)

        except Exception as e:
            self.logger.error(f"Error handling alert grouping: {e}")
            # Fallback: send individual message
            self.message_queue.put(message)

    def _send_grouped_alerts(self, group_key: str):
        """Send grouped alerts"""
        try:
            if (
                group_key not in self.grouped_alerts
                or not self.grouped_alerts[group_key]
            ):
                return

            messages = self.grouped_alerts[group_key]
            if not messages:
                return

            # Create grouped message
            first_message = messages[0]
            count = len(messages)

            title = f"🔔 Alert Group ({count} alerts)"

            if count == 1:
                # Send individual message
                self.message_queue.put(first_message)
            else:
                # Create summary message
                grouped_text = f"📊 <b>Alert Summary</b>\n"
                grouped_text += f"🔢 <b>Count:</b> {count} alerts\n"
                grouped_text += f"📅 <b>Time Range:</b> {messages[0].timestamp.strftime('%H:%M')} - {messages[-1].timestamp.strftime('%H:%M')}\n\n"

                # Group by type
                by_level = defaultdict(int)
                by_source = defaultdict(int)

                for msg in messages:
                    by_level[msg.level.value] += 1
                    by_source[msg.source] += 1

                grouped_text += "<b>📋 By Level:</b>\n"
                for level, count in by_level.items():
                    emoji = messages[
                        0
                    ]._get_level_emoji()  # Get emoji from first message
                    grouped_text += f"  {emoji} {level.upper()}: {count}\n"

                grouped_text += "\n<b>📡 By Source:</b>\n"
                for source, count in by_source.items():
                    grouped_text += f"  🔧 {source}: {count}\n"

                # Add recent messages
                grouped_text += "\n<b>🔍 Recent Messages:</b>\n"
                for msg in messages[-3:]:  # Last 3 messages
                    grouped_text += f"  • {msg.title[:50]}...\n"

                # Create grouped notification
                grouped_notification = NotificationMessage(
                    level=first_message.level,
                    title=title,
                    message=grouped_text,
                    chat_id=first_message.chat_id,
                    source="telegram_notifier",
                    category="grouped_alerts",
                )

                self.message_queue.put(grouped_notification)

            # Clear group and update last send time
            self.grouped_alerts[group_key] = []
            self.last_group_send[group_key] = datetime.now()

        except Exception as e:
            self.logger.error(f"Error sending grouped alerts: {e}")

    def _process_grouped_alerts(self):
        """Process any pending grouped alerts"""
        try:
            now = datetime.now()

            for group_key in list(self.grouped_alerts.keys()):
                if not self.grouped_alerts[group_key]:
                    continue

                last_send = self.last_group_send.get(group_key)
                if not last_send or now - last_send >= timedelta(
                    minutes=self.config.alert_grouping_window_minutes
                ):
                    self._send_grouped_alerts(group_key)

        except Exception as e:
            self.logger.error(f"Error processing grouped alerts: {e}")

    def _check_rate_limits(self) -> bool:
        """Check if rate limits allow sending"""
        try:
            now = datetime.now()

            # Clean old entries
            minute_ago = now - timedelta(minutes=1)
            hour_ago = now - timedelta(hours=1)

            while (
                self.rate_limiter["minute"]
                and self.rate_limiter["minute"][0] < minute_ago
            ):
                self.rate_limiter["minute"].popleft()

            while self.rate_limiter["hour"] and self.rate_limiter["hour"][0] < hour_ago:
                self.rate_limiter["hour"].popleft()

            # Check limits
            if len(self.rate_limiter["minute"]) >= self.config.max_messages_per_minute:
                return False

            if len(self.rate_limiter["hour"]) >= self.config.max_messages_per_hour:
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error checking rate limits: {e}")
            return True  # Allow on error

    def _retry_failed_messages(self):
        """Retry failed messages"""
        try:
            retry_queue = []

            # Collect messages to retry
            while not self.failed_messages.empty():
                try:
                    message = self.failed_messages.get_nowait()
                    if message.retry_count < self.config.max_retries:
                        retry_queue.append(message)
                    self.failed_messages.task_done()
                except queue.Empty:
                    break

            # Retry messages with delay
            for message in retry_queue:
                time.sleep(self.config.retry_delay)
                if self._check_rate_limits():
                    self._process_message(message)
                else:
                    self.failed_messages.put(message)

        except Exception as e:
            self.logger.error(f"Error retrying failed messages: {e}")

    def _test_bot_connection(self):
        """Test bot connection and permissions"""
        try:
            url = f"{self.config.api_base_url}/bot{self.config.bot_token}/getMe"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                result = response.json()
                if result.get("ok"):
                    bot_info = result.get("result", {})
                    self.logger.info(
                        f"Bot connected successfully: @{bot_info.get('username', 'unknown')}"
                    )
                else:
                    raise Exception(f"Bot API error: {result.get('description')}")
            else:
                raise Exception(f"HTTP {response.status_code}: {response.text}")

        except Exception as e:
            self.logger.error(f"Bot connection test failed: {e}")
            # Don't raise exception, allow initialization to continue

    def _initialize_database(self):
        """Initialize SQLite database for persistence"""
        try:
            if not self.config.enable_persistence:
                return

            self.db_conn = sqlite3.connect(self.config.db_path, check_same_thread=False)

            # Create table
            self.db_conn.execute(
                """
                CREATE TABLE IF NOT EXISTS telegram_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id TEXT UNIQUE NOT NULL,
                    level TEXT NOT NULL,
                    title TEXT NOT NULL,
                    message TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    chat_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    category TEXT NOT NULL,
                    sent BOOLEAN NOT NULL,
                    sent_at TEXT,
                    telegram_message_id INTEGER,
                    error TEXT,
                    retry_count INTEGER DEFAULT 0,
                    data TEXT
                )
            """
            )

            self.db_conn.commit()

        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
            self.db_conn = None

    def _persist_message(self, message: NotificationMessage):
        """Persist message to database"""
        try:
            if not self.db_conn:
                return

            self.db_conn.execute(
                """
                INSERT OR REPLACE INTO telegram_messages 
                (message_id, level, title, message, timestamp, chat_id, source, category,
                 sent, sent_at, telegram_message_id, error, retry_count, data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    message.message_id,
                    message.level.value,
                    message.title,
                    message.message,
                    message.timestamp.isoformat(),
                    str(message.chat_id),
                    message.source,
                    message.category,
                    message.sent,
                    message.sent_at.isoformat() if message.sent_at else None,
                    message.telegram_message_id,
                    message.error,
                    message.retry_count,
                    json.dumps(message.data),
                ),
            )

            self.db_conn.commit()

        except Exception as e:
            self.logger.error(f"Error persisting message: {e}")

    def _load_message_templates(self) -> Dict[str, str]:
        """Load message templates"""
        templates = {
            "trade_opened": "🔥 Trade Opened: {action} {symbol}\n💰 Entry: {price:.5f}\n📊 Volume: {volume} lots",
            "trade_closed": "💼 Trade Closed: {symbol}\n💰 Exit: {price:.5f}\n{profit_emoji} Profit: ${profit:.2f}",
            "signal_generated": "📊 Signal: {type} {symbol}\n🎯 Entry: {price:.5f}\n💪 Strength: {strength:.1%}",
            "system_alert": "🔧 System Alert: {component}\n⚠️ {description}",
            "performance_report": "📊 Performance Report\n🎯 Win Rate: {win_rate:.1%}\n💚 Profit: ${profit:.2f}",
        }
        return templates

    def get_statistics(self) -> Dict[str, Any]:
        """Get notification statistics"""
        try:
            stats = self.stats.copy()

            # Add queue sizes
            stats["queue_size"] = self.message_queue.qsize()
            stats["failed_queue_size"] = self.failed_messages.qsize()

            # Add rate limit status
            stats["rate_limit_minute"] = len(self.rate_limiter["minute"])
            stats["rate_limit_hour"] = len(self.rate_limiter["hour"])

            # Add service status
            stats["is_running"] = self.is_running

            return stats

        except Exception as e:
            self.logger.error(f"Error getting statistics: {e}")
            return {}

    def get_message_history(self, limit: int = 100) -> List[Dict]:
        """Get message history from database"""
        try:
            if not self.db_conn:
                return []

            cursor = self.db_conn.cursor()
            cursor.execute(
                """
                SELECT * FROM telegram_messages 
                ORDER BY timestamp DESC 
                LIMIT ?
            """,
                (limit,),
            )

            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()

            return [dict(zip(columns, row)) for row in rows]

        except Exception as e:
            self.logger.error(f"Error getting message history: {e}")
            return []

    def cleanup_old_messages(self):
        """Cleanup old messages from database"""
        try:
            if not self.db_conn:
                return

            cutoff_date = datetime.now() - timedelta(
                days=self.config.message_retention_days
            )

            cursor = self.db_conn.cursor()
            cursor.execute(
                """
                DELETE FROM telegram_messages 
                WHERE timestamp < ?
            """,
                (cutoff_date.isoformat(),),
            )

            deleted_count = cursor.rowcount
            self.db_conn.commit()

            if deleted_count > 0:
                self.logger.info(f"Cleaned up {deleted_count} old messages")

        except Exception as e:
            self.logger.error(f"Error cleaning up old messages: {e}")


# Convenience functions
def create_telegram_notifier(
    bot_token: str, chat_id: Union[str, int]
) -> TelegramNotifier:
    """Create TelegramNotifier with minimal configuration"""
    config = TelegramConfig(bot_token=bot_token, default_chat_id=chat_id)
    return TelegramNotifier(config)


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Example configuration (replace with your actual values)
    config = TelegramConfig(
        bot_token="YOUR_BOT_TOKEN",
        default_chat_id="YOUR_CHAT_ID",
        admin_chat_ids=["ADMIN_CHAT_ID"],
    )

    # Create and start notifier
    notifier = TelegramNotifier(config)
    notifier.start()

    try:
        # Send test messages
        notifier.send_message(
            NotificationLevel.INFO,
            "System Started",
            "AI Trading System has been started successfully!",
        )

        # Send trade notification
        notifier.send_trade_notification(
            "buy", "EURUSD", 1.1234, 0.1, stop_loss=1.1200, take_profit=1.1300
        )

        # Send signal notification
        notifier.send_signal_notification(
            "BUY", "GBPUSD", 0.85, 0.75, 1.2500, stop_loss=1.2450, take_profit=1.2600
        )

        # Send system alert
        notifier.send_system_alert(
            "High CPU Usage",
            "Trading Server",
            "CPU usage has exceeded 90%",
            NotificationLevel.WARNING,
            metric_value="92%",
            threshold="90%",
        )

        # Wait for messages to be sent
        time.sleep(5)

        # Get statistics
        stats = notifier.get_statistics()
        print("\nNotification Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    finally:
        # Stop notifier
        notifier.stop()
