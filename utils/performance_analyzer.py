"""
utils/performance_analyzer.py
Performance Analysis System
Comprehensive trading performance analysis for institutional Forex trading
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import sqlite3
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import classification_report, confusion_matrix
import warnings

warnings.filterwarnings("ignore")


class PerformanceMetric(Enum):
    """Performance metric types"""

    TOTAL_RETURN = "total_return"
    ANNUALIZED_RETURN = "annualized_return"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    RECOVERY_FACTOR = "recovery_factor"
    EXPECTANCY = "expectancy"
    KELLY_CRITERION = "kelly_criterion"
    VAR = "value_at_risk"
    CVAR = "conditional_var"


class TimeFrame(Enum):
    """Analysis time frames"""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    ALL_TIME = "all_time"


@dataclass
class TradeRecord:
    """Individual trade record"""

    trade_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    volume: float
    profit_loss: float
    profit_loss_pct: float
    commission: float = 0.0
    swap: float = 0.0
    duration_minutes: int = 0

    # Trade metadata
    signal_source: str = "unknown"
    signal_strength: float = 0.0
    signal_confidence: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    exit_reason: str = "unknown"

    # Risk metrics
    risk_amount: float = 0.0
    risk_reward_ratio: float = 0.0
    max_adverse_excursion: float = 0.0
    max_favorable_excursion: float = 0.0

    def __post_init__(self):
        if self.duration_minutes == 0:
            self.duration_minutes = int(
                (self.exit_time - self.entry_time).total_seconds() / 60
            )

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "side": self.side,
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat(),
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "volume": self.volume,
            "profit_loss": self.profit_loss,
            "profit_loss_pct": self.profit_loss_pct,
            "commission": self.commission,
            "swap": self.swap,
            "duration_minutes": self.duration_minutes,
            "signal_source": self.signal_source,
            "signal_strength": self.signal_strength,
            "signal_confidence": self.signal_confidence,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "exit_reason": self.exit_reason,
            "risk_amount": self.risk_amount,
            "risk_reward_ratio": self.risk_reward_ratio,
            "max_adverse_excursion": self.max_adverse_excursion,
            "max_favorable_excursion": self.max_favorable_excursion,
        }


@dataclass
class PerformanceReport:
    """Comprehensive performance report"""

    period_start: datetime
    period_end: datetime
    total_trades: int
    winning_trades: int
    losing_trades: int

    # Return metrics
    total_return: float
    total_return_pct: float
    annualized_return_pct: float

    # Risk metrics
    max_drawdown_pct: float
    max_drawdown_duration_days: int
    volatility_pct: float

    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Trading metrics
    win_rate_pct: float
    profit_factor: float
    expectancy: float

    # Additional metrics
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    avg_trade_duration_hours: float

    # Portfolio metrics
    starting_balance: float
    ending_balance: float
    peak_balance: float

    # Risk metrics
    value_at_risk_5pct: float
    conditional_var_5pct: float
    kelly_criterion: float

    # Detailed breakdowns
    monthly_returns: List[float] = field(default_factory=list)
    drawdown_series: List[float] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "total_return": self.total_return,
            "total_return_pct": self.total_return_pct,
            "annualized_return_pct": self.annualized_return_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "max_drawdown_duration_days": self.max_drawdown_duration_days,
            "volatility_pct": self.volatility_pct,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "win_rate_pct": self.win_rate_pct,
            "profit_factor": self.profit_factor,
            "expectancy": self.expectancy,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "largest_win": self.largest_win,
            "largest_loss": self.largest_loss,
            "avg_trade_duration_hours": self.avg_trade_duration_hours,
            "starting_balance": self.starting_balance,
            "ending_balance": self.ending_balance,
            "peak_balance": self.peak_balance,
            "value_at_risk_5pct": self.value_at_risk_5pct,
            "conditional_var_5pct": self.conditional_var_5pct,
            "kelly_criterion": self.kelly_criterion,
            "monthly_returns": self.monthly_returns,
            "drawdown_series": self.drawdown_series,
            "equity_curve": self.equity_curve,
        }


@dataclass
class AnalyzerConfig:
    """Performance analyzer configuration"""

    # Database settings
    db_path: str = "performance_analysis.db"
    enable_persistence: bool = True

    # Analysis settings
    risk_free_rate: float = 0.02  # 2% annual risk-free rate
    benchmark_return: float = 0.08  # 8% annual benchmark
    confidence_level: float = 0.95  # 95% confidence for VaR

    # Report settings
    min_trades_for_analysis: int = 10
    default_starting_balance: float = 10000.0

    # Chart settings
    enable_charts: bool = True
    chart_style: str = "seaborn"
    figure_size: Tuple[int, int] = (12, 8)

    # Alert thresholds
    max_drawdown_alert_pct: float = 0.20  # 20%
    min_sharpe_alert: float = 1.0
    min_win_rate_alert_pct: float = 0.40  # 40%


class PerformanceAnalyzer:
    """
    Comprehensive Trading Performance Analyzer
    Analyzes trading performance with institutional-grade metrics
    """

    def __init__(self, config: AnalyzerConfig = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or AnalyzerConfig()

        # Trade data storage
        self.trades: List[TradeRecord] = []
        self.db_conn = None

        # Analysis cache
        self.analysis_cache: Dict[str, Any] = {}
        self.last_analysis_time: Optional[datetime] = None

        # Benchmark data
        self.benchmark_returns: List[float] = []

        # Initialize
        if self.config.enable_persistence:
            self._initialize_database()

        # Set up plotting style
        if self.config.enable_charts:
            plt.style.use(self.config.chart_style)
            sns.set_palette("husl")

        self.logger.info("PerformanceAnalyzer initialized successfully")

    def add_trade(self, trade: TradeRecord):
        """Add a trade record for analysis"""
        try:
            self.trades.append(trade)

            # Persist to database
            if self.config.enable_persistence:
                self._persist_trade(trade)

            # Clear cache
            self.analysis_cache.clear()

            self.logger.debug(f"Added trade: {trade.trade_id}")

        except Exception as e:
            self.logger.error(f"Error adding trade: {e}")

    def load_trades_from_dataframe(self, df: pd.DataFrame):
        """Load trades from DataFrame"""
        try:
            required_columns = [
                "trade_id",
                "symbol",
                "side",
                "entry_time",
                "exit_time",
                "entry_price",
                "exit_price",
                "volume",
                "profit_loss",
            ]

            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            for _, row in df.iterrows():
                # Calculate profit_loss_pct if not provided
                profit_loss_pct = row.get("profit_loss_pct")
                if profit_loss_pct is None:
                    if row["side"].lower() == "buy":
                        profit_loss_pct = (
                            row["exit_price"] - row["entry_price"]
                        ) / row["entry_price"]
                    else:
                        profit_loss_pct = (
                            row["entry_price"] - row["exit_price"]
                        ) / row["entry_price"]

                trade = TradeRecord(
                    trade_id=str(row["trade_id"]),
                    symbol=str(row["symbol"]),
                    side=str(row["side"]).lower(),
                    entry_time=pd.to_datetime(row["entry_time"]),
                    exit_time=pd.to_datetime(row["exit_time"]),
                    entry_price=float(row["entry_price"]),
                    exit_price=float(row["exit_price"]),
                    volume=float(row["volume"]),
                    profit_loss=float(row["profit_loss"]),
                    profit_loss_pct=float(profit_loss_pct),
                    commission=float(row.get("commission", 0)),
                    swap=float(row.get("swap", 0)),
                    signal_source=str(row.get("signal_source", "unknown")),
                    signal_strength=float(row.get("signal_strength", 0)),
                    signal_confidence=float(row.get("signal_confidence", 0)),
                    stop_loss=row.get("stop_loss"),
                    take_profit=row.get("take_profit"),
                    exit_reason=str(row.get("exit_reason", "unknown")),
                    risk_amount=float(row.get("risk_amount", 0)),
                    risk_reward_ratio=float(row.get("risk_reward_ratio", 0)),
                    max_adverse_excursion=float(row.get("max_adverse_excursion", 0)),
                    max_favorable_excursion=float(
                        row.get("max_favorable_excursion", 0)
                    ),
                )

                self.add_trade(trade)

            self.logger.info(f"Loaded {len(df)} trades from DataFrame")

        except Exception as e:
            self.logger.error(f"Error loading trades from DataFrame: {e}")
            raise

    def generate_performance_report(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        starting_balance: Optional[float] = None,
    ) -> PerformanceReport:
        """Generate comprehensive performance report"""
        try:
            # Filter trades by date range
            filtered_trades = self._filter_trades_by_date(start_date, end_date)

            if len(filtered_trades) < self.config.min_trades_for_analysis:
                raise ValueError(
                    f"Insufficient trades for analysis. Need at least {self.config.min_trades_for_analysis}"
                )

            # Set default values
            if starting_balance is None:
                starting_balance = self.config.default_starting_balance

            if start_date is None:
                start_date = min(trade.entry_time for trade in filtered_trades)
            if end_date is None:
                end_date = max(trade.exit_time for trade in filtered_trades)

            # Calculate basic metrics
            total_trades = len(filtered_trades)
            winning_trades = len([t for t in filtered_trades if t.profit_loss > 0])
            losing_trades = len([t for t in filtered_trades if t.profit_loss < 0])

            # Calculate returns
            total_return = sum(trade.profit_loss for trade in filtered_trades)
            total_return_pct = total_return / starting_balance

            # Calculate annualized return
            period_days = (end_date - start_date).days
            if period_days > 0:
                annualized_return_pct = (1 + total_return_pct) ** (
                    365.25 / period_days
                ) - 1
            else:
                annualized_return_pct = 0

            # Calculate equity curve and drawdown
            equity_curve, drawdown_series = self._calculate_equity_curve(
                filtered_trades, starting_balance
            )

            # Calculate drawdown metrics
            max_drawdown_pct = min(drawdown_series) if drawdown_series else 0
            max_dd_duration = self._calculate_max_drawdown_duration(drawdown_series)

            # Calculate volatility
            returns = self._calculate_period_returns(filtered_trades, TimeFrame.DAILY)
            volatility_pct = (
                np.std(returns) * np.sqrt(252) if returns else 0
            )  # Annualized

            # Calculate risk-adjusted metrics
            excess_returns = [r - self.config.risk_free_rate / 252 for r in returns]
            sharpe_ratio = (
                np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
                if np.std(excess_returns) > 0
                else 0
            )

            # Sortino ratio (downside deviation)
            downside_returns = [r for r in excess_returns if r < 0]
            downside_std = np.std(downside_returns) if downside_returns else 0
            sortino_ratio = (
                np.mean(excess_returns) / downside_std * np.sqrt(252)
                if downside_std > 0
                else 0
            )

            # Calmar ratio
            calmar_ratio = (
                annualized_return_pct / abs(max_drawdown_pct)
                if max_drawdown_pct != 0
                else 0
            )

            # Trading metrics
            win_rate_pct = winning_trades / total_trades if total_trades > 0 else 0

            winning_profits = [
                t.profit_loss for t in filtered_trades if t.profit_loss > 0
            ]
            losing_profits = [
                t.profit_loss for t in filtered_trades if t.profit_loss < 0
            ]

            avg_win = np.mean(winning_profits) if winning_profits else 0
            avg_loss = np.mean(losing_profits) if losing_profits else 0

            profit_factor = (
                abs(sum(winning_profits) / sum(losing_profits))
                if losing_profits and sum(losing_profits) != 0
                else float("inf")
            )

            expectancy = (win_rate_pct * avg_win) + ((1 - win_rate_pct) * avg_loss)

            # Extreme values
            largest_win = max(winning_profits) if winning_profits else 0
            largest_loss = min(losing_profits) if losing_profits else 0

            # Average trade duration
            durations = [trade.duration_minutes for trade in filtered_trades]
            avg_trade_duration_hours = np.mean(durations) / 60 if durations else 0

            # Portfolio metrics
            ending_balance = starting_balance + total_return
            peak_balance = (
                starting_balance + max(equity_curve)
                if equity_curve
                else starting_balance
            )

            # Risk metrics
            if returns:
                var_5pct = np.percentile(returns, 5) * starting_balance
                cvar_5pct = (
                    np.mean([r for r in returns if r <= np.percentile(returns, 5)])
                    * starting_balance
                )
            else:
                var_5pct = 0
                cvar_5pct = 0

            # Kelly criterion
            kelly_criterion = self._calculate_kelly_criterion(filtered_trades)

            # Monthly returns
            monthly_returns = self._calculate_period_returns(
                filtered_trades, TimeFrame.MONTHLY
            )

            # Create report
            report = PerformanceReport(
                period_start=start_date,
                period_end=end_date,
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                total_return=total_return,
                total_return_pct=total_return_pct,
                annualized_return_pct=annualized_return_pct,
                max_drawdown_pct=max_drawdown_pct,
                max_drawdown_duration_days=max_dd_duration,
                volatility_pct=volatility_pct,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                win_rate_pct=win_rate_pct,
                profit_factor=profit_factor,
                expectancy=expectancy,
                avg_win=avg_win,
                avg_loss=avg_loss,
                largest_win=largest_win,
                largest_loss=largest_loss,
                avg_trade_duration_hours=avg_trade_duration_hours,
                starting_balance=starting_balance,
                ending_balance=ending_balance,
                peak_balance=peak_balance,
                value_at_risk_5pct=var_5pct,
                conditional_var_5pct=cvar_5pct,
                kelly_criterion=kelly_criterion,
                monthly_returns=monthly_returns,
                drawdown_series=drawdown_series,
                equity_curve=equity_curve,
            )

            # Cache the report
            cache_key = f"report_{start_date}_{end_date}_{starting_balance}"
            self.analysis_cache[cache_key] = report
            self.last_analysis_time = datetime.now()

            self.logger.info(f"Generated performance report for {total_trades} trades")
            return report

        except Exception as e:
            self.logger.error(f"Error generating performance report: {e}")
            raise

    def analyze_by_symbol(self) -> Dict[str, Dict]:
        """Analyze performance by trading symbol"""
        try:
            symbol_analysis = {}

            # Group trades by symbol
            symbol_groups = {}
            for trade in self.trades:
                if trade.symbol not in symbol_groups:
                    symbol_groups[trade.symbol] = []
                symbol_groups[trade.symbol].append(trade)

            # Analyze each symbol
            for symbol, trades in symbol_groups.items():
                if len(trades) < 5:  # Minimum trades for meaningful analysis
                    continue

                total_trades = len(trades)
                winning_trades = len([t for t in trades if t.profit_loss > 0])
                total_profit = sum(trade.profit_loss for trade in trades)

                win_rate = winning_trades / total_trades
                avg_profit_per_trade = total_profit / total_trades

                # Risk metrics
                profits = [trade.profit_loss for trade in trades]
                volatility = np.std(profits)
                max_loss = min(profits)
                max_gain = max(profits)

                symbol_analysis[symbol] = {
                    "total_trades": total_trades,
                    "winning_trades": winning_trades,
                    "win_rate": win_rate,
                    "total_profit": total_profit,
                    "avg_profit_per_trade": avg_profit_per_trade,
                    "volatility": volatility,
                    "max_gain": max_gain,
                    "max_loss": max_loss,
                    "profit_factor": self._calculate_profit_factor(trades),
                }

            return symbol_analysis

        except Exception as e:
            self.logger.error(f"Error analyzing by symbol: {e}")
            return {}

    def analyze_by_signal_source(self) -> Dict[str, Dict]:
        """Analyze performance by signal source"""
        try:
            signal_analysis = {}

            # Group trades by signal source
            signal_groups = {}
            for trade in self.trades:
                source = trade.signal_source
                if source not in signal_groups:
                    signal_groups[source] = []
                signal_groups[source].append(trade)

            # Analyze each signal source
            for source, trades in signal_groups.items():
                if len(trades) < 3:  # Minimum trades
                    continue

                total_trades = len(trades)
                winning_trades = len([t for t in trades if t.profit_loss > 0])
                total_profit = sum(trade.profit_loss for trade in trades)

                # Calculate metrics
                win_rate = winning_trades / total_trades
                avg_profit = total_profit / total_trades
                avg_strength = np.mean([trade.signal_strength for trade in trades])
                avg_confidence = np.mean([trade.signal_confidence for trade in trades])

                signal_analysis[source] = {
                    "total_trades": total_trades,
                    "winning_trades": winning_trades,
                    "win_rate": win_rate,
                    "total_profit": total_profit,
                    "avg_profit_per_trade": avg_profit,
                    "avg_signal_strength": avg_strength,
                    "avg_signal_confidence": avg_confidence,
                    "profit_factor": self._calculate_profit_factor(trades),
                }

            return signal_analysis

        except Exception as e:
            self.logger.error(f"Error analyzing by signal source: {e}")
            return {}

    def analyze_trade_timing(self) -> Dict[str, Any]:
        """Analyze trade timing patterns"""
        try:
            timing_analysis = {}

            # Analyze by hour of day
            hourly_profits = defaultdict(list)
            for trade in self.trades:
                hour = trade.entry_time.hour
                hourly_profits[hour].append(trade.profit_loss)

            hour_analysis = {}
            for hour, profits in hourly_profits.items():
                if len(profits) >= 3:
                    hour_analysis[hour] = {
                        "total_trades": len(profits),
                        "avg_profit": np.mean(profits),
                        "win_rate": len([p for p in profits if p > 0]) / len(profits),
                        "total_profit": sum(profits),
                    }

            timing_analysis["hourly"] = hour_analysis

            # Analyze by day of week
            daily_profits = defaultdict(list)
            for trade in self.trades:
                day = trade.entry_time.strftime("%A")
                daily_profits[day].append(trade.profit_loss)

            day_analysis = {}
            for day, profits in daily_profits.items():
                if len(profits) >= 3:
                    day_analysis[day] = {
                        "total_trades": len(profits),
                        "avg_profit": np.mean(profits),
                        "win_rate": len([p for p in profits if p > 0]) / len(profits),
                        "total_profit": sum(profits),
                    }

            timing_analysis["daily"] = day_analysis

            # Analyze trade duration vs profit
            durations = [trade.duration_minutes for trade in self.trades]
            profits = [trade.profit_loss for trade in self.trades]

            if len(durations) > 10:
                correlation = np.corrcoef(durations, profits)[0, 1]
                timing_analysis["duration_correlation"] = correlation

            return timing_analysis

        except Exception as e:
            self.logger.error(f"Error analyzing trade timing: {e}")
            return {}

    def calculate_risk_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive risk metrics"""
        try:
            if not self.trades:
                return {}

            profits = [trade.profit_loss for trade in self.trades]
            returns = [trade.profit_loss_pct for trade in self.trades]

            risk_metrics = {}

            # Value at Risk (VaR)
            if len(returns) >= 10:
                risk_metrics["var_1pct"] = np.percentile(returns, 1)
                risk_metrics["var_5pct"] = np.percentile(returns, 5)
                risk_metrics["var_10pct"] = np.percentile(returns, 10)

                # Conditional VaR (Expected Shortfall)
                var_5 = risk_metrics["var_5pct"]
                risk_metrics["cvar_5pct"] = np.mean([r for r in returns if r <= var_5])

            # Maximum Adverse Excursion
            mae_values = [
                trade.max_adverse_excursion
                for trade in self.trades
                if trade.max_adverse_excursion != 0
            ]
            if mae_values:
                risk_metrics["avg_mae"] = np.mean(mae_values)
                risk_metrics["max_mae"] = max(mae_values)

            # Maximum Favorable Excursion
            mfe_values = [
                trade.max_favorable_excursion
                for trade in self.trades
                if trade.max_favorable_excursion != 0
            ]
            if mfe_values:
                risk_metrics["avg_mfe"] = np.mean(mfe_values)
                risk_metrics["max_mfe"] = max(mfe_values)

            # Risk-Reward Ratios
            rr_ratios = [
                trade.risk_reward_ratio
                for trade in self.trades
                if trade.risk_reward_ratio != 0
            ]
            if rr_ratios:
                risk_metrics["avg_risk_reward"] = np.mean(rr_ratios)
                risk_metrics["median_risk_reward"] = np.median(rr_ratios)

            # Consecutive losses
            consecutive_losses = self._calculate_consecutive_losses()
            risk_metrics["max_consecutive_losses"] = consecutive_losses[
                "max_consecutive"
            ]
            risk_metrics["avg_consecutive_losses"] = consecutive_losses[
                "avg_consecutive"
            ]

            # Ulcer Index
            equity_curve, drawdown_series = self._calculate_equity_curve(
                self.trades, self.config.default_starting_balance
            )
            if drawdown_series:
                squared_drawdowns = [dd**2 for dd in drawdown_series]
                risk_metrics["ulcer_index"] = np.sqrt(np.mean(squared_drawdowns))

            return risk_metrics

        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {e}")
            return {}

    def create_performance_charts(self, output_dir: str = "charts") -> List[str]:
        """Create comprehensive performance charts"""
        try:
            if not self.config.enable_charts or not self.trades:
                return []

            import os

            os.makedirs(output_dir, exist_ok=True)

            chart_files = []

            # 1. Equity Curve
            equity_curve, _ = self._calculate_equity_curve(
                self.trades, self.config.default_starting_balance
            )
            if equity_curve:
                plt.figure(figsize=self.config.figure_size)
                dates = [trade.exit_time for trade in self.trades]
                plt.plot(dates, equity_curve, linewidth=2, color="blue")
                plt.title("Equity Curve", fontsize=16, fontweight="bold")
                plt.xlabel("Date")
                plt.ylabel("Account Balance")
                plt.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()

                chart_file = os.path.join(output_dir, "equity_curve.png")
                plt.savefig(chart_file, dpi=300, bbox_inches="tight")
                plt.close()
                chart_files.append(chart_file)

            # 2. Drawdown Chart
            _, drawdown_series = self._calculate_equity_curve(
                self.trades, self.config.default_starting_balance
            )
            if drawdown_series:
                plt.figure(figsize=self.config.figure_size)
                dates = [trade.exit_time for trade in self.trades]
                plt.fill_between(dates, drawdown_series, 0, alpha=0.6, color="red")
                plt.title("Drawdown Series", fontsize=16, fontweight="bold")
                plt.xlabel("Date")
                plt.ylabel("Drawdown (%)")
                plt.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()

                chart_file = os.path.join(output_dir, "drawdown.png")
                plt.savefig(chart_file, dpi=300, bbox_inches="tight")
                plt.close()
                chart_files.append(chart_file)

            # 3. Monthly Returns Heatmap
            monthly_returns = self._get_monthly_returns_matrix()
            if monthly_returns is not None and not monthly_returns.empty:
                plt.figure(figsize=(12, 6))
                sns.heatmap(
                    monthly_returns,
                    annot=True,
                    fmt=".1%",
                    cmap="RdYlGn",
                    center=0,
                    cbar_kws={"label": "Return (%)"},
                )
                plt.title("Monthly Returns Heatmap", fontsize=16, fontweight="bold")
                plt.tight_layout()

                chart_file = os.path.join(output_dir, "monthly_returns_heatmap.png")
                plt.savefig(chart_file, dpi=300, bbox_inches="tight")
                plt.close()
                chart_files.append(chart_file)

            # 4. Trade Distribution
            profits = [trade.profit_loss for trade in self.trades]
            plt.figure(figsize=self.config.figure_size)
            plt.hist(profits, bins=50, alpha=0.7, color="skyblue", edgecolor="black")
            plt.axvline(x=0, color="red", linestyle="--", linewidth=2)
            plt.title("Trade P&L Distribution", fontsize=16, fontweight="bold")
            plt.xlabel("Profit/Loss")
            plt.ylabel("Frequency")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            chart_file = os.path.join(output_dir, "trade_distribution.png")
            plt.savefig(chart_file, dpi=300, bbox_inches="tight")
            plt.close()
            chart_files.append(chart_file)

            # 5. Performance by Symbol
            symbol_analysis = self.analyze_by_symbol()
            if symbol_analysis:
                symbols = list(symbol_analysis.keys())
                win_rates = [symbol_analysis[s]["win_rate"] for s in symbols]
                total_profits = [symbol_analysis[s]["total_profit"] for s in symbols]

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

                # Win rates
                ax1.bar(symbols, win_rates, color="lightgreen", alpha=0.7)
                ax1.set_title("Win Rate by Symbol", fontweight="bold")
                ax1.set_ylabel("Win Rate")
                ax1.set_xticklabels(symbols, rotation=45)
                ax1.grid(True, alpha=0.3)

                # Total profits
                colors = ["green" if p > 0 else "red" for p in total_profits]
                ax2.bar(symbols, total_profits, color=colors, alpha=0.7)
                ax2.set_title("Total Profit by Symbol", fontweight="bold")
                ax2.set_ylabel("Total Profit")
                ax2.set_xticklabels(symbols, rotation=45)
                ax2.grid(True, alpha=0.3)
                ax2.axhline(y=0, color="black", linestyle="-", linewidth=1)

                plt.tight_layout()

                chart_file = os.path.join(output_dir, "performance_by_symbol.png")
                plt.savefig(chart_file, dpi=300, bbox_inches="tight")
                plt.close()
                chart_files.append(chart_file)

            self.logger.info(f"Created {len(chart_files)} performance charts")
            return chart_files

        except Exception as e:
            self.logger.error(f"Error creating performance charts: {e}")
            return []

    def export_detailed_report(self, filename: str = None) -> str:
        """Export detailed performance report to file"""
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"performance_report_{timestamp}.json"

            # Generate comprehensive report
            report = self.generate_performance_report()
            symbol_analysis = self.analyze_by_symbol()
            signal_analysis = self.analyze_by_signal_source()
            timing_analysis = self.analyze_trade_timing()
            risk_metrics = self.calculate_risk_metrics()

            # Compile full report
            full_report = {
                "performance_summary": report.to_dict(),
                "symbol_analysis": symbol_analysis,
                "signal_analysis": signal_analysis,
                "timing_analysis": timing_analysis,
                "risk_metrics": risk_metrics,
                "trade_count": len(self.trades),
                "analysis_timestamp": datetime.now().isoformat(),
            }

            # Export to JSON
            with open(filename, "w") as f:
                json.dump(full_report, f, indent=2, default=str)

            self.logger.info(f"Exported detailed report to {filename}")
            return filename

        except Exception as e:
            self.logger.error(f"Error exporting detailed report: {e}")
            return ""

    def get_performance_alerts(self) -> List[Dict[str, Any]]:
        """Get performance-based alerts"""
        try:
            alerts = []

            if len(self.trades) < self.config.min_trades_for_analysis:
                return alerts

            # Generate recent report
            report = self.generate_performance_report()

            # Check drawdown alert
            if abs(report.max_drawdown_pct) > self.config.max_drawdown_alert_pct:
                alerts.append(
                    {
                        "type": "high_drawdown",
                        "severity": "critical",
                        "message": f"Maximum drawdown ({report.max_drawdown_pct:.1%}) exceeds threshold ({self.config.max_drawdown_alert_pct:.1%})",
                        "current_value": report.max_drawdown_pct,
                        "threshold": self.config.max_drawdown_alert_pct,
                    }
                )

            # Check Sharpe ratio alert
            if report.sharpe_ratio < self.config.min_sharpe_alert:
                alerts.append(
                    {
                        "type": "low_sharpe_ratio",
                        "severity": "warning",
                        "message": f"Sharpe ratio ({report.sharpe_ratio:.2f}) below minimum threshold ({self.config.min_sharpe_alert:.2f})",
                        "current_value": report.sharpe_ratio,
                        "threshold": self.config.min_sharpe_alert,
                    }
                )

            # Check win rate alert
            if report.win_rate_pct < self.config.min_win_rate_alert_pct:
                alerts.append(
                    {
                        "type": "low_win_rate",
                        "severity": "warning",
                        "message": f"Win rate ({report.win_rate_pct:.1%}) below minimum threshold ({self.config.min_win_rate_alert_pct:.1%})",
                        "current_value": report.win_rate_pct,
                        "threshold": self.config.min_win_rate_alert_pct,
                    }
                )

            # Check consecutive losses
            risk_metrics = self.calculate_risk_metrics()
            max_consecutive = risk_metrics.get("max_consecutive_losses", 0)
            if max_consecutive >= 5:
                alerts.append(
                    {
                        "type": "consecutive_losses",
                        "severity": "warning",
                        "message": f"Maximum consecutive losses: {max_consecutive}",
                        "current_value": max_consecutive,
                        "threshold": 5,
                    }
                )

            return alerts

        except Exception as e:
            self.logger.error(f"Error getting performance alerts: {e}")
            return []

    # Helper methods
    def _filter_trades_by_date(
        self, start_date: Optional[datetime], end_date: Optional[datetime]
    ) -> List[TradeRecord]:
        """Filter trades by date range"""
        filtered_trades = self.trades.copy()

        if start_date:
            filtered_trades = [t for t in filtered_trades if t.exit_time >= start_date]

        if end_date:
            filtered_trades = [t for t in filtered_trades if t.exit_time <= end_date]

        return filtered_trades

    def _calculate_equity_curve(
        self, trades: List[TradeRecord], starting_balance: float
    ) -> Tuple[List[float], List[float]]:
        """Calculate equity curve and drawdown series"""
        try:
            equity_curve = []
            drawdown_series = []

            current_balance = starting_balance
            peak_balance = starting_balance

            for trade in sorted(trades, key=lambda x: x.exit_time):
                current_balance += trade.profit_loss - trade.commission - trade.swap
                equity_curve.append(current_balance)

                # Update peak
                if current_balance > peak_balance:
                    peak_balance = current_balance

                # Calculate drawdown
                drawdown = (current_balance - peak_balance) / peak_balance
                drawdown_series.append(drawdown)

            return equity_curve, drawdown_series

        except Exception as e:
            self.logger.error(f"Error calculating equity curve: {e}")
            return [], []

    def _calculate_period_returns(
        self, trades: List[TradeRecord], timeframe: TimeFrame
    ) -> List[float]:
        """Calculate returns for specified time period"""
        try:
            if not trades:
                return []

            # Sort trades by exit time
            sorted_trades = sorted(trades, key=lambda x: x.exit_time)

            # Group trades by period
            period_groups = {}

            for trade in sorted_trades:
                if timeframe == TimeFrame.DAILY:
                    period_key = trade.exit_time.date()
                elif timeframe == TimeFrame.WEEKLY:
                    period_key = trade.exit_time.isocalendar()[:2]  # (year, week)
                elif timeframe == TimeFrame.MONTHLY:
                    period_key = (trade.exit_time.year, trade.exit_time.month)
                elif timeframe == TimeFrame.QUARTERLY:
                    period_key = (
                        trade.exit_time.year,
                        (trade.exit_time.month - 1) // 3 + 1,
                    )
                elif timeframe == TimeFrame.YEARLY:
                    period_key = trade.exit_time.year
                else:
                    period_key = "all"

                if period_key not in period_groups:
                    period_groups[period_key] = []
                period_groups[period_key].append(trade)

            # Calculate returns for each period
            period_returns = []
            for period_trades in period_groups.values():
                period_profit = sum(trade.profit_loss for trade in period_trades)
                # Assume same starting balance for each period (simplified)
                period_return = period_profit / self.config.default_starting_balance
                period_returns.append(period_return)

            return period_returns

        except Exception as e:
            self.logger.error(f"Error calculating period returns: {e}")
            return []

    def _calculate_max_drawdown_duration(self, drawdown_series: List[float]) -> int:
        """Calculate maximum drawdown duration in days"""
        try:
            if not drawdown_series:
                return 0

            max_duration = 0
            current_duration = 0

            for dd in drawdown_series:
                if dd < 0:  # In drawdown
                    current_duration += 1
                else:  # Out of drawdown
                    max_duration = max(max_duration, current_duration)
                    current_duration = 0

            # Check if still in drawdown at the end
            max_duration = max(max_duration, current_duration)

            return max_duration

        except Exception as e:
            self.logger.error(f"Error calculating max drawdown duration: {e}")
            return 0

    def _calculate_kelly_criterion(self, trades: List[TradeRecord]) -> float:
        """Calculate Kelly Criterion for optimal position sizing"""
        try:
            winning_trades = [t for t in trades if t.profit_loss > 0]
            losing_trades = [t for t in trades if t.profit_loss < 0]

            if not winning_trades or not losing_trades:
                return 0

            win_rate = len(winning_trades) / len(trades)
            avg_win = np.mean([t.profit_loss_pct for t in winning_trades])
            avg_loss = abs(np.mean([t.profit_loss_pct for t in losing_trades]))

            if avg_loss == 0:
                return 0

            # Kelly formula: f = (bp - q) / b
            # where b = avg_win/avg_loss, p = win_rate, q = 1 - win_rate
            b = avg_win / avg_loss
            kelly = (b * win_rate - (1 - win_rate)) / b

            # Cap at reasonable levels
            return max(-1, min(1, kelly))

        except Exception as e:
            self.logger.error(f"Error calculating Kelly criterion: {e}")
            return 0

    def _calculate_profit_factor(self, trades: List[TradeRecord]) -> float:
        """Calculate profit factor for a set of trades"""
        try:
            winning_profits = sum(t.profit_loss for t in trades if t.profit_loss > 0)
            losing_profits = abs(
                sum(t.profit_loss for t in trades if t.profit_loss < 0)
            )

            if losing_profits == 0:
                return float("inf") if winning_profits > 0 else 0

            return winning_profits / losing_profits

        except Exception as e:
            self.logger.error(f"Error calculating profit factor: {e}")
            return 0

    def _calculate_consecutive_losses(self) -> Dict[str, float]:
        """Calculate consecutive loss statistics"""
        try:
            consecutive_counts = []
            current_consecutive = 0

            for trade in sorted(self.trades, key=lambda x: x.exit_time):
                if trade.profit_loss < 0:
                    current_consecutive += 1
                else:
                    if current_consecutive > 0:
                        consecutive_counts.append(current_consecutive)
                    current_consecutive = 0

            # Add final consecutive count if ended on losses
            if current_consecutive > 0:
                consecutive_counts.append(current_consecutive)

            if not consecutive_counts:
                return {"max_consecutive": 0, "avg_consecutive": 0}

            return {
                "max_consecutive": max(consecutive_counts),
                "avg_consecutive": np.mean(consecutive_counts),
            }

        except Exception as e:
            self.logger.error(f"Error calculating consecutive losses: {e}")
            return {"max_consecutive": 0, "avg_consecutive": 0}

    def _get_monthly_returns_matrix(self) -> Optional[pd.DataFrame]:
        """Get monthly returns as a matrix for heatmap"""
        try:
            if not self.trades:
                return None

            # Group trades by month
            monthly_profits = {}
            for trade in self.trades:
                year_month = (trade.exit_time.year, trade.exit_time.month)
                if year_month not in monthly_profits:
                    monthly_profits[year_month] = 0
                monthly_profits[year_month] += trade.profit_loss

            # Convert to returns (assuming fixed starting balance)
            monthly_returns = {
                year_month: profit / self.config.default_starting_balance
                for year_month, profit in monthly_profits.items()
            }

            # Create DataFrame
            data = []
            for (year, month), return_pct in monthly_returns.items():
                data.append({"Year": year, "Month": month, "Return": return_pct})

            if not data:
                return None

            df = pd.DataFrame(data)

            # Pivot to create matrix
            matrix = df.pivot(index="Year", columns="Month", values="Return")

            # Reorder columns to be calendar months
            month_order = list(range(1, 13))
            matrix = matrix.reindex(columns=month_order, fill_value=0)

            # Rename columns to month names
            month_names = [
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
            ]
            matrix.columns = month_names

            return matrix

        except Exception as e:
            self.logger.error(f"Error creating monthly returns matrix: {e}")
            return None

    def _initialize_database(self):
        """Initialize database for persistence"""
        try:
            self.db_conn = sqlite3.connect(self.config.db_path, check_same_thread=False)

            # Create trades table
            self.db_conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL NOT NULL,
                    volume REAL NOT NULL,
                    profit_loss REAL NOT NULL,
                    profit_loss_pct REAL NOT NULL,
                    commission REAL DEFAULT 0,
                    swap REAL DEFAULT 0,
                    duration_minutes INTEGER DEFAULT 0,
                    signal_source TEXT DEFAULT 'unknown',
                    signal_strength REAL DEFAULT 0,
                    signal_confidence REAL DEFAULT 0,
                    stop_loss REAL,
                    take_profit REAL,
                    exit_reason TEXT DEFAULT 'unknown',
                    risk_amount REAL DEFAULT 0,
                    risk_reward_ratio REAL DEFAULT 0,
                    max_adverse_excursion REAL DEFAULT 0,
                    max_favorable_excursion REAL DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            self.db_conn.commit()

        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
            self.db_conn = None

    def _persist_trade(self, trade: TradeRecord):
        """Persist trade to database"""
        try:
            if not self.db_conn:
                return

            self.db_conn.execute(
                """
                INSERT OR REPLACE INTO trades VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    trade.trade_id,
                    trade.symbol,
                    trade.side,
                    trade.entry_time.isoformat(),
                    trade.exit_time.isoformat(),
                    trade.entry_price,
                    trade.exit_price,
                    trade.volume,
                    trade.profit_loss,
                    trade.profit_loss_pct,
                    trade.commission,
                    trade.swap,
                    trade.duration_minutes,
                    trade.signal_source,
                    trade.signal_strength,
                    trade.signal_confidence,
                    trade.stop_loss,
                    trade.take_profit,
                    trade.exit_reason,
                    trade.risk_amount,
                    trade.risk_reward_ratio,
                    trade.max_adverse_excursion,
                    trade.max_favorable_excursion,
                    datetime.now().isoformat(),
                ),
            )

            self.db_conn.commit()

        except Exception as e:
            self.logger.error(f"Error persisting trade: {e}")


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create analyzer
    config = AnalyzerConfig()
    analyzer = PerformanceAnalyzer(config)

    # Create sample trades
    import random
    from datetime import timedelta

    start_date = datetime.now() - timedelta(days=90)

    for i in range(100):
        entry_time = start_date + timedelta(
            days=random.randint(0, 90), hours=random.randint(0, 23)
        )
        exit_time = entry_time + timedelta(minutes=random.randint(30, 1440))

        side = random.choice(["buy", "sell"])
        entry_price = 1.1000 + random.uniform(-0.0100, 0.0100)

        # Simulate 60% win rate
        if random.random() < 0.6:  # Winning trade
            if side == "buy":
                exit_price = entry_price + random.uniform(0.0010, 0.0050)
            else:
                exit_price = entry_price - random.uniform(0.0010, 0.0050)
        else:  # Losing trade
            if side == "buy":
                exit_price = entry_price - random.uniform(0.0010, 0.0030)
            else:
                exit_price = entry_price + random.uniform(0.0010, 0.0030)

        volume = 0.1
        if side == "buy":
            profit_loss = (exit_price - entry_price) * volume * 100000  # Standard lot
        else:
            profit_loss = (entry_price - exit_price) * volume * 100000

        profit_loss_pct = profit_loss / (entry_price * volume * 100000)

        trade = TradeRecord(
            trade_id=f"TRADE_{i+1:03d}",
            symbol="EURUSD",
            side=side,
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=entry_price,
            exit_price=exit_price,
            volume=volume,
            profit_loss=profit_loss,
            profit_loss_pct=profit_loss_pct,
            signal_source=random.choice(
                ["RSI", "MACD", "Bollinger", "Support_Resistance"]
            ),
            signal_strength=random.uniform(0.5, 1.0),
            signal_confidence=random.uniform(0.6, 1.0),
        )

        analyzer.add_trade(trade)

    # Generate performance report
    print("Generating performance report...")
    report = analyzer.generate_performance_report()

    # Display key metrics
    print(f"\n=== PERFORMANCE SUMMARY ===")
    print(f"Total Trades: {report.total_trades}")
    print(f"Win Rate: {report.win_rate_pct:.1%}")
    print(f"Total Return: {report.total_return_pct:.1%}")
    print(f"Annualized Return: {report.annualized_return_pct:.1%}")
    print(f"Max Drawdown: {report.max_drawdown_pct:.1%}")
    print(f"Sharpe Ratio: {report.sharpe_ratio:.2f}")
    print(f"Profit Factor: {report.profit_factor:.2f}")
    print(f"Expectancy: ${report.expectancy:.2f}")

    # Analyze by signal source
    signal_analysis = analyzer.analyze_by_signal_source()
    print(f"\n=== SIGNAL SOURCE ANALYSIS ===")
    for source, metrics in signal_analysis.items():
        print(
            f"{source}: {metrics['win_rate']:.1%} win rate, ${metrics['total_profit']:.2f} profit"
        )

    # Get performance alerts
    alerts = analyzer.get_performance_alerts()
    if alerts:
        print(f"\n=== PERFORMANCE ALERTS ===")
        for alert in alerts:
            print(f"{alert['severity'].upper()}: {alert['message']}")

    # Export detailed report
    report_file = analyzer.export_detailed_report()
    print(f"\nDetailed report exported to: {report_file}")

    # Create charts
    chart_files = analyzer.create_performance_charts()
    if chart_files:
        print(f"Created {len(chart_files)} performance charts")
