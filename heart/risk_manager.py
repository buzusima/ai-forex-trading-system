"""
Institutional-Grade Risk Management System for Forex Trading
Advanced position sizing, drawdown control, and risk monitoring
Author: Senior AI Developer
Version: 1.0.0 - Production Ready
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
import threading
from collections import deque, defaultdict
import math
import warnings
warnings.filterwarnings('ignore')

# Import configurations
import sys
sys.path.append('..')
from config.settings import RiskConfig
from sensor.mt5_connector import MT5Connector, TickData

class RiskLevel(Enum):
    """Risk level classifications"""
    VERY_LOW = "very_low"      # 0.1-0.5% per trade
    LOW = "low"                # 0.5-1.0% per trade  
    MODERATE = "moderate"      # 1.0-2.0% per trade
    HIGH = "high"              # 2.0-3.0% per trade
    VERY_HIGH = "very_high"    # 3.0%+ per trade

class PositionSizeMethod(Enum):
    """Position sizing methods"""
    FIXED_LOT = "fixed_lot"
    FIXED_PERCENT = "fixed_percent"
    KELLY_CRITERION = "kelly_criterion"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    ATR_BASED = "atr_based"
    SHARPE_OPTIMAL = "sharpe_optimal"
    VAR_BASED = "var_based"

class RiskEvent(Enum):
    """Risk event types"""
    DRAWDOWN_WARNING = "drawdown_warning"
    DRAWDOWN_LIMIT = "drawdown_limit"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    POSITION_LIMIT = "position_limit"
    CORRELATION_WARNING = "correlation_warning"
    VOLATILITY_SPIKE = "volatility_spike"
    MARGIN_WARNING = "margin_warning"

@dataclass
class RiskMetrics:
    """Real-time risk metrics"""
    timestamp: datetime
    total_exposure: float
    max_drawdown: float
    current_drawdown: float
    daily_pnl: float
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    sortino_ratio: float
    var_95: float  # Value at Risk 95%
    expected_shortfall: float
    correlation_risk: float
    margin_utilization: float
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'total_exposure': self.total_exposure,
            'max_drawdown': self.max_drawdown,
            'current_drawdown': self.current_drawdown,
            'daily_pnl': self.daily_pnl,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'var_95': self.var_95,
            'expected_shortfall': self.expected_shortfall,
            'correlation_risk': self.correlation_risk,
            'margin_utilization': self.margin_utilization
        }

@dataclass
class PositionSizeResult:
    """Position sizing calculation result"""
    recommended_lots: float
    risk_amount: float
    stop_loss_pips: float
    take_profit_pips: Optional[float]
    risk_reward_ratio: float
    confidence_level: float
    method_used: PositionSizeMethod
    warnings: List[str] = field(default_factory=list)

@dataclass
class TradeRiskAssessment:
    """Individual trade risk assessment"""
    symbol: str
    trade_type: str  # BUY/SELL
    entry_price: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    position_size: float
    risk_score: float  # 0-100
    confidence_score: float  # 0-100
    correlation_risk: float
    market_risk: float
    approved: bool
    rejection_reasons: List[str] = field(default_factory=list)

class RiskManager:
    """
    Institutional-Grade Risk Management System
    Comprehensive risk control for algorithmic trading
    """
    
    def __init__(self, config: Optional[RiskConfig] = None, mt5_connector: Optional[MT5Connector] = None):
        self.config = config or RiskConfig()
        self.mt5_connector = mt5_connector
        self.logger = self._setup_logger()
        
        # Risk state tracking
        self.account_balance = 0.0
        self.account_equity = 0.0
        self.initial_balance = 0.0
        self.peak_balance = 0.0
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        
        # Daily tracking
        self.daily_start_balance = 0.0
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.daily_wins = 0
        self.daily_losses = 0
        
        # Position tracking
        self.open_positions = {}
        self.total_exposure = 0.0
        self.symbol_exposure = defaultdict(float)
        self.correlation_matrix = {}
        
        # Performance tracking
        self.trade_history = deque(maxlen=1000)  # Last 1000 trades
        self.pnl_history = deque(maxlen=10000)   # Last 10000 data points
        self.returns_history = deque(maxlen=252) # Last 252 trading days
        
        # Risk events
        self.risk_events = deque(maxlen=100)
        self.risk_breaches = {}
        
        # Threading
        self.lock = threading.Lock()
        
        # Initialize
        self._initialize_risk_state()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup dedicated logger for risk manager"""
        logger = logging.getLogger("RiskManager")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.FileHandler("logs/risk_manager.log")
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _initialize_risk_state(self) -> None:
        """Initialize risk management state"""
        try:
            if self.mt5_connector and self.mt5_connector.is_connected():
                account_info = self.mt5_connector.get_account_info()
                if account_info:
                    self.account_balance = account_info['balance']
                    self.account_equity = account_info['equity']
                    self.initial_balance = self.account_balance
                    self.peak_balance = self.account_balance
                    self.daily_start_balance = self.account_balance
                    
                    self.logger.info(f"✅ Risk state initialized - Balance: {self.account_balance:.2f}")
            
        except Exception as e:
            self.logger.error(f"Risk state initialization error: {str(e)}")
    
    def update_account_state(self) -> None:
        """Update account state from MT5"""
        try:
            if not self.mt5_connector or not self.mt5_connector.is_connected():
                return
            
            with self.lock:
                account_info = self.mt5_connector.get_account_info()
                if account_info:
                    self.account_balance = account_info['balance']
                    self.account_equity = account_info['equity']
                    
                    # Update peak balance
                    if self.account_balance > self.peak_balance:
                        self.peak_balance = self.account_balance
                    
                    # Calculate current drawdown
                    self.current_drawdown = (self.peak_balance - self.account_balance) / self.peak_balance
                    
                    # Update max drawdown
                    if self.current_drawdown > self.max_drawdown:
                        self.max_drawdown = self.current_drawdown
                    
                    # Update daily PnL
                    self.daily_pnl = self.account_balance - self.daily_start_balance
                    
                    # Add to PnL history
                    self.pnl_history.append({
                        'timestamp': datetime.now(),
                        'balance': self.account_balance,
                        'equity': self.account_equity,
                        'pnl': self.daily_pnl,
                        'drawdown': self.current_drawdown
                    })
                
                # Update positions
                self._update_position_exposure()
                
        except Exception as e:
            self.logger.error(f"Account state update error: {str(e)}")
    
    def _update_position_exposure(self) -> None:
        """Update position exposure tracking"""
        try:
            if not self.mt5_connector:
                return
            
            positions = self.mt5_connector.get_positions()
            self.open_positions = {}
            self.total_exposure = 0.0
            self.symbol_exposure.clear()
            
            for pos in positions:
                symbol = pos['symbol']
                volume = pos['volume']
                price = pos['price_open']
                pos_type = pos['type']
                
                # Calculate exposure
                exposure = volume * price * 100000  # Standard lot conversion
                
                self.open_positions[pos['ticket']] = {
                    'symbol': symbol,
                    'volume': volume,
                    'exposure': exposure,
                    'type': pos_type,
                    'price': price,
                    'profit': pos.get('profit', 0)
                }
                
                self.total_exposure += exposure
                self.symbol_exposure[symbol] += exposure
                
        except Exception as e:
            self.logger.error(f"Position exposure update error: {str(e)}")
    
    def calculate_position_size(self, symbol: str, entry_price: float, 
                              stop_loss: float, risk_percent: Optional[float] = None,
                              method: PositionSizeMethod = PositionSizeMethod.VOLATILITY_ADJUSTED) -> PositionSizeResult:
        """
        Calculate optimal position size based on risk parameters
        
        Args:
            symbol: Trading symbol
            entry_price: Planned entry price
            stop_loss: Stop loss price
            risk_percent: Risk percentage (overrides config if provided)
            method: Position sizing method
        
        Returns:
            PositionSizeResult with recommended position size
        """
        try:
            warnings = []
            risk_percent = risk_percent or self.config.risk_per_trade
            
            # Get symbol information
            symbol_info = self.mt5_connector.get_symbol_info(symbol) if self.mt5_connector else None
            if not symbol_info:
                warnings.append("Symbol information not available")
                return PositionSizeResult(
                    recommended_lots=0.0,
                    risk_amount=0.0,
                    stop_loss_pips=0.0,
                    take_profit_pips=None,
                    risk_reward_ratio=0.0,
                    confidence_level=0.0,
                    method_used=method,
                    warnings=warnings
                )
            
            # Calculate stop loss distance in pips
            stop_loss_pips = abs(entry_price - stop_loss) / symbol_info.point
            
            if stop_loss_pips == 0:
                warnings.append("Stop loss distance is zero")
                return PositionSizeResult(
                    recommended_lots=0.0,
                    risk_amount=0.0,
                    stop_loss_pips=0.0,
                    take_profit_pips=None,
                    risk_reward_ratio=0.0,
                    confidence_level=0.0,
                    method_used=method,
                    warnings=warnings
                )
            
            # Calculate risk amount
            risk_amount = self.account_balance * (risk_percent / 100)
            
            # Apply position sizing method
            if method == PositionSizeMethod.FIXED_LOT:
                recommended_lots = self.config.fixed_lot_size
                
            elif method == PositionSizeMethod.FIXED_PERCENT:
                # Standard percentage-based sizing
                pip_value = symbol_info.tick_value
                recommended_lots = risk_amount / (stop_loss_pips * pip_value)
                
            elif method == PositionSizeMethod.VOLATILITY_ADJUSTED:
                # Adjust for recent volatility
                volatility_multiplier = self._calculate_volatility_adjustment(symbol)
                pip_value = symbol_info.tick_value
                base_lots = risk_amount / (stop_loss_pips * pip_value)
                recommended_lots = base_lots * volatility_multiplier
                
            elif method == PositionSizeMethod.ATR_BASED:
                # ATR-based position sizing
                atr = self._get_atr(symbol)
                if atr > 0:
                    atr_multiplier = min(2.0, max(0.5, stop_loss_pips / (atr * 10000)))
                    pip_value = symbol_info.tick_value
                    base_lots = risk_amount / (stop_loss_pips * pip_value)
                    recommended_lots = base_lots * atr_multiplier
                else:
                    # Fallback to fixed percent
                    pip_value = symbol_info.tick_value
                    recommended_lots = risk_amount / (stop_loss_pips * pip_value)
                    
            elif method == PositionSizeMethod.KELLY_CRITERION:
                # Kelly Criterion sizing
                kelly_fraction = self._calculate_kelly_fraction(symbol)
                recommended_lots = (self.account_balance * kelly_fraction) / (entry_price * 100000)
                
            elif method == PositionSizeMethod.VAR_BASED:
                # VaR-based sizing
                var_adjustment = self._calculate_var_adjustment(symbol)
                pip_value = symbol_info.tick_value
                base_lots = risk_amount / (stop_loss_pips * pip_value)
                recommended_lots = base_lots * var_adjustment
                
            else:
                # Default to fixed percent
                pip_value = symbol_info.tick_value
                recommended_lots = risk_amount / (stop_loss_pips * pip_value)
            
            # Apply constraints
            recommended_lots = max(symbol_info.min_lot, 
                                 min(symbol_info.max_lot, recommended_lots))
            
            # Round to lot step
            recommended_lots = round(recommended_lots / symbol_info.lot_step) * symbol_info.lot_step
            
            # Check position limits
            if self._check_position_limits(symbol, recommended_lots):
                warnings.append("Position size exceeds limits")
                recommended_lots *= 0.5  # Reduce by half if limits exceeded
            
            # Calculate confidence level
            confidence_level = self._calculate_confidence_level(symbol, method)
            
            return PositionSizeResult(
                recommended_lots=recommended_lots,
                risk_amount=risk_amount,
                stop_loss_pips=stop_loss_pips,
                take_profit_pips=None,
                risk_reward_ratio=0.0,
                confidence_level=confidence_level,
                method_used=method,
                warnings=warnings
            )
            
        except Exception as e:
            self.logger.error(f"Position size calculation error: {str(e)}")
            return PositionSizeResult(
                recommended_lots=0.0,
                risk_amount=0.0,
                stop_loss_pips=0.0,
                take_profit_pips=None,
                risk_reward_ratio=0.0,
                confidence_level=0.0,
                method_used=method,
                warnings=[f"Calculation error: {str(e)}"]
            )
    
    def assess_trade_risk(self, symbol: str, trade_type: str, entry_price: float,
                         stop_loss: Optional[float] = None, take_profit: Optional[float] = None,
                         position_size: float = 0.0) -> TradeRiskAssessment:
        """
        Comprehensive trade risk assessment
        
        Args:
            symbol: Trading symbol
            trade_type: BUY or SELL
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            position_size: Position size in lots
        
        Returns:
            TradeRiskAssessment with detailed risk analysis
        """
        try:
            rejection_reasons = []
            risk_score = 0.0
            confidence_score = 100.0
            
            # Update account state
            self.update_account_state()
            
            # Check daily loss limit
            if self.daily_pnl < -self.config.daily_loss_limit:
                rejection_reasons.append("Daily loss limit exceeded")
                risk_score += 30
            
            # Check drawdown limits
            if self.current_drawdown > self.config.max_drawdown_percent / 100:
                rejection_reasons.append("Maximum drawdown exceeded")
                risk_score += 40
            
            # Check total exposure
            symbol_info = self.mt5_connector.get_symbol_info(symbol) if self.mt5_connector else None
            if symbol_info and position_size > 0:
                new_exposure = position_size * entry_price * 100000
                total_new_exposure = self.total_exposure + new_exposure
                
                if total_new_exposure > self.account_balance * self.config.max_total_exposure:
                    rejection_reasons.append("Total exposure limit exceeded")
                    risk_score += 25
            
            # Check symbol exposure
            current_symbol_exposure = self.symbol_exposure.get(symbol, 0)
            max_symbol_exposure = self.account_balance * self.config.max_symbol_exposure
            
            if symbol_info and position_size > 0:
                new_symbol_exposure = current_symbol_exposure + (position_size * entry_price * 100000)
                if new_symbol_exposure > max_symbol_exposure:
                    rejection_reasons.append("Symbol exposure limit exceeded")
                    risk_score += 20
            
            # Check correlation risk
            correlation_risk = self._calculate_correlation_risk(symbol, trade_type)
            if correlation_risk > 0.7:
                rejection_reasons.append("High correlation risk with existing positions")
                risk_score += 15
            
            # Check market risk
            market_risk = self._calculate_market_risk(symbol)
            if market_risk > 0.8:
                rejection_reasons.append("High market volatility detected")
                risk_score += 10
            
            # Check margin requirements
            if self._check_margin_requirements(symbol, position_size, entry_price):
                rejection_reasons.append("Insufficient margin")
                risk_score += 35
            
            # Adjust confidence based on risk factors
            confidence_score = max(0, 100 - risk_score)
            
            # Final approval decision
            approved = (
                len(rejection_reasons) == 0 and
                risk_score < 50 and
                confidence_score > 60
            )
            
            return TradeRiskAssessment(
                symbol=symbol,
                trade_type=trade_type,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                risk_score=risk_score,
                confidence_score=confidence_score,
                correlation_risk=correlation_risk,
                market_risk=market_risk,
                approved=approved,
                rejection_reasons=rejection_reasons
            )
            
        except Exception as e:
            self.logger.error(f"Trade risk assessment error: {str(e)}")
            return TradeRiskAssessment(
                symbol=symbol,
                trade_type=trade_type,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                risk_score=100.0,
                confidence_score=0.0,
                correlation_risk=1.0,
                market_risk=1.0,
                approved=False,
                rejection_reasons=[f"Assessment error: {str(e)}"]
            )
    
    def get_risk_metrics(self) -> RiskMetrics:
        """Get current risk metrics"""
        try:
            self.update_account_state()
            
            # Calculate performance metrics
            win_rate = self._calculate_win_rate()
            profit_factor = self._calculate_profit_factor()
            sharpe_ratio = self._calculate_sharpe_ratio()
            sortino_ratio = self._calculate_sortino_ratio()
            var_95 = self._calculate_var_95()
            expected_shortfall = self._calculate_expected_shortfall()
            correlation_risk = self._calculate_portfolio_correlation_risk()
            margin_utilization = self._calculate_margin_utilization()
            
            return RiskMetrics(
                timestamp=datetime.now(),
                total_exposure=self.total_exposure,
                max_drawdown=self.max_drawdown,
                current_drawdown=self.current_drawdown,
                daily_pnl=self.daily_pnl,
                win_rate=win_rate,
                profit_factor=profit_factor,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                var_95=var_95,
                expected_shortfall=expected_shortfall,
                correlation_risk=correlation_risk,
                margin_utilization=margin_utilization
            )
            
        except Exception as e:
            self.logger.error(f"Risk metrics calculation error: {str(e)}")
            return RiskMetrics(
                timestamp=datetime.now(),
                total_exposure=0.0,
                max_drawdown=0.0,
                current_drawdown=0.0,
                daily_pnl=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                var_95=0.0,
                expected_shortfall=0.0,
                correlation_risk=0.0,
                margin_utilization=0.0
            )
    
    def check_risk_limits(self) -> List[RiskEvent]:
        """Check all risk limits and return violations"""
        violations = []
        
        try:
            self.update_account_state()
            
            # Check drawdown limits
            if self.current_drawdown > self.config.max_drawdown_percent / 100:
                violations.append(RiskEvent.DRAWDOWN_LIMIT)
                self._log_risk_event(RiskEvent.DRAWDOWN_LIMIT, 
                                   f"Drawdown: {self.current_drawdown:.2%}")
            
            elif self.current_drawdown > (self.config.max_drawdown_percent / 100) * 0.8:
                violations.append(RiskEvent.DRAWDOWN_WARNING)
                self._log_risk_event(RiskEvent.DRAWDOWN_WARNING,
                                   f"Drawdown: {self.current_drawdown:.2%}")
            
            # Check daily loss limit
            daily_loss_percent = abs(self.daily_pnl) / self.account_balance
            if self.daily_pnl < 0 and daily_loss_percent > self.config.daily_loss_limit / 100:
                violations.append(RiskEvent.DAILY_LOSS_LIMIT)
                self._log_risk_event(RiskEvent.DAILY_LOSS_LIMIT,
                                   f"Daily loss: {daily_loss_percent:.2%}")
            
            # Check position limits
            if self.total_exposure > self.account_balance * self.config.max_total_exposure:
                violations.append(RiskEvent.POSITION_LIMIT)
                self._log_risk_event(RiskEvent.POSITION_LIMIT,
                                   f"Total exposure: {self.total_exposure:.2f}")
            
            # Check correlation risk
            correlation_risk = self._calculate_portfolio_correlation_risk()
            if correlation_risk > 0.8:
                violations.append(RiskEvent.CORRELATION_WARNING)
                self._log_risk_event(RiskEvent.CORRELATION_WARNING,
                                   f"Correlation risk: {correlation_risk:.2f}")
            
            # Check margin utilization
            margin_util = self._calculate_margin_utilization()
            if margin_util > 0.9:
                violations.append(RiskEvent.MARGIN_WARNING)
                self._log_risk_event(RiskEvent.MARGIN_WARNING,
                                   f"Margin utilization: {margin_util:.2%}")
            
        except Exception as e:
            self.logger.error(f"Risk limit check error: {str(e)}")
        
        return violations
    
    def emergency_stop(self) -> bool:
        """Emergency stop - close all positions"""
        try:
            self.logger.warning("🚨 EMERGENCY STOP INITIATED")
            
            if not self.mt5_connector or not self.mt5_connector.is_connected():
                self.logger.error("MT5 not connected for emergency stop")
                return False
            
            positions = self.mt5_connector.get_positions()
            closed_count = 0
            
            for position in positions:
                try:
                    result = self.mt5_connector.close_position(position['ticket'])
                    if result.success:
                        closed_count += 1
                        self.logger.info(f"✅ Emergency closed position {position['ticket']}")
                    else:
                        self.logger.error(f"❌ Failed to close position {position['ticket']}: {result.error_message}")
                except Exception as e:
                    self.logger.error(f"Error closing position {position['ticket']}: {str(e)}")
            
            self.logger.warning(f"🚨 Emergency stop completed - Closed {closed_count} positions")
            return closed_count > 0
            
        except Exception as e:
            self.logger.error(f"Emergency stop error: {str(e)}")
            return False
    
    # Helper methods
    def _calculate_volatility_adjustment(self, symbol: str) -> float:
        """Calculate volatility-based adjustment factor"""
        try:
            if not self.mt5_connector:
                return 1.0
            
            # Get recent price data
            rates = self.mt5_connector.get_rates(symbol, 1, 20)  # 20 periods
            if rates is None or len(rates) < 10:
                return 1.0
            
            # Calculate volatility
            returns = rates['close'].pct_change().dropna()
            volatility = returns.std()
            
            # Normalize volatility (assuming 0.01 as baseline)
            baseline_vol = 0.01
            vol_ratio = volatility / baseline_vol
            
            # Return adjustment factor (inverse relationship)
            return max(0.5, min(2.0, 1.0 / vol_ratio))
            
        except Exception:
            return 1.0
    
    def _get_atr(self, symbol: str, period: int = 14) -> float:
        """Calculate Average True Range"""
        try:
            if not self.mt5_connector:
                return 0.0
            
            rates = self.mt5_connector.get_rates(symbol, 1, period + 1)
            if rates is None or len(rates) < period:
                return 0.0
            
            high = rates['high']
            low = rates['low']
            close = rates['close'].shift(1)
            
            tr1 = high - low
            tr2 = (high - close).abs()
            tr3 = (low - close).abs()
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(period).mean().iloc[-1]
            
            return atr if not pd.isna(atr) else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_kelly_fraction(self, symbol: str) -> float:
        """Calculate Kelly Criterion fraction"""
        try:
            # Simplified Kelly calculation based on recent trades
            if len(self.trade_history) < 10:
                return 0.02  # Conservative default
            
            # Get recent trades for this symbol
            symbol_trades = [t for t in self.trade_history if t.get('symbol') == symbol]
            if len(symbol_trades) < 5:
                return 0.02
            
            wins = [t['profit'] for t in symbol_trades if t['profit'] > 0]
            losses = [abs(t['profit']) for t in symbol_trades if t['profit'] < 0]
            
            if not wins or not losses:
                return 0.02
            
            win_rate = len(wins) / len(symbol_trades)
            avg_win = np.mean(wins)
            avg_loss = np.mean(losses)
            
            if avg_loss == 0:
                return 0.02
            
            # Kelly formula: f = (bp - q) / b
            # where b = avg_win/avg_loss, p = win_rate, q = 1-win_rate
            b = avg_win / avg_loss
            kelly_fraction = (b * win_rate - (1 - win_rate)) / b
            
            # Limit Kelly fraction for safety
            return max(0.01, min(0.05, kelly_fraction))
            
        except Exception:
            return 0.02
    
    def _calculate_var_adjustment(self, symbol: str) -> float:
        """Calculate VaR-based adjustment"""
        try:
            if len(self.pnl_history) < 30:
                return 1.0
            
            # Get recent PnL data
            recent_pnl = [p['pnl'] for p in list(self.pnl_history)[-30:]]
            returns = np.array(recent_pnl) / self.account_balance
            
            # Calculate 95% VaR
            var_95 = np.percentile(returns, 5)  # 5th percentile for losses
            
            # Adjust position size based on VaR
            target_var = -0.02  # Target 2% daily VaR
            if var_95 < target_var:
                return abs(target_var / var_95)
            
            return 1.0
            
        except Exception:
            return 1.0
    
    def _check_position_limits(self, symbol: str, lots: float) -> bool:
        """Check if position size exceeds limits"""
        try:
            current_exposure = self.symbol_exposure.get(symbol, 0)
            symbol_info = self.mt5_connector.get_symbol_info(symbol) if self.mt5_connector else None
            
            if symbol_info:
                new_exposure = lots * symbol_info.tick_value * 100000
                total_symbol_exposure = current_exposure + new_exposure
                max_allowed = self.account_balance * self.config.max_symbol_exposure
                
                return total_symbol_exposure > max_allowed
            
            return False
            
        except Exception:
            return True  # Conservative approach
    
    def _calculate_confidence_level(self, symbol: str, method: PositionSizeMethod) -> float:
        """Calculate confidence level for position sizing"""
        base_confidence = 80.0
        
        # Adjust based on method reliability
        method_adjustments = {
            PositionSizeMethod.FIXED_LOT: -10,
            PositionSizeMethod.FIXED_PERCENT: 0,
            PositionSizeMethod.VOLATILITY_ADJUSTED: 10,
            PositionSizeMethod.ATR_BASED: 15,
            PositionSizeMethod.KELLY_CRITERION: 5,
            PositionSizeMethod.VAR_BASED: 20
        }
        
        confidence = base_confidence + method_adjustments.get(method, 0)
        
        # Adjust based on data availability
        if not self.mt5_connector or not self.mt5_connector.is_connected():
            confidence -= 20
        
        return max(0, min(100, confidence))
    
    def _calculate_correlation_risk(self, symbol: str, trade_type: str) -> float:
        """Calculate correlation risk with existing positions"""
        try:
            if not self.open_positions:
                return 0.0
            
            # Simplified correlation calculation
            # In production, use proper correlation matrix
            correlation_risk = 0.0
            
            for pos in self.open_positions.values():
                if pos['symbol'] == symbol:
                    # Same symbol correlation
                    if (trade_type == "BUY" and pos['type'] == 0) or \
                       (trade_type == "SELL" and pos['type'] == 1):
                        correlation_risk += 0.3  # Same direction
                    else:
                        correlation_risk -= 0.2  # Opposite direction (hedging)
                
                # Cross-pair correlation (simplified)
                elif self._are_correlated_pairs(symbol, pos['symbol']):
                    correlation_risk += 0.2
            
            return max(0, min(1, correlation_risk))
            
        except Exception:
            return 0.5  # Moderate risk assumption
    
    def _calculate_market_risk(self, symbol: str) -> float:
        """Calculate current market risk level"""
        try:
            if not self.mt5_connector:
                return 0.5
            
            # Get recent volatility
            rates = self.mt5_connector.get_rates(symbol, 1, 20)
            if rates is None or len(rates) < 10:
                return 0.5
            
            # Calculate volatility
            returns = rates['close'].pct_change().dropna()
            current_vol = returns.std()
            
            # Historical average volatility (simplified)
            historical_vol = 0.01  # Baseline
            
            # Risk ratio
            risk_ratio = current_vol / historical_vol
            
            return max(0, min(1, risk_ratio - 1))
            
        except Exception:
            return 0.5
    
    def _check_margin_requirements(self, symbol: str, lots: float, price: float) -> bool:
        """Check if sufficient margin is available"""
        try:
            if not self.mt5_connector:
                return False
            
            account_info = self.mt5_connector.get_account_info()
            symbol_info = self.mt5_connector.get_symbol_info(symbol)
            
            if not account_info or not symbol_info:
                return False
            
            # Calculate required margin
            required_margin = lots * symbol_info.margin_required
            available_margin = account_info['margin_free']
            
            return required_margin > available_margin * 0.9  # 90% margin utilization limit
            
        except Exception:
            return True  # Conservative approach
    
    def _calculate_win_rate(self) -> float:
        """Calculate win rate from trade history"""
        if len(self.trade_history) == 0:
            return 0.0
        
        wins = sum(1 for trade in self.trade_history if trade.get('profit', 0) > 0)
        return wins / len(self.trade_history) * 100
    
    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor"""
        if len(self.trade_history) == 0:
            return 0.0
        
        total_profit = sum(trade.get('profit', 0) for trade in self.trade_history if trade.get('profit', 0) > 0)
        total_loss = abs(sum(trade.get('profit', 0) for trade in self.trade_history if trade.get('profit', 0) < 0))
        
        return total_profit / total_loss if total_loss > 0 else 0.0
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio"""
        if len(self.returns_history) < 10:
            return 0.0
        
        returns = np.array(list(self.returns_history))
        excess_returns = returns - 0.02/252  # Assuming 2% risk-free rate
        
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0.0
    
    def _calculate_sortino_ratio(self) -> float:
        """Calculate Sortino ratio"""
        if len(self.returns_history) < 10:
            return 0.0
        
        returns = np.array(list(self.returns_history))
        excess_returns = returns - 0.02/252
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return 0.0
        
        downside_std = np.std(downside_returns)
        return np.mean(excess_returns) / downside_std * np.sqrt(252) if downside_std > 0 else 0.0
    
    def _calculate_var_95(self) -> float:
        """Calculate 95% Value at Risk"""
        if len(self.pnl_history) < 30:
            return 0.0
        
        pnl_values = [p['pnl'] for p in list(self.pnl_history)[-30:]]
        return abs(np.percentile(pnl_values, 5))  # 5th percentile
    
    def _calculate_expected_shortfall(self) -> float:
        """Calculate Expected Shortfall (CVaR)"""
        if len(self.pnl_history) < 30:
            return 0.0
        
        pnl_values = [p['pnl'] for p in list(self.pnl_history)[-30:]]
        var_95 = np.percentile(pnl_values, 5)
        
        # Expected value of losses beyond VaR
        shortfall_values = [p for p in pnl_values if p <= var_95]
        return abs(np.mean(shortfall_values)) if shortfall_values else 0.0
    
    def _calculate_portfolio_correlation_risk(self) -> float:
        """Calculate overall portfolio correlation risk"""
        if len(self.open_positions) < 2:
            return 0.0
        
        # Simplified correlation risk calculation
        symbols = list(set(pos['symbol'] for pos in self.open_positions.values()))
        correlation_count = 0
        total_pairs = 0
        
        for i, symbol1 in enumerate(symbols):
            for symbol2 in symbols[i+1:]:
                total_pairs += 1
                if self._are_correlated_pairs(symbol1, symbol2):
                    correlation_count += 1
        
        return correlation_count / total_pairs if total_pairs > 0 else 0.0
    
    def _calculate_margin_utilization(self) -> float:
        """Calculate margin utilization percentage"""
        try:
            if not self.mt5_connector:
                return 0.0
            
            account_info = self.mt5_connector.get_account_info()
            if not account_info:
                return 0.0
            
            margin_used = account_info.get('margin', 0)
            margin_free = account_info.get('margin_free', 0)
            total_margin = margin_used + margin_free
            
            return margin_used / total_margin if total_margin > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _are_correlated_pairs(self, symbol1: str, symbol2: str) -> bool:
        """Check if two currency pairs are correlated"""
        # Simplified correlation mapping
        correlation_groups = [
            ["EURUSD", "GBPUSD", "AUDUSD"],  # Risk-on currencies
            ["USDJPY", "USDCHF"],           # USD strength pairs
            ["EURJPY", "GBPJPY", "AUDJPY"], # JPY crosses
        ]
        
        for group in correlation_groups:
            if symbol1 in group and symbol2 in group:
                return True
        
        return False
    
    def _log_risk_event(self, event: RiskEvent, details: str) -> None:
        """Log risk event"""
        event_data = {
            'timestamp': datetime.now(),
            'event': event.value,
            'details': details
        }
        
        self.risk_events.append(event_data)
        self.logger.warning(f"🚨 Risk Event: {event.value} - {details}")
    
    def reset_daily_stats(self) -> None:
        """Reset daily statistics (call at start of each trading day)"""
        with self.lock:
            self.daily_start_balance = self.account_balance
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.daily_wins = 0
            self.daily_losses = 0
            
            # Add to returns history
            if len(self.pnl_history) > 0:
                yesterday_pnl = list(self.pnl_history)[-1]['pnl']
                daily_return = yesterday_pnl / self.account_balance
                self.returns_history.append(daily_return)
        
        self.logger.info("📊 Daily statistics reset")

# Factory function
def create_risk_manager(config: Optional[RiskConfig] = None, 
                       mt5_connector: Optional[MT5Connector] = None) -> RiskManager:
    """Create risk manager with optional configuration"""
    return RiskManager(config, mt5_connector)

# Export main classes
__all__ = ['RiskManager', 'RiskMetrics', 'PositionSizeResult', 'TradeRiskAssessment', 
           'RiskLevel', 'PositionSizeMethod', 'RiskEvent', 'create_risk_manager']

if __name__ == "__main__":
    # Demo and testing
    print("🛡️ Risk Management System Testing")
    print("=" * 50)
    
    # Create risk manager
    risk_manager = create_risk_manager()
    
    # Test position sizing
    position_result = risk_manager.calculate_position_size(
        symbol="EURUSD",
        entry_price=1.1000,
        stop_loss=1.0950,
        method=PositionSizeMethod.VOLATILITY_ADJUSTED
    )
    
    print(f"📊 Position Size Calculation:")
    print(f"   Recommended Lots: {position_result.recommended_lots:.2f}")
    print(f"   Risk Amount: ${position_result.risk_amount:.2f}")
    print(f"   Stop Loss Pips: {position_result.stop_loss_pips:.1f}")
    print(f"   Confidence: {position_result.confidence_level:.1f}%")
    
    # Test trade assessment
    trade_assessment = risk_manager.assess_trade_risk(
        symbol="EURUSD",
        trade_type="BUY",
        entry_price=1.1000,
        stop_loss=1.0950,
        position_size=0.1
    )
    
    print(f"\n🎯 Trade Risk Assessment:")
    print(f"   Risk Score: {trade_assessment.risk_score:.1f}/100")
    print(f"   Confidence: {trade_assessment.confidence_score:.1f}%")
    print(f"   Approved: {'✅' if trade_assessment.approved else '❌'}")
    
    # Get risk metrics
    metrics = risk_manager.get_risk_metrics()
    print(f"\n📈 Risk Metrics:")
    print(f"   Current Drawdown: {metrics.current_drawdown:.2%}")
    print(f"   Win Rate: {metrics.win_rate:.1f}%")
    print(f"   Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    
    print("\n🎯 Risk Management System Ready!")