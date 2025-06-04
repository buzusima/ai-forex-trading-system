"""
Institutional-grade Forex AI Trading System
Portfolio Optimizer Module

This module provides advanced portfolio optimization using Modern Portfolio Theory,
risk parity, Kelly Criterion, and multi-objective optimization techniques.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm, t
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import asyncio
import threading
from collections import defaultdict
import json
import math
from abc import ABC, abstractmethod

class OptimizationType(Enum):
    """Portfolio optimization types"""
    MEAN_VARIANCE = "MEAN_VARIANCE"
    RISK_PARITY = "RISK_PARITY"
    MAXIMUM_SHARPE = "MAXIMUM_SHARPE"
    MINIMUM_VARIANCE = "MINIMUM_VARIANCE"
    MAXIMUM_DIVERSIFICATION = "MAXIMUM_DIVERSIFICATION"
    KELLY_CRITERION = "KELLY_CRITERION"
    BLACK_LITTERMAN = "BLACK_LITTERMAN"
    ROBUST_OPTIMIZATION = "ROBUST_OPTIMIZATION"

class RiskMeasure(Enum):
    """Risk measurement types"""
    VARIANCE = "VARIANCE"
    VOLATILITY = "VOLATILITY"
    VAR_95 = "VAR_95"
    VAR_99 = "VAR_99"
    CVAR_95 = "CVAR_95"
    CVAR_99 = "CVAR_99"
    MAX_DRAWDOWN = "MAX_DRAWDOWN"
    SEMI_VARIANCE = "SEMI_VARIANCE"

@dataclass
class OptimizationConstraints:
    """Portfolio optimization constraints"""
    min_weight: float = 0.0
    max_weight: float = 1.0
    max_position_count: int = 50
    min_position_size: float = 0.01
    max_portfolio_risk: float = 0.2
    target_return: Optional[float] = None
    max_correlation: float = 0.8
    max_sector_exposure: float = 0.3
    transaction_costs: float = 0.0001
    
    # Currency-specific constraints
    max_currency_exposure: Dict[str, float] = field(default_factory=dict)
    forbidden_pairs: List[str] = field(default_factory=list)
    required_pairs: List[str] = field(default_factory=list)

@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics"""
    expected_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    var_95: float
    cvar_95: float
    beta: float
    alpha: float
    information_ratio: float
    calmar_ratio: float
    omega_ratio: float
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

@dataclass
class OptimizationResult:
    """Portfolio optimization result"""
    weights: Dict[str, float]
    metrics: PortfolioMetrics
    optimization_type: OptimizationType
    optimization_time: float
    convergence_info: Dict[str, Any]
    timestamp: datetime
    
    def get_non_zero_weights(self) -> Dict[str, float]:
        """Get weights that are not zero"""
        return {symbol: weight for symbol, weight in self.weights.items() if abs(weight) > 1e-6}

class RiskModel:
    """Risk model for portfolio optimization"""
    
    def __init__(self, lookback_period: int = 252, half_life: int = 63):
        self.lookback_period = lookback_period
        self.half_life = half_life
        self.logger = logging.getLogger(__name__)
    
    def calculate_covariance_matrix(self, returns: pd.DataFrame, 
                                  method: str = "exponential") -> np.ndarray:
        """Calculate covariance matrix"""
        try:
            if method == "exponential":
                return self._exponential_covariance(returns)
            elif method == "shrinkage":
                return self._shrinkage_covariance(returns)
            elif method == "sample":
                return returns.cov().values
            else:
                raise ValueError(f"Unknown covariance method: {method}")
        except Exception as e:
            self.logger.error(f"Error calculating covariance matrix: {str(e)}")
            # Fallback to identity matrix
            n = len(returns.columns)
            return np.eye(n) * 0.01
    
    def _exponential_covariance(self, returns: pd.DataFrame) -> np.ndarray:
        """Calculate exponentially weighted covariance matrix"""
        decay_factor = 1 - 1/self.half_life
        weights = np.array([(decay_factor ** i) for i in range(len(returns))][::-1])
        weights = weights / weights.sum()
        
        # Center the returns
        mean_returns = returns.mean()
        centered_returns = returns - mean_returns
        
        # Calculate weighted covariance
        weighted_returns = centered_returns.values * weights.reshape(-1, 1)
        cov_matrix = np.cov(weighted_returns.T)
        
        return cov_matrix
    
    def _shrinkage_covariance(self, returns: pd.DataFrame) -> np.ndarray:
        """Calculate shrinkage covariance matrix (Ledoit-Wolf)"""
        sample_cov = returns.cov().values
        n_assets = sample_cov.shape[0]
        
        # Target matrix (identity scaled by average variance)
        avg_var = np.trace(sample_cov) / n_assets
        target = np.eye(n_assets) * avg_var
        
        # Shrinkage intensity (simplified version)
        shrinkage = 0.2  # Could be optimized
        
        shrunk_cov = shrinkage * target + (1 - shrinkage) * sample_cov
        return shrunk_cov
    
    def calculate_expected_returns(self, returns: pd.DataFrame, 
                                 method: str = "historical") -> np.ndarray:
        """Calculate expected returns"""
        if method == "historical":
            return returns.mean().values * 252  # Annualize
        elif method == "exponential":
            decay_factor = 1 - 1/self.half_life
            weights = np.array([(decay_factor ** i) for i in range(len(returns))][::-1])
            weights = weights / weights.sum()
            weighted_returns = (returns.values * weights.reshape(-1, 1)).sum(axis=0)
            return weighted_returns * 252
        else:
            raise ValueError(f"Unknown expected returns method: {method}")

class PortfolioOptimizer(ABC):
    """Abstract base class for portfolio optimizers"""
    
    def __init__(self, risk_model: RiskModel, constraints: OptimizationConstraints):
        self.risk_model = risk_model
        self.constraints = constraints
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def optimize(self, returns: pd.DataFrame, **kwargs) -> OptimizationResult:
        """Optimize portfolio"""
        pass
    
    def _validate_inputs(self, returns: pd.DataFrame) -> bool:
        """Validate optimization inputs"""
        if returns.empty:
            self.logger.error("Returns data is empty")
            return False
        
        if returns.isnull().any().any():
            self.logger.warning("Returns data contains NaN values")
            returns.fillna(0, inplace=True)
        
        if len(returns.columns) < 2:
            self.logger.error("Need at least 2 assets for optimization")
            return False
        
        return True
    
    def _apply_constraints(self, weights: np.ndarray, symbols: List[str]) -> np.ndarray:
        """Apply portfolio constraints"""
        # Normalize weights to sum to 1
        weights = weights / weights.sum()
        
        # Apply min/max weight constraints
        weights = np.clip(weights, self.constraints.min_weight, self.constraints.max_weight)
        
        # Apply position count constraint
        if np.sum(weights > self.constraints.min_position_size) > self.constraints.max_position_count:
            # Keep only the largest positions
            sorted_indices = np.argsort(weights)[::-1]
            keep_indices = sorted_indices[:self.constraints.max_position_count]
            new_weights = np.zeros_like(weights)
            new_weights[keep_indices] = weights[keep_indices]
            weights = new_weights
        
        # Renormalize
        if weights.sum() > 0:
            weights = weights / weights.sum()
        
        return weights
    
    def _calculate_portfolio_metrics(self, weights: np.ndarray, returns: pd.DataFrame, 
                                   benchmark_returns: Optional[pd.Series] = None) -> PortfolioMetrics:
        """Calculate portfolio performance metrics"""
        portfolio_returns = (returns * weights).sum(axis=1)
        
        # Basic metrics
        expected_return = portfolio_returns.mean() * 252
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = expected_return / volatility if volatility > 0 else 0
        
        # Downside metrics
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else volatility
        sortino_ratio = expected_return / downside_std if downside_std > 0 else 0
        
        # Drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # VaR and CVaR
        var_95 = np.percentile(portfolio_returns, 5) * np.sqrt(252)
        cvar_95 = portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)].mean() * np.sqrt(252)
        
        # Market-relative metrics
        beta = alpha = information_ratio = 0.0
        if benchmark_returns is not None:
            covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
            benchmark_variance = np.var(benchmark_returns)
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
            
            benchmark_return = benchmark_returns.mean() * 252
            alpha = expected_return - beta * benchmark_return
            
            excess_returns = portfolio_returns - benchmark_returns
            tracking_error = excess_returns.std() * np.sqrt(252)
            information_ratio = excess_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0
        
        # Additional ratios
        calmar_ratio = expected_return / abs(max_drawdown) if max_drawdown < 0 else 0
        
        # Omega ratio (simplified)
        threshold = 0
        gains = portfolio_returns[portfolio_returns > threshold].sum()
        losses = abs(portfolio_returns[portfolio_returns <= threshold].sum())
        omega_ratio = gains / losses if losses > 0 else float('inf')
        
        return PortfolioMetrics(
            expected_return=expected_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            var_95=var_95,
            cvar_95=cvar_95,
            beta=beta,
            alpha=alpha,
            information_ratio=information_ratio,
            calmar_ratio=calmar_ratio,
            omega_ratio=omega_ratio
        )

class MeanVarianceOptimizer(PortfolioOptimizer):
    """Mean-Variance optimization (Markowitz)"""
    
    def optimize(self, returns: pd.DataFrame, risk_aversion: float = 1.0, 
                **kwargs) -> OptimizationResult:
        """Optimize using mean-variance approach"""
        start_time = datetime.now()
        
        if not self._validate_inputs(returns):
            raise ValueError("Invalid input data")
        
        symbols = returns.columns.tolist()
        n_assets = len(symbols)
        
        # Calculate expected returns and covariance matrix
        expected_returns = self.risk_model.calculate_expected_returns(returns)
        cov_matrix = self.risk_model.calculate_covariance_matrix(returns)
        
        # Objective function: maximize utility = return - (risk_aversion/2) * variance
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            return -(portfolio_return - (risk_aversion / 2) * portfolio_variance)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # weights sum to 1
        ]
        
        if self.constraints.target_return is not None:
            constraints.append({
                'type': 'eq', 
                'fun': lambda x: np.dot(x, expected_returns) - self.constraints.target_return
            })
        
        # Bounds
        bounds = [(self.constraints.min_weight, self.constraints.max_weight) for _ in range(n_assets)]
        
        # Initial guess (equal weights)
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(
            objective, x0, method='SLSQP',
            bounds=bounds, constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if not result.success:
            self.logger.warning(f"Optimization did not converge: {result.message}")
        
        # Apply additional constraints
        optimal_weights = self._apply_constraints(result.x, symbols)
        
        # Calculate metrics
        metrics = self._calculate_portfolio_metrics(optimal_weights, returns)
        
        optimization_time = (datetime.now() - start_time).total_seconds()
        
        return OptimizationResult(
            weights=dict(zip(symbols, optimal_weights)),
            metrics=metrics,
            optimization_type=OptimizationType.MEAN_VARIANCE,
            optimization_time=optimization_time,
            convergence_info={
                'success': result.success,
                'message': result.message,
                'iterations': result.nit,
                'function_evaluations': result.nfev
            },
            timestamp=datetime.now(timezone.utc)
        )

class RiskParityOptimizer(PortfolioOptimizer):
    """Risk Parity optimization"""
    
    def optimize(self, returns: pd.DataFrame, **kwargs) -> OptimizationResult:
        """Optimize using risk parity approach"""
        start_time = datetime.now()
        
        if not self._validate_inputs(returns):
            raise ValueError("Invalid input data")
        
        symbols = returns.columns.tolist()
        n_assets = len(symbols)
        
        # Calculate covariance matrix
        cov_matrix = self.risk_model.calculate_covariance_matrix(returns)
        
        # Risk parity objective: minimize sum of squared risk contribution differences
        def objective(weights):
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            marginal_risk = np.dot(cov_matrix, weights)
            risk_contributions = weights * marginal_risk / portfolio_variance
            
            # Target is equal risk contribution (1/n for each asset)
            target_risk = 1.0 / n_assets
            risk_diff = risk_contributions - target_risk
            
            return np.sum(risk_diff ** 2)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # weights sum to 1
        ]
        
        # Bounds
        bounds = [(self.constraints.min_weight, self.constraints.max_weight) for _ in range(n_assets)]
        
        # Initial guess (equal weights)
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(
            objective, x0, method='SLSQP',
            bounds=bounds, constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if not result.success:
            self.logger.warning(f"Risk parity optimization did not converge: {result.message}")
        
        # Apply additional constraints
        optimal_weights = self._apply_constraints(result.x, symbols)
        
        # Calculate metrics
        metrics = self._calculate_portfolio_metrics(optimal_weights, returns)
        
        optimization_time = (datetime.now() - start_time).total_seconds()
        
        return OptimizationResult(
            weights=dict(zip(symbols, optimal_weights)),
            metrics=metrics,
            optimization_type=OptimizationType.RISK_PARITY,
            optimization_time=optimization_time,
            convergence_info={
                'success': result.success,
                'message': result.message,
                'iterations': result.nit,
                'function_evaluations': result.nfev
            },
            timestamp=datetime.now(timezone.utc)
        )

class MaximumSharpeOptimizer(PortfolioOptimizer):
    """Maximum Sharpe ratio optimization"""
    
    def optimize(self, returns: pd.DataFrame, risk_free_rate: float = 0.02, 
                **kwargs) -> OptimizationResult:
        """Optimize for maximum Sharpe ratio"""
        start_time = datetime.now()
        
        if not self._validate_inputs(returns):
            raise ValueError("Invalid input data")
        
        symbols = returns.columns.tolist()
        n_assets = len(symbols)
        
        # Calculate expected returns and covariance matrix
        expected_returns = self.risk_model.calculate_expected_returns(returns)
        cov_matrix = self.risk_model.calculate_covariance_matrix(returns)
        
        # Objective function: minimize negative Sharpe ratio
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            portfolio_std = np.sqrt(portfolio_variance)
            
            if portfolio_std == 0:
                return -np.inf
            
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std
            return -sharpe_ratio
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # weights sum to 1
        ]
        
        # Bounds
        bounds = [(self.constraints.min_weight, self.constraints.max_weight) for _ in range(n_assets)]
        
        # Initial guess (equal weights)
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(
            objective, x0, method='SLSQP',
            bounds=bounds, constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if not result.success:
            self.logger.warning(f"Maximum Sharpe optimization did not converge: {result.message}")
        
        # Apply additional constraints
        optimal_weights = self._apply_constraints(result.x, symbols)
        
        # Calculate metrics
        metrics = self._calculate_portfolio_metrics(optimal_weights, returns)
        
        optimization_time = (datetime.now() - start_time).total_seconds()
        
        return OptimizationResult(
            weights=dict(zip(symbols, optimal_weights)),
            metrics=metrics,
            optimization_type=OptimizationType.MAXIMUM_SHARPE,
            optimization_time=optimization_time,
            convergence_info={
                'success': result.success,
                'message': result.message,
                'iterations': result.nit,
                'function_evaluations': result.nfev
            },
            timestamp=datetime.now(timezone.utc)
        )

class KellyCriterionOptimizer(PortfolioOptimizer):
    """Kelly Criterion optimization"""
    
    def optimize(self, returns: pd.DataFrame, confidence_level: float = 0.95,
                **kwargs) -> OptimizationResult:
        """Optimize using Kelly Criterion"""
        start_time = datetime.now()
        
        if not self._validate_inputs(returns):
            raise ValueError("Invalid input data")
        
        symbols = returns.columns.tolist()
        n_assets = len(symbols)
        
        # Calculate expected returns and covariance matrix
        expected_returns = self.risk_model.calculate_expected_returns(returns)
        cov_matrix = self.risk_model.calculate_covariance_matrix(returns)
        
        # Kelly fraction calculation: f = μ / σ² for single asset
        # For portfolio: f = Σ⁻¹μ (simplified)
        try:
            inv_cov = np.linalg.pinv(cov_matrix)
            kelly_weights = np.dot(inv_cov, expected_returns)
            
            # Normalize and apply constraints
            if kelly_weights.sum() != 0:
                kelly_weights = kelly_weights / kelly_weights.sum()
            else:
                kelly_weights = np.ones(n_assets) / n_assets
            
            # Apply leverage constraint (Kelly can suggest >100% allocation)
            max_leverage = 1.0 / confidence_level  # Conservative approach
            if np.sum(np.abs(kelly_weights)) > max_leverage:
                kelly_weights = kelly_weights * max_leverage / np.sum(np.abs(kelly_weights))
            
            # Apply additional constraints
            optimal_weights = self._apply_constraints(kelly_weights, symbols)
            
        except np.linalg.LinAlgError:
            self.logger.error("Covariance matrix is singular, using equal weights")
            optimal_weights = np.ones(n_assets) / n_assets
        
        # Calculate metrics
        metrics = self._calculate_portfolio_metrics(optimal_weights, returns)
        
        optimization_time = (datetime.now() - start_time).total_seconds()
        
        return OptimizationResult(
            weights=dict(zip(symbols, optimal_weights)),
            metrics=metrics,
            optimization_type=OptimizationType.KELLY_CRITERION,
            optimization_time=optimization_time,
            convergence_info={
                'success': True,
                'message': 'Kelly criterion applied successfully',
                'iterations': 1,
                'function_evaluations': 1
            },
            timestamp=datetime.now(timezone.utc)
        )

class PortfolioRebalancer:
    """Portfolio rebalancing manager"""
    
    def __init__(self, transaction_cost: float = 0.0001, min_rebalance_threshold: float = 0.05):
        self.transaction_cost = transaction_cost
        self.min_rebalance_threshold = min_rebalance_threshold
        self.logger = logging.getLogger(__name__)
    
    def calculate_rebalancing_trades(self, current_weights: Dict[str, float],
                                   target_weights: Dict[str, float],
                                   portfolio_value: float) -> Dict[str, float]:
        """Calculate trades needed for rebalancing"""
        trades = {}
        total_cost = 0.0
        
        # Get all symbols
        all_symbols = set(current_weights.keys()) | set(target_weights.keys())
        
        for symbol in all_symbols:
            current_weight = current_weights.get(symbol, 0.0)
            target_weight = target_weights.get(symbol, 0.0)
            
            weight_diff = target_weight - current_weight
            
            # Only trade if difference is significant
            if abs(weight_diff) > self.min_rebalance_threshold:
                trade_value = weight_diff * portfolio_value
                trade_cost = abs(trade_value) * self.transaction_cost
                
                trades[symbol] = trade_value - np.sign(trade_value) * trade_cost
                total_cost += trade_cost
        
        return trades, total_cost
    
    def should_rebalance(self, current_weights: Dict[str, float],
                        target_weights: Dict[str, float]) -> bool:
        """Determine if portfolio should be rebalanced"""
        all_symbols = set(current_weights.keys()) | set(target_weights.keys())
        
        total_deviation = 0.0
        for symbol in all_symbols:
            current_weight = current_weights.get(symbol, 0.0)
            target_weight = target_weights.get(symbol, 0.0)
            total_deviation += abs(target_weight - current_weight)
        
        return total_deviation > self.min_rebalance_threshold

class AdvancedPortfolioOptimizer:
    """Advanced portfolio optimizer with multiple strategies"""
    
    def __init__(self, risk_model: RiskModel = None, constraints: OptimizationConstraints = None):
        self.risk_model = risk_model or RiskModel()
        self.constraints = constraints or OptimizationConstraints()
        self.rebalancer = PortfolioRebalancer()
        self.logger = logging.getLogger(__name__)
        
        # Initialize optimizers
        self.optimizers = {
            OptimizationType.MEAN_VARIANCE: MeanVarianceOptimizer(self.risk_model, self.constraints),
            OptimizationType.RISK_PARITY: RiskParityOptimizer(self.risk_model, self.constraints),
            OptimizationType.MAXIMUM_SHARPE: MaximumSharpeOptimizer(self.risk_model, self.constraints),
            OptimizationType.KELLY_CRITERION: KellyCriterionOptimizer(self.risk_model, self.constraints)
        }
        
        # Performance tracking
        self.optimization_history = []
        self.performance_history = []
    
    def optimize_portfolio(self, returns: pd.DataFrame, 
                          optimization_type: OptimizationType,
                          **kwargs) -> OptimizationResult:
        """Optimize portfolio using specified method"""
        try:
            optimizer = self.optimizers[optimization_type]
            result = optimizer.optimize(returns, **kwargs)
            
            # Store result
            self.optimization_history.append(result)
            
            # Log result
            non_zero_weights = result.get_non_zero_weights()
            self.logger.info(f"Portfolio optimization completed: "
                           f"Type: {optimization_type.value}, "
                           f"Sharpe: {result.metrics.sharpe_ratio:.3f}, "
                           f"Active positions: {len(non_zero_weights)}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Portfolio optimization failed: {str(e)}")
            raise
    
    def multi_objective_optimization(self, returns: pd.DataFrame,
                                   objectives: List[OptimizationType],
                                   weights: Optional[List[float]] = None) -> OptimizationResult:
        """Combine multiple optimization objectives"""
        if weights is None:
            weights = [1.0 / len(objectives)] * len(objectives)
        
        if len(weights) != len(objectives):
            raise ValueError("Number of weights must match number of objectives")
        
        # Optimize for each objective
        results = []
        for obj_type in objectives:
            result = self.optimize_portfolio(returns, obj_type)
            results.append(result)
        
        # Combine weights using specified weights
        symbols = returns.columns.tolist()
        combined_weights = np.zeros(len(symbols))
        
        for i, result in enumerate(results):
            result_weights = np.array([result.weights.get(symbol, 0.0) for symbol in symbols])
            combined_weights += weights[i] * result_weights
        
        # Normalize
        combined_weights = combined_weights / combined_weights.sum()
        
        # Calculate combined metrics
        combined_metrics = self.optimizers[OptimizationType.MEAN_VARIANCE]._calculate_portfolio_metrics(
            combined_weights, returns
        )
        
        return OptimizationResult(
            weights=dict(zip(symbols, combined_weights)),
            metrics=combined_metrics,
            optimization_type=OptimizationType.MEAN_VARIANCE,  # Combined type
            optimization_time=sum(r.optimization_time for r in results),
            convergence_info={'combined_objectives': [obj.value for obj in objectives]},
            timestamp=datetime.now(timezone.utc)
        )
    
    def robust_optimization(self, returns: pd.DataFrame, 
                          uncertainty_sets: Dict[str, float],
                          optimization_type: OptimizationType = OptimizationType.MEAN_VARIANCE) -> OptimizationResult:
        """Robust optimization considering parameter uncertainty"""
        n_scenarios = 100
        results = []
        
        # Generate scenarios based on uncertainty
        for _ in range(n_scenarios):
            # Perturb returns based on uncertainty sets
            perturbed_returns = returns.copy()
            
            for symbol, uncertainty in uncertainty_sets.items():
                if symbol in perturbed_returns.columns:
                    noise = np.random.normal(0, uncertainty, len(perturbed_returns))
                    perturbed_returns[symbol] += noise
            
            # Optimize for this scenario
            try:
                result = self.optimize_portfolio(perturbed_returns, optimization_type)
                results.append(result)
            except:
                continue
        
        if not results:
            raise ValueError("Robust optimization failed for all scenarios")
        
        # Average the weights across scenarios
        symbols = returns.columns.tolist()
        avg_weights = np.zeros(len(symbols))
        
        for result in results:
            result_weights = np.array([result.weights.get(symbol, 0.0) for symbol in symbols])
            avg_weights += result_weights
        
        avg_weights = avg_weights / len(results)
        avg_weights = avg_weights / avg_weights.sum()  # Normalize
        
        # Calculate metrics for average portfolio
        avg_metrics = self.optimizers[optimization_type]._calculate_portfolio_metrics(
            avg_weights, returns
        )
        
        return OptimizationResult(
            weights=dict(zip(symbols, avg_weights)),
            metrics=avg_metrics,
            optimization_type=OptimizationType.ROBUST_OPTIMIZATION,
            optimization_time=sum(r.optimization_time for r in results),
            convergence_info={
                'scenarios': len(results),
                'base_optimization': optimization_type.value
            },
            timestamp=datetime.now(timezone.utc)
        )
    
    def dynamic_rebalancing(self, current_weights: Dict[str, float],
                           returns: pd.DataFrame,
                           portfolio_value: float,
                           optimization_type: OptimizationType = OptimizationType.MEAN_VARIANCE) -> Dict[str, Any]:
        """Dynamic portfolio rebalancing"""
        # Get new optimal weights
        new_result = self.optimize_portfolio(returns, optimization_type)
        target_weights = new_result.weights
        
        # Check if rebalancing is needed
        should_rebalance = self.rebalancer.should_rebalance(current_weights, target_weights)
        
        if should_rebalance:
            trades, transaction_cost = self.rebalancer.calculate_rebalancing_trades(
                current_weights, target_weights, portfolio_value
            )
            
            return {
                'should_rebalance': True,
                'target_weights': target_weights,
                'trades': trades,
                'transaction_cost': transaction_cost,
                'expected_improvement': {
                    'sharpe_ratio': new_result.metrics.sharpe_ratio,
                    'expected_return': new_result.metrics.expected_return,
                    'volatility': new_result.metrics.volatility
                }
            }
        else:
            return {
                'should_rebalance': False,
                'current_weights': current_weights,
                'target_weights': target_weights,
                'deviation': sum(abs(target_weights.get(s, 0) - current_weights.get(s, 0)) 
                               for s in set(current_weights.keys()) | set(target_weights.keys()))
            }
    
    def backtest_strategy(self, returns: pd.DataFrame, 
                         optimization_type: OptimizationType,
                         rebalance_frequency: int = 21,  # Days
                         lookback_window: int = 252) -> Dict[str, Any]:
        """Backtest optimization strategy"""
        if len(returns) < lookback_window + rebalance_frequency:
            raise ValueError("Insufficient data for backtesting")
        
        portfolio_values = [1.0]  # Start with $1
        weights_history = []
        rebalance_dates = []
        
        for i in range(lookback_window, len(returns), rebalance_frequency):
            # Get training data
            train_data = returns.iloc[i-lookback_window:i]
            
            # Optimize portfolio
            try:
                result = self.optimize_portfolio(train_data, optimization_type)
                weights = result.weights
            except:
                # Use equal weights if optimization fails
                weights = {symbol: 1.0/len(returns.columns) for symbol in returns.columns}
            
            weights_history.append(weights)
            rebalance_dates.append(returns.index[i])
            
            # Calculate returns for holding period
            end_idx = min(i + rebalance_frequency, len(returns))
            period_returns = returns.iloc[i:end_idx]
            
            # Calculate portfolio return
            for _, day_returns in period_returns.iterrows():
                portfolio_return = sum(weights.get(symbol, 0.0) * day_returns[symbol] 
                                     for symbol in returns.columns)
                portfolio_values.append(portfolio_values[-1] * (1 + portfolio_return))
        
        # Calculate performance metrics
        portfolio_returns = pd.Series(portfolio_values).pct_change().dropna()
        
        total_return = portfolio_values[-1] - 1
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = portfolio_returns.mean() * 252 / volatility if volatility > 0 else 0
        
        max_value = max(portfolio_values)
        max_drawdown = (max_value - min(portfolio_values[portfolio_values.index(max_value):])) / max_value
        
        return {
            'total_return': total_return,
            'annualized_volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'portfolio_values': portfolio_values,
            'weights_history': weights_history,
            'rebalance_dates': rebalance_dates,
            'optimization_type': optimization_type.value
        }
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization history"""
        if not self.optimization_history:
            return {'message': 'No optimization history available'}
        
        recent_results = self.optimization_history[-10:]  # Last 10 optimizations
        
        return {
            'total_optimizations': len(self.optimization_history),
            'recent_sharpe_ratios': [r.metrics.sharpe_ratio for r in recent_results],
            'recent_optimization_times': [r.optimization_time for r in recent_results],
            'optimization_types_used': list(set(r.optimization_type.value for r in recent_results)),
            'average_sharpe_ratio': np.mean([r.metrics.sharpe_ratio for r in recent_results]),
            'average_optimization_time': np.mean([r.optimization_time for r in recent_results])
        }

# Example usage and testing
if __name__ == "__main__":
    # Generate sample returns data
    np.random.seed(42)
    symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD']
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    
    # Generate correlated returns
    n_assets = len(symbols)
    returns_data = np.random.multivariate_normal(
        mean=[0.0001] * n_assets,
        cov=np.eye(n_assets) * 0.0001 + 0.00002,
        size=len(dates)
    )
    
    returns_df = pd.DataFrame(returns_data, index=dates, columns=symbols)
    
    # Create optimizer
    constraints = OptimizationConstraints(
        min_weight=0.0,
        max_weight=0.4,
        max_position_count=3,
        min_position_size=0.05
    )
    
    optimizer = AdvancedPortfolioOptimizer(constraints=constraints)
    
    # Test different optimization methods
    print("Testing Portfolio Optimization Methods:")
    
    for opt_type in [OptimizationType.MEAN_VARIANCE, OptimizationType.RISK_PARITY, 
                     OptimizationType.MAXIMUM_SHARPE, OptimizationType.KELLY_CRITERION]:
        
        result = optimizer.optimize_portfolio(returns_df, opt_type)
        
        print(f"\n{opt_type.value}:")
        print(f"  Sharpe Ratio: {result.metrics.sharpe_ratio:.3f}")
        print(f"  Expected Return: {result.metrics.expected_return:.3f}")
        print(f"  Volatility: {result.metrics.volatility:.3f}")
        print(f"  Active Positions: {len(result.get_non_zero_weights())}")
        print(f"  Optimization Time: {result.optimization_time:.3f}s")
    
    # Test multi-objective optimization
    print(f"\nMulti-Objective Optimization:")
    multi_result = optimizer.multi_objective_optimization(
        returns_df, 
        [OptimizationType.MAXIMUM_SHARPE, OptimizationType.RISK_PARITY],
        weights=[0.7, 0.3]
    )
    print(f"  Sharpe Ratio: {multi_result.metrics.sharpe_ratio:.3f}")
    print(f"  Expected Return: {multi_result.metrics.expected_return:.3f}")
    print(f"  Volatility: {multi_result.metrics.volatility:.3f}")
    
    # Test backtesting
    print(f"\nBacktest Results:")
    backtest_result = optimizer.backtest_strategy(
        returns_df, OptimizationType.MAXIMUM_SHARPE, rebalance_frequency=21
    )
    print(f"  Total Return: {backtest_result['total_return']:.3f}")
    print(f"  Sharpe Ratio: {backtest_result['sharpe_ratio']:.3f}")
    print(f"  Max Drawdown: {backtest_result['max_drawdown']:.3f}")
    
    print("Portfolio optimization test completed successfully!")