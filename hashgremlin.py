#!/usr/bin/env python3
"""
HashGremlin - Infinite Bitcoin Trading AI Trainer v2.9
Created by: lolitemaultes
Description: Continuously trains an AI model on historical Bitcoin data using reinforcement learning
v2.9 Fixes: True KL divergence, risk-adjusted reward guardrails, stricter position cap release,
            unified LR policy, IQR-based best model filtering for maximum robustness
"""

import os
import sys
import json
import time
import random
import pickle
import hashlib
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, NamedTuple, Any
from dataclasses import dataclass, asdict
import warnings
import yaml
import subprocess
from collections import deque
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical, kl_divergence

# Add safe globals for NumPy scalars (PyTorch 2.6+ compatibility)
try:
    from torch.serialization import add_safe_globals
    import numpy as _np
    try:
        add_safe_globals([_np.core.multiarray.scalar])
    except Exception:
        try:
            add_safe_globals([_np._core.multiarray.scalar])
        except Exception:
            pass
except Exception:
    pass

import gymnasium as gym
from gymnasium import spaces

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.align import Align
from rich.text import Text
from rich import box

console = Console()

# Configuration
@dataclass
class Config:
    data_dir: Path = Path("./hashgremlin_data")
    model_dir: Path = Path("./hashgremlin_models")
    log_dir: Path = Path("./hashgremlin_logs")
    
    # Data settings
    start_date: str = "2019-01-01"
    data_file: str = "btc_historical_data.pkl"
    download_threads: int = 16
    
    # Training settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_envs: int = 8  # Parallel environments
    episode_hours: int = 24
    steps_per_update: int = 2048  # Collect this many steps before update
    batch_size: int = 256
    minibatch_size: int = 64
    update_epochs: int = 10
    actor_lr: float = 1e-4  # Separate actor learning rate
    critic_lr: float = 2e-4  # Separate critic learning rate
    min_actor_lr: float = 5e-5  # Minimum actor learning rate
    max_actor_lr: float = 2e-4  # Maximum actor learning rate
    min_critic_lr: float = 1e-4  # Minimum critic learning rate
    max_critic_lr: float = 4e-4  # Maximum critic learning rate
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.1  # Tighter clipping
    clip_range_value: float = 0.3  # Relaxed from 0.2 for better critic learning
    value_loss_coef: float = 1.0  # Increased from 0.5 for stronger critic signal
    max_grad_norm: float = 0.5
    target_kl: float = 0.10  # Increased from 0.08 to reduce early stops
    adaptive_lr: bool = True  # Enable adaptive learning rate
    
    # Model settings
    hidden_size: int = 256
    num_layers: int = 3
    
    # Save settings
    save_interval: int = 50  # Save every N updates
    eval_interval: int = 10  # Evaluate every N updates
    eval_episodes: int = 50  # Increased from 20 for more stable evaluation
    keep_best_n: int = 5  # Keep best N models by validation score
    keep_recent_n: int = 3  # Keep recent N models
    best_model_min_sharpe: float = 0.3  # Minimum Sharpe to be considered "best"
    best_model_max_std: float = 5.0  # Maximum Sharpe std to avoid lucky spikes
    best_model_max_iqr: float = 4.0  # Maximum Sharpe IQR to avoid jumpy windows
    best_model_min_win_rate: float = 0.55  # Minimum win rate for best model
    
    # Trading settings
    initial_balance: float = 10000.0
    commission_rate: float = 0.0015  # Increased from 0.001 to reduce churn
    slippage_bps: float = 2  # 2 basis points slippage
    min_portfolio_value: float = 100.0  # End episode if below this
    position_change_penalty: float = 0.0005  # Penalty for position changes
    min_position_change: float = 0.3  # Increased from 0.2 to further reduce churn
    trade_cooldown_steps: int = 5  # Increased from 3 to reduce overtrading
    position_deadband: float = 0.12  # 12% exposure dead-zone for flat positions
    time_penalty_base: float = -0.000005  # Base time penalty (scaled by exposure)
    reward_risk_adjusted: bool = True  # Enable risk-adjusted rewards
    reward_vol_window: int = 20  # Window for rolling volatility calculation
    temp_position_cap: float = 0.5  # Temporary cap until model proves itself
    position_cap_sharpe_threshold: float = 0.2  # Remove cap when median Sharpe >= this
    
    # Action space: position sizes (capped temporarily)
    position_sizes: List[float] = None
    
    def __post_init__(self):
        self.data_dir.mkdir(exist_ok=True)
        self.model_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        
        # Position sizes: conservative to full position (will be capped dynamically)
        if self.position_sizes is None:
            self.position_sizes = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
        
        # Validate batch size configuration
        if self.steps_per_update % self.num_envs != 0:
            console.print(f"[yellow]Warning: steps_per_update ({self.steps_per_update}) must be divisible by num_envs ({self.num_envs})[/yellow]")
            self.steps_per_update = (self.steps_per_update // self.num_envs) * self.num_envs
            console.print(f"[yellow]Adjusted to: {self.steps_per_update}[/yellow]")
        
        if self.steps_per_update % self.minibatch_size != 0:
            console.print(f"[yellow]Warning: minibatch_size ({self.minibatch_size}) should divide evenly into steps_per_update ({self.steps_per_update})[/yellow]")
    
    def save(self, path: Path):
        """Save configuration to YAML"""
        config_dict = asdict(self)
        # Convert Path objects to strings for YAML
        for key, value in config_dict.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
        with open(path, 'w') as f:
            yaml.dump(config_dict, f)
    
    @classmethod
    def load(cls, path: Path):
        """Load configuration from YAML"""
        with open(path, 'r') as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
        # Convert string paths back to Path objects
        for key in ['data_dir', 'model_dir', 'log_dir']:
            if key in data and isinstance(data[key], str):
                data[key] = Path(data[key])
        return cls(**data)
    
    def get_entropy_coef(self, update_num: int) -> float:
        """Get current entropy coefficient with faster cosine decay to lower floor"""
        # Faster decay to lower floor for more exploitation
        start = 0.012  # Start a bit lower
        end = 0.0005  # Lower floor for more exploitation
        T = 600  # Faster decay over 600 updates (was 800)
        
        progress = min(update_num / T, 1.0)
        # Cosine decay for smooth transition
        return end + 0.5 * (start - end) * (1 + np.cos(np.pi * progress))
    
    def get_position_cap(self, median_sharpe: float) -> float:
        """Get current position cap based on performance - stricter release threshold"""
        if median_sharpe >= self.position_cap_sharpe_threshold:
            return 1.0  # Remove cap
        else:
            return self.temp_position_cap  # Keep cap

config = Config()

# Helper function to convert NumPy types to Python types
def to_python_types(obj: Any) -> Any:
    """Recursively convert NumPy types to Python types for safe serialization"""
    if isinstance(obj, dict):
        return {k: to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python_types(v) for v in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    else:
        return obj

# Trajectory storage
class Trajectory(NamedTuple):
    states: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    old_action_logits: torch.Tensor  # Store full logits for true KL computation
    values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor

# Bitcoin Trading Environment
class BitcoinTradingEnv(gym.Env):
    """Custom Environment for Bitcoin Trading with improved execution"""
    
    def __init__(self, data: pd.DataFrame, episode_hours: int = 24, eval_mode: bool = False, position_cap: float = 1.0):
        super().__init__()
        
        self.data = data
        self.episode_hours = episode_hours
        self.episode_steps = episode_hours * 2  # 30-minute intervals
        self.eval_mode = eval_mode
        self.position_cap = position_cap  # Dynamic position cap
        
        # Action space: different position sizes
        self.action_space = spaces.Discrete(len(config.position_sizes))
        
        # Observation space: enhanced features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(30,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        
        # Random starting point (keep some data for validation)
        if self.eval_mode:
            # Use last 20% of data for evaluation
            eval_start = int(len(self.data) * 0.8)
            max_start = len(self.data) - self.episode_steps - 1
            self.start_idx = np.random.randint(eval_start, max_start)
        else:
            # Use first 80% for training
            max_start = min(int(len(self.data) * 0.8), len(self.data) - self.episode_steps - 1)
            self.start_idx = np.random.randint(0, max_start)
        
        self.current_step = 0
        
        # Trading state
        self.balance = config.initial_balance
        self.btc_held = 0
        self.total_value = self.balance
        self.prev_total_value = self.balance
        self.entry_price = 0
        self.position_size = 0  # Current position as fraction of portfolio
        self.prev_position_size = 0  # For tracking position changes
        self.trades = []
        self.trade_history = []
        self.executed_trades = 0  # Count only actual executed trades
        self.portfolio_vals = []  # Track portfolio values for metrics
        self.step_returns = []  # Track step returns for volatility
        self.position_changes = []  # Track position changes
        self.hold_times = []  # Track holding periods
        self.current_hold_time = 0
        self.max_adverse_excursion = 0  # Track worst drawdown during position
        self.cooldown = 0  # Trade cooldown counter
        self.exposures = []  # Track actual exposure over time
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        idx = self.start_idx + self.current_step
        window_size = 48  # Look back 24 hours
        
        if idx < window_size:
            window_data = self.data.iloc[:idx+1]
        else:
            window_data = self.data.iloc[idx-window_size+1:idx+1]
        
        prices = window_data['close'].values
        volumes = window_data['volume'].values if 'volume' in window_data.columns else np.ones(len(window_data))
        highs = window_data['high'].values if 'high' in window_data.columns else prices
        lows = window_data['low'].values if 'low' in window_data.columns else prices
        
        # Price features
        returns = np.diff(prices) / (prices[:-1] + 1e-8) if len(prices) > 1 else np.array([0])
        current_price = prices[-1]
        current_high = highs[-1] if len(highs) > 0 else current_price
        current_low = lows[-1] if len(lows) > 0 else current_price
        
        # Rolling statistics
        if len(prices) >= 20:
            sma_20 = np.mean(prices[-20:])
            std_20 = np.std(prices[-20:])
            sma_10 = np.mean(prices[-10:])
            std_10 = np.std(prices[-10:])
        else:
            sma_20 = sma_10 = np.mean(prices)
            std_20 = std_10 = np.std(prices) if len(prices) > 1 else 1e-8
        
        # Technical indicators
        rsi = self._calculate_rsi(prices)
        macd, signal = self._calculate_macd(prices)
        bb_upper, bb_lower = self._calculate_bollinger_bands(prices)
        
        # Volume features
        volume_mean = np.mean(volumes) if len(volumes) > 0 else 1
        volume_zscore = (volumes[-1] - volume_mean) / (np.std(volumes) + 1e-8) if len(volumes) > 1 else 0
        
        # Volatility features
        realized_vol = np.std(returns[-10:]) if len(returns) >= 10 else np.std(returns) if len(returns) > 1 else 0
        
        # Time features
        current_timestamp = self.data['timestamp'].iloc[idx]
        time_of_day = current_timestamp.hour + current_timestamp.minute / 60
        time_sin = np.sin(2 * np.pi * time_of_day / 24)
        time_cos = np.cos(2 * np.pi * time_of_day / 24)
        
        # Position features
        unrealized_pnl = ((current_price - self.entry_price) / self.entry_price if self.btc_held > 0 else 0)
        
        # Normalize price features
        price_z = (current_price - sma_20) / (std_20 + 1e-8)
        price_z_10 = (current_price - sma_10) / (std_10 + 1e-8)
        
        # Spread feature
        spread = (current_high - current_low) / (current_price + 1e-8)
        
        # Normalized trade counter
        trades_normalized = min(self.executed_trades / 20.0, 2.0)  # Use executed trades
        
        features = np.array([
            price_z,
            price_z_10,
            returns[-1] if len(returns) > 0 else 0,
            np.mean(returns[-5:]) if len(returns) >= 5 else np.mean(returns) if len(returns) > 0 else 0,
            realized_vol,
            rsi,
            macd,
            signal,
            (current_price - bb_upper) / (std_20 + 1e-8),
            (current_price - bb_lower) / (std_20 + 1e-8),
            volume_zscore,
            self.position_size,
            unrealized_pnl,
            self.balance / config.initial_balance,
            self.total_value / config.initial_balance,
            trades_normalized,
            self.current_step / self.episode_steps,
            time_sin,
            time_cos,
            spread,
        ] + list(returns[-10:] if len(returns) >= 10 else np.pad(returns, (10-len(returns), 0), 'constant')))
        
        # Guard feature dimension
        assert features.shape[0] == 30, f"Observation dimension {features.shape[0]} != 30"
        
        return features.astype(np.float32)
    
    def _calculate_rsi(self, prices, period=14):
        if len(prices) < 2:
            return 0.5
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        if len(gains) >= period:
            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])
        else:
            avg_gain = np.mean(gains) if len(gains) > 0 else 0
            avg_loss = np.mean(losses) if len(losses) > 0 else 0
        
        if avg_loss == 0:
            return 1.0
        
        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 1 - (1 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal_period=9):
        if len(prices) < slow:
            return 0, 0
        
        prices_series = pd.Series(prices)
        exp_fast = prices_series.ewm(span=fast, adjust=False).mean()
        exp_slow = prices_series.ewm(span=slow, adjust=False).mean()
        macd = exp_fast - exp_slow
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        
        macd_val = macd.iloc[-1] / prices[-1] if prices[-1] != 0 else 0
        signal_val = signal.iloc[-1] / prices[-1] if prices[-1] != 0 else 0
        
        return macd_val, signal_val
    
    def _calculate_bollinger_bands(self, prices, period=20, num_std=2):
        if len(prices) < period:
            current = prices[-1] if len(prices) > 0 else 0
            return current, current
        
        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        
        return upper, lower
    
    def step(self, action):
        # Current bar for observation
        current_idx = self.start_idx + self.current_step
        current_price = self.data['close'].iloc[current_idx]
        
        # Store previous value for reward calculation
        self.prev_total_value = self.total_value
        
        # Advance to next bar for execution
        self.current_step += 1
        
        if self.current_step >= self.episode_steps:
            done = True
            next_price = current_price
        else:
            done = False
            next_idx = self.start_idx + self.current_step
            # Execute at next bar's OPEN for realistic fills
            if 'open' in self.data.columns:
                next_price = self.data['open'].iloc[next_idx]
            else:
                next_price = self.data['close'].iloc[next_idx]
        
        # Get target position from action with cap
        target_position = min(config.position_sizes[action], self.position_cap)
        
        # Apply deadband for small positions
        if abs(target_position) < config.position_deadband:
            target_position = 0.0
        
        # Apply cooldown and threshold blocking
        if self.cooldown > 0:
            target_position = self.position_size  # Keep current position
            self.cooldown -= 1
        elif abs(target_position - self.position_size) < config.min_position_change:
            target_position = self.position_size  # Ignore small changes
        
        # Execute trade at next bar's open price with slippage
        slippage_mult = 1 + (config.slippage_bps / 10000)
        
        # Calculate target BTC based on TOTAL portfolio value
        portfolio_value = self.balance + self.btc_held * next_price
        target_btc = target_position * portfolio_value / (next_price * slippage_mult) if target_position > 0 else 0
        delta_btc = target_btc - self.btc_held
        
        # Track if we actually executed a trade
        trade_executed = False
        
        if delta_btc > 1e-12:  # Buy
            btc_to_buy = delta_btc
            cost = btc_to_buy * next_price * slippage_mult * (1 + config.commission_rate)
            
            if cost <= self.balance and btc_to_buy > 0:
                prev_btc = self.btc_held
                self.btc_held += btc_to_buy
                self.balance -= cost
                trade_executed = True
                
                # Reset MAE on fresh entry (0 → long)
                if prev_btc == 0:
                    self.max_adverse_excursion = 0
                
                # Weighted-average entry price
                if prev_btc > 0:
                    self.entry_price = ((self.entry_price * prev_btc) + (next_price * slippage_mult * btc_to_buy)) / (self.btc_held + 1e-8)
                else:
                    self.entry_price = next_price * slippage_mult
                
                self.trades.append(('buy', next_price, btc_to_buy, self.current_step))
                self.trade_history.append({
                    'step': self.current_step,
                    'action': 'buy',
                    'price': next_price,
                    'amount': btc_to_buy,
                    'cost': cost
                })
        
        elif delta_btc < -1e-12:  # Sell
            btc_to_sell = min(-delta_btc, self.btc_held)
            proceeds = btc_to_sell * next_price / slippage_mult * (1 - config.commission_rate)
            
            if btc_to_sell > 0:
                self.btc_held -= btc_to_sell
                self.balance += proceeds
                trade_executed = True
                
                self.trades.append(('sell', next_price, proceeds, self.current_step))
                self.trade_history.append({
                    'step': self.current_step,
                    'action': 'sell',
                    'price': next_price,
                    'amount': btc_to_sell,
                    'proceeds': proceeds
                })
        
        # Apply cooldown after successful trade
        if trade_executed:
            self.cooldown = max(self.cooldown, config.trade_cooldown_steps)
            self.executed_trades += 1  # Count only actual executed trades
        
        # Update total value (mark-to-market)
        if not done:
            current_close = self.data['close'].iloc[next_idx]
        else:
            current_close = next_price
        
        self.total_value = self.balance + (self.btc_held * current_close if self.btc_held > 0 else 0)
        
        # Recompute position size AFTER trade execution
        portfolio_after = self.balance + self.btc_held * current_close
        self.position_size = (self.btc_held * current_close) / (portfolio_after + 1e-8) if portfolio_after > 0 else 0
        
        # Track actual exposure (absolute position size)
        self.exposures.append(abs(self.position_size))
        
        # Track portfolio value for metrics
        self.portfolio_vals.append(self.total_value)
        
        # Calculate portfolio return
        portfolio_return = (self.total_value - self.prev_total_value) / (self.prev_total_value + 1e-8)
        self.step_returns.append(portfolio_return)
        
        # Calculate risk-adjusted reward if enabled
        if config.reward_risk_adjusted and len(self.step_returns) >= config.reward_vol_window:
            # Calculate rolling volatility with guardrail
            recent_returns = self.step_returns[-config.reward_vol_window:]
            rolling_vol = max(np.std(recent_returns), 1e-4)  # Floor at 1e-4 to prevent explosion
            # Risk-adjusted return (Sharpe-like) with clipping
            risk_adjusted_return = np.clip(portfolio_return / rolling_vol, -10.0, 10.0)
            base_reward = risk_adjusted_return
        else:
            base_reward = portfolio_return
        
        # Track position changes and holding times
        if self.position_size != self.prev_position_size:
            self.position_changes.append(abs(self.position_size - self.prev_position_size))
            if self.current_hold_time > 0:
                self.hold_times.append(self.current_hold_time)
            self.current_hold_time = 0
        else:
            self.current_hold_time += 1
        
        # Track max adverse excursion if in position
        if self.btc_held > 0 and self.entry_price > 0:
            current_pnl = (current_close - self.entry_price) / self.entry_price
            self.max_adverse_excursion = min(self.max_adverse_excursion, current_pnl)
        
        # Exposure-scaled time penalty (no bleed when flat)
        time_penalty = config.time_penalty_base * abs(self.position_size)
        
        # Transaction cost awareness
        trade_penalty = -0.0001 if len(self.trades) > 0 and self.trades[-1][-1] == self.current_step else 0
        
        # Position change penalty only for meaningful changes
        position_change_penalty = 0
        if config.position_change_penalty > 0 and hasattr(self, 'prev_position_size'):
            position_change = abs(self.position_size - self.prev_position_size)
            if position_change > config.min_position_change:  # Only penalize changes > threshold
                position_change_penalty = -config.position_change_penalty * position_change
        self.prev_position_size = self.position_size
        
        reward = base_reward + time_penalty + trade_penalty + position_change_penalty
        
        # Check for bankruptcy
        if self.total_value < config.min_portfolio_value:
            done = True
            reward -= 1.0  # Heavy penalty for bankruptcy
        
        # Calculate metrics for info
        total_return = (self.total_value - config.initial_balance) / config.initial_balance
        
        info = {
            'total_value': self.total_value,
            'balance': self.balance,
            'btc_held': self.btc_held,
            'position_size': self.position_size,
            'trades': self.executed_trades,  # Use actual executed trades count
            'return': total_return,
            'current_price': current_close,
            'step_reward': portfolio_return,
            'risk_adjusted_reward': base_reward,
            'cooldown': self.cooldown
        }
        
        if done:
            # Calculate final metrics
            info['sharpe'] = self._calculate_sharpe()
            info['max_drawdown'] = self._calculate_max_drawdown()
            info['win_rate'] = self._calculate_win_rate()
            info['avg_hold_time'] = np.mean(self.hold_times) if self.hold_times else 0
            info['avg_position_change'] = np.mean(self.position_changes) if self.position_changes else 0
            info['max_adverse_excursion'] = self.max_adverse_excursion
            info['avg_exposure'] = np.mean(self.exposures) if self.exposures else 0
        
        return self._get_observation(), reward, done, False, info
    
    def _calculate_sharpe(self):
        """Calculate Sharpe ratio from step returns"""
        if len(self.portfolio_vals) < 3:
            return 0.0
        
        pv = np.array(self.portfolio_vals, dtype=float)
        step_returns = np.diff(pv) / (pv[:-1] + 1e-8)
        
        if len(step_returns) == 0:
            return 0.0
        
        mean_return = np.mean(step_returns)
        std_return = np.std(step_returns) + 1e-12
        
        # Annualized Sharpe (30-min bars, ~17,520 per year)
        steps_per_year = 365 * 24 * 2
        sharpe = (mean_return / std_return) * np.sqrt(steps_per_year)
        
        return sharpe
    
    def _calculate_max_drawdown(self):
        """Calculate maximum drawdown from portfolio values"""
        if len(self.portfolio_vals) == 0:
            return 0.0
        
        peak = -float('inf')
        max_dd = 0.0
        
        for value in self.portfolio_vals:
            peak = max(peak, value)
            dd = (peak - value) / (peak + 1e-8)
            max_dd = max(max_dd, dd)
        
        return max_dd
    
    def _calculate_win_rate(self):
        """Calculate win rate of completed round trips"""
        if len(self.trade_history) < 2:
            return 0.0
        
        wins = 0
        total_round_trips = 0
        
        # Find buy-sell pairs
        buy_cost = None
        for trade in self.trade_history:
            if trade['action'] == 'buy':
                buy_cost = trade.get('cost', trade['price'] * trade['amount'])
            elif trade['action'] == 'sell' and buy_cost is not None:
                sell_proceeds = trade.get('proceeds', trade['price'] * trade['amount'])
                total_round_trips += 1
                if sell_proceeds > buy_cost:
                    wins += 1
                buy_cost = None  # Reset for next round trip
        
        return float(wins) / float(total_round_trips) if total_round_trips > 0 else 0.0

# Neural Network Model
class TradingNetwork(nn.Module):
    def __init__(self, input_size=30, hidden_size=256, num_layers=3, action_size=6):
        super().__init__()
        
        self.shared_layers = nn.ModuleList()
        
        # Input layer with layer norm
        self.input_norm = nn.LayerNorm(input_size)
        self.shared_layers.append(nn.Linear(input_size, hidden_size))
        
        # Hidden layers with residual connections
        for _ in range(num_layers - 1):
            self.shared_layers.append(nn.Linear(hidden_size, hidden_size))
        
        # Output heads
        self.actor_head = nn.Linear(hidden_size, action_size)
        self.critic_head = nn.Linear(hidden_size, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # Guard input dimension
        assert x.shape[-1] == 30, f"Input dimension {x.shape[-1]} != 30"
        
        x = self.input_norm(x)
        
        for i, layer in enumerate(self.shared_layers):
            residual = x if i > 0 else None
            x = F.relu(layer(x))
            
            # Residual connection for deeper layers
            if residual is not None and x.shape == residual.shape:
                x = x + residual * 0.5
            
            # Dropout for regularization
            if i > 0 and self.training:
                x = F.dropout(x, p=0.1)
        
        action_logits = self.actor_head(x)
        value = self.critic_head(x)
        
        return action_logits, value

# PPO Agent with throttled adaptive LR
class PPOAgent:
    def __init__(self, model: TradingNetwork):
        self.model = model
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        # Separate optimizers for actor and critic
        actor_params = list(model.actor_head.parameters())
        critic_params = list(model.critic_head.parameters())
        shared_params = []
        for layer in model.shared_layers:
            shared_params.extend(layer.parameters())
        shared_params.extend(model.input_norm.parameters())
        
        # Actor gets shared + actor head
        self.actor_optimizer = optim.Adam(shared_params + actor_params, lr=config.actor_lr, eps=1e-5)
        # Critic gets ONLY critic head
        self.critic_optimizer = optim.Adam(critic_params, lr=config.critic_lr, eps=1e-5)
        
        # Track current learning rates
        self.current_actor_lr = config.actor_lr
        self.current_critic_lr = config.critic_lr
        
        # Track validation performance for LR adjustment
        self.last_val_return = None
        
        # AMP setup for faster training on modern GPUs
        self.use_amp = config.device == "cuda" and torch.cuda.is_available()
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Update counter for entropy decay
        self.update_count = 0
    
    def adjust_learning_rate(self, kl_divergence: float, val_return: float = None):
        """Throttled adaptive learning rate adjustment"""
        if not config.adaptive_lr:
            return
        
        # Store validation return if provided
        if val_return is not None:
            self.last_val_return = val_return
        
        # Only allow increases if KL is low AND performance is positive
        if kl_divergence < 0.5 * config.target_kl and self.last_val_return is not None and self.last_val_return >= 0:
            # Can increase learning rate
            factor = 1.1
        elif kl_divergence > 1.5 * config.target_kl:
            # KL too high, reduce learning rate
            factor = 0.9
        else:
            # Default: decay slowly
            factor = 0.995
        
        # Adjust actor learning rate
        new_actor_lr = self.current_actor_lr * factor
        new_actor_lr = np.clip(new_actor_lr, config.min_actor_lr, config.max_actor_lr)
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = new_actor_lr
        self.current_actor_lr = new_actor_lr
        
        # Adjust critic learning rate
        new_critic_lr = self.current_critic_lr * factor
        new_critic_lr = np.clip(new_critic_lr, config.min_critic_lr, config.max_critic_lr)
        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] = new_critic_lr
        self.current_critic_lr = new_critic_lr
    
    def get_action_and_value(self, state, deterministic=False):
        """Get action, value, log_prob, and logits for a state"""
        state = torch.FloatTensor(state).to(self.device)
        
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        
        with torch.no_grad():
            action_logits, value = self.model(state)
            
            if deterministic:
                # Greedy action for evaluation
                action = torch.argmax(action_logits, dim=-1)
                dist = Categorical(logits=action_logits)
                log_prob = dist.log_prob(action)
            else:
                # Sample from distribution for training
                dist = Categorical(logits=action_logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
        
        return action.cpu().numpy(), value.cpu().numpy(), log_prob.cpu().numpy(), action_logits.cpu().numpy()
    
    def compute_gae(self, rewards, values, dones, num_envs):
        """Compute Generalized Advantage Estimation"""
        T = len(rewards) // num_envs
        N = num_envs
        
        # Reshape to [T, N] for per-env computation
        rewards_tn = rewards.reshape(T, N)
        values_tn = values.reshape(T, N)
        dones_tn = dones.reshape(T, N)
        
        advantages = np.zeros_like(rewards_tn)
        last_adv = np.zeros(N)
        
        for t in reversed(range(T)):
            if t == T - 1:
                next_val = np.zeros(N)
                next_done = np.ones(N)
            else:
                next_val = values_tn[t + 1]
                next_done = dones_tn[t + 1]
            
            delta = rewards_tn[t] + config.gamma * next_val * (1 - next_done) - values_tn[t]
            last_adv = delta + config.gamma * config.gae_lambda * (1 - next_done) * last_adv
            advantages[t] = last_adv
        
        returns = advantages + values_tn
        
        # Flatten back to original shape
        advantages = advantages.reshape(T * N)
        returns = returns.reshape(T * N)
        
        return advantages, returns
    
    def update(self, trajectories: Trajectory):
        """Update model using PPO with multiple epochs and minibatches"""
        self.update_count += 1
        
        # Get current entropy coefficient using new schedule
        entropy_coef = config.get_entropy_coef(self.update_count)
        
        # Move to device
        states = trajectories.states.to(self.device)
        actions = trajectories.actions.to(self.device)
        old_log_probs = trajectories.log_probs.to(self.device)
        old_action_logits = trajectories.old_action_logits.to(self.device)  # For true KL
        advantages = trajectories.advantages.to(self.device)
        returns = trajectories.returns.to(self.device)
        
        # Normalize advantages and detach to prevent gradients
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = advantages.detach()  # Critical: make sure it's a constant for PPO
        
        # Precompute parameter groups for gradient clipping
        actor_params = list(self.model.actor_head.parameters())
        shared_params = []
        for layer in self.model.shared_layers:
            shared_params.extend(layer.parameters())
        shared_params.extend(self.model.input_norm.parameters())
        critic_params = list(self.model.critic_head.parameters())
        
        # Training metrics
        total_loss_epoch = []
        policy_loss_epoch = []
        value_loss_epoch = []
        entropy_epoch = []
        kl_divergence_epoch = []
        
        # Multiple epochs
        for epoch in range(config.update_epochs):
            # Generate random indices for minibatches
            indices = np.random.permutation(len(states))
            
            for start in range(0, len(states), config.minibatch_size):
                end = start + config.minibatch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_old_action_logits = old_action_logits[batch_indices]  # For true KL
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Zero gradients for both optimizers first
                self.actor_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)
                
                # Use AMP for faster training
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        # Single forward pass for both losses
                        action_logits, values = self.model(batch_states)
                        dist = Categorical(logits=action_logits)
                        log_probs = dist.log_prob(batch_actions)
                        entropy = dist.entropy().mean()
                        
                        # Calculate ratio
                        ratio = torch.exp(log_probs - batch_old_log_probs)
                        
                        # Clipped surrogate loss
                        surr1 = ratio * batch_advantages
                        surr2 = torch.clamp(ratio, 1 - config.clip_range, 1 + config.clip_range) * batch_advantages
                        policy_loss = -torch.min(surr1, surr2).mean()
                        
                        # Value loss with clipping
                        values = values.squeeze()
                        value_pred_clipped = batch_returns + torch.clamp(
                            values - batch_returns, -config.clip_range_value, config.clip_range_value
                        )
                        value_losses = (values - batch_returns) ** 2
                        value_losses_clipped = (value_pred_clipped - batch_returns) ** 2
                        value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                        
                        # Compute both losses
                        actor_loss = policy_loss - entropy_coef * entropy
                        critic_loss = config.value_loss_coef * value_loss
                    
                    # Backward both losses BEFORE any optimizer steps
                    self.scaler.scale(actor_loss).backward(retain_graph=True)
                    self.scaler.scale(critic_loss).backward()
                    
                    # Unscale and clip gradients for both optimizers
                    self.scaler.unscale_(self.actor_optimizer)
                    nn.utils.clip_grad_norm_(shared_params + actor_params, config.max_grad_norm)
                    
                    self.scaler.unscale_(self.critic_optimizer)
                    nn.utils.clip_grad_norm_(critic_params, config.max_grad_norm)
                    
                    # Step both optimizers
                    self.scaler.step(self.actor_optimizer)
                    self.scaler.step(self.critic_optimizer)
                    
                    # Update scaler once
                    self.scaler.update()
                else:
                    # Standard training without AMP
                    action_logits, values = self.model(batch_states)
                    dist = Categorical(logits=action_logits)
                    log_probs = dist.log_prob(batch_actions)
                    entropy = dist.entropy().mean()
                    
                    ratio = torch.exp(log_probs - batch_old_log_probs)
                    
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1 - config.clip_range, 1 + config.clip_range) * batch_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    values = values.squeeze()
                    value_pred_clipped = batch_returns + torch.clamp(
                        values - batch_returns, -config.clip_range_value, config.clip_range_value
                    )
                    value_losses = (values - batch_returns) ** 2
                    value_losses_clipped = (value_pred_clipped - batch_returns) ** 2
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                    
                    # Compute both losses
                    actor_loss = policy_loss - entropy_coef * entropy
                    critic_loss = config.value_loss_coef * value_loss
                    
                    # Backward both losses BEFORE any optimizer steps
                    actor_loss.backward(retain_graph=True)
                    critic_loss.backward()
                    
                    # Clip gradients
                    nn.utils.clip_grad_norm_(shared_params + actor_params, config.max_grad_norm)
                    nn.utils.clip_grad_norm_(critic_params, config.max_grad_norm)
                    
                    # Step both optimizers
                    self.actor_optimizer.step()
                    self.critic_optimizer.step()
                
                # Track metrics
                with torch.no_grad():
                    # Compute true KL divergence between old and new distributions  
                    old_dist = Categorical(logits=batch_old_action_logits)
                    new_dist = Categorical(logits=action_logits)
                    kl = kl_divergence(old_dist, new_dist).mean()
                    kl_divergence_epoch.append(kl.item())
                
                total_loss = policy_loss + config.value_loss_coef * value_loss - entropy_coef * entropy
                total_loss_epoch.append(total_loss.item())
                policy_loss_epoch.append(policy_loss.item())
                value_loss_epoch.append(value_loss.item())
                entropy_epoch.append(entropy.item())
            
            # Early stopping if KL divergence too high
            mean_kl = np.mean(kl_divergence_epoch)
            if mean_kl > config.target_kl:
                console.print(f"[yellow]Early stopping at epoch {epoch+1}/{config.update_epochs} due to KL divergence: {mean_kl:.4f} > {config.target_kl:.4f}[/yellow]")
                break
        
        # Throttled adaptive learning rate adjustment
        mean_kl = np.mean(kl_divergence_epoch)
        self.adjust_learning_rate(mean_kl)
        
        return {
            'total_loss': np.mean(total_loss_epoch),
            'policy_loss': np.mean(policy_loss_epoch),
            'value_loss': np.mean(value_loss_epoch),
            'entropy': np.mean(entropy_epoch),
            'entropy_coef': entropy_coef,
            'kl_divergence': mean_kl,
            'update_epochs': epoch + 1,
            'actor_lr': self.current_actor_lr,
            'critic_lr': self.current_critic_lr
        }

# Data Manager
class DataManager:
    def __init__(self):
        self.data_path = config.data_dir / config.data_file
        self.data = None
    
    def download_data(self):
        """Download historical Bitcoin data with proper pagination"""
        console.print("\n[bold cyan]📊 HashGremlin Data Downloader[/bold cyan]\n")
        
        if self.data_path.exists():
            console.print("[yellow]⚠️  Existing data file found. Checking integrity...[/yellow]")
            if self._verify_data():
                console.print("[green]✅ Data file is valid. Loading...[/green]")
                self.load_data()
                console.print(f"[green]📈 Loaded {len(self.data)} data points[/green]")
                return
            else:
                console.print("[red]❌ Data file is corrupted. Re-downloading...[/red]")
        
        # Download from Binance with proper pagination
        console.print("[cyan]Downloading from Binance...[/cyan]")
        self.data = self._download_binance_data_paginated()
        
        if self.data is not None and len(self.data) > 0:
            self._save_data()
            console.print(f"[green]✅ Successfully downloaded {len(self.data)} data points[/green]")
        else:
            # Fallback to synthetic data
            console.print("[yellow]⚠️  Download failed. Generating synthetic data for testing...[/yellow]")
            self.data = self._generate_synthetic_data()
            self._save_data()
    
    def _download_binance_data_paginated(self):
        """Download data from Binance with proper pagination"""
        url = "https://api.binance.com/api/v3/klines"
        all_data = []
        
        start_date = datetime.strptime(config.start_date, "%Y-%m-%d")
        end_date = datetime.now()
        
        start_ms = int(start_date.timestamp() * 1000)
        end_ms = int(end_date.timestamp() * 1000)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            
            # Estimate total requests
            total_days = (end_date - start_date).days
            estimated_requests = (total_days * 48) // 1000 + 1
            
            task = progress.add_task("[cyan]Downloading Bitcoin history...", total=estimated_requests)
            
            current_start = start_ms
            request_count = 0
            
            while current_start < end_ms:
                try:
                    params = {
                        'symbol': 'BTCUSDT',
                        'interval': '30m',
                        'startTime': current_start,
                        'endTime': min(current_start + 30 * 24 * 60 * 60 * 1000, end_ms),  # 30 days max
                        'limit': 1000
                    }
                    
                    response = requests.get(url, params=params, timeout=10)
                    response.raise_for_status()
                    
                    candles = response.json()
                    
                    if not candles:
                        break
                    
                    all_data.extend(candles)
                    
                    # Move to next batch
                    last_close_time = candles[-1][6]
                    current_start = last_close_time + 1
                    
                    request_count += 1
                    progress.update(task, advance=1)
                    
                    # Rate limiting
                    time.sleep(0.1)
                    
                except requests.exceptions.RequestException as e:
                    console.print(f"[red]Request error: {e}[/red]")
                    time.sleep(1)
                    continue
        
        if all_data:
            df = pd.DataFrame(all_data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Convert and clean
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df = df.astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})
            
            # Remove duplicates and sort
            df = df.drop_duplicates(subset=['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Verify continuity
            expected_diff = pd.Timedelta(minutes=30)
            time_diffs = df['timestamp'].diff()
            gaps = time_diffs[time_diffs > expected_diff * 1.5]
            
            if len(gaps) > 0:
                console.print(f"[yellow]⚠️  Found {len(gaps)} gaps in data[/yellow]")
                # Forward-fill gaps
                df = df.set_index('timestamp').resample('30min').ffill().reset_index()
                console.print(f"[green]✅ Gaps filled using forward-fill[/green]")
            
            return df
        
        return None
    
    def _generate_synthetic_data(self):
        """Generate synthetic data for testing"""
        dates = pd.date_range(start=config.start_date, end=datetime.now(), freq='30min')
        
        # Generate realistic price movement
        np.random.seed(42)
        prices = [20000]
        volatility = 0.02
        
        for i in range(len(dates) - 1):
            # Volatility clustering
            volatility = volatility * 0.95 + np.random.gamma(2, 0.001)
            volatility = np.clip(volatility, 0.005, 0.05)
            
            # Price movement
            change = np.random.normal(0.0001, volatility)
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 100))
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
            'close': prices,
            'volume': np.random.lognormal(8, 1.5, len(dates))
        })
        
        return df
    
    def _save_data(self):
        """Save data to disk"""
        with open(self.data_path, 'wb') as f:
            pickle.dump(self.data, f)
        
        # Save checksum
        checksum = self._calculate_checksum()
        with open(self.data_path.with_suffix('.md5'), 'w') as f:
            f.write(checksum)
    
    def _verify_data(self):
        """Verify data integrity"""
        if not self.data_path.exists():
            return False
        
        checksum_path = self.data_path.with_suffix('.md5')
        if not checksum_path.exists():
            return False
        
        with open(checksum_path, 'r') as f:
            stored_checksum = f.read()
        
        current_checksum = self._calculate_checksum()
        
        return stored_checksum == current_checksum
    
    def _calculate_checksum(self):
        """Calculate MD5 checksum of data file"""
        hash_md5 = hashlib.md5()
        with open(self.data_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def load_data(self):
        """Load data from disk"""
        with open(self.data_path, 'rb') as f:
            self.data = pickle.load(f)

# Model Manager with safe loading
class ModelManager:
    def __init__(self):
        self.current_version = 0
        self.model_prefix = "hashgremlin_model"
        self.best_models = []  # Track best models by validation score
    
    def save_model(self, agent: PPOAgent, update_num: int, metrics: Dict, is_best: bool = False, seed: int = 42):
        """Save model checkpoint with safe types"""
        self.current_version += 1
        
        # Get git commit hash if available
        try:
            git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()[:8]
        except:
            git_hash = 'no-git'
        
        # Convert metrics to Python types for safe serialization
        safe_metrics = to_python_types(metrics)
        
        checkpoint = {
            'version': self.current_version,
            'update_num': update_num,
            'model_state': agent.model.state_dict(),
            'actor_optimizer_state': agent.actor_optimizer.state_dict(),
            'critic_optimizer_state': agent.critic_optimizer.state_dict(),
            'metrics': safe_metrics,
            'seed': seed,
            'timestamp': datetime.now().isoformat(),
            'git_hash': git_hash,
            'update_count': agent.update_count,
            'last_val_return': agent.last_val_return
        }
        
        if is_best:
            filename = config.model_dir / f"{self.model_prefix}_best_v{self.current_version:06d}_sharpe{safe_metrics.get('val_sharpe', 0):.3f}.pt"
            self.best_models.append((safe_metrics.get('val_sharpe', 0), filename))
            self.best_models.sort(key=lambda x: x[0], reverse=True)
            
            # Keep only top N best models
            while len(self.best_models) > config.keep_best_n:
                _, old_file = self.best_models.pop()
                if old_file.exists():
                    old_file.unlink()
                    # Also remove sidecar JSON if it exists
                    old_json = old_file.with_suffix('.json')
                    if old_json.exists():
                        old_json.unlink()
        else:
            filename = config.model_dir / f"{self.model_prefix}_v{self.current_version:06d}.pt"
        
        torch.save(checkpoint, filename)
        
        # Clean up old regular models
        self._cleanup_old_models()
        
        return filename
    
    def load_latest_model(self, agent: PPOAgent):
        """Load the latest model checkpoint with safe fallback"""
        # Try to load best model first
        best_models = list(config.model_dir.glob(f"{self.model_prefix}_best_*.pt"))
        
        if best_models:
            # Parse Sharpe values from filenames and sort
            model_sharpes = []
            for model_file in best_models:
                try:
                    sharpe_str = model_file.stem.split('_sharpe')[-1]
                    sharpe_val = float(sharpe_str)
                    model_sharpes.append((sharpe_val, model_file))
                except:
                    continue
            
            if model_sharpes:
                model_sharpes.sort(key=lambda x: x[0], reverse=True)
                latest_file = model_sharpes[0][1]
            else:
                latest_file = sorted(best_models)[-1]
        else:
            # No best models, load most recent regular model
            model_files = sorted(config.model_dir.glob(f"{self.model_prefix}_*.pt"))
            if not model_files:
                return None, 0
            latest_file = model_files[-1]
        
        # Try safe loading first, fallback to unsafe if needed
        try:
            # Try PyTorch 2.6+ safe loading
            checkpoint = torch.load(latest_file, map_location=config.device, weights_only=True)
        except Exception as e:
            # Fallback to unsafe loading (safe for our own files)
            try:
                checkpoint = torch.load(latest_file, map_location=config.device, weights_only=False)
            except Exception as e2:
                # Older PyTorch versions
                checkpoint = torch.load(latest_file, map_location=config.device)
        
        agent.model.load_state_dict(checkpoint['model_state'])
        
        # Handle optimizer states (might be single or separate)
        if 'actor_optimizer_state' in checkpoint:
            agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state'])
            agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state'])
        elif 'optimizer_state' in checkpoint:
            # Old single optimizer format
            agent.actor_optimizer.load_state_dict(checkpoint['optimizer_state'])
            agent.critic_optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        # Restore update count for entropy decay
        if 'update_count' in checkpoint:
            agent.update_count = checkpoint['update_count']
        
        # Restore validation return for LR adjustment
        if 'last_val_return' in checkpoint:
            agent.last_val_return = checkpoint['last_val_return']
        
        self.current_version = checkpoint['version']
        
        return checkpoint, checkpoint.get('update_num', 0)
    
    def _cleanup_old_models(self):
        """Keep only recent models"""
        regular_models = sorted([f for f in config.model_dir.glob(f"{self.model_prefix}_v*.pt") 
                                if 'best' not in f.name])
        
        if len(regular_models) > config.keep_recent_n:
            for file in regular_models[:-config.keep_recent_n]:
                file.unlink()

# Training Statistics
class TrainingStats:
    def __init__(self):
        self.episode_rewards = deque(maxlen=1000)
        self.episode_returns = deque(maxlen=1000)
        self.episode_trades = deque(maxlen=1000)
        self.episode_lengths = deque(maxlen=1000)
        self.step_rewards = deque(maxlen=10000)
        self.losses = deque(maxlen=100)
        self.val_sharpes = deque(maxlen=100)
        self.val_returns = deque(maxlen=100)
        self.val_max_dds = deque(maxlen=100)
        self.start_time = time.time()
        self.total_episodes = 0
        self.total_updates = 0
        self.total_steps = 0
        self.current_position_cap = config.temp_position_cap
    
    def update_episodes(self, rewards, returns, trades, lengths, step_rewards):
        """Update episode statistics"""
        self.episode_rewards.extend(rewards)
        self.episode_returns.extend(returns)
        self.episode_trades.extend(trades)
        self.episode_lengths.extend(lengths)
        self.step_rewards.extend(step_rewards)
        self.total_episodes += len(rewards)
        self.total_steps += sum(lengths)
    
    def update_losses(self, loss_dict):
        """Update loss statistics"""
        self.losses.append(loss_dict)
        self.total_updates += 1
    
    def update_validation(self, sharpe, total_return, max_dd=0):
        """Update validation statistics"""
        self.val_sharpes.append(sharpe)
        self.val_returns.append(total_return)
        self.val_max_dds.append(max_dd)
    
    def update_position_cap(self, position_cap):
        """Update current position cap"""
        self.current_position_cap = position_cap
    
    def get_summary(self):
        """Get training summary"""
        if not self.episode_rewards:
            return {}
        
        elapsed_time = time.time() - self.start_time
        episodes_per_hour = (self.total_episodes / max(elapsed_time, 1)) * 3600.0
        steps_per_second = self.total_steps / elapsed_time if elapsed_time > 0 else 0
        
        return {
            'total_episodes': self.total_episodes,
            'total_updates': self.total_updates,
            'total_steps': self.total_steps,
            'avg_reward': np.mean(self.episode_rewards),
            'avg_return': np.mean(self.episode_returns),
            'avg_trades': np.mean(self.episode_trades),
            'avg_ep_length': np.mean(self.episode_lengths) if self.episode_lengths else 0,
            'avg_step_reward': np.mean(self.step_rewards) if self.step_rewards else 0,
            'best_return': max(self.episode_returns) if self.episode_returns else 0,
            'worst_return': min(self.episode_returns) if self.episode_returns else 0,
            'val_sharpe': np.mean(self.val_sharpes) if self.val_sharpes else 0,
            'val_return': np.mean(self.val_returns) if self.val_returns else 0,
            'val_max_dd': np.mean(self.val_max_dds) if self.val_max_dds else 0,
            'episodes_per_hour': episodes_per_hour,
            'steps_per_second': steps_per_second,
            'training_hours': elapsed_time / 3600,
            'position_cap': self.current_position_cap
        }

# Main Trainer
class HashGremlinTrainer:
    def __init__(self, seed: int = 42):
        # Set random seeds for reproducibility
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        self.data_manager = DataManager()
        self.model_manager = ModelManager()
        self.stats = TrainingStats()
        
        # Initialize model
        self.model = TradingNetwork(
            input_size=30,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            action_size=len(config.position_sizes)
        )
        self.agent = PPOAgent(self.model)
        
        # TensorBoard
        self.writer = SummaryWriter(config.log_dir)
        
        # KL tracking
        self.kl_history = deque(maxlen=100)
        
        # Track median Sharpe for position cap adjustment
        self.median_sharpe_history = deque(maxlen=10)
        
        self.running = False
        self.live_display = None
        self.training_started = False
    
    def initialize(self):
        """Initialize the trainer"""
        self._display_header()
        
        # Save configuration
        config.save(config.model_dir / "config.yaml")
        
        # Download/load data
        self.data_manager.download_data()
        
        # Load existing model if available
        checkpoint, start_update = self.model_manager.load_latest_model(self.agent)
        
        if checkpoint:
            console.print(f"\n[green]✅ Loaded model v{checkpoint['version']} from update {start_update}[/green]")
            self.stats.total_updates = start_update
            self.stats.total_episodes = checkpoint.get('metrics', {}).get('total_episodes', 0)
        else:
            console.print("\n[yellow]🆕 Starting fresh training session[/yellow]")
            start_update = 0
        
        return start_update
    
    def _display_header(self):
        """Display program header"""
        header = """
        ╔═══════════════════════════════════════════════════════════════════╗
        ║                                                                   ║
        ║     ██╗  ██╗ █████╗ ███████╗██╗  ██╗                              ║
        ║     ██║  ██║██╔══██╗██╔════╝██║  ██║                              ║
        ║     ███████║███████║███████╗███████║                              ║
        ║     ██╔══██║██╔══██║╚════██║██╔══██║                              ║
        ║     ██║  ██║██║  ██║███████║██║  ██║                              ║
        ║     ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝                              ║
        ║                                                                   ║
        ║      ██████╗ ██████╗ ███████╗███╗   ███╗██╗     ██╗███╗   ██╗     ║
        ║     ██╔════╝ ██╔══██╗██╔════╝████╗ ████║██║     ██║████╗  ██║     ║
        ║     ██║  ███╗██████╔╝█████╗  ██╔████╔██║██║     ██║██╔██╗ ██║     ║
        ║     ██║   ██║██╔══██╗██╔══╝  ██║╚██╔╝██║██║     ██║██║╚██╗██║     ║
        ║     ╚██████╔╝██║  ██║███████╗██║ ╚═╝ ██║███████╗██║██║ ╚████║     ║
        ║      ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝╚══════╝╚═╝╚═╝  ╚═══╝     ║
        ║                                                                   ║
        ║                      Created by lolitemaultes                     ║
        ║                           Version 2.9                             ║
        ║                                                                   ║
        ╚═══════════════════════════════════════════════════════════════════╝
        """
        
        console.print(header, style="bold cyan")
        console.print(f"\n[bold]System Info:[/bold]")
        console.print(f"  • Device: {config.device.upper()}")
        if config.device == "cuda":
            console.print(f"  • GPU: {torch.cuda.get_device_name(0)}")
            console.print(f"  • CUDA Version: {torch.version.cuda}")
        console.print(f"  • Random Seed: {self.seed}")
        console.print(f"  • Parallel Environments: {config.num_envs}")
        console.print(f"  • Episode Duration: {config.episode_hours} hours")
        console.print(f"  • Position Sizes: {config.position_sizes} (capped at {config.temp_position_cap:.0%} until Sharpe ≥ {config.position_cap_sharpe_threshold:.1f})")
        console.print(f"  • Actor LR: {config.actor_lr} [{config.min_actor_lr:.1e}, {config.max_actor_lr:.1e}]")
        console.print(f"  • Critic LR: {config.critic_lr} [{config.min_critic_lr:.1e}, {config.max_critic_lr:.1e}]")
        console.print(f"  • Entropy Decay: 0.012 → 0.0005 over 600 updates (fast cosine)")
        console.print(f"  • Commission: {config.commission_rate:.2%}, Min Position Change: {config.min_position_change:.0%}")
        console.print(f"  • Trade Cooldown: {config.trade_cooldown_steps} steps, Deadband: {config.position_deadband:.0%}")
        console.print(f"  • Eval Episodes: {config.eval_episodes} (median Sharpe + IQR robustness)")
        console.print(f"  • Risk-Adjusted Rewards: {'Enabled with guardrails' if config.reward_risk_adjusted else 'Disabled'}")
        console.print(f"  • Time Penalty: Exposure-scaled (no bleed when flat)")
        console.print(f"  • KL Divergence: True categorical KL (not approximation)")
    
    def collect_trajectories(self):
        """Collect trajectories from parallel environments"""
        # Get current position cap
        current_cap = config.get_position_cap(np.median(self.median_sharpe_history) if self.median_sharpe_history else -1.0)
        self.stats.update_position_cap(current_cap)
        
        envs = [BitcoinTradingEnv(self.data_manager.data, config.episode_hours, position_cap=current_cap) 
                for _ in range(config.num_envs)]
        
        all_states = []
        all_actions = []
        all_log_probs = []
        all_old_action_logits = []  # Store full logits for true KL
        all_values = []
        all_rewards = []
        all_dones = []
        
        episode_rewards = []
        episode_returns = []
        episode_trades = []
        episode_lengths = []
        step_rewards = []
        episode_win_rates = []
        episode_hold_times = []
        
        # Track episode metrics for each env
        env_episode_rewards = [0.0] * config.num_envs
        env_episode_lengths = [0] * config.num_envs
        
        # Reset all environments
        states = np.array([env.reset()[0] for env in envs])
        
        for step in range(config.steps_per_update // config.num_envs):
            # Get actions for all environments
            actions, values, log_probs, action_logits = self.agent.get_action_and_value(states)
            
            # Step all environments
            next_states = []
            rewards = []
            dones = []
            
            for i, env in enumerate(envs):
                next_state, reward, done, _, info = env.step(actions[i])
                next_states.append(next_state)
                rewards.append(reward)
                dones.append(done)
                
                # Accumulate true episode reward
                env_episode_rewards[i] += reward
                env_episode_lengths[i] += 1
                step_rewards.append(info.get('step_reward', reward))
                
                if done:
                    # Record true accumulated reward and other metrics
                    episode_rewards.append(env_episode_rewards[i])
                    episode_returns.append(info['return'])
                    episode_trades.append(info['trades'])
                    episode_lengths.append(env_episode_lengths[i])
                    episode_win_rates.append(info.get('win_rate', 0))
                    episode_hold_times.append(info.get('avg_hold_time', 0))
                    
                    # Log episode metrics to TensorBoard
                    if len(episode_returns) > 0 and len(episode_returns) % 10 == 0:
                        self.writer.add_scalar('Episode/Win_Rate', np.mean(episode_win_rates[-10:]), self.stats.total_episodes + len(episode_returns))
                        self.writer.add_scalar('Episode/Avg_Hold_Time', np.mean(episode_hold_times[-10:]), self.stats.total_episodes + len(episode_returns))
                    
                    # Reset tracking for this env
                    env_episode_rewards[i] = 0.0
                    env_episode_lengths[i] = 0
                    next_states[i] = env.reset()[0]
            
            # Store trajectories
            all_states.append(states)
            all_actions.append(actions)
            all_log_probs.append(log_probs)
            all_old_action_logits.append(action_logits)  # Store full logits
            all_values.append(values.flatten())
            all_rewards.append(rewards)
            all_dones.append(dones)
            
            states = np.array(next_states)
        
        # Flatten trajectories
        all_states = np.concatenate(all_states, axis=0)
        all_actions = np.concatenate(all_actions, axis=0)
        all_log_probs = np.concatenate(all_log_probs, axis=0)
        all_old_action_logits = np.concatenate(all_old_action_logits, axis=0)  # Flatten logits
        all_values = np.concatenate(all_values, axis=0)
        all_rewards = np.concatenate(all_rewards, axis=0)
        all_dones = np.concatenate(all_dones, axis=0)
        
        # Compute GAE
        advantages, returns = self.agent.compute_gae(all_rewards, all_values, all_dones, config.num_envs)
        
        # Create trajectory object
        trajectory = Trajectory(
            states=torch.FloatTensor(all_states),
            actions=torch.LongTensor(all_actions),
            log_probs=torch.FloatTensor(all_log_probs),
            old_action_logits=torch.FloatTensor(all_old_action_logits),  # Include logits
            values=torch.FloatTensor(all_values),
            rewards=torch.FloatTensor(all_rewards),
            dones=torch.FloatTensor(all_dones),
            advantages=torch.FloatTensor(advantages),
            returns=torch.FloatTensor(returns)
        )
        
        return trajectory, episode_rewards, episode_returns, episode_trades, episode_lengths, step_rewards
    
    def evaluate(self):
        """Evaluate model on validation data with multiple episodes"""
        # Set model to eval mode for deterministic evaluation
        self.agent.model.eval()
        
        try:
            # Get current position cap
            current_cap = config.get_position_cap(np.median(self.median_sharpe_history) if self.median_sharpe_history else -1.0)
            
            episode_rewards = []
            total_returns = []
            sharpes = []
            max_drawdowns = []
            trade_win_rates = []  # Win rates from individual episodes
            avg_hold_times = []
            avg_exposures = []
            
            # Track wins across all episodes
            episode_wins = 0
            
            for _ in range(config.eval_episodes):
                eval_env = BitcoinTradingEnv(self.data_manager.data, config.episode_hours, eval_mode=True, position_cap=current_cap)
                state, _ = eval_env.reset()
                episode_reward = 0
                
                while True:
                    # Use deterministic action for evaluation
                    action, _, _, _ = self.agent.get_action_and_value(state, deterministic=True)
                    state, reward, done, _, info = eval_env.step(action[0])
                    episode_reward += reward
                    
                    if done:
                        episode_rewards.append(episode_reward)
                        episode_return = info['return']
                        total_returns.append(episode_return)
                        
                        # Count episode as win if positive return
                        if episode_return > 0:
                            episode_wins += 1
                        
                        sharpes.append(info.get('sharpe', 0))
                        max_drawdowns.append(info.get('max_drawdown', 0))
                        trade_win_rates.append(info.get('win_rate', 0))  # Trade-level win rate
                        avg_hold_times.append(info.get('avg_hold_time', 0))
                        avg_exposures.append(info.get('avg_exposure', 0))
                        break
            
            # Calculate episode-level win rate (% of profitable episodes)
            episode_win_rate = float(episode_wins) / float(config.eval_episodes)
            
            # Calculate IQR for robustness
            sharpe_q1 = np.percentile(sharpes, 25)
            sharpe_q3 = np.percentile(sharpes, 75)
            sharpe_iqr = sharpe_q3 - sharpe_q1
            
            return {
                'val_reward': np.mean(episode_rewards),
                'val_return': np.mean(total_returns),
                'val_sharpe': np.median(sharpes),  # Use median for stability
                'val_sharpe_mean': np.mean(sharpes),  # Also track mean
                'val_sharpe_iqr': sharpe_iqr,  # Interquartile range
                'val_sharpe_q1': sharpe_q1,  # First quartile
                'val_sharpe_q3': sharpe_q3,  # Third quartile
                'val_max_dd': np.mean(max_drawdowns),
                'val_sharpe_std': np.std(sharpes),
                'val_return_std': np.std(total_returns),
                'val_win_rate': episode_win_rate,  # Episode-level win rate
                'val_trade_win_rate': np.mean(trade_win_rates),  # Trade-level win rate
                'val_avg_hold_time': np.mean(avg_hold_times),
                'val_avg_exposure': np.mean(avg_exposures)
            }
        finally:
            # Set model back to train mode
            self.agent.model.train()
    
    def training_loop(self, start_update: int):
        """Main training loop"""
        update_num = start_update
        self.training_started = True
        
        with Live(self._create_display_layout(), refresh_per_second=1, console=console) as live:
            self.live_display = live
            
            while self.running:
                try:
                    # Collect trajectories
                    trajectory, ep_rewards, ep_returns, ep_trades, ep_lengths, step_rewards = self.collect_trajectories()
                    
                    # Update statistics
                    self.stats.update_episodes(ep_rewards, ep_returns, ep_trades, ep_lengths, step_rewards)
                    
                    # Update model
                    loss_dict = self.agent.update(trajectory)
                    self.stats.update_losses(loss_dict)
                    
                    # Track KL divergence
                    current_kl = loss_dict.get('kl_divergence', 0)
                    self.kl_history.append(current_kl)
                    
                    update_num += 1
                    
                    # Log to TensorBoard
                    for key, value in loss_dict.items():
                        self.writer.add_scalar(f'Loss/{key}', value, update_num)
                    
                    # Log current entropy coefficient
                    current_entropy_coef = config.get_entropy_coef(update_num)
                    self.writer.add_scalar('Loss/entropy_coefficient', current_entropy_coef, update_num)
                    
                    # Log KL moving average
                    if self.kl_history:
                        self.writer.add_scalar('Loss/kl_divergence_avg', np.mean(self.kl_history), update_num)
                    
                    # Log position cap
                    self.writer.add_scalar('Training/position_cap', self.stats.current_position_cap, update_num)
                    
                    if len(ep_rewards) > 0:
                        self.writer.add_scalar('Episode/Mean_Reward', np.mean(ep_rewards), update_num)
                        self.writer.add_scalar('Episode/Mean_Return', np.mean(ep_returns), update_num)
                        self.writer.add_scalar('Episode/Mean_Trades', np.mean(ep_trades), update_num)
                        self.writer.add_scalar('Episode/Mean_Length', np.mean(ep_lengths), update_num)
                    
                    if len(step_rewards) > 0:
                        self.writer.add_scalar('Step/Mean_Reward', np.mean(step_rewards), update_num)
                    
                    # Evaluate periodically
                    if update_num % config.eval_interval == 0:
                        # Store previous best BEFORE updating stats
                        prev_best_sharpe = max(self.stats.val_sharpes) if self.stats.val_sharpes else -1e9
                        
                        eval_metrics = self.evaluate()
                        
                        # Update median Sharpe history for position cap
                        self.median_sharpe_history.append(eval_metrics['val_sharpe'])
                        
                        # Pass validation return to agent for LR adjustment
                        self.agent.last_val_return = eval_metrics['val_return']
                        
                        self.stats.update_validation(
                            eval_metrics['val_sharpe'],  # Now using median Sharpe
                            eval_metrics['val_return'],
                            eval_metrics.get('val_max_dd', 0)
                        )
                        
                        # Log all metrics to TensorBoard
                        for key, value in eval_metrics.items():
                            self.writer.add_scalar(f'Validation/{key}', value, update_num)
                        
                        # Additional metrics for better analysis
                        self.writer.add_scalar('Validation/sharpe_median', eval_metrics['val_sharpe'], update_num)
                        self.writer.add_scalar('Validation/sharpe_mean', eval_metrics['val_sharpe_mean'], update_num)
                        self.writer.add_scalar('Validation/sharpe_iqr', eval_metrics['val_sharpe_iqr'], update_num)
                        self.writer.add_scalar('Validation/episode_win_rate', eval_metrics.get('val_win_rate', 0), update_num)
                        self.writer.add_scalar('Validation/trade_win_rate', eval_metrics.get('val_trade_win_rate', 0), update_num)
                        self.writer.add_scalar('Validation/avg_hold_time', eval_metrics.get('val_avg_hold_time', 0), update_num)
                        self.writer.add_scalar('Validation/avg_exposure', eval_metrics.get('val_avg_exposure', 0), update_num)
                        
                        # Check if this is the best model with tighter criteria including IQR
                        is_best = (eval_metrics['val_sharpe'] > prev_best_sharpe and 
                                 eval_metrics['val_sharpe'] > config.best_model_min_sharpe and  # Require meaningful positive Sharpe
                                 eval_metrics['val_sharpe_std'] < config.best_model_max_std and  # Avoid high variance spikes
                                 eval_metrics['val_sharpe_iqr'] < config.best_model_max_iqr and  # Avoid jumpy windows
                                 eval_metrics['val_win_rate'] > config.best_model_min_win_rate)  # Require consistent wins
                        
                        if is_best:
                            console.print(f"[bold green]🏆 New best model! Sharpe (median): {eval_metrics['val_sharpe']:.3f} [Q1: {eval_metrics['val_sharpe_q1']:.3f}, Q3: {eval_metrics['val_sharpe_q3']:.3f}][/bold green]")
                            console.print(f"[dim]Episode Win Rate: {eval_metrics.get('val_win_rate', 0):.2%}, Trade Win Rate: {eval_metrics.get('val_trade_win_rate', 0):.2%}, Avg Hold: {eval_metrics.get('val_avg_hold_time', 0):.1f} steps, Avg Exposure: {eval_metrics.get('val_avg_exposure', 0):.2%}[/dim]")
                            # 💾 Save best model immediately
                            summary = self.stats.get_summary()
                            summary.update(eval_metrics)  # Include eval metrics in summary
                            best_file = self.model_manager.save_model(self.agent, update_num, summary, is_best=True, seed=self.seed)
                            console.print(f"[green]💾 Saved best model v{self.model_manager.current_version}[/green]")
                            
                            # Save eval metrics to sidecar JSON for easy comparison
                            if best_file:
                                metrics_file = best_file.with_suffix('.json')
                                with open(metrics_file, 'w') as f:
                                    json.dump({
                                        'update_num': update_num,
                                        'eval_metrics': eval_metrics,
                                        'summary': summary,
                                        'timestamp': datetime.now().isoformat()
                                    }, f, indent=2)
                                console.print(f"[dim]📊 Saved metrics to {metrics_file.name}[/dim]")
                        
                        # Check if we should lift position cap
                        new_cap = config.get_position_cap(eval_metrics['val_sharpe'])
                        if new_cap != self.stats.current_position_cap:
                            self.stats.update_position_cap(new_cap)
                            if new_cap == 1.0:
                                console.print(f"[bold green]🚀 Position cap removed! Median Sharpe {eval_metrics['val_sharpe']:.3f} >= {config.position_cap_sharpe_threshold:.1f}[/bold green]")
                            else:
                                console.print(f"[yellow]Position cap adjusted to {new_cap:.0%}[/yellow]")
                    
                    # Save regular checkpoints periodically
                    if update_num % config.save_interval == 0:
                        summary = self.stats.get_summary()
                        model_file = self.model_manager.save_model(self.agent, update_num, summary, is_best=False, seed=self.seed)
                        console.print(f"\n[green]💾 Saved checkpoint v{self.model_manager.current_version} at update {update_num}[/green]")
                    
                    # Update display
                    live.update(self._create_display_layout())
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    console.print(f"[red]Training error: {e}[/red]")
                    # Reset optimizers and scaler to clean state for next iteration
                    self.agent.actor_optimizer.zero_grad(set_to_none=True)
                    self.agent.critic_optimizer.zero_grad(set_to_none=True)
                    if self.agent.use_amp:
                        # Reset AMP scaler to fresh state
                        self.agent.scaler = torch.cuda.amp.GradScaler()
                    import traceback
                    traceback.print_exc()
                    time.sleep(1)
    
    def _create_display_layout(self):
        """Create live display layout"""
        summary = self.stats.get_summary()
        
        # Main stats table
        stats_table = Table(title="📊 Training Statistics", box=box.ROUNDED)
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")
        
        stats_table.add_row("Total Episodes", f"{summary.get('total_episodes', 0):,}")
        stats_table.add_row("Total Updates", f"{summary.get('total_updates', 0):,}")
        stats_table.add_row("Total Steps", f"{summary.get('total_steps', 0):,}")
        stats_table.add_row("Episodes/Hour", f"{summary.get('episodes_per_hour', 0):.1f}")
        stats_table.add_row("Steps/Second", f"{summary.get('steps_per_second', 0):.1f}")
        stats_table.add_row("Avg Episode Length", f"{summary.get('avg_ep_length', 0):.1f}")
        stats_table.add_row("Training Hours", f"{summary.get('training_hours', 0):.2f}")
        
        # Add position cap status
        position_cap = summary.get('position_cap', 1.0)
        if position_cap < 1.0:
            stats_table.add_row("Position Cap", f"[yellow]{position_cap:.0%}[/yellow]")
        else:
            stats_table.add_row("Position Cap", f"[green]None[/green]")
        
        # Performance table
        perf_table = Table(title="💰 Performance Metrics", box=box.ROUNDED)
        perf_table.add_column("Metric", style="cyan")
        perf_table.add_column("Value", style="green")
        
        perf_table.add_row("Avg Episode Reward", f"{summary.get('avg_reward', 0):.4f}")
        perf_table.add_row("Avg Portfolio Return %", f"{summary.get('avg_return', 0)*100:.2f}%")
        perf_table.add_row("Best Return %", f"{summary.get('best_return', 0)*100:.2f}%")
        perf_table.add_row("Worst Return %", f"{summary.get('worst_return', 0)*100:.2f}%")
        perf_table.add_row("Avg Trades/Episode", f"{summary.get('avg_trades', 0):.1f}")
        perf_table.add_row("Avg Step Reward", f"{summary.get('avg_step_reward', 0):.6f}")
        
        val_sharpe = summary.get('val_sharpe', 0)  # This is now median Sharpe
        sharpe_color = "green" if val_sharpe > 0 else "red"
        perf_table.add_row("Val Sharpe (median)", f"[{sharpe_color}]{val_sharpe:.3f}[/{sharpe_color}]")
        
        # Add IQR if available
        if self.stats.val_sharpes and len(self.stats.val_sharpes) >= config.eval_episodes:
            recent_sharpes = list(self.stats.val_sharpes)[-config.eval_episodes:]
            q1 = np.percentile(recent_sharpes, 25)
            q3 = np.percentile(recent_sharpes, 75)
            perf_table.add_row("Sharpe IQR", f"[{q1:.3f}, {q3:.3f}]")
        
        perf_table.add_row("Val Return %", f"{summary.get('val_return', 0)*100:.2f}%")
        perf_table.add_row("Val Max DD %", f"{summary.get('val_max_dd', 0)*100:.2f}%")
        
        # Loss table
        if self.stats.losses:
            loss_table = Table(title="📉 Loss & Learning Metrics", box=box.ROUNDED)
            loss_table.add_column("Metric", style="cyan")
            loss_table.add_column("Value", style="yellow")
            
            recent_loss = self.stats.losses[-1]
            loss_table.add_row("Total Loss", f"{recent_loss.get('total_loss', 0):.4f}")
            loss_table.add_row("Policy Loss", f"{recent_loss.get('policy_loss', 0):.4f}")
            loss_table.add_row("Value Loss", f"{recent_loss.get('value_loss', 0):.4f}")
            loss_table.add_row("Entropy", f"{recent_loss.get('entropy', 0):.4f}")
            loss_table.add_row("Entropy Coef", f"{recent_loss.get('entropy_coef', 0):.5f}")
            
            # Enhanced KL tracking
            current_kl = recent_loss.get('kl_divergence', 0)
            avg_kl = np.mean(self.kl_history) if self.kl_history else 0
            kl_color = "yellow" if current_kl > config.target_kl * 0.8 else "green"
            loss_table.add_row("KL Divergence", f"[{kl_color}]{current_kl:.4f}[/{kl_color}]")
            loss_table.add_row("KL Avg (100)", f"{avg_kl:.4f}")
            
            # Learning rates with trend indicators
            actor_lr = recent_loss.get('actor_lr', 0)
            critic_lr = recent_loss.get('critic_lr', 0)
            
            # Show trend if we have history
            lr_trend = ""
            if len(self.stats.losses) > 1 and 'actor_lr' in self.stats.losses[-2]:
                prev_actor_lr = self.stats.losses[-2]['actor_lr']
                if actor_lr < prev_actor_lr * 0.99:
                    lr_trend = " ↓"
                elif actor_lr > prev_actor_lr * 1.01:
                    lr_trend = " ↑"
                else:
                    lr_trend = " →"
            
            loss_table.add_row("Actor LR", f"{actor_lr:.2e}{lr_trend}")
            loss_table.add_row("Critic LR", f"{critic_lr:.2e}")
            
            # Show if early stopping triggered
            if recent_loss.get('update_epochs', 0) < config.update_epochs:
                loss_table.add_row("Early Stop", f"Yes (epoch {recent_loss.get('update_epochs', 0)})")
        else:
            loss_table = Table(title="📉 Loss & Learning Metrics", box=box.ROUNDED)
            loss_table.add_column("Waiting for first update...", style="dim")
        
        # Create layout
        layout = Layout()
        layout.split_column(
            Layout(Panel(Align.center(Text("HashGremlin v2.9 - Bitcoin Trading AI", style="bold cyan"))), size=3),
            Layout(name="main", ratio=1)
        )
        
        layout["main"].split_row(
            Layout(stats_table),
            Layout(perf_table),
            Layout(loss_table)
        )
        
        return layout
    
    def run(self):
        """Main training loop"""
        try:
            start_update = self.initialize()
            
            console.print("\n[bold green]🚀 Starting infinite training...[/bold green]")
            console.print("[dim]The AI will continuously improve its trading strategy[/dim]\n")
            console.print("[dim]Press Ctrl+C to stop training[/dim]\n")
            
            self.running = True
            
            # Start training
            self.training_loop(start_update)
            
        except KeyboardInterrupt:
            console.print("\n\n[yellow]⚠️  Training interrupted by user[/yellow]")
        except Exception as e:
            console.print(f"\n[red]❌ Fatal error: {e}[/red]")
            import traceback
            traceback.print_exc()
        finally:
            self.running = False
            
            # Final save if training actually started
            if self.training_started and self.stats.total_updates > 0:
                summary = self.stats.get_summary()
                self.model_manager.save_model(self.agent, self.stats.total_updates, summary, seed=self.seed)
                console.print(f"\n[green]✅ Final model saved (v{self.model_manager.current_version})[/green]")
            
            self.writer.close()
            
            # Only show thank you message if training actually happened
            if self.training_started:
                console.print("\n[bold cyan]Thank you for using HashGremlin v2.9![/bold cyan]")
                console.print("[dim]Your AI trader has been saved and will resume from here next time.[/dim]\n")

# Entry point
def main():
    # Check dependencies
    package_imports = {
        'torch': 'torch',
        'pandas': 'pandas', 
        'numpy': 'numpy',
        'requests': 'requests',
        'rich': 'rich',
        'gymnasium': 'gymnasium',
        'tensorboard': 'tensorboard',
        'pyyaml': 'yaml'
    }
    
    missing = []
    for package, import_name in package_imports.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(package)
    
    if missing:
        console.print(f"[red]Missing required packages: {', '.join(missing)}[/red]")
        console.print(f"[yellow]Install with: pip install {' '.join(missing)}[/yellow]")
        sys.exit(1)
    
    # Parse optional seed argument
    seed = 42  # Default seed
    if len(sys.argv) > 1:
        try:
            seed = int(sys.argv[1])
            console.print(f"[green]Using random seed: {seed}[/green]")
        except ValueError:
            console.print(f"[yellow]Invalid seed '{sys.argv[1]}', using default: {seed}[/yellow]")
    
    # Set global random seed
    random.seed(seed)
    
    trainer = HashGremlinTrainer(seed=seed)
    trainer.run()

if __name__ == "__main__":
    main()
