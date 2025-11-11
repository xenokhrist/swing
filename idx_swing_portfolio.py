#!/usr/bin/env python3
"""
IDX Swing Screener - Portfolio Manager & Risk Control
Handles position tracking, portfolio heat, and exit management

Author: Portfolio Management Module
Version: 2.0.0
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import json
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# POSITION TRACKING
# ============================================================================

@dataclass
class Position:
    """Active position tracking"""
    symbol: str
    entry_date: str
    entry_price: float
    shares: int
    stop_loss: float
    target_1: float
    target_2: float
    
    # Current state
    current_price: float = 0.0
    days_held: int = 0
    
    # P&L tracking
    unrealized_pnl_idr: float = 0.0
    unrealized_pnl_pct: float = 0.0
    
    # Exit tracking
    exit_triggered: bool = False
    exit_reason: Optional[str] = None
    exit_date: Optional[str] = None
    exit_price: Optional[float] = None
    realized_pnl_idr: Optional[float] = None
    realized_pnl_pct: Optional[float] = None
    
    # Metadata
    setup_quality: str = "B"
    original_signal_score: float = 0.0
    
    def update(self, current_price: float, current_date: str):
        """Update position with current market data"""
        self.current_price = current_price
        entry_datetime = datetime.strptime(self.entry_date, '%Y-%m-%d')
        current_datetime = datetime.strptime(current_date, '%Y-%m-%d')
        self.days_held = (current_datetime - entry_datetime).days
        
        position_value = self.shares * current_price
        cost_basis = self.shares * self.entry_price
        
        self.unrealized_pnl_idr = position_value - cost_basis
        self.unrealized_pnl_pct = (current_price / self.entry_price - 1) * 100
    
    def check_exit(self, current_price: float, current_date: str, max_days: int = 10) -> Tuple[bool, str]:
        """
        Check if position should be exited
        Returns (should_exit, reason)
        """
        self.update(current_price, current_date)
        
        # Stop loss hit
        if current_price <= self.stop_loss:
            return True, f"Stop loss ({self.stop_loss:.0f})"
        
        # Target 1 hit (take partial profit in real trading)
        if current_price >= self.target_1:
            return True, f"Target 1 ({self.target_1:.0f})"
        
        # Time stop (holding too long)
        if self.days_held >= max_days:
            return True, f"Time stop ({self.days_held} days)"
        
        # Trailing stop (if price went above entry by 10%, trail stop to breakeven)
        if current_price >= self.entry_price * 1.10:
            adjusted_stop = self.entry_price * 1.01  # 1% above entry
            if current_price <= adjusted_stop:
                return True, f"Trailing stop ({adjusted_stop:.0f})"
        
        return False, ""
    
    def close_position(self, exit_price: float, exit_date: str, reason: str):
        """Mark position as closed"""
        self.exit_triggered = True
        self.exit_price = exit_price
        self.exit_date = exit_date
        self.exit_reason = reason
        
        gross_pnl = (exit_price - self.entry_price) * self.shares
        
        # Account for transaction costs (both entry and exit)
        entry_cost = self.entry_price * self.shares * 0.0040  # 0.40% round-trip
        exit_cost = exit_price * self.shares * 0.0040
        
        self.realized_pnl_idr = gross_pnl - entry_cost - exit_cost
        self.realized_pnl_pct = (self.realized_pnl_idr / (self.entry_price * self.shares)) * 100

@dataclass
class Portfolio:
    """Portfolio state and risk management"""
    total_capital: float
    max_positions: int = 5
    max_portfolio_heat: float = 0.06
    
    positions: List[Position] = field(default_factory=list)
    closed_positions: List[Position] = field(default_factory=list)
    
    def add_position(self, position: Position) -> bool:
        """Add new position if risk limits allow"""
        if len(self.positions) >= self.max_positions:
            logger.warning(f"Cannot add {position.symbol}: max positions reached")
            return False
        
        # Check portfolio heat
        current_heat = self.get_portfolio_heat()
        position_risk = abs(position.entry_price - position.stop_loss) / position.entry_price
        new_heat = current_heat + position_risk
        
        if new_heat > self.max_portfolio_heat:
            logger.warning(f"Cannot add {position.symbol}: would exceed portfolio heat limit")
            return False
        
        self.positions.append(position)
        logger.info(f"Added position: {position.symbol} ({position.shares} shares @ {position.entry_price:.0f})")
        return True
    
    def update_positions(self, market_data: Dict[str, float], current_date: str):
        """Update all positions with current prices"""
        for pos in self.positions:
            if pos.symbol in market_data:
                current_price = market_data[pos.symbol]
                
                # Check exit conditions
                should_exit, reason = pos.check_exit(current_price, current_date)
                
                if should_exit:
                    pos.close_position(current_price, current_date, reason)
                    self.closed_positions.append(pos)
                    logger.info(f"Closed {pos.symbol}: {reason} | P&L: {pos.realized_pnl_idr:,.0f} IDR ({pos.realized_pnl_pct:+.1f}%)")
        
        # Remove closed positions from active
        self.positions = [p for p in self.positions if not p.exit_triggered]
    
    def get_portfolio_heat(self) -> float:
        """Calculate current portfolio risk (heat)"""
        total_risk = 0.0
        for pos in self.positions:
            position_risk = abs(pos.entry_price - pos.stop_loss) / pos.entry_price
            total_risk += position_risk
        return total_risk
    
    def get_portfolio_value(self) -> float:
        """Get current portfolio value"""
        cash = self.total_capital
        
        # Subtract capital deployed in positions
        for pos in self.positions:
            cash -= pos.entry_price * pos.shares
        
        # Add current position values
        for pos in self.positions:
            cash += pos.current_price * pos.shares
        
        return cash
    
    def get_statistics(self) -> Dict:
        """Calculate portfolio performance statistics"""
        if not self.closed_positions:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_win_pct': 0.0,
                'avg_loss_pct': 0.0,
                'profit_factor': 0.0,
                'total_pnl_idr': 0.0,
                'total_pnl_pct': 0.0
            }
        
        wins = [p for p in self.closed_positions if p.realized_pnl_idr and p.realized_pnl_idr > 0]
        losses = [p for p in self.closed_positions if p.realized_pnl_idr and p.realized_pnl_idr <= 0]
        
        total_win_amount = sum(p.realized_pnl_idr for p in wins)
        total_loss_amount = abs(sum(p.realized_pnl_idr for p in losses))
        
        return {
            'total_trades': len(self.closed_positions),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': len(wins) / len(self.closed_positions) * 100,
            'avg_win_pct': np.mean([p.realized_pnl_pct for p in wins]) if wins else 0.0,
            'avg_loss_pct': np.mean([p.realized_pnl_pct for p in losses]) if losses else 0.0,
            'profit_factor': total_win_amount / total_loss_amount if total_loss_amount > 0 else 0.0,
            'total_pnl_idr': sum(p.realized_pnl_idr for p in self.closed_positions if p.realized_pnl_idr),
            'total_pnl_pct': (sum(p.realized_pnl_idr for p in self.closed_positions if p.realized_pnl_idr) / self.total_capital) * 100
        }
    
    def save_state(self, filename: str = "portfolio_state.json"):
        """Save portfolio state to file"""
        state = {
            'total_capital': self.total_capital,
            'timestamp': datetime.now().isoformat(),
            'active_positions': [
                {
                    'symbol': p.symbol,
                    'entry_date': p.entry_date,
                    'entry_price': p.entry_price,
                    'shares': p.shares,
                    'stop_loss': p.stop_loss,
                    'target_1': p.target_1,
                    'target_2': p.target_2,
                    'current_price': p.current_price,
                    'unrealized_pnl_idr': p.unrealized_pnl_idr,
                    'unrealized_pnl_pct': p.unrealized_pnl_pct,
                    'days_held': p.days_held
                }
                for p in self.positions
            ],
            'closed_positions': [
                {
                    'symbol': p.symbol,
                    'entry_date': p.entry_date,
                    'entry_price': p.entry_price,
                    'shares': p.shares,
                    'exit_date': p.exit_date,
                    'exit_price': p.exit_price,
                    'exit_reason': p.exit_reason,
                    'realized_pnl_idr': p.realized_pnl_idr,
                    'realized_pnl_pct': p.realized_pnl_pct,
                    'days_held': p.days_held
                }
                for p in self.closed_positions
            ],
            'statistics': self.get_statistics()
        }
        
        with open(filename, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Portfolio state saved to {filename}")

# ============================================================================
# TRADE EXECUTION SIMULATOR
# ============================================================================

class TradeExecutor:
    """
    Simulates realistic trade execution for backtesting
    Accounts for slippage, partial fills, and market impact
    """
    
    def __init__(self, slippage_bps: float = 15, min_liquidity_pct: float = 0.05):
        self.slippage_bps = slippage_bps
        self.min_liquidity_pct = min_liquidity_pct  # Max 5% of daily volume
    
    def execute_entry(self, 
                      signal_price: float, 
                      desired_shares: int,
                      avg_daily_volume: float) -> Tuple[bool, float, int, str]:
        """
        Simulate entry execution
        Returns: (success, fill_price, filled_shares, message)
        """
        # Check if order size is reasonable vs liquidity
        max_shares = int(avg_daily_volume * self.min_liquidity_pct)
        
        if desired_shares > max_shares:
            filled_shares = max_shares
            message = f"Partial fill: {filled_shares}/{desired_shares} (liquidity constraint)"
        else:
            filled_shares = desired_shares
            message = "Full fill"
        
        # Apply slippage (worse price for entry)
        slippage_factor = 1 + (self.slippage_bps / 10000)
        fill_price = signal_price * slippage_factor
        
        return True, fill_price, filled_shares, message
    
    def execute_exit(self,
                     signal_price: float,
                     shares: int,
                     avg_daily_volume: float) -> Tuple[bool, float, str]:
        """
        Simulate exit execution
        Returns: (success, fill_price, message)
        """
        # Check liquidity
        max_shares = int(avg_daily_volume * self.min_liquidity_pct)
        
        if shares > max_shares:
            message = f"WARNING: Exit size ({shares}) exceeds 5% of daily volume"
        else:
            message = "Normal exit"
        
        # Apply slippage (worse price for exit)
        slippage_factor = 1 - (self.slippage_bps / 10000)
        fill_price = signal_price * slippage_factor
        
        return True, fill_price, message

# ============================================================================
# REPORTING & VISUALIZATION
# ============================================================================

class PerformanceReporter:
    """Generate performance reports and metrics"""
    
    @staticmethod
    def print_signal_report(signals: List, market_regime: str):
        """Print formatted signal report"""
        if not signals:
            print(f"\n{'='*100}")
            print(f"NO VALID SIGNALS FOUND | Market Regime: {market_regime}")
            print(f"{'='*100}")
            return
        
        print(f"\n{'='*100}")
        print(f"SWING TRADE SIGNALS | Market Regime: {market_regime} | Signals: {len(signals)}")
        print(f"{'='*100}")
        
        # Group by quality
        a_grade = [s for s in signals if s.setup_quality == 'A']
        b_grade = [s for s in signals if s.setup_quality == 'B']
        c_grade = [s for s in signals if s.setup_quality == 'C']
        
        print(f"\nQuality Distribution: A-Grade ({len(a_grade)}) | B-Grade ({len(b_grade)}) | C-Grade ({len(c_grade)})")
        print(f"\n{'Rank':<4} {'Symbol':<10} {'Grade':<6} {'Score':<6} {'Price':<8} {'Target':<8} {'R:R':<7} {'Risk%':<7} {'Size(IDR)':<12} {'Hold'}")
        print(f"{'-'*100}")
        
        for i, sig in enumerate(signals[:15], 1):  # Top 15
            symbol_clean = sig.symbol.replace('.JK', '')
            
            print(f"{i:<4} {symbol_clean:<10} {sig.setup_quality:<6} {sig.technical_score:<6.1f} "
                  f"{sig.current_price:<8.0f} {sig.target_1:<8.0f} {sig.risk_reward_t1:<7.1f} "
                  f"{sig.risk_pct*100:<7.1f} {sig.position_size_idr:<12,.0f} {sig.holding_period}")
        
        print(f"\n{'-'*100}")
        
        # Top 3 detailed
        print(f"\nTOP 3 RECOMMENDATIONS:")
        print(f"{'='*100}")
        
        for i, sig in enumerate(signals[:3], 1):
            symbol_clean = sig.symbol.replace('.JK', '')
            
            print(f"\n🎯 #{i} {symbol_clean} | Grade: {sig.setup_quality} | Score: {sig.technical_score:.1f}/10")
            print(f"   Entry: {sig.entry_price:.0f} IDR")
            print(f"   Stop:  {sig.stop_loss:.0f} IDR ({sig.risk_pct*100:.1f}%)")
            print(f"   Target 1: {sig.target_1:.0f} IDR ({sig.reward_pct_t1*100:.1f}% | R:R {sig.risk_reward_t1:.1f}:1)")
            print(f"   Target 2: {sig.target_2:.0f} IDR ({sig.reward_pct_t2*100:.1f}% | R:R {sig.risk_reward_t2:.1f}:1)")
            print(f"   Position: {sig.position_size_shares:,} shares = {sig.position_size_idr:,.0f} IDR")
            print(f"   Max Loss: {sig.max_loss_idr:,.0f} IDR")
            print(f"   Setup: {sig.entry_reason}")
            print(f"   Hold: {sig.holding_period} | Confidence: {sig.confidence}")
            
            if sig.warnings:
                print(f"   ⚠️  Warnings: {', '.join(sig.warnings)}")
    
    @staticmethod
    def print_portfolio_report(portfolio: Portfolio):
        """Print portfolio status report"""
        stats = portfolio.get_statistics()
        
        print(f"\n{'='*100}")
        print(f"PORTFOLIO STATUS")
        print(f"{'='*100}")
        
        print(f"\nCapital: {portfolio.total_capital:,.0f} IDR")
        print(f"Portfolio Value: {portfolio.get_portfolio_value():,.0f} IDR")
        print(f"Portfolio Heat: {portfolio.get_portfolio_heat():.1%} / {portfolio.max_portfolio_heat:.1%}")
        
        print(f"\nActive Positions: {len(portfolio.positions)}/{portfolio.max_positions}")
        if portfolio.positions:
            print(f"\n{'Symbol':<10} {'Entry':<10} {'Current':<10} {'P&L%':<8} {'Days':<5} {'Status'}")
            print(f"{'-'*70}")
            for pos in portfolio.positions:
                status = "🟢" if pos.unrealized_pnl_pct > 0 else "🔴"
                print(f"{pos.symbol.replace('.JK', ''):<10} {pos.entry_price:<10.0f} "
                      f"{pos.current_price:<10.0f} {pos.unrealized_pnl_pct:<8.1f} "
                      f"{pos.days_held:<5} {status}")
        
        if stats['total_trades'] > 0:
            print(f"\nClosed Trades: {stats['total_trades']}")
            print(f"Win Rate: {stats['win_rate']:.1f}% ({stats['wins']}W / {stats['losses']}L)")
            print(f"Avg Win: {stats['avg_win_pct']:+.1f}% | Avg Loss: {stats['avg_loss_pct']:+.1f}%")
            print(f"Profit Factor: {stats['profit_factor']:.2f}")
            print(f"Total P&L: {stats['total_pnl_idr']:,.0f} IDR ({stats['total_pnl_pct']:+.1f}%)")
    
    @staticmethod
    def export_signals_to_csv(signals: List, filename: str = None):
        """Export signals to CSV for further analysis"""
        if not filename:
            filename = f"swing_signals_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        
        data = []
        for sig in signals:
            data.append({
                'Symbol': sig.symbol.replace('.JK', ''),
                'Date': sig.signal_date,
                'Grade': sig.setup_quality,
                'Score': round(sig.technical_score, 2),
                'Current_Price': sig.current_price,
                'Entry_Price': sig.entry_price,
                'Stop_Loss': sig.stop_loss,
                'Target_1': sig.target_1,
                'Target_2': sig.target_2,
                'Risk_Pct': round(sig.risk_pct * 100, 2),
                'Reward_Pct_T1': round(sig.reward_pct_t1 * 100, 2),
                'RR_Ratio': round(sig.risk_reward_t1, 2),
                'Position_Shares': sig.position_size_shares,
                'Position_IDR': sig.position_size_idr,
                'Max_Loss_IDR': sig.max_loss_idr,
                'Holding_Period': sig.holding_period,
                'Confidence': sig.confidence,
                'ADX': round(sig.trend_strength, 1),
                'Entry_Reason': sig.entry_reason,
                'Warnings': '; '.join(sig.warnings) if sig.warnings else ''
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        
        logger.info(f"Signals exported to {filename}")
        return filename

# ============================================================================
# END OF PORTFOLIO MANAGER
# ============================================================================
