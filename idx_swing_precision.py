#!/usr/bin/env python3
"""
IDX Swing Screener - Precision Entry Module
Advanced entry logic: Buy on weakness, respect price structure, layered entries

Philosophy:
- Don't chase strength - wait for pullbacks
- Use support levels and price structure
- Scale into positions (partial entries)
- Invalidation levels (structure breaks)

Author: Precision Entry System
Version: 2.1.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# PRICE STRUCTURE DEFINITIONS
# ============================================================================

class PriceStructure(Enum):
    """Price structure classification"""
    HIGHER_HIGHS_HIGHER_LOWS = "HH_HL"  # Uptrend
    LOWER_HIGHS_LOWER_LOWS = "LH_LL"    # Downtrend
    CONSOLIDATION = "CONSOLIDATION"      # Range-bound
    BREAKOUT_PULLBACK = "BREAKOUT_PB"    # Broke resistance, pulling back
    FAILED_BREAKOUT = "FAILED_BO"        # Broke resistance then failed
    ACCUMULATION = "ACCUMULATION"        # Building base

class EntryType(Enum):
    """Entry execution strategy"""
    LIMIT_AT_SUPPORT = "LIMIT_SUPPORT"       # Limit order at support
    SCALE_IN_WEAKNESS = "SCALE_WEAKNESS"     # Multiple entries on pullback
    BREAKOUT_RETEST = "BREAKOUT_RETEST"      # Enter on retest of breakout
    FAILED_BREAKDOWN = "FAILED_BREAKDOWN"    # Enter when breakdown fails
    ACCUMULATION_ZONE = "ACCUMULATION"       # Enter in accumulation range

@dataclass
class SupportResistanceLevel:
    """Price level with strength rating"""
    price: float
    level_type: str  # "support" or "resistance"
    strength: float  # 0-10 (how many times tested)
    last_test_days_ago: int
    volume_at_level: float  # Average volume when price was here
    
    def get_buffer_range(self, atr: float) -> Tuple[float, float]:
        """Get price range around level (for limit orders)"""
        buffer = atr * 0.3  # 30% of ATR as buffer
        if self.level_type == "support":
            return (self.price - buffer, self.price + buffer)
        else:
            return (self.price - buffer, self.price + buffer)

@dataclass
class PrecisionEntry:
    """Detailed entry plan with multiple levels"""
    symbol: str
    entry_strategy: EntryType
    
    # Price structure context
    structure: PriceStructure
    structure_invalidation: float  # If price goes here, structure broken
    
    # Support/Resistance
    key_support: float
    secondary_support: float
    resistance: float
    
    # Layered entry plan
    entry_zone_high: float   # Start scaling in here
    entry_zone_low: float    # Aggressive entry here
    optimal_entry: float     # Best risk/reward entry
    
    # Position sizing (scaled entries)
    entry_1_price: float     # 30% position
    entry_1_size_pct: float
    entry_2_price: float     # 40% position
    entry_2_size_pct: float
    entry_3_price: float     # 30% position
    entry_3_size_pct: float
    
    # Risk management
    stop_loss: float
    trailing_stop_activation: float  # When to start trailing
    
    # Targets (scaled exits)
    target_1: float  # Take 50% profit
    target_2: float  # Take 30% profit
    target_3: float  # Let 20% run
    
    # Invalidation
    max_days_to_trigger: int  # If doesn't trigger in X days, cancel
    
    # Context
    entry_reason: str
    risk_rating: str  # "Low", "Medium", "High"
    confidence: float  # 0-1
    
    # Technical confluence at entry
    ema_support: bool
    volume_profile_support: bool
    fib_level: Optional[float]  # If at Fibonacci level
    prior_resistance_as_support: bool

# ============================================================================
# PRICE STRUCTURE ANALYZER
# ============================================================================

class PriceStructureAnalyzer:
    """
    Analyzes price structure to identify:
    - Swing highs and lows
    - Support and resistance levels
    - Trend structure
    - Key pivot points
    """
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.swing_lookback = 5  # Look 5 bars left and right for swings
    
    def identify_swing_points(self) -> Tuple[pd.Series, pd.Series]:
        """
        Identify swing highs and swing lows
        Returns: (swing_highs, swing_lows) as boolean series
        """
        high = self.data['High']
        low = self.data['Low']
        
        # Swing high: highest point in 2*lookback+1 window
        swing_highs = pd.Series(False, index=self.data.index)
        swing_lows = pd.Series(False, index=self.data.index)
        
        for i in range(self.swing_lookback, len(self.data) - self.swing_lookback):
            # Check if this is a swing high
            window_high = high[i-self.swing_lookback:i+self.swing_lookback+1]
            if high.iloc[i] == window_high.max():
                swing_highs.iloc[i] = True
            
            # Check if this is a swing low
            window_low = low[i-self.swing_lookback:i+self.swing_lookback+1]
            if low.iloc[i] == window_low.min():
                swing_lows.iloc[i] = True
        
        return swing_highs, swing_lows
    
    def identify_structure(self) -> PriceStructure:
        """
        Classify current price structure
        """
        swing_highs, swing_lows = self.identify_swing_points()
        
        # Get recent swing points (last 20 days)
        recent_data = self.data.tail(20)
        recent_highs = recent_data['High'][swing_highs.tail(20)]
        recent_lows = recent_data['Low'][swing_lows.tail(20)]
        
        if len(recent_highs) < 2 or len(recent_lows) < 2:
            return PriceStructure.CONSOLIDATION
        
        # Check for higher highs and higher lows (uptrend)
        highs_increasing = recent_highs.is_monotonic_increasing
        lows_increasing = recent_lows.is_monotonic_increasing
        
        # Check for lower highs and lower lows (downtrend)
        highs_decreasing = recent_highs.is_monotonic_decreasing
        lows_decreasing = recent_lows.is_monotonic_decreasing
        
        latest_price = self.data['Close'].iloc[-1]
        resistance = self.data['High'].tail(20).max()
        
        # Classify structure
        if highs_increasing and lows_increasing:
            return PriceStructure.HIGHER_HIGHS_HIGHER_LOWS
        
        elif highs_decreasing and lows_decreasing:
            return PriceStructure.LOWER_HIGHS_LOWER_LOWS
        
        # Check for breakout pullback (price above resistance, pulling back)
        elif latest_price > resistance * 0.98 and latest_price < resistance * 1.03:
            # Recently broke resistance
            if self.data['Close'].iloc[-5] < resistance * 0.98:
                return PriceStructure.BREAKOUT_PULLBACK
        
        # Check for accumulation (tight range with increasing volume)
        price_range = (self.data['High'].tail(10).max() - self.data['Low'].tail(10).min())
        avg_price = self.data['Close'].tail(10).mean()
        range_pct = (price_range / avg_price) * 100
        
        if range_pct < 5:  # Less than 5% range
            vol_increasing = self.data['Volume'].tail(10).is_monotonic_increasing
            if vol_increasing:
                return PriceStructure.ACCUMULATION
            else:
                return PriceStructure.CONSOLIDATION
        
        return PriceStructure.CONSOLIDATION
    
    def find_support_resistance_levels(self, num_levels: int = 3) -> List[SupportResistanceLevel]:
        """
        Find key support and resistance levels using:
        - Swing points
        - High volume nodes
        - Round numbers
        - EMA levels
        """
        levels = []
        swing_highs, swing_lows = self.identify_swing_points()
        
        # Get swing lows as support
        support_prices = self.data.loc[swing_lows, 'Low'].tail(10)
        support_volumes = self.data.loc[swing_lows, 'Volume'].tail(10)
        
        for price, volume in zip(support_prices, support_volumes):
            # Count how many times this level was tested
            tests = ((self.data['Low'] >= price * 0.98) & 
                    (self.data['Low'] <= price * 1.02)).sum()
            
            # Find last test
            last_test_idx = self.data[
                (self.data['Low'] >= price * 0.98) & 
                (self.data['Low'] <= price * 1.02)
            ].index[-1] if tests > 0 else self.data.index[-1]
            
            days_ago = len(self.data) - self.data.index.get_loc(last_test_idx)
            
            strength = min(tests / 3 * 10, 10)  # Normalize to 0-10
            
            levels.append(SupportResistanceLevel(
                price=float(price),
                level_type="support",
                strength=strength,
                last_test_days_ago=days_ago,
                volume_at_level=float(volume)
            ))
        
        # Get swing highs as resistance
        resistance_prices = self.data.loc[swing_highs, 'High'].tail(10)
        resistance_volumes = self.data.loc[swing_highs, 'Volume'].tail(10)
        
        for price, volume in zip(resistance_prices, resistance_volumes):
            tests = ((self.data['High'] >= price * 0.98) & 
                    (self.data['High'] <= price * 1.02)).sum()
            
            if tests > 0:
                last_test_idx = self.data[
                    (self.data['High'] >= price * 0.98) & 
                    (self.data['High'] <= price * 1.02)
                ].index[-1]
                days_ago = len(self.data) - self.data.index.get_loc(last_test_idx)
            else:
                days_ago = len(self.data)
            
            strength = min(tests / 3 * 10, 10)
            
            levels.append(SupportResistanceLevel(
                price=float(price),
                level_type="resistance",
                strength=strength,
                last_test_days_ago=days_ago,
                volume_at_level=float(volume)
            ))
        
        # Sort by strength and recency
        levels.sort(key=lambda x: (x.strength, -x.last_test_days_ago), reverse=True)
        
        return levels[:num_levels * 2]  # Return top levels

# ============================================================================
# PRECISION ENTRY GENERATOR
# ============================================================================

class PrecisionEntryGenerator:
    """
    Generates precise entry plans with:
    - Multiple entry levels (scale in on weakness)
    - Structure-based stops
    - Layered profit targets
    """
    
    def __init__(self, config):
        self.config = config
    
    def generate_precision_entry(self, 
                                 symbol: str,
                                 data: pd.DataFrame,
                                 base_signal: Dict) -> Optional[PrecisionEntry]:
        """
        Generate precision entry plan from base signal
        
        Args:
            symbol: Stock symbol
            data: Price data with indicators
            base_signal: Basic signal from screener (score, stop, target, etc.)
        
        Returns:
            PrecisionEntry plan or None if no valid entry setup
        """
        try:
            # Analyze price structure
            analyzer = PriceStructureAnalyzer(data)
            structure = analyzer.identify_structure()
            levels = analyzer.find_support_resistance_levels()
            
            if not levels:
                return None
            
            latest = data.iloc[-1]
            current_price = float(latest['Close'])
            atr = float(latest.get('ATR', current_price * 0.03))
            
            # Filter support and resistance levels
            supports = [l for l in levels if l.level_type == "support" and l.price < current_price]
            resistances = [l for l in levels if l.level_type == "resistance" and l.price > current_price]
            
            if not supports:
                # No clear support, use ATR-based support
                key_support = current_price - (atr * 2)
                secondary_support = current_price - (atr * 3)
            else:
                # Use nearest strong support
                key_support = supports[0].price
                secondary_support = supports[1].price if len(supports) > 1 else key_support - atr
            
            if not resistances:
                resistance = current_price + (atr * 4)
            else:
                resistance = resistances[0].price
            
            # Determine entry strategy based on structure
            entry_strategy, entry_plan = self._determine_entry_strategy(
                structure, current_price, key_support, resistance, atr, data
            )
            
            if entry_plan is None:
                return None
            
            # Calculate stop loss (below key support or structure invalidation)
            stop_loss = self._calculate_structure_stop(
                structure, key_support, secondary_support, atr
            )
            
            # Validate risk (stop not too wide)
            risk_pct = (current_price - stop_loss) / current_price
            if risk_pct > self.config.MAX_STOP_LOSS_PCT:
                logger.warning(f"{symbol}: Stop too wide ({risk_pct*100:.1f}%)")
                return None
            
            if risk_pct < self.config.MIN_STOP_LOSS_PCT:
                stop_loss = current_price * (1 - self.config.MIN_STOP_LOSS_PCT)
                risk_pct = self.config.MIN_STOP_LOSS_PCT
            
            # Calculate layered targets
            risk_amount = current_price - stop_loss
            target_1 = current_price + (risk_amount * 1.5)  # 1.5R
            target_2 = current_price + (risk_amount * 2.5)  # 2.5R
            target_3 = current_price + (risk_amount * 4.0)  # 4R (runner)
            
            # Cap targets at resistance levels
            if target_1 > resistance * 0.98:
                target_1 = resistance * 0.97
            if target_2 > resistance * 1.05:
                target_2 = resistance * 1.03
            
            # Structure invalidation level (if price goes here, setup broken)
            invalidation = self._calculate_invalidation_level(structure, secondary_support, atr)
            
            # Check for technical confluence
            ema_20 = float(latest.get('EMA_20', 0))
            ema_support = abs(key_support - ema_20) / ema_20 < 0.02 if ema_20 > 0 else False
            
            # Check if prior resistance is now support
            prior_resistance_support = False
            if resistances and len(resistances) > 0:
                for res_level in resistances:
                    if abs(key_support - res_level.price) / res_level.price < 0.03:
                        prior_resistance_support = True
                        break
            
            # Calculate Fibonacci retracement (if in pullback)
            fib_level = None
            if structure == PriceStructure.BREAKOUT_PULLBACK:
                recent_high = data['High'].tail(10).max()
                recent_low = data['Low'].tail(20).min()
                fib_382 = recent_high - (recent_high - recent_low) * 0.382
                fib_500 = recent_high - (recent_high - recent_low) * 0.500
                fib_618 = recent_high - (recent_high - recent_low) * 0.618
                
                # Check if current price near any fib level
                if abs(current_price - fib_618) / current_price < 0.02:
                    fib_level = 0.618
                elif abs(current_price - fib_500) / current_price < 0.02:
                    fib_level = 0.500
                elif abs(current_price - fib_382) / current_price < 0.02:
                    fib_level = 0.382
            
            # Determine confidence based on confluence
            confluence_score = 0
            if ema_support:
                confluence_score += 1
            if prior_resistance_support:
                confluence_score += 1
            if fib_level is not None:
                confluence_score += 1
            if supports and supports[0].strength >= 7:
                confluence_score += 1
            
            confidence = min(confluence_score / 4, 1.0)
            
            # Risk rating
            if risk_pct < 0.04:
                risk_rating = "Low"
            elif risk_pct < 0.06:
                risk_rating = "Medium"
            else:
                risk_rating = "High"
            
            # Build entry reason
            reasons = [f"{structure.value} structure"]
            if ema_support:
                reasons.append("EMA support")
            if prior_resistance_support:
                reasons.append("S/R flip")
            if fib_level:
                reasons.append(f"Fib {fib_level}")
            
            entry_reason = " + ".join(reasons)
            
            # Trailing stop activation (when up 1R)
            trailing_activation = current_price + risk_amount
            
            return PrecisionEntry(
                symbol=symbol,
                entry_strategy=entry_strategy,
                structure=structure,
                structure_invalidation=invalidation,
                key_support=key_support,
                secondary_support=secondary_support,
                resistance=resistance,
                entry_zone_high=entry_plan['entry_zone_high'],
                entry_zone_low=entry_plan['entry_zone_low'],
                optimal_entry=entry_plan['optimal_entry'],
                entry_1_price=entry_plan['entry_1_price'],
                entry_1_size_pct=entry_plan['entry_1_size_pct'],
                entry_2_price=entry_plan['entry_2_price'],
                entry_2_size_pct=entry_plan['entry_2_size_pct'],
                entry_3_price=entry_plan['entry_3_price'],
                entry_3_size_pct=entry_plan['entry_3_size_pct'],
                stop_loss=stop_loss,
                trailing_stop_activation=trailing_activation,
                target_1=target_1,
                target_2=target_2,
                target_3=target_3,
                max_days_to_trigger=5,
                entry_reason=entry_reason,
                risk_rating=risk_rating,
                confidence=confidence,
                ema_support=ema_support,
                volume_profile_support=supports[0].volume_at_level > data['Volume'].tail(20).mean() if supports else False,
                fib_level=fib_level,
                prior_resistance_as_support=prior_resistance_support
            )
            
        except Exception as e:
            logger.error(f"Error generating precision entry for {symbol}: {str(e)}")
            return None
    
    def _determine_entry_strategy(self, 
                                  structure: PriceStructure,
                                  current_price: float,
                                  support: float,
                                  resistance: float,
                                  atr: float,
                                  data: pd.DataFrame) -> Tuple[EntryType, Optional[Dict]]:
        """
        Determine optimal entry strategy based on price structure
        Returns: (entry_type, entry_plan_dict)
        """
        
        if structure == PriceStructure.HIGHER_HIGHS_HIGHER_LOWS:
            # Uptrend: Wait for pullback to support, scale in
            entry_zone_high = current_price * 0.98  # Start scaling at 2% pullback
            entry_zone_low = support * 1.01  # Aggressive entry just above support
            optimal_entry = (entry_zone_high + entry_zone_low) / 2
            
            # Scale in: 30% at high, 40% at mid, 30% at low
            return EntryType.SCALE_IN_WEAKNESS, {
                'entry_zone_high': entry_zone_high,
                'entry_zone_low': entry_zone_low,
                'optimal_entry': optimal_entry,
                'entry_1_price': entry_zone_high,
                'entry_1_size_pct': 0.30,
                'entry_2_price': optimal_entry,
                'entry_2_size_pct': 0.40,
                'entry_3_price': entry_zone_low,
                'entry_3_size_pct': 0.30
            }
        
        elif structure == PriceStructure.BREAKOUT_PULLBACK:
            # Breakout pullback: Enter on retest of breakout level
            breakout_level = resistance
            entry_zone_high = breakout_level * 1.01
            entry_zone_low = breakout_level * 0.98
            optimal_entry = breakout_level
            
            # Concentrate entry at optimal level
            return EntryType.BREAKOUT_RETEST, {
                'entry_zone_high': entry_zone_high,
                'entry_zone_low': entry_zone_low,
                'optimal_entry': optimal_entry,
                'entry_1_price': breakout_level * 1.005,  # 50% just above
                'entry_1_size_pct': 0.50,
                'entry_2_price': breakout_level,  # 30% at level
                'entry_2_size_pct': 0.30,
                'entry_3_price': breakout_level * 0.995,  # 20% below
                'entry_3_size_pct': 0.20
            }
        
        elif structure == PriceStructure.CONSOLIDATION:
            # Consolidation: Enter at bottom of range
            range_bottom = support
            range_top = resistance
            range_mid = (range_top + range_bottom) / 2
            
            # Only enter if price near bottom third of range
            if current_price > range_mid:
                return EntryType.LIMIT_AT_SUPPORT, None  # Too high, wait
            
            entry_zone_low = range_bottom * 1.005
            entry_zone_high = range_bottom * 1.02
            optimal_entry = (entry_zone_low + entry_zone_high) / 2
            
            return EntryType.LIMIT_AT_SUPPORT, {
                'entry_zone_high': entry_zone_high,
                'entry_zone_low': entry_zone_low,
                'optimal_entry': optimal_entry,
                'entry_1_price': entry_zone_low,
                'entry_1_size_pct': 0.40,
                'entry_2_price': optimal_entry,
                'entry_2_size_pct': 0.40,
                'entry_3_price': entry_zone_high,
                'entry_3_size_pct': 0.20
            }
        
        elif structure == PriceStructure.ACCUMULATION:
            # Accumulation: Enter anywhere in zone, full position
            entry_zone_low = support * 1.01
            entry_zone_high = current_price * 1.02
            optimal_entry = (entry_zone_low + entry_zone_high) / 2
            
            return EntryType.ACCUMULATION_ZONE, {
                'entry_zone_high': entry_zone_high,
                'entry_zone_low': entry_zone_low,
                'optimal_entry': optimal_entry,
                'entry_1_price': optimal_entry,
                'entry_1_size_pct': 1.00,  # Single entry
                'entry_2_price': optimal_entry,
                'entry_2_size_pct': 0.00,
                'entry_3_price': optimal_entry,
                'entry_3_size_pct': 0.00
            }
        
        else:
            # Default: scale in on weakness
            entry_zone_high = current_price * 0.98
            entry_zone_low = support * 1.02
            optimal_entry = (entry_zone_high + entry_zone_low) / 2
            
            return EntryType.SCALE_IN_WEAKNESS, {
                'entry_zone_high': entry_zone_high,
                'entry_zone_low': entry_zone_low,
                'optimal_entry': optimal_entry,
                'entry_1_price': entry_zone_high,
                'entry_1_size_pct': 0.30,
                'entry_2_price': optimal_entry,
                'entry_2_size_pct': 0.40,
                'entry_3_price': entry_zone_low,
                'entry_3_size_pct': 0.30
            }
    
    def _calculate_structure_stop(self,
                                  structure: PriceStructure,
                                  key_support: float,
                                  secondary_support: float,
                                  atr: float) -> float:
        """Calculate stop loss based on structure (not arbitrary percentage)"""
        
        if structure == PriceStructure.HIGHER_HIGHS_HIGHER_LOWS:
            # Stop below key support with buffer
            return key_support * 0.98
        
        elif structure == PriceStructure.BREAKOUT_PULLBACK:
            # Stop below breakout level (now support)
            return key_support * 0.97
        
        elif structure == PriceStructure.CONSOLIDATION:
            # Stop below range bottom
            return key_support * 0.98
        
        elif structure == PriceStructure.ACCUMULATION:
            # Stop below accumulation zone
            return secondary_support * 0.99
        
        else:
            # Default: ATR-based stop below support
            return key_support - (atr * 0.5)
    
    def _calculate_invalidation_level(self,
                                      structure: PriceStructure,
                                      secondary_support: float,
                                      atr: float) -> float:
        """
        Calculate structure invalidation level
        If price goes here, the setup is completely broken (cancel order)
        """
        if structure == PriceStructure.HIGHER_HIGHS_HIGHER_LOWS:
            # If price breaks below secondary support, uptrend broken
            return secondary_support * 0.97
        
        elif structure == PriceStructure.BREAKOUT_PULLBACK:
            # If price goes back below breakout, failed breakout
            return secondary_support * 0.98
        
        else:
            # Default: below secondary support
            return secondary_support * 0.98

# ============================================================================
# ENTRY ORDER GENERATOR
# ============================================================================

@dataclass
class EntryOrder:
    """Actionable limit order"""
    order_id: int
    symbol: str
    order_type: str  # "LIMIT", "STOP_LIMIT"
    price: float
    shares: int
    size_pct: float  # Percentage of total position
    valid_until: str  # Date string
    notes: str

class EntryOrderGenerator:
    """Generate actionable limit orders from precision entry plan"""
    
    @staticmethod
    def generate_orders(precision_entry: PrecisionEntry,
                       total_shares: int,
                       capital: float) -> List[EntryOrder]:
        """
        Convert precision entry plan to executable limit orders
        
        Returns list of limit orders to place
        """
        orders = []
        order_id = 1
        
        from datetime import datetime, timedelta
        valid_until = (datetime.now() + timedelta(days=precision_entry.max_days_to_trigger)).strftime('%Y-%m-%d')
        
        # Entry 1 (highest price, smallest size)
        if precision_entry.entry_1_size_pct > 0:
            shares_1 = int(total_shares * precision_entry.entry_1_size_pct)
            if shares_1 >= 100:  # Minimum 1 lot
                orders.append(EntryOrder(
                    order_id=order_id,
                    symbol=precision_entry.symbol,
                    order_type="LIMIT",
                    price=precision_entry.entry_1_price,
                    shares=shares_1,
                    size_pct=precision_entry.entry_1_size_pct,
                    valid_until=valid_until,
                    notes=f"Entry 1: {precision_entry.entry_1_size_pct*100:.0f}% at {precision_entry.entry_1_price:.0f}"
                ))
                order_id += 1
        
        # Entry 2 (middle price, larger size)
        if precision_entry.entry_2_size_pct > 0:
            shares_2 = int(total_shares * precision_entry.entry_2_size_pct)
            if shares_2 >= 100:
                orders.append(EntryOrder(
                    order_id=order_id,
                    symbol=precision_entry.symbol,
                    order_type="LIMIT",
                    price=precision_entry.entry_2_price,
                    shares=shares_2,
                    size_pct=precision_entry.entry_2_size_pct,
                    valid_until=valid_until,
                    notes=f"Entry 2: {precision_entry.entry_2_size_pct*100:.0f}% at {precision_entry.entry_2_price:.0f}"
                ))
                order_id += 1
        
        # Entry 3 (lowest price, smaller size)
        if precision_entry.entry_3_size_pct > 0:
            shares_3 = int(total_shares * precision_entry.entry_3_size_pct)
            if shares_3 >= 100:
                orders.append(EntryOrder(
                    order_id=order_id,
                    symbol=precision_entry.symbol,
                    order_type="LIMIT",
                    price=precision_entry.entry_3_price,
                    shares=shares_3,
                    size_pct=precision_entry.entry_3_size_pct,
                    valid_until=valid_until,
                    notes=f"Entry 3: {precision_entry.entry_3_size_pct*100:.0f}% at {precision_entry.entry_3_price:.0f}"
                ))