#!/usr/bin/env python3
"""
IDX Daily Swing Screener - Core Engine
Reliable 3-10 Day Swing Trading System for Indonesian Stock Market

Design Philosophy:
- Use ONLY end-of-day data (no look-ahead bias)
- Conservative assumptions (real slippage, real costs)
- Regime-aware (don't fight the tape)
- Risk-first approach (survive first, profit second)

Author: Rebuilt for Production Reliability
Version: 2.0.0
"""

import yfinance as yf
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
import warnings
import time

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

class TradingConfig:
    """Centralized configuration - ADJUST THESE FOR YOUR RISK TOLERANCE"""
    
    # Capital & Risk Management
    TOTAL_CAPITAL = 100_000_000  # IDR (100 million = ~$6,500 USD)
    MAX_POSITION_SIZE = 0.15  # 15% max per position
    MAX_PORTFOLIO_HEAT = 0.06  # 6% total risk across all positions
    MAX_POSITIONS = 5  # Maximum concurrent positions
    
    # IDX Market Specifics
    MIN_PRICE_IDR = 100  # Avoid penny stocks
    MAX_PRICE_IDR = 50_000  # Avoid very expensive stocks (liquidity)
    MIN_AVG_VOLUME = 500_000  # Minimum 500K shares/day
    MIN_AVG_VALUE_IDR = 500_000_000  # 500M IDR daily value (~$32K)
    
    # Transaction Costs (REALISTIC for IDX)
    COMMISSION_BPS = 25  # 0.25% (0.15% buy + 0.10% sell typical)
    SLIPPAGE_BPS = 15  # 0.15% realistic slippage
    VAT_RATE = 0.11  # 11% VAT on commission
    TOTAL_COST_BPS = COMMISSION_BPS + SLIPPAGE_BPS  # ~0.40% round-trip
    
    # Swing Trading Parameters
    HOLDING_PERIOD_MIN = 3  # days
    HOLDING_PERIOD_MAX = 10  # days
    LOOKBACK_PERIOD = 60  # days for technical analysis
    
    # Risk/Reward Requirements
    MIN_RISK_REWARD = 2.0  # Must have 2:1 reward:risk minimum
    MAX_STOP_LOSS_PCT = 0.08  # 8% maximum stop loss
    MIN_STOP_LOSS_PCT = 0.03  # 3% minimum (too tight = noise)
    
    # Technical Filters
    MIN_ADX = 20  # Minimum trend strength
    RSI_OVERSOLD = 35  # Below this = oversold
    RSI_OVERBOUGHT = 70  # Above this = overbought
    RSI_NEUTRAL_LOW = 40
    RSI_NEUTRAL_HIGH = 60

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class SwingSignal:
    """Clean signal output structure"""
    symbol: str
    signal_date: str
    
    # Pricing
    current_price: float
    entry_price: float  # Recommended entry (slight premium for execution)
    stop_loss: float
    target_1: float  # Conservative target (2:1)
    target_2: float  # Aggressive target (3:1)
    
    # Risk Metrics
    risk_pct: float
    reward_pct_t1: float
    reward_pct_t2: float
    risk_reward_t1: float
    risk_reward_t2: float
    
    # Position Sizing
    position_size_idr: float
    position_size_shares: int
    max_loss_idr: float
    
    # Quality Scores
    technical_score: float  # 0-10
    trend_strength: float  # ADX value
    volume_score: float  # 0-10
    setup_quality: str  # "A", "B", "C"
    
    # Context
    market_regime: str
    confidence: str  # "High", "Medium", "Low"
    holding_period: str  # "3-5 days" or "5-10 days"
    
    # Reasoning
    entry_reason: str
    warnings: List[str]

class MarketRegime(NamedTuple):
    """Market regime classification"""
    regime: str  # "Bull", "Bear", "Neutral", "Volatile"
    confidence: float  # 0-1
    trend_direction: int  # 1, 0, -1
    volatility_level: str  # "Low", "Normal", "High"
    tradeable: bool

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(f'swing_screener_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CORE SCREENER ENGINE
# ============================================================================

class IDXSwingScreener:
    """
    Production-grade daily swing screener for Indonesian stocks
    Uses ONLY end-of-day data with proper lookahead bias prevention
    """
    
    def __init__(self, config: TradingConfig = None):
        self.config = config or TradingConfig()
        self.market_regime: Optional[MarketRegime] = None
        logger.info("IDX Swing Screener initialized")
        logger.info(f"Capital: {self.config.TOTAL_CAPITAL:,.0f} IDR")
        logger.info(f"Max positions: {self.config.MAX_POSITIONS}")
    
    # ========================================================================
    # DATA ACQUISITION (with retry logic)
    # ========================================================================
    
    def fetch_data(self, symbol: str, period_days: int = 90) -> Optional[pd.DataFrame]:
        """
        Fetch and validate stock data
        Returns None if data is insufficient or invalid
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=f"{period_days}d", interval="1d")
                
                if data is None or len(data) < self.config.LOOKBACK_PERIOD:
                    if attempt < max_retries - 1:
                        time.sleep(1)
                        continue
                    logger.warning(f"{symbol}: Insufficient data ({len(data) if data is not None else 0} days)")
                    return None
                
                # Validate required columns
                required = ['Open', 'High', 'Low', 'Close', 'Volume']
                if not all(col in data.columns for col in required):
                    logger.warning(f"{symbol}: Missing required columns")
                    return None
                
                # Check for data quality
                if data['Close'].isna().sum() > 5:  # Allow max 5 missing days
                    logger.warning(f"{symbol}: Too many missing values")
                    return None
                
                # Fill forward then backward for small gaps
                data = data.ffill().bfill()
                
                return data
                
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                logger.error(f"{symbol}: Error fetching data - {str(e)}")
                return None
        
        return None
    
    # ========================================================================
    # TECHNICAL INDICATORS (NO LOOK-AHEAD BIAS)
    # ========================================================================
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators with STRICT no look-ahead bias
        All indicators use only PAST data (proper shifting)
        """
        df = data.copy()
        
        # === MOVING AVERAGES ===
        # These represent "what we knew at close of previous day"
        df['SMA_10'] = df['Close'].rolling(10).mean().shift(1)
        df['SMA_20'] = df['Close'].rolling(20).mean().shift(1)
        df['SMA_50'] = df['Close'].rolling(50).mean().shift(1)
        df['SMA_200'] = df['Close'].rolling(200).mean().shift(1)
        
        df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean().shift(1)
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean().shift(1)
        
        # === MOMENTUM (RSI) ===
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['RSI'] = (100 - (100 / (1 + rs))).shift(1)  # SHIFTED!
        
        # === TREND STRENGTH (ADX) ===
        # Calculate +DM and -DM
        high_diff = df['High'].diff()
        low_diff = -df['Low'].diff()
        
        pos_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        neg_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        # True Range
        tr1 = df['High'] - df['Low']
        tr2 = abs(df['High'] - df['Close'].shift(1))
        tr3 = abs(df['Low'] - df['Close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.rolling(14).mean()
        
        pos_di = 100 * (pos_dm.rolling(14).mean() / atr)
        neg_di = 100 * (neg_dm.rolling(14).mean() / atr)
        
        dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di).replace(0, np.nan)
        df['ADX'] = dx.rolling(14).mean().shift(1)  # SHIFTED!
        
        # === VOLATILITY ===
        df['ATR'] = atr.shift(1)  # SHIFTED!
        df['ATR_Pct'] = (df['ATR'] / df['Close']) * 100
        
        # Bollinger Bands
        bb_mid = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['BB_Upper'] = (bb_mid + 2 * bb_std).shift(1)
        df['BB_Lower'] = (bb_mid - 2 * bb_std).shift(1)
        df['BB_Width'] = ((df['BB_Upper'] - df['BB_Lower']) / bb_mid).shift(1)
        
        # === VOLUME ===
        df['Volume_SMA_20'] = df['Volume'].rolling(20).mean().shift(1)
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
        
        # Average dollar volume (for liquidity check)
        df['Avg_Value'] = (df['Close'] * df['Volume']).rolling(20).mean().shift(1)
        
        # === PRICE ACTION ===
        # Support/Resistance (20-day)
        df['Resistance'] = df['High'].rolling(20).max().shift(1)
        df['Support'] = df['Low'].rolling(20).min().shift(1)
        
        # Recent momentum
        df['Return_5d'] = (df['Close'] / df['Close'].shift(5) - 1).shift(1)
        df['Return_10d'] = (df['Close'] / df['Close'].shift(10) - 1).shift(1)
        
        return df
    
    # ========================================================================
    # MARKET REGIME DETECTION
    # ========================================================================
    
    def detect_market_regime(self, index_symbol: str = '^JKSE') -> MarketRegime:
        """
        Detect current market regime using JCI (Jakarta Composite Index)
        Critical for knowing if we should trade at all
        """
        try:
            data = self.fetch_data(index_symbol, period_days=120)
            if data is None or len(data) < 60:
                logger.warning("Cannot assess market regime - using neutral")
                return MarketRegime("Neutral", 0.5, 0, "Normal", True)
            
            data = self.calculate_indicators(data)
            latest = data.iloc[-1]
            
            # Trend assessment
            price = latest['Close']
            sma_50 = latest['SMA_50']
            sma_200 = latest['SMA_200']
            
            # Trend direction
            if pd.notna(sma_50) and pd.notna(sma_200):
                if price > sma_50 > sma_200:
                    trend = 1  # Uptrend
                elif price < sma_50 < sma_200:
                    trend = -1  # Downtrend
                else:
                    trend = 0  # Mixed
            else:
                trend = 0
            
            # Volatility assessment
            atr_pct = latest['ATR_Pct']
            if pd.isna(atr_pct):
                vol_level = "Normal"
            elif atr_pct > 3.0:
                vol_level = "High"
            elif atr_pct < 1.5:
                vol_level = "Low"
            else:
                vol_level = "Normal"
            
            # Regime classification
            adx = latest['ADX']
            if pd.isna(adx):
                adx = 20  # Neutral default
            
            if trend == 1 and adx > 25:
                regime = "Bull"
                confidence = min(adx / 50, 1.0)
                tradeable = True
            elif trend == -1 and adx > 25:
                regime = "Bear"
                confidence = min(adx / 50, 1.0)
                tradeable = False  # Don't trade in bear market
            elif vol_level == "High":
                regime = "Volatile"
                confidence = 0.7
                tradeable = False  # Too risky
            else:
                regime = "Neutral"
                confidence = 0.6
                tradeable = True
            
            result = MarketRegime(regime, confidence, trend, vol_level, tradeable)
            logger.info(f"Market Regime: {regime} (confidence: {confidence:.1%}, tradeable: {tradeable})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error detecting market regime: {str(e)}")
            return MarketRegime("Unknown", 0.5, 0, "Normal", True)
    
    # ========================================================================
    # SIGNAL GENERATION (THE CORE LOGIC)
    # ========================================================================
    
    def generate_signal(self, symbol: str, data: pd.DataFrame, use_precision_entry: bool = True) -> Optional[SwingSignal]:
        """
        Generate swing trade signal if setup is valid
        
        Args:
            symbol: Stock symbol
            data: Price data with indicators
            use_precision_entry: If True, generate precision entry plan
        
        Returns None if no valid setup exists
        """
        try:
            if len(data) < self.config.LOOKBACK_PERIOD:
                return None
            
            latest = data.iloc[-1]
            prev = data.iloc[-2]
            
            # === BASIC FILTERS ===
            price = float(latest['Close'])
            volume = float(latest['Volume'])
            avg_volume = float(latest['Volume_SMA_20'])
            avg_value = float(latest['Avg_Value'])
            
            # Price range filter
            if price < self.config.MIN_PRICE_IDR or price > self.config.MAX_PRICE_IDR:
                return None
            
            # Liquidity filters
            if volume < self.config.MIN_AVG_VOLUME:
                return None
            
            if pd.notna(avg_value) and avg_value < self.config.MIN_AVG_VALUE_IDR:
                return None
            
            # === TECHNICAL SETUP SCORING ===
            score = 0.0
            warnings = []
            
            # 1. Trend alignment (30% weight)
            sma_10 = latest['SMA_10']
            sma_20 = latest['SMA_20']
            sma_50 = latest['SMA_50']
            
            if pd.notna(sma_10) and pd.notna(sma_20) and pd.notna(sma_50):
                if price > sma_10 > sma_20 > sma_50:
                    score += 3.0  # Perfect alignment
                elif price > sma_20 > sma_50:
                    score += 2.0  # Good alignment
                elif price > sma_50:
                    score += 1.0  # Weak bullish
                else:
                    return None  # Bearish - no trade
            else:
                warnings.append("Incomplete MA data")
                score += 1.0  # Benefit of doubt
            
            # 2. Momentum (RSI) - looking for pullbacks in uptrends (25% weight)
            rsi = latest['RSI']
            if pd.notna(rsi):
                if self.config.RSI_NEUTRAL_LOW <= rsi <= self.config.RSI_NEUTRAL_HIGH:
                    score += 2.5  # Ideal range
                elif self.config.RSI_OVERSOLD <= rsi < self.config.RSI_NEUTRAL_LOW:
                    score += 2.0  # Oversold bounce
                elif rsi > self.config.RSI_OVERBOUGHT:
                    warnings.append(f"RSI overbought ({rsi:.1f})")
                    score += 0.5  # Risky
                else:
                    score += 1.0  # Acceptable
            else:
                warnings.append("Missing RSI")
                return None
            
            # 3. Trend strength (ADX) (20% weight)
            adx = latest['ADX']
            if pd.notna(adx):
                if adx >= 30:
                    score += 2.0  # Strong trend
                elif adx >= self.config.MIN_ADX:
                    score += 1.5  # Acceptable trend
                else:
                    return None  # Too weak, skip
            else:
                warnings.append("Missing ADX")
                return None
            
            # 4. Volume confirmation (15% weight)
            vol_ratio = latest['Volume_Ratio']
            if pd.notna(vol_ratio):
                if vol_ratio >= 1.5:
                    score += 1.5  # Strong volume
                elif vol_ratio >= 1.0:
                    score += 1.0  # Normal volume
                else:
                    score += 0.3  # Weak volume
                    warnings.append(f"Low volume ({vol_ratio:.1f}x)")
            
            # 5. Price action near support (10% weight)
            support = latest['Support']
            resistance = latest['Resistance']
            
            if pd.notna(support) and pd.notna(resistance):
                range_position = (price - support) / (resistance - support)
                if 0.2 <= range_position <= 0.5:
                    score += 1.0  # Good entry zone
                elif 0.5 < range_position <= 0.7:
                    score += 0.7  # Acceptable
                else:
                    score += 0.3  # Suboptimal
            
            # Normalize score to 0-10
            technical_score = min(score, 10.0)
            
            # Minimum score required
            if technical_score < 5.0:
                return None  # Not good enough
            
            # === RISK/REWARD CALCULATION ===
            # Use ATR-based stops
            atr = latest['ATR']
            if pd.isna(atr) or atr <= 0:
                return None
            
            # Stop loss: 1.5-2.0 ATR below current price (or support, whichever is higher)
            atr_stop = price - (1.8 * atr)
            support_stop = support * 0.98 if pd.notna(support) else atr_stop
            
            stop_loss = max(atr_stop, support_stop)
            stop_loss = max(stop_loss, price * (1 - self.config.MAX_STOP_LOSS_PCT))
            
            risk_pct = (price - stop_loss) / price
            
            # Check risk bounds
            if risk_pct > self.config.MAX_STOP_LOSS_PCT:
                return None  # Risk too high
            if risk_pct < self.config.MIN_STOP_LOSS_PCT:
                stop_loss = price * (1 - self.config.MIN_STOP_LOSS_PCT)  # Widen stop
                risk_pct = self.config.MIN_STOP_LOSS_PCT
            
            # Targets: conservative (2:1) and aggressive (3:1)
            risk_amount = price - stop_loss
            target_1 = price + (risk_amount * 2.0)  # 2R
            target_2 = price + (risk_amount * 3.0)  # 3R
            
            # Check if targets are realistic (below resistance + margin)
            if pd.notna(resistance):
                if target_1 > resistance * 1.05:
                    target_1 = resistance * 1.03  # Adjust to just above resistance
                if target_2 > resistance * 1.10:
                    target_2 = resistance * 1.07
            
            reward_pct_t1 = (target_1 - price) / price
            reward_pct_t2 = (target_2 - price) / price
            
            rr_t1 = reward_pct_t1 / risk_pct if risk_pct > 0 else 0
            rr_t2 = reward_pct_t2 / risk_pct if risk_pct > 0 else 0
            
            # Must meet minimum R:R
            if rr_t1 < self.config.MIN_RISK_REWARD:
                return None  # Reward not worth the risk
            
            # === POSITION SIZING (risk-based) ===
            risk_amount_idr = risk_pct * price
            max_risk_idr = self.config.TOTAL_CAPITAL * self.config.MAX_PORTFOLIO_HEAT / self.config.MAX_POSITIONS
            
            position_size_shares = int(max_risk_idr / risk_amount_idr)
            position_size_idr = position_size_shares * price
            
            # Check position size constraints
            max_position_idr = self.config.TOTAL_CAPITAL * self.config.MAX_POSITION_SIZE
            if position_size_idr > max_position_idr:
                position_size_shares = int(max_position_idr / price)
                position_size_idr = position_size_shares * price
            
            # Must be able to buy at least 100 shares (1 lot in IDX)
            if position_size_shares < 100:
                return None
            
            max_loss_idr = position_size_shares * risk_amount_idr
            
            # === SETUP QUALITY CLASSIFICATION ===
            if technical_score >= 8.0 and rr_t1 >= 2.5 and adx >= 30:
                setup_quality = "A"
                confidence = "High"
                holding_period = "3-5 days"
            elif technical_score >= 6.5 and rr_t1 >= 2.0:
                setup_quality = "B"
                confidence = "Medium"
                holding_period = "5-8 days"
            else:
                setup_quality = "C"
                confidence = "Low"
                holding_period = "7-10 days"
            
            # === ENTRY REASONING ===
            reasons = []
            if price > sma_20 and pd.notna(sma_20):
                reasons.append("Above 20 SMA")
            if self.config.RSI_OVERSOLD <= rsi <= self.config.RSI_NEUTRAL_HIGH:
                reasons.append(f"RSI pullback ({rsi:.0f})")
            if vol_ratio >= 1.2:
                reasons.append("Volume surge")
            if adx >= 30:
                reasons.append(f"Strong trend (ADX {adx:.0f})")
            
            entry_reason = " | ".join(reasons[:3])  # Top 3 reasons
            
            # === MARKET REGIME CONTEXT ===
            regime_str = self.market_regime.regime if self.market_regime else "Unknown"
            
            # Entry price (add small premium for realistic execution)
            entry_price = price * 1.001  # 0.1% above close (realistic)
            
            return SwingSignal(
                symbol=symbol,
                signal_date=latest.name.strftime('%Y-%m-%d') if hasattr(latest.name, 'strftime') else str(latest.name),
                current_price=price,
                entry_price=entry_price,
                stop_loss=stop_loss,
                target_1=target_1,
                target_2=target_2,
                risk_pct=risk_pct,
                reward_pct_t1=reward_pct_t1,
                reward_pct_t2=reward_pct_t2,
                risk_reward_t1=rr_t1,
                risk_reward_t2=rr_t2,
                position_size_idr=position_size_idr,
                position_size_shares=position_size_shares,
                max_loss_idr=max_loss_idr,
                technical_score=technical_score,
                trend_strength=adx,
                volume_score=min(vol_ratio * 5, 10),  # Scale to 0-10
                setup_quality=setup_quality,
                market_regime=regime_str,
                confidence=confidence,
                holding_period=holding_period,
                entry_reason=entry_reason,
                warnings=warnings
            )
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {str(e)}")
            return None
    
    # ========================================================================
    # MAIN SCREENING FUNCTION
    # ========================================================================
    
    def screen(self, symbols: List[str]) -> List[SwingSignal]:
        """
        Screen a list of symbols and return valid signals
        Sorted by setup quality and technical score
        """
        logger.info(f"Starting screen of {len(symbols)} symbols...")
        
        # Check market regime first
        self.market_regime = self.detect_market_regime()
        
        if not self.market_regime.tradeable:
            logger.warning(f"Market regime is {self.market_regime.regime} - trading not recommended")
            # Don't return empty - still show what would trigger in better conditions
        
        signals = []
        failed = 0
        
        for i, symbol in enumerate(symbols, 1):
            if i % 10 == 0:
                logger.info(f"Progress: {i}/{len(symbols)}")
            
            try:
                data = self.fetch_data(symbol)
                if data is None:
                    failed += 1
                    continue
                
                data = self.calculate_indicators(data)
                signal = self.generate_signal(symbol, data)
                
                if signal is not None:
                    signals.append(signal)
                    logger.info(f"✓ {symbol}: Score {signal.technical_score:.1f}, "
                              f"R:R {signal.risk_reward_t1:.1f}:1, Grade {signal.setup_quality}")
                
            except Exception as e:
                logger.error(f"Error screening {symbol}: {str(e)}")
                failed += 1
        
        # Sort by quality, then score
        quality_order = {'A': 0, 'B': 1, 'C': 2}
        signals.sort(key=lambda x: (quality_order.get(x.setup_quality, 3), -x.technical_score))
        
        logger.info(f"\nScreen complete: {len(signals)} signals from {len(symbols)} symbols")
        logger.info(f"Failed to fetch: {failed} symbols")
        
        return signals

# ============================================================================
# END OF CORE ENGINE
# ============================================================================
