#!/usr/bin/env python3
"""
IDX Swing Screener - Enhanced with Precision Entry
Integrates structure-based precision entries with original screener

Usage:
    python idx_swing_enhanced.py --mode screen --precision
    python idx_swing_enhanced.py --mode screen --precision --detail 3

Author: Enhanced Main Application
Version: 2.1.0
"""

import argparse
import sys
from datetime import datetime
from typing import List
import logging

# Import core modules
from idx_swing_core import IDXSwingScreener, TradingConfig, SwingSignal
from idx_swing_portfolio import Portfolio, PerformanceReporter
from idx_swing_precision import (
    PrecisionEntryGenerator,
    PrecisionEntry,
    PrecisionEntryFormatter,
    EntryOrderGenerator,
    PriceStructure
)

logger = logging.getLogger(__name__)

# ============================================================================
# ENHANCED SCREENING WITH PRECISION ENTRY
# ============================================================================

class EnhancedScreener:
    """Wrapper that adds precision entry to base screener"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.base_screener = IDXSwingScreener(config)
        self.precision_generator = PrecisionEntryGenerator(config)
    
    def screen_with_precision(self, symbols: List[str]) -> List[tuple]:
        """
        Screen stocks and generate precision entry plans
        Returns: List of (base_signal, precision_entry) tuples
        """
        logger.info("Starting enhanced screening with precision entries...")
        
        # First, get market regime
        self.base_screener.market_regime = self.base_screener.detect_market_regime()
        
        if not self.base_screener.market_regime.tradeable:
            logger.warning(f"Market regime is {self.base_screener.market_regime.regime} - not recommended")
        
        results = []
        
        for symbol in symbols:
            try:
                # Fetch and calculate indicators
                data = self.base_screener.fetch_data(symbol)
                if data is None:
                    continue
                
                data = self.base_screener.calculate_indicators(data)
                
                # Generate base signal
                base_signal = self.base_screener.generate_signal(symbol, data)
                if base_signal is None:
                    continue
                
                # Generate precision entry plan
                precision_entry = self.precision_generator.generate_precision_entry(
                    symbol=symbol,
                    data=data,
                    base_signal={
                        'score': base_signal.technical_score,
                        'stop': base_signal.stop_loss,
                        'target': base_signal.target_1
                    }
                )
                
                if precision_entry is not None:
                    results.append((base_signal, precision_entry))
                    logger.info(f"✓ {symbol}: Score {base_signal.technical_score:.1f}, "
                              f"Structure: {precision_entry.structure.value}, "
                              f"Entry: {precision_entry.optimal_entry:.0f}")
                
            except Exception as e:
                logger.error(f"Error screening {symbol}: {str(e)}")
        
        # Sort by base signal score
        results.sort(key=lambda x: x[0].technical_score, reverse=True)
        
        logger.info(f"Enhanced screening complete: {len(results)} precision setups found")
        return results

# ============================================================================
# ENHANCED REPORTING
# ============================================================================

def print_precision_results(results: List[tuple], market_regime: str, detail_count: int = 3):
    """
    Print results with precision entry plans
    
    Args:
        results: List of (base_signal, precision_entry) tuples
        market_regime: Current market regime
        detail_count: Number of detailed plans to show (default 3)
    """
    if not results:
        print(f"\n{'='*90}")
        print(f"NO PRECISION SETUPS FOUND | Market Regime: {market_regime}")
        print(f"{'='*90}")
        return
    
    print(f"\n{'='*90}")
    print(f"PRECISION ENTRY SIGNALS | Market Regime: {market_regime} | Setups: {len(results)}")
    print(f"{'='*90}")
    
    # Summary table
    print(f"\n{'Rank':<4} {'Symbol':<10} {'Structure':<18} {'Strategy':<18} {'Entry Zone':<20} {'Risk%':<7} {'Conf'}")
    print(f"{'-'*90}")
    
    for i, (base_signal, precision_entry) in enumerate(results[:15], 1):
        symbol_clean = base_signal.symbol.replace('.JK', '')
        entry_zone = f"{precision_entry.entry_zone_low:.0f}-{precision_entry.entry_zone_high:.0f}"
        risk_pct = ((precision_entry.optimal_entry - precision_entry.stop_loss) / precision_entry.optimal_entry) * 100
        
        print(f"{i:<4} {symbol_clean:<10} {precision_entry.structure.value:<18} "
              f"{precision_entry.entry_strategy.value[:16]:<18} {entry_zone:<20} "
              f"{risk_pct:<7.1f} {precision_entry.confidence*100:.0f}%")
    
    # Detailed execution plans for top N
    print(f"\n{'='*90}")
    print(f"DETAILED EXECUTION PLANS (TOP {detail_count})")
    print(f"{'='*90}")
    
    for i, (base_signal, precision_entry) in enumerate(results[:detail_count], 1):
        # Calculate total shares from base signal
        total_shares = base_signal.position_size_shares
        capital = base_signal.position_size_idr
        
        # Format precision entry plan
        plan = PrecisionEntryFormatter.format_entry_plan(
            precision_entry, total_shares, capital
        )
        print(plan)
        
        # Generate orders
        orders = EntryOrderGenerator.generate_orders(
            precision_entry, total_shares, capital
        )
        
        # Print order list
        order_list = PrecisionEntryFormatter.format_order_list(orders)
        print(order_list)
    
    # Quick reference card for remaining signals
    if len(results) > detail_count:
        print(f"\n{'='*90}")
        print(f"QUICK REFERENCE: Remaining {len(results) - detail_count} Setups")
        print(f"{'='*90}\n")
        
        for i, (base_signal, precision_entry) in enumerate(results[detail_count:], detail_count + 1):
            symbol_clean = base_signal.symbol.replace('.JK', '')
            print(f"{i}. {symbol_clean}: {precision_entry.structure.value} | "
                  f"Entry {precision_entry.entry_zone_low:.0f}-{precision_entry.entry_zone_high:.0f} | "
                  f"Stop {precision_entry.stop_loss:.0f} | "
                  f"Target {precision_entry.target_1:.0f}")
    
    # Trading tips based on market structure
    print(f"\n{'='*90}")
    print("💡 TRADING TIPS FOR TODAY")
    print(f"{'='*90}")
    
    # Analyze dominant structures
    structures = [pe.structure for _, pe in results]
    structure_counts = {}
    for s in structures:
        structure_counts[s] = structure_counts.get(s, 0) + 1
    
    dominant_structure = max(structure_counts, key=structure_counts.get) if structures else None
    
    if dominant_structure == PriceStructure.HIGHER_HIGHS_HIGHER_LOWS:
        print("\n📈 Market Character: Strong uptrends dominating")
        print("   → Strategy: Wait for pullbacks to support, don't chase")
        print("   → Use layered entries (scale in on weakness)")
        print("   → Trail stops aggressively once up 1R")
    
    elif dominant_structure == PriceStructure.BREAKOUT_PULLBACK:
        print("\n🚀 Market Character: Breakout-pullback setups")
        print("   → Strategy: Enter on retest of breakout level")
        print("   → Concentrate position at breakout level")
        print("   → Stop below breakout (tight risk)")
    
    elif dominant_structure == PriceStructure.CONSOLIDATION:
        print("\n📊 Market Character: Range-bound action")
        print("   → Strategy: Buy near support, sell near resistance")
        print("   → Use limit orders at range bottom")
        print("   → Take profits quickly at mid-range or top")
    
    elif dominant_structure == PriceStructure.ACCUMULATION:
        print("\n💰 Market Character: Accumulation zones forming")
        print("   → Strategy: Enter anywhere in zone, hold longer")
        print("   → Full position (no scaling needed)")
        print("   → Be patient - breakouts take time")
    
    # Risk management reminder
    print(f"\n⚠️  RISK MANAGEMENT REMINDERS:")
    print(f"   1. Place ALL limit orders at once (scale-in strategy)")
    print(f"   2. Set stop loss IMMEDIATELY when first entry fills")
    print(f"   3. Cancel pending orders if structure breaks (invalidation level)")
    print(f"   4. Don't move stops lower - structure-based stops are FINAL")
    print(f"   5. Scale out at targets - don't get greedy")

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def load_tickers(filename: str = "idx_tickers.csv") -> List[str]:
    """Load tickers from CSV"""
    try:
        with open(filename, 'r') as f:
            tickers = []
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    ticker = line.upper()
                    if not ticker.endswith('.JK'):
                        ticker += '.JK'
                    tickers.append(ticker)
            return list(dict.fromkeys(tickers))  # Remove duplicates
    except FileNotFoundError:
        logger.warning(f"Ticker file {filename} not found, using defaults")
        return get_default_tickers()

def get_default_tickers() -> List[str]:
    """Default ticker list"""
    return [
        'BBCA.JK', 'BBRI.JK', 'BMRI.JK', 'BBNI.JK',
        'ASII.JK', 'UNTR.JK', 'TLKM.JK', 'UNVR.JK',
        'INDF.JK', 'PGAS.JK', 'PTBA.JK', 'ADRO.JK',
        'ICBP.JK', 'KLBF.JK', 'MYOR.JK'
    ]

def export_precision_signals(results: List[tuple], filename: str = None):
    """Export precision signals to CSV"""
    if not filename:
        filename = f"precision_signals_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    
    import pandas as pd
    
    data = []
    for base_signal, precision_entry in results:
        # Generate orders
        orders = EntryOrderGenerator.generate_orders(
            precision_entry,
            base_signal.position_size_shares,
            base_signal.position_size_idr
        )
        
        # Flatten orders into CSV row
        order_prices = [o.price for o in orders]
        order_shares = [o.shares for o in orders]
        order_pcts = [o.size_pct for o in orders]
        
        data.append({
            'Symbol': base_signal.symbol.replace('.JK', ''),
            'Date': datetime.now().strftime('%Y-%m-%d'),
            'Structure': precision_entry.structure.value,
            'Strategy': precision_entry.entry_strategy.value,
            'Score': round(base_signal.technical_score, 2),
            'Confidence': round(precision_entry.confidence * 100, 0),
            
            # Entry zone
            'Entry_Zone_Low': precision_entry.entry_zone_low,
            'Entry_Zone_High': precision_entry.entry_zone_high,
            'Optimal_Entry': precision_entry.optimal_entry,
            
            # Layered entries
            'Entry_1_Price': order_prices[0] if len(order_prices) > 0 else None,
            'Entry_1_Shares': order_shares[0] if len(order_shares) > 0 else None,
            'Entry_1_Pct': f"{order_pcts[0]*100:.0f}%" if len(order_pcts) > 0 else None,
            
            'Entry_2_Price': order_prices[1] if len(order_prices) > 1 else None,
            'Entry_2_Shares': order_shares[1] if len(order_shares) > 1 else None,
            'Entry_2_Pct': f"{order_pcts[1]*100:.0f}%" if len(order_pcts) > 1 else None,
            
            'Entry_3_Price': order_prices[2] if len(order_prices) > 2 else None,
            'Entry_3_Shares': order_shares[2] if len(order_shares) > 2 else None,
            'Entry_3_Pct': f"{order_pcts[2]*100:.0f}%" if len(order_pcts) > 2 else None,
            
            # Risk management
            'Stop_Loss': precision_entry.stop_loss,
            'Structure_Invalidation': precision_entry.structure_invalidation,
            
            # Targets
            'Target_1': precision_entry.target_1,
            'Target_2': precision_entry.target_2,
            'Target_3': precision_entry.target_3,
            
            # Key levels
            'Key_Support': precision_entry.key_support,
            'Secondary_Support': precision_entry.secondary_support,
            'Resistance': precision_entry.resistance,
            
            # Position sizing
            'Total_Shares': base_signal.position_size_shares,
            'Position_Value_IDR': base_signal.position_size_idr,
            'Max_Loss_IDR': base_signal.max_loss_idr,
            
            # Context
            'Entry_Reason': precision_entry.entry_reason,
            'Risk_Rating': precision_entry.risk_rating,
            'Valid_Until': (datetime.now()).strftime('%Y-%m-%d'),
            
            # Confluence
            'EMA_Support': precision_entry.ema_support,
            'SR_Flip': precision_entry.prior_resistance_as_support,
            'Fib_Level': precision_entry.fib_level if precision_entry.fib_level else '',
        })
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    
    logger.info(f"Precision signals exported to {filename}")
    return filename

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='IDX Swing Screener - Enhanced with Precision Entry',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--mode', choices=['screen'], default='screen',
                       help='Operating mode (precision entry only in screen mode)')
    
    parser.add_argument('--precision', action='store_true',
                       help='Use precision entry (structure-based scaling)')
    
    parser.add_argument('--detail', type=int, default=3,
                       help='Number of detailed execution plans to show (default: 3)')
    
    parser.add_argument('--tickers', type=str, default='idx_tickers.csv',
                       help='Ticker list CSV file')
    
    parser.add_argument('--min-score', type=float, default=5.5,
                       help='Minimum technical score (default: 5.5)')
    
    args = parser.parse_args()
    
    # Banner
    print("\n" + "="*90)
    print("IDX SWING SCREENER v2.1.0 - PRECISION ENTRY SYSTEM")
    print("Structure-Based Scaling Entries | Buy on Weakness | Respect Price Action")
    print("="*90)
    
    # Load tickers
    tickers = load_tickers(args.tickers)
    logger.info(f"Loaded {len(tickers)} tickers")
    
    # Configuration
    config = TradingConfig()
    
    try:
        if args.precision:
            # Enhanced screening with precision entry
            screener = EnhancedScreener(config)
            results = screener.screen_with_precision(tickers)
            
            # Filter by minimum score
            results = [(bs, pe) for bs, pe in results if bs.technical_score >= args.min_score]
            
            # Print results
            regime = screener.base_screener.market_regime.regime if screener.base_screener.market_regime else "Unknown"
            print_precision_results(results, regime, args.detail)
            
            # Export to CSV
            if results:
                csv_file = export_precision_signals(results)
                print(f"\n📊 Precision signals exported to: {csv_file}")
                print("\n💡 Import into Excel/Google Sheets for tracking")
        
        else:
            # Standard screening (backward compatible)
            print("\n⚠️  Running in standard mode (no precision entry)")
            print("Add --precision flag for structure-based precision entries\n")
            
            screener = IDXSwingScreener(config)
            signals = screener.screen(tickers)
            
            regime = screener.market_regime.regime if screener.market_regime else "Unknown"
            PerformanceReporter.print_signal_report(signals, regime)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)
    
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        print(f"\n❌ FATAL ERROR: {str(e)}")
        sys.exit(1)
    
    print("\n" + "="*90)
    print("✅ Screening complete - Good luck with your trades!")
    print("="*90 + "\n")

if __name__ == "__main__":
    main()
