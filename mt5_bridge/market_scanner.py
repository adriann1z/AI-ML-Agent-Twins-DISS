"""
mt5_bridge/market_scanner.py
Scans live XAUUSD price data for:
  - Regime detection (volatility + trend based)
  - Signal conditions (RSI extremes, Bollinger breakouts, MACD crossovers)
  - Spread filter (avoid wide-spread entries)
Works on live MT5 prices OR synthetic prices when MT5 unavailable.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from collections import deque


@dataclass
class ScanResult:
    timestamp:       str
    price:           float
    spread:          float
    detected_regime: str
    regime_color:    str
    volatility:      float
    trend_strength:  float    # 0=no trend, 1=strong trend
    rsi:             float
    bb_position:     float    # 0=lower band, 1=upper band
    macd_cross:      str      # "bullish" / "bearish" / "none"
    spread_ok:       bool
    signal_summary:  str


REGIME_COLORS = {
    "High Volatility":  "#ff6d00",
    "Bull Trend":       "#00e676",
    "Bear Trend":       "#f44336",
    "Consolidation":    "#90caf9",
    "Ranging":          "#fbbf24",
    "Unknown":          "#78909c",
}

MAX_SPREAD_PIPS = 5.0   # reject entries if spread > 5 pips


class MarketScanner:
    """
    Maintains a rolling price buffer and runs regime + signal detection
    on each new tick.
    """

    def __init__(self, buffer_size: int = 300):
        self.prices  = deque(maxlen=buffer_size)
        self.spreads = deque(maxlen=buffer_size)
        self.scan_log: List[ScanResult] = []

    def update(self, price: float, spread: float = 0.0,
               timestamp: str = "") -> Optional[ScanResult]:
        self.prices.append(price)
        self.spreads.append(spread)

        if len(self.prices) < 30:
            return None

        p   = np.array(self.prices)
        ret = np.diff(p) / (p[:-1] + 1e-10)

        vol            = self._volatility(ret, 20)
        trend_strength = self._trend_strength(p, 50)
        regime, color  = self._detect_regime(vol, trend_strength, ret)
        rsi            = self._rsi(p, 14)
        bb_pos         = self._bb_position(p, 20)
        macd_cross     = self._macd_cross(p)
        spread_pips    = spread / 0.1   # XAUUSD: 1 pip = 0.1
        spread_ok      = spread_pips <= MAX_SPREAD_PIPS

        signal_parts = []
        if rsi < 0.30:
            signal_parts.append("RSI oversold")
        elif rsi > 0.70:
            signal_parts.append("RSI overbought")
        if bb_pos < 0.10:
            signal_parts.append("At BB lower")
        elif bb_pos > 0.90:
            signal_parts.append("At BB upper")
        if macd_cross != "none":
            signal_parts.append(f"MACD {macd_cross}")
        if not spread_ok:
            signal_parts.append(f"⚠ Wide spread ({spread_pips:.1f} pips)")

        result = ScanResult(
            timestamp       = timestamp or "live",
            price           = round(price, 4),
            spread          = round(spread, 5),
            detected_regime = regime,
            regime_color    = color,
            volatility      = round(vol * 100, 4),
            trend_strength  = round(trend_strength, 3),
            rsi             = round(rsi, 3),
            bb_position     = round(bb_pos, 3),
            macd_cross      = macd_cross,
            spread_ok       = spread_ok,
            signal_summary  = " · ".join(signal_parts) if signal_parts else "No signals",
        )
        self.scan_log.append(result)
        return result

    # ---------------------------------------------------------------- #
    #  Detection methods                                                 #
    # ---------------------------------------------------------------- #

    def _volatility(self, ret: np.ndarray, n: int) -> float:
        if len(ret) < n:
            return 0.0
        return float(np.std(ret[-n:]))

    def _trend_strength(self, p: np.ndarray, n: int) -> float:
        """Linear regression R² over last n bars as trend strength."""
        if len(p) < n:
            return 0.0
        y = p[-n:]
        x = np.arange(n)
        xm, ym = x.mean(), y.mean()
        ss_tot = ((y - ym) ** 2).sum()
        ss_res = ((y - (xm * np.cov(x, y)[0, 1] / (np.var(x) + 1e-10)
                        * (x - xm) + ym)) ** 2).sum()
        r2 = 1 - ss_res / (ss_tot + 1e-10)
        return float(np.clip(r2, 0, 1))

    def _detect_regime(self, vol: float, trend: float,
                       ret: np.ndarray) -> tuple:
        mean_ret = float(ret[-20:].mean()) if len(ret) >= 20 else 0.0

        if vol > 0.008:
            return "High Volatility", REGIME_COLORS["High Volatility"]
        if trend > 0.6 and mean_ret > 0.0002:
            return "Bull Trend", REGIME_COLORS["Bull Trend"]
        if trend > 0.6 and mean_ret < -0.0002:
            return "Bear Trend", REGIME_COLORS["Bear Trend"]
        if vol < 0.002:
            return "Consolidation", REGIME_COLORS["Consolidation"]
        return "Ranging", REGIME_COLORS["Ranging"]

    def _rsi(self, p: np.ndarray, n: int = 14) -> float:
        if len(p) < n + 1:
            return 0.5
        d = np.diff(p[-(n+1):])
        g = np.where(d > 0, d, 0.0).mean()
        l = np.where(d < 0, -d, 0.0).mean()
        return float(1.0 - 1.0 / (1.0 + g / (l + 1e-10)))

    def _bb_position(self, p: np.ndarray, n: int = 20) -> float:
        if len(p) < n:
            return 0.5
        w   = p[-n:]
        mu  = w.mean()
        sig = w.std()
        return float(np.clip((p[-1] - (mu - 2*sig)) / (4*sig + 1e-10), 0, 1))

    def _macd_cross(self, p: np.ndarray) -> str:
        if len(p) < 28:
            return "none"
        def ema(arr, s):
            a, e = 2/(s+1), float(arr[0])
            for x in arr[1:]:
                e = a*float(x) + (1-a)*e
            return e

        macd_now  = ema(p[-12:], 12) - ema(p[-26:], 26)
        macd_prev = ema(p[-13:-1], 12) - ema(p[-27:-1], 26)

        if macd_prev < 0 and macd_now > 0:
            return "bullish"
        if macd_prev > 0 and macd_now < 0:
            return "bearish"
        return "none"

    def recent_scan_df(self, n: int = 20):
        import pandas as pd
        recent = self.scan_log[-n:]
        if not recent:
            return pd.DataFrame()
        return pd.DataFrame([{
            "Time":     r.timestamp,
            "Price":    f"${r.price:,.2f}",
            "Regime":   r.detected_regime,
            "Vol %":    f"{r.volatility:.4f}",
            "RSI":      f"{r.rsi:.3f}",
            "BB Pos":   f"{r.bb_position:.3f}",
            "MACD":     r.macd_cross,
            "Spread OK": "✓" if r.spread_ok else "✗",
            "Signals":  r.signal_summary,
        } for r in reversed(recent)])
