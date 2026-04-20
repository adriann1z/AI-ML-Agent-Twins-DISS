"""
data_generator.py  –  v2
Synthetic XAUUSD price stream.
Improvements over v1:
  - AR(1) return autocorrelation (realistic momentum)
  - Jump diffusion (sudden price shocks)
  - 7 distinct regimes with clearer drift signatures
  - Bid-ask spread simulation
  - Regime transition log for dashboard annotation
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class MarketRegime:
    name:       str
    drift:      float
    volatility: float
    ar_coef:    float    # AR(1) return autocorrelation
    jump_prob:  float    # probability of jump per step
    jump_scale: float    # std of jump magnitude
    duration:   int
    color:      str


REGIMES: List[MarketRegime] = [
    MarketRegime("Bull Trend",      +0.0009, 0.003, 0.15, 0.002, 0.008, 380, "#00e676"),
    MarketRegime("High Volatility", +0.0001, 0.013, 0.05, 0.010, 0.025, 280, "#ff6d00"),
    MarketRegime("Bear Trend",      -0.0007, 0.004, 0.12, 0.003, 0.010, 360, "#f44336"),
    MarketRegime("Consolidation",   +0.0000, 0.002, 0.02, 0.001, 0.004, 230, "#90caf9"),
    MarketRegime("Flash Crash",     -0.0020, 0.020, 0.00, 0.030, 0.050, 180, "#ce93d8"),
    MarketRegime("Recovery Rally",  +0.0012, 0.007, 0.20, 0.005, 0.015, 400, "#80cbc4"),
    MarketRegime("Ranging Market",  +0.0000, 0.003, 0.08, 0.002, 0.006, 270, "#fbbf24"),
]


class SyntheticXAUUSD:
    """
    Non-stationary synthetic XAUUSD price stream.
    Models: GARCH(1,1) volatility + AR(1) momentum + jump diffusion.
    """

    def __init__(self, seed: int = 42, base_price: float = 1950.0):
        self.rng        = np.random.default_rng(seed)
        self.price      = base_price
        self.step       = 0
        self.vol_carry  = 0.0
        self.prev_ret   = 0.0
        self.history: List[dict] = []
        self.regime_transitions: List[dict] = []
        self._build_schedule()

    def _build_schedule(self):
        self._schedule: List[MarketRegime] = []
        for r in REGIMES:
            self._schedule.extend([r] * r.duration)
        self._slen = len(self._schedule)

    def current_regime(self) -> MarketRegime:
        return self._schedule[self.step % self._slen]

    def next_price(self) -> dict:
        regime = self.current_regime()
        prev   = self._schedule[(self.step - 1) % self._slen] if self.step > 0 else regime
        regime_changed = regime.name != prev.name

        if regime_changed:
            self.regime_transitions.append({
                "step":  self.step,
                "from":  prev.name,
                "to":    regime.name,
                "color": regime.color,
            })

        # GARCH(1,1)
        alpha, beta = 0.10, 0.84
        innov = self.rng.standard_normal()
        self.vol_carry = np.sqrt(max(1e-10,
            (1 - alpha - beta) * regime.volatility ** 2
            + alpha * (innov * regime.volatility) ** 2
            + beta * self.vol_carry ** 2
        ))
        eff_vol = max(self.vol_carry, regime.volatility * 0.4)

        # Return: drift + AR(1) + GARCH noise + jump
        jump = (self.rng.normal(0, regime.jump_scale)
                if self.rng.random() < regime.jump_prob else 0.0)
        ret  = regime.drift + regime.ar_coef * self.prev_ret + eff_vol * innov + jump
        self.prev_ret = ret
        self.price = max(100.0, self.price * (1.0 + ret))

        spread = self.price * 0.0001
        record = {
            "step":           self.step,
            "price":          round(self.price, 4),
            "bid":            round(self.price - spread / 2, 4),
            "ask":            round(self.price + spread / 2, 4),
            "return":         round(ret, 6),
            "jump":           round(jump, 6),
            "volatility":     round(eff_vol, 6),
            "regime":         regime.name,
            "regime_color":   regime.color,
            "regime_changed": regime_changed,
        }
        self.history.append(record)
        self.step += 1
        return record

    def generate_batch(self, n: int) -> List[dict]:
        return [self.next_price() for _ in range(n)]

    def prices_array(self, last_n: Optional[int] = None) -> np.ndarray:
        arr = np.array([r["price"] for r in self.history])
        return arr if last_n is None else arr[-last_n:]
