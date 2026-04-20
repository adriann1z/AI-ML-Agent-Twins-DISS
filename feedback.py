"""
feedback.py  –  v3
Outcome evaluation with expanded metrics:
  - Sharpe ratio of PnL proxy
  - Win rate per regime
  - Max drawdown
  - Decision Quality Index (DQI)
  - Actual supervised labels for post-hoc learning
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from collections import defaultdict


OUTCOME_HORIZON = 5   # evaluate decision N steps after entry


@dataclass
class OutcomeRecord:
    step: int
    twin_name: str
    direction: int
    confidence: float
    epistemic: float
    entropy: float
    acted: bool
    correct: bool
    actual_label: int
    pnl_proxy: float
    price_in: float
    price_out: float
    regime: str


class OutcomeEvaluator:
    """
    Registers decisions and resolves them after OUTCOME_HORIZON steps.
    Maintains a full log with rich metrics.
    """

    def __init__(self, horizon: int = OUTCOME_HORIZON):
        self.horizon = horizon
        self.pending: List[dict] = []
        self.log: List[OutcomeRecord] = []

    def register(
        self,
        step: int,
        twin_name: str,
        direction: int,
        confidence: float,
        epistemic: float,
        entropy: float,
        acted: bool,
        price_in: float,
        sequence: np.ndarray,
        regime: str,
    ):
        self.pending.append(
            {
                "step": step,
                "twin_name": twin_name,
                "direction": direction,
                "confidence": confidence,
                "epistemic": epistemic,
                "entropy": entropy,
                "acted": acted,
                "price_in": price_in,
                "sequence": sequence,
                "regime": regime,
                "resolve_at": step + self.horizon,
            }
        )

    def resolve(self, current_step: int, current_price: float) -> List[Tuple[OutcomeRecord, np.ndarray]]:
        resolved, remaining = [], []
        for p in self.pending:
            if current_step >= p["resolve_at"]:
                actual_up = current_price > p["price_in"]
                actual_label = 1 if actual_up else 0
                predicted_up = p["direction"] == 1
                correct = actual_up == predicted_up
                ret = (current_price - p["price_in"]) / (p["price_in"] + 1e-10)
                pnl = ret if predicted_up else -ret

                rec = OutcomeRecord(
                    step=p["step"],
                    twin_name=p["twin_name"],
                    direction=p["direction"],
                    confidence=p["confidence"],
                    epistemic=p["epistemic"],
                    entropy=p["entropy"],
                    acted=p["acted"],
                    correct=correct,
                    actual_label=actual_label,
                    pnl_proxy=pnl,
                    price_in=p["price_in"],
                    price_out=current_price,
                    regime=p["regime"],
                )
                self.log.append(rec)
                resolved.append((rec, p["sequence"]))
            else:
                remaining.append(p)
        self.pending = remaining
        return resolved

    def _acted_log(self) -> List[OutcomeRecord]:
        return [r for r in self.log if r.acted]

    def cumulative_pnl(self) -> List[float]:
        total, series = 0.0, []
        for r in self._acted_log():
            total += r.pnl_proxy
            series.append(total)
        return series

    def rolling_dqi(self, window: int = 50) -> List[float]:
        acted = self._acted_log()
        if not acted:
            return []
        dqi = []
        for i in range(len(acted)):
            chunk = acted[max(0, i - window + 1): i + 1]
            dqi.append(float(np.mean([1 if r.correct else 0 for r in chunk])))
        return dqi

    def sharpe_ratio(self, window: int = 200) -> float:
        pnls = [r.pnl_proxy for r in self._acted_log()[-window:]]
        if len(pnls) < 5:
            return 0.0
        arr = np.array(pnls)
        return float(arr.mean() / (arr.std() + 1e-10) * np.sqrt(252))

    def max_drawdown(self) -> float:
        cpnl = self.cumulative_pnl()
        if not cpnl:
            return 0.0
        arr = np.array(cpnl)
        peak = np.maximum.accumulate(arr)
        dd = (arr - peak) / (np.abs(peak) + 1e-10)
        return float(dd.min())

    def win_rate(self) -> float:
        acted = self._acted_log()
        if not acted:
            return 0.0
        return float(np.mean([1 if r.correct else 0 for r in acted]))

    def regime_win_rates(self) -> dict:
        per_regime = defaultdict(list)
        for r in self._acted_log():
            per_regime[r.regime].append(1 if r.correct else 0)
        return {k: round(float(np.mean(v)), 3) for k, v in per_regime.items()}

    def total_decisions(self) -> int:
        return len(self._acted_log())
