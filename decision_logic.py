"""
decision_logic.py  –  v2
Enhanced gating logic that considers confidence, entropy, and cooldown.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Decision:
    twin_name:  str
    step:       int
    price:      float
    direction:  int
    confidence: float
    epistemic:  float
    entropy:    float
    acted:      bool
    reason:     str
    signal:     str   # "BUY" / "SELL" / "ABSTAIN"


def gate_decision(
    twin_name:  str,
    step:       int,
    price:      float,
    direction:  int,
    confidence: float,
    epistemic:  float,
    entropy:    float,
    will_act:   bool,
    conf_threshold: float,
    entropy_threshold: float,
) -> Decision:
    if will_act:
        signal = "BUY" if direction == 1 else "SELL"
        reason = (f"conf={confidence:.3f} ≥ {conf_threshold} · "
                  f"H={entropy:.3f} ≤ {entropy_threshold:.2f} → {signal}")
    else:
        signal = "ABSTAIN"
        if confidence < conf_threshold:
            reason = f"conf={confidence:.3f} < {conf_threshold} → abstain"
        elif entropy > entropy_threshold:
            reason = f"entropy={entropy:.3f} > {entropy_threshold:.2f} → abstain (high uncertainty)"
        else:
            reason = "cooldown active → abstain"

    return Decision(
        twin_name=twin_name, step=step, price=price,
        direction=direction, confidence=confidence,
        epistemic=epistemic, entropy=entropy,
        acted=will_act, reason=reason, signal=signal,
    )
