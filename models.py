"""
models.py  –  v3
Upgrades over v2:
  - Calibrated confidence derived from directional certainty + uncertainty
  - Replay learning only from resolved outcomes (no self-labelling bias)
  - Optional learning from abstained but resolved samples
  - Thread clamp for stable CPU execution
  - Extra bias diagnostics for BUY/SELL imbalance
"""

import math
import os
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from features import FEATURE_DIM


try:
    torch.set_num_threads(max(1, int(os.getenv("TWIN_TORCH_THREADS", "1"))))
except Exception:
    pass


class TemporalAttention(nn.Module):
    """Scaled dot-product style attention over LSTM time steps."""

    def __init__(self, hidden: int):
        super().__init__()
        self.attn = nn.Linear(hidden, 1)

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        scores = self.attn(lstm_out).squeeze(-1)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return (lstm_out * weights).sum(dim=1)


class TwinLSTM(nn.Module):
    """
    2-layer LSTM with temporal attention and dual output heads.
    The confidence head is retained for backwards-compatible checkpoint loading,
    but runtime confidence is calibrated from the direction distribution.
    """

    def __init__(self, feat_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.hidden = hidden
        self.dropout_p = dropout

        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.attention = TemporalAttention(hidden)
        self.dropout_layer = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(hidden)
        self.direction_head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out, _ = self.lstm(x)
        context = self.attention(out)
        last = out[:, -1, :]
        fused = self.dropout_layer(context + last)
        if fused.shape[0] > 1:
            fused = self.bn(fused)
        direction = torch.sigmoid(self.direction_head(fused)).squeeze(-1)
        confidence = torch.sigmoid(self.confidence_head(fused)).squeeze(-1)
        return direction, confidence

    def mc_predict(self, x: torch.Tensor, n_samples: int = 20) -> Tuple[float, float, float, float]:
        """
        Monte Carlo Dropout inference.
        Returns: (mean_direction, calibrated_confidence, epistemic_uncertainty, entropy)
        """
        self.train()
        dirs = []
        with torch.no_grad():
            for _ in range(n_samples):
                d, _ = self.forward(x)
                dirs.append(d.item())
        self.eval()

        dir_arr = np.array(dirs, dtype=np.float32)
        mean_dir = float(dir_arr.mean())
        epistemic = float(dir_arr.std())
        p = float(np.clip(mean_dir, 1e-6, 1 - 1e-6))
        entropy = float(-p * math.log(p) - (1 - p) * math.log(1 - p))

        directional_edge = float(np.clip(abs(mean_dir - 0.5) * 2.0, 0.0, 1.0))
        entropy_term = float(np.clip(1.0 - entropy / math.log(2), 0.0, 1.0))
        epistemic_term = float(np.clip(1.0 - epistemic / 0.25, 0.0, 1.0))
        calibrated_conf = float(
            np.clip(
                0.55 * directional_edge + 0.25 * entropy_term + 0.20 * epistemic_term,
                0.0,
                1.0,
            )
        )
        return mean_dir, calibrated_conf, epistemic, entropy


class ReplayBuffer:
    def __init__(self, maxlen: int = 1500):
        self.maxlen = maxlen
        self.sequences = deque(maxlen=maxlen)
        self.labels = deque(maxlen=maxlen)

    def push(self, seq: np.ndarray, label: int):
        self.sequences.append(seq.astype(np.float32))
        self.labels.append(int(label))

    def sample(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.sequences) == 0:
            raise ValueError("ReplayBuffer is empty")
        n = min(n, len(self.sequences))
        idx = np.random.choice(len(self.sequences), size=n, replace=False)
        seqs = np.stack([self.sequences[i] for i in idx])
        labels = np.array([self.labels[i] for i in idx], dtype=np.float32)
        return torch.tensor(seqs), torch.tensor(labels)

    def class_balance(self) -> dict:
        if not self.labels:
            return {"buy": 0, "sell": 0, "buy_ratio": 0.5}
        arr = np.array(self.labels, dtype=np.int32)
        buys = int(arr.sum())
        sells = int(len(arr) - buys)
        return {"buy": buys, "sell": sells, "buy_ratio": float(arr.mean())}

    def __len__(self):
        return len(self.sequences)


@dataclass
class RetrainEvent:
    step: int
    trigger: str
    acc_before: float
    acc_after: float
    loss: float
    grad_norm: float
    version: int


@dataclass
class DecisionRecord:
    step: int
    price: float
    direction: int
    confidence: float
    epistemic: float
    entropy: float
    acted: bool
    correct: Optional[bool]
    regime: str


class DigitalTwin:
    """
    Wraps TwinLSTM with adaptation policy:
      - Time-based and performance-based retraining
      - MC-dropout uncertainty quantification
      - Cooldown risk control
      - Outcome learning only from resolved labels
    """

    def __init__(
        self,
        name: str,
        confidence_threshold: float,
        retrain_every: int,
        lr: float,
        hidden: int = 96,
        dropout: float = 0.2,
        mc_samples: int = 20,
        perf_retrain_window: int = 50,
        perf_retrain_thresh: float = 0.50,
        cooldown_trigger: int = 3,
        cooldown_steps: int = 10,
        entropy_threshold: float = 0.65,
        model_weight: float = 0.5,
        heuristic_weight: float = 0.5,
        min_edge: float = 0.0,
        min_retrain_gap: Optional[int] = None,
    ):
        self.name = name
        self.trade_tag = name.split()[0]
        self.conf_threshold = confidence_threshold
        self.retrain_every = retrain_every
        self.lr = lr
        self.mc_samples = mc_samples
        self.perf_retrain_window = perf_retrain_window
        self.perf_retrain_thresh = perf_retrain_thresh
        self.cooldown_trigger = cooldown_trigger
        self.cooldown_steps = cooldown_steps
        self.entropy_threshold = entropy_threshold
        total_weight = max(model_weight + heuristic_weight, 1e-6)
        self.model_weight = model_weight / total_weight
        self.heuristic_weight = heuristic_weight / total_weight
        self.min_edge = float(np.clip(min_edge, 0.0, 0.95))
        self.min_retrain_gap = int(min_retrain_gap or max(30, retrain_every // 4))

        self.model = TwinLSTM(FEATURE_DIM, hidden, dropout)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
        )
        self.criterion = nn.BCELoss()

        self.replay = ReplayBuffer(maxlen=1500)
        self.step = 0
        self.version = 1
        self.retrain_events: List[RetrainEvent] = []
        self.decision_log: List[DecisionRecord] = []
        self.loss_history: List[float] = []

        self.outcome_window: deque = deque(maxlen=200)
        self.confidence_history: deque = deque(maxlen=500)
        self.epistemic_history: deque = deque(maxlen=500)
        self.entropy_history: deque = deque(maxlen=500)
        self.direction_prob_history: deque = deque(maxlen=500)
        self.recent_outcomes: deque = deque(maxlen=perf_retrain_window)
        self.consecutive_losses = 0
        self.cooldown_remaining = 0
        self.last_retrain_step = 0

    def _heuristic_edge(self, sequence: np.ndarray) -> float:
        """Feature-space directional edge in [-1, 1] to stabilise bias-prone model output."""
        last = np.asarray(sequence[-1], dtype=np.float32)
        score = 0.0
        score += 0.28 * np.tanh(float(last[1]) * 22.0)      # return_5
        score += 0.18 * np.tanh(float(last[2]) * 14.0)      # return_20
        score += 0.16 * np.tanh(float(last[6]) * 150.0)     # macd
        score += 0.10 * np.tanh(float(last[7]) * 120.0)     # macd_signal
        score += 0.10 * np.tanh((float(last[5]) - 0.5) * 3.0)   # rsi
        score += 0.10 * np.tanh(float(last[10]) * 18.0)     # momentum_10
        score += 0.08 * np.tanh((float(last[12]) - 0.5) * 2.5)  # bb_position
        return float(np.clip(score, -1.0, 1.0))

    def predict(self, sequence: np.ndarray) -> Tuple[int, float, float, float, bool]:
        x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
        raw_dir, _, epistemic, entropy = self.model.mc_predict(x, n_samples=self.mc_samples)
        model_edge = float(np.clip((raw_dir - 0.5) * 2.0, -1.0, 1.0))
        heuristic_edge = self._heuristic_edge(sequence)
        ensemble_edge = float(
            np.clip(
                self.model_weight * model_edge + self.heuristic_weight * heuristic_edge,
                -1.0,
                1.0,
            )
        )
        mean_dir = float(np.clip(0.5 + 0.5 * ensemble_edge, 0.001, 0.999))
        direction = int(mean_dir > 0.5)

        directional_conf = abs(ensemble_edge)
        entropy_term = float(np.clip(1.0 - entropy / math.log(2), 0.0, 1.0))
        epistemic_term = float(np.clip(1.0 - epistemic / 0.25, 0.0, 1.0))
        mean_conf = float(
            np.clip(
                0.60 * directional_conf + 0.25 * entropy_term + 0.15 * epistemic_term,
                0.0,
                1.0,
            )
        )

        in_cooldown = self.cooldown_remaining > 0
        edge_ok = directional_conf >= self.min_edge
        conf_ok = mean_conf >= self.conf_threshold
        entropy_ok = entropy <= self.entropy_threshold
        will_act = edge_ok and conf_ok and entropy_ok and not in_cooldown

        self.confidence_history.append(mean_conf)
        self.epistemic_history.append(epistemic)
        self.entropy_history.append(entropy)
        self.direction_prob_history.append(mean_dir)

        return direction, mean_conf, epistemic, entropy, will_act

    def record_outcome(
        self,
        sequence: np.ndarray,
        direction: int,
        confidence: float,
        epistemic: float,
        entropy: float,
        acted: bool,
        correct: Optional[bool],
        price: float,
        regime: str,
        actual_label: Optional[int] = None,
    ):
        """Log outcome, update replay buffer with resolved labels, then retrain if needed."""
        self.step += 1
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1

        resolved_label = actual_label
        if resolved_label is None and correct is not None:
            resolved_label = direction if correct else (1 - direction)

        if resolved_label is not None:
            self.replay.push(sequence, int(resolved_label))

        outcome_val = -1
        if acted and correct is not None:
            outcome_val = 1 if correct else 0
            self.recent_outcomes.append(outcome_val)
            if correct:
                self.consecutive_losses = 0
            else:
                self.consecutive_losses += 1
                if self.consecutive_losses >= self.cooldown_trigger:
                    self.cooldown_remaining = self.cooldown_steps
                    self.consecutive_losses = 0

        self.outcome_window.append(outcome_val)
        self.decision_log.append(
            DecisionRecord(
                step=self.step,
                price=price,
                direction=direction,
                confidence=confidence,
                epistemic=epistemic,
                entropy=entropy,
                acted=acted,
                correct=correct,
                regime=regime,
            )
        )

        self._check_retrain()

    def _check_retrain(self):
        if len(self.replay) < 64:
            return
        if self.step - self.last_retrain_step < self.min_retrain_gap:
            return
        trigger = None
        if (
            len(self.recent_outcomes) >= self.perf_retrain_window
            and np.mean(list(self.recent_outcomes)) < self.perf_retrain_thresh
        ):
            trigger = "performance"
        elif self.step > 0 and self.step % self.retrain_every == 0:
            trigger = "time"
        if trigger:
            self._retrain(trigger)

    def _retrain(self, trigger: str):
        acc_before = self.rolling_accuracy()
        self.model.train()
        total_loss = 0.0
        total_gnorm = 0.0
        epochs = 6

        for _ in range(epochs):
            seqs, labels = self.replay.sample(min(256, len(self.replay)))
            self.optimizer.zero_grad()
            dir_out, _ = self.model(seqs)
            loss = self.criterion(dir_out, labels)
            loss.backward()
            gnorm = nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
            total_gnorm += float(gnorm)

        avg_loss = total_loss / epochs
        avg_gnorm = total_gnorm / epochs
        self.scheduler.step(avg_loss)
        self.loss_history.append(avg_loss)
        self.model.eval()

        self.version += 1
        self.last_retrain_step = self.step
        self.recent_outcomes.clear()
        acc_after = self.rolling_accuracy()
        self.retrain_events.append(
            RetrainEvent(
                step=self.step,
                trigger=trigger,
                acc_before=round(acc_before, 4),
                acc_after=round(acc_after, 4),
                loss=round(avg_loss, 4),
                grad_norm=round(avg_gnorm, 4),
                version=self.version,
            )
        )

    def pretrain(self, sequences: List[np.ndarray], labels: List[int], epochs: int = 18):
        self.model.train()
        for seq, lbl in zip(sequences, labels):
            self.replay.push(seq, lbl)
        for _ in range(epochs):
            if len(self.replay) < 32:
                break
            seqs, lbls = self.replay.sample(min(256, len(self.replay)))
            self.optimizer.zero_grad()
            dir_out, _ = self.model(seqs)
            loss = self.criterion(dir_out, lbls)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        self.model.eval()

    def rolling_accuracy(self, window: int = 50) -> float:
        acted = [x for x in list(self.outcome_window)[-window:] if x != -1]
        return float(np.mean(acted)) if acted else 0.5

    def abstention_rate(self, window: int = 50) -> float:
        recent = list(self.outcome_window)[-window:]
        if not recent:
            return 0.0
        return sum(1 for x in recent if x == -1) / len(recent)

    def mean_confidence(self, window: int = 50) -> float:
        recent = list(self.confidence_history)[-window:]
        return float(np.mean(recent)) if recent else 0.5

    def confidence_std(self, window: int = 50) -> float:
        recent = list(self.confidence_history)[-window:]
        return float(np.std(recent)) if recent else 0.0

    def mean_epistemic(self, window: int = 50) -> float:
        recent = list(self.epistemic_history)[-window:]
        return float(np.mean(recent)) if recent else 0.0

    def mean_entropy(self, window: int = 50) -> float:
        recent = list(self.entropy_history)[-window:]
        return float(np.mean(recent)) if recent else 0.0

    def mean_direction_probability(self, window: int = 50) -> float:
        recent = list(self.direction_prob_history)[-window:]
        return float(np.mean(recent)) if recent else 0.5

    def buy_bias(self, window: int = 50) -> float:
        return self.mean_direction_probability(window) - 0.5

    def decisions_made(self) -> int:
        return sum(1 for r in self.decision_log if r.acted)

    def total_retrains(self) -> int:
        return len(self.retrain_events)

    def current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def dqi_series(self, window: int = 50) -> List[float]:
        acted = [r for r in self.decision_log if r.acted and r.correct is not None]
        if not acted:
            return []
        dqi = []
        for i in range(len(acted)):
            chunk = acted[max(0, i - window + 1): i + 1]
            dqi.append(float(np.mean([1 if r.correct else 0 for r in chunk])))
        return dqi

    def confidence_series(self, last_n: int = 300) -> List[float]:
        return list(self.confidence_history)[-last_n:]

    def epistemic_series(self, last_n: int = 300) -> List[float]:
        return list(self.epistemic_history)[-last_n:]

    def entropy_series(self, last_n: int = 300) -> List[float]:
        return list(self.entropy_history)[-last_n:]

    def regime_accuracy(self) -> dict:
        from collections import defaultdict

        regime_results = defaultdict(list)
        for r in self.decision_log:
            if r.acted and r.correct is not None:
                regime_results[r.regime].append(1 if r.correct else 0)
        return {k: round(float(np.mean(v)), 3) for k, v in regime_results.items()}


def build_twins() -> Tuple[DigitalTwin, DigitalTwin]:
    twin_a = DigitalTwin(
        name="Twin-A (Conservative)",
        confidence_threshold=0.30,
        retrain_every=500,
        lr=0.0003,
        hidden=96,
        dropout=0.25,
        mc_samples=24,
        perf_retrain_window=60,
        perf_retrain_thresh=0.46,
        cooldown_trigger=4,
        cooldown_steps=18,
        entropy_threshold=0.72,
        model_weight=0.62,
        heuristic_weight=0.38,
        min_edge=0.12,
        min_retrain_gap=120,
    )
    twin_b = DigitalTwin(
        name="Twin-B (Plastic)",
        confidence_threshold=0.50,
        retrain_every=200,
        lr=0.0010,
        hidden=96,
        dropout=0.15,
        mc_samples=20,
        perf_retrain_window=36,
        perf_retrain_thresh=0.48,
        cooldown_trigger=3,
        cooldown_steps=8,
        entropy_threshold=0.72,
        model_weight=0.55,
        heuristic_weight=0.45,
        min_edge=0.10,
        min_retrain_gap=60,
    )
    return twin_a, twin_b
