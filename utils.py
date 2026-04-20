"""
utils.py  –  v4
Simulation factory and transfer utility.
Upgrades:
  - Horizon-aligned supervised labels
  - Class balancing to reduce BUY-only collapse
  - Auto-save with state versioning
"""

import random
from typing import List, Sequence, Tuple

import numpy as np

from data_generator import SyntheticXAUUSD
from feedback import OUTCOME_HORIZON, OutcomeEvaluator
from features import FeatureEngine, SequenceBuffer
from models import build_twins
from twin_state import save_twins

PRETRAIN_STEPS = 900


def to_csv_bytes(df) -> bytes:
    if df is None or getattr(df, "empty", True):
        return b""
    return df.to_csv(index=False).encode("utf-8")


def _build_labelled_sequences(history: Sequence[dict]) -> Tuple[List[np.ndarray], List[int]]:
    feat_engine = FeatureEngine()
    seq_buffer = SequenceBuffer()
    candidate_sequences: List[np.ndarray] = []
    candidate_prices: List[float] = []

    for tick in history:
        feat = feat_engine.update(tick["price"])
        if feat is None:
            continue
        seq_buffer.push(feat)
        if seq_buffer.ready():
            candidate_sequences.append(seq_buffer.get_sequence().copy())
            candidate_prices.append(float(tick["price"]))

    if len(candidate_sequences) <= OUTCOME_HORIZON:
        return [], []

    sequences: List[np.ndarray] = []
    labels: List[int] = []
    for idx in range(len(candidate_sequences) - OUTCOME_HORIZON):
        future_price = candidate_prices[idx + OUTCOME_HORIZON]
        current_price = candidate_prices[idx]
        label = 1 if future_price > current_price else 0
        sequences.append(candidate_sequences[idx])
        labels.append(label)
    return sequences, labels


def _rebalance_binary_dataset(sequences: Sequence[np.ndarray], labels: Sequence[int]) -> Tuple[List[np.ndarray], List[int]]:
    if not sequences:
        return [], []

    ones = [i for i, lbl in enumerate(labels) if lbl == 1]
    zeros = [i for i, lbl in enumerate(labels) if lbl == 0]
    if not ones or not zeros:
        idx = list(range(len(labels)))
    else:
        target = min(max(len(ones), len(zeros)), min(len(ones), len(zeros)) * 2)
        rng = np.random.default_rng(123)
        ones_sel = rng.choice(ones, size=target, replace=len(ones) < target)
        zeros_sel = rng.choice(zeros, size=target, replace=len(zeros) < target)
        idx = np.concatenate([ones_sel, zeros_sel]).tolist()
        rng.shuffle(idx)

    return [np.array(sequences[i], copy=True) for i in idx], [int(labels[i]) for i in idx]


def pretrain_twins(generator, twin_a, twin_b) -> List[dict]:
    history = [generator.next_price() for _ in range(PRETRAIN_STEPS)]
    sequences, labels = _build_labelled_sequences(history)
    balanced_sequences, balanced_labels = _rebalance_binary_dataset(sequences, labels)

    twin_a.pretrain(balanced_sequences, balanced_labels, epochs=18)
    twin_b.pretrain(balanced_sequences, balanced_labels, epochs=18)
    return history


def make_fresh_simulation(seed: int = None, auto_save: bool = True):
    if seed is None:
        seed = random.randint(0, 99999)

    generator = SyntheticXAUUSD(seed=seed)
    twin_a, twin_b = build_twins()
    pretrain_history = pretrain_twins(generator, twin_a, twin_b)

    if auto_save:
        save_twins(twin_a, twin_b, source="simulation_pretrain")

    feat_engine = FeatureEngine()
    seq_buffer = SequenceBuffer()
    for tick in pretrain_history:
        feat = feat_engine.update(tick["price"])
        if feat is not None:
            seq_buffer.push(feat)

    return (
        generator,
        twin_a,
        twin_b,
        feat_engine,
        seq_buffer,
        OutcomeEvaluator(),
        OutcomeEvaluator(),
        pretrain_history,
    )


def save_current_twins(twin_a, twin_b, source: str = "mid_simulation"):
    return save_twins(twin_a, twin_b, source=source)
