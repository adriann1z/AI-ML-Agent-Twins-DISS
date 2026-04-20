"""
mt5_bridge/live_feed.py
Bridges live MT5 tick data (or synthetic fallback) into the
digital twin feature pipeline.
Handles: price normalisation, feature extraction, sequence building.
"""

import numpy as np
from datetime import datetime
from typing import Optional, List
from collections import deque
import sys
import os

# Allow importing from parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features import FeatureEngine, SequenceBuffer, SEQUENCE_LENGTH, FEATURE_DIM
from data_generator import SyntheticXAUUSD


class LiveFeedAdapter:
    """
    Accepts price ticks from either MT5 or a synthetic generator
    and outputs feature sequences ready for LSTM inference.

    When MT5 is unavailable, automatically falls back to the
    synthetic XAUUSD generator so the dashboard always works.
    """

    def __init__(self, use_synthetic_fallback: bool = True, seed: int = 42):
        self.feat_engine  = FeatureEngine(maxlen=600)
        self.seq_buffer   = SequenceBuffer()
        self.price_buffer = deque(maxlen=1000)
        self.tick_count   = 0
        self.last_price   = None
        self.source       = "unknown"

        # Synthetic fallback
        self._synthetic   = None
        if use_synthetic_fallback:
            self._synthetic = SyntheticXAUUSD(seed=seed)
            self.source = "synthetic"

    def ingest_tick(self, price: float, spread: float = 0.0,
                    timestamp: Optional[datetime] = None
                    ) -> Optional[np.ndarray]:
        """
        Push a live price tick through the feature pipeline.
        Returns a (seq_len, feat_dim) array when the buffer is full,
        else None.
        """
        self.last_price = price
        self.price_buffer.append(price)
        self.tick_count += 1

        feat = self.feat_engine.update(price)
        if feat is None:
            return None

        self.seq_buffer.push(feat)
        if not self.seq_buffer.ready():
            return None

        return self.seq_buffer.get_sequence()

    def synthetic_tick(self) -> dict:
        """
        Pull the next synthetic tick (fallback mode).
        Returns the full tick dict including price, regime, etc.
        """
        if self._synthetic is None:
            raise RuntimeError("Synthetic generator not initialised")
        tick = self._synthetic.next_price()
        self.source = "synthetic"
        return tick

    def prices_array(self, last_n: int = 200) -> np.ndarray:
        arr = np.array(self.price_buffer)
        return arr[-last_n:] if len(arr) >= last_n else arr

    def warmup_from_prices(self, prices: List[float]):
        """Pre-fill the feature engine from historical prices."""
        for p in prices:
            feat = self.feat_engine.update(p)
            if feat is not None:
                self.seq_buffer.push(feat)
        self.source = "warmed-up"
