"""
features.py  –  v2
Feature engineering pipeline.
Expanded from 12 → 16 features:
  + Bollinger Band position
  + Average True Range (ATR proxy)
  + Return autocorrelation (lag-1)
  + Skewness of recent returns
All features normalised; extreme values clipped.
"""

import numpy as np
from collections import deque
from typing import Optional


SEQUENCE_LENGTH = 20    # LSTM look-back window
FEATURE_DIM     = 16    # number of features per timestep

FEATURE_NAMES = [
    "return_1", "return_5", "return_20",
    "vol_10", "vol_20",
    "rsi_14",
    "macd", "macd_signal",
    "zscore_20",
    "momentum_5", "momentum_10",
    "hl_ratio_10",
    "bb_position",     # Bollinger band position (0=lower, 1=upper)
    "atr_proxy",       # Average True Range proxy (normalised)
    "autocorr_lag1",   # Lag-1 return autocorrelation
    "ret_skew",        # Skewness of recent returns
]


class FeatureEngine:
    """
    Maintains rolling price/return buffers and computes features incrementally.
    """

    def __init__(self, maxlen: int = 500):
        self.prices  = deque(maxlen=maxlen)
        self.returns = deque(maxlen=maxlen)

    def update(self, price: float) -> Optional[np.ndarray]:
        if len(self.prices) > 0:
            ret = (price - self.prices[-1]) / (self.prices[-1] + 1e-10)
            self.returns.append(ret)
        self.prices.append(price)

        if len(self.prices) < 30:
            return None

        p = np.array(self.prices)
        r = np.array(self.returns)

        features = np.array([
            self._return(r, 1),
            self._return(r, 5),
            self._return(r, 20),
            self._vol(r, 10),
            self._vol(r, 20),
            self._rsi(p, 14),
            self._macd(p),
            self._macd_signal(p),
            self._zscore(p, 20),
            self._momentum(p, 5),
            self._momentum(p, 10),
            self._hl_ratio(p, 10),
            self._bb_position(p, 20),
            self._atr_proxy(p, 14),
            self._autocorr_lag1(r, 20),
            self._skewness(r, 20),
        ], dtype=np.float32)

        features = np.nan_to_num(features, nan=0.0, posinf=5.0, neginf=-5.0)
        features = np.clip(features, -5.0, 5.0)
        return features

    # ---------------------------------------------------------------- #
    #  Feature implementations                                          #
    # ---------------------------------------------------------------- #

    def _return(self, r, n):
        return float(np.sum(r[-n:])) if len(r) >= n else 0.0

    def _vol(self, r, n):
        return float(np.std(r[-n:]) * 100) if len(r) >= n else 0.0

    def _rsi(self, p, n=14):
        if len(p) < n + 1:
            return 0.5
        d = np.diff(p[-(n+1):])
        g = np.where(d > 0, d, 0.0).mean()
        l = np.where(d < 0, -d, 0.0).mean()
        return float(1.0 - 1.0 / (1.0 + g / (l + 1e-10)))

    def _ema(self, p, span):
        alpha = 2.0 / (span + 1)
        ema = float(p[0])
        for x in p[1:]:
            ema = alpha * float(x) + (1 - alpha) * ema
        return ema

    def _macd(self, p):
        if len(p) < 26:
            return 0.0
        return float((self._ema(p[-12:], 12) - self._ema(p[-26:], 26))
                     / (abs(self._ema(p[-26:], 26)) + 1e-10))

    def _macd_signal(self, p):
        if len(p) < 35:
            return 0.0
        macds = []
        for i in range(9):
            end = len(p) - i
            sub = p[max(0, end - 26):end]
            if len(sub) >= 26:
                macds.insert(0, (self._ema(sub[-12:], 12) - self._ema(sub, 26))
                             / (abs(self._ema(sub, 26)) + 1e-10))
        return float(self._ema(np.array(macds), 9)) if len(macds) >= 2 else 0.0

    def _zscore(self, p, n):
        if len(p) < n:
            return 0.0
        w = p[-n:]
        return float((p[-1] - w.mean()) / (w.std() + 1e-10))

    def _momentum(self, p, n):
        if len(p) < n + 1:
            return 0.0
        return float((p[-1] - p[-(n+1)]) / (abs(p[-(n+1)]) + 1e-10))

    def _hl_ratio(self, p, n):
        if len(p) < n:
            return 0.5
        w = p[-n:]
        rng = w.max() - w.min()
        return float((p[-1] - w.min()) / (rng + 1e-10))

    def _bb_position(self, p, n=20):
        if len(p) < n:
            return 0.5
        w   = p[-n:]
        mu  = w.mean()
        sig = w.std()
        upper = mu + 2 * sig
        lower = mu - 2 * sig
        band  = upper - lower
        return float(np.clip((p[-1] - lower) / (band + 1e-10), 0.0, 1.0))

    def _atr_proxy(self, p, n=14):
        """True range proxy using price differences."""
        if len(p) < n + 1:
            return 0.0
        trs  = np.abs(np.diff(p[-(n+1):]))
        atr  = trs.mean()
        return float(atr / (abs(p[-1]) + 1e-10) * 100)

    def _autocorr_lag1(self, r, n=20):
        if len(r) < n + 1:
            return 0.0
        w = np.array(r[-n:])
        if w.std() < 1e-10:
            return 0.0
        return float(np.corrcoef(w[:-1], w[1:])[0, 1])

    def _skewness(self, r, n=20):
        if len(r) < n:
            return 0.0
        w  = np.array(r[-n:])
        mu = w.mean()
        sd = w.std()
        if sd < 1e-10:
            return 0.0
        return float(np.mean(((w - mu) / sd) ** 3))


class SequenceBuffer:
    """Accumulates feature vectors into fixed-length LSTM input sequences."""

    def __init__(self, seq_len: int = SEQUENCE_LENGTH, feat_dim: int = FEATURE_DIM):
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.buffer  = deque(maxlen=seq_len)

    def push(self, feat: np.ndarray):
        self.buffer.append(feat.copy())

    def ready(self) -> bool:
        return len(self.buffer) == self.seq_len

    def get_sequence(self) -> np.ndarray:
        """Returns (seq_len, feat_dim) array."""
        return np.array(self.buffer, dtype=np.float32)
