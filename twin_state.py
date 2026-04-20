"""
twin_state.py
Persistent state for trained digital twins.
Shared between app.py and live_dashboard.py.
"""

import json
import os
import pickle
from datetime import datetime
from typing import Optional

import torch

STATE_VERSION = 2
STATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".twin_state")
TWIN_A_FILE = os.path.join(STATE_DIR, "twin_a.pt")
TWIN_B_FILE = os.path.join(STATE_DIR, "twin_b.pt")
META_FILE = os.path.join(STATE_DIR, "meta.json")
BUFFER_FILE = os.path.join(STATE_DIR, "buffers.pkl")


def _ensure_dir():
    os.makedirs(STATE_DIR, exist_ok=True)


def _load_torch_state(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _read_meta() -> Optional[dict]:
    if not os.path.exists(META_FILE):
        return None
    try:
        with open(META_FILE) as f:
            return json.load(f)
    except Exception:
        return None


def save_twins(twin_a, twin_b, source: str = "simulation"):
    _ensure_dir()

    torch.save(twin_a.model.state_dict(), TWIN_A_FILE)
    torch.save(twin_b.model.state_dict(), TWIN_B_FILE)

    with open(BUFFER_FILE, "wb") as f:
        pickle.dump(
            {
                "a_seqs": list(twin_a.replay.sequences),
                "a_labels": list(twin_a.replay.labels),
                "b_seqs": list(twin_b.replay.sequences),
                "b_labels": list(twin_b.replay.labels),
            },
            f,
        )

    meta = {
        "state_version": STATE_VERSION,
        "saved_at": datetime.now().isoformat(),
        "source": source,
        "twin_a_version": twin_a.version,
        "twin_b_version": twin_b.version,
        "twin_a_step": twin_a.step,
        "twin_b_step": twin_b.step,
        "twin_a_retrains": twin_a.total_retrains(),
        "twin_b_retrains": twin_b.total_retrains(),
        "twin_a_acc": round(twin_a.rolling_accuracy(), 4),
        "twin_b_acc": round(twin_b.rolling_accuracy(), 4),
        "twin_a_bias": round(twin_a.buy_bias(), 4),
        "twin_b_bias": round(twin_b.buy_bias(), 4),
        "twin_a_replay_balance": twin_a.replay.class_balance(),
        "twin_b_replay_balance": twin_b.replay.class_balance(),
        "twin_a_retrains_log": [
            {
                "step": e.step,
                "trigger": e.trigger,
                "acc_before": e.acc_before,
                "acc_after": e.acc_after,
            }
            for e in twin_a.retrain_events
        ],
        "twin_b_retrains_log": [
            {
                "step": e.step,
                "trigger": e.trigger,
                "acc_before": e.acc_before,
                "acc_after": e.acc_after,
            }
            for e in twin_b.retrain_events
        ],
    }
    with open(META_FILE, "w") as f:
        json.dump(meta, f, indent=2)

    return meta


def load_twins(twin_a, twin_b) -> Optional[dict]:
    if not state_exists():
        return None

    try:
        twin_a.model.load_state_dict(_load_torch_state(TWIN_A_FILE))
        twin_b.model.load_state_dict(_load_torch_state(TWIN_B_FILE))
        twin_a.model.eval()
        twin_b.model.eval()

        with open(BUFFER_FILE, "rb") as f:
            buf = pickle.load(f)

        from collections import deque

        twin_a.replay.sequences = deque(buf.get("a_seqs", []), maxlen=twin_a.replay.maxlen)
        twin_a.replay.labels = deque(buf.get("a_labels", []), maxlen=twin_a.replay.maxlen)
        twin_b.replay.sequences = deque(buf.get("b_seqs", []), maxlen=twin_b.replay.maxlen)
        twin_b.replay.labels = deque(buf.get("b_labels", []), maxlen=twin_b.replay.maxlen)

        meta = _read_meta() or {}
        twin_a.version = meta.get("twin_a_version", 1)
        twin_b.version = meta.get("twin_b_version", 1)
        return meta
    except Exception:
        return None


def state_exists() -> bool:
    meta = _read_meta()
    if not meta or meta.get("state_version") != STATE_VERSION:
        return False
    return all(os.path.exists(p) for p in [TWIN_A_FILE, TWIN_B_FILE, META_FILE, BUFFER_FILE])


def load_meta() -> Optional[dict]:
    meta = _read_meta()
    if not meta or meta.get("state_version") != STATE_VERSION:
        return None
    return meta


def delete_state():
    for f in [TWIN_A_FILE, TWIN_B_FILE, META_FILE, BUFFER_FILE]:
        if os.path.exists(f):
            os.remove(f)
