
from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
for path in (CURRENT_DIR, PARENT_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from data_generator import SyntheticXAUUSD
try:
    from live_feed import LiveFeedAdapter
except ModuleNotFoundError:
    # Support running the file directly from the dissertation_v3 folder.
    from live_feed import LiveFeedAdapter
from mt5_bridge.mt5_connector import MT5Connector
from mt5_bridge.mt5_trade_engine import MT5TradeEngine
from mt5_bridge.paper_engine import PaperTradingEngine
from features import FeatureEngine, SequenceBuffer
from feedback import OutcomeEvaluator
from twin_state import load_twins, save_twins, state_exists
from utils import to_csv_bytes

try:
    from models import build_twins
except Exception:
    build_twins = None


APP_NAME = "Twin Learning Execution"
APP_TAGLINE = "Institutional-grade twin learning execution, diagnostics, and market intelligence."
LIVE_CANDLE_SECONDS = 5


@dataclass
class RuntimeInit:
    connector: MT5Connector
    engine: Any
    feed: LiveFeedAdapter
    twin_a: Any
    twin_b: Any
    feat_engine: FeatureEngine
    seq_buffer: SequenceBuffer
    eval_a: OutcomeEvaluator
    eval_b: OutcomeEvaluator
    diagnostics: Dict[str, Any]
    requested_mode: str
    actual_mode: str
    note: str
    connected: bool
    connected_at: datetime
    runtime_step: int = 0


def _blank_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["time", "open", "high", "low", "close", "source"])


def _signal_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["time", "twin", "signal", "score", "confidence", "price", "source", "reason"])


def ensure_session_defaults() -> None:
    defaults = {
        "theme_mode": "Dark",
        "show_broker_credentials": False,
        "mt5_login": "",
        "mt5_password": "",
        "mt5_server": "ICMarketsSC-Demo",
        "terminal_path_ui": r"C:\Program Files\MetaTrader 5 IC Markets Global\terminal64.exe",
        "preferred_symbol_input": "XAUUSD",
        "requested_mode_ui": "Live MT5",
        "allow_live_ui": True,
        "confirm_live_ui": True,
        "lot_size_ui": 0.01,
        "sl_pips_ui": 150.0,
        "tp_pips_ui": 250.0,
        "ticks_per_refresh_ui": 1,
        "autorun_sleep_ui": 0.35,
        "autorun_ui": False,
        "deploy_count": 0,
        "deploy_history": [],
        "live_init": None,
        "ticks_df": _blank_df(),
        "signals_df": _signal_df(),
        "twin_a_curve": pd.DataFrame(columns=["time", "value"]),
        "twin_b_curve": pd.DataFrame(columns=["time", "value"]),
        "last_force_result": "",
        "paper_balance": 10000.0,
        "manual_a_conf": 0.50,
        "manual_b_conf": 0.50,
        "page_notice": "",
        "execution_audit": [],
        "preferred_symbol_runtime": "XAUUSD",
        "fallback_generator": SyntheticXAUUSD(seed=77),
        "last_twin_signals": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def theme_tokens() -> Dict[str, str]:
    if st.session_state.theme_mode == "Light":
        return {
            "bg": "#F4F8FF",
            "bg2": "#EAF2FF",
            "card": "rgba(255,255,255,0.92)",
            "text": "#0A1A33",
            "muted": "#58708F",
            "border": "rgba(67,118,196,0.20)",
            "accent": "#5AA8FF",
            "accent2": "#7A7DFF",
            "success": "#22C55E",
            "warning": "#F59E0B",
            "danger": "#EF4444",
            "chart_bg": "rgba(234,242,255,0.88)",
            "chip": "rgba(255,255,255,0.75)",
        }
    return {
        "bg": "#050D1A",
        "bg2": "#071428",
        "card": "linear-gradient(180deg, rgba(8,18,34,0.96), rgba(5,12,24,0.94))",
        "text": "#EAF2FF",
        "muted": "#9DB2D1",
        "border": "rgba(100,170,255,0.22)",
        "accent": "#59C2FF",
        "accent2": "#7B7DFF",
        "success": "#22C55E",
        "warning": "#F59E0B",
        "danger": "#FF5C7A",
        "chart_bg": "rgba(8,18,34,0.55)",
        "chip": "rgba(9,18,35,0.76)",
    }


def inject_css() -> None:
    t = theme_tokens()
    st.markdown(
        f"""
        <style>
        [data-testid="stHeader"], #MainMenu, footer {{visibility:hidden; height:0;}}
        .block-container {{padding-top: 1.05rem; padding-bottom: 1rem; max-width: 98rem;}}
        .stApp {{
            background:
              radial-gradient(circle at top right, rgba(89,194,255,0.08), transparent 24%),
              radial-gradient(circle at top left, rgba(123,125,255,0.07), transparent 18%),
              {t["bg"]};
            color: {t["text"]};
        }}
        section[data-testid="stSidebar"] {{
            background:
              linear-gradient(180deg, rgba(9,18,35,0.98), rgba(4,10,20,0.98));
            border-right: 1px solid {t["border"]};
        }}
        div[data-testid="stExpander"] {{
            border: 1px solid {t["border"]};
            border-radius: 18px;
            background: rgba(7,17,31,0.58);
            overflow: hidden;
        }}
        .stTabs [data-baseweb="tab-list"] {{
            gap: .5rem;
            padding: .25rem;
            border-radius: 18px;
            background: rgba(8,18,34,0.6);
            border: 1px solid {t["border"]};
            margin-bottom: 1rem;
        }}
        .stTabs [data-baseweb="tab"] {{
            height: auto;
            border-radius: 14px;
            padding: .7rem 1rem;
            color: {t["muted"]};
        }}
        .stTabs [aria-selected="true"] {{
            background: linear-gradient(135deg, rgba(89,194,255,.18), rgba(123,125,255,.14)) !important;
            color: {t["text"]} !important;
            border: 1px solid rgba(89,194,255,.22);
        }}
        .tle-card {{
            border-radius: 24px;
            padding: 1rem 1.15rem;
            border: 1px solid {t["border"]};
            background: {t["card"]};
            box-shadow: 0 12px 32px rgba(0,0,0,0.16);
            backdrop-filter: blur(8px);
        }}
        .tle-card-tight {{
            border-radius: 18px;
            padding: .9rem 1rem;
            border: 1px solid {t["border"]};
            background: rgba(7,17,31,0.72);
        }}
        .tle-metric {{
            min-height: 118px;
        }}
        .tle-grid-two {{
            display:grid;
            grid-template-columns: 2.1fr 1fr;
            gap: 16px;
        }}
        .tle-stack {{
            display:flex;
            flex-direction:column;
            gap: 12px;
        }}
        .tle-status-dot {{
            width: 10px;
            height: 10px;
            border-radius: 999px;
            display: inline-block;
            margin-right: .5rem;
        }}
        .tle-eyebrow {{
            color: {t["accent"]};
            font-size: .74rem;
            text-transform: uppercase;
            letter-spacing: .22em;
            margin-bottom: .55rem;
            font-weight: 700;
        }}
        .tle-banner-ok {{
            border-radius: 18px;
            padding: .92rem 1rem;
            border: 1px solid rgba(34,197,94,.35);
            background: linear-gradient(90deg, rgba(10,40,26,.96), rgba(6,22,18,.92));
        }}
        .tle-banner-warn {{
            border-radius: 18px;
            padding: .92rem 1rem;
            border: 1px solid rgba(245,158,11,.35);
            background: linear-gradient(90deg, rgba(44,28,8,.96), rgba(28,18,8,.92));
        }}
        .tle-banner-danger {{
            border-radius: 18px;
            padding: .92rem 1rem;
            border: 1px solid rgba(255,92,122,.35);
            background: linear-gradient(90deg, rgba(48,14,22,.96), rgba(28,10,16,.92));
        }}
        .tle-chip {{
            display:inline-block;
            margin:.20rem .35rem .20rem 0;
            padding:.48rem .84rem;
            border-radius:999px;
            border:1px solid {t["border"]};
            background: {t["chip"]};
            font-size:.84rem;
            color: {t["text"]};
        }}
        .tle-logo {{
            width: 48px;
            height: 48px;
            border-radius: 16px;
            display:flex;
            align-items:center;
            justify-content:center;
            font-weight: 800;
            font-size: 1.15rem;
            color: white;
            background: linear-gradient(135deg, {t["accent"]}, {t["accent2"]});
            box-shadow: 0 10px 24px rgba(89,194,255,.22);
        }}
        .tle-title {{
            font-size: 2rem;
            font-weight: 800;
            line-height: 1.02;
            margin-bottom: .25rem;
        }}
        .tle-sub {{
            color: {t["muted"]};
            font-size: .95rem;
        }}
        .tle-section {{
            font-size: 1.15rem;
            font-weight: 700;
            margin: .2rem 0 .75rem 0;
        }}
        .tle-section-sub {{
            color: {t["muted"]};
            font-size: .9rem;
            margin-bottom: .8rem;
        }}
        .tle-kv {{
            display:flex;
            justify-content:space-between;
            gap: 12px;
            border-bottom:1px solid rgba(125,145,180,0.10);
            padding:.46rem 0;
        }}
        .tle-kv:last-child {{
            border-bottom:none;
        }}
        .tle-small {{
            color:{t["muted"]};
            font-size:.83rem;
        }}
        .tle-sidebar-title {{
            font-size: 1.2rem;
            font-weight: 800;
            margin-bottom: .25rem;
        }}
        .tle-sidebar-sub {{
            color:{t["muted"]};
            font-size: .85rem;
            margin-bottom: 1rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def reset_runtime_state() -> None:
    st.session_state.live_init = None
    st.session_state.ticks_df = _blank_df()
    st.session_state.signals_df = _signal_df()
    st.session_state.twin_a_curve = pd.DataFrame(columns=["time", "value"])
    st.session_state.twin_b_curve = pd.DataFrame(columns=["time", "value"])
    st.session_state.last_force_result = ""
    st.session_state.execution_audit = []
    st.session_state.page_notice = ""
    st.session_state.last_twin_signals = {}
    st.session_state.manual_a_conf = 0.50
    st.session_state.manual_b_conf = 0.50


def prime_runtime_state(connector: MT5Connector) -> None:
    s = st.session_state
    s.ticks_df = _blank_df()
    s.signals_df = _signal_df()
    s.twin_a_curve = pd.DataFrame(columns=["time", "value"])
    s.twin_b_curve = pd.DataFrame(columns=["time", "value"])
    s.last_force_result = ""
    s.last_twin_signals = {}
    s.manual_a_conf = 0.50
    s.manual_b_conf = 0.50

    bars: List[Dict[str, Any]] = []
    try:
        bars = connector.get_recent_bars(180)
    except Exception:
        bars = []

    if bars:
        df = pd.DataFrame(bars)
        if not df.empty:
            df["source"] = f"MT5 | {connector.symbol or s.preferred_symbol_input}"
            s.ticks_df = df[["time", "open", "high", "low", "close", "source"]].tail(240).reset_index(drop=True)
            return

    tick = connector.get_current_tick()
    if tick is not None:
        price = float(getattr(tick, "mid", 0.0) or 0.0)
        if price > 0:
            now = datetime.now().replace(microsecond=0)
            s.ticks_df = pd.DataFrame(
                [
                    {
                        "time": now,
                        "open": price,
                        "high": price,
                        "low": price,
                        "close": price,
                        "source": f"MT5 | {connector.symbol or s.preferred_symbol_input}",
                    }
                ]
            )


def append_audit(message: str) -> None:
    st.session_state.execution_audit.append(
        {"time": datetime.now().strftime("%H:%M:%S"), "message": message}
    )
    st.session_state.execution_audit = st.session_state.execution_audit[-150:]


def _memory_badge() -> str:
    return "Loaded" if state_exists() else "Empty"


def _make_chips(items: List[str]) -> str:
    return "".join(f'<span class="tle-chip">{item}</span>' for item in items)


def _notice_box_html(message: str, level: str = "info") -> str:
    klass = "tle-banner-ok"
    if level == "warn":
        klass = "tle-banner-warn"
    elif level == "danger":
        klass = "tle-banner-danger"
    return f'<div class="{klass}">{message}</div>'


def section_intro(title: str, subtitle: str = "") -> None:
    st.markdown(
        f"""
        <div style="margin:.15rem 0 .85rem 0;">
            <div class="tle-section">{title}</div>
            {f'<div class="tle-section-sub">{subtitle}</div>' if subtitle else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def current_runtime() -> Optional[RuntimeInit]:
    return st.session_state.live_init


def current_mode_truth() -> Tuple[str, str]:
    rt = current_runtime()
    if not rt:
        return st.session_state.requested_mode_ui, "Paper"
    return rt.requested_mode, rt.actual_mode


def build_engine(
    connector: MT5Connector,
    requested_mode: str,
    allow_live: bool,
    lot_size: float,
    sl_pips: float,
    tp_pips: float,
) -> Tuple[Any, str, str]:
    if requested_mode == "Live MT5":
        if not connector.connected:
            note = "Live requested, but MT5 connection failed — paper fallback active."
            return PaperTradingEngine(starting_balance=10000.0), "Paper", note
        if not allow_live:
            note = "Live requested, but confirmation box was not ticked — paper fallback active."
            return PaperTradingEngine(starting_balance=10000.0), "Paper", note
        if not connector.can_trade():
            note = "Live requested, but MT5 trade permissions are disabled — paper fallback active."
            return PaperTradingEngine(starting_balance=10000.0), "Paper", note

        engine = MT5TradeEngine(
            connector=connector,
            lot_size=lot_size,
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            allow_live=True,
        )
        note = f"Real broker routing is active. Orders should appear in MT5 under {connector.symbol or 'XAUUSD'} with broker tickets."
        return engine, "Live MT5", note

    return PaperTradingEngine(starting_balance=10000.0), "Paper", "Paper engine active."


def _build_runtime_twins(connector: MT5Connector) -> Tuple[Any, Any, FeatureEngine, SequenceBuffer]:
    twin_a = twin_b = None
    if build_twins is not None:
        twin_a, twin_b = build_twins()
        if state_exists():
            try:
                load_twins(twin_a, twin_b)
            except Exception:
                pass

    feat_engine = FeatureEngine(maxlen=600)
    seq_buffer = SequenceBuffer()
    warm_prices = []

    try:
        recent = connector.get_recent_prices(180)
        if recent is not None:
            warm_prices = [float(x) for x in recent.tolist()]
    except Exception:
        warm_prices = []

    if not warm_prices:
        gen = SyntheticXAUUSD(seed=91)
        warm_prices = [float(gen.next_price()["price"]) for _ in range(180)]

    for price in warm_prices:
        feat = feat_engine.update(price)
        if feat is not None:
            seq_buffer.push(feat)

    return twin_a, twin_b, feat_engine, seq_buffer


def initialise(
    login: str,
    password: str,
    server: str,
    preferred_symbol: str,
    terminal_path: str,
    requested_mode: str,
    allow_live: bool,
    lot_size: float,
    sl_pips: float,
    tp_pips: float,
) -> None:
    s = st.session_state

    with st.spinner("Connecting to MT5..."):
        conn = MT5Connector(
            login=int(login) if str(login).strip() else None,
            password=password or None,
            server=server or None,
            preferred_symbol=preferred_symbol or "XAUUSD",
            terminal_path=terminal_path or None,
        )
        connected = conn.connect()

    twin_a, twin_b, feat_engine, seq_buffer = _build_runtime_twins(conn)

    engine, actual_mode, note = build_engine(
        connector=conn,
        requested_mode=requested_mode,
        allow_live=allow_live,
        lot_size=lot_size,
        sl_pips=sl_pips,
        tp_pips=tp_pips,
    )

    feed = LiveFeedAdapter(connector=conn, use_synthetic_fallback=True)
    diagnostics = {
        "connected": connected,
        "status": getattr(conn, "status_msg", ""),
        "requested_mode": requested_mode,
        "actual_mode": actual_mode,
        "symbol": getattr(conn, "symbol", preferred_symbol or "XAUUSD"),
        "trade_allowed": bool(conn.can_trade()) if connected else False,
        "account": conn.account_summary() if connected else {},
        "terminal": conn.terminal_summary() if connected else {},
    }

    s.live_init = RuntimeInit(
        connector=conn,
        engine=engine,
        feed=feed,
        twin_a=twin_a,
        twin_b=twin_b,
        feat_engine=feat_engine,
        seq_buffer=seq_buffer,
        eval_a=OutcomeEvaluator(),
        eval_b=OutcomeEvaluator(),
        diagnostics=diagnostics,
        requested_mode=requested_mode,
        actual_mode=actual_mode,
        note=note,
        connected=connected,
        connected_at=datetime.now(),
    )
    prime_runtime_state(conn)
    s.deploy_count += 1
    s.preferred_symbol_runtime = diagnostics["symbol"]
    s.deploy_history.append(
        {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "requested_mode": requested_mode,
            "actual_mode": actual_mode,
            "server": server,
            "symbol": diagnostics["symbol"],
            "terminal_path": terminal_path,
            "status": getattr(conn, "status_msg", ""),
        }
    )
    s.deploy_history = s.deploy_history[-25:]
    append_audit(f"CONNECT & DEPLOY → {diagnostics['status'] or actual_mode}")


def submit_force_order(direction: str) -> str:
    s = st.session_state
    rt = current_runtime()
    if rt is None:
        return "Live system not initialised."
    if not hasattr(rt, "engine") or rt.engine is None:
        return "MT5 engine not attached."
    engine = rt.engine
    try:
        if direction.upper() == "BUY":
            result = engine.force_buy()
        else:
            result = engine.force_sell()
        ticket = getattr(result, "position_ticket", None) or getattr(result, "broker_ticket", None)
        msg = f"Order sent successfully: {direction.upper()} ticket={ticket}"
        append_audit(msg)
        return msg
    except Exception as exc:
        msg = f"Execution error: {exc}"
        append_audit(msg)
        return msg


def submit_flatten_all() -> str:
    rt = current_runtime()
    if rt is None or not hasattr(rt, "engine") or rt.engine is None:
        return "MT5 engine not attached."
    try:
        current_price = latest_price_value()
        rt.engine.close_all(current_price)
        msg = "Flatten all submitted."
        append_audit(msg)
        return msg
    except Exception as exc:
        msg = f"Flatten failed: {exc}"
        append_audit(msg)
        return msg


def latest_price_value() -> float:
    df = st.session_state.ticks_df
    if df.empty:
        return 0.0
    return safe_float(df.iloc[-1]["close"])


def _signal_score(price: float) -> float:
    df = st.session_state.ticks_df
    if len(df) < 6:
        return 0.0
    close = df["close"].astype(float)
    short = close.tail(5).mean()
    long = close.tail(min(20, len(close))).mean()
    denom = max(long, 1e-9)
    return (short - long) / denom


def _signal_label(score: float) -> str:
    if score > 0.0009:
        return "BUY"
    if score < -0.0009:
        return "SELL"
    return "HOLD"


def fetch_price_tick() -> Tuple[float, str]:
    rt = current_runtime()
    if rt and rt.connector and getattr(rt.connector, "using_live", False):
        tick = rt.connector.get_current_tick()
        if tick is not None:
            mid = getattr(tick, "mid", None)
            if mid is None:
                bid = getattr(tick, "bid", None)
                ask = getattr(tick, "ask", None)
                if bid is not None and ask is not None:
                    mid = (float(bid) + float(ask)) / 2.0
            if mid is not None:
                return float(mid), f"MT5 | {rt.connector.symbol or 'XAUUSD'}"

        last_live = latest_price_value()
        if last_live > 0:
            return float(last_live), f"MT5 | {rt.connector.symbol or 'XAUUSD'} (stale)"

    if rt and rt.feed is not None:
        tick = rt.feed.get_current_tick()
        if tick is not None:
            mid = getattr(tick, "mid", None)
            if mid is None:
                bid = getattr(tick, "bid", None)
                ask = getattr(tick, "ask", None)
                if bid is not None and ask is not None:
                    mid = (float(bid) + float(ask)) / 2.0
            if mid is not None:
                if rt.connector and getattr(rt.connector, "using_live", False):
                    return float(mid), f"MT5 · {rt.connector.symbol or 'XAUUSD'}"
                return float(mid), "Synthetic Fallback"

    gen = st.session_state.fallback_generator
    row = gen.next_price()
    return float(row["price"]), "Synthetic Fallback"


def append_tick(price: float, source: str) -> None:
    s = st.session_state
    now = datetime.now().replace(microsecond=0)
    df = s.ticks_df.copy()

    if df.empty:
        df = pd.DataFrame(
            [
                {
                    "time": now,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "source": source,
                }
            ]
        )
    else:
        last_time = pd.Timestamp(df.iloc[-1]["time"]).to_pydatetime()
        if (now - last_time).total_seconds() >= LIVE_CANDLE_SECONDS:
            row = {
                "time": now,
                "open": safe_float(df.iloc[-1]["close"]),
                "high": price,
                "low": price,
                "close": price,
                "source": source,
            }
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            idx = df.index[-1]
            df.loc[idx, "high"] = max(safe_float(df.loc[idx, "high"]), price)
            df.loc[idx, "low"] = min(safe_float(df.loc[idx, "low"]), price)
            df.loc[idx, "close"] = price
            df.loc[idx, "source"] = source

    s.ticks_df = df.tail(400).reset_index(drop=True)


def _canonical_twin_name(twin: Any) -> str:
    if twin is None:
        return "Twin"
    return getattr(twin, "trade_tag", str(getattr(twin, "name", "Twin")).split()[0])


def _latest_signal_for(twin_name: str) -> Optional[Dict[str, Any]]:
    signal_map = st.session_state.last_twin_signals
    return signal_map.get(twin_name)


def update_twin_telemetry(price: float, source: str) -> None:
    s = st.session_state
    rt = current_runtime()
    now = datetime.now()

    if rt is None or rt.twin_a is None or rt.twin_b is None:
        score = _signal_score(price)
        signal = _signal_label(score)
        a_prev = safe_float(s.manual_a_conf, 0.5)
        b_prev = safe_float(s.manual_b_conf, 0.5)
        a_val = max(0.0, min(1.0, a_prev * 0.85 + (0.5 + score * 80) * 0.15))
        b_val = max(0.0, min(1.0, b_prev * 0.85 + (0.5 - score * 70) * 0.15))
        s.manual_a_conf = a_val
        s.manual_b_conf = b_val
        s.twin_a_curve = pd.concat(
            [s.twin_a_curve, pd.DataFrame([{"time": now, "value": a_val}])],
            ignore_index=True,
        ).tail(240)
        s.twin_b_curve = pd.concat(
            [s.twin_b_curve, pd.DataFrame([{"time": now, "value": b_val}])],
            ignore_index=True,
        ).tail(240)
        s.signals_df = pd.concat(
            [
                s.signals_df,
                pd.DataFrame(
                    [
                        {
                            "time": now,
                            "twin": "Consensus",
                            "signal": signal,
                            "score": float(score),
                            "confidence": 0.5,
                            "price": float(price),
                            "source": source,
                            "reason": "fallback score",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        ).tail(240)
        return

    feat = rt.feat_engine.update(price)
    if feat is None:
        return
    rt.seq_buffer.push(feat)
    if not rt.seq_buffer.ready():
        return

    rt.runtime_step += 1
    seq = rt.seq_buffer.get_sequence()
    regime = source.replace("MT5 | ", "").replace("MT5 Â· ", "") if source.startswith("MT5") else "LiveFlow"
    signal_rows = []

    for twin, curve_key, evalr in [
        (rt.twin_a, "twin_a_curve", rt.eval_a),
        (rt.twin_b, "twin_b_curve", rt.eval_b),
    ]:
        twin_name = _canonical_twin_name(twin)
        direction, confidence, epistemic, entropy, acted = twin.predict(seq)
        signal = "BUY" if acted and direction == 1 else "SELL" if acted else "ABSTAIN"
        score = confidence if direction == 1 else -confidence
        reason = (
            f"act conf={confidence:.3f} H={entropy:.3f}"
            if acted
            else f"abstain conf={confidence:.3f} H={entropy:.3f}"
        )

        s.manual_a_conf = confidence if twin_name == "Twin-A" else s.manual_a_conf
        s.manual_b_conf = confidence if twin_name == "Twin-B" else s.manual_b_conf
        s[curve_key] = pd.concat(
            [s[curve_key], pd.DataFrame([{"time": now, "value": float(confidence)}])],
            ignore_index=True,
        ).tail(240)

        signal_rows.append(
            {
                "time": now,
                "twin": twin_name,
                "signal": signal,
                "score": float(score),
                "confidence": float(confidence),
                "price": float(price),
                "source": source,
                "reason": reason,
            }
        )
        s.last_twin_signals[twin_name] = signal_rows[-1]

        evalr.register(
            rt.runtime_step,
            twin_name,
            direction,
            confidence,
            epistemic,
            entropy,
            acted,
            price,
            seq.copy(),
            regime,
        )

        if hasattr(rt.engine, "get_open_position"):
            open_order = rt.engine.get_open_position(twin_name)
        else:
            open_order = None

        intended_direction = "BUY" if direction == 1 else "SELL"
        if acted:
            if open_order is not None and getattr(open_order, "direction", "") != intended_direction:
                if hasattr(rt.engine, "close_twin"):
                    rt.engine.close_twin(twin_name, price)
                    append_audit(f"{twin_name} reversed {getattr(open_order, 'direction', '')} -> {intended_direction}")
                open_order = None

            if open_order is None and hasattr(rt.engine, "place_order"):
                order = rt.engine.place_order(
                    intended_direction,
                    price,
                    twin_name,
                    confidence,
                    entropy,
                    regime,
                )
                if order is not None:
                    append_audit(f"{twin_name} {intended_direction} @ {price:.2f} conf={confidence:.3f}")

        for record, record_seq in evalr.resolve(rt.runtime_step, price):
            twin.record_outcome(
                record_seq,
                record.direction,
                record.confidence,
                record.epistemic,
                record.entropy,
                record.acted,
                record.correct if record.acted else None,
                record.price_out,
                record.regime,
                actual_label=record.actual_label,
            )

    if signal_rows:
        s.signals_df = pd.concat(
            [s.signals_df, pd.DataFrame(signal_rows)],
            ignore_index=True,
        ).tail(400)


def step_runtime() -> None:
    ticks_per_refresh = int(st.session_state.ticks_per_refresh_ui)
    for _ in range(max(1, ticks_per_refresh)):
        price, source = fetch_price_tick()
        append_tick(price, source)
        update_twin_telemetry(price, source)

    rt = current_runtime()
    if rt and hasattr(rt.engine, "update_prices"):
        try:
            rt.engine.update_prices(latest_price_value())
        except Exception:
            pass


def render_header() -> None:
    s = st.session_state
    rt = current_runtime()
    requested, actual = current_mode_truth()
    source = "Synthetic Fallback"
    symbol = s.preferred_symbol_runtime or s.preferred_symbol_input
    account_text = "—"
    broker_text = "—"
    server_text = s.mt5_server or "—"

    if rt:
        symbol = getattr(rt.connector, "symbol", symbol) or symbol
        if rt.connected:
            source = f"MT5 · {symbol}"
            account_summary = rt.connector.account_summary()
            account_text = "DEMO" if account_summary else "—"
            terminal_summary = rt.connector.terminal_summary()
            broker_text = terminal_summary.get("company", "—") if terminal_summary else "—"
        else:
            source = "Synthetic Fallback"

    chips = [
        f"Source · {source}",
        f"Requested · {requested}",
        f"Actual · {actual}",
        f"Symbol · {symbol}",
        f"Account · {account_text}",
        f"Broker · {broker_text}",
        f"Server · {server_text}",
        f"Memory · {_memory_badge()}",
        f"Time · {datetime.now().strftime('%d %b %Y %H:%M:%S')}",
        f"Deploys · {s.deploy_count}",
    ]

    st.markdown(
        f"""
        <div class="tle-card">
            <div style="display:flex; gap:14px; align-items:flex-start;">
                <div class="tle-logo">TLE</div>
                <div style="flex:1;">
                    <div class="tle-title">{APP_NAME}</div>
                    <div class="tle-sub">{APP_TAGLINE}</div>
                    <div style="margin-top:.45rem; color:#9DB2D1; font-size:.92rem;">
                        Model stack · 2-layer LSTM + temporal attention + confidence / uncertainty heads ·
                        Twin-A and Twin-B adapt online from resolved outcomes only.
                    </div>
                    <div style="margin-top:.7rem;">{_make_chips(chips)}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if rt:
        if rt.actual_mode == "Live MT5":
            st.markdown(_notice_box_html(rt.note, "info"), unsafe_allow_html=True)
        elif rt.requested_mode == "Live MT5":
            st.markdown(_notice_box_html(rt.note, "danger"), unsafe_allow_html=True)
        else:
            st.markdown(_notice_box_html(rt.note, "warn"), unsafe_allow_html=True)


def render_metric_cards() -> None:
    s = st.session_state
    rt = current_runtime()
    price = latest_price_value()
    spread = 0.0
    tick_age = "n/a"
    conn_label = "PAPER"

    if rt and rt.connector:
        tick = rt.connector.get_current_tick()
        if tick is not None:
            bid = getattr(tick, "bid", None)
            ask = getattr(tick, "ask", None)
            if bid is not None and ask is not None:
                spread = float(ask) - float(bid)
            ts = getattr(tick, "timestamp", None)
            if ts is not None:
                try:
                    if isinstance(ts, datetime):
                        age = datetime.now() - ts
                    else:
                        age = datetime.now() - datetime.fromtimestamp(float(ts))
                    tick_age = f"{max(age.total_seconds(), 0.0):.1f}s"
                except Exception:
                    tick_age = "n/a"

    if rt:
        conn_label = "LIVE MT5" if rt.actual_mode == "Live MT5" else "PAPER"

    engine = rt.engine if rt else None
    closed_pnl = safe_float(engine.total_pnl(), 0.0) if engine and hasattr(engine, "total_pnl") else 0.0
    equity = safe_float(engine.equity(price), 10000.0) if engine and hasattr(engine, "equity") else safe_float(s.paper_balance, 10000.0)
    sharpe = safe_float(engine.sharpe_ratio(), 0.0) if engine and hasattr(engine, "sharpe_ratio") else 0.0
    max_dd = safe_float(engine.max_drawdown(), 0.0) if engine and hasattr(engine, "max_drawdown") else 0.0
    open_count = len(getattr(engine, "_open", {})) if engine else 0

    metrics = [
        ("PRICE", f"{price:,.2f}", symbol_label()),
        ("EQUITY", f"${equity:,.2f}", f"Balance ${equity:,.2f}"),
        ("CLOSED PNL", f"${closed_pnl:+,.2f}", f"Win rate {safe_float(engine.win_rate(), 0.0):.1%}" if engine and hasattr(engine, "win_rate") else "Win rate n/a"),
        ("SPREAD", f"{spread:.5f}", "Latency n/a"),
        ("TICK AGE", tick_age, f"Open orders {open_count}"),
        ("SHARPE", f"{sharpe:.3f}", f"Max DD {max_dd:.4f}"),
        ("CONNECTION", conn_label, getattr(rt.connector, "status_msg", "Not connected") if rt else "Not connected"),
    ]

    cols = st.columns(len(metrics))
    for col, (label, value, sub) in zip(cols, metrics):
        with col:
            st.markdown(
                f"""
                <div class="tle-card tle-metric">
                    <div class="tle-small">{label}</div>
                    <div style="font-size:2rem;font-weight:800;margin-top:.1rem;">{value}</div>
                    <div class="tle-small" style="margin-top:.25rem;">{sub}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def symbol_label() -> str:
    rt = current_runtime()
    if rt and rt.connector and rt.connector.symbol:
        return rt.connector.symbol
    return st.session_state.preferred_symbol_input


def plot_price_chart() -> go.Figure:
    t = theme_tokens()
    df = st.session_state.ticks_df.copy()
    fig = go.Figure()
    if not df.empty:
        df["ema_fast"] = df["close"].ewm(span=9, adjust=False).mean()
        df["ema_slow"] = df["close"].ewm(span=21, adjust=False).mean()
        fig.add_trace(
            go.Candlestick(
                x=df["time"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name="XAUUSD",
                increasing_line_color="#22C55E",
                increasing_fillcolor="#22C55E",
                decreasing_line_color="#FF5C7A",
                decreasing_fillcolor="#FF5C7A",
                whiskerwidth=0.4,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df["time"],
                y=df["ema_fast"],
                mode="lines",
                name="EMA 9",
                line=dict(width=1.6, color="#59C2FF"),
                opacity=0.95,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df["time"],
                y=df["ema_slow"],
                mode="lines",
                name="EMA 21",
                line=dict(width=1.2, color="#F59E0B"),
                opacity=0.9,
            )
        )
        rt = current_runtime()
        engine = rt.engine if rt else None
        if engine and hasattr(engine, "orders") and engine.orders:
            buy_x, buy_y, sell_x, sell_y = [], [], [], []
            for o in engine.orders:
                if o.direction == "BUY":
                    buy_x.append(o.timestamp)
                    buy_y.append(o.price_in)
                else:
                    sell_x.append(o.timestamp)
                    sell_y.append(o.price_in)
            if buy_x:
                fig.add_trace(
                    go.Scatter(
                        x=buy_x,
                        y=buy_y,
                        mode="markers",
                        name="BUY",
                        marker=dict(size=11, symbol="triangle-up", color="#22C55E", line=dict(width=1, color="#04111F")),
                    )
                )
            if sell_x:
                fig.add_trace(
                    go.Scatter(
                        x=sell_x,
                        y=sell_y,
                        mode="markers",
                        name="SELL",
                        marker=dict(size=11, symbol="triangle-down", color="#FF5C7A", line=dict(width=1, color="#04111F")),
                    )
                )
        last_row = df.iloc[-1]
        fig.add_hline(
            y=safe_float(last_row["close"]),
            line_width=1,
            line_dash="dot",
            line_color="rgba(234,242,255,0.28)",
        )
    fig.update_layout(
        title="Live Price",
        height=420,
        margin=dict(l=16, r=16, t=38, b=10),
        xaxis_rangeslider_visible=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=t["chart_bg"],
        font=dict(color=t["text"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0),
        hovermode="x unified",
        uirevision="live-price-chart",
        dragmode="pan",
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(125,145,180,0.08)",
            showspikes=True,
            spikemode="across",
            spikecolor="rgba(234,242,255,0.35)",
            spikesnap="cursor",
            rangeslider=dict(visible=False),
            zeroline=False,
        ),
        yaxis=dict(
            side="right",
            showgrid=True,
            gridcolor="rgba(125,145,180,0.10)",
            tickformat=",.2f",
            zeroline=False,
            fixedrange=False,
        ),
    )
    return fig


def plot_signal_chart() -> go.Figure:
    t = theme_tokens()
    fig = go.Figure()
    df = st.session_state.signals_df.copy()
    if not df.empty:
        mapping = {"BUY": 1, "HOLD": 0, "SELL": -1, "ABSTAIN": 0}
        if "twin" in df.columns:
            colors = {"Twin-A": "#FF9B54", "Twin-B": "#4ADE80", "Consensus": "#59C2FF"}
            for twin_name, chunk in df.groupby("twin"):
                fig.add_trace(
                    go.Scatter(
                        x=chunk["time"],
                        y=chunk["signal"].map(mapping).fillna(0),
                        mode="lines+markers",
                        name=twin_name,
                        line=dict(width=2.4, color=colors.get(twin_name, "#59C2FF"), shape="hv"),
                        marker=dict(size=7),
                    )
                )
        else:
            fig.add_trace(
                go.Scatter(
                    x=df["time"],
                    y=df["signal"].map(mapping),
                    mode="lines+markers",
                    name="Signals",
                )
            )
    fig.update_layout(
        title="Signal Tape",
        height=420,
        margin=dict(l=16, r=16, t=38, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=t["chart_bg"],
        font=dict(color=t["text"]),
        hovermode="x unified",
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(125,145,180,0.08)",
            showspikes=True,
            spikemode="across",
            spikecolor="rgba(234,242,255,0.28)",
        ),
        yaxis=dict(
            tickvals=[-1, 0, 1],
            ticktext=["SELL", "FLAT", "BUY"],
            range=[-1.2, 1.2],
            showgrid=True,
            gridcolor="rgba(125,145,180,0.10)",
            zeroline=True,
            zerolinecolor="rgba(234,242,255,0.18)",
        ),
    )
    return fig


def plot_confidence_curve(df: pd.DataFrame, title: str, color: str) -> go.Figure:
    t = theme_tokens()
    fig = go.Figure()
    if not df.empty:
        fig.add_trace(
            go.Scatter(
                x=df["time"],
                y=df["value"],
                mode="lines",
                name=title,
                line=dict(width=2, color=color),
                fill="tozeroy",
                fillcolor=f"rgba({int(color[1:3], 16)},{int(color[3:5], 16)},{int(color[5:7], 16)},0.10)",
            )
        )
    fig.update_layout(
        title=title,
        height=290,
        margin=dict(l=16, r=16, t=38, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=t["chart_bg"],
        font=dict(color=t["text"]),
        yaxis=dict(range=[0, 1]),
        hovermode="x unified",
        xaxis=dict(showgrid=True, gridcolor="rgba(125,145,180,0.08)"),
        yaxis_showgrid=True,
        yaxis_gridcolor="rgba(125,145,180,0.10)",
    )
    return fig


def render_overview_tab() -> None:
    section_intro("Market Overview", "Live pricing, execution posture, and twin confidence at a glance.")
    render_metric_cards()

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_price_chart(), use_container_width=True)
    with c2:
        st.plotly_chart(plot_signal_chart(), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(plot_confidence_curve(st.session_state.twin_a_curve, "Twin-A Confidence", "#FF9B54"), use_container_width=True)
    with c4:
        st.plotly_chart(plot_confidence_curve(st.session_state.twin_b_curve, "Twin-B Confidence", "#4ADE80"), use_container_width=True)


def render_twin_intelligence_tab() -> None:
    section_intro("Twin Intelligence", "Compare directional bias, confidence strength, and internal agreement between both twins.")
    rt = current_runtime()
    a = safe_float(st.session_state.manual_a_conf, 0.5)
    b = safe_float(st.session_state.manual_b_conf, 0.5)
    leader = "Twin-A" if a > b else "Twin-B" if b > a else "Draw"
    consensus = "Consensus" if abs(a - b) < 0.08 else "Disagreement"
    sig_a = _latest_signal_for("Twin-A") or {}
    sig_b = _latest_signal_for("Twin-B") or {}
    twin_a = rt.twin_a if rt and rt.twin_a is not None else None
    twin_b = rt.twin_b if rt and rt.twin_b is not None else None

    cols = st.columns(3)
    data = [
        ("Twin-A", a, "Orange lane"),
        ("Twin-B", b, "Green lane"),
        ("Leader", max(a, b), f"{leader} · {consensus}"),
    ]
    for col, (title, value, sub) in zip(cols, data):
        with col:
            st.markdown(
                f"""
                <div class="tle-card">
                    <div class="tle-small">{title}</div>
                    <div style="font-size:2rem;font-weight:800;">{value:.3f}</div>
                    <div class="tle-small">{sub}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    table = pd.DataFrame(
        [
            {
                "Twin": "Twin-A",
                "Direction Bias": sig_a.get("signal", "ABSTAIN"),
                "Confidence": round(a, 3),
                "Current Trade": "OPEN" if rt and rt.engine and rt.engine.get_open_position("Twin-A") else "FLAT",
                "PnL": "See Orders tab",
                "Accuracy": f"{twin_a.rolling_accuracy():.3f}" if twin_a else "n/a",
                "Trades": twin_a.decisions_made() if twin_a else "n/a",
                "Loss Ratio": f"{1.0 - twin_a.rolling_accuracy():.3f}" if twin_a else "n/a",
                "Regime": sig_a.get("source", "Adaptive"),
                "Learning State": f"v{twin_a.version}" if twin_a else "Online",
            },
            {
                "Twin": "Twin-B",
                "Direction Bias": sig_b.get("signal", "ABSTAIN"),
                "Confidence": round(b, 3),
                "Current Trade": "OPEN" if rt and rt.engine and rt.engine.get_open_position("Twin-B") else "FLAT",
                "PnL": "See Orders tab",
                "Accuracy": f"{twin_b.rolling_accuracy():.3f}" if twin_b else "n/a",
                "Trades": twin_b.decisions_made() if twin_b else "n/a",
                "Loss Ratio": f"{1.0 - twin_b.rolling_accuracy():.3f}" if twin_b else "n/a",
                "Regime": sig_b.get("source", "Adaptive"),
                "Learning State": f"v{twin_b.version}" if twin_b else "Online",
            },
        ]
    )
    st.dataframe(table, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_confidence_curve(st.session_state.twin_a_curve, "Twin-A Decision Timeline", "#FF9B54"), use_container_width=True)
    with c2:
        st.plotly_chart(plot_confidence_curve(st.session_state.twin_b_curve, "Twin-B Decision Timeline", "#4ADE80"), use_container_width=True)


def open_positions_df() -> pd.DataFrame:
    rt = current_runtime()
    if not rt or not hasattr(rt.engine, "_open"):
        return pd.DataFrame()
    rows = []
    current_price = latest_price_value()
    for twin_name, order in getattr(rt.engine, "_open", {}).items():
        pnl = 0.0
        if order.direction == "BUY":
            pnl = current_price - safe_float(order.price_in)
        else:
            pnl = safe_float(order.price_in) - current_price
        rows.append(
            {
                "Twin": twin_name,
                "Dir": order.direction,
                "Entry": f"{safe_float(order.price_in):.2f}",
                "Now": f"{current_price:.2f}",
                "PnL": f"{pnl:+.2f}",
                "Lots": safe_float(order.lot_size),
                "SL": f"{safe_float(order.sl_price):.2f}",
                "TP": f"{safe_float(order.tp_price):.2f}",
                "Regime": order.regime,
                "Broker Ticket": order.position_ticket or order.broker_ticket or "—",
            }
        )
    return pd.DataFrame(rows)


def render_orders_tab() -> None:
    rt = current_runtime()
    if rt is None:
        st.info("No engine initialised yet.")
        return
    section_intro("Execution Desk", "Manual order controls, open exposure, and routing feedback from the active engine.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("🔥 FORCE BUY TEST", use_container_width=True):
            msg = submit_force_order("BUY")
            if "successfully" in msg.lower():
                st.success(msg)
            else:
                st.error(msg)
            st.session_state.last_force_result = msg
    with c2:
        if st.button("🔥 FORCE SELL TEST", use_container_width=True):
            msg = submit_force_order("SELL")
            if "successfully" in msg.lower():
                st.success(msg)
            else:
                st.error(msg)
            st.session_state.last_force_result = msg
    with c3:
        if st.button("🧨 FLATTEN ALL", use_container_width=True):
            msg = submit_flatten_all()
            if "submitted" in msg.lower():
                st.success(msg)
            else:
                st.error(msg)
            st.session_state.last_force_result = msg
    with c4:
        st.markdown(
            f"""
            <div class="tle-card" style="padding:.9rem 1rem;">
                <div class="tle-small">Last Broker Result</div>
                <div style="font-weight:700;margin-top:.2rem;">{st.session_state.last_force_result or '—'}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Open Positions")
    pos_df = open_positions_df()
    if pos_df.empty:
        st.info("No open positions recorded in the dashboard engine.")
    else:
        st.dataframe(pos_df, use_container_width=True, hide_index=True)

    st.markdown("### Order History")
    if hasattr(rt.engine, "recent_orders_df"):
        hist = rt.engine.recent_orders_df(50)
        if hist.empty:
            st.info("No orders recorded yet.")
        else:
            st.dataframe(hist, use_container_width=True, hide_index=True)
    else:
        st.info("Order history not available for current engine.")

    st.markdown("### Execution Audit Log")
    audit_df = pd.DataFrame(st.session_state.execution_audit)
    if audit_df.empty:
        st.info("No execution audit events yet.")
    else:
        st.dataframe(audit_df.iloc[::-1], use_container_width=True, hide_index=True)


def render_broker_tab() -> None:
    rt = current_runtime()
    if rt is None:
        st.info("No broker session initialised yet.")
        return
    section_intro("Broker Diagnostics", "Connection truth, deploy history, and the exact terminal/account details behind this session.")

    conn = rt.connector
    account = conn.account_summary() if rt.connected else {}
    terminal = conn.terminal_summary() if rt.connected else {}

    broker_rows = [
        ("Status", getattr(conn, "status_msg", "—")),
        ("Requested Mode", rt.requested_mode),
        ("Actual Mode", rt.actual_mode),
        ("Terminal Path", st.session_state.terminal_path_ui),
        ("Account Login", st.session_state.mt5_login or "—"),
        ("Server", st.session_state.mt5_server or "—"),
        ("Mode", "DEMO" if account else "—"),
        ("Trade Allowed", str(conn.can_trade())),
        ("Terminal Trade Allowed", str(terminal.get("trade_allowed", "â€”") if terminal else "â€”")),
        ("Broker", terminal.get("company", "—") if terminal else "—"),
        ("Symbol", getattr(conn, "symbol", "—")),
        ("Terminal Name", terminal.get("name", "—") if terminal else "—"),
        ("Terminal Company", terminal.get("company", "—") if terminal else "—"),
        ("Balance", account.get("balance", "—") if account else "—"),
        ("Equity", account.get("equity", "—") if account else "—"),
        ("Margin Free", account.get("margin_free", "—") if account else "—"),
    ]
    st.dataframe(pd.DataFrame(broker_rows, columns=["Field", "Value"]), use_container_width=True, hide_index=True)

    st.markdown("### Deploy History")
    dh = pd.DataFrame(st.session_state.deploy_history)
    if dh.empty:
        st.info("No deploy history yet.")
    else:
        st.dataframe(dh.iloc[::-1], use_container_width=True, hide_index=True)


def render_exports_tab() -> None:
    section_intro("Exports", "Download live ticks and signal traces for external analysis or reporting.")
    st.markdown("### Export Center")
    ticks_df = st.session_state.ticks_df.copy()
    sig_df = st.session_state.signals_df.copy()

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download ticks CSV",
            to_csv_bytes(ticks_df),
            file_name="tle_ticks.csv",
            use_container_width=True,
        )
    with c2:
        st.download_button(
            "Download signals CSV",
            to_csv_bytes(sig_df),
            file_name="tle_signals.csv",
            use_container_width=True,
        )


def render_header_v2() -> None:
    s = st.session_state
    rt = current_runtime()
    requested, actual = current_mode_truth()
    source = "Synthetic Fallback"
    symbol = s.preferred_symbol_runtime or s.preferred_symbol_input
    broker_text = "-"
    server_text = s.mt5_server or "-"

    if rt:
        symbol = getattr(rt.connector, "symbol", symbol) or symbol
        if rt.connected:
            source = f"MT5 | {symbol}"
            terminal_summary = rt.connector.terminal_summary()
            broker_text = terminal_summary.get("company", "-") if terminal_summary else "-"

    chips = [
        f"Source | {source}",
        f"Requested | {requested}",
        f"Actual | {actual}",
        f"Symbol | {symbol}",
        f"Broker | {broker_text}",
        f"Server | {server_text}",
        f"Memory | {_memory_badge()}",
        f"Deploys | {s.deploy_count}",
    ]
    live_color = "#22C55E" if rt and rt.actual_mode == "Live MT5" else "#F59E0B"
    live_label = "Broker Live" if rt and rt.actual_mode == "Live MT5" else "Paper Safety"

    st.markdown(
        f"""
        <div class="tle-grid-two">
            <div class="tle-card">
                <div class="tle-eyebrow">Trading Terminal</div>
                <div style="display:flex; gap:14px; align-items:flex-start;">
                    <div class="tle-logo">TLE</div>
                    <div style="flex:1;">
                        <div class="tle-title">{APP_NAME}</div>
                        <div class="tle-sub">{APP_TAGLINE}</div>
                        <div style="margin-top:.55rem; color:#9DB2D1; font-size:.92rem;">
                            Twin-led execution for MT5 with a cleaner operator flow, clearer broker truth, and faster manual controls.
                        </div>
                        <div style="margin-top:.7rem;">{_make_chips(chips)}</div>
                    </div>
                </div>
            </div>
            <div class="tle-stack">
                <div class="tle-card-tight">
                    <div class="tle-small">Session Mode</div>
                    <div style="font-size:1.35rem;font-weight:800;margin:.25rem 0;">
                        <span class="tle-status-dot" style="background:{live_color};"></span>{live_label}
                    </div>
                    <div class="tle-small">Requested {requested} | Actual {actual}</div>
                </div>
                <div class="tle-card-tight">
                    <div class="tle-small">Venue</div>
                    <div style="font-size:1.08rem;font-weight:700;margin:.25rem 0;">{server_text}</div>
                    <div class="tle-small">{source} | {broker_text}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if rt:
        if rt.actual_mode == "Live MT5":
            st.markdown(_notice_box_html(rt.note, "info"), unsafe_allow_html=True)
        elif rt.requested_mode == "Live MT5":
            st.markdown(_notice_box_html(rt.note, "danger"), unsafe_allow_html=True)
        else:
            st.markdown(_notice_box_html(rt.note, "warn"), unsafe_allow_html=True)


def render_sidebar_v2() -> None:
    s = st.session_state
    with st.sidebar:
        st.markdown(
            """
            <div class="tle-sidebar-title">Control Center</div>
            <div class="tle-sidebar-sub">Broker setup, execution rules, and runtime controls in one place.</div>
            """,
            unsafe_allow_html=True,
        )
        st.selectbox("Theme", ["Dark", "Light"], key="theme_mode")

        with st.expander("Broker Connection", expanded=True):
            st.toggle("Show broker credentials", key="show_broker_credentials")
            st.text_input("Login", key="mt5_login")
            st.text_input("Password", key="mt5_password", type="password")
            st.text_input("Server", key="mt5_server")
            st.text_input("Terminal Path (.exe)", key="terminal_path_ui")
            st.text_input("Preferred Symbol", key="preferred_symbol_input")

        with st.expander("Execution Rules", expanded=True):
            st.selectbox("Execution Mode", ["Paper", "Live MT5"], key="requested_mode_ui")
            st.checkbox("Enable real MT5 order routing", key="allow_live_ui")
            if s.requested_mode_ui == "Live MT5":
                st.checkbox("I confirm I want live broker order routing", key="confirm_live_ui")
            c1, c2 = st.columns(2)
            with c1:
                st.number_input("Lot Size", min_value=0.01, max_value=5.0, step=0.01, key="lot_size_ui")
                st.number_input("Stop Loss", min_value=5.0, max_value=1000.0, step=5.0, key="sl_pips_ui")
            with c2:
                st.number_input("Take Profit", min_value=5.0, max_value=1000.0, step=5.0, key="tp_pips_ui")

        with st.expander("Runtime Controls", expanded=False):
            st.slider("Ticks per refresh", 1, 20, key="ticks_per_refresh_ui")
            st.slider("Auto-run pause (sec)", 0.10, 1.50, key="autorun_sleep_ui")
            st.toggle("Auto-run", key="autorun_ui")
            if st.button("Step +1 Tick", use_container_width=True):
                step_runtime()
                st.rerun()

        with st.expander("State & Tools", expanded=False):
            if build_twins is not None and st.button("Save Twin State", use_container_width=True):
                try:
                    ta, tb = build_twins()
                    save_twins(ta, tb)
                    st.success("Twin state saved.")
                except Exception as exc:
                    st.error(str(exc))

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Connect", use_container_width=True, type="primary"):
                allow_live = bool(
                    s.allow_live_ui
                    and (
                        s.requested_mode_ui != "Live MT5"
                        or s.get("confirm_live_ui", False)
                    )
                )
                initialise(
                    login=s.mt5_login,
                    password=s.mt5_password,
                    server=s.mt5_server,
                    preferred_symbol=s.preferred_symbol_input,
                    terminal_path=s.terminal_path_ui,
                    requested_mode=s.requested_mode_ui,
                    allow_live=allow_live,
                    lot_size=float(s.lot_size_ui),
                    sl_pips=float(s.sl_pips_ui),
                    tp_pips=float(s.tp_pips_ui),
                )
                st.rerun()
        with c2:
            if st.button("Reset", use_container_width=True):
                reset_runtime_state()
                st.rerun()

        if st.session_state.last_force_result:
            st.info(st.session_state.last_force_result)


def render_sidebar() -> None:
    s = st.session_state
    with st.sidebar:
        st.selectbox("Theme", ["Dark", "Light"], key="theme_mode")
        st.markdown("### Broker Login")
        st.toggle("Show broker credentials", key="show_broker_credentials")
        st.text_input("Login", key="mt5_login")
        st.text_input("Password", key="mt5_password", type="password")
        st.text_input("Server", key="mt5_server")
        st.text_input("Terminal Path (.exe)", key="terminal_path_ui")
        st.text_input("Preferred Symbol", key="preferred_symbol_input")

        st.markdown("### Execution")
        st.selectbox("Execution Mode", ["Paper", "Live MT5"], key="requested_mode_ui")
        st.checkbox("Enable real MT5 order routing", key="allow_live_ui")
        if s.requested_mode_ui == "Live MT5":
            st.checkbox("I confirm I want live broker order routing", key="confirm_live_ui")
        st.number_input("Lot Size", min_value=0.01, max_value=5.0, step=0.01, key="lot_size_ui")
        st.number_input("Stop Loss (pips)", min_value=5.0, max_value=1000.0, step=5.0, key="sl_pips_ui")
        st.number_input("Take Profit (pips)", min_value=5.0, max_value=1000.0, step=5.0, key="tp_pips_ui")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("CONNECT & DEPLOY", use_container_width=True):
                allow_live = bool(
                    s.allow_live_ui
                    and (
                        s.requested_mode_ui != "Live MT5"
                        or s.get("confirm_live_ui", False)
                    )
                )
                initialise(
                    login=s.mt5_login,
                    password=s.mt5_password,
                    server=s.mt5_server,
                    preferred_symbol=s.preferred_symbol_input,
                    terminal_path=s.terminal_path_ui,
                    requested_mode=s.requested_mode_ui,
                    allow_live=allow_live,
                    lot_size=float(s.lot_size_ui),
                    sl_pips=float(s.sl_pips_ui),
                    tp_pips=float(s.tp_pips_ui),
                )
                st.rerun()

        with c2:
            if st.button("RESET SESSION", use_container_width=True):
                reset_runtime_state()
                st.rerun()

        st.slider("Ticks per refresh", 1, 20, key="ticks_per_refresh_ui")
        st.slider("Auto-run pause (sec)", 0.10, 1.50, key="autorun_sleep_ui")
        st.toggle("Auto-run", key="autorun_ui")

        st.markdown("### Quick Actions")
        if st.button("+1 TICK", use_container_width=True):
            step_runtime()
            st.rerun()

        if build_twins is not None and st.button("SAVE TWIN STATE", use_container_width=True):
            try:
                ta, tb = build_twins()
                save_twins(ta, tb)
                st.success("Twin state saved.")
            except Exception as exc:
                st.error(str(exc))

        if st.session_state.last_force_result:
            st.info(st.session_state.last_force_result)


def maybe_autorun() -> bool:
    if st.session_state.autorun_ui:
        step_runtime()
        return True
    return False


def finish_autorun_cycle(autorun_active: bool) -> None:
    if not autorun_active:
        return
    time.sleep(float(st.session_state.autorun_sleep_ui))
    st.rerun()


def warm_start_data() -> None:
    if st.session_state.ticks_df.empty:
        for _ in range(30):
            step_runtime()


def main() -> None:
    st.set_page_config(page_title=APP_NAME, layout="wide")
    ensure_session_defaults()
    inject_css()
    render_sidebar_v2()
    autorun_active = maybe_autorun()
    warm_start_data()

    render_header_v2()
    tabs = st.tabs(
        ["Command Center", "Twin Models", "Execution", "Broker", "Exports"]
    )
    with tabs[0]:
        render_overview_tab()
    with tabs[1]:
        render_twin_intelligence_tab()
    with tabs[2]:
        render_orders_tab()
    with tabs[3]:
        render_broker_tab()
    with tabs[4]:
        render_exports_tab()
    finish_autorun_cycle(autorun_active)


if __name__ == "__main__":
    main()
