"""
app.py  –  v4  |  STEP 1: Simulation Training Dashboard

Rewritten to provide a safer, more diagnostic training workflow.
Key improvements added in this version:
1. Robust signal logging so downstream tables never fail on missing keys.
2. Safe dataframe rendering helpers instead of inline conditional expressions.
3. Seed control for reproducible simulations.
4. Batch stepping controls for faster simulation runs.
5. Training health cards with calibration and abstention diagnostics.
6. Signal mix and regime coverage summaries.
7. Recent decision and retrain logs with safe defaults.
8. CSV exports for ticks, signals, and retrains.
9. Readiness / transfer health banner before deployment.
10. Export freshness tracking so you know what the live dashboard will load.
"""

import io
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from twin_state import load_meta, state_exists
from utils import make_fresh_simulation, save_current_twins

# ── Page ──────────────────────────────────────────────────────────── #
st.set_page_config(
    page_title="Twin Training · XAUUSD",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@300;400;500&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;background:#080c14;color:#b8c4d4;}
h1,h2,h3,h4{font-family:'IBM Plex Mono',monospace;}
.stApp{background:#080c14;}
section[data-testid="stSidebar"]{background:#0c1220;border-right:1px solid #182030;}
.mc{background:linear-gradient(135deg,#0f1928,#111e2e);border:1px solid #1c2e42;
    border-radius:10px;padding:13px 17px;margin-bottom:9px;position:relative;overflow:hidden;}
.mc::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;}
.mc.ca::before{background:linear-gradient(90deg,#38bdf8,#0ea5e9);}
.mc.cb::before{background:linear-gradient(90deg,#fb923c,#f97316);}
.mc.cn::before{background:linear-gradient(90deg,#34d399,#10b981);}
.mc.cr::before{background:linear-gradient(90deg,#f87171,#ef4444);}
.mc.cx::before{background:linear-gradient(90deg,#a78bfa,#8b5cf6);}
.mc .lb{font-family:'IBM Plex Mono',monospace;font-size:9px;letter-spacing:2.5px;
        text-transform:uppercase;color:#3d5470;margin-bottom:5px;}
.mc .vl{font-family:'IBM Plex Mono',monospace;font-size:21px;font-weight:600;color:#dde6f0;}
.mc .sb{font-size:10px;color:#3d5470;margin-top:2px;}
.export-box{background:#0a1f12;border:1px solid #1a4a28;border-left:3px solid #22c55e;
            border-radius:8px;padding:13px 16px;margin-bottom:10px;font-size:11px;color:#3a7a4a;}
.warn-box{background:#23130a;border:1px solid #7a3e16;border-left:3px solid #fb923c;
          border-radius:8px;padding:13px 16px;margin-bottom:10px;font-size:11px;color:#b98955;}
.stTabs [data-baseweb="tab-list"]{background:#0c1220;border-bottom:1px solid #182030;}
.stTabs [data-baseweb="tab"]{font-family:'IBM Plex Mono',monospace;font-size:10px;
    letter-spacing:1.5px;color:#3d5470;padding:11px 18px;}
.stTabs [aria-selected="true"]{color:#38bdf8!important;
    border-bottom:2px solid #38bdf8!important;background:transparent!important;}
.stButton>button{font-family:'IBM Plex Mono',monospace;font-size:11px;
    background:#0f1928;color:#38bdf8;border:1px solid #1c3050;
    border-radius:6px;padding:9px 14px;width:100%;}
.stButton>button:hover{background:#182a42;border-color:#38bdf8;}
</style>
""",
    unsafe_allow_html=True,
)

COL_A = "#38bdf8"
COL_B = "#fb923c"
BG = "#080c14"
GR = "#141e2e"
BL = dict(
    paper_bgcolor=BG,
    plot_bgcolor=BG,
    font=dict(family="IBM Plex Mono,monospace", color="#8aa0b8", size=10),
    margin=dict(l=48, r=18, t=34, b=34),
    xaxis=dict(gridcolor=GR, zerolinecolor=GR),
    yaxis=dict(gridcolor=GR, zerolinecolor=GR),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=GR),
)

# ── State ─────────────────────────────────────────────────────────── #
def init() -> None:
    defaults = dict(
        ready=False,
        gen=None,
        ta=None,
        tb=None,
        fe=None,
        sb=None,
        ea=None,
        eb=None,
        ph=[],
        ticks=[],
        da=[],
        db=[],
        step=0,
        last_meta=load_meta(),
        selected_seed=42,
        last_export_step=None,
    )
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def mc(label: str, value: str, sub: str = "", cls: str = "cn") -> str:
    return (
        f'<div class="mc {cls}"><div class="lb">{label}</div>'
        f'<div class="vl">{value}</div><div class="sb">{sub}</div></div>'
    )


def fig(height: int = 300, title: str = "") -> go.Figure:
    f = go.Figure()
    layout = dict(**BL, height=height)
    if title:
        layout["title"] = dict(text=title, font=dict(size=11, color="#6a8aaa"))
    f.update_layout(**layout)
    return f


def safe_show_df(df: pd.DataFrame, empty_message: str) -> None:
    if df is not None and not df.empty:
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.caption(empty_message)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    if df is None or df.empty:
        return b""
    return df.to_csv(index=False).encode("utf-8")


def launch(seed: int) -> None:
    progress = st.progress(0, text="Building synthetic XAUUSD stream…")
    progress.progress(15, text="Pre-training Twin-A and Twin-B on 600 warm-up steps…")
    result = make_fresh_simulation(seed=seed, auto_save=True)
    progress.progress(90, text="Saving trained weights to disk…")
    time.sleep(0.2)
    progress.progress(100, text="Ready.")
    time.sleep(0.25)
    progress.empty()

    generator, ta, tb, fe, sb, ea, eb, ph = result
    s = st.session_state
    s.gen = generator
    s.ta = ta
    s.tb = tb
    s.fe = fe
    s.sb = sb
    s.ea = ea
    s.eb = eb
    s.ph = ph
    s.ticks = []
    s.da = []
    s.db = []
    s.step = 0
    s.ready = True
    s.last_meta = load_meta()
    s.last_export_step = 0


def make_signal_label(direction: int, acted: bool) -> str:
    if not acted:
        return "ABSTAIN"
    return "BUY" if direction == 1 else "SELL"


def step_sim() -> None:
    s = st.session_state
    tick = s.gen.next_price()
    price = tick["price"]
    step = s.step
    s.ticks.append(tick)

    feat = s.fe.update(price)
    if feat is None:
        s.step += 1
        return

    s.sb.push(feat)
    if not s.sb.ready():
        s.step += 1
        return

    seq = s.sb.get_sequence()
    for twin, evaluator, decision_log in [(s.ta, s.ea, s.da), (s.tb, s.eb, s.db)]:
        direction, confidence, epistemic, entropy, acted = twin.predict(seq)
        signal = make_signal_label(direction, acted)
        decision_log.append(
            {
                "step": step,
                "price": price,
                "direction": direction,
                "confidence": confidence,
                "epistemic": epistemic,
                "entropy": entropy,
                "acted": acted,
                "signal": signal,
                "regime": tick.get("regime", "N/A"),
                "regime_color": tick.get("regime_color", COL_A),
                "time": tick.get("time", f"sim-{step}"),
            }
        )
        evaluator.register(
            step,
            twin.name[:6],
            direction,
            confidence,
            epistemic,
            entropy,
            acted,
            price,
            seq,
            tick.get("regime", "N/A"),
        )
        for record, record_seq in evaluator.resolve(step, price):
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

    s.step += 1


def price_chart(ticks: List[Dict], window: int = 400) -> go.Figure:
    f = fig(300, "XAUUSD Synthetic Price")
    if not ticks:
        return f
    recent = ticks[-window:]
    f.add_trace(
        go.Scatter(
            x=[t["step"] for t in recent],
            y=[t["price"] for t in recent],
            mode="lines",
            line=dict(color="#7a9bb8", width=1.5),
            name="XAUUSD",
        )
    )
    for point in recent:
        if point.get("regime_changed"):
            f.add_vline(
                x=point["step"],
                line=dict(color=point.get("regime_color", COL_A), width=1, dash="dash"),
                annotation_text=point.get("regime", "Regime"),
                annotation_font=dict(size=8, color=point.get("regime_color", COL_A)),
                annotation_position="top left",
            )
    return f


def dqi_chart(ea, eb) -> go.Figure:
    f = fig(320, "Decision Quality Index ±1σ")
    for fn, col, name in [(ea.rolling_dqi, COL_A, "Twin-A"), (eb.rolling_dqi, COL_B, "Twin-B")]:
        dqi = fn()
        if not dqi:
            continue
        arr = np.array(dqi)
        std = pd.Series(arr).rolling(20, min_periods=1).std().fillna(0).values
        x = list(range(len(arr)))
        fill = "rgba(56,189,248,0.10)" if col == COL_A else "rgba(251,146,60,0.10)"
        f.add_trace(go.Scatter(x=x, y=(arr + std).tolist(), mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        f.add_trace(go.Scatter(x=x, y=(arr - std).tolist(), mode="lines", line=dict(width=0), fill="tonexty", fillcolor=fill, showlegend=False, hoverinfo="skip"))
        f.add_trace(go.Scatter(x=x, y=dqi, mode="lines", line=dict(color=col, width=2), name=f"{name} DQI"))
    f.add_hline(y=0.5, line=dict(color="#2a3f58", width=1, dash="dot"), annotation_text="Baseline 0.50", annotation_font=dict(size=9, color="#3d5470"))
    f.update_layout(yaxis=dict(**BL["yaxis"], range=[0, 1]))
    return f


def pnl_chart(ea, eb) -> go.Figure:
    f = fig(240, "Cumulative PnL Proxy")
    for ev, col, name in [(ea, COL_A, "Twin-A"), (eb, COL_B, "Twin-B")]:
        curve = ev.cumulative_pnl()
        if curve:
            fill = "rgba(56,189,248,0.07)" if col == COL_A else "rgba(251,146,60,0.07)"
            f.add_trace(
                go.Scatter(
                    y=curve,
                    mode="lines",
                    line=dict(color=col, width=2),
                    name=name,
                    fill="tozeroy",
                    fillcolor=fill,
                )
            )
    f.add_hline(y=0, line=dict(color="#2a3f58", width=1))
    return f


def conf_chart(da: List[Dict], db: List[Dict], window: int = 300) -> go.Figure:
    f = fig(240, "Confidence Over Time")
    for logs, col, name, threshold in [(da, COL_A, "Twin-A", 0.70), (db, COL_B, "Twin-B", 0.55)]:
        recent = logs[-window:]
        if not recent:
            continue
        f.add_trace(
            go.Scatter(
                x=[d["step"] for d in recent],
                y=[d["confidence"] for d in recent],
                mode="lines",
                line=dict(color=col, width=1.2),
                name=name,
            )
        )
        f.add_hline(y=threshold, line=dict(color=col, width=0.8, dash="dash"))
    f.update_layout(yaxis=dict(**BL["yaxis"], range=[0, 1]))
    return f


def epist_chart(da: List[Dict], db: List[Dict], window: int = 300) -> go.Figure:
    f = fig(220, "Epistemic Uncertainty (MC Dropout σ)")
    for logs, col, name in [(da, COL_A, "Twin-A"), (db, COL_B, "Twin-B")]:
        recent = logs[-window:]
        if recent:
            f.add_trace(
                go.Scatter(
                    x=[d["step"] for d in recent],
                    y=[d["epistemic"] for d in recent],
                    mode="lines",
                    line=dict(color=col, width=1.1),
                    name=name,
                )
            )
    return f


def regime_chart(ea, eb) -> go.Figure:
    win_a = ea.regime_win_rates()
    win_b = eb.regime_win_rates()
    regimes = sorted(set(list(win_a) + list(win_b)))
    f = fig(260, "Win Rate by Regime")
    if not regimes:
        return f
    f.add_trace(go.Bar(x=regimes, y=[win_a.get(r, 0) for r in regimes], name="Twin-A", marker_color=COL_A, opacity=0.8))
    f.add_trace(go.Bar(x=regimes, y=[win_b.get(r, 0) for r in regimes], name="Twin-B", marker_color=COL_B, opacity=0.8))
    f.add_hline(y=0.5, line=dict(color="#2a3f58", width=1, dash="dot"))
    f.update_layout(barmode="group", yaxis=dict(**BL["yaxis"], range=[0, 1]))
    return f


def loss_chart(ta, tb) -> go.Figure:
    f = fig(200, "Training Loss per Retrain Event")
    if ta.loss_history:
        f.add_trace(go.Scatter(y=ta.loss_history, mode="lines+markers", line=dict(color=COL_A, width=1.5), marker=dict(size=4), name="Twin-A"))
    if tb.loss_history:
        f.add_trace(go.Scatter(y=tb.loss_history, mode="lines+markers", line=dict(color=COL_B, width=1.5), marker=dict(size=4), name="Twin-B"))
    return f


def recent_signal_df(decisions_a: List[Dict], decisions_b: List[Dict], n: int = 12) -> pd.DataFrame:
    if not decisions_a or not decisions_b:
        return pd.DataFrame()
    rows = []
    for a, b in zip(decisions_a[-n:], decisions_b[-n:]):
        rows.append(
            {
                "Step": a.get("step", ""),
                "Price": f"${a.get('price', 0):,.2f}",
                "Regime": a.get("regime", "N/A"),
                "A": a.get("signal", "N/A"),
                "A Conf": f"{a.get('confidence', 0):.3f}",
                "B": b.get("signal", "N/A"),
                "B Conf": f"{b.get('confidence', 0):.3f}",
            }
        )
    return pd.DataFrame(rows)


def retrain_df(twin) -> pd.DataFrame:
    rows = []
    for ev in twin.retrain_events[-25:]:
        rows.append(
            {
                "Step": ev.step,
                "Trigger": ev.trigger,
                "Acc Before": ev.acc_before,
                "Acc After": ev.acc_after,
                "Loss": ev.loss,
                "Grad Norm": ev.grad_norm,
                "Version": ev.version,
            }
        )
    return pd.DataFrame(rows)


def signal_mix_df(decisions: List[Dict], twin_name: str) -> pd.DataFrame:
    if not decisions:
        return pd.DataFrame()
    df = pd.DataFrame(decisions)
    counts = df["signal"].value_counts().to_dict() if "signal" in df else {}
    return pd.DataFrame(
        [
            {"Twin": twin_name, "Signal": signal, "Count": count}
            for signal, count in sorted(counts.items())
        ]
    )


def regime_coverage_df(ticks: List[Dict]) -> pd.DataFrame:
    if not ticks:
        return pd.DataFrame()
    df = pd.DataFrame(ticks)
    counts = df["regime"].value_counts().sort_values(ascending=False)
    return pd.DataFrame(
        [{"Regime": regime, "Steps": int(steps), "Share": f"{steps / len(df):.1%}"} for regime, steps in counts.items()]
    )


def training_health_rows(ta, tb, ea, eb) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Twin": "Twin-A",
                "Threshold": "0.70 / H≤0.60",
                "Retrain": "500 steps",
                "Decisions": ea.total_decisions(),
                "Win Rate": f"{ea.win_rate():.3f}",
                "Rolling Acc": f"{ta.rolling_accuracy():.3f}",
                "Calibration Gap": f"{abs(ta.mean_confidence() - ta.rolling_accuracy()):.3f}",
                "Sharpe": f"{ea.sharpe_ratio():.3f}",
                "Max DD": f"{ea.max_drawdown():.4f}",
                "Retrains": ta.total_retrains(),
                "LR": f"{ta.current_lr():.6f}",
            },
            {
                "Twin": "Twin-B",
                "Threshold": "0.55 / H≤0.80",
                "Retrain": "200 steps",
                "Decisions": eb.total_decisions(),
                "Win Rate": f"{eb.win_rate():.3f}",
                "Rolling Acc": f"{tb.rolling_accuracy():.3f}",
                "Calibration Gap": f"{abs(tb.mean_confidence() - tb.rolling_accuracy()):.3f}",
                "Sharpe": f"{eb.sharpe_ratio():.3f}",
                "Max DD": f"{eb.max_drawdown():.4f}",
                "Retrains": tb.total_retrains(),
                "LR": f"{tb.current_lr():.6f}",
            },
        ]
    )


def export_freshness_banner() -> None:
    if not state_exists():
        return
    meta = load_meta()
    if not meta:
        return
    st.markdown(
        f"""<div class="export-box">
            <strong style="color:#22c55e;">✓ Trained twins saved</strong> ·
            {meta.get('saved_at', '')[:19].replace('T', ' ')} ·
            Source: {meta.get('source', '?')} ·
            A acc: {meta.get('twin_a_acc', '?')} · B acc: {meta.get('twin_b_acc', '?')} ·
            <strong>Run live_dashboard.py to deploy →</strong>
            </div>""",
        unsafe_allow_html=True,
    )


def main() -> None:
    init()
    s = st.session_state

    with st.sidebar:
        st.markdown(
            """<div style="font-family:'IBM Plex Mono',monospace;font-size:11px;
            color:#38bdf8;letter-spacing:2px;margin-bottom:2px;">⚙ STEP 1 · TRAIN</div>
            <div style="font-size:9px;color:#2a3f58;margin-bottom:16px;
            font-family:'IBM Plex Mono',monospace;">Simulate → Export → Deploy Live</div>
            """,
            unsafe_allow_html=True,
        )

        seed = st.number_input("Simulation seed", min_value=1, max_value=999999, value=int(s.selected_seed), step=1)
        s.selected_seed = seed
        show_n = st.slider("Chart window", 100, 1200, 400, 50)
        speed = st.slider("Ticks per refresh", 1, 50, 8)
        auto = st.toggle("Auto-run", value=False)

        st.markdown("---")
        if st.button("▶ LAUNCH / RESET TRAINING"):
            launch(seed=seed)
            st.rerun()

        if s.ready:
            if st.button("💾 EXPORT TWINS TO LIVE"):
                with st.spinner("Saving weights, buffers, metadata…"):
                    s.last_meta = save_current_twins(s.ta, s.tb, source=f"step_{s.step}")
                    s.last_export_step = s.step
                st.success(f"Saved at simulation step {s.step}.")

            st.markdown("---")
            if not auto:
                if st.button(f"⏩ +{speed} ticks"):
                    for _ in range(speed):
                        step_sim()
                    st.rerun()
                if st.button("⏩ +1 tick"):
                    step_sim()
                    st.rerun()

        if s.ready:
            st.markdown(
                f"""<div style="font-family:'IBM Plex Mono',monospace;
                font-size:9px;color:#2a3f58;line-height:2.0;margin-top:8px;">
                READY · <span style="color:#22c55e">YES</span><br>
                STEP · <span style="color:#dde6f0">{s.step}</span><br>
                BUFFER A · <span style="color:#dde6f0">{len(s.ta.replay)}</span><br>
                BUFFER B · <span style="color:#dde6f0">{len(s.tb.replay)}</span><br>
                LAST EXPORT · <span style="color:#dde6f0">{s.last_export_step if s.last_export_step is not None else 'pretrain'}</span>
                </div>""",
                unsafe_allow_html=True,
            )

    h1, h2 = st.columns([3, 1])
    with h1:
        st.markdown(
            """<h1 style="font-size:19px;color:#dde6f0;margin-bottom:2px;">
            ⚙ Digital Twin Training · XAUUSD</h1>
            <p style="font-size:9px;color:#2a3f58;margin:0;font-family:'IBM Plex Mono',monospace;">
            STEP 1 · Train on synthetic data → Export → Load in live dashboard</p>
            """,
            unsafe_allow_html=True,
        )
    with h2:
        if s.ready and s.ticks:
            latest_tick = s.ticks[-1]
            color = "#22c55e" if latest_tick.get("return", 0) > 0 else "#ef4444"
            st.markdown(
                f"""<div style="text-align:right;font-family:'IBM Plex Mono',monospace;">
                <div style="font-size:9px;color:#2a3f58;">SYNTHETIC XAUUSD</div>
                <div style="font-size:22px;color:{color};">${latest_tick['price']:,.2f}</div>
                <div style="font-size:9px;color:#2a3f58;">{latest_tick.get('regime', 'N/A')} · step {s.step}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    export_freshness_banner()

    if not s.ready:
        st.markdown(
            """<div style="background:#111827;border:1px solid #1e2a3a;
            border-radius:8px;padding:22px;margin-top:12px;">
            <h3 style="font-family:'IBM Plex Mono',monospace;font-size:13px;color:#38bdf8;margin-bottom:14px;">
            How the training pipeline works</h3>
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;">
            <div style="background:#0c1928;border:1px solid #1a3044;border-radius:6px;padding:13px;">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:#38bdf8;margin-bottom:5px;">
            STEP 1 · TRAIN</div>
            <div style="font-size:11px;color:#6a8aaa;line-height:1.7;">
            Launch simulation. Twins pre-train on 600 synthetic warm-up steps.
            Continue running to let twins adapt across all 7 market regimes.
            Weights auto-saved after pretrain.</div></div>
            <div style="background:#0c1928;border:1px solid #1a3044;border-radius:6px;padding:13px;">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:#fb923c;margin-bottom:5px;">
            STEP 2 · EXPORT</div>
            <div style="font-size:11px;color:#6a8aaa;line-height:1.7;">
            Click Export Twins to Live at any point. Saves LSTM weights,
            temporal attention params, replay buffer, and metadata to the .twin_state folder.</div></div>
            <div style="background:#0c1928;border:1px solid #1a3044;border-radius:6px;padding:13px;">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:#22c55e;margin-bottom:5px;">
            STEP 3 · DEPLOY</div>
            <div style="font-size:11px;color:#6a8aaa;line-height:1.7;">
            Open live_dashboard.py. Twins load saved weights automatically.
            Replay buffers are transferred so first live retraining uses both synthetic and live data.</div></div>
            </div></div>""",
            unsafe_allow_html=True,
        )
        return

    ta, tb = s.ta, s.tb
    ea, eb = s.ea, s.eb

    calibration_a = abs(ta.mean_confidence() - ta.rolling_accuracy())
    calibration_b = abs(tb.mean_confidence() - tb.rolling_accuracy())
    if calibration_a > 0.15 or calibration_b > 0.15:
        st.markdown(
            f"""<div class="warn-box">
            ⚠ Calibration watch · Twin-A gap: {calibration_a:.3f} · Twin-B gap: {calibration_b:.3f}.
            Large gaps can mean the twins sound more confident than their realised accuracy.
            </div>""",
            unsafe_allow_html=True,
        )

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "📈 Market Feed",
            "⚖️ Twin Comparison",
            "🎯 Uncertainty",
            "🌍 Regime Analysis",
            "🔄 Adaptation Log",
            "📡 Transfer to Live",
        ]
    )

    with tab1:
        st.plotly_chart(price_chart(s.ticks, show_n), use_container_width=True)
        if s.ticks:
            latest_tick = s.ticks[-1]
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(mc("PRICE", f"${latest_tick['price']:,.2f}", f"step {latest_tick.get('step', s.step)}"), unsafe_allow_html=True)
            with c2:
                st.markdown(mc("RETURN", f"{latest_tick.get('return', 0) * 100:+.4f}%", "per-step"), unsafe_allow_html=True)
            with c3:
                st.markdown(mc("VOL", f"{latest_tick.get('volatility', 0) * 100:.4f}%", "GARCH"), unsafe_allow_html=True)
            with c4:
                st.markdown(mc("REGIME", latest_tick.get("regime", "N/A"), f"jump={latest_tick.get('jump', 0) * 100:+.3f}%"), unsafe_allow_html=True)

        safe_show_df(recent_signal_df(s.da, s.db, n=12), "No paired twin decisions yet.")

        c1, c2 = st.columns(2)
        with c1:
            safe_show_df(signal_mix_df(s.da, "Twin-A"), "Twin-A has no signal history yet.")
        with c2:
            safe_show_df(signal_mix_df(s.db, "Twin-B"), "Twin-B has no signal history yet.")

    with tab2:
        st.plotly_chart(dqi_chart(ea, eb), use_container_width=True)
        st.plotly_chart(pnl_chart(ea, eb), use_container_width=True)
        safe_show_df(training_health_rows(ta, tb, ea, eb), "No training health rows yet.")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(mc("TWIN-A WIN RATE", f"{ea.win_rate():.3f}", f"{ea.total_decisions()} decisions · Sharpe {ea.sharpe_ratio():.3f}", "ca"), unsafe_allow_html=True)
            st.markdown(mc("TWIN-A ROLLING ACC", f"{ta.rolling_accuracy():.3f}", f"Abstain {ta.abstention_rate() * 100:.1f}% · v{ta.version}", "ca"), unsafe_allow_html=True)
        with c2:
            st.markdown(mc("TWIN-B WIN RATE", f"{eb.win_rate():.3f}", f"{eb.total_decisions()} decisions · Sharpe {eb.sharpe_ratio():.3f}", "cb"), unsafe_allow_html=True)
            st.markdown(mc("TWIN-B ROLLING ACC", f"{tb.rolling_accuracy():.3f}", f"Abstain {tb.abstention_rate() * 100:.1f}% · v{tb.version}", "cb"), unsafe_allow_html=True)

    with tab3:
        st.plotly_chart(conf_chart(s.da, s.db, show_n), use_container_width=True)
        st.plotly_chart(epist_chart(s.da, s.db, show_n), use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(mc("TWIN-A CONF", f"{ta.mean_confidence():.3f}", f"σ={ta.confidence_std():.3f} · H={ta.mean_entropy():.3f}", "ca"), unsafe_allow_html=True)
            st.markdown(mc("TWIN-A CALIBRATION", f"{calibration_a:.3f}", "| confidence - realised accuracy |", "cx"), unsafe_allow_html=True)
        with c2:
            st.markdown(mc("TWIN-B CONF", f"{tb.mean_confidence():.3f}", f"σ={tb.confidence_std():.3f} · H={tb.mean_entropy():.3f}", "cb"), unsafe_allow_html=True)
            st.markdown(mc("TWIN-B CALIBRATION", f"{calibration_b:.3f}", "| confidence - realised accuracy |", "cx"), unsafe_allow_html=True)

    with tab4:
        st.plotly_chart(regime_chart(ea, eb), use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            win_a = ea.regime_win_rates()
            if win_a:
                safe_show_df(pd.DataFrame([{"Regime": k, "Win Rate": v, "vs Base": f"{v - 0.5:+.3f}"} for k, v in win_a.items()]), "Twin-A has no regime stats yet.")
        with c2:
            win_b = eb.regime_win_rates()
            if win_b:
                safe_show_df(pd.DataFrame([{"Regime": k, "Win Rate": v, "vs Base": f"{v - 0.5:+.3f}"} for k, v in win_b.items()]), "Twin-B has no regime stats yet.")

        safe_show_df(regime_coverage_df(s.ticks), "No regime coverage yet.")

        if getattr(s.gen, "regime_transitions", None):
            st.markdown("##### Regime Transition Log")
            safe_show_df(pd.DataFrame([{"Step": t["step"], "From": t["from"], "To": t["to"]} for t in s.gen.regime_transitions]), "No regime transitions yet.")

    with tab5:
        st.plotly_chart(loss_chart(ta, tb), use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            safe_show_df(retrain_df(ta), "No Twin-A retrains yet.")
        with c2:
            safe_show_df(retrain_df(tb), "No Twin-B retrains yet.")

    with tab6:
        st.markdown(
            """<div style="font-family:'IBM Plex Mono',monospace;font-size:10px;
            color:#22c55e;letter-spacing:2px;margin-bottom:14px;">
            📡 TRANSFER LEARNED WEIGHTS → LIVE DASHBOARD</div>""",
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f"""<div style="background:#0c1928;border:1px solid #1a3044;
                border-radius:8px;padding:15px;font-size:11px;color:#6a8aaa;line-height:1.9;">
                <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;
                color:#38bdf8;margin-bottom:8px;">TWIN-A STATE</div>
                Version: <strong style="color:#dde6f0">v{ta.version}</strong><br>
                Retrains: <strong style="color:#dde6f0">{ta.total_retrains()}</strong><br>
                Buffer size: <strong style="color:#dde6f0">{len(ta.replay)}</strong><br>
                Rolling acc: <strong style="color:#dde6f0">{ta.rolling_accuracy():.4f}</strong><br>
                Mean conf: <strong style="color:#dde6f0">{ta.mean_confidence():.4f}</strong><br>
                Mean H: <strong style="color:#dde6f0">{ta.mean_entropy():.4f}</strong><br>
                Current LR: <strong style="color:#dde6f0">{ta.current_lr():.6f}</strong><br>
                Regimes seen: <strong style="color:#dde6f0">{len(ea.regime_win_rates())}</strong>
                </div>""",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"""<div style="background:#0c1928;border:1px solid #1a3044;
                border-radius:8px;padding:15px;font-size:11px;color:#6a8aaa;line-height:1.9;">
                <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;
                color:#fb923c;margin-bottom:8px;">TWIN-B STATE</div>
                Version: <strong style="color:#dde6f0">v{tb.version}</strong><br>
                Retrains: <strong style="color:#dde6f0">{tb.total_retrains()}</strong><br>
                Buffer size: <strong style="color:#dde6f0">{len(tb.replay)}</strong><br>
                Rolling acc: <strong style="color:#dde6f0">{tb.rolling_accuracy():.4f}</strong><br>
                Mean conf: <strong style="color:#dde6f0">{tb.mean_confidence():.4f}</strong><br>
                Mean H: <strong style="color:#dde6f0">{tb.mean_entropy():.4f}</strong><br>
                Current LR: <strong style="color:#dde6f0">{tb.current_lr():.6f}</strong><br>
                Regimes seen: <strong style="color:#dde6f0">{len(eb.regime_win_rates())}</strong>
                </div>""",
                unsafe_allow_html=True,
            )

        if s.last_meta:
            m = s.last_meta
            st.markdown(
                f"""<div class="export-box">
                LAST EXPORT · {m.get('saved_at', '')[:19].replace('T', ' ')} ·
                Source: {m.get('source', '?')} ·
                A v{m.get('twin_a_version', '?')} acc={m.get('twin_a_acc', '?')} ·
                B v{m.get('twin_b_version', '?')} acc={m.get('twin_b_acc', '?')}
                </div>""",
                unsafe_allow_html=True,
            )

        st.markdown(
            """<div style="background:#0c1928;border:1px solid #1a3044;
            border-radius:6px;padding:13px 15px;margin-top:10px;
            font-size:11px;color:#4a7090;line-height:1.8;">
            <strong style="color:#8aa0b8;">What gets transferred:</strong><br>
            LSTM weights + attention layer parameters ·
            Replay buffer contents ·
            Training metadata (version, accuracy, retrain history) ·
            Learning-rate schedule effects already embedded in the trained weights.<br><br>
            The live dashboard loads these before its first prediction.
            The transferred replay buffer means the first live retraining event
            blends synthetic experience with new live market observations.
            </div>""",
            unsafe_allow_html=True,
        )

        ticks_df = pd.DataFrame(s.ticks)
        signals_df = pd.concat([
            pd.DataFrame(s.da).assign(twin="Twin-A") if s.da else pd.DataFrame(),
            pd.DataFrame(s.db).assign(twin="Twin-B") if s.db else pd.DataFrame(),
        ], ignore_index=True)
        retrains_df = pd.concat([
            retrain_df(ta).assign(Twin="Twin-A") if not retrain_df(ta).empty else pd.DataFrame(),
            retrain_df(tb).assign(Twin="Twin-B") if not retrain_df(tb).empty else pd.DataFrame(),
        ], ignore_index=True)

        d1, d2, d3 = st.columns(3)
        with d1:
            st.download_button("⬇ Download ticks CSV", to_csv_bytes(ticks_df), file_name="simulation_ticks.csv", mime="text/csv")
        with d2:
            st.download_button("⬇ Download signals CSV", to_csv_bytes(signals_df), file_name="simulation_signals.csv", mime="text/csv")
        with d3:
            st.download_button("⬇ Download retrains CSV", to_csv_bytes(retrains_df), file_name="simulation_retrains.csv", mime="text/csv")

    if auto and s.ready:
        for _ in range(speed):
            step_sim()
        time.sleep(0.12)
        st.rerun()


if __name__ == "__main__":
    main()
