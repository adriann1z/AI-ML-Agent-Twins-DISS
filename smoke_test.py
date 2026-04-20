"""Minimal smoke test for the dissertation project."""

from models import build_twins
from mt5_bridge.live_feed import LiveFeedAdapter
from mt5_bridge.paper_engine import PaperTradingEngine
from twin_state import load_twins, state_exists


def main():
    assert state_exists(), "Expected a compatible saved twin state"
    ta, tb = build_twins()
    meta = load_twins(ta, tb)
    assert meta is not None, "Failed to load saved twins"

    feed = LiveFeedAdapter(use_synthetic_fallback=True)
    counts = {"A": {"buy": 0, "sell": 0, "act": 0}, "B": {"buy": 0, "sell": 0, "act": 0}}
    for _ in range(760):
        tick = feed.synthetic_tick()
        price = tick["price"]
        seq = feed.ingest_tick(price, spread=price * 0.0001)
        if seq is None:
            continue
        for twin, tag in [(ta, "A"), (tb, "B")]:
            direction, confidence, epistemic, entropy, acted = twin.predict(seq)
            if acted:
                counts[tag]["act"] += 1
                counts[tag]["buy" if direction == 1 else "sell"] += 1

    assert counts["A"]["act"] > 0 and counts["B"]["act"] > 0, counts
    assert counts["A"]["buy"] > 0 and counts["A"]["sell"] > 0, counts
    assert counts["B"]["buy"] > 0 and counts["B"]["sell"] > 0, counts
    assert counts["B"]["act"] > counts["A"]["act"], counts

    engine = PaperTradingEngine(starting_balance=10000.0)
    order = engine.place_order("BUY", 2000.0, "Twin-A", 0.61, 0.68, "Bull Trend")
    assert order is not None
    engine.update_prices(order.tp_price)
    assert len(engine.closed_orders()) == 1

    print("SMOKE TEST PASSED")
    print(counts)


if __name__ == "__main__":
    main()
