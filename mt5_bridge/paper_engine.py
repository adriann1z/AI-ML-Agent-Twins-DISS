"""
mt5_bridge/paper_engine.py
Virtual paper trading engine.
Manages simulated orders, P&L calculation, and position tracking.
NO real orders are sent to the broker under any circumstances.
"""

import numpy as np
from datetime import datetime
from typing import List, Optional, Dict
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class PaperOrder:
    order_id:    int
    timestamp:   datetime
    symbol:      str
    direction:   str          # "BUY" / "SELL"
    price_in:    float
    sl_price:    float        # stop-loss (paper)
    tp_price:    float        # take-profit (paper)
    lot_size:    float
    twin_name:   str
    confidence:  float
    entropy:     float
    regime:      str
    status:      str = "OPEN"
    price_out:   Optional[float] = None
    pnl_pips:    float = 0.0
    pnl_usd:     float = 0.0
    closed_at:   Optional[datetime] = None
    close_reason: str = ""


class PaperTradingEngine:
    """
    Simulates order placement and management on live prices.
    Tracks equity curve, drawdown, win/loss ratio, and per-twin P&L.

    Risk management (paper):
      - Max 1 open position per twin at a time
      - Fixed SL: 15 pips, TP: 25 pips (configurable)
      - Virtual lot size: 0.01 (micro lot)
      - Point value for XAUUSD: $1 per pip per 0.1 lot
    """

    POINT        = 0.1     # 1 pip = $0.10 per 0.01 lot
    LOT          = 0.01
    SL_PIPS      = 150     # stop-loss in pips (15.0 USD move)
    TP_PIPS      = 250     # take-profit in pips (25.0 USD move)
    PIP_SIZE     = 0.1     # XAUUSD: 1 pip = $0.10

    def __init__(self, starting_balance: float = 10_000.0):
        self.balance         = starting_balance
        self.starting_balance = starting_balance
        self.orders: List[PaperOrder] = []
        self._order_id       = 1
        self._open: Dict[str, PaperOrder] = {}   # twin_name → open order

    # ---------------------------------------------------------------- #
    #  Order management                                                  #
    # ---------------------------------------------------------------- #

    def place_order(self, direction: str, price: float,
                    twin_name: str, confidence: float,
                    entropy: float, regime: str) -> Optional[PaperOrder]:
        """
        Place a virtual paper order.
        Returns None if twin already has an open position.
        """
        if twin_name in self._open:
            return None   # one position per twin max

        sl, tp = self._calc_sl_tp(direction, price)

        order = PaperOrder(
            order_id   = self._order_id,
            timestamp  = datetime.now(),
            symbol     = "XAUUSD",
            direction  = direction,
            price_in   = price,
            sl_price   = sl,
            tp_price   = tp,
            lot_size   = self.LOT,
            twin_name  = twin_name,
            confidence = confidence,
            entropy    = entropy,
            regime     = regime,
        )
        self.orders.append(order)
        self._open[twin_name] = order
        self._order_id += 1
        return order

    def _calc_sl_tp(self, direction: str, price: float):
        sl_delta = self.SL_PIPS * self.PIP_SIZE
        tp_delta = self.TP_PIPS * self.PIP_SIZE
        if direction == "BUY":
            return price - sl_delta, price + tp_delta
        else:
            return price + sl_delta, price - tp_delta

    def update_prices(self, current_price: float):
        """
        Called on every new tick. Checks SL/TP for all open positions.
        Closes positions automatically if hit.
        """
        closed_twins = []
        for twin_name, order in self._open.items():
            hit, reason = self._check_exit(order, current_price)
            if hit:
                self._close_order(order, current_price, reason)
                closed_twins.append(twin_name)

        for t in closed_twins:
            del self._open[t]

    def _check_exit(self, order: PaperOrder, price: float):
        if order.direction == "BUY":
            if price <= order.sl_price:
                return True, "SL hit"
            if price >= order.tp_price:
                return True, "TP hit"
        else:  # SELL
            if price >= order.sl_price:
                return True, "SL hit"
            if price <= order.tp_price:
                return True, "TP hit"
        return False, ""

    def _close_order(self, order: PaperOrder, price: float, reason: str):
        order.status      = "CLOSED"
        order.price_out   = price
        order.closed_at   = datetime.now()
        order.close_reason = reason

        if order.direction == "BUY":
            pips = (price - order.price_in) / self.PIP_SIZE
        else:
            pips = (order.price_in - price) / self.PIP_SIZE

        pnl_usd = pips * self.PIP_SIZE * (order.lot_size / 0.01)
        order.pnl_pips = round(pips, 1)
        order.pnl_usd  = round(pnl_usd, 2)
        self.balance  += pnl_usd

    def close_all(self, current_price: float):
        """Force-close all open positions (e.g. end of session)."""
        for twin_name, order in list(self._open.items()):
            self._close_order(order, current_price, "manual close")
        self._open.clear()

    def close_twin(self, twin_name: str, current_price: float):
        order = self._open.get(twin_name)
        if order is None:
            return
        self._close_order(order, current_price, "signal reverse")
        self._open.pop(twin_name, None)

    def has_open_position(self, twin_name: str) -> bool:
        return twin_name in self._open

    def get_open_position(self, twin_name: str) -> Optional[PaperOrder]:
        return self._open.get(twin_name)

    # ---------------------------------------------------------------- #
    #  Metrics                                                           #
    # ---------------------------------------------------------------- #

    def closed_orders(self) -> List[PaperOrder]:
        return [o for o in self.orders if o.status == "CLOSED"]

    def equity(self, current_price: float) -> float:
        """Balance + unrealised P&L of open positions."""
        unrealised = 0.0
        for order in self._open.values():
            if order.direction == "BUY":
                pips = (current_price - order.price_in) / self.PIP_SIZE
            else:
                pips = (order.price_in - current_price) / self.PIP_SIZE
            unrealised += pips * self.PIP_SIZE * (order.lot_size / 0.01)
        return round(self.balance + unrealised, 2)

    def total_pnl(self) -> float:
        return round(sum(o.pnl_usd for o in self.closed_orders()), 2)

    def win_rate(self) -> float:
        closed = self.closed_orders()
        if not closed:
            return 0.0
        return round(sum(1 for o in closed if o.pnl_usd > 0) / len(closed), 3)

    def max_drawdown(self) -> float:
        pnls = [o.pnl_usd for o in self.closed_orders()]
        if not pnls:
            return 0.0
        equity = np.cumsum(pnls) + self.starting_balance
        peak   = np.maximum.accumulate(equity)
        dd     = (equity - peak) / (peak + 1e-10)
        return float(dd.min())

    def sharpe_ratio(self) -> float:
        pnls = [o.pnl_usd for o in self.closed_orders()]
        if len(pnls) < 5:
            return 0.0
        arr = np.array(pnls)
        return float(arr.mean() / (arr.std() + 1e-10) * np.sqrt(252))

    def equity_curve(self) -> List[float]:
        curve = [self.starting_balance]
        running = self.starting_balance
        for o in self.closed_orders():
            running += o.pnl_usd
            curve.append(running)
        return curve

    def per_twin_stats(self) -> dict:
        stats = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0, "pips": 0.0})
        for o in self.closed_orders():
            s = stats[o.twin_name]
            s["trades"] += 1
            s["pnl"]    += o.pnl_usd
            s["pips"]   += o.pnl_pips
            if o.pnl_usd > 0:
                s["wins"] += 1
        return dict(stats)

    def recent_orders_df(self, n: int = 30):
        import pandas as pd
        recent = self.orders[-n:]
        if not recent:
            return pd.DataFrame()
        return pd.DataFrame([{
            "ID":        o.order_id,
            "Time":      o.timestamp.strftime("%H:%M:%S"),
            "Twin":      o.twin_name,
            "Dir":       o.direction,
            "Price In":  f"{o.price_in:.2f}",
            "Price Out": f"{o.price_out:.2f}" if o.price_out else "OPEN",
            "PnL $":     f"{o.pnl_usd:+.2f}" if o.status == "CLOSED" else "—",
            "Pips":      f"{o.pnl_pips:+.1f}" if o.status == "CLOSED" else "—",
            "Status":    o.status,
            "Reason":    o.close_reason or "—",
            "Regime":    o.regime,
            "Conf":      f"{o.confidence:.3f}",
        } for o in reversed(recent)])
