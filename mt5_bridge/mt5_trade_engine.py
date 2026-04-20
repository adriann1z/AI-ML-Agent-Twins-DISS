from __future__ import annotations

"""Real MT5 trade engine with local order tracking for the dashboard."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import MetaTrader5 as mt5

    MT5_AVAILABLE = True
except ImportError:  # pragma: no cover
    MT5_AVAILABLE = False
    mt5 = None


@dataclass
class BrokerOrder:
    order_id: int
    timestamp: datetime
    symbol: str
    direction: str
    price_in: float
    sl_price: float
    tp_price: float
    lot_size: float
    twin_name: str
    confidence: float
    entropy: float
    regime: str
    status: str = "OPEN"
    price_out: Optional[float] = None
    pnl_pips: float = 0.0
    pnl_usd: float = 0.0
    closed_at: Optional[datetime] = None
    close_reason: str = ""
    broker_ticket: Optional[int] = None
    position_ticket: Optional[int] = None


class MT5TradeEngine:
    PIP_SIZE = 0.1
    MAGIC_A = 550001
    MAGIC_B = 550002

    def __init__(
        self,
        connector,
        lot_size: float = 0.01,
        sl_pips: float = 150.0,
        tp_pips: float = 250.0,
        allow_live: bool = False,
    ):
        self.connector = connector
        self.symbol = connector.symbol
        self.lot_size = float(lot_size)
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.allow_live = bool(allow_live)

        self.starting_balance = float(connector.account_summary().get("balance", 10000.0))
        self.balance = self.starting_balance
        self.orders: List[BrokerOrder] = []
        self._open: Dict[str, BrokerOrder] = {}
        self._next_id = 1
        self.last_error = ""

    def _magic(self, twin_name: str) -> int:
        return self.MAGIC_A if self._canonical_twin_name(twin_name) == "Twin-A" else self.MAGIC_B

    def _canonical_twin_name(self, twin_name: str) -> str:
        name = str(twin_name or "").strip()
        if name.startswith("Twin-A"):
            return "Twin-A"
        if name.startswith("Twin-B"):
            return "Twin-B"
        return name or "Twin-A"

    def _filling_modes(self) -> List[int]:
        if not MT5_AVAILABLE:
            return [0]

        order_ioc = getattr(mt5, "ORDER_FILLING_IOC", 1)
        order_fok = getattr(mt5, "ORDER_FILLING_FOK", 0)
        order_return = getattr(mt5, "ORDER_FILLING_RETURN", order_ioc)

        preferred: List[int] = []
        info = mt5.symbol_info(self.symbol)
        if info is not None:
            fill_mask = int(getattr(info, "filling_mode", 0) or 0)
            symbol_ioc = getattr(mt5, "SYMBOL_FILLING_IOC", 2)
            symbol_fok = getattr(mt5, "SYMBOL_FILLING_FOK", 1)

            if fill_mask & symbol_ioc:
                preferred.append(order_ioc)
            if fill_mask & symbol_fok:
                preferred.append(order_fok)

        for candidate in (order_ioc, order_fok, order_return):
            if candidate not in preferred:
                preferred.append(candidate)
        return preferred

    def _send_market_request(self, request: dict, failure_prefix: str) -> Optional[Any]:
        if not MT5_AVAILABLE:
            self.last_error = "MetaTrader5 package not available"
            return None

        invalid_fill = getattr(mt5, "TRADE_RETCODE_INVALID_FILL", 10030)
        success_codes = {
            getattr(mt5, "TRADE_RETCODE_DONE", 10009),
            getattr(mt5, "TRADE_RETCODE_PLACED", 10008),
        }
        attempts: List[str] = []

        for filling in self._filling_modes():
            payload = dict(request)
            payload["type_filling"] = filling
            result = mt5.order_send(payload)
            if result is None:
                self.last_error = f"{failure_prefix} order_send returned None {mt5.last_error()}"
                return None

            retcode = int(getattr(result, "retcode", -1))
            if retcode in success_codes:
                self.last_error = ""
                return result

            comment = str(getattr(result, "comment", "")).strip()
            attempts.append(f"fill={filling} retcode={retcode} comment={comment or '-'}")
            if retcode != invalid_fill:
                self.last_error = f"{failure_prefix} " + " | ".join(attempts)
                return None

        self.last_error = f"{failure_prefix} " + " | ".join(attempts)
        return None

    def _calc_sl_tp(self, direction: str, price: float):
        sl_delta = self.sl_pips * self.PIP_SIZE
        tp_delta = self.tp_pips * self.PIP_SIZE
        if direction == "BUY":
            return price - sl_delta, price + tp_delta
        return price + sl_delta, price - tp_delta

    def _positions_for_symbol(self):
        if not MT5_AVAILABLE:
            return []
        positions = mt5.positions_get(symbol=self.symbol)
        return list(positions or [])

    def has_open_position(self, twin_name: str) -> bool:
        self.sync_positions()
        return twin_name in self._open and self._open[twin_name].status == "OPEN"

    def get_open_position(self, twin_name: str) -> Optional[BrokerOrder]:
        self.sync_positions()
        return self._open.get(twin_name)

    def _match_position(self, order: BrokerOrder):
        positions = self._positions_for_symbol()
        for pos in positions:
            magic = int(getattr(pos, "magic", 0))
            comment = str(getattr(pos, "comment", ""))
            if magic == self._magic(order.twin_name) or order.twin_name in comment:
                return pos
        return None

    def place_order(
        self,
        direction: str,
        price: float,
        twin_name: str,
        confidence: float,
        entropy: float,
        regime: str,
    ) -> Optional[BrokerOrder]:
        self.sync_positions()
        if not self.allow_live:
            self.last_error = "Live routing disabled"
            return None
        if not self.connector.connected or not self.connector.can_trade():
            self.last_error = "MT5 not connected or trading not allowed"
            return None
        twin_name = self._canonical_twin_name(twin_name)
        if twin_name in self._open:
            self.last_error = "Position already open"
            return None

        tick = self.connector.get_current_tick()
        if tick is None:
            self.last_error = "No live tick available"
            return None

        deal_price = tick.ask if direction == "BUY" else tick.bid
        sl_price, tp_price = self._calc_sl_tp(direction, deal_price)
        order_type = mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": self.lot_size,
            "type": order_type,
            "price": deal_price,
            "sl": sl_price,
            "tp": tp_price,
            "deviation": 20,
            "magic": self._magic(twin_name),
            "comment": f"{twin_name} dashboard",
            "type_time": mt5.ORDER_TIME_GTC,
        }
        result = self._send_market_request(request, "order_send failed")
        if result is None:
            return None

        order = BrokerOrder(
            order_id=self._next_id,
            timestamp=datetime.now(),
            symbol=self.symbol,
            direction=direction,
            price_in=float(getattr(result, "price", 0.0) or deal_price),
            sl_price=sl_price,
            tp_price=tp_price,
            lot_size=self.lot_size,
            twin_name=twin_name,
            confidence=confidence,
            entropy=entropy,
            regime=regime,
            broker_ticket=int(getattr(result, "order", 0) or 0),
            position_ticket=int(getattr(result, "deal", 0) or 0),
        )
        self._next_id += 1
        self.orders.append(order)
        self._open[twin_name] = order
        self.last_error = ""
        self.sync_positions()
        return order

    def _position_profit(self, pos) -> float:
        return float(getattr(pos, "profit", 0.0))

    def _position_price_open(self, pos) -> float:
        return float(getattr(pos, "price_open", 0.0))

    def sync_positions(self) -> None:
        if not MT5_AVAILABLE:
            return
        self.balance = float(self.connector.account_summary().get("balance", self.balance))
        still_open: Dict[str, BrokerOrder] = {}

        for twin_name, order in list(self._open.items()):
            pos = self._match_position(order)
            if pos is not None:
                order.position_ticket = int(getattr(pos, "ticket", 0) or order.position_ticket or 0)
                order.price_in = self._position_price_open(pos) or order.price_in
                still_open[twin_name] = order
                continue

            if order.status == "OPEN":
                self._mark_closed_from_history(order)

        self._open = still_open

    def _mark_closed_from_history(self, order: BrokerOrder) -> None:
        order.status = "CLOSED"
        order.closed_at = datetime.now()
        order.close_reason = order.close_reason or "broker closed"

        close_price = None
        profit = None
        if MT5_AVAILABLE:
            frm = order.timestamp - timedelta(days=1)
            to = datetime.now() + timedelta(minutes=1)
            try:
                deals = mt5.history_deals_get(frm, to)
            except Exception:
                deals = None
            if deals:
                candidates = []
                for d in deals:
                    magic = int(getattr(d, "magic", 0))
                    comment = str(getattr(d, "comment", ""))
                    entry = int(getattr(d, "entry", 0))
                    if magic == self._magic(order.twin_name) or order.twin_name in comment:
                        if entry != 0:
                            candidates.append(d)
                if candidates:
                    last = sorted(candidates, key=lambda x: getattr(x, "time", 0))[-1]
                    close_price = float(getattr(last, "price", 0.0) or 0.0)
                    profit = float(getattr(last, "profit", 0.0) or 0.0)

        if close_price is None:
            tick = self.connector.get_current_tick()
            close_price = tick.mid if tick else order.price_in

        order.price_out = close_price
        if order.direction == "BUY":
            pips = (close_price - order.price_in) / self.PIP_SIZE
        else:
            pips = (order.price_in - close_price) / self.PIP_SIZE

        if profit is None:
            profit = pips * self.PIP_SIZE * (order.lot_size / 0.01)

        order.pnl_pips = round(float(pips), 1)
        order.pnl_usd = round(float(profit), 2)
        self.balance = float(self.connector.account_summary().get("balance", self.balance))

    def update_prices(self, current_price: float):
        self.sync_positions()

    def force_buy(self) -> BrokerOrder:
        if not self.allow_live:
            raise RuntimeError("Live routing disabled")
        if not self.connector.connected or not self.connector.can_trade():
            raise RuntimeError("MT5 not connected or trading not allowed")

        tick = self.connector.get_current_tick()
        if tick is None:
            raise RuntimeError(f"No tick data for {self.symbol}")

        order = self.place_order(
            direction="BUY",
            price=float(tick.ask),
            twin_name="Twin-A",
            confidence=1.0,
            entropy=0.0,
            regime="ManualTest",
        )
        if order is None:
            raise RuntimeError(self.last_error or "Force buy failed")
        return order

    def force_sell(self) -> BrokerOrder:
        if not self.allow_live:
            raise RuntimeError("Live routing disabled")
        if not self.connector.connected or not self.connector.can_trade():
            raise RuntimeError("MT5 not connected or trading not allowed")

        tick = self.connector.get_current_tick()
        if tick is None:
            raise RuntimeError(f"No tick data for {self.symbol}")

        order = self.place_order(
            direction="SELL",
            price=float(tick.bid),
            twin_name="Twin-B",
            confidence=1.0,
            entropy=0.0,
            regime="ManualTest",
        )
        if order is None:
            raise RuntimeError(self.last_error or "Force sell failed")
        return order

    def _close_one(self, order: BrokerOrder) -> bool:
        if not MT5_AVAILABLE:
            self.last_error = "MetaTrader5 package not available"
            return False

        pos = self._match_position(order)
        if pos is None:
            self._mark_closed_from_history(order)
            self._open.pop(order.twin_name, None)
            return True

        close_type = mt5.ORDER_TYPE_SELL if order.direction == "BUY" else mt5.ORDER_TYPE_BUY
        tick = self.connector.get_current_tick()
        if tick is None:
            self.last_error = "No live tick available for close"
            return False

        price = tick.bid if order.direction == "BUY" else tick.ask
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": float(getattr(pos, "volume", order.lot_size)),
            "type": close_type,
            "position": int(getattr(pos, "ticket", 0)),
            "price": price,
            "deviation": 20,
            "magic": self._magic(order.twin_name),
            "comment": f"{order.twin_name} manual close",
            "type_time": mt5.ORDER_TIME_GTC,
        }
        result = self._send_market_request(request, "close failed")
        if result is None:
            return False

        order.close_reason = "manual close"
        self._mark_closed_from_history(order)
        self._open.pop(order.twin_name, None)
        self.last_error = ""
        return True

    def close_all(self, current_price: float):
        for order in list(self._open.values()):
            self._close_one(order)

    def close_twin(self, twin_name: str, current_price: float):
        order = self._open.get(self._canonical_twin_name(twin_name))
        if order is not None:
            self._close_one(order)

    def closed_orders(self) -> List[BrokerOrder]:
        return [o for o in self.orders if o.status == "CLOSED"]

    def equity(self, current_price: float) -> float:
        account = self.connector.account_summary()
        if account:
            return round(float(account.get("equity", self.balance)), 2)

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
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / (peak + 1e-10)
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
        stats = {
            "Twin-A": {"trades": 0, "wins": 0, "pnl": 0.0, "pips": 0.0},
            "Twin-B": {"trades": 0, "wins": 0, "pnl": 0.0, "pips": 0.0},
        }
        for o in self.closed_orders():
            s = stats[o.twin_name]
            s["trades"] += 1
            s["pnl"] += o.pnl_usd
            s["pips"] += o.pnl_pips
            if o.pnl_usd > 0:
                s["wins"] += 1
        return stats

    def recent_orders_df(self, n: int = 30):
        import pandas as pd

        recent = self.orders[-n:]
        if not recent:
            return pd.DataFrame()
        return pd.DataFrame(
            [
                {
                    "ID": o.order_id,
                    "Time": o.timestamp.strftime("%H:%M:%S"),
                    "Twin": o.twin_name,
                    "Dir": o.direction,
                    "Price In": f"{o.price_in:.2f}",
                    "Price Out": f"{o.price_out:.2f}" if o.price_out else "OPEN",
                    "PnL $": f"{o.pnl_usd:+.2f}" if o.status == "CLOSED" else "—",
                    "Pips": f"{o.pnl_pips:+.1f}" if o.status == "CLOSED" else "—",
                    "Status": o.status,
                    "Reason": o.close_reason or "—",
                    "Regime": o.regime,
                    "Conf": f"{o.confidence:.3f}",
                    "Broker Ticket": o.position_ticket or o.broker_ticket or "—",
                }
                for o in reversed(recent)
            ]
        )
