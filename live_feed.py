from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    mt5 = None


@dataclass
class TickData:
    bid: float
    ask: float
    mid: float
    timestamp: datetime


class LiveFeedAdapter:
    def __init__(self, connector=None, use_synthetic_fallback: bool = True):
        self.connector = connector
        self.use_synthetic_fallback = use_synthetic_fallback

    def get_current_tick(self) -> Optional[TickData]:
        # ✅ Try MT5 first
        if MT5_AVAILABLE and self.connector and self.connector.connected:
            try:
                symbol = self.connector.symbol
                tick = mt5.symbol_info_tick(symbol)

                if tick is not None:
                    bid = float(tick.bid)
                    ask = float(tick.ask)
                    mid = (bid + ask) / 2.0

                    return TickData(
                        bid=bid,
                        ask=ask,
                        mid=mid,
                        timestamp=datetime.now()
                    )
            except Exception:
                pass

        # ⚠️ Fallback (synthetic)
        if self.use_synthetic_fallback:
            import random
            price = 1900 + random.uniform(-5, 5)
            return TickData(
                bid=price - 0.1,
                ask=price + 0.1,
                mid=price,
                timestamp=datetime.now()
            )

        return None