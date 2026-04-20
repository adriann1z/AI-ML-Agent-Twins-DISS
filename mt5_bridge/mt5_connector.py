from __future__ import annotations

"""
Robust MT5 connector for the dissertation dashboard.

Highlights:
- Supports explicit login with dashboard credentials.
- Can target a specific MT5 terminal executable path.
- Handles multiple installed terminals more safely.
- Resolves preferred symbol variants while preferring exact XAUUSD.
- Exposes detailed diagnostics so the UI can explain fallback clearly.
"""

import glob
import os
import platform
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import MetaTrader5 as mt5

    MT5_AVAILABLE = True
except ImportError:  # pragma: no cover
    MT5_AVAILABLE = False
    mt5 = None


@dataclass
class LiveTick:
    timestamp: datetime
    symbol: str
    bid: float
    ask: float
    mid: float
    spread: float
    volume: float


class MT5Connector:
    def __init__(
        self,
        login: Optional[int] = None,
        password: Optional[str] = None,
        server: Optional[str] = None,
        preferred_symbol: str = "XAUUSD",
        terminal_path: Optional[str] = None,
        fallback_synthetic: bool = True,
    ):
        self.login = int(login) if login not in (None, "") else None
        self.password = password or None
        self.server = (server or "").strip() or None
        self.preferred_symbol = (preferred_symbol or "XAUUSD").strip() or "XAUUSD"
        self.terminal_path = self._normalize_path(terminal_path)
        self.fallback = fallback_synthetic

        self.connected = False
        self.using_live = False
        self.symbol = self.preferred_symbol
        self.status_msg = "Not connected"
        self.last_error: Optional[tuple] = None
        self.connection_details: Dict[str, Any] = {}

    def _normalize_path(self, value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        value = str(value).strip().strip('"')
        return value or None

    def _candidate_paths(self) -> List[Optional[str]]:
        paths: List[Optional[str]] = []
        if self.terminal_path:
            paths.append(self.terminal_path)

        if platform.system().lower().startswith("win"):
            patterns = [
                r"C:\Program Files\*\terminal64.exe",
                r"C:\Program Files (x86)\*\terminal64.exe",
                r"C:\Users\*\AppData\Roaming\MetaQuotes\Terminal\*\terminal64.exe",
            ]
            for pattern in patterns:
                for candidate in glob.glob(pattern):
                    if candidate not in paths:
                        paths.append(candidate)

        if len(paths) > 1:
            paths = sorted(paths, key=self._path_priority)

        paths.append(None)
        return paths

    def _path_priority(self, path: Optional[str]) -> tuple:
        if not path:
            return (9, 9, "")

        norm = str(path).lower()

        if self.terminal_path and os.path.normcase(path) == os.path.normcase(self.terminal_path):
            return (0, 0, norm)

        server = (self.server or "").lower()
        wants_ic_markets = "icmarkets" in server or "ic markets" in server
        wants_fxpro = "fxpro" in server or "fx pro" in server

        if wants_ic_markets:
            broker_rank = 0 if "ic markets" in norm or "icmarkets" in norm else 5
        elif wants_fxpro:
            broker_rank = 0 if "fxpro" in norm or "fx pro" in norm else 5
        else:
            broker_rank = 0 if "ic markets" in norm or "icmarkets" in norm else 1

        roaming_rank = 1 if "metaquotes\\terminal" in norm else 0
        return (1, broker_rank + roaming_rank, norm)

    def _shutdown(self) -> None:
        if MT5_AVAILABLE:
            try:
                mt5.shutdown()
            except Exception:
                pass

    def _initialise_once(self, path: Optional[str]) -> bool:
        kwargs: Dict[str, Any] = {}
        if path:
            kwargs["path"] = path
        ok = mt5.initialize(**kwargs)
        self.last_error = mt5.last_error()
        return bool(ok)

    def _account_matches(self, info: Any) -> bool:
        if info is None:
            return False
        login_ok = True if self.login is None else int(getattr(info, "login", -1)) == int(self.login)
        server_ok = True
        if self.server:
            server_ok = str(getattr(info, "server", "")).strip().lower() == self.server.strip().lower()
        return login_ok and server_ok

    def _attempt_login(self) -> bool:
        if self.login is None or not self.password or not self.server:
            return False
        ok = mt5.login(login=int(self.login), password=self.password, server=self.server)
        self.last_error = mt5.last_error()
        return bool(ok)

    def _attempt_initialize_with_credentials(self, path: Optional[str]) -> bool:
        if self.login is None or not self.password or not self.server:
            return False
        self._shutdown()
        kwargs: Dict[str, Any] = {
            "login": int(self.login),
            "password": self.password,
            "server": self.server,
        }
        if path:
            kwargs["path"] = path
        try:
            ok = mt5.initialize(**kwargs)
        except TypeError:
            ok = False
        self.last_error = mt5.last_error()
        return bool(ok)

    def _resolve_symbol(self) -> bool:
        preferred = self.preferred_symbol.strip()
        candidates = [
            preferred,
            preferred.upper(),
            "XAUUSD",
            "XAUUSD.",
            "XAUUSDm",
            "XAUUSD.a",
            "GOLD",
            "GOLD.",
        ]

        for symbol in candidates:
            info = mt5.symbol_info(symbol)
            if info is not None:
                mt5.symbol_select(symbol, True)
                self.symbol = symbol
                return True

        symbols = mt5.symbols_get()
        if not symbols:
            return False
        gold_like = []
        for item in symbols:
            name = str(getattr(item, "name", ""))
            u = name.upper()
            if "XAUUSD" in u or u.startswith("GOLD"):
                gold_like.append(name)
        if not gold_like:
            return False
        chosen = sorted(gold_like, key=lambda x: (0 if x.upper() == preferred.upper() else 1, len(x)))[0]
        mt5.symbol_select(chosen, True)
        self.symbol = chosen
        return True

    def connect(self) -> bool:
        self.connected = False
        self.using_live = False
        self.connection_details = {"attempts": []}

        if not MT5_AVAILABLE:
            self.status_msg = "MetaTrader5 package not installed"
            return False

        for path in self._candidate_paths():
            attempt: Dict[str, Any] = {"path": path or "AUTO"}
            self._shutdown()
            if not self._initialise_once(path):
                attempt["status"] = f"initialize failed {self.last_error}"
                self.connection_details["attempts"].append(attempt)
                continue

            term = mt5.terminal_info()
            acct = mt5.account_info()
            attempt["terminal"] = getattr(term, "name", None)
            attempt["company"] = getattr(term, "company", None)
            attempt["server_before_login"] = getattr(acct, "server", None) if acct else None
            attempt["login_before_login"] = getattr(acct, "login", None) if acct else None

            logged_in = False
            if acct is not None and self._account_matches(acct):
                logged_in = True
                attempt["status"] = "using existing matching session"
            else:
                if self.login is not None and self.password and self.server:
                    logged_in = self._attempt_login()
                    attempt["login_attempt"] = "mt5.login"
                    attempt["login_result"] = logged_in
                    attempt["login_error"] = self.last_error
                    if not logged_in:
                        logged_in = self._attempt_initialize_with_credentials(path)
                        attempt["initialize_with_credentials"] = logged_in
                        attempt["initialize_with_credentials_error"] = self.last_error
                else:
                    logged_in = acct is not None
                    attempt["status"] = "no credentials supplied; using existing session if available"

            acct = mt5.account_info()
            if not logged_in and acct is None:
                attempt["status"] = f"authorization failed {self.last_error}"
                self.connection_details["attempts"].append(attempt)
                continue

            if acct is None:
                attempt["status"] = "connected but no account info"
                self.connection_details["attempts"].append(attempt)
                continue

            if self.login is not None and int(getattr(acct, "login", -1)) != int(self.login):
                attempt["status"] = f"wrong account attached {getattr(acct, 'login', None)}"
                self.connection_details["attempts"].append(attempt)
                continue

            if self.server and str(getattr(acct, "server", "")).strip().lower() != self.server.strip().lower():
                attempt["status"] = f"wrong server attached {getattr(acct, 'server', None)}"
                self.connection_details["attempts"].append(attempt)
                continue

            if not self._resolve_symbol():
                attempt["status"] = "connected but could not resolve symbol"
                self.connection_details["attempts"].append(attempt)
                continue

            symbol_info = mt5.symbol_info(self.symbol)
            if symbol_info is None:
                attempt["status"] = "resolved symbol but symbol_info failed"
                self.connection_details["attempts"].append(attempt)
                continue

            tick = mt5.symbol_info_tick(self.symbol)
            self.last_error = mt5.last_error()
            self.connected = tick is not None
            self.using_live = self.connected
            if not self.connected:
                attempt["status"] = f"symbol resolved but no tick available {self.last_error}"
                self.connection_details["attempts"].append(attempt)
                continue

            mode = "DEMO" if int(getattr(acct, "trade_mode", 0)) == 0 else "LIVE"
            self.status_msg = (
                f"Connected · {getattr(acct, 'server', 'Unknown')} · {mode} · "
                f"Login {getattr(acct, 'login', '—')} · Symbol {self.symbol}"
            )
            attempt["status"] = "connected"
            self.connection_details["selected_path"] = path
            self.connection_details["account_login"] = getattr(acct, "login", None)
            self.connection_details["account_server"] = getattr(acct, "server", None)
            self.connection_details["symbol"] = self.symbol
            self.connection_details["terminal"] = self.terminal_summary()
            self.connection_details["account"] = self.account_summary()
            self.connection_details["attempts"].append(attempt)
            return True

        self._shutdown()
        self.status_msg = f"terminal64.exe init failed {self.last_error}"
        return False

    def disconnect(self) -> None:
        self._shutdown()
        self.connected = False
        self.using_live = False
        self.status_msg = "Disconnected"

    def can_trade(self) -> bool:
        if not self.connected or not MT5_AVAILABLE:
            return False
        acct = mt5.account_info()
        term = mt5.terminal_info()
        return bool(getattr(acct, "trade_allowed", False)) and bool(getattr(term, "trade_allowed", False))

    def get_current_tick(self) -> Optional[LiveTick]:
        if not self.connected or not MT5_AVAILABLE:
            return None
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            self.last_error = mt5.last_error()
            return None
        bid = float(getattr(tick, "bid", 0.0))
        ask = float(getattr(tick, "ask", 0.0))
        mid = (bid + ask) / 2 if bid and ask else bid or ask
        return LiveTick(
            timestamp=datetime.fromtimestamp(int(getattr(tick, "time", datetime.now().timestamp()))),
            symbol=self.symbol,
            bid=bid,
            ask=ask,
            mid=mid,
            spread=max(ask - bid, 0.0),
            volume=float(getattr(tick, "volume", 0.0)),
        )

    def get_recent_prices(self, n: int = 500) -> Optional[np.ndarray]:
        if not self.connected or not MT5_AVAILABLE:
            return None
        bars = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M1, 0, n)
        if bars is None or len(bars) == 0:
            return None
        return np.array([float(b[4]) for b in bars], dtype=float)

    def get_recent_bars(self, n: int = 120, timeframe: Optional[int] = None) -> List[Dict[str, Any]]:
        if not self.connected or not MT5_AVAILABLE:
            return []
        tf = timeframe if timeframe is not None else mt5.TIMEFRAME_M1
        bars = mt5.copy_rates_from_pos(self.symbol, tf, 0, n)
        if bars is None:
            return []
        out: List[Dict[str, Any]] = []
        for b in bars:
            out.append(
                {
                    "time": datetime.fromtimestamp(int(b[0])),
                    "open": float(b[1]),
                    "high": float(b[2]),
                    "low": float(b[3]),
                    "close": float(b[4]),
                    "tick_volume": float(b[5]),
                }
            )
        return out

    def account_summary(self) -> Dict[str, Any]:
        if not MT5_AVAILABLE:
            return {}
        info = mt5.account_info()
        if info is None:
            return {}
        return {
            "login": getattr(info, "login", None),
            "balance": float(getattr(info, "balance", 0.0)),
            "equity": float(getattr(info, "equity", 0.0)),
            "margin": float(getattr(info, "margin", 0.0)),
            "free_margin": float(getattr(info, "margin_free", 0.0)),
            "profit": float(getattr(info, "profit", 0.0)),
            "currency": getattr(info, "currency", ""),
            "leverage": getattr(info, "leverage", ""),
            "server": getattr(info, "server", ""),
            "mode": "DEMO" if int(getattr(info, "trade_mode", 0)) == 0 else "LIVE",
            "trade_allowed": bool(getattr(info, "trade_allowed", False)),
        }

    def terminal_summary(self) -> Dict[str, Any]:
        if not MT5_AVAILABLE:
            return {}
        info = mt5.terminal_info()
        if info is None:
            return {}
        return {
            "name": getattr(info, "name", ""),
            "company": getattr(info, "company", ""),
            "connected": bool(getattr(info, "connected", False)),
            "trade_allowed": bool(getattr(info, "trade_allowed", False)),
            "dlls_allowed": bool(getattr(info, "dlls_allowed", False)),
            "path": self.connection_details.get("selected_path") or self.terminal_path or "AUTO",
        }
