"""Tests for the shared market-data helper layer.

``src.market_data`` is the source-resolution + normalization layer shared by
the MCP server and the agent ``get_market_data`` tool. It shipped (with the
#270 global data layer) without dedicated tests. These cover the
network-free logic: source detection, row capping, JSON-safety, and the
``fetch_market_data`` orchestration via an injected stub loader.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.market_data import (
    DEFAULT_MAX_ROWS,
    _json_safe,
    cap_rows,
    detect_source,
    fetch_market_data,
    fetch_market_data_json,
)


# --------------------------------------------------------------------------
# detect_source
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "code,expected",
    [
        ("600519.SH", "tencent"),
        ("000001.SZ", "tencent"),
        ("430139.BJ", "tencent"),
        ("AAPL.US", "yahoo"),
        ("700.HK", "tencent"),
        ("00700.HK", "tencent"),
        ("RELIANCE.NS", "yahoo"),  # India NSE
        ("TCS.NS", "yahoo"),
        ("M&M.NS", "yahoo"),  # ampersand in ticker
        ("BAJAJ-AUTO.NS", "yahoo"),  # hyphen in ticker
        ("500325.BO", "yahoo"),  # India BSE (numeric scrip code)
        ("TD.TO", "yahoo"),  # Canada TSX
        ("BBD-B.TO", "yahoo"),  # hyphenated TSX class symbol
        ("PNG.V", "yahoo"),  # Canada TSX Venture
        ("BTC-USDT", "okx"),
        ("ETH/USDT", "ccxt"),
        ("EUR/USD", "mt5"),  # forex pair → mt5 chain head (registry fallback)
        ("XAU/USD", "mt5"),  # metals share the forex route
        ("EURUSD.FX", "mt5"),
        ("XAUUSD.FX", "mt5"),
        ("local:my_file", "local"),
        # Yahoo futures / forex suffix conventions (#718) — must not fall to the
        # ``tushare`` default (which routed them to China-market loaders).
        ("GC=F", "yahoo"),  # gold future
        ("CL=F", "yahoo"),  # crude future
        ("EURUSD=X", "yahoo"),  # forex pair
        ("JPY=X", "yahoo"),
        ("something_weird", "tushare"),  # documented fallback
    ],
)
def test_detect_source(code: str, expected: str) -> None:
    assert detect_source(code) == expected


def test_yahoo_loader_accepts_futures_and_forex_suffixes() -> None:
    """The yahoo direct loader must accept =F/=X, not just equity suffixes (#718)."""
    from backtest.loaders.yahoo_loader import _is_supported

    assert _is_supported("GC=F") is True
    assert _is_supported("EURUSD=X") is True
    assert _is_supported("AAPL.US") is True  # unchanged
    assert _is_supported("TD.TO") is True
    assert _is_supported("PNG.V") is True
    assert _is_supported("600519.SH") is False  # A-share still not yahoo


def test_fetch_market_data_auto_routes_yahoo_suffix_symbols() -> None:
    """auto mode groups GC=F/EURUSD=X under yahoo, not the tushare/China chain (#718)."""
    seen_sources: list[str] = []

    class _StubLoader:
        def fetch(self, codes, start, end, *, interval="1D"):  # noqa: ANN001
            index = pd.DatetimeIndex(pd.to_datetime(["2024-01-02"]))
            return {
                code: pd.DataFrame(
                    {"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [0.0]},
                    index=index,
                )
                for code in codes
            }

    def _resolver(source: str):
        seen_sources.append(source)
        return _StubLoader

    out = fetch_market_data(
        codes=["GC=F", "EURUSD=X", "TD.TO", "PNG.V"],
        start_date="2024-01-01",
        end_date="2024-01-03",
        source="auto",
        loader_resolver=_resolver,
    )

    assert "_unresolved" not in out
    assert all(code in out for code in ("GC=F", "EURUSD=X", "TD.TO", "PNG.V"))
    # First source tried must be yahoo (not tushare/akshare from the China chain).
    assert seen_sources and seen_sources[0] == "yahoo"


# --------------------------------------------------------------------------
# cap_rows
# --------------------------------------------------------------------------


def test_cap_rows_passthrough_when_under_limit() -> None:
    rows = [{"a": i} for i in range(3)]
    assert cap_rows(rows, 250) is rows


def test_cap_rows_zero_means_no_cap() -> None:
    rows = [{"a": i} for i in range(1000)]
    assert cap_rows(rows, 0) is rows


def test_cap_rows_negative_falls_back_to_default() -> None:
    rows = [{"a": i} for i in range(DEFAULT_MAX_ROWS + 10)]
    out = cap_rows(rows, -5)
    # Negative max_rows is treated as DEFAULT_MAX_ROWS -> truncated payload.
    assert isinstance(out, dict)
    assert out["truncated"] is True


def test_cap_rows_samples_with_stride_and_pins_last() -> None:
    rows = [{"a": i} for i in range(10)]
    out = cap_rows(rows, 4)
    assert isinstance(out, dict)
    assert out["rows"] == 10
    assert out["truncated"] is True
    # Even stride of ceil(10/4)=3 plus the pinned final bar.
    assert out["data"][0] == {"a": 0}
    assert out["data"][-1] == {"a": 9}  # last bar always pinned
    assert out["returned"] == len(out["data"])


# --------------------------------------------------------------------------
# _json_safe
# --------------------------------------------------------------------------


def test_json_safe_non_finite_becomes_none() -> None:
    assert _json_safe(float("nan")) is None
    assert _json_safe(float("inf")) is None
    assert _json_safe(float("-inf")) is None


def test_json_safe_timestamp_isoformat() -> None:
    assert _json_safe(pd.Timestamp("2026-01-01")) == "2026-01-01T00:00:00"


def test_json_safe_numpy_scalar_unwrapped() -> None:
    out = _json_safe(np.int64(5))
    assert out == 5
    assert not isinstance(out, np.integer)


def test_json_safe_plain_value_passthrough() -> None:
    assert _json_safe("hello") == "hello"
    assert _json_safe(3.5) == 3.5


# --------------------------------------------------------------------------
# fetch_market_data (stub loader — no network)
# --------------------------------------------------------------------------


class _StubLoader:
    """Returns a fixed 2-row OHLCV frame for every requested code."""

    def __init__(self) -> None:
        pass

    def fetch(self, codes, start_date, end_date, interval="1D"):
        idx = pd.to_datetime(["2026-01-01", "2026-01-02"])
        idx.name = "trade_date"
        return {
            code: pd.DataFrame({"close": [1.0, 2.0], "volume": [100, 200]}, index=idx)
            for code in codes
        }


class _BadLoader:
    def __init__(self) -> None:
        pass

    def fetch(self, *args, **kwargs):
        raise RuntimeError("loader exploded")


class _PartialLoader:
    """Returns data for only the first requested code."""

    def __init__(self) -> None:
        pass

    def fetch(self, codes, start_date, end_date, interval="1D"):
        idx = pd.to_datetime(["2026-01-01"])
        idx.name = "trade_date"
        return {codes[0]: pd.DataFrame({"close": [1.0]}, index=idx)}


class _LocalAliasLoader:
    """Mirrors the local loader, which returns keys without ``local:``."""

    def fetch(self, codes, start_date, end_date, interval="1D"):
        idx = pd.to_datetime(["2026-01-01"])
        idx.name = "trade_date"
        clean = codes[0].split(":", 1)[-1]
        return {clean: pd.DataFrame({"close": [1.0]}, index=idx)}


class _TorontoOnlyLoader:
    """Serves only ``.TO`` symbols — mimics Yahoo after a listing moved
    TSX Venture -> TSX main (HIVE.V 404s, HIVE.TO resolves)."""

    def fetch(self, codes, start_date, end_date, interval="1D"):
        idx = pd.to_datetime(["2026-01-01", "2026-01-02"])
        idx.name = "trade_date"
        out = {}
        for code in codes:
            if code.upper().endswith(".TO"):
                out[code] = pd.DataFrame(
                    {"close": [4.24, 4.30], "volume": [100, 200]}, index=idx
                )
        return out


class _VentureOnlyLoader:
    """Serves only ``.V`` symbols — the reverse (TSX -> TSX Venture move)."""

    def fetch(self, codes, start_date, end_date, interval="1D"):
        idx = pd.to_datetime(["2026-01-01", "2026-01-02"])
        idx.name = "trade_date"
        out = {}
        for code in codes:
            if code.upper().endswith(".V"):
                out[code] = pd.DataFrame(
                    {"close": [0.55, 0.60], "volume": [300, 400]}, index=idx
                )
        return out


def test_fetch_explicit_source_normalizes_rows() -> None:
    out = fetch_market_data(
        codes=["AAPL.US"],
        start_date="2026-01-01",
        end_date="2026-01-02",
        source="yahoo",
        loader_resolver=lambda src: _StubLoader,
    )
    assert "AAPL.US" in out
    rows = out["AAPL.US"]
    assert rows[0]["trade_date"] == "2026-01-01T00:00:00"  # index reset + isoformat
    assert rows[0]["close"] == 1.0


def test_fetch_auto_groups_by_detected_source() -> None:
    seen: dict[str, list[str]] = {}

    def resolver(src: str):
        seen[src] = []
        return _StubLoader

    out = fetch_market_data(
        codes=["AAPL.US", "BTC-USDT"],
        start_date="2026-01-01",
        end_date="2026-01-02",
        source="auto",
        loader_resolver=resolver,
    )
    # AAPL.US -> yahoo, BTC-USDT -> okx: two distinct loader groups resolved.
    assert set(seen) == {"yahoo", "okx"}
    assert "AAPL.US" in out and "BTC-USDT" in out


def test_fetch_auto_hk_walks_hk_chain_not_us_chain() -> None:
    """HK symbols must degrade through the hk_equity chain, not the US one.

    A source-name-only chain lookup would match HK's yahoo membership against
    the us_equity chain first, where the attempt budget exhausts on the
    US-only stooq/sina loaders and never reaches eastmoney/akshare.
    """
    from backtest.loaders.base import NoAvailableSourceError

    attempts: list[str] = []

    def resolver(src: str):
        attempts.append(src)
        if src == "eastmoney":
            return _StubLoader
        raise NoAvailableSourceError(f"{src} unavailable in test")

    out = fetch_market_data(
        codes=["00700.HK"],
        start_date="2026-01-01",
        end_date="2026-01-02",
        source="auto",
        loader_resolver=resolver,
    )
    assert attempts[:2] == ["tencent", "eastmoney"]
    assert "stooq" not in attempts and "sina" not in attempts
    assert "_unresolved" not in out
    assert "00700.HK" in out


def test_fetch_auto_hk_akshare_reachable_within_default_budget() -> None:
    """akshare (Eastmoney-backed HK daily) must be reachable within the
    default ``max_fallback_attempts`` when every earlier HK source is down."""
    from backtest.loaders.base import NoAvailableSourceError

    attempts: list[str] = []

    def resolver(src: str):
        attempts.append(src)
        if src == "akshare":
            return _StubLoader
        raise NoAvailableSourceError(f"{src} unavailable in test")

    out = fetch_market_data(
        codes=["09988.HK"],
        start_date="2026-01-01",
        end_date="2026-01-02",
        source="auto",
        loader_resolver=resolver,
    )
    assert "_unresolved" not in out
    assert "09988.HK" in out
    assert attempts[-1] == "akshare"
    assert len(attempts) <= 5


def test_fetch_auto_india_walks_india_chain() -> None:
    """India symbols must degrade through the india_equity chain (chain
    selection is market-aware for every market, not just HK)."""
    from backtest.loaders.base import NoAvailableSourceError

    attempts: list[str] = []

    def resolver(src: str):
        attempts.append(src)
        if src == "yfinance":
            return _StubLoader
        raise NoAvailableSourceError(f"{src} unavailable in test")

    out = fetch_market_data(
        codes=["RELIANCE.NS"],
        start_date="2026-01-01",
        end_date="2026-01-02",
        source="auto",
        loader_resolver=resolver,
    )
    assert attempts == ["yahoo", "yfinance"]
    assert "_unresolved" not in out
    assert "RELIANCE.NS" in out


def test_fetch_auto_us_still_walks_us_chain() -> None:
    """US routing is unchanged by the market-aware chain selection."""
    from backtest.loaders.base import NoAvailableSourceError

    attempts: list[str] = []

    def resolver(src: str):
        attempts.append(src)
        if src == "stooq":
            return _StubLoader
        raise NoAvailableSourceError(f"{src} unavailable in test")

    out = fetch_market_data(
        codes=["AAPL.US"],
        start_date="2026-01-01",
        end_date="2026-01-02",
        source="auto",
        loader_resolver=resolver,
    )
    assert attempts[:2] == ["yahoo", "stooq"]
    assert "_unresolved" not in out
    assert "AAPL.US" in out


def test_fetch_loader_error_falls_through_to_unresolved() -> None:
    out = fetch_market_data(
        codes=["X.US"],
        start_date="2026-01-01",
        end_date="2026-01-02",
        source="yahoo",
        loader_resolver=lambda src: _BadLoader,
    )
    assert out["_unresolved"] == ["X.US"]


def test_fetch_missing_symbol_listed_as_unresolved() -> None:
    out = fetch_market_data(
        codes=["A.US", "B.US"],
        start_date="2026-01-01",
        end_date="2026-01-02",
        source="yahoo",
        loader_resolver=lambda src: _PartialLoader,
    )
    assert "A.US" in out
    assert out["_unresolved"] == ["B.US"]


def test_fetch_local_result_alias_is_not_unresolved() -> None:
    out = fetch_market_data(
        codes=["local:AAPL.US"],
        start_date="2026-01-01",
        end_date="2026-01-02",
        source="auto",
        loader_resolver=lambda src: _LocalAliasLoader,
    )

    assert "AAPL.US" in out
    assert "_unresolved" not in out


# --------------------------------------------------------------------------
# Canadian venue-alias fallback (.V <-> .TO) — moved listings resolve via the
# sibling venue's symbol instead of landing in _unresolved.
# --------------------------------------------------------------------------


def test_fetch_canadian_v_falls_back_to_to_sibling() -> None:
    """HIVE.V 404s (listing moved to TSX) -> HIVE.TO served under HIVE.V."""
    out = fetch_market_data(
        codes=["HIVE.V"],
        start_date="2026-01-01",
        end_date="2026-01-02",
        source="auto",
        loader_resolver=lambda src: _TorontoOnlyLoader,
        include_provenance=True,
    )
    assert "_unresolved" not in out
    assert "HIVE.V" in out
    assert out["HIVE.V"][0]["close"] == 4.24
    prov = out["_provenance"]["HIVE.V"]
    assert prov["venue_fallback"] is True
    assert prov["resolved_symbol"] == "HIVE.TO"


def test_fetch_canadian_to_falls_back_to_v_sibling() -> None:
    """Reverse direction: a .TO symbol whose only live venue is .V."""
    out = fetch_market_data(
        codes=["HIVE.TO"],
        start_date="2026-01-01",
        end_date="2026-01-02",
        source="auto",
        loader_resolver=lambda src: _VentureOnlyLoader,
        include_provenance=True,
    )
    assert "_unresolved" not in out
    assert "HIVE.TO" in out
    assert out["HIVE.TO"][0]["close"] == 0.55
    prov = out["_provenance"]["HIVE.TO"]
    assert prov["venue_fallback"] is True
    assert prov["resolved_symbol"] == "HIVE.V"


def test_fetch_canadian_aliases_sibling_already_resolved() -> None:
    """When both venues are requested and only the sibling resolves, the
    missing one is aliased to the resolved sibling's bars with no extra fetch."""
    calls: list[list[str]] = []

    class _RecordingTorontoLoader:
        def fetch(self, codes, start_date, end_date, interval="1D"):
            calls.append(list(codes))
            idx = pd.to_datetime(["2026-01-01", "2026-01-02"])
            idx.name = "trade_date"
            return {
                code: pd.DataFrame(
                    {"close": [4.24, 4.30], "volume": [100, 200]}, index=idx
                )
                for code in codes
                if code.upper().endswith(".TO")
            }

    out = fetch_market_data(
        codes=["HIVE.V", "HIVE.TO"],
        start_date="2026-01-01",
        end_date="2026-01-02",
        source="auto",
        loader_resolver=lambda src: _RecordingTorontoLoader,
        include_provenance=True,
    )
    assert "_unresolved" not in out
    assert "HIVE.V" in out and "HIVE.TO" in out
    assert out["HIVE.V"] == out["HIVE.TO"]  # aliased — identical bars
    # The .V symbol must be aliased from the already-resolved .TO sibling:
    # exactly one group fetch (both codes together), no separate re-fetch of
    # the sibling after the fact.
    assert len(calls) == 1
    assert set(calls[0]) == {"HIVE.V", "HIVE.TO"}
    assert out["_provenance"]["HIVE.V"]["resolved_symbol"] == "HIVE.TO"
    assert out["_provenance"]["HIVE.V"]["venue_fallback"] is True


def test_fetch_non_canadian_symbol_unaffected_by_venue_fallback() -> None:
    """A non-Canadian symbol that fails must stay _unresolved — the sibling
    fallback only ever fires for .TO/.V symbols."""
    out = fetch_market_data(
        codes=["AAPL.US"],
        start_date="2026-01-01",
        end_date="2026-01-02",
        source="auto",
        loader_resolver=lambda src: _TorontoOnlyLoader,
    )
    assert out["_unresolved"] == ["AAPL.US"]


def test_ca_venue_sibling_swaps_suffix_only() -> None:
    """Unit check for the sibling helper, incl. hyphenated class bases."""
    from src.market_data import _ca_venue_sibling

    assert _ca_venue_sibling("HIVE.V") == "HIVE.TO"
    assert _ca_venue_sibling("HIVE.TO") == "HIVE.V"
    assert _ca_venue_sibling("BBD-B.TO") == "BBD-B.V"
    assert _ca_venue_sibling("PNG.V") == "PNG.TO"
    assert _ca_venue_sibling("AAPL.US") is None
    assert _ca_venue_sibling("local:HIVE.V") is None
    assert _ca_venue_sibling("BTC-USDT") is None


# --------------------------------------------------------------------------
# fetch_market_data_json
# --------------------------------------------------------------------------


def test_fetch_json_is_strict_and_parseable() -> None:
    payload = fetch_market_data_json(
        codes=["AAPL.US"],
        start_date="2026-01-01",
        end_date="2026-01-02",
        source="yahoo",
        loader_resolver=lambda src: _StubLoader,
    )
    parsed = json.loads(payload)  # must be valid JSON
    assert "AAPL.US" in parsed


def test_fetch_json_rejects_nan_via_allow_nan_false() -> None:
    class _NanLoader:
        def __init__(self) -> None:
            pass

        def fetch(self, codes, start_date, end_date, interval="1D"):
            idx = pd.to_datetime(["2026-01-01"])
            idx.name = "trade_date"
            # A NaN close must be sanitized to null by _json_safe, so strict
            # JSON (allow_nan=False) still succeeds.
            return {codes[0]: pd.DataFrame({"close": [float("nan")]}, index=idx)}

    payload = fetch_market_data_json(
        codes=["A.US"],
        start_date="2026-01-01",
        end_date="2026-01-02",
        source="yahoo",
        loader_resolver=lambda src: _NanLoader,
    )
    parsed = json.loads(payload)
    assert parsed["A.US"][0]["close"] is None
