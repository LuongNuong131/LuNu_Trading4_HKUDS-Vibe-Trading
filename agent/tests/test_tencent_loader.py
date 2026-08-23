"""Tests for Tencent's daily-only market-data contract."""

from __future__ import annotations

import json
import urllib.request

import pandas as pd
import pytest

from backtest.loaders import tencent_loader


class _FakeResponse:
    def __init__(self, payload: str) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload.encode("utf-8")

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc) -> bool:
        return False


def _hk_kline_payload(tencent_code: str) -> str:
    return json.dumps(
        {
            "code": 0,
            "data": {
                tencent_code: {
                    "day": [
                        ["2026-01-05", "466.4", "471.8", "475.0", "462.8", "31791979"],
                        ["2026-01-06", "470.0", "475.2", "479.8", "462.0", "31100240"],
                    ]
                }
            },
        }
    )


def _patch_http(monkeypatch, urls: list[str], payload: str) -> None:
    def fake_urlopen(req, timeout=None, **kwargs):  # noqa: ANN001, ANN002
        urls.append(req.full_url)
        return _FakeResponse(payload)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(
        tencent_loader,
        "cached_loader_fetch",
        lambda **kwargs: kwargs["fetch"](),
    )


def test_intraday_request_does_not_return_daily_bars(monkeypatch) -> None:
    calls: list[str] = []
    daily = pd.DataFrame(
        {
            "open": [10.0],
            "high": [11.0],
            "low": [9.0],
            "close": [10.5],
            "volume": [100.0],
        },
        index=pd.DatetimeIndex([pd.Timestamp("2026-01-05")]),
    )
    loader = tencent_loader.DataLoader()
    monkeypatch.setattr(
        tencent_loader,
        "cached_loader_fetch",
        lambda **kwargs: kwargs["fetch"](),
    )
    monkeypatch.setattr(
        loader,
        "_fetch_one",
        lambda code, start, end: calls.append(code) or daily,
    )

    result = loader.fetch(
        ["600519.SH"],
        "2026-01-01",
        "2026-01-31",
        interval="1m",
    )

    assert result == {}
    assert calls == []


def test_hk_equity_maps_to_hk_prefix_and_parses(monkeypatch) -> None:
    urls: list[str] = []
    _patch_http(monkeypatch, urls, _hk_kline_payload("hk00700"))

    result = tencent_loader.DataLoader().fetch(
        ["00700.HK"], "2026-01-01", "2026-01-31",
    )

    assert len(urls) == 1
    assert "param=hk00700,day," in urls[0]
    df = result["00700.HK"]
    assert len(df) == 2
    # Tencent kline rows are [date, open, close, high, low, volume].
    assert df.iloc[0]["open"] == 466.4
    assert df.iloc[0]["close"] == 471.8
    assert df.iloc[0]["high"] == 475.0
    assert df.iloc[0]["low"] == 462.8


def test_short_hk_code_is_zero_padded(monkeypatch) -> None:
    urls: list[str] = []
    _patch_http(monkeypatch, urls, _hk_kline_payload("hk00700"))

    result = tencent_loader.DataLoader().fetch(
        ["700.HK"], "2026-01-01", "2026-01-31",
    )

    assert "param=hk00700,day," in urls[0]
    assert "700.HK" in result


def _daily_page(dates: list[str]) -> pd.DataFrame:
    """Build a normalized daily frame indexed by trade date."""
    index = pd.to_datetime(dates)
    return pd.DataFrame(
        {
            "open": [1.0] * len(index),
            "high": [1.0] * len(index),
            "low": [1.0] * len(index),
            "close": [1.0] * len(index),
            "volume": [1.0] * len(index),
        },
        index=index,
    )


def _full_page(start: str) -> pd.DataFrame:
    """A page of exactly _PAGE_SIZE bars, i.e. one the API truncated."""
    dates = pd.date_range(start, periods=tencent_loader._PAGE_SIZE, freq="D")
    return _daily_page([d.strftime("%Y-%m-%d") for d in dates])


def test_pagination_walks_past_the_500_bar_cap(monkeypatch) -> None:
    """A multi-year window must not stop at the API's per-response cap."""
    loader = tencent_loader.DataLoader()
    requested: list[str] = []

    def fake_page(code, start, end):  # noqa: ANN001
        requested.append(start)
        if len(requested) == 1:
            return _full_page("2020-01-01")
        return _daily_page(["2021-06-01", "2021-06-02"])

    monkeypatch.setattr(loader, "_request_page", fake_page)
    df = loader._fetch_one("600519.SH", "2020-01-01", "2021-12-31")

    assert len(requested) == 2
    # The second page resumes the day after the first page's last bar.
    assert requested[1] == "2021-05-15"
    assert len(df) == tencent_loader._PAGE_SIZE + 2
    assert df.index.is_monotonic_increasing


def test_overlapping_pages_are_deduplicated(monkeypatch) -> None:
    """Rows repeated across pages must collapse, not double-count."""
    loader = tencent_loader.DataLoader()
    pages = [_full_page("2020-01-01"), _daily_page(["2021-05-14", "2021-05-15"])]

    monkeypatch.setattr(
        loader, "_request_page", lambda code, start, end: pages.pop(0),
    )
    df = loader._fetch_one("600519.SH", "2020-01-01", "2021-12-31")

    assert df.index.duplicated().sum() == 0
    assert len(df) == tencent_loader._PAGE_SIZE + 1


def test_exhausting_the_page_cap_raises_instead_of_truncating(monkeypatch) -> None:
    """Hitting _MAX_PAGES must fail loudly, like the other bounded loaders."""
    loader = tencent_loader.DataLoader()
    starts = ["2000-01-01"]

    def fake_page(code, start, end):  # noqa: ANN001
        page = _full_page(start)
        starts.append(start)
        return page

    monkeypatch.setattr(loader, "_request_page", fake_page)
    with pytest.raises(ValueError, match="incomplete tencent history"):
        loader._fetch_one("600519.SH", "2000-01-01", "2026-12-31")


def test_a_failed_page_raises_instead_of_returning_partial_history(
    monkeypatch,
) -> None:
    """A network failure mid-walk must not read downstream as a short series."""
    loader = tencent_loader.DataLoader()
    calls = {"n": 0}
    sleeps: list[float] = []

    def fake_page(code, start, end):  # noqa: ANN001
        calls["n"] += 1
        if calls["n"] == 1:
            return _full_page("2020-01-01")
        raise OSError("connection reset")

    monkeypatch.setattr(loader, "_request_page", fake_page)
    monkeypatch.setattr(tencent_loader.time, "sleep", sleeps.append)
    with pytest.raises(ValueError, match="incomplete tencent history"):
        loader._fetch_one("600519.SH", "2020-01-01", "2021-12-31")

    # One good page, then the retry budget spent on the failing one.
    assert calls["n"] == 1 + tencent_loader._PAGE_RETRIES
    assert len(sleeps) == tencent_loader._PAGE_RETRIES - 1


def test_a_short_page_ends_the_walk(monkeypatch) -> None:
    """A page below the cap means the window is served; stop requesting."""
    loader = tencent_loader.DataLoader()
    calls: list[str] = []

    def fake_page(code, start, end):  # noqa: ANN001
        calls.append(start)
        return _daily_page(["2026-01-05", "2026-01-06"])

    monkeypatch.setattr(loader, "_request_page", fake_page)
    df = loader._fetch_one("600519.SH", "2026-01-01", "2026-01-31")

    assert calls == ["2026-01-01"]
    assert len(df) == 2


def test_a_bar_on_the_end_date_itself_is_not_dropped(monkeypatch) -> None:
    """When the next start lands exactly on end_date, that day still counts."""
    loader = tencent_loader.DataLoader()
    first = _full_page("2020-01-01")
    boundary = first.index.max() + pd.Timedelta(days=1)
    end_date = boundary.strftime("%Y-%m-%d")
    requested: list[str] = []

    def fake_page(code, start, end):  # noqa: ANN001
        requested.append(start)
        if len(requested) == 1:
            return first
        return _daily_page([end_date])

    monkeypatch.setattr(loader, "_request_page", fake_page)
    df = loader._fetch_one("600519.SH", "2020-01-01", end_date)

    assert requested == ["2020-01-01", end_date]
    assert len(df) == tencent_loader._PAGE_SIZE + 1
    assert df.index.max() == boundary
