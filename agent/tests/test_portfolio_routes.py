from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api import portfolio_routes


class _Service:
    def __init__(self, **kwargs):
        self.progress_callback = kwargs.get("progress_callback")

    def latest(self):
        return {"snapshot_id": "latest"}

    def refresh(self):
        return {"snapshot_id": "new"}

    def settings(self):
        return {
            "display_currency": "USD",
            "sources": [{"connection_id": "ibkr", "enabled": True}],
        }

    def save_settings(self, payload):
        return payload

    def sources(self):
        return [{"profile_id": "ibkr-live-official-mcp-readonly"}]

    def history(self, limit):
        return [{"id": "one", "limit": limit}]

    def reconnect_source(self, source_id):
        return {"status": "ok", "authorized": True, "source_id": source_id}

    def export_csv(self):
        return "broker,symbol\nibkr,AAPL\n"

    def analysis_context(self):
        return {"privacy": "sanitized"}


def test_portfolio_routes_are_readonly_and_return_expected_shapes(monkeypatch):
    monkeypatch.setattr(portfolio_routes, "PortfolioService", _Service)
    app = FastAPI()
    portfolio_routes.register_portfolio_routes(app)
    client = TestClient(app)

    assert client.get("/api/portfolio").json()["snapshot"]["snapshot_id"] == "latest"
    assert (
        client.post("/api/portfolio/refresh").json()["snapshot"]["snapshot_id"] == "new"
    )
    refresh = client.get("/api/portfolio/refresh-status").json()["refresh"]
    assert refresh["running"] is False
    assert refresh["sources"] == refresh["brokers"]
    assert (
        client.get("/api/portfolio/history?limit=12").json()["history"][0]["limit"]
        == 12
    )
    settings = client.get("/api/portfolio/settings").json()
    assert settings["settings"]["display_currency"] == "USD"
    saved = client.put(
        "/api/portfolio/settings",
        json={
            "display_currency": "CNY",
            "sources": [
                {
                    "connection_id": "ibkr",
                    "label": "Main",
                    "enabled": True,
                    "order": 0,
                    "include_cash": True,
                }
            ],
        },
    )
    assert saved.status_code == 200
    assert saved.json()["settings"]["display_currency"] == "CNY"
    assert (
        client.post("/api/portfolio/sources/ibkr/reconnect").json()["source_id"]
        == "ibkr"
    )
    assert (
        client.get("/api/portfolio/analysis-context").json()["context"]["privacy"]
        == "sanitized"
    )
    exported = client.get("/api/portfolio/export.csv")
    assert exported.status_code == 200
    assert "ibkr,AAPL" in exported.text
