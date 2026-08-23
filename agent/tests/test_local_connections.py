from __future__ import annotations

import json

import pytest

from src.trading.connections import ConnectionStore
from src.trading.credentials import CredentialStore
from src.trading.local_plugins import discover_plugins, parse_manifest
from src.trading.plugin_scaffold import scaffold_connector, validate_connector


class _MemoryCredentials:
    def __init__(self):
        self.values: dict[tuple[str, str], str] = {}

    def get_password(self, service_name: str, username: str):
        return self.values.get((service_name, username))

    def set_password(self, service_name: str, username: str, password: str):
        self.values[(service_name, username)] = password

    def delete_password(self, service_name: str, username: str):
        self.values.pop((service_name, username), None)


def test_connection_registry_never_serializes_credentials(tmp_path):
    credentials = CredentialStore(_MemoryCredentials())
    store = ConnectionStore(tmp_path / "connections.json", credential_store=credentials)
    connection = store.create(
        "main-binance",
        "binance-live-sdk-readonly",
        "Main Binance",
    )
    credentials.save(connection.id, {"api_key": "secret-value"})

    payload = (tmp_path / "connections.json").read_text(encoding="utf-8")
    assert "secret-value" not in payload
    assert connection.credential_ref == "connector-config://binance"
    assert (tmp_path / "connections.json").stat().st_mode & 0o777 == 0o600


def test_connection_registry_normalizes_ids_before_duplicate_checks(tmp_path):
    store = ConnectionStore(
        tmp_path / "connections.json",
        credential_store=CredentialStore(_MemoryCredentials()),
    )
    store.create("main-account", "binance-live-sdk-readonly", "Main account")

    with pytest.raises(ValueError, match="already exists"):
        store.create(" MAIN-ACCOUNT ", "binance-live-sdk-readonly", "Replacement")

    assert store.get("main-account").label == "Main account"


def test_connection_registry_rejects_control_characters_in_labels(tmp_path):
    store = ConnectionStore(
        tmp_path / "connections.json",
        credential_store=CredentialStore(_MemoryCredentials()),
    )

    with pytest.raises(ValueError, match="printable"):
        store.create("main-account", "binance-live-sdk-readonly", "Main\naccount")


def test_scaffold_generates_a_valid_readonly_connector(tmp_path):
    target = scaffold_connector("sample-broker", tmp_path)
    plugin = validate_connector(target)

    assert plugin.profile.id == "sample-broker-live-readonly"
    assert plugin.profile.readonly is True
    assert {"account.read", "positions.read"}.issubset(plugin.profile.capabilities)
    assert [field.name for field in plugin.credential_fields] == [
        "api_key",
        "api_secret",
    ]


def test_manifest_rejects_write_capabilities(tmp_path):
    target = scaffold_connector("unsafe-broker", tmp_path)
    manifest = target / "connector.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["profile"]["capabilities"].append("orders.place")
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="read capabilities only"):
        parse_manifest(manifest)


def test_manifest_rejects_unknown_non_read_capabilities(tmp_path):
    target = scaffold_connector("unsafe-operation", tmp_path)
    manifest = target / "connector.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["profile"]["capabilities"].append("trade.execute")
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="trade.execute"):
        parse_manifest(manifest)


def test_discovery_isolates_invalid_local_plugins(tmp_path):
    scaffold_connector("good-broker", tmp_path)
    bad = tmp_path / "bad-broker"
    bad.mkdir()
    (bad / "connector.json").write_text("{}", encoding="utf-8")

    plugins, errors = discover_plugins(tmp_path)

    assert [plugin.profile.connector for plugin in plugins] == ["good-broker"]
    assert errors[0]["directory"] == "bad-broker"


def test_installed_local_plugin_runs_through_the_trading_read_interface(
    tmp_path,
    monkeypatch,
):
    from src.trading import connections, local_plugins
    from src.trading.service import get_account, get_positions

    monkeypatch.setattr(local_plugins, "get_runtime_root", lambda: tmp_path)
    monkeypatch.setattr(connections, "get_runtime_root", lambda: tmp_path)
    target = scaffold_connector("sample", tmp_path / "connectors")
    manifest = target / "connector.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["auth"]["fields"] = []
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    (target / "adapter.py").write_text(
        """
def check_status(*, credentials, config):
    return {"status": "ok", "readonly": True}

def get_account_snapshot(*, credentials, config):
    return {"account": {"portfolio_value": "42", "currency": "USD"}}

def get_positions(*, credentials, config):
    return {"positions": [{"symbol": "DEMO", "quantity": 1, "current_price": 42}]}
""",
        encoding="utf-8",
    )
    registry = ConnectionStore()
    registry.create("my-sample", "sample-live-readonly", "My Sample")

    assert (
        get_account("sample-live-readonly", connection_id="my-sample")["account"][
            "portfolio_value"
        ]
        == "42"
    )
    assert (
        get_positions("sample-live-readonly", connection_id="my-sample")["positions"][
            0
        ]["symbol"]
        == "DEMO"
    )
