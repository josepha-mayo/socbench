"""Production application configuration tests."""

from __future__ import annotations

import pytest

from socbench.api.app import _cors_origins, _trusted_hosts, create_app


def test_development_defaults_are_local_only(monkeypatch):
    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    monkeypatch.delenv("TRUSTED_HOSTS", raising=False)
    monkeypatch.setenv("APP_ENV", "development")

    assert _cors_origins() == ["http://localhost:3000", "http://127.0.0.1:3000"]
    assert _trusted_hosts() == ["localhost", "127.0.0.1", "testserver"]


def test_production_requires_trusted_hosts(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("TRUSTED_HOSTS", raising=False)

    with pytest.raises(RuntimeError, match="TRUSTED_HOSTS"):
        create_app()


def test_production_disables_docs_and_cross_origin_by_default(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("TRUSTED_HOSTS", "api.example.com")
    monkeypatch.delenv("CORS_ORIGINS", raising=False)

    app = create_app()

    assert _cors_origins() == []
    assert app.docs_url is None
    assert app.openapi_url is None


def test_production_rejects_wildcard_cors(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("TRUSTED_HOSTS", "api.example.com")
    monkeypatch.setenv("CORS_ORIGINS", "*")

    with pytest.raises(RuntimeError, match="explicit origins"):
        create_app()
