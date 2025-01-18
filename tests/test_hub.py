"""Tests for the Hub API client."""

import pytest
from vlmrun.client import Client
from vlmrun.client.base_requestor import APIError


def test_hub_health(monkeypatch):
    """Test hub health check endpoint."""
    def mock_request(*args, **kwargs):
        assert kwargs["method"] == "GET"
        assert kwargs["url"] == "/hub/health"
        return {"status": "ok", "hub_version": "1.0.0"}, 200, {}

    client = Client(api_key="test-key")
    monkeypatch.setattr(client.requestor, "request", mock_request)
    
    response = client.hub.get_health()
    assert response.status == "ok"
    assert response.hub_version == "1.0.0"
    

def test_hub_health_error(monkeypatch):
    """Test hub health check error handling."""
    def mock_request(*args, **kwargs):
        raise Exception("Connection error")

    client = Client(api_key="test-key")
    monkeypatch.setattr(client.requestor, "request", mock_request)
    
    with pytest.raises(APIError) as exc_info:
        client.hub.get_health()
    assert "Failed to check hub health" in str(exc_info.value)


def test_hub_list_domains(monkeypatch):
    """Test listing hub domains."""
    expected_domains = [
        "document.invoice",
        "document.receipt",
        "document.utility_bill"
    ]
    
    def mock_request(*args, **kwargs):
        assert kwargs["method"] == "GET"
        assert kwargs["url"] == "/hub/domains"
        return expected_domains, 200, {}

    client = Client(api_key="test-key")
    monkeypatch.setattr(client.requestor, "request", mock_request)
    
    response = client.hub.list_domains()
    assert response.domains == expected_domains


def test_hub_list_domains_error(monkeypatch):
    """Test listing hub domains error handling."""
    def mock_request(*args, **kwargs):
        raise Exception("Connection error")

    client = Client(api_key="test-key")
    monkeypatch.setattr(client.requestor, "request", mock_request)
    
    with pytest.raises(APIError) as exc_info:
        client.hub.list_domains()
    assert "Failed to list domains" in str(exc_info.value)


def test_hub_get_schema(monkeypatch):
    """Test getting schema for a domain."""
    expected_schema = {
        "schema_json": {
            "type": "object",
            "properties": {
                "invoice_number": {"type": "string"},
                "total_amount": {"type": "number"}
            }
        },
        "schema_version": "1.0.0",
        "schema_hash": "abcd1234"
    }
    
    def mock_request(*args, **kwargs):
        assert kwargs["method"] == "POST"
        assert kwargs["url"] == "/hub/schema"
        assert kwargs["data"] == {"domain": "document.invoice"}
        return expected_schema, 200, {}

    client = Client(api_key="test-key")
    monkeypatch.setattr(client.requestor, "request", mock_request)
    
    response = client.hub.get_schema("document.invoice")
    assert response.schema_json == expected_schema["schema_json"]
    assert response.schema_version == expected_schema["schema_version"]
    assert response.schema_hash == expected_schema["schema_hash"]


def test_hub_get_schema_error(monkeypatch):
    """Test getting schema error handling."""
    def mock_request(*args, **kwargs):
        raise Exception("Domain not found")

    client = Client(api_key="test-key")
    monkeypatch.setattr(client.requestor, "request", mock_request)
    
    with pytest.raises(APIError) as exc_info:
        client.hub.get_schema("invalid.domain")
    assert "Failed to get schema for domain invalid.domain" in str(exc_info.value)
