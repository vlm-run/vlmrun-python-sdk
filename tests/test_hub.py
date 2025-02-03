"""Tests for hub operations."""

from vlmrun.client.types import (
    HubInfoResponse,
    HubSchemaQueryResponse,
    HubDomainsResponse,
)


def test_hub_info(mock_client):
    """Test hub info retrieval."""
    client = mock_client
    response = client.hub.info()
    assert isinstance(response, HubInfoResponse)
    assert isinstance(response.version, str)
    assert response.version == "0.1.0"


def test_hub_list_domains(mock_client):
    """Test listing hub domains."""
    client = mock_client
    response = client.hub.list_domains()
    assert isinstance(response, HubDomainsResponse)
    assert isinstance(response.domains, list)
    assert all(isinstance(domain, str) for domain in response.domains)
    assert "document.invoice" in response.domains


def test_hub_get_schema(mock_client):
    """Test schema retrieval for a domain."""
    client = mock_client
    domain = "document.invoice"
    response = client.hub.get_schema(domain)
    assert isinstance(response, HubSchemaQueryResponse)
    assert isinstance(response.schema_json, dict)
    assert isinstance(response.schema_version, str)
    assert isinstance(response.schema_hash, str)
    assert len(response.schema_hash) == 8
