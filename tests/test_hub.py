"""Tests for hub operations."""

from vlmrun.client.types import (
    HubInfoResponse,
    HubSchemaResponse,
    HubDomainInfo,
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
    assert isinstance(response, list)
    assert all(isinstance(domain, HubDomainInfo) for domain in response)
    assert "document.invoice" in [domain.domain for domain in response]


def test_hub_get_schema(mock_client):
    """Test schema retrieval for a domain."""
    client = mock_client
    domain = "document.invoice"
    response = client.hub.get_schema(domain)
    assert isinstance(response, HubSchemaResponse)
    assert isinstance(response.json_schema, dict)
    assert isinstance(response.schema_version, str)
    assert isinstance(response.schema_hash, str)
