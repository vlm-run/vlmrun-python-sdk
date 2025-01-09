def test_client():
    """Test client initialization."""
    from vlmrun.client import Client

    client = Client(api_key="test-key")
    assert client.api_key == "test-key"
    assert hasattr(client, "image")
    assert client is not None
