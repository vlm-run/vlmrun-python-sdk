def test_client():
    from vlmrun.client import Client

    client = Client(api_key="test-key")
    assert client is not None
