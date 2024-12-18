def test_client():
    from vlmrun.client import Client

    client = Client()
    assert client is not None
