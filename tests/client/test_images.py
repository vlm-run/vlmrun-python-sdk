"""Tests for the Images class."""

import pytest
import requests
from unittest.mock import patch
from pathlib import Path

from vlmrun.client import Client


def test_image_generate_with_path():
    """Test image.generate with a file path."""
    client = Client(api_key="test-key")
    
    with patch("vlmrun.client.images.requests.post") as mock_post:
        # Setup mock response
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "id": "123",
            "status": "success",
            "result": {"text": "sample prediction"}
        }
        
        # Test with minimal parameters
        response = client.image.generate(
            image="tests/fixtures/sample.jpg",
            model="vlm-1"
        )
        
        # Verify response
        assert response["id"] == "123"
        assert response["status"] == "success"
        assert response["result"]["text"] == "sample prediction"
        
        # Verify request
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == "https://api.vlm.run/v1/image/generate"
        assert kwargs["headers"]["Authorization"] == "Bearer test-key"
        assert kwargs["headers"]["Content-Type"] == "application/json"
        assert "image" in kwargs["json"]
        assert kwargs["json"]["model"] == "vlm-1"


def test_image_generate_with_all_params():
    """Test image.generate with all optional parameters."""
    client = Client(api_key="test-key")
    
    with patch("vlmrun.client.images.requests.post") as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"id": "456", "status": "success"}
        
        # Test with all parameters
        response = client.image.generate(
            image="tests/fixtures/sample.jpg",
            model="vlm-1",
            domain="document.invoice",
            detail="hi",
            json_schema={"type": "object"},
            callback_url="https://example.com/callback",
            metadata={"session_id": "test-session"}
        )
        
        # Verify response
        assert response["id"] == "456"
        assert response["status"] == "success"
        
        # Verify all parameters were sent
        args, kwargs = mock_post.call_args
        payload = kwargs["json"]
        assert payload["model"] == "vlm-1"
        assert payload["domain"] == "document.invoice"
        assert payload["detail"] == "hi"
        assert payload["json_schema"] == {"type": "object"}
        assert payload["callback_url"] == "https://example.com/callback"
        assert payload["metadata"] == {"session_id": "test-session"}


def test_image_generate_error_handling():
    """Test error handling in image.generate."""
    client = Client(api_key="test-key")
    
    with patch("vlmrun.client.images.requests.post") as mock_post, \
         pytest.raises(FileNotFoundError):
        # Test with non-existent file
        client.image.generate(
            image="non_existent.jpg",
            model="vlm-1"
        )
    
    with patch("vlmrun.client.images.requests.post") as mock_post:
        # Test API error
        mock_post.return_value.status_code = 400
        mock_post.return_value.raise_for_status.side_effect = \
            requests.exceptions.HTTPError("Bad Request")
        
        with pytest.raises(requests.exceptions.HTTPError):
            client.image.generate(
                image="tests/fixtures/sample.jpg",
                model="vlm-1"
            )
