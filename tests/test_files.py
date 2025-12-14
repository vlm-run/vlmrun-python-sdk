"""Tests for files operations."""

from vlmrun.client.types import FileResponse, PresignedUrlRequest, PresignedUrlResponse


def test_list_files(mock_client):
    """Test listing files."""
    response = mock_client.files.list()
    assert isinstance(response, list)
    assert all(isinstance(file, FileResponse) for file in response)
    assert len(response) > 0
    assert response[0].id == "file1"
    assert response[0].filename == "test.txt"


def test_upload_file(mock_client, tmp_path):
    """Test uploading a file."""
    # Create a temporary file
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")

    response = mock_client.files.upload(test_file)
    assert isinstance(response, FileResponse)
    assert response.id == "file1"
    assert response.filename == str(test_file)


def test_get_file(mock_client):
    """Test getting file metadata."""
    response = mock_client.files.get("file1")
    assert isinstance(response, FileResponse)
    assert response.id == "file1"
    assert response.filename == "test.txt"
    assert response.bytes == 10


def test_delete_file(mock_client):
    """Test deleting a file."""
    response = mock_client.files.delete("file1")
    assert isinstance(response, FileResponse)
    assert response.id == "file1"


def test_generate_presigned_url(mock_client):
    """Test generating a presigned URL."""
    params = PresignedUrlRequest(filename="test.pdf", purpose="assistants")
    response = mock_client.files.generate_presigned_url(params)
    assert isinstance(response, PresignedUrlResponse)
    assert response.id == "presigned1"
    assert response.filename == "test.pdf"
    assert response.url is not None
    assert "storage.example.com" in response.url


def test_generate_presigned_url_without_purpose(mock_client):
    """Test generating a presigned URL without specifying purpose."""
    params = PresignedUrlRequest(filename="document.pdf")
    response = mock_client.files.generate_presigned_url(params)
    assert isinstance(response, PresignedUrlResponse)
    assert response.filename == "document.pdf"
