"""Tests for the Agent resource."""

from datetime import datetime
from vlmrun.client.types import AgentCreationResponse


class TestAgentCreationResponse:
    """Test the AgentCreationResponse model."""

    def test_agent_creation_response_creation(self):
        """Test creating an AgentCreationResponse instance."""
        response_data = {
            "id": "test-agent-123",
            "name": "Test Agent",
            "version": "1.0.0",
            "description": "A test agent for testing purposes",
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "output_json_schema": {
                "type": "object",
                "properties": {"result": {"type": "string"}},
            },
            "output_json_sample": {"result": "test result"},
            "input_type": "document",
            "input_json_schema": None,
        }

        response = AgentCreationResponse(**response_data)

        assert response.id == "test-agent-123"
        assert response.name == "Test Agent"
        assert response.version == "1.0.0"
        assert response.description == "A test agent for testing purposes"
        assert response.input_type == "document"
        assert response.input_json_schema is None
        assert "result" in response.output_json_schema["properties"]
        assert response.output_json_sample["result"] == "test result"

    def test_agent_creation_response_with_input_schema(self):
        """Test creating an AgentCreationResponse with input schema."""
        response_data = {
            "id": "test-agent-456",
            "name": "Test Agent with Input Schema",
            "version": "2.0.0",
            "description": "A test agent with input schema",
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "output_json_schema": {
                "type": "object",
                "properties": {"output": {"type": "string"}},
            },
            "output_json_sample": {"output": "test output"},
            "input_type": "mixed",
            "input_json_schema": {
                "type": "object",
                "properties": {"input": {"type": "string"}},
            },
        }

        response = AgentCreationResponse(**response_data)

        assert response.id == "test-agent-456"
        assert response.input_type == "mixed"
        assert response.input_json_schema is not None
        assert "input" in response.input_json_schema["properties"]

    def test_agent_creation_response_input_types(self):
        """Test that all input types are accepted."""
        valid_input_types = ["text", "document", "image", "video", "audio", "mixed"]

        for input_type in valid_input_types:
            response_data = {
                "id": f"test-agent-{input_type}",
                "name": f"Test Agent {input_type}",
                "version": "1.0.0",
                "description": f"A test agent for {input_type}",
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
                "output_json_schema": {"type": "object", "properties": {}},
                "output_json_sample": {},
                "input_type": input_type,
                "input_json_schema": None,
            }

            response = AgentCreationResponse(**response_data)
            assert response.input_type == input_type
