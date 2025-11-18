"""Tests for the Agent resource."""

import pytest
from datetime import datetime
from vlmrun.client.types import (
    AgentCreationResponse,
    AgentCreationConfig,
    AgentExecutionConfig,
    AgentExecutionResponse,
    AgentInfo,
    CreditUsage,
)


class TestAgentCreationResponse:
    """Test the AgentCreationResponse model."""

    def test_agent_creation_response_creation(self):
        """Test creating an AgentCreationResponse instance."""
        response_data = {
            "id": "test-agent-123",
            "name": "test-agent:1.0.0",
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "status": "completed",
        }

        response = AgentCreationResponse(**response_data)

        assert response.id == "test-agent-123"
        assert response.name == "test-agent:1.0.0"
        assert response.status == "completed"


class TestAgentMethods:
    """Test the Agent API methods."""

    def test_agent_get_by_name_and_version(self, mock_client):
        """Test getting an agent by name and version."""
        client = mock_client
        response = client.agent.get(name="test-agent:1.0.0")

        assert isinstance(response, AgentInfo)
        assert response.name == "test-agent:1.0.0"
        assert response.description == "Test agent description"
        assert response.prompt == "Test agent prompt"
        assert response.status == "completed"

    def test_agent_get_by_id(self, mock_client):
        """Test getting an agent by ID."""
        client = mock_client
        response = client.agent.get(id="agent-123")

        assert isinstance(response, AgentInfo)
        assert response.id == "agent-123"
        assert response.name == "agent-agent-123"

    def test_agent_get_validation_error(self, mock_client):
        """Test that get method validates input parameters."""
        client = mock_client

        with pytest.raises(
            ValueError, match="Only one of `id` or `name` or `prompt` can be provided."
        ):
            client.agent.get(id="agent-123", name="test-agent")

        with pytest.raises(
            ValueError, match="Either `id` or `name` or `prompt` must be provided."
        ):
            client.agent.get()

    def test_agent_list(self, mock_client):
        """Test listing all agents."""
        client = mock_client
        response = client.agent.list()

        assert isinstance(response, list)
        assert len(response) == 2
        assert all(isinstance(agent, AgentInfo) for agent in response)
        assert response[0].name.startswith("test-agent-1")
        assert response[1].name.startswith("test-agent-2")

    def test_agent_create(self, mock_client):
        """Test creating an agent."""
        client = mock_client
        config = AgentCreationConfig(
            prompt="Create a test agent",
            json_schema={
                "type": "object",
                "properties": {"result": {"type": "string"}},
            },
        )

        response = client.agent.create(
            config=config, name="new-agent", inputs={"test": "input"}
        )

        assert isinstance(response, AgentCreationResponse)
        assert response.name == "new-agent"
        assert response.status == "pending"

    def test_agent_create_validation_error(self, mock_client):
        """Test that create method validates config has prompt."""
        client = mock_client
        config = AgentCreationConfig()

        with pytest.raises(
            ValueError,
            match="Prompt is not provided as a request parameter, please provide a prompt.",
        ):
            client.agent.create(config=config)

    def test_agent_execute(self, mock_client):
        """Test executing an agent."""
        client = mock_client
        config = AgentExecutionConfig(
            prompt="Execute this agent",
            json_schema={
                "type": "object",
                "properties": {"output": {"type": "string"}},
            },
        )

        response = client.agent.execute(
            name="test-agent:1.0.0",
            inputs={"input": "test data"},
            config=config,
        )

        assert isinstance(response, AgentExecutionResponse)
        assert response.name == "test-agent:1.0.0"
        assert response.status == "completed"
        assert response.response == {"result": "execution result"}
        assert isinstance(response.usage, CreditUsage)
        assert response.usage.credits_used == 50

    def test_agent_execute_without_version(self, mock_client):
        """Test executing an agent without specifying version."""
        client = mock_client

        response = client.agent.execute(name="test-agent")

        assert isinstance(response, AgentExecutionResponse)
        assert response.name == "test-agent"

    def test_agent_execute_batch_mode_required(self, mock_client):
        """Test that execute method requires batch mode."""
        client = mock_client

        with pytest.raises(
            NotImplementedError, match="Batch mode is required for agent execution"
        ):
            client.agent.execute(name="test-agent", batch=False)


class TestAgentConfigModels:
    """Test the Agent configuration models."""

    def test_agent_creation_config(self):
        """Test AgentCreationConfig model."""
        config = AgentCreationConfig(
            prompt="Test prompt",
            json_schema={
                "type": "object",
                "properties": {"result": {"type": "string"}},
            },
        )

        assert config.prompt == "Test prompt"
        assert config.json_schema is not None
        assert config.response_model is None

    def test_agent_execution_config(self):
        """Test AgentExecutionConfig model."""
        config = AgentExecutionConfig(
            prompt="Execute prompt",
            json_schema={
                "type": "object",
                "properties": {"output": {"type": "string"}},
            },
        )

        assert config.prompt == "Execute prompt"
        assert config.json_schema is not None
        assert config.response_model is None

    def test_config_validation_error(self):
        """Test that config validates response_model and json_schema are not both provided."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            result: str

        with pytest.raises(ValueError, match="cannot be used together"):
            AgentCreationConfig(
                prompt="Test", response_model=TestModel, json_schema={"type": "object"}
            )


class TestAgentCompletions:
    """Test the Agent OpenAI completions integration."""

    def test_agent_completions_property_exists(self, mock_client):
        """Test that agent.completions property exists and returns OpenAI Completions object."""
        client = mock_client

        completions = client.agent.completions

        assert hasattr(completions, "create")
        assert hasattr(completions, "stream")
        assert callable(completions.create)

    def test_agent_async_completions_property_exists(self, mock_client):
        """Test that agent.async_completions property exists and returns AsyncCompletions object."""
        client = mock_client

        async_completions = client.agent.async_completions

        assert hasattr(async_completions, "create")
        assert callable(async_completions.create)

    def test_agent_completions_cached_property(self, mock_client):
        """Test that completions property is cached."""
        client = mock_client

        completions1 = client.agent.completions
        completions2 = client.agent.completions

        assert completions1 is completions2

    def test_agent_async_completions_cached_property(self, mock_client):
        """Test that async_completions property is cached."""
        client = mock_client

        async_completions1 = client.agent.async_completions
        async_completions2 = client.agent.async_completions

        assert async_completions1 is async_completions2

    def test_client_openai_property_enhanced(self, mock_client):
        """Test that client.openai property is enhanced with timeout and max_retries."""
        client = mock_client

        openai_client = client.openai

        assert hasattr(openai_client, "chat")
        assert hasattr(openai_client.chat, "completions")

    def test_client_async_openai_property_exists(self, mock_client):
        """Test that client.async_openai property exists."""
        client = mock_client

        async_openai_client = client.async_openai

        assert hasattr(async_openai_client, "chat")
        assert hasattr(async_openai_client.chat, "completions")
