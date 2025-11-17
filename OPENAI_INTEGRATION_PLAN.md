# OpenAI SDK Integration Plan for VLMRun Python SDK

## Overview

This document outlines the plan to integrate OpenAI SDK's `chat.completions` API into the VLMRun Python SDK, allowing users to access the new agent chat completions endpoint through a familiar OpenAI-compatible interface.

## Goals

1. Add optional dependency on the OpenAI SDK
2. Expose `client.agent.completions` that maps to OpenAI's `client.chat.completions` object
3. Support both synchronous and asynchronous operations
4. Automatically configure the OpenAI client with the correct base URL and API key
5. Minimize code duplication by leveraging the OpenAI SDK directly

## Current State Analysis

### Existing Infrastructure

1. **VLMRun Client** (`vlmrun/client/client.py`):
   - Already has an `openai` cached property that creates an OpenAI client
   - Current implementation: `OpenAI(api_key=self.api_key, base_url=f"{self.base_url}/openai")`
   - This is used for general OpenAI compatibility

2. **Agent Resource** (`vlmrun/client/agent.py`):
   - Current methods: `create()`, `execute()`, `get()`, `list()`, `get_by_id()`
   - Handles VLMRun-specific agent operations
   - Does not currently expose OpenAI-style chat completions

3. **Dependencies** (`pyproject.toml`):
   - OpenAI SDK is currently only in `[test]` optional dependencies
   - Not available by default for users

4. **OpenAI SDK Structure**:
   - `OpenAI` client has `chat.completions` resource
   - `AsyncOpenAI` client has async version of `chat.completions`
   - Main method: `create()` with extensive parameters for chat completion
   - Additional methods: `stream()`, `parse()`, `list()`, `retrieve()`, etc.

## Proposed Solution

### Architecture

We will create a new property on the `Agent` class called `completions` that returns an OpenAI `Completions` or `AsyncCompletions` object configured with the agent-specific base URL.

```
VLMRun client
  └── agent (Agent resource)
      ├── create()          # Existing VLMRun agent methods
      ├── execute()
      ├── get()
      ├── list()
      └── completions       # NEW: OpenAI chat.completions proxy
          ├── create()      # OpenAI's create method
          ├── stream()      # OpenAI's stream method
          └── ...           # All other OpenAI methods
```

### Implementation Details

#### 1. Update Dependencies

**File**: `pyproject.toml`

Add OpenAI SDK as an optional dependency group:

```toml
[project.optional-dependencies]
openai = ["openai>=1.0.0"]
all = [
    "numpy>=1.24.0",
    "pypdfium2>=4.30.0",
    "openai>=1.0.0",  # Add to all
]
```

This allows users to install with:
- `pip install vlmrun[openai]` - for OpenAI integration only
- `pip install vlmrun[all]` - for all optional features

#### 2. Create Agent Completions Wrapper

**File**: `vlmrun/client/agent.py`

Add two new cached properties to the `Agent` class:

```python
from functools import cached_property
from vlmrun.client.exceptions import DependencyError

class Agent:
    # ... existing methods ...
    
    @cached_property
    def completions(self):
        """OpenAI-compatible chat completions interface (synchronous).
        
        Returns an OpenAI Completions object configured to use the VLMRun
        agent endpoint. This allows you to use the familiar OpenAI API
        for chat completions.
        
        Example:
            ```python
            from vlmrun import VLMRun
            
            client = VLMRun(api_key="your-key")
            
            response = client.agent.completions.create(
                model="vlm-1",
                messages=[
                    {"role": "user", "content": "Hello!"}
                ]
            )
            ```
        
        Raises:
            DependencyError: If openai package is not installed
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise DependencyError(
                message="OpenAI SDK is not installed",
                suggestion="Install it with `pip install vlmrun[openai]` or `pip install openai`",
                error_type="missing_dependency",
            )
        
        # Create OpenAI client with agent-specific base URL
        base_url = f"{self._client.base_url}/openai"
        openai_client = OpenAI(
            api_key=self._client.api_key,
            base_url=base_url,
            timeout=self._client.timeout,
            max_retries=self._client.max_retries,
        )
        
        return openai_client.chat.completions
    
    @cached_property
    def async_completions(self):
        """OpenAI-compatible chat completions interface (asynchronous).
        
        Returns an OpenAI AsyncCompletions object configured to use the VLMRun
        agent endpoint. This allows you to use the familiar OpenAI async API
        for chat completions.
        
        Example:
            ```python
            from vlmrun import VLMRun
            import asyncio
            
            client = VLMRun(api_key="your-key")
            
            async def main():
                response = await client.agent.async_completions.create(
                    model="vlm-1",
                    messages=[
                        {"role": "user", "content": "Hello!"}
                    ]
                )
                return response
            
            asyncio.run(main())
            ```
        
        Raises:
            DependencyError: If openai package is not installed
        """
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise DependencyError(
                message="OpenAI SDK is not installed",
                suggestion="Install it with `pip install vlmrun[openai]` or `pip install openai`",
                error_type="missing_dependency",
            )
        
        # Create AsyncOpenAI client with agent-specific base URL
        base_url = f"{self._client.base_url}/openai"
        async_openai_client = AsyncOpenAI(
            api_key=self._client.api_key,
            base_url=base_url,
            timeout=self._client.timeout,
            max_retries=self._client.max_retries,
        )
        
        return async_openai_client.chat.completions
```

#### 3. Update Client-Level OpenAI Property (Optional Enhancement)

**File**: `vlmrun/client/client.py`

The existing `openai` property could be enhanced to also support async:

```python
@cached_property
def openai(self):
    """OpenAI client (synchronous)."""
    try:
        from openai import OpenAI as _OpenAI
    except ImportError:
        raise DependencyError(
            message="OpenAI client is not installed",
            suggestion="Install it with `pip install vlmrun[openai]` or `pip install openai`",
            error_type="missing_dependency",
        )

    return _OpenAI(
        api_key=self.api_key, 
        base_url=f"{self.base_url}/openai",
        timeout=self.timeout,
        max_retries=self.max_retries,
    )

@cached_property
def async_openai(self):
    """OpenAI client (asynchronous)."""
    try:
        from openai import AsyncOpenAI as _AsyncOpenAI
    except ImportError:
        raise DependencyError(
            message="OpenAI client is not installed",
            suggestion="Install it with `pip install vlmrun[openai]` or `pip install openai`",
            error_type="missing_dependency",
        )

    return _AsyncOpenAI(
        api_key=self.api_key,
        base_url=f"{self.base_url}/openai",
        timeout=self.timeout,
        max_retries=self.max_retries,
    )
```

### Usage Examples

#### Synchronous Usage

```python
from vlmrun import VLMRun

# Initialize client with agent base URL
client = VLMRun(
    api_key="your-vlmrun-api-key",
    base_url="https://agent.vlm.run/v1"  # Agent-specific base URL
)

# Use OpenAI-style chat completions
response = client.agent.completions.create(
    model="vlm-1",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ],
    temperature=0.7,
    max_tokens=100
)

print(response.choices[0].message.content)
```

#### Asynchronous Usage

```python
from vlmrun import VLMRun
import asyncio

async def main():
    client = VLMRun(
        api_key="your-vlmrun-api-key",
        base_url="https://agent.vlm.run/v1"
    )
    
    response = await client.agent.async_completions.create(
        model="vlm-1",
        messages=[
            {"role": "user", "content": "Hello!"}
        ]
    )
    
    print(response.choices[0].message.content)

asyncio.run(main())
```

#### Streaming Usage

```python
from vlmrun import VLMRun

client = VLMRun(
    api_key="your-vlmrun-api-key",
    base_url="https://agent.vlm.run/v1"
)

stream = client.agent.completions.create(
    model="vlm-1",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## Testing Strategy

### Unit Tests

**File**: `tests/test_agent_completions.py` (new file)

```python
import pytest
from vlmrun import VLMRun
from vlmrun.client.exceptions import DependencyError

def test_agent_completions_requires_openai():
    """Test that accessing completions without openai installed raises error."""
    # This test would need to mock the import to simulate missing openai
    pass

def test_agent_completions_sync(mock_vlmrun_client):
    """Test synchronous completions access."""
    client = VLMRun(api_key="test-key")
    
    # Should return OpenAI Completions object
    completions = client.agent.completions
    assert completions is not None
    assert hasattr(completions, 'create')
    assert hasattr(completions, 'stream')

def test_agent_completions_async(mock_vlmrun_client):
    """Test asynchronous completions access."""
    client = VLMRun(api_key="test-key")
    
    # Should return OpenAI AsyncCompletions object
    async_completions = client.agent.async_completions
    assert async_completions is not None
    assert hasattr(async_completions, 'create')

def test_agent_completions_base_url():
    """Test that completions uses correct base URL."""
    client = VLMRun(
        api_key="test-key",
        base_url="https://agent.vlm.run/v1"
    )
    
    # The OpenAI client should be configured with /openai suffix
    # This would need to inspect the internal OpenAI client configuration
    pass
```

### Integration Tests

**File**: `tests/integration/test_agent_completions_integration.py` (new file)

```python
import pytest
from vlmrun import VLMRun
import os

@pytest.mark.skipif(
    not os.getenv("VLMRUN_API_KEY"),
    reason="Requires VLMRUN_API_KEY environment variable"
)
def test_agent_completions_create():
    """Test actual chat completion request."""
    client = VLMRun(base_url="https://agent.vlm.run/v1")
    
    response = client.agent.completions.create(
        model="vlm-1",
        messages=[
            {"role": "user", "content": "Say 'test successful'"}
        ],
        max_tokens=10
    )
    
    assert response.choices[0].message.content is not None
    assert len(response.choices) > 0

@pytest.mark.skipif(
    not os.getenv("VLMRUN_API_KEY"),
    reason="Requires VLMRUN_API_KEY environment variable"
)
@pytest.mark.asyncio
async def test_agent_async_completions_create():
    """Test actual async chat completion request."""
    client = VLMRun(base_url="https://agent.vlm.run/v1")
    
    response = await client.agent.async_completions.create(
        model="vlm-1",
        messages=[
            {"role": "user", "content": "Say 'async test successful'"}
        ],
        max_tokens=10
    )
    
    assert response.choices[0].message.content is not None
```

## Documentation Updates

### README.md

Add a new section on OpenAI compatibility:

```markdown
### OpenAI-Compatible Chat Completions

The VLMRun SDK provides OpenAI-compatible chat completions through the agent endpoint:

```python
from vlmrun import VLMRun

client = VLMRun(
    api_key="your-key",
    base_url="https://agent.vlm.run/v1"
)

# Use familiar OpenAI API
response = client.agent.completions.create(
    model="vlm-1",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)
```

For async support:

```python
response = await client.agent.async_completions.create(
    model="vlm-1",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

**Installation**: Install with OpenAI support using `pip install vlmrun[openai]`
```

### API Documentation

Create or update documentation for the new completions interface, explaining:
- How it differs from the existing `agent.execute()` method
- When to use OpenAI-style completions vs VLMRun-style agent execution
- All supported OpenAI parameters
- Limitations or differences from standard OpenAI API

## Migration Path

### For Existing Users

Existing VLMRun agent functionality remains unchanged. Users can continue using:
- `client.agent.create()`
- `client.agent.execute()`
- `client.agent.get()`
- `client.agent.list()`

### For New Users

New users wanting OpenAI compatibility can:
1. Install with `pip install vlmrun[openai]`
2. Use `client.agent.completions.create()` for familiar OpenAI-style interface

## Implementation Checklist

- [ ] Update `pyproject.toml` to add `openai` optional dependency
- [ ] Add `completions` property to `Agent` class in `vlmrun/client/agent.py`
- [ ] Add `async_completions` property to `Agent` class
- [ ] Optionally enhance `client.openai` and add `client.async_openai` properties
- [ ] Create unit tests in `tests/test_agent_completions.py`
- [ ] Create integration tests (if backend is ready)
- [ ] Update README.md with usage examples
- [ ] Update API documentation
- [ ] Add type hints and docstrings
- [ ] Run linting: `make lint`
- [ ] Run tests: `make test`
- [ ] Create PR with detailed description

## Considerations and Trade-offs

### Advantages

1. **Zero Implementation Overhead**: We leverage the OpenAI SDK directly, so we don't need to implement or maintain the chat completions API
2. **Automatic Updates**: When OpenAI SDK adds new features, they're automatically available
3. **Familiar Interface**: Users already familiar with OpenAI API can use VLMRun immediately
4. **Type Safety**: OpenAI SDK provides excellent type hints and IDE support
5. **Optional Dependency**: Users who don't need OpenAI compatibility don't pay the dependency cost

### Disadvantages

1. **Additional Dependency**: Adds OpenAI SDK as a dependency (though optional)
2. **Version Compatibility**: Need to ensure OpenAI SDK version compatibility
3. **Two APIs**: Users might be confused about when to use `agent.execute()` vs `agent.completions.create()`

### Design Decisions

1. **Why cached_property?**
   - Lazy initialization: OpenAI client only created when needed
   - Avoids import errors for users without OpenAI SDK installed
   - Reuses same client instance for multiple calls

2. **Why separate sync/async properties?**
   - OpenAI SDK uses separate `OpenAI` and `AsyncOpenAI` classes
   - Clearer API: users explicitly choose sync or async
   - Avoids runtime detection of async context

3. **Why not wrap the OpenAI client?**
   - Wrapping would require maintaining compatibility with OpenAI API
   - Direct exposure means automatic feature parity
   - Less code to maintain and test

## Future Enhancements

1. **Convenience Methods**: Add helper methods that combine VLMRun-specific features with OpenAI API
2. **Unified Interface**: Consider creating a unified interface that works for both agent execution styles
3. **Response Conversion**: Add utilities to convert between OpenAI responses and VLMRun response types
4. **Model Aliases**: Map VLMRun model names to OpenAI-compatible names

## Questions for Backend Team

1. What is the exact base URL for the agent chat completions endpoint?
   - Assumed: `https://agent.vlm.run/v1/openai`
2. Are all OpenAI chat completion parameters supported?
3. Are there any VLMRun-specific extensions to the OpenAI API?
4. What models should be available through this interface?
5. Are there rate limits or quotas different from standard VLMRun API?

## Timeline Estimate

- **Implementation**: 2-3 hours
  - Dependency updates: 15 minutes
  - Agent class updates: 1 hour
  - Unit tests: 1 hour
  - Documentation: 30 minutes
- **Testing**: 1 hour
  - Local testing with mock backend
  - Integration testing (if backend ready)
- **Review and Iteration**: 1-2 hours

**Total**: 4-6 hours
