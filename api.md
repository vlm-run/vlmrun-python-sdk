# Shared Types

```python
from vlm.types import PredictionResponse
```

# Vlm

Types:

```python
from vlm.types import HealthResponse
```

Methods:

- <code title="get /v1/health">client.<a href="./src/vlm/_client.py">health</a>() -> <a href="./src/vlm/types/health_response.py">object</a></code>

# OpenAI

Types:

```python
from vlm.types import OpenAIHealthResponse
```

Methods:

- <code title="get /v1/openai/health">client.openai.<a href="./src/vlm/resources/openai/openai.py">health</a>() -> <a href="./src/vlm/types/openai_health_response.py">object</a></code>

## ChatCompletions

Types:

```python
from vlm.types.openai import Completion
```

Methods:

- <code title="post /v1/openai/chat/completions">client.openai.chat_completions.<a href="./src/vlm/resources/openai/chat_completions.py">create</a>(\*\*<a href="src/vlm/types/openai/chat_completion_create_params.py">params</a>) -> <a href="./src/vlm/types/openai/completion.py">Completion</a></code>

## Models

Types:

```python
from vlm.types.openai import ChatModel, Model
```

Methods:

- <code title="get /v1/openai/models/{model}">client.openai.models.<a href="./src/vlm/resources/openai/models.py">retrieve</a>(model) -> <a href="./src/vlm/types/openai/chat_model.py">ChatModel</a></code>
- <code title="get /v1/openai/models">client.openai.models.<a href="./src/vlm/resources/openai/models.py">list</a>() -> <a href="./src/vlm/types/openai/model.py">Model</a></code>

# Experimental

Types:

```python
from vlm.types import ExperimentalHealthResponse
```

Methods:

- <code title="get /v1/experimental/health">client.experimental.<a href="./src/vlm/resources/experimental/experimental.py">health</a>() -> <a href="./src/vlm/types/experimental_health_response.py">object</a></code>

## Image

### Embeddings

Methods:

- <code title="post /v1/experimental/image/embeddings">client.experimental.image.embeddings.<a href="./src/vlm/resources/experimental/image/embeddings.py">create</a>(\*\*<a href="src/vlm/types/experimental/image/embedding_create_params.py">params</a>) -> <a href="./src/vlm/types/shared/prediction_response.py">PredictionResponse</a></code>

## Document

### Embeddings

Methods:

- <code title="post /v1/experimental/document/embeddings">client.experimental.document.embeddings.<a href="./src/vlm/resources/experimental/document/embeddings.py">create</a>(\*\*<a href="src/vlm/types/experimental/document/embedding_create_params.py">params</a>) -> <a href="./src/vlm/types/shared/prediction_response.py">PredictionResponse</a></code>

# Models

Types:

```python
from vlm.types import ModelInfoResponse, ModelListResponse
```

Methods:

- <code title="get /v1/models">client.models.<a href="./src/vlm/resources/models.py">list</a>() -> <a href="./src/vlm/types/model_list_response.py">ModelListResponse</a></code>

# Files

Types:

```python
from vlm.types import StoreFileResponse, FileListResponse
```

Methods:

- <code title="post /v1/files">client.files.<a href="./src/vlm/resources/files.py">create</a>(\*\*<a href="src/vlm/types/file_create_params.py">params</a>) -> <a href="./src/vlm/types/store_file_response.py">StoreFileResponse</a></code>
- <code title="get /v1/files/{file_id}">client.files.<a href="./src/vlm/resources/files.py">retrieve</a>(file_id) -> <a href="./src/vlm/types/store_file_response.py">StoreFileResponse</a></code>
- <code title="get /v1/files">client.files.<a href="./src/vlm/resources/files.py">list</a>(\*\*<a href="src/vlm/types/file_list_params.py">params</a>) -> <a href="./src/vlm/types/file_list_response.py">FileListResponse</a></code>

# Response

Methods:

- <code title="get /v1/response/{id}">client.response.<a href="./src/vlm/resources/response.py">retrieve</a>(id) -> <a href="./src/vlm/types/shared/prediction_response.py">PredictionResponse</a></code>

# Document

Methods:

- <code title="post /v1/document/generate">client.document.<a href="./src/vlm/resources/document.py">generate</a>(\*\*<a href="src/vlm/types/document_generate_params.py">params</a>) -> <a href="./src/vlm/types/shared/prediction_response.py">PredictionResponse</a></code>

# Audio

Methods:

- <code title="post /v1/audio/generate">client.audio.<a href="./src/vlm/resources/audio.py">generate</a>(\*\*<a href="src/vlm/types/audio_generate_params.py">params</a>) -> <a href="./src/vlm/types/shared/prediction_response.py">PredictionResponse</a></code>

# Image

Methods:

- <code title="post /v1/image/generate">client.image.<a href="./src/vlm/resources/image.py">generate</a>(\*\*<a href="src/vlm/types/image_generate_params.py">params</a>) -> <a href="./src/vlm/types/shared/prediction_response.py">PredictionResponse</a></code>

# Web

Methods:

- <code title="post /v1/web/generate">client.web.<a href="./src/vlm/resources/web.py">generate</a>(\*\*<a href="src/vlm/types/web_generate_params.py">params</a>) -> <a href="./src/vlm/types/shared/prediction_response.py">PredictionResponse</a></code>

# Schema

Methods:

- <code title="post /v1/schema/generate">client.schema.<a href="./src/vlm/resources/schema.py">generate</a>(\*\*<a href="src/vlm/types/schema_generate_params.py">params</a>) -> <a href="./src/vlm/types/shared/prediction_response.py">PredictionResponse</a></code>
