## 🖥️ Command Line Interface

VLM Run provides a powerful CLI for interacting with the platform. The CLI supports major operations including dataset management, model fine-tuning, predictions, and more.

### Installation

The CLI is automatically installed with the Python package:

```bash
pip install vlmrun
```


### Configuration

Configure your API key and other settings using the `vlmrun config` command:

```bash
## Set your API key
vlmrun config set --api-key YOUR_API_KEY
```

View current configuration:

```bash
vlmrun config show
```


### Key Features

#### Dataset Management

Create a dataset from a directory of files
```bash
vlmrun datasets create --directory ./images --domain document.invoice --dataset-name "Invoice Dataset" --dataset-type images
```

List your datasets
```bash
vlmrun datasets list
```

#### Model Operations

List available models
```bash
vlmrun models list
```

Fine-tune a model
```bash
vlmrun fine-tuning create --model base_model --training-file training_file_id
```

#### Predictions

Generate predictions from a document
```bash
vlmrun generate document path/to/document.pdf --domain document.invoice
```

List predictions
```bash
vlmrun predictions list
```

Get prediction details
```bash
vlmrun predictions get PREDICTION_ID
```

#### Hub Operations

List available domains
```bash
vlmrun hub list
```

View schema for a domain
```bash
vlmrun hub schema document.invoice
```


### Full Command Reference

Here are the main command groups available:

- `vlmrun config` - Manage configuration settings
- `vlmrun datasets` - Dataset operations
- `vlmrun files` - File management
- `vlmrun fine-tuning` - Model fine-tuning operations
- `vlmrun generate` - Generate predictions
- `vlmrun hub` - Access domain schemas and information
- `vlmrun models` - Model operations
- `vlmrun predictions` - Manage and monitor predictions

For detailed help on any command, use the `--help` flag:

```bash
vlmrun --help
vlmrun <command> --help
```
