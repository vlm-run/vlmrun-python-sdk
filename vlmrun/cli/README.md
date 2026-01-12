# VLM Run CLI

Visual AI from your terminal. Chat with VLM Run's Orion visual AI agent to process images, videos, and documents with natural language.

## Installation

The CLI is included as an extra in the vlmrun package:

```bash
# Install vlmrun with CLI support
pip install "vlmrun[cli]"

# Or with uv
uv pip install "vlmrun[cli]"
```

## Quick Start

1. **Get your API key** at [app.vlm.run](https://app.vlm.run)

2. **Set your API key:**
   ```bash
   export VLMRUN_API_KEY="your-api-key-here"
   ```

3. **Start using VLM Run:**
   ```bash
   vlmrun chat "What's in this image?" -i photo.jpg
   ```

## Commands

### `vlmrun chat` - Visual AI Chat

The main command for interacting with VLM Run's Orion visual AI agent.
Artifacts are automatically downloaded to `~/.vlmrun/cache/artifacts/<session_id>/`.

**Prompt Sources (in order of precedence):**
1. Command line argument
2. -p option (text, file path, or 'stdin')
3. Auto-detected piped stdin

```bash
# Basic usage
vlmrun chat "What's in this image?" -i photo.jpg
vlmrun chat -p "What's in this image?" -i photo.jpg

# From a file
vlmrun chat -p long_prompt.txt -i photo.jpg

# Pipe prompt from stdin
echo "Describe this image" | vlmrun chat - -i photo.jpg
echo "Describe this image" | vlmrun chat -p stdin -i photo.jpg

# Compare multiple images
vlmrun chat "Compare these two images" -i img1.jpg -i img2.jpg

# Save to custom directory (by default, artifacts are saved to ~/.vlmrun/cache/artifacts/)
vlmrun chat "Detect faces and visualize" -i photo.jpg -o ./results/

# Skip artifact download (by default, artifacts are downloaded)
vlmrun chat "Quick description" -i photo.jpg -nd
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--prompt` | `-p` | Prompt text, file path, or 'stdin' for piped input |
| `--input` | `-i` | Input file(s): images, videos, or documents (repeatable) |
| `--output` | `-o` | Directory for artifacts (default: ~/.vlmrun/cache/artifacts/) |
| `--model` | `-m` | Model (default: `vlmrun-orion-1:auto`) |
| `--json` | `-j` | Output raw JSON response (default: text) |
| `--no-stream` | `-ns` | Do not stream the response in real-time (default: stream) |
| `--no-download` | `-nd` | Skip artifact download (default: download artifacts) |

**Models:**

| Model | Description |
|-------|-------------|
| `vlmrun-orion-1:fast` | Optimized for speed and quick responses |
| `vlmrun-orion-1:auto` | Automatically selects best approach (default) |
| `vlmrun-orion-1:pro` | Most capable, for complex multi-step workflows |


## Examples

### Image Understanding

```bash
# Describe an image
vlmrun chat "Describe everything you see in detail" -i photo.jpg

# Object detection
vlmrun chat "Detect all objects and list them with their locations" -i scene.jpg

# Face detection with visualization
vlmrun chat "Detect all faces and draw bounding boxes around them" -i crowd.jpg

# OCR / Text extraction
vlmrun chat "Extract all text from this image" -i document.png

# Compare images
vlmrun chat "What are the differences between these two images?" -i before.jpg -i after.jpg
```

### Image Generation

```bash
# Text to image
vlmrun chat "Generate a photorealistic image of a cozy cabin in a snowy forest at night"

# Style-based generation
vlmrun chat "Generate an artistic watercolor painting of a mountain landscape" -o ./art/

# Image editing
vlmrun chat "Remove the background from this image" -i product.jpg
```

### Video Processing

```bash
# Summarize a video
vlmrun chat "Summarize this video in 3 bullet points" -i meeting.mp4

# Extract highlights
# This should generate 3 artifacts (.mp4 clips) in the output directory (default: ~/.vlmrun/cache/artifacts/)
vlmrun chat "Find the top 3 most interesting moments and create clips" -i sports.mp4

# Transcription
vlmrun chat "Transcribe this video with timestamps" -i lecture.mp4 --json

# Search within video
vlmrun chat "Find all moments where someone mentions 'AI'" -i interview.mp4
```

### Document Processing

```bash
# Invoice extraction
vlmrun chat "Extract vendor, items, and total from this invoice" -i invoice.pdf --json

# Medical records
vlmrun chat "Extract patient info, diagnoses, and medications" -i medical_record.pdf --json

# Contract analysis
vlmrun chat "Summarize the key terms and obligations" -i contract.pdf

# Multi-page analysis
vlmrun chat "Analyze this document and classify each page type" -i multi_page.pdf
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `VLMRUN_API_KEY` | Your VLM Run API key (required) |
| `VLMRUN_CACHE_DIR` | Custom cache directory (default: `~/.vlmrun/cache/artifacts`) |

## How It Works

The CLI uses the official [VLM Run Python SDK](https://pypi.org/project/vlmrun/) to communicate with VLM Run's Orion visual AI agent. Orion is a unified visual agent that can:

- **See**: Process images, videos, and documents
- **Reason**: Understand context and relationships
- **Act**: Execute multi-step visual workflows with specialized tools

Unlike basic vision-language models, Orion orchestrates specialized computer vision tools (OCR, detection, segmentation, generation, etc.) to deliver precise, production-ready results.

## Additional Commands

VLM Run CLI also provides additional commands for advanced operations:

### Configuration

```bash
# Set your API key
vlmrun config set --api-key YOUR_API_KEY

# View current configuration
vlmrun config show
```

### Dataset Management

```bash
# Create a dataset from a directory of files
vlmrun datasets create --directory ./images --domain document.invoice --dataset-name "Invoice Dataset" --dataset-type images

# List your datasets
vlmrun datasets list
```

### Model Operations

```bash
# List available models
vlmrun models list

# Fine-tune a model
vlmrun fine-tuning create --model base_model --training-file training_file_id
```

### Predictions

```bash
# Generate predictions from a document
vlmrun generate document path/to/document.pdf --domain document.invoice

# List predictions
vlmrun predictions list

# Get prediction details
vlmrun predictions get PREDICTION_ID
```

### Hub Operations

```bash
# List available domains
vlmrun hub list

# View schema for a domain
vlmrun hub schema document.invoice
```

### Full Command Reference

Here are the main command groups available:

- `vlmrun chat` - Chat with Orion visual AI agent
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

## Support

- **Documentation**: [docs.vlm.run](https://docs.vlm.run)
- **Chat with Orion**: [chat.vlm.run](https://chat.vlm.run)
- **API Reference**: [docs.vlm.run/agents/api-reference](https://docs.vlm.run/agents/api-reference)
- **Cookbooks**: [github.com/vlm-run/vlmrun-cookbook](https://github.com/vlm-run/vlmrun-cookbook)
