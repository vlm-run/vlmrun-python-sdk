---
name: vlmrun-cli
description: VLM Run CLI for visual AI tasks using Orion agent. Use when users want to process images, generate images, analyze videos, extract data from documents, detect faces/objects, perform OCR, or any visual AI task from the command line. Supports piping prompts, template files, and skill files.
---

# VLM Run CLI Skill

Chat with VLM Run's Orion visual AI agent from the command line.

## Prerequisites

1. Install: `pip install "vlmrun[cli]"` (or `uv pip install "vlmrun[cli]"`)
2. Get API key at [app.vlm.run](https://app.vlm.run)
3. Set key: `export VLMRUN_API_KEY="your-key-here"`

## Command Reference

### Chat (Primary Command)

```bash
vlmrun chat [PROMPT] [OPTIONS]
```

**Prompt Sources (in order of precedence):**
1. Piped input from stdin
2. Command line argument

**Options:**
- `-i, --input FILE` - Input file (image/video/document). Repeatable.
- `-o, --output DIR` - Output directory (default: `~/.vlm/cache/artifacts/<session_id>/`)
- `-m, --model MODEL` - Model (default: `vlmrun-orion-1:auto`)
- `--stream` - Stream response
- `-j, --json` - JSON output
- `-nd, --no-download` - Skip artifact download

### Models

| Model | Use Case |
|-------|----------|
| `vlmrun-orion-1:fast` | Quick responses, simple tasks |
| `vlmrun-orion-1:auto` | Balanced (default) |
| `vlmrun-orion-1:pro` | Complex multi-step workflows |

## Examples

### Basic Usage

```bash
vlmrun chat "What's in this image?" -i photo.jpg
vlmrun chat "Compare these" -i img1.jpg -i img2.jpg
vlmrun chat "Extract data" -i invoice.pdf --json
```

### Piping Prompts

```bash
# Pipe from echo
echo "Describe this image in detail" | vlmrun chat -i photo.jpg

# Pipe from file
cat analysis_prompt.txt | vlmrun chat -i document.pdf

# Pipe from command
curl -s https://example.com/prompt.txt | vlmrun chat -i data.pdf
```

### Templates with Variables

```bash
# Template file: extract.prompt
# Extract {{fields}} from this document.
# Output as {{format}}.

vlmrun chat -t extract.prompt -V "fields=name,date,total" -V "format=json" -i invoice.pdf
```

Template variable formats: `{{var}}`, `${var}`, `$(var)`

### Skills for Specialized Behavior

```bash
# Single skill
vlmrun chat "Process this" -i record.pdf -s medical.md

# Multiple skills
vlmrun chat "Analyze" -i doc.pdf -s analysis.md -s formatting.md -s compliance.md
```

### Combined Usage

```bash
# Template + variables + skills + input
vlmrun chat -t task.prompt -V "output=table" -s domain.md -i data.pdf

# Piped prompt + skills
cat prompt.txt | vlmrun chat -i image.jpg -s style.md
```

## Task Mapping

| User Request | Command |
|-------------|---------|
| Describe image | `vlmrun chat "describe" -i img.jpg` |
| Pipe long prompt | `cat prompt.txt \| vlmrun chat -i img.jpg` |
| Use template | `vlmrun chat -t template.prompt -i doc.pdf` |
| With variables | `vlmrun chat -t t.prompt -V "key=value"` |
| Add skills | `vlmrun chat "analyze" -s skill.md -i doc.pdf` |
| Multiple skills | `vlmrun chat "task" -s a.md -s b.md -i file.pdf` |

## Creating Skills

Skills are markdown files that customize agent behavior:

```markdown
---
name: invoice-extraction
---
# Invoice Extraction Skill

When processing invoices:
- Extract vendor name and address
- List all line items with quantities and prices
- Calculate subtotal, tax, and total
- Identify payment terms and due date
```

## Output Handling

- **Default**: Artifacts saved to `~/.vlm/cache/artifacts/<session_id>/`
- **Custom**: Use `-o` to specify output directory
- **Skip**: Use `-nd` to skip artifact download
- **JSON**: Use `-j` for machine-readable output

## Skills Management

Manage reusable skills (SKILL.md bundles) via the `vlmrun skills` subcommand.

### List Skills

```bash
vlmrun skills list
vlmrun skills list --grouped          # latest version per name
vlmrun skills list --limit 50 --asc   # pagination & sort
```

### Get a Skill

```bash
vlmrun skills get my-skill                  # by name (latest version)
vlmrun skills get my-skill -V 20260312-abc  # by name + version
vlmrun skills get <uuid>                    # by ID
```

### Upload a Skill

Upload a local folder containing a `SKILL.md` (and optional supporting files).
Name and description are parsed from the SKILL.md YAML frontmatter automatically.

```bash
vlmrun skills upload ./my-skill
```

### Download a Skill

```bash
vlmrun skills download my-skill                 # → ~/.vlmrun/skills/my-skill/
vlmrun skills download my-skill -V 0.2.0        # specific version
vlmrun skills download <uuid> -o ./local-dir    # custom output
```

### Create a Skill (from prompt or session)

```bash
vlmrun skills create --prompt "Extract invoice fields"
vlmrun skills create --prompt-file task.txt --schema schema.json
vlmrun skills create --session-id <session-uuid>
```

## Utility Commands

```bash
vlmrun config --show   # View configuration
vlmrun cache --show    # View cache
vlmrun cache --clear   # Clear cache
```
