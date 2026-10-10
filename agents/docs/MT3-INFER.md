# MT3-Infer

Production-ready, unified inference toolkit for the MT3 music transcription model family.

MT3-Infer provides a clean, framework-neutral API for running music transcription inference across multiple MT3 implementations with a single consistent interface.

---

## What's New

- **v0.1.3 (Latest):** Google Colab support, removed `note_seq` dependency, `transformers 4.44+` compatibility
- **v0.1.1:** Fixed YAML config files inclusion in package distribution
- **v0.1.0:** Initial release with 3 production-ready models:
  - MR-MT3
  - MT3-PyTorch
  - YourMT3

## Features

- **Unified API:** One interface for all MT3 variants
- **Production Ready:** Clean, tested, ~8 MB package size
- **Auto-Download:** Automatic checkpoint downloads on first use
- **4 Download Methods:** Auto, Python API, CLI, standalone script
- **3 Models:** MR-MT3, MT3-PyTorch, YourMT3
- **Framework Isolated:** Clean PyTorch / TensorFlow / JAX separation
- **CLI Tool:** `mt3-infer` command-line interface
- **Reproducible:** Pinned dependencies, verified checkpoints
- **Google Colab:** Ready-to-use notebook with audio preview

---

## Quick Start

### Installation

MT3-Infer is available on PyPI.

```bash
# Using pip
pip install mt3-infer

# Using UV
uv pip install mt3-infer
```

### Simple Transcription

```python
from mt3_infer import transcribe

# Transcribe audio to MIDI
# Auto-downloads checkpoint on first use
midi = transcribe(audio, sr=16000)
midi.save("output.mid")
```

### Model Selection

```python
# Use MR-MT3 model
midi = transcribe(audio, model="mr_mt3")

# Use MT3-PyTorch model
midi = transcribe(audio, model="mt3_pytorch")

# Use YourMT3 model
midi = transcribe(audio, model="yourmt3")
```

### Download Checkpoints

```bash
# Download all models
mt3-infer download --all

# Download specific models
mt3-infer download mr_mt3 mt3_pytorch

# List available models
mt3-infer list

# Transcribe audio via CLI
mt3-infer transcribe input.wav -o output.mid -m mr_mt3
```

MR-MT3 weights are downloaded directly from `gudgud1014/MR-MT3` on Hugging Face.

Checkpoints are stored under:

```text
.mt3_checkpoints/<model>
```

To use another checkpoint directory:

```bash
export MT3_CHECKPOINT_DIR=/data/models/mt3
```

Or in a `.env` file:

```text
MT3_CHECKPOINT_DIR=/data/models/mt3
```

---

## Supported Models

| Model | Framework | Speed | Notes Detected | Size | Features |
|---|---|---:|---:|---:|---|
| MR-MT3 | PyTorch | 57x real-time | 116 | 176 MB | Optimized for speed |
| MT3-PyTorch | PyTorch | 12x real-time | 147 | 176 MB | Official architecture with auto-filtering |
| YourMT3 | PyTorch | ~15x real-time | 118 | 536 MB | 8-stem separation, Perceiver-TF + MoE |

Performance benchmarks were measured on an NVIDIA RTX 4090 with PyTorch 2.7.1 and CUDA 12.6.

The default `yourmt3` model downloads the `YPTF.MoE+Multi (noPS)` checkpoint.

---

## Advanced Usage

### Explicit Model Loading

```python
from mt3_infer import load_model

model = load_model("mt3_pytorch", device="cuda")
midi = model.transcribe(audio, sr=16000)
```

### Explore Available Models

```python
from mt3_infer import list_models, get_model_info

models = list_models()

for name, info in models.items():
    print(f"{name}: {info['description']}")
```

Get information about a specific model:

```python
info = get_model_info("mr_mt3")

print(
    f"Speed: "
    f"{info['metadata']['performance']['speed_x_realtime']}x real-time"
)
```

### Disable Auto-Download

```python
from mt3_infer import load_model

model = load_model("mr_mt3", auto_download=False)
```

### Control MT3-PyTorch Instrument Filtering

Automatic filtering is enabled by default:

```python
model = load_model("mt3_pytorch")
```

Disable filtering:

```python
model = load_model(
    "mt3_pytorch",
    auto_filter=False
)
```

### Override Checkpoint Directory

```bash
export MT3_CHECKPOINT_DIR=/mnt/shared/mt3
```

Then:

```bash
uv run python -c \
"from mt3_infer import download_model; download_model('yourmt3')"

uv run mt3-infer download --all
```

Programmatically check the resolved location:

```python
from mt3_infer import download_model

path = download_model("mt3_pytorch")
print(path)
```

### Download Programmatically

```python
from mt3_infer import download_model

download_model("mr_mt3")
download_model("mt3_pytorch")
download_model("yourmt3")
```

---

## Diagnostics & Troubleshooting

Additional diagnostics are available under:

```text
examples/diagnostics/
```

Included scripts:

- `download_mt3_pytorch.py` — manual vs. automatic checkpoint download walkthrough
- `test_all_models.py` — loads all registered models and runs a short transcription
- `test_checkpoint_download.py` — verifies checkpoints land in `MT3_CHECKPOINT_DIR`
- `test_yourmt3.py` — full audio-to-MIDI flow for the YourMT3 MoE model

Run one with:

```bash
uv run python examples/diagnostics/<script>.py
```

---

## Installation Options

### Basic Installation

```bash
pip install mt3-infer
```

### Development Installation

```bash
git clone https://github.com/openmirlab/mt3-infer.git
cd mt3-infer

uv sync --extra torch --extra dev
```

Or:

```bash
pip install -e ".[torch,dev]"
```

### Optional Dependencies

```bash
# PyTorch backend
pip install mt3-infer[torch]

# TensorFlow backend
pip install mt3-infer[tensorflow]

# All backends
pip install mt3-infer[all]

# Development tools
pip install mt3-infer[dev]

# MIDI synthesis
pip install mt3-infer[synthesis]
```

---

## CLI Tool

The `mt3-infer` CLI provides access to the package functionality.

### Download Checkpoints

```bash
mt3-infer download --all
mt3-infer download mr_mt3 mt3_pytorch
```

### List Models

```bash
mt3-infer list
```

### Transcribe Audio

```bash
mt3-infer transcribe input.wav -o output.mid

mt3-infer transcribe input.wav \
    -m mr_mt3

mt3-infer transcribe input.wav \
    --device cuda
```

### Help

```bash
mt3-infer --help
mt3-infer download --help
```

---

## Download Methods

MT3-Infer supports four checkpoint download methods.

### 1. Automatic Download

```python
midi = transcribe(audio)
```

### 2. Python API

```python
from mt3_infer import download_model

download_model("mr_mt3")
```

### 3. CLI

```bash
mt3-infer download --all
```

### 4. Standalone Script

```bash
python tools/download_all_checkpoints.py
```

---

## Project Status

### Completed Features

- Core infrastructure (`MT3Base` interface, utilities)
- 3 production adapters:
  - MR-MT3
  - MT3-PyTorch
  - YourMT3
- Public API:
  - `transcribe()`
  - `load_model()`
- Model registry with aliases
- Checkpoint download system
- CLI tool
- Production cleanup
- Comprehensive documentation

### Package Statistics

| Component | Size |
|---|---:|
| Source code | ~5 MB |
| Vendor dependencies | ~3 MB |
| Documentation | 284 KB |
| Total source package | ~8 MB |
| With downloaded models | ~882 MB |

### Roadmap

- **v0.2.0:** Batch processing and additional optimizations
- **v0.3.0:** ONNX export and streaming inference
- **v1.0.0:** Full test coverage and additional features

Magenta MT3 using JAX/Flax is excluded because of dependency conflicts with the PyTorch ecosystem.

---

## Architecture

```text
mt3_infer/
├── __init__.py
├── api.py
├── base.py
├── cli.py
├── exceptions.py
├── adapters/
│   ├── mr_mt3.py
│   ├── mt3_pytorch.py
│   ├── yourmt3.py
│   └── vocab_utils.py
├── config/
│   └── checkpoints.yaml
├── utils/
│   ├── audio.py
│   ├── midi.py
│   ├── download.py
│   └── framework.py
└── models/
    ├── mr_mt3/
    ├── mt3_pytorch/
    └── yourmt3/
```

---

## Development

### Setup

```bash
uv sync --extra torch --extra dev
```

Run tests:

```bash
uv run pytest
```

Run tests with coverage:

```bash
uv run pytest \
    --cov=mt3_infer \
    --cov-report=html
```

Lint:

```bash
uv run ruff check .
uv run ruff check --fix .
```

Type checking:

```bash
uv run mypy mt3_infer/
```

### Using UV

Use:

```bash
uv run python script.py
uv run pytest
```

rather than:

```bash
python script.py
pytest
```

---

## Integration with `worzpro-demo`

Add the package to `pyproject.toml`:

```toml
[tool.uv.sources]
mt3-infer = {
    git = "https://github.com/openmirlab/mt3-infer",
    extras = ["torch"]
}
```

Then:

```python
from mt3_infer import transcribe

midi = transcribe(audio, sr=16000)
```

---

## Examples

The repository contains examples including:

- `public_api_demo.py` — main usage example
- `synthesize_all_models.py` — compare all models
- `demo_midi_synthesis.py` — MIDI synthesis demo
- `test_download.py` — download validation
- `compare_models.py` — model comparison

---

## License

MIT License.

The project includes code adapted from:

- Magenta MT3 — Apache-2.0
- MR-MT3 — MIT
- MT3-PyTorch
- YourMT3 — Apache-2.0

---

## Citation

If using MT3-Infer in research, cite the original MT3 work:

```bibtex
@inproceedings{hawthorne2022mt3,
  title={Multi-Task Multitrack Music Transcription},
  author={Hawthorne, Curtis and others},
  booktitle={ISMIR},
  year={2022}
}
```

---

## Package Information

- **License:** MIT
- **Python:** >= 3.9
- **Tags:** `audio`, `midi`, `mir`, `mt3`, `music`, `transcription`
- **Intended audience:** Developers, Science/Research
- **Topic:** Sound/Audio Analysis, Artificial Intelligence
