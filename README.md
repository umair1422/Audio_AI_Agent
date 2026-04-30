# Expert Service Voice Agent (Local)

Local voice agent (Mac friendly) built on:

- **STT**: `whisper.cpp` (default: `base` model for low latency)
- **Retrieval**: your existing **BM25 + typo tolerance**
- **LLM**: **Ollama** (`phi3:mini`) used only for **final routing / re-ranking**
- **TTS**: **Piper**
- **Call transport**: **LiveKit** (use `agents-playground.livekit.io` as the UI)

## 1) Python setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) System dependencies

### ffmpeg (required)

```bash
brew install ffmpeg
```

### whisper.cpp (required)

Install whisper.cpp via Homebrew:

```bash
brew install whisper-cpp
```

On macOS, Homebrew provides the CLI as **`whisper-cli`**. You can verify:

```bash
which whisper-cli
```

Download a GGML Whisper model. For **low latency**, use **base**:

```bash
pip install -U huggingface_hub
python - <<'PY'
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id="ggerganov/whisper.cpp",
    filename="ggml-base.bin",
    local_dir="models",
    local_dir_use_symlinks=False,
)
print("Downloaded to:", path)
PY
```

Set:

```bash
export WHISPER_CPP_BIN="whisper-cli"
export WHISPER_CPP_MODEL="models/ggml-base.bin"
```

## Expert services data (required)

Place your services JSON file at:

- `ExpertServices.json` (this file is **ignored by git**)

If your file is elsewhere, set:

```bash
export EXPERT_SERVICES_JSON="/absolute/path/to/ExpertServices.json"
```

### Ollama (required)

Install and run Ollama, then pull the model:

```bash
ollama pull phi3:mini
```

### Piper (required)

Install Piper TTS Python package:

```bash
pip install -r requirements.txt
```

Download a piper `.onnx` voice model and set (your current file lives under `models/piper/en/en_US/lessac/medium/...`):

```bash
export PIPER_MODEL="models/piper/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
```

Many voices also ship with an accompanying `.onnx.json`. If your voice needs it, set:

```bash
export PIPER_MODEL_JSON="models/piper/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
```

## 3) LiveKit (agents-playground.livekit.io) real-time “call agent”

If you want the best “audio call” experience (echo cancellation, jitter buffering, real-time duplex audio), use LiveKit.

### Setup
- Create a LiveKit Cloud project (or run your own LiveKit server)
- Get:
  - `LIVEKIT_URL` (wss://…)
  - `LIVEKIT_API_KEY`
  - `LIVEKIT_API_SECRET`
- Pick a room name, e.g. `demo`

### Run the agent (joins the LiveKit room)

```bash
export LIVEKIT_URL="wss://<your-livekit-url>"
export LIVEKIT_API_KEY="<key>"
export LIVEKIT_API_SECRET="<secret>"
export LIVEKIT_ROOM="demo"
export LIVEKIT_IDENTITY="expert-service-agent"

python livekit_realtime_agent.py
```

Then open `agents-playground.livekit.io`, connect to the same project + room, and talk to the agent.

Notes:
- This agent uses your local stack: whisper.cpp STT + BM25 + Ollama rerank-on-final + Piper TTS.
- Ensure `PIPER_MODEL`, `WHISPER_CPP_MODEL`, and `OLLAMA_MODEL=phi3:mini` are set correctly before running.
- Put secrets in a local `.env` file (not committed). This repo includes `.gitignore` entries for `.env`.

## Environment variables

- `WHISPER_CPP_BIN` (default: `whisper-cli`)
- `WHISPER_CPP_MODEL` (default: `models/ggml-base.bin`)
- `WHISPER_CPP_LANG` (default: `en`)
- `FFMPEG_BIN` (default: `ffmpeg`)
- `EXPERT_SERVICES_JSON` (default: `ExpertServices.json`)
- `OLLAMA_BASE_URL` (default: `http://127.0.0.1:11434`)
- `OLLAMA_MODEL` (default: `phi3:mini`)
- `OLLAMA_TIMEOUT_S` (default: `20`)
- `PIPER_MODEL` (default: `models/piper/en/en_US/lessac/medium/en_US-lessac-medium.onnx`)
- `PIPER_MODEL_JSON` (optional)

# AI Search Service with Typo Handling

An intelligent service search engine that handles typos, spelling errors, and provides semantic search capabilities.

## Features

### 1. **Semantic Search**
- Uses `SentenceTransformers` (all-MiniLM-L6-v2) for deep semantic understanding
- Understands intent beyond exact keywords
- Example: "fix leaking taps" matches plumbing service even without exact words

### 2. **Fuzzy Matching for Typo Tolerance**
- Handles common typos automatically
- Uses `fuzzywuzzy` (optional) or built-in `difflib` for fuzzy matching
- Examples:
  - `plumbing reapir` → finds "Plumbing Repair Service"
  - `dearmtology` → finds "Dermatology Consultation"
  - `acne treatmnt` → finds "Acne Treatment"

### 3. **Spell Correction**
- Automatically corrects misspellings using service vocabulary
- Suggests corrections before searching
- Example: `woemens helath` → searches for `womens health`

### 4. **Hybrid Scoring**
- Combines semantic (70%) + fuzzy matching (30%) for best results
- Automatically adjusts based on match type:
  - **Semantic**: For contextual relevance
  - **Fuzzy**: For typo/spelling tolerance
  - **Hybrid**: When both methods match

### 5. **Intent Detection**
- Detects user intent: `greeting`, `service_search`, `help`, or `unknown`
- Handles non-search queries gracefully

## Installation

### Basic (Semantic + Keyword Fallback)
```bash
pip install sentence-transformers numpy
```

### With Enhanced Fuzzy Matching
```bash
pip install sentence-transformers numpy fuzzywuzzy python-Levenshtein
```

## Usage

### Command Line
```bash
# Interactive search
python AI_Searchpy.py services.json

# Single query
python AI_Searchpy.py services.json "plumbing repair"

# With custom results count
python AI_Searchpy.py services.json "botox" --top-k 5

# With custom embedding model
python AI_Searchpy.py services.json "acne" --model "all-MiniLM-L6-v2"
```

### Test Typo Handling
```bash
python test_search.py
```

### In Python
```python
from pathlib import Path
from AI_Searchpy import build_search_engine

engine = build_search_engine(Path("services.json"))

# Search with typo
results = engine.search("plumbing reapir", top_k=3)
for result in results:
    print(f"{result.service.title} ({result.score:.2f}) - {result.match_type}")
```

## JSON Format

Services should be in this format:
```json
[
  {
    "title": "Service Name",
    "description": "Detailed description of the service",
    "tags": ["tag1", "tag2", "tag3"]
  }
]
```

## How It Works

### Search Pipeline

1. **Input**: User query (potentially with typos)
2. **Tokenization**: Break query into tokens
3. **Spell Correction**: Correct tokens using vocabulary
4. **Scoring**:
   - **Semantic Score**: Embed corrected query, compare with service embeddings
   - **Fuzzy Score**: Check for token matches with typo tolerance
5. **Combination**: Blend scores (70% semantic + 30% fuzzy)
6. **Ranking**: Sort by combined score, filter by threshold
7. **Output**: Top K results with match quality info

### Example: Typo Handling

Query: `plumbing reapir` (typo in "repair")

```
1. Tokenize: ["plumbing", "reapir"]
2. Correct: ["plumbing", "repair"] (reapir → repair via fuzzy match)
3. Semantic: Search for "plumbing repair" in embeddings
4. Fuzzy: Match "plumbing" & "repair" in service text
5. Combine: 70% semantic + 30% fuzzy
6. Result: "Plumbing Repair Service" ✓
```

## Tuning Parameters

Edit these in `AI_Searchpy.py`:

```python
INTENT_THRESHOLD = 0.33      # Confidence threshold for intent detection
RESULT_THRESHOLD = 0.25      # Minimum score to return a result
```

Fuzzy matching thresholds in search methods:
- `find_best_fuzzy_match(threshold=0.6)`: Token matching threshold
- `correct_spelling(threshold=0.8)`: Spell correction confidence

## Limitations & Trade-offs

| Aspect | Pros | Cons |
|--------|------|------|
| **Semantic Search** | Understands context | Slower, needs model download |
| **Fuzzy Matching** | Fast, typo-tolerant | May match unrelated words |
| **70/30 Split** | Balanced approach | May need tuning for your data |

## Performance Notes

- First run: ~15-30s (downloads embedding model)
- Subsequent runs: <100ms for search
- Memory: ~200MB for embeddings

## FAQ

**Q: What if services.json has 1000+ items?**
- Still works, slightly slower. Consider batching for large catalogs.

**Q: Can I use a different embedding model?**
- Yes: `--model "distiluse-base-multilingual-cased-v2"` (smaller/faster)

**Q: How are ties broken when scores are equal?**
- By original service index order.

**Q: Can I add custom spell-checking dictionary?**
- Yes, modify the `vocabulary` set in `ServiceSearchEngine.__init__`
