# S2S local project

Local speech2speech system. All dependencies and models are stored locally in this project folder. No external API calls required.

## Stack

- **ASR**: faster-whisper
- **LLM**: Ollama (configurable via `OLLAMA_MODEL`)
- **TTS**: OpenVoice (voice cloning) / Coqui TTS / Piper (fallback)
- **RAG**: sentence-transformers + FAISS (work in progress, more features coming)
- **Audio**: PyAudio

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp env.example .env
```

**TTS Options**:
- **OpenVoice** (recommended): Download checkpoints from the link in error message, place in `./checkpoints/`
- **Piper** (fallback): Download binary from [Piper releases](https://github.com/rhasspy/piper/releases) and model from [Piper voices](https://huggingface.co/rhasspy/piper-voices), place in `./models/piper/` (binary as `piper`, model as `.onnx` file)
- **Coqui TTS**: Auto-downloads on first use (requires Python < 3.13)

**Ollama**: `ollama serve` (configure model via `OLLAMA_MODEL` in `.env`)

## Run

```bash
cd src && python main.py
```

Input: text or voice (ENTER for voice mode)  
Output: text or voice (`/mode text` or `/mode voice`)

## Commands

- `insert info` - Add to knowledge base
- `print info` - View knowledge base
- `delete info` - Clear knowledge base
- `use rag <query>` - Explicit RAG query
- `/mode text|voice` - Toggle output mode

## Config

Edit `.env`:

- `OLLAMA_MODEL` - Ollama model name (default: `qwen3:1.7b`). Set to any installed Ollama model (e.g., `llama3:8b`, `mistral:7b`, `phi3:mini`)
- `WHISPER_MODEL_SIZE` - Whisper size (default: `base.en`)
- `WHISPER_DEVICE` - `cpu` or `cuda`
- `EMBEDDING_MODEL` - Sentence transformer (default: `BAAI/bge-small-en-v1.5`)
- `RAG_TOP_K` - Retrieval count (default: `3`)
- `DISABLE_REASONING` - Filter reasoning tags (default: `true`)


## Architecture

```
src/
├── main.py      # Orchestration
├── audio.py     # I/O
├── asr.py       # faster-whisper
├── tts.py       # Piper TTS
├── tts_openvoice.py  # OpenVoice
├── tts_coqui.py # Coqui TTS
├── llm.py       # Ollama client
└── rag.py       # FAISS + embeddings
```

**RAG**: Currently opt-in via `use rag` command. Normal queries don't use context. This feature is work in progress - more functionality will be added in the future.
