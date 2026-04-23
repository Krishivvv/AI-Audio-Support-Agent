# AI Audio Support Agent

A full-stack audio-powered customer support agent built with Speech-to-Text (STT), a LangChain ReAct LLM agent with Retrieval-Augmented Generation (RAG), and Text-to-Speech (TTS). Users speak a question, the system transcribes it, queries a knowledge base, generates a response, and speaks it back — all in one pipeline.

## Pipeline

```
Audio Input → STT (Whisper) → LLM Agent + RAG (Groq / LLaMA) → TTS (Edge TTS) → Audio Output
```

## Features

- **Voice-to-voice** customer support pipeline
- **RAG knowledge base** with 16 customer support documents stored in ChromaDB
- **Semantic search** using sentence-transformers (`all-MiniLM-L6-v2`)
- **FastAPI** backend with REST endpoints
- **Streamlit UI** with text chat, audio upload/recording, health monitor, and transcript display
- Fully async pipeline — non-blocking throughout

## Stack

| Layer | Technology |
|---|---|
| STT | OpenAI Whisper (local, `base` model) |
| LLM | Groq API — LLaMA 3.3 70B Versatile |
| RAG | ChromaDB + sentence-transformers |
| TTS | Microsoft Edge TTS (free, no API key) |
| API | FastAPI + Uvicorn |
| UI | Streamlit |

## Project Structure

```
audio_support_agent/
├── src/
│   ├── stt/
│   │   └── base_stt.py          # Whisper STT with soundfile + ffmpeg fallback
│   ├── llm/
│   │   └── agent.py             # LangChain ReAct agent with ChromaDB RAG
│   ├── tts/
│   │   └── base_tts.py          # Edge TTS synthesis
│   ├── api/
│   │   └── server.py            # FastAPI server
│   ├── utils/
│   │   └── kb_test.py           # Knowledge base debug utility
│   └── pipeline.py              # STT → LLM → TTS orchestrator
├── docs/
│   └── RAG_IMPLEMENTATION_GUIDE.md
├── tests/
├── streamlit_app.py             # Streamlit UI
├── requirements.txt
└── .env.example
```

## Setup

### 1. Clone & install

```bash
git clone https://github.com/Krishivvv/AI-Audio-Support-Agent.git
cd AI-Audio-Support-Agent

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and set your API keys. At minimum you need a Groq key (free at [console.groq.com](https://console.groq.com)):

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

STT uses local Whisper and TTS uses Edge TTS — both are free with no API key required.

### 3. Run

**Terminal 1 — API server:**
```bash
python -m src.api.server
```

**Terminal 2 — Streamlit UI:**
```bash
streamlit run streamlit_app.py
```

Open **http://localhost:8501** for the UI, or **http://localhost:8000/docs** for the raw API.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | API info |
| `GET` | `/health` | Component health check |
| `POST` | `/chat/text` | Text → LLM → text response |
| `POST` | `/chat/audio` | Audio → STT → LLM → TTS → audio response |
| `GET` | `/chat/audio/{text}` | TTS-only: text to audio file |
| `POST` | `/debug/stt` | STT-only: transcribe an audio file |

### Quick test

```bash
# Health check
curl http://localhost:8000/health

# Text chat
curl -X POST http://localhost:8000/chat/text \
  -H "Content-Type: application/json" \
  -d '{"text": "What is your return policy?"}'

# Audio pipeline
curl -X POST http://localhost:8000/chat/audio \
  -F "audio=@my_question.wav" --output response.mp3
```

## Knowledge Base

The RAG system contains 16 customer support documents across these categories:

- **Returns** — 30-day policy, return steps, non-returnable items
- **Shipping** — methods, times, international shipping, order tracking
- **Support** — contact info, response times
- **Warranty & Technical** — product warranty, tech support hours
- **Account & Orders** — account management, order modifications
- **Payment & Billing** — accepted payment methods, invoices
- **Products** — availability, size guides

ChromaDB stores embeddings persistently in `data/chroma_db/` — no re-ingestion needed on restart.

## How the RAG Works

1. User query is embedded using `all-MiniLM-L6-v2`
2. ChromaDB returns the top-3 most semantically similar documents
3. Documents are injected into the LangChain ReAct agent's context
4. LLaMA 3.3 generates a grounded response from the retrieved content

See [docs/RAG_IMPLEMENTATION_GUIDE.md](docs/RAG_IMPLEMENTATION_GUIDE.md) for technical details.

## Audio Tips

- WAV format at 16 kHz mono gives the best Whisper accuracy
- For other formats (MP3, OGG, FLAC), install [ffmpeg](https://ffmpeg.org/download.html) and add it to your PATH
- The pipeline also accepts audio recorded directly in the Streamlit UI (requires `sounddevice`)

## Troubleshooting

| Problem | Fix |
|---|---|
| `ImportError` | Run `pip install -r requirements.txt` from the project root |
| Port 8000 in use | Kill the existing process or change `SERVER_PORT` in `.env` |
| Empty transcription | Speak clearly; use 16 kHz WAV; check microphone levels |
| No audio output | Verify internet connection (Edge TTS needs it) |
| Whisper slow | Switch to `"model": "tiny"` in `stt_config` inside `server.py` |
