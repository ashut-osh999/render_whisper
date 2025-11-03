# app.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import tempfile
from faster_whisper import WhisperModel
from typing import List, Dict, Any

# CONFIG
MODEL_SIZE = os.environ.get("WHISPER_MODEL", "base")  # "tiny", "base", "small", "medium", "large"
LANG = os.environ.get("WHISPER_LANG", None)  # e.g. "hi" or None for auto
CONCURRENCY = int(os.environ.get("WHISPER_CONCURRENCY", "1"))

app = FastAPI(title="Audio2SRT Whisper API")

# Allow hostinger domain or wildcard for testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
print(f"Loading Whisper model: {MODEL_SIZE}")
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")

@app.get("/health")
async def health():
    return {"status":"ok"}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), language: str = None):
    # Save incoming file temporarily
    try:
        suffix = os.path.splitext(file.filename)[1] or ".mp3"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
            content = await file.read()
            tmp.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")

    # Transcribe using faster_whisper
    try:
        transcribe_kwargs = {}
        if language:
            transcribe_kwargs["language"] = language
        elif LANG:
            transcribe_kwargs["language"] = LANG

        segments: List[Dict[str, Any]] = []
        full_text_parts = []
        # transcribe returns segments generator
        for segment in model.transcribe(tmp_path, **transcribe_kwargs)[1]:
            seg = {
                "id": int(segment.index),
                "start": float(segment.start),
                "end": float(segment.end),
                "text": segment.text.strip()
            }
            segments.append(seg)
            full_text_parts.append(segment.text.strip())

        result = {
            "text": " ".join([p for p in full_text_parts if p]),
            "segments": segments
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")
    finally:
        try:
            os.remove(tmp_path)
        except:
            pass

    return result

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=False)
    
