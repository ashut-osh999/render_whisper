# app.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import tempfile
from faster_whisper import WhisperModel
from typing import Optional, Dict, Any

# ---------------- CONFIG ----------------
MODEL_SIZE = os.environ.get("WHISPER_MODEL", "base")  # e.g. "tiny", "base", "small", "medium", "large-v3"
LANG = os.environ.get("WHISPER_LANG", None)  # e.g. "hi" for Hindi, "en" for English
DEVICE = os.environ.get("DEVICE", "cpu")

# ---------------- APP INIT ----------------
app = FastAPI(title="Audio2SRT Whisper API", version="1.0")

# Enable CORS (for testing / production adjust this)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://wbstatus.in"],  # TODO: replace with specific domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- MODEL LOAD ----------------
print(f"🚀 Loading Whisper model: {MODEL_SIZE} on {DEVICE}")
try:
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type="int8")
    print("✅ Whisper model loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    raise RuntimeError(f"Model load error: {e}")


# ---------------- ROUTES ----------------
@app.get("/health")
async def health() -> Dict[str, str]:
    """Check API health"""
    return {"status": "ok"}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), language: Optional[str] = None):
    """Transcribe an uploaded audio file"""
    tmp_path = None
    try:
        # Save incoming file temporarily
        suffix = os.path.splitext(file.filename)[1] or ".mp3"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
            content = await file.read()
            tmp.write(content)

        if not os.path.exists(tmp_path):
            raise HTTPException(status_code=500, detail="Temporary file not saved correctly.")

        # Prepare transcription parameters
        transcribe_kwargs = {}
        if language:
            transcribe_kwargs["language"] = language
        elif LANG:
            transcribe_kwargs["language"] = LANG

        # Run transcription
        segments_gen, info = model.transcribe(tmp_path, **transcribe_kwargs)

        segments = []
        full_text_parts = []

        for i, segment in enumerate(segments_gen):
            # ✅ Safe access to start/end/text
            seg_data = {
                "id": i,
                "start": float(getattr(segment, "start", 0)),
                "end": float(getattr(segment, "end", 0)),
                "text": getattr(segment, "text", "").strip()
            }
            segments.append(seg_data)
            full_text_parts.append(seg_data["text"])

        if not segments:
            raise HTTPException(status_code=400, detail="No speech segments detected in the audio.")

        result = {
            "text": " ".join(full_text_parts).strip(),
            "segments": segments,
            "info": {
                "language": getattr(info, "language", None),
                "duration": getattr(info, "duration", None)
            }
        }

        return result

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception as cleanup_err:
                print(f"⚠️ Could not remove temp file: {cleanup_err}")


# ---------------- RUN SERVER ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
