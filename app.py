# app.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import tempfile
from faster_whisper import WhisperModel
from typing import Optional, Dict
from googletrans import Translator

# ---------------- CONFIG ----------------
MODEL_SIZE = os.environ.get("WHISPER_MODEL", "small")  # good balance for Hindi
DEVICE = os.environ.get("DEVICE", "cpu")
COMPUTE_TYPE = os.environ.get("COMPUTE_TYPE", "int8")

# ---------------- APP INIT ----------------
app = FastAPI(title="🎧 Whisper Transcriber", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://wbstatus.in",
        "https://render-whisper.onrender.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- MODEL LOAD ----------------
print(f"🚀 Loading Whisper model: {MODEL_SIZE} on {DEVICE} ({COMPUTE_TYPE})")
try:
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    print("✅ Whisper model loaded successfully!")
except Exception as e:
    print(f"❌ Model load failed: {e}")
    raise RuntimeError(f"Model load error: {e}")

translator = Translator()  # For Hindi normalization

# ---------------- ROUTES ----------------
@app.get("/health")
async def health() -> Dict[str, str]:
    """Health check endpoint"""
    return {"status": "ok", "model": MODEL_SIZE, "device": DEVICE}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), language: Optional[str] = None):
    """Transcribe and auto-normalize Hindi (Devanagari)"""
    tmp_path = None
    try:
        # Save uploaded file temporarily
        suffix = os.path.splitext(file.filename)[1] or ".mp3"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(await file.read())

        if not os.path.exists(tmp_path):
            raise HTTPException(status_code=500, detail="Temporary file not saved correctly.")

        print(f"🎵 Received file: {file.filename}")

        # Configure transcription
        transcribe_kwargs = {
            "beam_size": 5,
            "vad_filter": True,
            "temperature": 0.2,
        }

        if language:
            transcribe_kwargs["language"] = language
            print(f"🌐 Using provided language: {language}")
        else:
            print("🌐 Auto-detecting language...")

        # Run transcription
        segments_gen, info = model.transcribe(tmp_path, **transcribe_kwargs)
        detected_lang = getattr(info, "language", "unknown")
        print(f"✅ Detected Language: {detected_lang}")

        segments, full_text_parts = [], []

        for i, segment in enumerate(segments_gen):
            seg_text = getattr(segment, "text", "").strip()
            seg_data = {
                "id": i,
                "start": float(getattr(segment, "start", 0)),
                "end": float(getattr(segment, "end", 0)),
                "text": seg_text
            }
            segments.append(seg_data)
            full_text_parts.append(seg_text)

        if not segments:
            raise HTTPException(status_code=400, detail="No speech detected in the audio.")

        final_text = " ".join(full_text_parts).strip()

        # ---------------- AUTO NORMALIZATION ----------------
        translated_text = final_text
        if detected_lang in ["hi", "ur", "unknown"]:
            try:
                print("🪄 Normalizing to Devanagari Hindi script...")
                translated_text = translator.translate(final_text, dest="hi").text
            except Exception as t_err:
                print(f"⚠️ Translation fallback failed: {t_err}")

        result = {
            "original_text": final_text,
            "translated_text": translated_text,
            "segments": segments,
            "info": {
                "detected_language": detected_lang,
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
                print(f"⚠️ Temp cleanup failed: {cleanup_err}")


# ---------------- RUN SERVER ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
