from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
import tempfile
from faster_whisper import WhisperModel
from typing import Optional, Dict
from deep_translator import GoogleTranslator


# ---------------- CONFIG ----------------
MODEL_SIZE = os.environ.get("WHISPER_MODEL", "small")
DEVICE = os.environ.get("DEVICE", "cpu")
COMPUTE_TYPE = os.environ.get("COMPUTE_TYPE", "int8")

# ---------------- APP INIT ----------------
app = FastAPI(title="🎧 Whisper Transcriber", version="3.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # allow all
    allow_credentials=True,
    allow_methods=["*"],     # POST + OPTIONS included
    allow_headers=["*"],
)

# ---------------- OPTIONS FIX (critical!) ----------------
@app.options("/transcribe")
async def options_transcribe():
    return JSONResponse(
        status_code=200,
        content={"message": "OK"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )


# ---------------- MODEL LOAD ----------------
print(f"🚀 Loading Whisper model: {MODEL_SIZE} on {DEVICE} ({COMPUTE_TYPE})")
try:
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    print("✅ Whisper model loaded successfully!")
except Exception as e:
    print(f"❌ Model load failed: {e}")
    raise RuntimeError(f"Model load error: {e}")


# ---------------- ROUTES ----------------
@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "model": MODEL_SIZE, "device": DEVICE}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), language: Optional[str] = None):

    tmp_path = None
    try:
        # save temp file
        suffix = os.path.splitext(file.filename)[1] or ".mp3"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(await file.read())

        print(f"🎵 Received file: {file.filename}")

        transcribe_kwargs = {
            "beam_size": 5,
            "vad_filter": True,
            "temperature": 0.2,
        }

        if language:
            transcribe_kwargs["language"] = language
            print("🌐 Using provided language:", language)
        else:
            print("🌐 Auto-detecting language...")

        # Whisper transcribe
        segments_gen, info = model.transcribe(tmp_path, **transcribe_kwargs)
        detected_lang = getattr(info, "language", "unknown")

        print(f"✅ Detected Language: {detected_lang}")

        segments = []
        texts = []

        for i, seg in enumerate(segments_gen):
            text = seg.text.strip()
            segments.append({
                "id": i,
                "start": float(seg.start),
                "end": float(seg.end),
                "text": text
            })
            texts.append(text)

        final_text = " ".join(texts).strip()

        # Translate / Normalize
        translated_text = final_text
        if detected_lang in ["hi", "ur", "unknown"]:
            try:
                translated_text = GoogleTranslator(
                    source="auto",
                    target="hi"
                ).translate(final_text)
            except Exception as tr:
                print("⚠️ Translation failed:", tr)

        return {
            "original_text": final_text,
            "translated_text": translated_text,
            "segments": segments,
            "info": {
                "detected_language": detected_lang,
                "duration": getattr(info, "duration", None)
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass


# ---------------- RUN SERVER ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
