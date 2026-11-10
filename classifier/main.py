"""FastAPI service for Dooboo audio classification and provider proxies."""
from __future__ import annotations

import logging
import mimetypes
import os
import tempfile
from pathlib import Path
from typing import Literal

import httpx
import joblib
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from features import FEATURE_COUNT, FEATURE_NAMES, FEATURE_VERSION, extract_features

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR.parent / ".env", override=False)
logger = logging.getLogger("dooboo.api")

MODEL_PATH = Path(os.getenv("DOOBOO_MODEL_PATH", BASE_DIR / "random_forest_model.joblib"))
MAX_UPLOAD_BYTES = int(os.getenv("DOOBOO_MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_TRANSCRIPTION_MODEL = os.getenv("OPENAI_TRANSCRIPTION_MODEL", "gpt-transcribe").strip()
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-6.1-terra").strip()
TYPECAST_API_KEY = os.getenv("TYPECAST_API_KEY", "").strip()
TYPECAST_VOICE_ID = os.getenv("TYPECAST_VOICE_ID", "").strip()

ALLOWED_EXTENSIONS = {".wav", ".m4a", ".mp3", ".flac", ".ogg", ".webm", ".mp4", ".mpeg", ".mpga"}
MIME_TYPES = {
    ".wav": "audio/wav", ".m4a": "audio/mp4", ".mp3": "audio/mpeg",
    ".flac": "audio/flac", ".ogg": "audio/ogg", ".webm": "audio/webm",
    ".mp4": "audio/mp4", ".mpeg": "audio/mpeg", ".mpga": "audio/mpeg",
}

app = FastAPI(title="Dooboo Voice Assistant API", version="3.0.0")
model_bundle = None


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10_000)
    tone: str = Field(default="near_normal", max_length=32)


class SpeechRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2_000)
    tone: Literal["smart", "whisper"] = "smart"


def _extract_response_text(payload: dict) -> str:
    direct_text = payload.get("output_text")
    if isinstance(direct_text, str) and direct_text.strip():
        return direct_text.strip()

    fragments = []
    for item in payload.get("output", []):
        if item.get("type") != "message":
            continue
        for part in item.get("content", []):
            if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                fragments.append(part["text"])
    text = "".join(fragments).strip()
    if not text:
        raise ValueError("OpenAI response did not contain output text")
    return text


def _check_model_bundle(bundle: object) -> None:
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise RuntimeError("Unsupported model artifact; retrain with train_model.py.")
    if bundle.get("feature_version") != FEATURE_VERSION:
        raise RuntimeError("Model feature version does not match the inference pipeline.")
    if bundle.get("feature_count") != FEATURE_COUNT:
        raise RuntimeError("Model feature count does not match the inference pipeline.")
    if bundle.get("feature_names") != list(FEATURE_NAMES):
        raise RuntimeError("Model feature order does not match the inference pipeline.")
    model = bundle["model"]
    if not hasattr(model, "predict") or getattr(model, "n_features_in_", None) != FEATURE_COUNT:
        raise RuntimeError("Model is missing or has an incompatible input schema.")
    if set(getattr(model, "classes_", [])) != set(bundle.get("classes", [])):
        raise RuntimeError("Model class metadata is inconsistent.")


@app.on_event("startup")
def load_model() -> None:
    global model_bundle
    if not MODEL_PATH.is_file():
        model_bundle = None
        logger.warning("Classifier artifact not found at %s", MODEL_PATH)
        return
    loaded = joblib.load(MODEL_PATH)
    _check_model_bundle(loaded)
    model_bundle = loaded
    logger.info("Loaded classifier artifact: %s", MODEL_PATH)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": model_bundle is not None,
        "feature_version": FEATURE_VERSION,
        "openai_configured": bool(OPENAI_API_KEY),
        "typecast_configured": bool(TYPECAST_API_KEY and TYPECAST_VOICE_ID),
    }


@app.post("/predict-audio")
async def predict_audio(file: UploadFile = File(...)) -> dict:
    if model_bundle is None:
        raise HTTPException(status_code=503, detail="Classifier model is not installed.")
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".wav", ".m4a", ".mp3", ".flac", ".ogg", ".webm"}:
        raise HTTPException(status_code=415, detail="Unsupported audio format.")
    temp_path = None
    try:
        content = await file.read(MAX_UPLOAD_BYTES + 1)
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded audio is empty.")
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Audio exceeds upload size limit.")
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
            temp.write(content)
            temp_path = temp.name
        features = await run_in_threadpool(extract_features, temp_path)
        sample = features.reshape(1, -1)
        model = model_bundle["model"]
        result = {"prediction": str(model.predict(sample)[0])}
        if hasattr(model, "predict_proba"):
            result["confidence"] = float(np.max(model.predict_proba(sample)[0]))
        return result
    except HTTPException:
        raise
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=f"Unable to process audio: {exc}") from exc
    except Exception as exc:
        logger.exception("Audio classification failed")
        raise HTTPException(status_code=500, detail="Audio classification failed.") from exc
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        await file.close()


@app.post("/transcribe-audio")
async def transcribe_audio(file: UploadFile = File(...)) -> dict:
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=503, detail="Speech transcription is not configured on the server.")
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Unsupported audio format.")
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    await file.close()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded audio is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Audio exceeds upload size limit.")

    filename = Path(file.filename or "recording.wav").name
    media_type = MIME_TYPES.get(suffix) or mimetypes.guess_type(filename)[0] or "application/octet-stream"
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
            response = await client.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                data=[("model", OPENAI_TRANSCRIPTION_MODEL), ("languages[]", "ko")],
                files={"file": (filename, content, media_type)},
            )
            response.raise_for_status()
            payload = response.json()
        transcript = payload.get("text")
        if not isinstance(transcript, str) or not transcript.strip():
            raise ValueError("Transcription response did not contain text")
        return {"text": transcript.strip()}
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=504, detail="Speech transcription timed out.") from exc
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        logger.warning("Speech transcription provider request failed: %s", type(exc).__name__)
        raise HTTPException(status_code=502, detail="Speech transcription provider request failed.") from exc


@app.post("/chat")
async def chat(request: ChatRequest) -> dict:
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=503, detail="Chat generation is not configured on the server.")
    instructions = (
        "당신은 한국어 음성비서 두부입니다. 답변은 자연스럽고 간결한 한국어로 합니다. "
        "사용자의 말투를 조롱하거나 진단하지 말고, 음성 어조 분류 결과는 응답의 말투를 "
        "조절하는 참고 정보로만 사용합니다. 분류 결과: " + request.tone
    )
    body = {
        "model": OPENAI_CHAT_MODEL,
        "instructions": instructions,
        "input": request.message,
        "max_output_tokens": 512,
        "store": False,
    }
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
            response = await client.post(
                "https://api.openai.com/v1/responses",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
                json=body,
            )
            response.raise_for_status()
            payload = response.json()
        return {"text": _extract_response_text(payload)}
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=504, detail="Chat generation timed out.") from exc
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        logger.warning("Chat provider request failed: %s", type(exc).__name__)
        raise HTTPException(status_code=502, detail="Chat provider request failed.") from exc


@app.post("/synthesize-speech")
async def synthesize_speech(request: SpeechRequest) -> Response:
    if not TYPECAST_API_KEY or not TYPECAST_VOICE_ID:
        raise HTTPException(status_code=503, detail="Speech synthesis is not configured on the server.")
    body = {
        "model": "ssfm-v30",
        "voice_id": TYPECAST_VOICE_ID,
        "text": request.text,
        "prompt": {"emotion_type": request.tone},
        "output": {"audio_format": "wav"},
    }
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(90.0, connect=10.0)) as client:
            response = await client.post(
                "https://api.typecast.ai/v1/text-to-speech",
                headers={"X-API-KEY": TYPECAST_API_KEY, "Content-Type": "application/json"},
                json=body,
            )
            response.raise_for_status()
        if not response.content:
            raise ValueError("Typecast returned an empty audio response")
        content_type = response.headers.get("content-type", "").lower()
        if "audio/" not in content_type and "octet-stream" not in content_type:
            raise ValueError("Typecast did not return an audio payload")
        return Response(content=response.content, media_type="audio/wav")
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=504, detail="Speech synthesis timed out.") from exc
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("Speech synthesis provider request failed: %s", type(exc).__name__)
        raise HTTPException(status_code=502, detail="Speech synthesis provider request failed.") from exc


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=False)
