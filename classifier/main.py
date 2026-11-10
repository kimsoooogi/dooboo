"""FastAPI service for Dooboo audio classification and provider proxies."""
from __future__ import annotations

import logging
import mimetypes
import os
import tempfile
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Literal

import httpx
import joblib
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

try:
    from .features import FEATURE_COUNT, FEATURE_NAMES, FEATURE_VERSION, extract_features
except ImportError:  # Running with the classifier directory as the working directory.
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

CLASSIFICATION_EXTENSIONS = {".wav", ".m4a", ".mp3", ".flac", ".ogg", ".webm"}
TRANSCRIPTION_EXTENSIONS = {".wav", ".m4a", ".mp3", ".mp4", ".mpeg", ".mpga", ".webm"}
MIME_TYPES = {
    ".wav": "audio/wav", ".m4a": "audio/mp4", ".mp3": "audio/mpeg",
    ".mp4": "audio/mp4", ".mpeg": "audio/mpeg", ".mpga": "audio/mpeg",
    ".webm": "audio/webm", ".flac": "audio/flac", ".ogg": "audio/ogg",
}
RATE_LIMIT_REQUESTS = int(os.getenv("DOOBOO_RATE_LIMIT_REQUESTS", "30"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("DOOBOO_RATE_LIMIT_WINDOW_SECONDS", "60"))
_request_timestamps: dict[str, deque[float]] = defaultdict(deque)

if MAX_UPLOAD_BYTES <= 0:
    raise RuntimeError("DOOBOO_MAX_UPLOAD_BYTES must be a positive integer.")
if RATE_LIMIT_REQUESTS <= 0 or RATE_LIMIT_WINDOW_SECONDS <= 0:
    raise RuntimeError("Rate-limit environment variables must be positive integers.")

app = FastAPI(title="Dooboo Voice Assistant API", version="3.1.0")
model_bundle = None


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10_000)
    tone: str = Field(default="near_normal", max_length=32)


class SpeechRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2_000)
    tone: Literal["smart", "whisper"] = "smart"


def _rate_limit(request: Request) -> None:
    """Simple per-process IP throttle; use an edge gateway for distributed deployments."""
    client_host = request.client.host if request.client else "unknown"
    now = time.monotonic()
    timestamps = _request_timestamps[client_host]
    cutoff = now - RATE_LIMIT_WINDOW_SECONDS
    while timestamps and timestamps[0] <= cutoff:
        timestamps.popleft()
    if len(timestamps) >= RATE_LIMIT_REQUESTS:
        raise HTTPException(status_code=429, detail="Too many requests. Please retry shortly.")
    timestamps.append(now)
    if len(_request_timestamps) > 10_000:
        for host in list(_request_timestamps):
            if not _request_timestamps[host] or _request_timestamps[host][-1] <= cutoff:
                _request_timestamps.pop(host, None)


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
    expected_classes = {
        "far_loud", "far_normal", "far_small",
        "near_loud", "near_normal", "near_small",
    }
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
    if set(getattr(model, "classes_", [])) != expected_classes:
        raise RuntimeError("Model must contain exactly the six Dooboo tone classes.")
    if set(bundle.get("classes", [])) != expected_classes:
        raise RuntimeError("Model class metadata is inconsistent.")


def _valid_typecast_voice_id(value: str) -> bool:
    return value.startswith("tc_") or value.startswith("uc_")


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
        "typecast_configured": bool(TYPECAST_API_KEY and _valid_typecast_voice_id(TYPECAST_VOICE_ID)),
    }


@app.post("/predict-audio")
async def predict_audio(request: Request, file: UploadFile = File(...)) -> dict:
    _rate_limit(request)
    if model_bundle is None:
        raise HTTPException(status_code=503, detail="Classifier model is not installed.")
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in CLASSIFICATION_EXTENSIONS:
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
        prediction = await run_in_threadpool(model.predict, sample)
        result = {"prediction": str(prediction[0])}
        if hasattr(model, "predict_proba"):
            probabilities = await run_in_threadpool(model.predict_proba, sample)
            result["confidence"] = float(np.max(probabilities[0]))
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
async def transcribe_audio(request: Request, file: UploadFile = File(...)) -> dict:
    _rate_limit(request)
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=503, detail="Speech transcription is not configured on the server.")
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in TRANSCRIPTION_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Unsupported transcription audio format.")
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    filename = Path(file.filename or "recording.wav").name
    await file.close()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded audio is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Audio exceeds upload size limit.")
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
async def chat(request: Request, payload: ChatRequest) -> dict:
    _rate_limit(request)
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=503, detail="Chat generation is not configured on the server.")
    instructions = (
        "당신은 한국어 음성비서 두부입니다. 답변은 자연스럽고 간결한 한국어로 합니다. "
        "사용자의 말투를 조롱하거나 진단하지 말고, 음성 어조 분류 결과는 응답 말투를 "
        "조절하는 참고 정보로만 사용합니다. 분류 결과: " + payload.tone
    )
    body = {
        "model": OPENAI_CHAT_MODEL,
        "instructions": instructions,
        "input": payload.message,
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
            response_payload = response.json()
        return {"text": _extract_response_text(response_payload)}
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=504, detail="Chat generation timed out.") from exc
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        logger.warning("Chat provider request failed: %s", type(exc).__name__)
        raise HTTPException(status_code=502, detail="Chat provider request failed.") from exc


@app.post("/synthesize-speech")
async def synthesize_speech(request: Request, payload: SpeechRequest) -> Response:
    _rate_limit(request)
    if not TYPECAST_API_KEY or not _valid_typecast_voice_id(TYPECAST_VOICE_ID):
        raise HTTPException(status_code=503, detail="Speech synthesis is not configured on the server.")
    prompt = (
        {"emotion_type": "smart"}
        if payload.tone == "smart"
        else {"emotion_type": "preset", "emotion_preset": "whisper"}
    )
    body = {
        "model": "ssfm-v30",
        "voice_id": TYPECAST_VOICE_ID,
        "text": payload.text,
        "prompt": prompt,
        "output": {"audio_format": "wav", "volume": 100},
    }
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(90.0, connect=10.0)) as client:
            response = await client.post(
                "https://api.typecast.ai/v1/text-to-speech",
                headers={"X-API-KEY": TYPECAST_API_KEY, "Content-Type": "application/json"},
                json=body,
            )
            response.raise_for_status()
        content_type = response.headers.get("content-type", "").lower()
        if not response.content or (
            "audio/" not in content_type and "octet-stream" not in content_type
        ):
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
