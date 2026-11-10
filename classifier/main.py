"""FastAPI inference service for the Dooboo tone classifier."""
from pathlib import Path
import os, tempfile
import joblib
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool
from features import FEATURE_COUNT, FEATURE_VERSION, extract_features

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("DOOBOO_MODEL_PATH", BASE_DIR / "random_forest_model.joblib"))
MAX_UPLOAD_BYTES = int(os.getenv("DOOBOO_MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))
ALLOWED_EXTENSIONS = {".wav", ".m4a", ".mp3", ".flac", ".ogg"}
app = FastAPI(title="Dooboo Tone Classifier", version="2.0.0")
model_bundle = None

@app.on_event("startup")
def load_model():
    global model_bundle
    if not MODEL_PATH.is_file():
        model_bundle = None
        return
    bundle = joblib.load(MODEL_PATH)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise RuntimeError("Unsupported model artifact; retrain with train_model.py.")
    if bundle.get("feature_version") != FEATURE_VERSION or bundle.get("feature_count") != FEATURE_COUNT:
        raise RuntimeError("Model feature schema does not match inference pipeline.")
    model_bundle = bundle

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model_bundle is not None, "feature_version": FEATURE_VERSION}

@app.post("/predict-audio")
async def predict_audio(file: UploadFile = File(...)):
    if model_bundle is None:
        raise HTTPException(status_code=503, detail="Model missing; train it with train_model.py.")
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
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
        raise HTTPException(status_code=500, detail="Audio inference failed.") from exc
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        await file.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=False)
