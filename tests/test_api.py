from io import BytesIO

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from classifier import main as api
from classifier.features import FEATURE_COUNT, FEATURE_NAMES, FEATURE_VERSION


class DummyClassifier:
    classes_ = np.array([
        "far_loud", "far_normal", "far_small",
        "near_loud", "near_normal", "near_small",
    ])
    n_features_in_ = FEATURE_COUNT

    def predict(self, X):
        assert X.shape == (1, FEATURE_COUNT)
        return np.array(["near_normal"])

    def predict_proba(self, X):
        return np.array([[0.01, 0.02, 0.02, 0.10, 0.80, 0.05]])


def _wav_bytes(duration: float = 1.5) -> bytes:
    sample_rate = 22_050
    time = np.arange(int(sample_rate * duration), dtype=np.float32) / sample_rate
    audio = 0.3 * np.sin(2 * np.pi * 220 * time) + 0.1 * np.sin(2 * np.pi * 440 * time)
    buffer = BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV")
    return buffer.getvalue()


def _valid_bundle():
    model = DummyClassifier()
    return {
        "model": model,
        "feature_version": FEATURE_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "feature_count": FEATURE_COUNT,
        "classes": model.classes_.tolist(),
    }


def test_health_is_available_without_model(monkeypatch):
    monkeypatch.setattr(api, "model_bundle", None)
    with TestClient(api.app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is False


def test_prediction_returns_label_and_confidence(monkeypatch):
    with TestClient(api.app) as client:
        monkeypatch.setattr(api, "model_bundle", _valid_bundle())
        response = client.post(
            "/predict-audio",
            files={"file": ("sample.wav", _wav_bytes(), "audio/wav")},
        )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["prediction"] == "near_normal"
    assert 0.0 <= payload["confidence"] <= 1.0


def test_prediction_rejects_unsupported_extension(monkeypatch):
    with TestClient(api.app) as client:
        monkeypatch.setattr(api, "model_bundle", _valid_bundle())
        response = client.post(
            "/predict-audio",
            files={"file": ("sample.txt", b"not audio", "text/plain")},
        )
    assert response.status_code == 415


def test_prediction_requires_model(monkeypatch):
    with TestClient(api.app) as client:
        monkeypatch.setattr(api, "model_bundle", None)
        response = client.post(
            "/predict-audio",
            files={"file": ("sample.wav", _wav_bytes(), "audio/wav")},
        )
    assert response.status_code == 503


def test_provider_endpoints_fail_closed_when_unconfigured(monkeypatch):
    monkeypatch.setattr(api, "OPENAI_API_KEY", "")
    monkeypatch.setattr(api, "TYPECAST_API_KEY", "")
    monkeypatch.setattr(api, "TYPECAST_VOICE_ID", "")
    with TestClient(api.app) as client:
        transcribe = client.post(
            "/transcribe-audio",
            files={"file": ("sample.wav", _wav_bytes(), "audio/wav")},
        )
        chat = client.post("/chat", json={"message": "안녕"})
        speech = client.post("/synthesize-speech", json={"text": "안녕"})
    assert transcribe.status_code == 503
    assert chat.status_code == 503
    assert speech.status_code == 503


def test_model_bundle_rejects_feature_order_mismatch():
    bundle = _valid_bundle()
    bundle["feature_names"] = list(reversed(FEATURE_NAMES))
    try:
        api._check_model_bundle(bundle)
    except RuntimeError:
        return
    raise AssertionError("Expected model schema mismatch to be rejected")


def test_typecast_preset_payload_contract():
    request = api.SpeechRequest(text="조용히 말할게", tone="whisper")
    assert request.tone == "whisper"
