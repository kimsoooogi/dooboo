from pathlib import Path

import numpy as np
import soundfile as sf

from classifier.features import FEATURE_COUNT, FEATURE_NAMES, extract_features


def _write_test_tone(path: Path, duration: float = 1.5) -> None:
    sample_rate = 22_050
    time = np.arange(int(sample_rate * duration), dtype=np.float32) / sample_rate
    audio = 0.3 * np.sin(2 * np.pi * 220 * time) + 0.1 * np.sin(2 * np.pi * 440 * time)
    sf.write(path, audio, sample_rate)


def test_feature_schema_is_332_dimensions():
    assert FEATURE_COUNT == 332
    assert len(FEATURE_NAMES) == 332
    assert len(set(FEATURE_NAMES)) == 332


def test_feature_extractor_returns_finite_vector(tmp_path):
    audio_path = tmp_path / "sample.wav"
    _write_test_tone(audio_path)

    features = extract_features(audio_path)

    assert features.shape == (FEATURE_COUNT,)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()


def test_feature_extractor_rejects_missing_file(tmp_path):
    missing = tmp_path / "missing.wav"
    try:
        extract_features(missing)
    except FileNotFoundError:
        return
    raise AssertionError("Expected FileNotFoundError")
