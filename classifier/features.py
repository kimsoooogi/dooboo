"""Shared audio feature extraction used by training and inference."""
from pathlib import Path
import numpy as np
import librosa

SAMPLE_RATE = 22050
FEATURE_VERSION = "v1"
FEATURE_NAMES = (
    [f"mfcc_mean_{i}" for i in range(13)] + [f"mfcc_std_{i}" for i in range(13)]
    + [f"chroma_mean_{i}" for i in range(12)] + [f"chroma_std_{i}" for i in range(12)]
    + [f"mel_mean_{i}" for i in range(128)] + [f"mel_std_{i}" for i in range(128)]
    + [f"contrast_mean_{i}" for i in range(7)] + [f"contrast_std_{i}" for i in range(7)]
    + [f"tonnetz_mean_{i}" for i in range(6)] + [f"tonnetz_std_{i}" for i in range(6)]
)
FEATURE_COUNT = len(FEATURE_NAMES)


def extract_features(file_path: str | Path) -> np.ndarray:
    """Return a fixed-order 332-dimensional audio feature vector."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")
    y, sr = librosa.load(path.as_posix(), sr=SAMPLE_RATE, mono=True)
    if y.size == 0:
        raise ValueError("Audio file contains no samples")

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    vector = np.concatenate([
        np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
        np.mean(chroma, axis=1), np.std(chroma, axis=1),
        np.mean(mel, axis=1), np.std(mel, axis=1),
        np.mean(contrast, axis=1), np.std(contrast, axis=1),
        np.mean(tonnetz, axis=1), np.std(tonnetz, axis=1),
    ]).astype(np.float32)
    if vector.shape != (FEATURE_COUNT,) or not np.isfinite(vector).all():
        raise ValueError(f"Expected {FEATURE_COUNT} finite features; got {vector.shape}")
    return vector
