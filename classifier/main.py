from fastapi import FastAPI, UploadFile, File
import joblib
import numpy as np
import librosa
from pydub import AudioSegment
import os

app = FastAPI()

# 학습된 Random Forest 모델 로드
model = joblib.load("random_forest_model.joblib")

# 상위 10개 피처 인덱스 (전체 161차원 중 중요도 기반 선정)
TOP_FEATURE_INDICES = [308, 33, 170, 315, 169, 41, 43, 172, 44, 168]


def extract_features(file_path: str) -> np.ndarray:
    """
    오디오 파일에서 161차원 피처를 추출한 뒤 상위 10개만 반환합니다.

    지원 형식: .wav, .m4a
    추출 피처: MFCC, Mel-spectrogram, Chroma STFT, Spectral Contrast, Tonnetz
    """
    # m4a → wav 변환
    if file_path.endswith(".m4a"):
        audio = AudioSegment.from_file(file_path, format="m4a")
        temp_path = "temp.wav"
        audio.export(temp_path, format="wav")
        y, sr = librosa.load(temp_path)
        os.remove(temp_path)
    else:
        y, sr = librosa.load(file_path)

    # 피처 추출
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mel   = librosa.feature.melspectrogram(y=y, sr=sr)
    chroma   = librosa.feature.chroma_stft(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    tonnetz  = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)

    # mean + std 집계 → 161차원 벡터
    feature_vector = np.concatenate([
        np.mean(mfccs, axis=1),    np.std(mfccs, axis=1),
        np.mean(mel, axis=1),      np.std(mel, axis=1),
        np.mean(chroma, axis=1),   np.std(chroma, axis=1),
        np.mean(contrast, axis=1), np.std(contrast, axis=1),
        np.mean(tonnetz, axis=1),  np.std(tonnetz, axis=1),
    ])

    # 상위 10개 피처만 선택 (추론 속도 최적화)
    return feature_vector[TOP_FEATURE_INDICES]


@app.post("/predict-audio")
async def predict_audio(file: UploadFile = File(...)):
    """
    음성 파일을 받아 어조 클래스를 반환합니다.

    반환값 예시: "near_normal", "far_loud", "near_small" 등 6개 클래스
    """
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    features   = extract_features(temp_path)
    os.remove(temp_path)

    prediction = model.predict([features])
    return {"prediction": str(prediction[0])}


# 실행: uvicorn main:app --reload
