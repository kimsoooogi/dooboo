# 두부 (Dooboo) — 음성 어조 인식 AI 음성비서

> 사용자의 음성 톤(속삭임 / 일반 / 외침)을 분류하고, 결과를 응답 음성 스타일에 반영하는 음성비서 프로젝트

## 어조 분류 파이프라인

학습과 추론은 `classifier/features.py`의 동일한 `extract_features()` 함수를 사용합니다.

| 특징 | 집계 | 차원 |
|---|---|---:|
| MFCC | 평균 + 표준편차 | 26 |
| Chroma STFT | 평균 + 표준편차 | 24 |
| Mel-spectrogram | 평균 + 표준편차 | 256 |
| Spectral Contrast | 평균 + 표준편차 | 14 |
| Tonnetz | 평균 + 표준편차 | 12 |
| **합계** | | **332** |

피처 순서는 MFCC → Chroma → Mel → Spectral Contrast → Tonnetz로 고정합니다. 모델 파일에는 피처 버전과 차원 정보가 저장되며 서버는 정보가 일치하지 않는 모델을 거부합니다.

## 학습 데이터

저장소에 오디오 파일은 포함하지 않습니다. `data/` 아래에 `far_loud`, `far_normal`, `far_small`, `near_loud`, `near_normal`, `near_small` 폴더를 만들고 각 클래스의 오디오를 넣으세요. 지원 형식은 WAV, M4A, MP3, FLAC, OGG입니다. 일부 형식은 FFmpeg가 필요할 수 있습니다.

## 모델 학습

```bash
pip install -r requirements.txt
cd classifier
python train_model.py --data-dir ../data --output random_forest_model.joblib
```

클래스별 최소 2개 이상의 정상 오디오 파일을 요구하며 stratified 80:20 홀드아웃 분할로 평가합니다. 동일 원본 음원에서 파생한 샘플이 있다면 원본 음원 단위 그룹 분할로 데이터 누수를 방지해야 합니다.

## API 실행

```bash
cd classifier
uvicorn main:app --host 0.0.0.0 --port 8001
```

모델 파일이 없으면 서버는 시작되지만 `/predict-audio`는 HTTP 503을 반환합니다. 상태 확인은 `GET /health`입니다.

```bash
curl -X POST http://localhost:8001/predict-audio -F "file=@sample.wav"
```

## 보안

- 저장소에 노출된 Typecast 토큰과 Porcupine 키는 제공자 콘솔에서 폐기·재발급해야 합니다.
- OpenAI 및 Typecast 비밀키는 모바일 앱에 넣지 마세요. 앱 바이너리에서 추출될 수 있으므로 서버 측 프록시와 환경변수로 관리해야 합니다.
- 실제 배포에서는 HTTPS, 인증, 요청 제한을 적용하세요. 학습 데이터와 모델 파일은 저장소에 커밋하지 않습니다.

## 성능 수치

기존 노트북의 Random Forest 96.67%, SVM 94.17% 출력은 기존 CSV와 당시 분할에 대한 기록이며 새 파이프라인의 재현 결과가 아닙니다. 새 학습 스크립트의 평가 결과를 확인한 뒤 README에 검증된 수치를 기재해야 합니다. 기존 94% 정확도 및 지연 시간 40% 단축 주장은 독립 재현 자료가 확인되기 전까지 검증된 성과로 간주하지 않습니다.
