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

저장소에 오디오 파일은 포함하지 않습니다. `data/` 아래에 `far_loud`, `far_normal`, `far_small`, `near_loud`, `near_normal`, `near_small` 폴더를 만들고 각 클래스의 오디오를 넣으세요. 지원 형식은 WAV, M4A, MP3, FLAC, OGG입니다. M4A·MP3 등은 FFmpeg 디코더가 필요하므로 시스템에 FFmpeg를 설치해야 합니다.

## 모델 학습

```bash
# Run all commands from the repository root.
pip install -r requirements.txt
python -m classifier.train_model --data-dir data --output classifier/random_forest_model.joblib
```

클래스별 최소 5개 이상의 정상 오디오 파일을 요구하며 stratified 80:20 홀드아웃 분할로 평가합니다. 동일 원본 음원에서 파생한 샘플이 있다면 원본 음원 단위 그룹 분할로 데이터 누수를 방지해야 합니다.

동일한 원본 녹음에서 여러 조각을 만든 경우 `groups.csv`에 `file,group_id` 열을 두고 각 조각에 같은 `group_id`를 지정한 다음 `--groups-csv groups.csv`를 추가하세요. 파일 이름은 `data/` 폴더 기준 상대 경로여야 합니다. 이 옵션은 같은 원본 녹음의 조각이 학습·평가 세트에 동시에 들어가는 것을 방지합니다.

## API 실행

```bash
uvicorn classifier.main:app --host 0.0.0.0 --port 8001
```

모델 파일이 없으면 서버는 시작되지만 `/predict-audio`는 HTTP 503을 반환합니다. 상태 확인은 `GET /health`, 모델 준비 상태 확인은 `GET /ready`입니다. `/health`는 프로세스 생존 확인용이며 모델 미설치 시에도 `200`을 반환할 수 있습니다. 배포 오케스트레이터의 준비 상태 확인에는 `/ready`를 사용하세요.

```bash
curl -X POST http://localhost:8001/predict-audio -F "file=@sample.wav"
```

## 보안

- 저장소에 노출된 Typecast 토큰과 Porcupine 키는 제공자 콘솔에서 폐기·재발급해야 합니다.
- OpenAI 및 Typecast 비밀키는 모바일 앱에 넣지 마세요. 앱 바이너리에서 추출될 수 있으므로 서버 측 프록시와 환경변수로 관리해야 합니다.
- 실제 배포에서는 HTTPS, 인증, 요청 제한을 적용하세요. 학습 데이터와 모델 파일은 저장소에 커밋하지 않습니다.

## Flutter 앱 실행

이 저장소에는 원래 Android/iOS 플랫폼 스캐폴딩과 웨이크워드 파일이 포함되어 있지 않았습니다. 앱은 빠진 자산에 의존하지 않도록 버튼 기반 녹음으로 정리했습니다.

```bash
cd app
flutter create --platforms=android,ios .
flutter pub get
flutter run --dart-define=DOOBOO_API_BASE_URL=http://10.0.2.2:8001
```

Android 에뮬레이터에서 `10.0.2.2`는 개발 PC의 localhost를 가리킵니다. 실제 휴대전화에서는 PC의 LAN 주소를 사용해야 하며, Android 개발 빌드는 로컬 HTTP 사용을 위해 `android:usesCleartextTraffic="true"`가 필요할 수 있습니다. 운영 환경에서는 HTTPS를 사용하세요.

## Flutter 앱 설정 및 보안 한계

Flutter 앱은 `DOOBOO_API_BASE_URL`만 필요하며, OpenAI와 Typecast 키는 백엔드의 `.env`에만 보관합니다. 앱에서 외부 제공업체를 직접 호출하지 않습니다.

**중요:** `--dart-define`은 저장소에서 문자열을 제거할 뿐, 모바일 앱 바이너리에 포함된 키를 보호하지 않습니다. 이 프로젝트는 OpenAI·Typecast 호출을 백엔드 프록시로 이동했습니다. 다만 공용 배포에서는 별도의 사용자 인증과 엣지/API 게이트웨이 보호가 필요합니다. 현재 백엔드의 IP별 제한은 단일 프로세스에만 적용되는 기본 방어이며, 다중 인스턴스의 분산 제한이나 사용자 계정 인증을 대체하지 않습니다.

앱의 기본 녹음 흐름은 `/transcribe-audio`와 `/predict-audio`를 병렬 호출하고, `/chat` 및 `/synthesize-speech`를 순차 호출합니다. 외부 제공업체 호출은 서버의 `OPENAI_API_KEY`, `TYPECAST_API_KEY`, `TYPECAST_VOICE_ID` 설정이 필요합니다.

## 성능 수치

기존 노트북의 Random Forest 96.67%, SVM 94.17% 출력은 기존 CSV와 당시 분할에 대한 기록이며 새 파이프라인의 재현 결과가 아닙니다. 새 학습 스크립트의 평가 결과를 확인한 뒤 README에 검증된 수치를 기재해야 합니다. 기존 94% 정확도 및 지연 시간 40% 단축 주장은 독립 재현 자료가 확인되기 전까지 검증된 성과로 간주하지 않습니다.

## Backend 환경 설정

```bash
cp .env.example .env
# .env에 OPENAI_API_KEY, TYPECAST_API_KEY, TYPECAST_VOICE_ID 입력
pip install -r requirements.txt
uvicorn classifier.main:app --host 0.0.0.0 --port 8001
```

`.env`는 Git에 커밋하지 마세요. Typecast voice ID는 실제 API voice ID(`tc_` 또는 `uc_` 접두사)여야 합니다. 노출된 과거 API 키는 반드시 제공자 콘솔에서 폐기·재발급해야 합니다.
