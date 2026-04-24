# 🫘 두부 (Dooboo) — 음성 어조 인식 AI 음성비서

> 사용자의 **음성 톤(속삭임 / 일반 / 외침)**을 실시간으로 감지하여 그에 맞는 어조와 볼륨으로 응답하는 초개인화 음성비서 서비스

<br>

## 📌 프로젝트 개요

| 항목 | 내용 |
|------|------|
| 개발 기간 | 2024.03.04 ~ 2024.11.25 |
| 팀 구성 | 2인 |
| 담당 역할 | 데이터 파이프라인 설계, 멀티 모델 벤치마킹, FastAPI 서버 구축 |
| 사용 기술 | Python, FastAPI, librosa, scikit-learn, Flutter, Whisper, GPT-4 |

<br>

## 🏗️ 시스템 아키텍처

```
[사용자 음성 입력]
       │
       ▼
[Porcupine 웨이크워드 감지] ← "두부야"
       │
       ├──────────────────────┐
       ▼                      ▼
[Whisper STT]         [어조 분류 모델]
(텍스트 변환)          (FastAPI 서버)
       │                      │
       └──────────┬───────────┘
                  ▼
            [GPT-4 응답 생성]
            (어조 컨텍스트 반영)
                  │
                  ▼
            [TTS 파라미터 동적 조정]
            (볼륨 / 속도 / 피치)
                  │
                  ▼
            [Flutter UI 출력]
```

<br>

## 🎯 핵심 성과

- **정확도 94% 달성** — 초기 81% 대비 13%p 향상
- **지연 시간 40% 단축** — FastAPI 비동기 처리 도입
- **6개 클래스 분류** — far/near × loud/normal/small

<br>

## 📁 프로젝트 구조

```
dooboo/
├── classifier/
│   ├── main.py                  # FastAPI 서버 (어조 분류 API)
│   ├── model_trainer.ipynb      # 모델 학습 및 벤치마킹
│   └── data_processing.ipynb   # 오디오 피처 추출 파이프라인
│
├── app/
│   └── lib/main.dart            # Flutter 모바일 UI
│
├── docs/
│   └── 06_prototype.ipynb       # 초기 프로토타입 (Google STT + Tacotron2)
│
└── requirements.txt
```

<br>

## 🧠 어조 분류 모델

### 피처 엔지니어링 (161차원)

| 피처 | 설명 | 차원 |
|------|------|------|
| MFCC (mean + std) | 음색 특징 | 26 |
| Mel-spectrogram (mean + std) | 청각 인지 주파수 에너지 | 256 |
| Chroma STFT (mean + std) | 음의 높낮이 | 24 |
| Spectral Contrast (mean + std) | 주파수 대비 특성 | 14 |
| Tonnetz (mean + std) | 조화 관계 | 12 |

### 알고리즘 벤치마킹 결과

| 모델 | 정확도 | 비고 |
|------|--------|------|
| Decision Tree | ~75% | 과적합 문제 |
| SVM | ~80% | 데이터 스케일 민감 |
| **Random Forest** | **94%** | ✅ 최종 채택 |
| Sequential (DL) | ~89% | 추론 속도 느림 |

<br>

## 🛠️ 이슈 및 해결

### Issue 1. 모델 정확도 정체 (81% → 94%)
- **원인**: 단순 MFCC만으로는 원거리 속삭임 vs 근거리 속삭임 구분 어려움
- **해결**: 161차원 피처 엔지니어링 + 4개 알고리즘 대조 실험
- **결과**: Random Forest 최종 채택, 94% 달성

### Issue 2. 실시간 처리 지연 문제
- **원인**: 동기 처리 방식 병목, 고중량 DL 모델 매번 로드
- **해결**: FastAPI 비동기 파이프라인 + joblib 기반 ML 모델 채택
- **결과**: 전체 지연 시간 40% 단축

### Issue 3. 협업 데이터 불일치
- **원인**: 피처 순서, 스케일링 방식 개인별 상이
- **해결**: 공통 데이터 명세서 제작 및 `data_processing.ipynb` 모듈화
- **결과**: 데이터 오염 제거, 통합 시간 단축

<br>

## 🚀 실행 방법

```bash
# 의존성 설치
pip install -r requirements.txt

# FastAPI 서버 실행
cd classifier
uvicorn main:app --reload

# API 테스트
curl -X POST "http://localhost:8000/predict-audio" \
     -F "file=@sample.wav"
```

<br>

## 📦 기술 스택

`Python` `FastAPI` `scikit-learn` `librosa` `Flutter` `Dart`
`OpenAI Whisper` `GPT-4` `Porcupine` `joblib`

<br>

## 📎 관련 자료

- [졸업전시 발표 영상](https://youtu.be/x6LAp4JKrEg)
