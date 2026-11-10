# Dooboo Flutter client

## Requirements

- Flutter stable with Dart 3.9 or later.
- A running Dooboo FastAPI backend.
- Microphone permission on the target device.

## Create platform scaffolding

This repository originally contained only `app/lib/main.dart`; Android/iOS platform folders and generated project metadata were absent. Create them once:

```bash
cd app
flutter create --platforms=android,ios .
flutter pub get
```

## Run

Android emulator:

```bash
flutter run --dart-define=DOOBOO_API_BASE_URL=http://10.0.2.2:8001
```

Physical Android/iOS device on the same LAN:

```bash
flutter run --dart-define=DOOBOO_API_BASE_URL=http://YOUR_COMPUTER_LAN_IP:8001
```

Use HTTPS for deployment. Android may require a development-only cleartext traffic exception for local HTTP in `android/app/src/main/AndroidManifest.xml`; do not carry that exception into a production release.

## Audio flow

1. Tap **녹음 시작** and grant microphone permission.
2. Tap **녹음 종료 및 분석**.
3. The client uploads the same WAV recording to transcription and tone-classification endpoints in parallel.
4. The recognized text and tone label are sent to chat generation.
5. The text reply is synthesized by Typecast and played in the app.

The client has no OpenAI or Typecast API keys. Configure these only on the backend using the repository-root `.env` file. The app currently uses manual recording because the previous repository did not include its Porcupine wake-word model files.
