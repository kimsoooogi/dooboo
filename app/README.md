# Dooboo Flutter client

## Requirements

- Flutter stable with Dart 3.10 or later.
- A running Dooboo FastAPI backend.
- Microphone permission on the target device.

## Create platform scaffolding

This repository originally contained only `app/lib/main.dart`; Android/iOS platform folders and generated project metadata were absent. Create them once:

```bash
cd app
flutter create --platforms=android,ios .
flutter pub get
```

## Platform permissions

After running `flutter create`, add microphone permissions to the generated platform files:

- Android: add `<uses-permission android:name="android.permission.RECORD_AUDIO" />` directly under the root `<manifest>` element in `android/app/src/main/AndroidManifest.xml`.
- iOS: add `NSMicrophoneUsageDescription` to `ios/Runner/Info.plist` with a clear explanation such as `두부가 음성을 녹음해 요청 내용을 인식합니다.`.
- For Android development against a local `http://` backend only, add `android:usesCleartextTraffic="true"` to the `<application>` element. Do not enable cleartext traffic in production; deploy HTTPS.

## Run
Android emulator:

```bash
flutter run --dart-define=DOOBOO_API_BASE_URL=http://10.0.2.2:8001
```

Physical Android/iOS device on the same LAN:

```bash
flutter run --dart-define=DOOBOO_API_BASE_URL=http://YOUR_COMPUTER_LAN_IP:8001
```

Use HTTPS for deployment. Do not ship the Android development-only cleartext traffic exception in a production release.

## Audio flow

1. Tap **녹음 시작** and grant microphone permission.
2. Tap **녹음 종료 및 분석**.
3. The client uploads the same WAV recording to transcription and tone-classification endpoints in parallel.
4. The recognized text and tone label are sent to chat generation.
5. The text reply is synthesized by Typecast and played in the app.

The client has no OpenAI or Typecast API keys. Configure these only on the backend using the repository-root `.env` file. The app currently uses manual recording because the previous repository did not include its Porcupine wake-word model files.
