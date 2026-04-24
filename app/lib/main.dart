import 'dart:io';
import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:porcupine_flutter/porcupine_manager.dart';
import 'package:flutter_sound/flutter_sound.dart';
import 'package:path_provider/path_provider.dart';
import 'package:http/http.dart' as http;
import 'package:permission_handler/permission_handler.dart';
import 'package:lottie/lottie.dart';
import 'package:audioplayers/audioplayers.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'AI Voice Assistant',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: VoiceAssistant(),
      debugShowCheckedModeBanner: false,
    );
  }
}

class VoiceAssistant extends StatefulWidget {
  @override
  _VoiceAssistantState createState() => _VoiceAssistantState();
}

enum AssistantState { waiting, listening, processing, answering }

class _VoiceAssistantState extends State<VoiceAssistant> {
  AssistantState _currentState = AssistantState.waiting;
  late AnimationController _controller;
  late PorcupineManager _porcupineManager;
  FlutterSoundRecorder? _recorder;
  String predictionResult = "";
  String voiceTone = "";
  String detectedText = " ";
  String generatedResponse = " ";
  String? _filePath;
  bool _isRecording = false;
  
  final String fastApiUrl = 'http://192.168.219.101:8001/predict-audio';
  final String apiKey = String.fromEnvironment('OPENAI_API_KEY');
  final String openAiGptUrl = 'https://api.openai.com/v1/chat/completions';
  final String url = "https://typecast.ai/api/speak";

  AudioPlayer audioPlayer = AudioPlayer();

  @override
  void initState() {
    super.initState();
    _initializeRecorder(); 
    _initializePorcupine();
    checkLocalNetworkAccess('192.168.0.3', 8001);
  }

  void checkLocalNetworkAccess(String ipAddress, int port) async {
    try {
      final socket = await Socket.connect(ipAddress, port, timeout: Duration(seconds: 5));
      print('Successfully connected to the local network');
      socket.destroy();
    } catch (e) {
      print('Failed to connect to the local network: $e');
      showDialog(
        context: context,
        builder: (context) => AlertDialog(
          title: Text("Network Error"),
          content: Text("Failed to connect to the server. Please check your network."),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context),
              child: Text("OK"),
            ),
          ],
        ),
      );
    }
  }

  void _initializePorcupine() async {
    _porcupineManager = await PorcupineManager.fromKeywordPaths(
      'RDB3NAY14hKp9XO5sT4UvbAduZSpB+tufB6AXnie6bm62kNlPmYsyQ==',
      ["assets/wake_word.ppn"],
      wakeWordCallback,
      modelPath: "assets/porcupine_params_ko.pv");
    _porcupineManager.start();
  }
  
  Future<void> _initializeRecorder() async {
    _recorder = FlutterSoundRecorder();
    await _recorder!.openRecorder();
    await _requestMicrophonePermission();
  }

  void wakeWordCallback(int keywordIndex) async {
      if (keywordIndex == 0) {
        // 웨이크 워드가 감지되었을 때 알림 소리를 재생합니다.
        await audioPlayer.setSourceAsset('notification_sound.mp3');
        await audioPlayer.resume();

        // 상태를 listening으로 전환하고 녹음을 시작합니다.
        setState(() => _currentState = AssistantState.listening);
        _startRecording();
    }
  }

  Future<void> _startRecording() async {
    if (_isRecording) return;

    Directory tempDir = await getTemporaryDirectory();
    _filePath = "${tempDir.path}/recording.wav";
    await _recorder!.startRecorder(toFile: _filePath, codec: Codec.pcm16WAV);
    setState(() => _isRecording = true);

    Future.delayed(Duration(seconds: 3), () {
      if (_isRecording) _stopRecording();
    });
  }

  Future<void> _stopRecording() async {
    if (!_isRecording) return;

    await _recorder!.stopRecorder();
    setState(() => _isRecording = false);
    setState(() => _currentState = AssistantState.processing);

    if (_filePath != null) {
      await _transcribeAudioWithOpenAI(_filePath!);
      await _sendFileToFastAPI(_filePath!);
      await _sendMessageToGPT(detectedText);
    }
    setState(() => _currentState = AssistantState.answering);
  }

  Future<void> _requestMicrophonePermission() async {
    var status = await Permission.microphone.status;
    if (!status.isGranted) await Permission.microphone.request();
  }

  Future<void> _transcribeAudioWithOpenAI(String filePath) async {
    try {
      var uri = Uri.parse('https://api.openai.com/v1/audio/transcriptions');
      var request = http.MultipartRequest('POST', uri)
        ..headers['Authorization'] = 'Bearer $openAiApiKey'
        ..files.add(await http.MultipartFile.fromPath('file', filePath))
        ..fields['model'] = 'whisper-1'
        ..fields['language'] = 'ko';

      var response = await request.send();
      if (response.statusCode == 200) {
        var responseData = await response.stream.bytesToString();
        var jsonData = jsonDecode(responseData);
        setState(() => detectedText = jsonData['text']);
      }
    } catch (e) {
      print('Error transcribing audio with Whisper API: $e');
    }
  }

  Future<void> _sendFileToFastAPI(String filePath) async {
    try {
      var uri = Uri.parse(fastApiUrl);
      var request = http.MultipartRequest('POST', uri)
        ..files.add(await http.MultipartFile.fromPath('file', filePath));

      var response = await request.send();
      if (response.statusCode == 200) {
        predictionResult = await response.stream.bytesToString();
        _adjustVolumeBasedOnPrediction(predictionResult);
      }
    } catch (e) {
      print('Error sending file to FastAPI: $e');
    }
  }

  void _adjustVolumeBasedOnPrediction(String prediction) {
    print(prediction);
    if (prediction.contains('loud')) {
      voiceTone = "calm";
      audioPlayer.setVolume(1.0);
    }
    else if (prediction.contains('small')){
        voiceTone = "whisper";
        audioPlayer.setVolume(0.1);
    } 
    else {
      audioPlayer.setVolume(0.5);
      voiceTone = "calm";
    }
  }

  Future<void> _sendMessageToGPT(String message) async {
    var uri = Uri.parse(openAiGptUrl);
    var response = await http.post(
      uri,
      headers: {'Authorization': 'Bearer $openAiApiKey', 'Content-Type': 'application/json'},
      body: jsonEncode({"model": "gpt-4", "messages": [{"role": "user", "content": message}]}),
    );

    if (response.statusCode == 200) {
      var responseData = jsonDecode(utf8.decode(response.bodyBytes));
      generatedResponse = responseData['choices'][0]['message']['content'];
      await _speak(generatedResponse);
    }
  }
  
  Future<void> _speak(String text) async {
  // Typecast API URL 및 헤더 설정
  final url = Uri.parse("https://typecast.ai/api/speak");
  final headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer __plt4PtZvzGLAW9kyoFn7YQSUsYJPpPQkWmqwb4eZoTy', // Typecast API 토큰 입력
  };
  print(voiceTone);
  // 요청 페이로드 생성
  final payload = jsonEncode({
    "actor_id": "622964d6255364be41659078", // 적절한 actor_id로 교체
    "text": text, // 음성으로 변환할 텍스트
    "lang": "auto",
    "tempo": 1,
    "volume": 100,
    "pitch": 0,
    "xapi_hd": true,
    "max_seconds": 60,
    "model_version": "latest",
    "xapi_audio_format": "wav",
    "emotion_tone_preset": "emotion-prompt",
    "emotion_prompt": voiceTone
  });

  try {
    // Typecast API 요청 보내기
    var response = await http.post(url, headers: headers, body: payload);

    // 성공적으로 응답을 받은 경우
    if (response.statusCode == 200) {
      final speakUrl = jsonDecode(response.body)['result']['speak_v2_url'];

      for (int i=0; i<120; i++){
        final statusResponse = await http.get(Uri.parse(speakUrl), headers: headers);
        final result = jsonDecode(statusResponse.body)['result'];

        if (result['status'] == 'done') {
        // 4. 오디오 파일 다운로드
        final audioResponse = await http.get(Uri.parse(result['audio_download_url']));
        final filePath = await _saveAudioToTempFile(audioResponse.bodyBytes);

        // 5. 오디오 파일 재생
        await audioPlayer.setSourceDeviceFile(filePath);
        await audioPlayer.resume();
        break;
      } else {
        print("status: ${result['status']}, waiting 1 second");
        await Future.delayed(Duration(seconds: 1));
      }
      }
    } else {
      print("Typecast API 음성재생 실패: ${response.statusCode}, ${response.reasonPhrase}");
    }
  } catch (e) {
    print("Typecast API 호출 중 오류 발생: $e");
  }
}

  Future<String> _saveAudioToTempFile(Uint8List audioBytes) async {
  final dir = await getTemporaryDirectory();
  final filePath = '${dir.path}/out.wav';
  final file = File(filePath);
  await file.writeAsBytes(audioBytes);
  return filePath;
}

  @override
  void dispose() {
    _controller.dispose();
    _porcupineManager.stop();
    _recorder!.closeRecorder();
    super.dispose();
  }

@override
Widget build(BuildContext context) {
  return Scaffold(
    body: Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          // 상황에 맞는 UI 상태에 따라 출력
          if (_currentState == AssistantState.waiting) ...[
            Text('두부를 불러주세요.'),
            Image.asset('assets/stillWave.png'),
          ] else if (_currentState == AssistantState.listening) ...[
            Text('듣고 있습니다...'),
            Lottie.asset('assets/movingWave.json'),
          ] else if (_currentState == AssistantState.processing) ...[
            Text("$detectedText"),
            Image.asset('assets/stillWave.png'),
            Text('답변을 생성 중입니다...'),
          ] else if (_currentState == AssistantState.answering) ...[
            Text("$detectedText"),
            Image.asset('assets/stillWave.png'),
            Text("$generatedResponse"),
          ],
        ],
      ),
    ),
  );
}
}