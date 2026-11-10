import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:audioplayers/audioplayers.dart';
import 'package:flutter/material.dart';
import 'package:flutter_sound/flutter_sound.dart';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const DoobooApp());
}

class DoobooApp extends StatelessWidget {
  const DoobooApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: '두부',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF557B67)),
        useMaterial3: true,
      ),
      home: const VoiceAssistantPage(),
    );
  }
}

enum AssistantState { initializing, waiting, recording, processing, answering, error }

class VoiceAssistantPage extends StatefulWidget {
  const VoiceAssistantPage({super.key});

  @override
  State<VoiceAssistantPage> createState() => _VoiceAssistantPageState();
}

class _VoiceAssistantPageState extends State<VoiceAssistantPage> {
  static const String _apiBaseUrl = String.fromEnvironment(
    'DOOBOO_API_BASE_URL',
    defaultValue: 'http://10.0.2.2:8001',
  );
  static const Duration _requestTimeout = Duration(seconds: 120);

  final FlutterSoundRecorder _recorder = FlutterSoundRecorder();
  final AudioPlayer _player = AudioPlayer();

  AssistantState _state = AssistantState.initializing;
  bool _recorderOpened = false;
  String _transcript = '';
  String _answer = '';
  String _error = '';
  String? _recordingPath;
  Timer? _recordingLimitTimer;
  StreamSubscription<void>? _playerCompleteSubscription;
  String? _speechFilePath;

  bool _isFinalizingRecording = false;

  bool get _isBusy =>
      _state == AssistantState.initializing ||
      _state == AssistantState.processing ||
      _isFinalizingRecording;

  @override
  void initState() {
    super.initState();
    _playerCompleteSubscription = _player.onPlayerComplete.listen((_) {
      unawaited(_removeSpeechFile());
    });
    _initializeAudio();
  }

  Future<void> _initializeAudio() async {
    try {
      await _recorder.openRecorder();
      _recorderOpened = true;
      if (!mounted) {
        await _recorder.closeRecorder();
        _recorderOpened = false;
        return;
      }
      setState(() => _state = AssistantState.waiting);
    } catch (error) {
      _showError('녹음기를 초기화하지 못했습니다: $error');
    }
  }

  Uri _endpoint(String path) => Uri.parse(_apiBaseUrl).resolve(path);

  Map<String, String> get _jsonHeaders => const {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      };

  Future<Map<String, dynamic>> _decodeJsonResponse(http.Response response) async {
    Map<String, dynamic> body = {};
    try {
      final decoded = jsonDecode(utf8.decode(response.bodyBytes));
      if (decoded is Map<String, dynamic>) body = decoded;
    } on FormatException {
      // Fall through to status-based error.
    }
    if (response.statusCode < 200 || response.statusCode >= 300) {
      final detail = body['detail'];
      throw HttpException(
        detail is String ? detail : '서버 요청 실패 (${response.statusCode})',
      );
    }
    return body;
  }

  Future<Map<String, dynamic>> _uploadAudio(String path, String endpoint) async {
    final request = http.MultipartRequest('POST', _endpoint(endpoint))
      ..files.add(await http.MultipartFile.fromPath('file', path));
    final streamed = await request.send().timeout(_requestTimeout);
    final response = await http.Response.fromStream(streamed).timeout(_requestTimeout);
    return _decodeJsonResponse(response);
  }

  Future<void> _startRecording() async {
    if (!_recorderOpened || _state != AssistantState.waiting) return;
    try {
      var permission = await Permission.microphone.status;
      if (!permission.isGranted) permission = await Permission.microphone.request();
      if (!permission.isGranted) {
        throw Exception('마이크 권한이 허용되지 않았습니다.');
      }
      if (!mounted) return;

      final directory = await getTemporaryDirectory();
      final filename = 'dooboo_${DateTime.now().microsecondsSinceEpoch}.wav';
      _recordingPath = '${directory.path}/$filename';

      await _recorder.startRecorder(
        toFile: _recordingPath,
        codec: Codec.pcm16WAV,
        sampleRate: 22050,
        numChannels: 1,
      );

      if (!mounted) return;
      setState(() {
        _transcript = '';
        _answer = '';
        _error = '';
        _state = AssistantState.recording;
      });
      _recordingLimitTimer?.cancel();
      _recordingLimitTimer = Timer(const Duration(seconds: 60), () {
        if (mounted && _state == AssistantState.recording) {
          unawaited(_stopRecordingAndProcess());
        }
      });
    } catch (error) {
      try {
        if (_recorderOpened && await _recorder.isRecording) {
          await _recorder.stopRecorder();
        }
      } catch (_) {
        // Recorder may already have stopped after a plugin error.
      }
      final path = _recordingPath;
      _recordingPath = null;
      if (path != null) {
        try {
          final file = File(path);
          if (await file.exists()) await file.delete();
        } catch (_) {
          // Temporary file cleanup is best-effort.
        }
      }
      _showError('녹음을 시작하지 못했습니다: $error');
    }
  }

  Future<void> _removeSpeechFile() async {
    final path = _speechFilePath;
    _speechFilePath = null;
    if (path == null) return;
    try {
      final file = File(path);
      if (await file.exists()) await file.delete();
    } catch (_) {
      // Audio file cleanup is best-effort.
    }
  }

  Future<void> _stopRecordingAndProcess() async {
    if (_state != AssistantState.recording || _isFinalizingRecording) return;
    _isFinalizingRecording = true;
    String? completedAudioPath;
    _recordingLimitTimer?.cancel();
    try {
      final recordedPath = await _recorder.stopRecorder();
      final audioPath = recordedPath ?? _recordingPath;
      completedAudioPath = audioPath;
      if (audioPath == null || !await File(audioPath).exists()) {
        throw Exception('녹음 파일을 찾을 수 없습니다.');
      }

      if (mounted) setState(() => _state = AssistantState.processing);

      final results = await Future.wait([
        _uploadAudio(audioPath, '/transcribe-audio'),
        _uploadAudio(audioPath, '/predict-audio'),
      ]);

      final transcript = results[0]['text'];
      final prediction = results[1]['prediction'];
      if (transcript is! String || transcript.trim().isEmpty) {
        throw Exception('음성 인식 결과가 비어 있습니다.');
      }
      if (prediction is! String || prediction.trim().isEmpty) {
        throw Exception('어조 분류 결과가 비어 있습니다.');
      }

      if (mounted) setState(() => _transcript = transcript);

      final chatResponse = await http
          .post(
            _endpoint('/chat'),
            headers: _jsonHeaders,
            body: jsonEncode({'message': transcript, 'tone': prediction}),
          )
          .timeout(_requestTimeout);
      final chatData = await _decodeJsonResponse(chatResponse);
      final answer = chatData['text'];
      if (answer is! String || answer.trim().isEmpty) {
        throw Exception('두부의 답변이 비어 있습니다.');
      }
      if (mounted) setState(() => _answer = answer);

      final speechResponse = await http
          .post(
            _endpoint('/synthesize-speech'),
            headers: _jsonHeaders,
            body: jsonEncode({
              'text': answer,
              'tone': prediction.contains('small') ? 'whisper' : 'smart',
            }),
          )
          .timeout(_requestTimeout);
      if (speechResponse.statusCode < 200 || speechResponse.statusCode >= 300) {
        throw HttpException('음성 합성 요청 실패 (${speechResponse.statusCode})');
      }
      if (speechResponse.bodyBytes.isEmpty) {
        throw Exception('음성 합성 결과가 비어 있습니다.');
      }

      await _player.stop();
      await _removeSpeechFile();
      final directory = await getTemporaryDirectory();
      final outputPath =
          '${directory.path}/dooboo_reply_${DateTime.now().microsecondsSinceEpoch}.wav';
      final outputFile = File(outputPath);
      await outputFile.writeAsBytes(speechResponse.bodyBytes, flush: true);
      _speechFilePath = outputPath;
      await _player.play(DeviceFileSource(outputPath));

      if (mounted) setState(() => _state = AssistantState.answering);
    } catch (error) {
      _showError('처리 중 오류가 발생했습니다: $error');
    } finally {
      if (completedAudioPath != null) {
        try {
          final file = File(completedAudioPath);
          if (await file.exists()) await file.delete();
        } catch (_) {
          // Temporary files are best-effort cleanup only.
        }
      }
      _recordingPath = null;
      _isFinalizingRecording = false;
      if (mounted && _state == AssistantState.processing) {
        setState(() => _state = AssistantState.waiting);
      }
    }
  }

  void _showError(String message) {
    if (!mounted) return;
    setState(() {
      _error = message;
      _state = AssistantState.error;
    });
  }

  void _reset() {
    if (!mounted) return;
    unawaited(_player.stop());
    unawaited(_removeSpeechFile());
    setState(() {
      _state = AssistantState.waiting;
      _error = '';
    });
  }

  @override
  void dispose() {
    _recordingLimitTimer?.cancel();
    unawaited(_playerCompleteSubscription?.cancel());
    unawaited(_removeSpeechFile());
    unawaited(_player.dispose());
    if (_recorderOpened) unawaited(_recorder.closeRecorder());
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final statusText = switch (_state) {
      AssistantState.initializing => '초기화 중입니다...',
      AssistantState.waiting => '두부에게 말을 걸어보세요.',
      AssistantState.recording => '듣고 있습니다...',
      AssistantState.processing => '음성을 분석하고 답변을 만들고 있습니다...',
      AssistantState.answering => '답변을 완료했습니다.',
      AssistantState.error => '오류가 발생했습니다.',
    };

    return Scaffold(
      appBar: AppBar(title: const Text('두부 음성비서')),
      body: SafeArea(
        child: Center(
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 560),
            child: Padding(
              padding: const EdgeInsets.all(24),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  Icon(
                    _state == AssistantState.recording
                        ? Icons.graphic_eq
                        : Icons.record_voice_over,
                    size: 72,
                    color: Theme.of(context).colorScheme.primary,
                  ),
                  const SizedBox(height: 20),
                  Text(
                    statusText,
                    textAlign: TextAlign.center,
                    style: Theme.of(context).textTheme.titleLarge,
                  ),
                  const SizedBox(height: 24),
                  if (_state == AssistantState.processing)
                    const Center(child: CircularProgressIndicator()),
                  if (_transcript.isNotEmpty) ...[
                    const Text('인식한 내용', style: TextStyle(fontWeight: FontWeight.bold)),
                    const SizedBox(height: 6),
                    Text(_transcript),
                    const SizedBox(height: 16),
                  ],
                  if (_answer.isNotEmpty) ...[
                    const Text('두부의 답변', style: TextStyle(fontWeight: FontWeight.bold)),
                    const SizedBox(height: 6),
                    Text(_answer),
                    const SizedBox(height: 16),
                  ],
                  if (_error.isNotEmpty) ...[
                    Text(
                      _error,
                      style: TextStyle(color: Theme.of(context).colorScheme.error),
                    ),
                    const SizedBox(height: 16),
                  ],
                  FilledButton.icon(
                    onPressed: _isBusy
                        ? null
                        : _state == AssistantState.recording
                            ? _stopRecordingAndProcess
                            : _state == AssistantState.error
                                ? _reset
                                : _startRecording,
                    icon: Icon(
                      _state == AssistantState.recording ? Icons.stop : Icons.mic,
                    ),
                    label: Text(
                      _state == AssistantState.recording
                          ? '녹음 종료 및 분석'
                          : _state == AssistantState.error
                              ? '다시 시도'
                              : '녹음 시작',
                    ),
                  ),
                  if (_state == AssistantState.answering) ...[
                    const SizedBox(height: 8),
                    OutlinedButton(
                      onPressed: _reset,
                      child: const Text('새 대화'),
                    ),
                  ],
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}
