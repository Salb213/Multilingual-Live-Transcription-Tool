# Design

- Dual-stream capture: mic + WASAPI loopback
- VAD ? chunking ? faster-whisper
- Language detection per chunk
- Translate PL/SK ? DE (EN/DE left as-is)
- Autosave every 15 min to /calls/YYYY-MM-DD/talkN-HH-MM.txt
- GUI: Start/Pause/Stop, device selector, autosave countdown, translation checkbox
- Later: pyannote diarization + overlap, VB-CABLE per-app routing, auto-stop on window close
