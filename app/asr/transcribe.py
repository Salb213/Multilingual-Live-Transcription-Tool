from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from faster_whisper import WhisperModel

@dataclass
class ASRConfig:
    model_name: str = "medium"
    device: str = "cpu"
    compute_type: str = "int8"
    beam_size: int = 3
    vad_filter: bool = True
    intra_threads: int = 2
    inter_threads: int = 1

class ASREngine:
    def __init__(self, cfg: ASRConfig | None = None):
        self.cfg = cfg or ASRConfig()
        self.model = WhisperModel(
            self.cfg.model_name,
            device=self.cfg.device,
            compute_type=self.cfg.compute_type,
            intra_threads=self.cfg.intra_threads,
            inter_threads=self.cfg.inter_threads,
        )

    def transcribe(
        self,
        audio: np.ndarray,
        sr: int,
        *,
        forced_lang: str | None = "de",
        return_segments: bool = False,
    ) -> Tuple[str, str, float] | Tuple[str, str, float, List[Any]]:
        if sr != 16000:
            raise ValueError("Whisper expects 16 kHz mono audio (got %s Hz)" % sr)
        segments, info = self.model.transcribe(
            audio,
            language=forced_lang,
            beam_size=self.cfg.beam_size,
            vad_filter=self.cfg.vad_filter,
            vad_parameters=dict(min_silence_duration_ms=200),
        )
        segments = list(segments) if segments else []
        text = "".join(s.text for s in segments).strip()
        lang = (forced_lang or info.language or "").lower()
        prob = 0.0 if forced_lang else float(info.language_probability or 0.0)
        if return_segments:
            return text, lang, prob, segments
        return text, lang, prob

    def detect_language(self, audio: np.ndarray, sr: int) -> Tuple[str, float]:
        if sr != 16000:
            raise ValueError("Whisper expects 16 kHz mono audio")
        _, info = self.model.transcribe(audio, task="lang_id", vad_filter=False)
        return (info.language or "").lower(), float(info.language_probability or 0.0)
