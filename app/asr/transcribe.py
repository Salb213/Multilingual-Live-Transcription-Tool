from dataclasses import dataclass
import numpy as np
from faster_whisper import WhisperModel

@dataclass
class ASRConfig:
    model_name: str = "base"
    device: str = "cpu"
    compute_type: str = "int8"
    vad_filter: bool = True
    beam_size: int = 1

class ASREngine:
    def __init__(self, cfg: ASRConfig):
        self.cfg = cfg
        self.model = WhisperModel(
            cfg.model_name,
            device=cfg.device,
            compute_type=cfg.compute_type,
        )

    def transcribe(self, audio: np.ndarray, sr: int):
        segs, info = self.model.transcribe(
            audio,
            beam_size=self.cfg.beam_size,
            vad_filter=self.cfg.vad_filter,
            vad_parameters=dict(min_silence_duration_ms=200),
        )
        text = "".join(s.text for s in segs).strip()
        lang = (info.language or "").lower()
        prob = float(info.language_probability or 0.0)
        return text, lang, prob
