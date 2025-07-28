import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly
from math import gcd

TARGET_SR = 16000

def device_default_sr(index: int) -> int:
    d = sd.query_devices(index)
    return int(d.get("default_samplerate", 48000) or 48000)

def resample_to_16k(x: np.ndarray, src_sr: int) -> np.ndarray:
    if src_sr == TARGET_SR:
        return x.astype(np.float32, copy=False)
    g = gcd(src_sr, TARGET_SR)
    up = TARGET_SR // g
    down = src_sr // g
    y = resample_poly(x, up, down, axis=0)
    return y.astype(np.float32, copy=False)

def record_block(dev_index: int, seconds: float) -> np.ndarray:
    sr = device_default_sr(dev_index)
    frames = int(seconds * sr)
    data = sd.rec(frames, samplerate=sr, channels=1, dtype="float32", device=dev_index)
    sd.wait()
    return resample_to_16k(data, sr)
