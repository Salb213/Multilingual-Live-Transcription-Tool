from datetime import datetime
from pathlib import Path
import time

class AutoSaver:
    def __init__(self, root: Path, interval_minutes: int = 15):
        self.root = Path(root)
        self.interval = interval_minutes * 60
        self.session_dir = self._session_dir()
        self.idx = 1
        self._t0 = time.time()
        self._fh = None
        self._open_new()

    def _session_dir(self) -> Path:
        d = self.root / "calls" / datetime.now().strftime("%Y-%m-%d")
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _open_new(self):
        ts = datetime.now().strftime("%H-%M")
        p = self.session_dir / f"talk{self.idx}-{ts}.txt"
        self._fh = open(p, "w", encoding="utf-8")
        self._fh.write(f"# Session started {datetime.now().isoformat()}\n")
        self._fh.flush()

    def write(self, line: str):
        self._fh.write(line + "\n")
        self._fh.flush()
        if time.time() - self._t0 >= self.interval:
            self.rotate()

    def rotate(self):
        self._fh.close()
        self.idx += 1
        self._t0 = time.time()
        self._open_new()

    def close(self):
        if self._fh:
            self._fh.close()
            self._fh = None
