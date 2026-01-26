import threading
import time

import jax


class PeakMemSampler:
    def __init__(self, sample_every_s: float = 0.005, device: jax.Device | None = None) -> None:
        self.sample_every_s = sample_every_s
        self.device = device or jax.devices()[0]
        self._stop = threading.Event()
        self.peak = 0

    def _sample(self) -> None:
        while not self._stop.is_set():
            stats = self.device.memory_stats()
            cur = 0
            if stats is not None:
                cur = stats.get("bytes_in_use", 0) or stats.get("device_memory_in_use", 0) or 0
            if cur > self.peak:
                self.peak = int(cur)
            time.sleep(self.sample_every_s)

    def __enter__(self) -> "PeakMemSampler":
        self.t = threading.Thread(target=self._sample, daemon=True)
        self.t.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        self.t.join()
