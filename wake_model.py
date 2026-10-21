"""
wake_model.py

Local "hey jarvis" detector (openWakeWord). While it is armed, Whisper stays
idle. If the package or the model file is missing, detection is disabled and
the assistant keeps the old transcript filter.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

FRAME = 1280  # 80 ms at 16 kHz, what openWakeWord scores one step at a time
THRESHOLD = 0.5

Predict = Callable[[np.ndarray], dict]


class HeyJarvisDetector:
    def __init__(self, predict: Optional[Predict] = None, reset: Optional[Callable[[], None]] = None) -> None:
        self._predict = predict
        self._reset = reset
        self._buf = np.zeros(0, dtype=np.int16)
        self.available = predict is not None

    def reset(self) -> None:
        self._buf = np.zeros(0, dtype=np.int16)
        if self._reset is not None:
            self._reset()

    def consume(self, pcm: bytes) -> bool:
        """True when this audio contains the wake phrase."""
        if not self.available or not pcm:
            return False
        samples = np.frombuffer(pcm, dtype=np.int16)
        self._buf = np.concatenate([self._buf, samples])
        hit = False
        while len(self._buf) >= FRAME:
            frame = self._buf[:FRAME]
            self._buf = self._buf[FRAME:]
            scores = self._predict(frame) or {}
            if isinstance(scores, dict):
                value = max(scores.values(), default=0)
            else:
                value = scores
            if value >= THRESHOLD:
                hit = True
        return hit


def load() -> HeyJarvisDetector:
    try:
        import openwakeword
        from openwakeword.model import Model

        openwakeword.utils.download_models()
        model = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")
    except Exception as exc:  # noqa: BLE001 - missing model must not stop the app
        print(f"[wake] hey jarvis model unavailable ({exc}); using the transcript filter")
        return HeyJarvisDetector()
    print("[wake] hey jarvis model ready")
    reset = model.reset if hasattr(model, "reset") else None
    return HeyJarvisDetector(predict=model.predict, reset=reset)
