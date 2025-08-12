from __future__ import annotations

from typing import Optional
try:
    import simpleaudio as sa
except Exception:
    sa = None


class AudioCuePlayer:
    def __init__(self, enable_audio: bool = False):
        self.enable_audio = enable_audio and sa is not None

    def play_ping(self):
        if not self.enable_audio:
            return
        # 440 Hz tone for 120 ms
        import numpy as np
        sample_rate = 44100
        t = np.linspace(0, 0.12, int(0.12 * sample_rate), False)
        tone = np.sin(2 * np.pi * 440 * t) * 0.2
        audio = (tone * (2**15 - 1)).astype(np.int16)
        sa.play_buffer(audio, 1, 2, sample_rate)

    def play_feedback(self, _text: str):
        # Placeholder: just ping for now
        self.play_ping()