from abc import ABC, abstractmethod
from typing import Callable
import numpy as np


class STTProvider(ABC):
    def __init__(self, on_partial: Callable[[str], None], on_final: Callable[[str], None]):
        self.on_partial = on_partial
        self.on_final = on_final

    @abstractmethod
    def feed_audio(self, audio: np.ndarray):
        """Feed audio chunk to the STT engine."""

    @abstractmethod
    def finalize(self):
        """Signal end of speech, flush any remaining audio."""

    @abstractmethod
    def reset(self):
        """Reset state for a new utterance."""
