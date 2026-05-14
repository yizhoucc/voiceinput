from abc import ABC, abstractmethod
from typing import Callable
import numpy as np


class STTProvider(ABC):
    def __init__(self, on_partial: Callable[[str], None], on_final: Callable[[str], None],
                 on_commit: Callable[[str], None] | None = None):
        self.on_partial = on_partial
        self.on_final = on_final
        self.on_commit = on_commit or (lambda _: None)

    def warmup(self):
        """Load model / warm up. Override in subclasses."""

    @abstractmethod
    def feed_audio(self, audio: np.ndarray):
        pass

    @abstractmethod
    def finalize(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    def prepare_finalize(self):
        """Reset commit cursor so finalize processes all audio. Override if needed."""
