from abc import ABC, abstractmethod
from typing import Callable


class LLMProvider(ABC):
    @abstractmethod
    def polish(self, text: str, context_before: str = "", context_after: str = "") -> str:
        """Polish transcribed text. context_before/after are overlap segments."""
