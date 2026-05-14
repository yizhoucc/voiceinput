from abc import ABC, abstractmethod


class LLMProvider(ABC):
    @abstractmethod
    def polish(self, text: str, context_before: str = "") -> str:
        """Polish transcribed text. context_before is the previous segment for overlap context."""
