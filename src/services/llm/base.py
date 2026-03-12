from abc import ABC, abstractmethod
from collections.abc import Iterator


class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def stream_generate(self, prompt: str) -> Iterator[str]:
        raise NotImplementedError
