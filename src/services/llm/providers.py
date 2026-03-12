import logging
from collections.abc import Iterator

from ollama import Client

from src.core.config import get_settings
from src.services.llm.base import LLMClient

logger = logging.getLogger(__name__)


class MockLLMClient(LLMClient):
    def generate(self, prompt: str) -> str:
        return (
            "Explanation: This clause sets contractual terms that may create obligations.\n"
            "User obligation: Review deadlines, payment terms, and compliance duties in the clause.\n"
            "Risk factors: Check for broad liability, high penalties, or auto-renew conditions.\n"
            "Related Indian law citation: Validate against Indian Contract Act, 1872 and dispute forums."
        )

    def stream_generate(self, prompt: str) -> Iterator[str]:
        text = self.generate(prompt)
        for token in text.split(" "):
            yield token + " "


class OllamaLLMClient(LLMClient):
    def __init__(self) -> None:
        settings = get_settings()
        self.model = settings.ollama_model
        self.temperature = settings.ollama_temperature
        headers = {}
        if settings.ollama_api_key:
            headers["Authorization"] = f"Bearer {settings.ollama_api_key}"
        self.client = Client(
            host=settings.ollama_base_url,
            headers=headers or None,
            timeout=settings.ollama_timeout_seconds,
        )

    def generate(self, prompt: str) -> str:
        response = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            options={"temperature": self.temperature},
        )
        return str(response.get("message", {}).get("content", ""))

    def stream_generate(self, prompt: str) -> Iterator[str]:
        stream = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            options={"temperature": self.temperature},
        )
        for part in stream:
            content = str(part.get("message", {}).get("content", ""))
            if not content:
                continue
            yield content


def get_llm_client() -> LLMClient:
    settings = get_settings()
    provider = settings.llm_provider.lower().strip()

    if provider == "ollama":
        return OllamaLLMClient()

    logger.info("Using mock LLM provider")
    return MockLLMClient()
