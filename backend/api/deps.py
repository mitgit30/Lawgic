from functools import lru_cache

from src.services.legal_assistant_service import LegalAssistantService


@lru_cache(maxsize=1)
def get_legal_assistant_service() -> LegalAssistantService:
    return LegalAssistantService()
