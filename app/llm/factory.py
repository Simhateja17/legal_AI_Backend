from app.core.config import get_settings
from app.core.exceptions import LLMProviderNotFoundError
from app.llm.base import BaseLLMProvider


def create_llm_provider() -> BaseLLMProvider:
    """Instantiate the configured LLM provider."""
    settings = get_settings()
    provider = settings.llm_provider

    if provider == "azure":
        from app.llm.azure_openai import AzureOpenAIProvider
        return AzureOpenAIProvider()

    if provider == "gemini":
        from app.llm.gemini import GeminiProvider
        return GeminiProvider(model_name="gemini-3-flash-preview")

    raise LLMProviderNotFoundError(provider)
