from functools import lru_cache
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from src.core.config import settings

def create_llm(model_id: str, temperature: float, max_token:int, provider:str = settings.model_provider) :
    if provider == "litellm" :
        return ChatOpenAI(
            model=model_id,
            temperature=temperature,
            base_url=settings.litellm_base_url,
            api_key=settings.litellm_api_key
        )

@lru_cache(maxsize=100)
def get_llm(
        model_id: str,
        temperature: float,
        max_tokens: int = 15000
    ) -> BaseChatModel:
    return create_llm(model_id, temperature, max_tokens)