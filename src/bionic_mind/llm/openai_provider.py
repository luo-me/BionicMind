from __future__ import annotations

import os
from typing import Any

from loguru import logger

from bionic_mind.llm.base import LLMProvider, LLMResponse


class OpenAIProvider(LLMProvider):
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        embedding_model: str = "text-embedding-3-small",
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url
        self.embedding_model = embedding_model
        self._client = None
        self._async_client = None

    def _get_async_client(self):
        if self._async_client is None:
            try:
                from openai import AsyncOpenAI

                kwargs: dict[str, Any] = {"api_key": self.api_key}
                if self.base_url:
                    kwargs["base_url"] = self.base_url
                self._async_client = AsyncOpenAI(**kwargs)
            except ImportError:
                raise ImportError("openai package not installed. Run: pip install openai")
        return self._async_client

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI

                kwargs: dict[str, Any] = {"api_key": self.api_key}
                if self.base_url:
                    kwargs["base_url"] = self.base_url
                self._client = OpenAI(**kwargs)
            except ImportError:
                raise ImportError("openai package not installed. Run: pip install openai")
        return self._client

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        client = self._get_async_client()
        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            choice = response.choices[0]
            usage = response.usage
            return LLMResponse(
                content=choice.message.content or "",
                model=response.model,
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                finish_reason=choice.finish_reason or "",
            )
        except Exception as e:
            logger.error(f"OpenAI chat failed: {e}")
            raise

    async def embed(self, text: str) -> list[float]:
        client = self._get_client()
        try:
            response = client.embeddings.create(
                model=self.embedding_model,
                input=text,
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"OpenAI embed failed: {e}")
            raise

    def get_model_name(self) -> str:
        return self.model

    def is_available(self) -> bool:
        return bool(self.api_key)
