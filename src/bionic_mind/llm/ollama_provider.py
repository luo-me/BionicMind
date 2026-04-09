from __future__ import annotations

from typing import Any

from loguru import logger

from bionic_mind.llm.base import LLMProvider, LLMResponse


class OllamaProvider(LLMProvider):
    def __init__(
        self,
        model: str = "qwen2.5:7b",
        base_url: str = "http://localhost:11434",
        embedding_model: str = "nomic-embed-text",
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.embedding_model = embedding_model
        self._session = None

    async def _get_session(self):
        if self._session is None:
            import aiohttp

            self._session = aiohttp.ClientSession()
        return self._session

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        session = await self._get_session()
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        try:
            async with session.post(f"{self.base_url}/api/chat", json=payload) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"Ollama API error {resp.status}: {text}")
                data = await resp.json()
                return LLMResponse(
                    content=data.get("message", {}).get("content", ""),
                    model=data.get("model", self.model),
                    prompt_tokens=data.get("prompt_eval_count", 0),
                    completion_tokens=data.get("eval_count", 0),
                    finish_reason="stop" if data.get("done") else "",
                )
        except Exception as e:
            logger.error(f"Ollama chat failed: {e}")
            raise

    async def embed(self, text: str) -> list[float]:
        session = await self._get_session()
        payload = {
            "model": self.embedding_model,
            "prompt": text,
        }
        try:
            async with session.post(f"{self.base_url}/api/embeddings", json=payload) as resp:
                if resp.status != 200:
                    text_resp = await resp.text()
                    raise RuntimeError(f"Ollama embed error {resp.status}: {text_resp}")
                data = await resp.json()
                return data.get("embedding", [])
        except Exception as e:
            logger.error(f"Ollama embed failed: {e}")
            raise

    async def close(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    def get_model_name(self) -> str:
        return self.model

    def is_available(self) -> bool:
        return True
