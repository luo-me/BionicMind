from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class LLMResponse:
    content: str
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = ""
    metadata: dict[str, Any] | None = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMProvider(ABC):
    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        ...

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        ...

    @abstractmethod
    def get_model_name(self) -> str:
        ...

    @abstractmethod
    def is_available(self) -> bool:
        ...
