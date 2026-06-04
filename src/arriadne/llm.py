"""
LLM Provider Abstraction Layer

Supports: OpenAI, Anthropic, Ollama, any OpenAI-compatible API.
All providers exposed through a unified async interface.

Usage:
    from arriadne.llm import LLMProvider

    llm = LLMProvider.from_config({
        "provider": "openai",
        "model": "gpt-4o-mini",
        "api_key": "sk-...",
    })
    result = await llm.complete("Extract facts from this text: ...")
"""

from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger("arriadne.llm")


@dataclass
class LLMMessage:
    """A single message for LLM completion."""

    role: str  # "system", "user", "assistant"
    content: str

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class LLMResponse:
    """Response from an LLM completion."""

    content: str
    model: str
    usage: Dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0
    raw: Any = None

    @property
    def total_tokens(self) -> int:
        return self.usage.get("total_tokens", 0)

    def json(self) -> Any:
        """Parse response content as JSON."""
        text = self.content.strip()
        # Strip markdown code fences
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            text = "\n".join(lines)
        return json.loads(text)


class BaseLLMProvider(ABC):
    """Abstract base for LLM providers."""

    @abstractmethod
    async def complete(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None,
    ) -> LLMResponse:
        ...

    @abstractmethod
    def is_available(self) -> bool:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


class OpenAIProvider(BaseLLMProvider):
    """OpenAI and OpenAI-compatible providers (OpenRouter, Together, etc.)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4o-mini",
        organization: Optional[str] = None,
        timeout: float = 60.0,
    ):
        try:
            import openai
        except ImportError:
            raise ImportError("pip install openai")

        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._model = model
        self._timeout = timeout
        self._client = openai.AsyncOpenAI(
            api_key=self._api_key,
            base_url=base_url,
            organization=organization,
            timeout=timeout,
        )

    @property
    def name(self) -> str:
        return f"openai:{self._model}"

    def is_available(self) -> bool:
        return bool(self._api_key)

    async def complete(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None,
    ) -> LLMResponse:
        kwargs: Dict[str, Any] = {
            "model": self._model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format

        t0 = time.monotonic()
        resp = await self._client.chat.completions.create(**kwargs)
        latency = (time.monotonic() - t0) * 1000

        choice = resp.choices[0]
        usage = {}
        if resp.usage:
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
            }

        return LLMResponse(
            content=choice.message.content or "",
            model=resp.model,
            usage=usage,
            latency_ms=latency,
            raw=resp,
        )


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        timeout: float = 60.0,
    ):
        try:
            import anthropic
        except ImportError:
            raise ImportError("pip install anthropic")

        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._model = model
        self._client = anthropic.AsyncAnthropic(
            api_key=self._api_key,
            timeout=timeout,
        )

    @property
    def name(self) -> str:
        return f"anthropic:{self._model}"

    def is_available(self) -> bool:
        return bool(self._api_key)

    async def complete(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None,
    ) -> LLMResponse:
        # Extract system message (Anthropic handles it separately)
        system_text = ""
        user_messages = []
        for m in messages:
            if m.role == "system":
                system_text = m.content
            else:
                user_messages.append(m.to_dict())

        kwargs: Dict[str, Any] = {
            "model": self._model,
            "messages": user_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system_text:
            kwargs["system"] = system_text

        t0 = time.monotonic()
        resp = await self._client.messages.create(**kwargs)
        latency = (time.monotonic() - t0) * 1000

        content_text = ""
        for block in resp.content:
            if hasattr(block, "text"):
                content_text += block.text

        usage = {}
        if resp.usage:
            usage = {
                "prompt_tokens": resp.usage.input_tokens,
                "completion_tokens": resp.usage.output_tokens,
                "total_tokens": resp.usage.input_tokens + resp.usage.output_tokens,
            }

        return LLMResponse(
            content=content_text,
            model=resp.model,
            usage=usage,
            latency_ms=latency,
            raw=resp,
        )


class OllamaProvider(BaseLLMProvider):
    """Ollama local LLM server."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: str = "llama3.2",
        timeout: float = 120.0,
    ):
        self._base_url = (base_url or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self._model = model
        self._timeout = timeout

    @property
    def name(self) -> str:
        return f"ollama:{self._model}"

    def is_available(self) -> bool:
        try:
            import httpx
            resp = httpx.get(f"{self._base_url}/api/tags", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False

    async def complete(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None,
    ) -> LLMResponse:
        import httpx

        payload: Dict[str, Any] = {
            "model": self._model,
            "messages": [m.to_dict() for m in messages],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if response_format and response_format.get("type") == "json_object":
            payload["format"] = "json"

        t0 = time.monotonic()
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(f"{self._base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
        latency = (time.monotonic() - t0) * 1000

        content = data.get("message", {}).get("content", "")
        usage = {}
        if "prompt_eval_count" in data:
            usage = {
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
            }

        return LLMResponse(
            content=content,
            model=self._model,
            usage=usage,
            latency_ms=latency,
            raw=data,
        )


class LLMProvider:
    """
    Unified LLM interface. Auto-detects provider from config or environment.

    Usage:
        llm = LLMProvider.from_config({
            "provider": "openai",  # or "anthropic", "ollama", "openrouter"
            "model": "gpt-4o-mini",
            "api_key": "sk-...",
        })

        # Or auto-detect from environment:
        llm = LLMProvider.auto_detect()

        result = await llm.complete([
            LLMMessage("system", "You are a memory extraction engine."),
            LLMMessage("user", "Extract facts from: I love Paris."),
        ])
    """

    def __init__(self, provider: BaseLLMProvider):
        self._provider = provider
        self._call_count = 0
        self._total_tokens = 0
        self._total_latency_ms = 0.0

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "LLMProvider":
        """Create from a configuration dict."""
        provider_name = config.get("provider", "openai").lower()
        model = config.get("model", "")
        api_key = config.get("api_key", "")
        base_url = config.get("base_url", "")

        if provider_name == "anthropic":
            inner = AnthropicProvider(
                api_key=api_key,
                model=model or "claude-sonnet-4-20250514",
                timeout=config.get("timeout", 60.0),
            )
        elif provider_name == "ollama":
            inner = OllamaProvider(
                base_url=base_url or None,
                model=model or "llama3.2",
                timeout=config.get("timeout", 120.0),
            )
        else:
            # OpenAI, OpenRouter, Together, etc.
            url = base_url or None
            if provider_name == "openrouter" and not url:
                url = "https://openrouter.ai/api/v1"
            elif provider_name == "together" and not url:
                url = "https://api.together.xyz/v1"
            elif provider_name == "lmstudio" and not url:
                url = "http://localhost:1234/v1"

            inner = OpenAIProvider(
                api_key=api_key,
                base_url=url,
                model=model or "gpt-4o-mini",
                timeout=config.get("timeout", 60.0),
            )

        return cls(inner)

    @classmethod
    def auto_detect(cls) -> "LLMProvider":
        """Auto-detect available provider from environment."""
        # Check OpenAI first
        if os.environ.get("OPENAI_API_KEY"):
            model = os.environ.get("ARIDADNE_LLM_MODEL", "gpt-4o-mini")
            return cls.from_config({"provider": "openai", "model": model})

        # Check Anthropic
        if os.environ.get("ANTHROPIC_API_KEY"):
            model = os.environ.get("ARIDADNE_LLM_MODEL", "claude-sonnet-4-20250514")
            return cls.from_config({"provider": "anthropic", "model": model})

        # Check Ollama
        ollama = OllamaProvider()
        if ollama.is_available():
            return cls(ollama)

        raise ValueError(
            "No LLM provider available. Set one of:\n"
            "  - OPENAI_API_KEY environment variable\n"
            "  - ANTHROPIC_API_KEY environment variable\n"
            "  - Ollama running on localhost:11434\n"
            "  - Pass provider config to LLMProvider.from_config()"
        )

    @property
    def name(self) -> str:
        return self._provider.name

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "provider": self.name,
            "calls": self._call_count,
            "total_tokens": self._total_tokens,
            "total_latency_ms": round(self._total_latency_ms, 1),
            "avg_latency_ms": round(
                self._total_latency_ms / max(1, self._call_count), 1
            ),
        }

    def is_available(self) -> bool:
        return self._provider.is_available()

    async def complete(
        self,
        messages: Union[List[LLMMessage], List[Dict[str, str]]],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None,
    ) -> LLMResponse:
        """Complete a chat conversation."""
        # Normalize input
        normalized = []
        for m in messages:
            if isinstance(m, dict):
                normalized.append(LLMMessage(m["role"], m["content"]))
            else:
                normalized.append(m)

        resp = await self._provider.complete(
            normalized,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )

        self._call_count += 1
        self._total_tokens += resp.total_tokens
        self._total_latency_ms += resp.latency_ms

        return resp

    def complete_sync(
        self,
        messages: Union[List[LLMMessage], List[Dict[str, str]]],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None,
    ) -> LLMResponse:
        """Synchronous wrapper for complete()."""
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Already in an async context, use a new thread
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    self.complete(messages, temperature, max_tokens, response_format),
                )
                return future.result(timeout=120)
        else:
            return asyncio.run(
                self.complete(messages, temperature, max_tokens, response_format)
            )
