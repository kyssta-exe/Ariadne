"""
LLM Provider Abstraction Layer

Supports: OpenAI, Anthropic, Ollama, Gemini, Cohere, DeepSeek, Groq, Mistral, xAI,
any OpenAI-compatible API.
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


class GoogleGeminiProvider(BaseLLMProvider):
    """Google Gemini API via google-generativeai SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.0-flash",
        timeout: float = 60.0,
    ):
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("pip install google-generativeai")

        self._api_key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY", "")
        self._model = model
        self._timeout = timeout
        genai.configure(api_key=self._api_key)
        self._client = genai.GenerativeModel(model)

    @property
    def name(self) -> str:
        return f"gemini:{self._model}"

    def is_available(self) -> bool:
        return bool(self._api_key)

    async def complete(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None,
    ) -> LLMResponse:
        import asyncio
        import google.generativeai as genai

        # Build content list for Gemini
        contents = []
        system_instruction = None
        for m in messages:
            if m.role == "system":
                system_instruction = m.content
            else:
                role = "model" if m.role == "assistant" else "user"
                contents.append({"role": role, "parts": [m.content]})

        gen_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        t0 = time.monotonic()
        # Use synchronous generate_content in a thread to avoid blocking
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            None,
            lambda: self._client.generate_content(
                contents,
                generation_config=gen_config,
                system_instruction=system_instruction,
            ),
        )
        latency = (time.monotonic() - t0) * 1000

        content_text = resp.text if resp.text else ""
        usage = {}
        if hasattr(resp, "usage_metadata") and resp.usage_metadata:
            um = resp.usage_metadata
            usage = {
                "prompt_tokens": getattr(um, "prompt_token_count", 0),
                "completion_tokens": getattr(um, "candidates_token_count", 0),
                "total_tokens": getattr(um, "total_token_count", 0),
            }

        return LLMResponse(
            content=content_text,
            model=self._model,
            usage=usage,
            latency_ms=latency,
            raw=resp,
        )


class CohereProvider(BaseLLMProvider):
    """Cohere API via cohere SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "command-r-plus",
        timeout: float = 60.0,
    ):
        try:
            import cohere
        except ImportError:
            raise ImportError("pip install cohere")

        self._api_key = api_key or os.environ.get("COHERE_API_KEY", "")
        self._model = model
        self._timeout = timeout
        self._client = cohere.ClientV2(api_key=self._api_key)

    @property
    def name(self) -> str:
        return f"cohere:{self._model}"

    def is_available(self) -> bool:
        return bool(self._api_key)

    async def complete(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None,
    ) -> LLMResponse:
        import asyncio

        # Build Cohere message format
        cohere_messages = []
        for m in messages:
            cohere_messages.append({"role": m.role, "content": m.content})

        t0 = time.monotonic()
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            None,
            lambda: self._client.chat(
                model=self._model,
                messages=cohere_messages,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
        )
        latency = (time.monotonic() - t0) * 1000

        content_text = ""
        if resp.message and resp.message.content:
            content_text = resp.message.content[0].text

        usage = {}
        if hasattr(resp, "usage") and resp.usage:
            tokens = getattr(resp.usage, "tokens", None)
            if isinstance(tokens, dict):
                usage = {
                    "prompt_tokens": tokens.get("input_tokens", 0),
                    "completion_tokens": tokens.get("output_tokens", 0),
                    "total_tokens": tokens.get("input_tokens", 0) + tokens.get("output_tokens", 0),
                }

        return LLMResponse(
            content=content_text,
            model=self._model,
            usage=usage,
            latency_ms=latency,
            raw=resp,
        )


class DeepSeekProvider(BaseLLMProvider):
    """DeepSeek API — OpenAI-compatible."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "deepseek-chat",
        timeout: float = 60.0,
    ):
        self._inner = OpenAIProvider(
            api_key=api_key or os.environ.get("DEEPSEEK_API_KEY", ""),
            base_url="https://api.deepseek.com",
            model=model,
            timeout=timeout,
        )

    @property
    def name(self) -> str:
        return f"deepseek:{self._inner._model}"

    def is_available(self) -> bool:
        return self._inner.is_available()

    async def complete(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None,
    ) -> LLMResponse:
        return await self._inner.complete(messages, temperature, max_tokens, response_format)


class GroqProvider(BaseLLMProvider):
    """Groq API — OpenAI-compatible."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.3-70b-versatile",
        timeout: float = 60.0,
    ):
        self._inner = OpenAIProvider(
            api_key=api_key or os.environ.get("GROQ_API_KEY", ""),
            base_url="https://api.groq.com/openai",
            model=model,
            timeout=timeout,
        )

    @property
    def name(self) -> str:
        return f"groq:{self._inner._model}"

    def is_available(self) -> bool:
        return self._inner.is_available()

    async def complete(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None,
    ) -> LLMResponse:
        return await self._inner.complete(messages, temperature, max_tokens, response_format)


class MistralProvider(BaseLLMProvider):
    """Mistral AI API — OpenAI-compatible."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "mistral-large-latest",
        timeout: float = 60.0,
    ):
        self._inner = OpenAIProvider(
            api_key=api_key or os.environ.get("MISTRAL_API_KEY", ""),
            base_url="https://api.mistral.ai",
            model=model,
            timeout=timeout,
        )

    @property
    def name(self) -> str:
        return f"mistral:{self._inner._model}"

    def is_available(self) -> bool:
        return self._inner.is_available()

    async def complete(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None,
    ) -> LLMResponse:
        return await self._inner.complete(messages, temperature, max_tokens, response_format)


class xAIProvider(BaseLLMProvider):
    """xAI (Grok) API — OpenAI-compatible."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "grok-2",
        timeout: float = 60.0,
    ):
        self._inner = OpenAIProvider(
            api_key=api_key or os.environ.get("XAI_API_KEY", ""),
            base_url="https://api.x.ai",
            model=model,
            timeout=timeout,
        )

    @property
    def name(self) -> str:
        return f"xai:{self._inner._model}"

    def is_available(self) -> bool:
        return self._inner.is_available()

    async def complete(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None,
    ) -> LLMResponse:
        return await self._inner.complete(messages, temperature, max_tokens, response_format)


# ──────────────────────────────────────────────────────────────
# Provider Registry
# ──────────────────────────────────────────────────────────────

PROVIDER_REGISTRY: Dict[str, type] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "ollama": OllamaProvider,
    "gemini": GoogleGeminiProvider,
    "google": GoogleGeminiProvider,
    "cohere": CohereProvider,
    "deepseek": DeepSeekProvider,
    "groq": GroqProvider,
    "mistral": MistralProvider,
    "xai": xAIProvider,
    "grok": xAIProvider,
}


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

        # Check known provider names that have dedicated classes
        if provider_name in PROVIDER_REGISTRY:
            provider_cls = PROVIDER_REGISTRY[provider_name]
            # Build kwargs for the provider
            kwargs: Dict[str, Any] = {}
            if provider_name in ("ollama",):
                kwargs["base_url"] = base_url or None
                kwargs["model"] = model or "llama3.2"
                kwargs["timeout"] = config.get("timeout", 120.0)
            else:
                if api_key:
                    kwargs["api_key"] = api_key
                if model:
                    kwargs["model"] = model
                kwargs["timeout"] = config.get("timeout", 60.0)
            return cls(provider_cls(**kwargs))

        # Fallback: OpenAI-compatible aliases
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

        # Check Gemini
        if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
            model = os.environ.get("ARIDADNE_LLM_MODEL", "gemini-2.0-flash")
            return cls.from_config({"provider": "gemini", "model": model})

        # Check Cohere
        if os.environ.get("COHERE_API_KEY"):
            model = os.environ.get("ARIDADNE_LLM_MODEL", "command-r-plus")
            return cls.from_config({"provider": "cohere", "model": model})

        # Check Groq
        if os.environ.get("GROQ_API_KEY"):
            model = os.environ.get("ARIDADNE_LLM_MODEL", "llama-3.3-70b-versatile")
            return cls.from_config({"provider": "groq", "model": model})

        # Check DeepSeek
        if os.environ.get("DEEPSEEK_API_KEY"):
            model = os.environ.get("ARIDADNE_LLM_MODEL", "deepseek-chat")
            return cls.from_config({"provider": "deepseek", "model": model})

        # Check Mistral
        if os.environ.get("MISTRAL_API_KEY"):
            model = os.environ.get("ARIDADNE_LLM_MODEL", "mistral-large-latest")
            return cls.from_config({"provider": "mistral", "model": model})

        # Check xAI
        if os.environ.get("XAI_API_KEY"):
            model = os.environ.get("ARIDADNE_LLM_MODEL", "grok-2")
            return cls.from_config({"provider": "xai", "model": model})

        # Check Ollama
        ollama = OllamaProvider()
        if ollama.is_available():
            return cls(ollama)

        raise ValueError(
            "No LLM provider available. Set one of:\n"
            "  - OPENAI_API_KEY environment variable\n"
            "  - ANTHROPIC_API_KEY environment variable\n"
            "  - GOOGLE_API_KEY or GEMINI_API_KEY environment variable\n"
            "  - COHERE_API_KEY environment variable\n"
            "  - GROQ_API_KEY environment variable\n"
            "  - DEEPSEEK_API_KEY environment variable\n"
            "  - MISTRAL_API_KEY environment variable\n"
            "  - XAI_API_KEY environment variable\n"
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
