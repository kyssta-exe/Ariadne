"""Tests for LLM and embedding provider registries and availability detection."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from arriadne.llm import (
    AnthropicProvider,
    BaseLLMProvider,
    CohereProvider,
    DeepSeekProvider,
    GoogleGeminiProvider,
    GroqProvider,
    MistralProvider,
    OpenAIProvider,
    PROVIDER_REGISTRY,
    LLMProvider,
    xAIProvider,
)
from arriadne.embeddings import (
    BGEEmbedding,
    CohereEmbedding,
    CustomEmbedding,
    EMBEDDING_REGISTRY,
    EmbeddingProvider,
    JinaEmbedding,
    KeywordEmbedding,
    NomicEmbedding,
    OnnxEmbedding,
    OpenAIEmbedding,
    VoyageEmbedding,
    auto_detect_provider,
)


# ──────────────────────────────────────────────────────────────
# LLM Provider Registry
# ──────────────────────────────────────────────────────────────


class TestLLMProviderRegistry:
    def test_registry_contains_all_providers(self):
        expected = {
            "openai", "anthropic", "ollama", "gemini", "google",
            "cohere", "deepseek", "groq", "mistral", "xai", "grok",
            "callable",
        }
        assert set(PROVIDER_REGISTRY.keys()) == expected

    def test_registry_values_are_classes(self):
        for name, cls in PROVIDER_REGISTRY.items():
            assert isinstance(cls, type), f"{name} is not a class"
            assert issubclass(cls, BaseLLMProvider), f"{name} does not inherit BaseLLMProvider"

    def test_registry_lookup_gemini(self):
        assert PROVIDER_REGISTRY["gemini"] is GoogleGeminiProvider
        assert PROVIDER_REGISTRY["google"] is GoogleGeminiProvider

    def test_registry_lookup_xai(self):
        assert PROVIDER_REGISTRY["xai"] is xAIProvider
        assert PROVIDER_REGISTRY["grok"] is xAIProvider

    def test_registry_lookup_others(self):
        assert PROVIDER_REGISTRY["cohere"] is CohereProvider
        assert PROVIDER_REGISTRY["deepseek"] is DeepSeekProvider
        assert PROVIDER_REGISTRY["groq"] is GroqProvider
        assert PROVIDER_REGISTRY["mistral"] is MistralProvider


# ──────────────────────────────────────────────────────────────
# LLM Provider is_available() — mocked SDK imports
# ──────────────────────────────────────────────────────────────


class TestLLMAvailability:
    def test_openai_not_available_without_key(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
            provider = OpenAIProvider(api_key="")
            assert not provider.is_available()

    def test_openai_available_with_key(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
            provider = OpenAIProvider(api_key="sk-test")
            assert provider.is_available()

    def test_anthropic_not_available_without_key(self):
        with patch.dict("sys.modules", {"anthropic": MagicMock()}):
            provider = AnthropicProvider(api_key="")
            assert not provider.is_available()

    def test_anthropic_available_with_key(self):
        with patch.dict("sys.modules", {"anthropic": MagicMock()}):
            provider = AnthropicProvider(api_key="test-key")
            assert provider.is_available()

    def test_gemini_not_available_without_key(self):
        mock_genai = MagicMock()
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "", "GEMINI_API_KEY": ""}, clear=False):
            with patch.dict("sys.modules", {"google": MagicMock(), "google.generativeai": mock_genai}):
                provider = GoogleGeminiProvider(api_key="")
                assert not provider.is_available()

    def test_gemini_available_with_key(self):
        mock_genai = MagicMock()
        with patch.dict(os.environ, {"GOOGLE_API_KEY": ""}, clear=False):
            with patch.dict("sys.modules", {"google": MagicMock(), "google.generativeai": mock_genai}):
                provider = GoogleGeminiProvider(api_key="test-key")
                assert provider.is_available()

    def test_gemini_env_var_google_api_key(self):
        mock_genai = MagicMock()
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "from-google"}, clear=False):
            with patch.dict("sys.modules", {"google": MagicMock(), "google.generativeai": mock_genai}):
                provider = GoogleGeminiProvider()
                assert provider._api_key == "from-google"
                assert provider.is_available()

    def test_gemini_env_var_gemini_api_key(self):
        mock_genai = MagicMock()
        with patch.dict(os.environ, {"GEMINI_API_KEY": "from-gemini", "GOOGLE_API_KEY": ""}, clear=False):
            with patch.dict("sys.modules", {"google": MagicMock(), "google.generativeai": mock_genai}):
                provider = GoogleGeminiProvider()
                assert provider._api_key == "from-gemini"
                assert provider.is_available()

    def test_cohere_not_available_without_key(self):
        with patch.dict("sys.modules", {"cohere": MagicMock()}):
            provider = CohereProvider(api_key="")
            assert not provider.is_available()

    def test_cohere_available_with_key(self):
        with patch.dict("sys.modules", {"cohere": MagicMock()}):
            provider = CohereProvider(api_key="test-key")
            assert provider.is_available()

    def test_deepseek_not_available_without_key(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": ""}, clear=False):
            provider = DeepSeekProvider(api_key="")
            assert not provider.is_available()

    def test_deepseek_available_with_key(self):
        provider = DeepSeekProvider(api_key="test-key")
        assert provider.is_available()

    def test_groq_not_available_without_key(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": ""}, clear=False):
            provider = GroqProvider(api_key="")
            assert not provider.is_available()

    def test_groq_available_with_key(self):
        provider = GroqProvider(api_key="test-key")
        assert provider.is_available()

    def test_mistral_not_available_without_key(self):
        with patch.dict(os.environ, {"MISTRAL_API_KEY": ""}, clear=False):
            provider = MistralProvider(api_key="")
            assert not provider.is_available()

    def test_mistral_available_with_key(self):
        provider = MistralProvider(api_key="test-key")
        assert provider.is_available()

    def test_xai_not_available_without_key(self):
        with patch.dict(os.environ, {"XAI_API_KEY": ""}, clear=False):
            provider = xAIProvider(api_key="")
            assert not provider.is_available()

    def test_xai_available_with_key(self):
        provider = xAIProvider(api_key="test-key")
        assert provider.is_available()


# ──────────────────────────────────────────────────────────────
# LLM Provider names
# ──────────────────────────────────────────────────────────────


class TestLLMProviderNames:
    def test_openai_name(self):
        p = OpenAIProvider(api_key="sk-test", model="gpt-4o")
        assert p.name == "openai:gpt-4o"

    def test_anthropic_name(self):
        with patch.dict("sys.modules", {"anthropic": MagicMock()}):
            p = AnthropicProvider(api_key="test", model="claude-3-opus")
            assert p.name == "anthropic:claude-3-opus"

    def test_gemini_name(self):
        mock_genai = MagicMock()
        with patch.dict("sys.modules", {"google": MagicMock(), "google.generativeai": mock_genai}):
            p = GoogleGeminiProvider(api_key="test", model="gemini-2.0-flash")
            assert p.name == "gemini:gemini-2.0-flash"

    def test_cohere_name(self):
        with patch.dict("sys.modules", {"cohere": MagicMock()}):
            p = CohereProvider(api_key="test", model="command-r-plus")
            assert p.name == "cohere:command-r-plus"

    def test_deepseek_name(self):
        p = DeepSeekProvider(api_key="test")
        assert p.name == "deepseek:deepseek-chat"

    def test_groq_name(self):
        p = GroqProvider(api_key="test")
        assert p.name == "groq:llama-3.3-70b-versatile"

    def test_mistral_name(self):
        p = MistralProvider(api_key="test")
        assert p.name == "mistral:mistral-large-latest"

    def test_xai_name(self):
        p = xAIProvider(api_key="test")
        assert p.name == "xai:grok-2"


# ──────────────────────────────────────────────────────────────
# LLMProvider.from_config() integration
# ──────────────────────────────────────────────────────────────


class TestLLMProviderFromConfig:
    def test_from_config_deepseek(self):
        llm = LLMProvider.from_config({"provider": "deepseek", "api_key": "test"})
        assert isinstance(llm._provider, DeepSeekProvider)
        assert llm.name == "deepseek:deepseek-chat"

    def test_from_config_groq(self):
        llm = LLMProvider.from_config({"provider": "groq", "api_key": "test"})
        assert isinstance(llm._provider, GroqProvider)

    def test_from_config_mistral(self):
        llm = LLMProvider.from_config({"provider": "mistral", "api_key": "test"})
        assert isinstance(llm._provider, MistralProvider)

    def test_from_config_xai(self):
        llm = LLMProvider.from_config({"provider": "xai", "api_key": "test"})
        assert isinstance(llm._provider, xAIProvider)

    def test_from_config_openai_compat_alias(self):
        llm = LLMProvider.from_config({"provider": "openrouter", "api_key": "test"})
        assert isinstance(llm._provider, OpenAIProvider)

    def test_from_config_together(self):
        llm = LLMProvider.from_config({"provider": "together", "api_key": "test"})
        assert isinstance(llm._provider, OpenAIProvider)

    def test_from_config_lmstudio(self):
        llm = LLMProvider.from_config({"provider": "lmstudio", "api_key": "test"})
        assert isinstance(llm._provider, OpenAIProvider)

    def test_from_config_gemini(self):
        mock_genai = MagicMock()
        with patch.dict("sys.modules", {"google": MagicMock(), "google.generativeai": mock_genai}):
            llm = LLMProvider.from_config({"provider": "gemini", "api_key": "test"})
            assert isinstance(llm._provider, GoogleGeminiProvider)

    def test_from_config_cohere(self):
        with patch.dict("sys.modules", {"cohere": MagicMock()}):
            llm = LLMProvider.from_config({"provider": "cohere", "api_key": "test"})
            assert isinstance(llm._provider, CohereProvider)

    def test_from_config_anthropic(self):
        with patch.dict("sys.modules", {"anthropic": MagicMock()}):
            llm = LLMProvider.from_config({"provider": "anthropic", "api_key": "test"})
            assert isinstance(llm._provider, AnthropicProvider)


# ──────────────────────────────────────────────────────────────
# Embedding Provider Registry
# ──────────────────────────────────────────────────────────────


class TestEmbeddingProviderRegistry:
    def test_registry_contains_all_providers(self):
        expected = {
            "onnx", "sentence-transformers", "nomic", "bge",
            "openai", "cohere", "jina", "voyage", "keyword", "custom",
        }
        assert set(EMBEDDING_REGISTRY.keys()) == expected

    def test_registry_values_are_classes(self):
        for name, cls in EMBEDDING_REGISTRY.items():
            assert isinstance(cls, type), f"{name} is not a class"
            assert issubclass(cls, EmbeddingProvider), f"{name} does not inherit EmbeddingProvider"

    def test_registry_lookup_nomic(self):
        assert EMBEDDING_REGISTRY["nomic"] is NomicEmbedding

    def test_registry_lookup_bge(self):
        assert EMBEDDING_REGISTRY["bge"] is BGEEmbedding

    def test_registry_lookup_openai(self):
        assert EMBEDDING_REGISTRY["openai"] is OpenAIEmbedding

    def test_registry_lookup_cohere(self):
        assert EMBEDDING_REGISTRY["cohere"] is CohereEmbedding

    def test_registry_lookup_jina(self):
        assert EMBEDDING_REGISTRY["jina"] is JinaEmbedding

    def test_registry_lookup_voyage(self):
        assert EMBEDDING_REGISTRY["voyage"] is VoyageEmbedding


# ──────────────────────────────────────────────────────────────
# Embedding Provider availability
# ──────────────────────────────────────────────────────────────


class TestEmbeddingAvailability:
    def test_keyword_always_available(self):
        kw = KeywordEmbedding()
        assert kw.dimension == 384
        assert kw.name == "keyword"
        emb = kw.embed("test text")
        assert emb.shape == (384,)

    def test_keyword_embed_batch(self):
        kw = KeywordEmbedding(dimension=128)
        embs = kw.embed_batch(["hello", "world"])
        assert embs.shape == (2, 128)

    def test_custom_embedding(self):
        def fn(texts):
            return [[1.0] * 10 for _ in texts]
        ce = CustomEmbedding(fn, dimension=10, name="test-custom")
        assert ce.name == "test-custom"
        emb = ce.embed("hello")
        assert emb.shape == (10,)

    def test_onnx_embedding_default(self):
        ox = OnnxEmbedding()
        assert ox.name == "onnx:all-MiniLM-L6-v2"

    def test_nomic_embedding(self):
        nm = NomicEmbedding()
        assert nm.name == "nomic-embed-text-v1.5"

    def test_bge_embedding(self):
        bge = BGEEmbedding()
        assert bge.name == "bge-small-en-v1.5"


# ──────────────────────────────────────────────────────────────
# auto_detect_provider with preferred
# ──────────────────────────────────────────────────────────────


class TestAutoDetectProvider:
    def test_preferred_keyword(self):
        p = auto_detect_provider(preferred="keyword")
        assert isinstance(p, KeywordEmbedding)

    def test_preferred_onnx(self):
        p = auto_detect_provider(preferred="onnx")
        assert isinstance(p, OnnxEmbedding)

    def test_preferred_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            auto_detect_provider(preferred="nonexistent")

    def test_auto_detect_falls_back_to_keyword(self):
        # With no ONNX, no sentence-transformers, no API keys,
        # should fall back to keyword
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "",
            "COHERE_API_KEY": "",
            "JINA_API_KEY": "",
        }, clear=False):
            # Patch out ONNX and sentence-transformers import attempts
            with patch.dict("sys.modules", {
                "onnxruntime": None,
                "huggingface_hub": None,
                "tokenizers": None,
                "sentence_transformers": None,
            }):
                p = auto_detect_provider(dimension=128)
                assert isinstance(p, KeywordEmbedding)
                assert p.dimension == 128
