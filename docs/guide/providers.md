---
title: Providers
description: Complete LLM and embedding provider support — 12 LLM providers, 10 embedding providers
---

# Providers

Ariadne supports a wide range of LLM and embedding providers out of the box. Every provider is lazy-loaded — missing dependencies don't break the system.

## LLM Providers

| Provider | Class | Default Model | Environment Variable |
|----------|-------|--------------|---------------------|
| OpenAI | `OpenAIProvider` | `gpt-4o-mini` | `OPENAI_API_KEY` |
| Anthropic | `AnthropicProvider` | `claude-sonnet-4-20250514` | `ANTHROPIC_API_KEY` |
| Ollama | `OllamaProvider` | `llama3.2` | (local server) |
| Google Gemini | `GoogleGeminiProvider` | `gemini-2.0-flash` | `GEMINI_API_KEY` |
| Cohere | `CohereProvider` | `command-r-plus` | `COHERE_API_KEY` |
| DeepSeek | `DeepSeekProvider` | `deepseek-chat` | `DEEPSEEK_API_KEY` |
| Groq | `GroqProvider` | `llama-3.3-70b-versatile` | `GROQ_API_KEY` |
| Mistral | `MistralProvider` | `mistral-large-latest` | `MISTRAL_API_KEY` |
| xAI (Grok) | `xAIProvider` | `grok-2` | `XAI_API_KEY` |
| OpenRouter | `OpenAIProvider` | varies | `OPENROUTER_API_KEY` |
| Together AI | `OpenAIProvider` | varies | `TOGETHER_API_KEY` |
| LM Studio | `OpenAIProvider` | varies | (local server) |

### Using the Provider Registry

```python
from arriadne.llm import LLMProvider, PROVIDER_REGISTRY

# Auto-detect from environment
llm = LLMProvider.auto_detect()

# Explicit config
llm = LLMProvider.from_config({
    "provider": "gemini",
    "model": "gemini-2.0-flash",
})

# Direct instantiation
from arriadne.llm import GroqProvider
provider = GroqProvider(api_key="gsk_...", model="llama-3.3-70b-versatile")
llm = LLMProvider(provider)
```

### OpenAI-Compatible Providers

DeepSeek, Groq, Mistral, and xAI all use the OpenAI-compatible API format. You can also use any other OpenAI-compatible endpoint:

```python
from arriadne.llm import OpenAIProvider, LLMProvider

provider = OpenAIProvider(
    api_key="your-key",
    base_url="https://your-custom-endpoint.com/v1",
    model="custom-model-name",
)
llm = LLMProvider(provider)
```

## Embedding Providers

| Provider | Class | Default Model | Dimensions | Environment Variable |
|----------|-------|--------------|:----------:|---------------------|
| ONNX (zero-config) | `OnnxEmbedding` | `all-MiniLM-L6-v2` | 384 | (auto-download) |
| Sentence Transformers | `SentenceTransformerEmbedding` | `all-MiniLM-L6-v2` | 384 | (local model) |
| Nomic (ONNX) | `NomicEmbedding` | `nomic-embed-text-v1.5` | 768 | (auto-download) |
| BGE (ONNX) | `BGEEmbedding` | `bge-small-en-v1.5` | 384 | (auto-download) |
| OpenAI | `OpenAIEmbedding` | `text-embedding-3-small` | 1536 | `OPENAI_API_KEY` |
| Cohere | `CohereEmbedding` | `embed-english-v3.0` | 1024 | `COHERE_API_KEY` |
| Jina | `JinaEmbedding` | `jina-embeddings-v3` | 1024 | `JINA_API_KEY` |
| Voyage | `VoyageEmbedding` | `voyage-3` | 1024 | `VOYAGE_API_KEY` |
| Keyword (fallback) | `KeywordEmbedding` | TF-IDF hash | configurable | (none needed) |
| Custom | `CustomEmbedding` | user-defined | user-defined | (none) |

### Auto-Detection

```python
from arriadne import AriadneMemory

# Automatically selects the best available provider
mem = AriadneMemory("my_memory.db")

# Check which provider was selected
print(f"Embedding provider: {mem.embedding_provider.name}")
print(f"Dimensions: {mem.embedding_dimension}")
```

### Explicit Selection

```python
from arriadne.embeddings import OnnxEmbedding, CohereEmbedding, EMBEDDING_REGISTRY

# By name
from arriadne.embeddings import EMBEDDING_REGISTRY
provider = EMBEDDING_REGISTRY["nomic"](dimension=768)

# Direct instantiation
provider = CohereEmbedding(api_key="your-key", dimension=1024)
```

### Embedding Provider Selection Order

When using `auto_detect_provider()`, the fallback order is:

1. ONNX (auto-downloads model, zero config)
2. Sentence Transformers (if PyTorch installed)
3. Nomic (ONNX variant, 768-dim)
4. BGE (ONNX variant, 384-dim)
5. OpenAI (requires API key)
6. Cohere (requires API key)
7. Jina (requires API key)
8. Keyword fallback (no model needed)

## Integration with Hermes

The Hermes plugin automatically selects the best available provider:

```yaml
# ~/.hermes/config.yaml
memory:
  provider: ariadne
  ariadne:
    embedding:
      provider: onnx  # or nomic, bge, openai, cohere, etc.
    llm:
      provider: ollama
      model: llama3.2
```
