"""
LangChain Integration for Ariadne

Provides:
- AriadneVectorStore: LangChain VectorStore implementation
- AriadneRetriever: LangChain Retriever for RAG pipelines

Usage:
    from arriadne.integrations.langchain import AriadneVectorStore

    store = AriadneVectorStore(db_path="memory.db")
    store.add_texts(["Paris is the capital of France", "London is the capital of England"])
    results = store.similarity_search("capital of France", k=5)

    # As a retriever for RAG:
    retriever = store.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke("What is the capital of France?")
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger("arriadne.integrations.langchain")


def _check_langchain():
    """Check that LangChain is available."""
    try:
        from langchain_core.vectorstores import VectorStore
        from langchain_core.embeddings import Embeddings
        from langchain_core.documents import Document
        return VectorStore, Embeddings, Document
    except ImportError:
        raise ImportError(
            "pip install langchain-core  # Required for LangChain integration"
        )


class AriadneVectorStore:
    """
    LangChain-compatible VectorStore backed by Ariadne.

    Implements the key VectorStore interface:
    - add_texts / add_documents
    - similarity_search / similarity_search_with_score
    - delete
    - as_retriever
    """

    def __init__(
        self,
        db_path: str = "ariadne_langchain.db",
        embedding_dim: int = 384,
        embedding_provider: Optional[str] = None,
        llm_config: Optional[Dict] = None,
        **kwargs: Any,
    ):
        from arriadne.interface import AriadneMemory

        self._memory = AriadneMemory(
            db_path=db_path,
            embedding_dim=embedding_dim,
            embedding_provider=embedding_provider,
            llm_config=llm_config,
        )

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Optional[Any] = None,
        metadatas: Optional[List[Dict]] = None,
        **kwargs: Any,
    ) -> "AriadneVectorStore":
        """Create store and add texts in one step."""
        store = cls(**kwargs)
        store.add_texts(texts, metadatas=metadatas)
        return store

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add texts to the store. Returns list of IDs."""
        ids = []
        for i, text in enumerate(texts):
            meta = metadatas[i] if metadatas else {}
            result = self._memory.remember(
                content=text,
                metadata=meta,
            )
            mid = result.get("memory_id")
            ids.append(str(mid) if mid else str(i))
        return ids

    def add_documents(self, documents: List[Any], **kwargs: Any) -> List[str]:
        """Add LangChain Document objects."""
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return self.add_texts(texts, metadatas=metadatas)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Any]:
        """Search for similar texts. Returns list of Documents."""
        try:
            from langchain_core.documents import Document
        except ImportError:
            raise ImportError("pip install langchain-core")

        results = self._memory.recall(query, k=k)

        docs = []
        for r in results:
            docs.append(Document(
                page_content=r["content"],
                metadata={
                    "id": r["id"],
                    "score": r.get("score", 0),
                    "memory_type": r.get("memory_type", ""),
                    "importance": r.get("importance", 5),
                },
            ))
        return docs

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[tuple]:
        """Search with relevance scores. Returns (Document, score) tuples."""
        try:
            from langchain_core.documents import Document
        except ImportError:
            raise ImportError("pip install langchain-core")

        results = self._memory.recall(query, k=k)

        docs_with_scores = []
        for r in results:
            doc = Document(
                page_content=r["content"],
                metadata={
                    "id": r["id"],
                    "memory_type": r.get("memory_type", ""),
                    "importance": r.get("importance", 5),
                },
            )
            docs_with_scores.append((doc, r.get("score", 0)))
        return docs_with_scores

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> bool:
        """Delete memories by ID."""
        if not ids:
            return False
        success = True
        for mid in ids:
            if not self._memory.delete(mid):
                success = False
        return success

    def as_retriever(self, **kwargs: Any) -> Any:
        """Return a LangChain Retriever wrapping this store."""
        try:
            from langchain_core.retrievers import BaseRetriever
            from langchain_core.documents import Document
        except ImportError:
            raise ImportError("pip install langchain-core")

        store = self

        class AriadneRetriever(BaseRetriever):
            """Retriever that searches Ariadne memory."""

            search_kwargs: Dict[str, Any] = {}

            def _get_relevant_documents(self, query: str, **kwargs2: Any) -> List[Document]:
                k = self.search_kwargs.get("k", 4)
                return store.similarity_search(query, k=k)

        return AriadneRetriever(search_kwargs=kwargs.get("search_kwargs", {"k": 4}))

    def get_by_ids(self, ids: Sequence[str]) -> List[Any]:
        """Get documents by IDs."""
        try:
            from langchain_core.documents import Document
        except ImportError:
            raise ImportError("pip install langchain-core")

        docs = []
        for mid in ids:
            result = self._memory.get(mid)
            if result:
                docs.append(Document(
                    page_content=result["content"],
                    metadata={
                        "id": result["id"],
                        "topic": result.get("topic", ""),
                        "importance": result.get("importance", 5),
                    },
                ))
        return docs

    def _select_relevance_score_fn(self):
        """Return the relevance score function."""
        return lambda score: score  # Already 0-1

    @property
    def memory(self):
        """Access the underlying AriadneMemory instance."""
        return self._memory


class AriadneRetriever:
    """
    Standalone LangChain Retriever for Ariadne.

    Usage:
        from arriadne.integrations.langchain import AriadneRetriever
        retriever = AriadneRetriever(db_path="memory.db")
        docs = retriever.invoke("What is the capital of France?")
    """

    def __init__(
        self,
        db_path: str = "ariadne_retriever.db",
        k: int = 4,
        **kwargs: Any,
    ):
        self._store = AriadneVectorStore(db_path=db_path, **kwargs)
        self._k = k

    def invoke(self, query: str, **kwargs: Any) -> List[Any]:
        """Retrieve relevant documents."""
        k = kwargs.get("k", self._k)
        return self._store.similarity_search(query, k=k)

    def batch(self, queries: List[str], **kwargs: Any) -> List[List[Any]]:
        """Retrieve for multiple queries."""
        return [self.invoke(q, **kwargs) for q in queries]
