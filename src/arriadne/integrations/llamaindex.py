"""
LlamaIndex Integration for Ariadne

Provides:
- AriadneVectorStore: LlamaIndex VectorStore implementation

Usage:
    from arriadne.integrations.llamaindex import AriadneLlamaIndexStore

    store = AriadneLlamaIndexStore(db_path="memory.db")
    storage_context = store.get_storage_context()

    # Use with LlamaIndex:
    from llama_index.core import VectorStoreIndex
    index = VectorStoreIndex.from_vector_store(store)
    query_engine = index.as_query_engine()
    response = query_engine.query("What is Ariadne?")
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("arriadne.integrations.llamaindex")


class AriadneLlamaIndexStore:
    """
    LlamaIndex-compatible VectorStore backed by Ariadne.

    Implements the key LlamaIndex VectorStore interface:
    - add / delete
    - query (similarity search)
    - get
    """

    def __init__(
        self,
        db_path: str = "ariadne_llamaindex.db",
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
    def from_documents(
        cls,
        documents: List[Any],
        **kwargs: Any,
    ) -> "AriadneLlamaIndexStore":
        """Create store and add documents."""
        store = cls(**kwargs)
        for doc in documents:
            text = doc.text if hasattr(doc, "text") else str(doc)
            metadata = doc.metadata if hasattr(doc, "metadata") else {}
            store._memory.remember(content=text, metadata=metadata)
        return store

    def add(self, nodes: List[Any], **kwargs: Any) -> List[str]:
        """Add LlamaIndex Node objects."""
        ids = []
        for node in nodes:
            text = node.text if hasattr(node, "text") else str(node)
            metadata = {}
            if hasattr(node, "metadata"):
                metadata = node.metadata
            elif hasattr(node, "node_info"):
                metadata = node.node_info

            result = self._memory.remember(content=text, metadata=metadata)
            mid = result.get("memory_id")
            ids.append(str(mid) if mid else "")
        return ids

    def delete(self, ref_doc_id: str, **kwargs: Any) -> None:
        """Delete a document by reference ID."""
        self._memory.delete(ref_doc_id)

    def query(
        self,
        query: Any,
        **kwargs: Any,
    ) -> List[Any]:
        """
        Query the vector store.

        Args:
            query: A VectorStoreQuery object or a string query
        """
        try:
            from llama_index.core.schema import TextNode, QueryBundle
        except ImportError:
            raise ImportError("pip install llama-index-core")

        # Extract query string
        if isinstance(query, str):
            query_str = query
        elif hasattr(query, "query_str"):
            query_str = query.query_str
        elif hasattr(query, "query_embedding"):
            query_str = str(query)
        else:
            query_str = str(query)

        k = kwargs.get("similarity_top_k", 4)
        results = self._memory.recall(query_str, k=k)

        nodes = []
        for r in results:
            node = TextNode(
                text=r["content"],
                id_=str(r["id"]),
                metadata={
                    "score": r.get("score", 0),
                    "memory_type": r.get("memory_type", ""),
                    "importance": r.get("importance", 5),
                },
            )
            nodes.append(node)

        return nodes

    def get_storage_context(self) -> Any:
        """Return a LlamaIndex StorageContext wrapping this store."""
        try:
            from llama_index.core import StorageContext
        except ImportError:
            raise ImportError("pip install llama-index-core")

        return StorageContext.from_defaults(vector_store=self)

    @property
    def memory(self):
        """Access the underlying AriadneMemory instance."""
        return self._memory
