"""API routes for the finance addon."""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

from arriadne.addons import AddonRegistry

logger = logging.getLogger(__name__)

# Lazy router — created on first access
_router = None


def _get_router():
    """Create and return the FastAPI router."""
    global _router
    if _router is not None:
        return _router

    try:
        from fastapi import APIRouter, File, HTTPException, Query, UploadFile
    except ImportError:
        return None

    _router = APIRouter()

    @_router.get("/health")
    def health():
        return {"status": "ok", "addon": "ariadne-finance"}

    @_router.post("/ingest")
    async def ingest_document(
        file: UploadFile = File(...),
        importance: float = Query(0.5, ge=0, le=1),
        db_path: str = Query("arriadne.db"),
    ):
        """Ingest a financial document (PDF, Excel, CSV) into Ariadne."""
        from arriadne.config import AriadneConfig
        from arriadne.interface import AriadneMemory

        # Save uploaded file to temp
        suffix = Path(file.filename or "upload").suffix or ".bin"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            # Find extractor
            registry = AddonRegistry()
            registry.discover()
            extractor = registry.get_extractor_for_file(tmp_path)
            registry.shutdown()

            if extractor is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"No extractor available for file type: {suffix}",
                )

            # Extract
            result = extractor.extract(tmp_path)
            text = result.get("content", "")
            entities = result.get("entities", [])

            if not text.strip():
                raise HTTPException(status_code=400, detail="No content extracted")

            # Store in Ariadne
            config = AriadneConfig(db_path=db_path)
            mem = AriadneMemory(config=config)

            entity_names = [e["value"] for e in entities if e.get("type") in ("ticker", "sector")]

            # Chunk and store
            chunks = _chunk(text)
            stored = 0
            for i, chunk in enumerate(chunks):
                r = mem.remember(
                    content=chunk,
                    memory_type="episodic",
                    importance=importance,
                    entities=entity_names if entity_names else None,
                    metadata={
                        "source": file.filename,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "file_type": suffix,
                    },
                )
                if r["status"] == "created":
                    stored += 1

            mem.close()

            return {
                "ok": True,
                "stored": stored,
                "total_chunks": len(chunks),
                "entities": entity_names,
                "content_length": len(text),
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error("Ingest failed: %s", e)
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @_router.get("/tickers")
    def extract_tickers(text: str = Query(..., description="Text to scan for tickers")):
        """Extract stock ticker symbols from text."""
        from arriadne_finance.extractors import recognize_tickers
        return {"tickers": recognize_tickers(text)}

    @_router.get("/classify")
    def classify_sector(text: str = Query(..., description="Text to classify")):
        """Classify text into market sectors."""
        from arriadne_finance.extractors import classify_sector
        return {"sectors": classify_sector(text)}

    return _router


def _chunk(content: str, max_chars: int = 2000) -> list[str]:
    """Split content into chunks."""
    if len(content) <= max_chars:
        return [content]
    chunks = []
    paragraphs = content.split("\n\n")
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 > max_chars:
            if current:
                chunks.append(current.strip())
            current = para
        else:
            current = current + "\n\n" + para if current else para
    if current.strip():
        chunks.append(current.strip())
    return chunks


# Module-level router for direct import
router = _get_router()
