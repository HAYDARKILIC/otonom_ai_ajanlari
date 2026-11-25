"""Uçtan uca hiyerarşik RAG hattı.

Şunları birbirine bağlar: parçalayıcı → kodlayıcı → vektör deposu; erişim
anında çocuk parçalardan ebeveynlerine (veya cümle penceresi stratejisinde
çevredeki pencereye) geri genişletme yapılır.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.rag_engine.chunking import (
    Chunk,
    HierarchicalChunker,
    SentenceWindowChunker,
)
from src.rag_engine.embeddings import SentenceEncoder
from src.rag_engine.parent_store import SQLiteParentStore
from src.rag_engine.vector_stores import ScoredHit, VectorStore, build_store
from src.utils.config import load_yaml_config
from src.utils.logging import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class RetrievedDocument:
    """Yeniden sıralayıcıya / LLM'e verilen, ebeveyne genişletilmiş metinli belge."""

    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any]


class HierarchicalRAG:
    """Ebeveyn-çocuk hiyerarşik erişim hattı."""

    def __init__(
        self,
        chunker: HierarchicalChunker,
        encoder: SentenceEncoder,
        child_store: VectorStore,
        parent_lookup: MutableMapping[str, str] | None = None,
    ) -> None:
        self.chunker = chunker
        self.encoder = encoder
        self.child_store = child_store
        # child.parent_id -> parent_text eşlemesi. Bellek içi bir ``dict`` not
        # defterleri için yeterlidir; indeks ile sorgunun farklı süreçlerde
        # çalıştığı durumlarda kalıcı bir :class:`SQLiteParentStore` verin.
        self.parent_lookup: MutableMapping[str, str] = {} if parent_lookup is None else parent_lookup

    @classmethod
    def from_config(cls, path: str) -> HierarchicalRAG:
        cfg = load_yaml_config(path)
        chunker = HierarchicalChunker(
            parent_chars=cfg["chunker"].get("parent_chars", 4000),
            child_chars=cfg["chunker"].get("child_chars", 1000),
            child_overlap_chars=cfg["chunker"].get("child_overlap_chars", 100),
        )
        encoder = SentenceEncoder(
            model_name=cfg["encoder"]["model"], device=cfg["encoder"].get("device"),
        )
        store_cfg = dict(cfg["vector_store"])
        parent_path = store_cfg.pop("parent_store", None)
        child_store = build_store(**store_cfg)
        if parent_path is None and "persist_dir" in store_cfg:
            # Varsayılan: ebeveynler vektör deposunun yanında, koleksiyona özgü bir dosyada.
            collection = store_cfg.get("collection", "default")
            parent_path = Path(store_cfg["persist_dir"]) / f"{collection}_parents.sqlite"
        parents = SQLiteParentStore(parent_path) if parent_path else {}
        return cls(chunker=chunker, encoder=encoder, child_store=child_store, parent_lookup=parents)

    # ------------------------------------------------------------------ #
    # Veri alma                                                          #
    # ------------------------------------------------------------------ #

    def ingest_document(self, text: str, metadata: dict[str, Any]) -> int:
        parents, children = self.chunker.chunk(text, metadata)
        parent_map = {p.chunk_id: p.text for p in parents}
        if isinstance(self.parent_lookup, SQLiteParentStore):
            self.parent_lookup.update_many(parent_map)
        else:
            self.parent_lookup.update(parent_map)
        if not children:
            return 0
        vectors = self.encoder.encode([c.text for c in children])
        self.child_store.upsert(
            ids=[c.chunk_id for c in children],
            vectors=vectors,
            texts=[c.text for c in children],
            metadatas=[c.metadata | {"parent_id": c.parent_id or ""} for c in children],
        )
        log.info("rag_ingest_done", parents=len(parents), children=len(children))
        return len(children)

    # ------------------------------------------------------------------ #
    # Erişim                                                             #
    # ------------------------------------------------------------------ #

    def retrieve(
        self,
        query: str,
        k: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedDocument]:
        qvec = self.encoder.encode_one(query)
        hits = self.child_store.query(qvec, k=k, where=where)
        return self._expand_to_parents(hits)

    def _expand_to_parents(self, hits: list[ScoredHit]) -> list[RetrievedDocument]:
        seen_parents: set[str] = set()
        out: list[RetrievedDocument] = []
        for hit in hits:
            parent_id = hit.metadata.get("parent_id") or ""
            # Tekilleştir: tek bir ebeveyn çoğu zaman birden çok çocuk isabetine katkıda bulunur.
            if parent_id in seen_parents:
                continue
            seen_parents.add(parent_id)
            parent_text = self.parent_lookup.get(parent_id, hit.text)
            out.append(
                RetrievedDocument(
                    chunk_id=parent_id or hit.chunk_id,
                    text=parent_text,
                    score=hit.score,
                    metadata=hit.metadata,
                )
            )
        return out


class SentenceWindowRAG:
    """Cümle penceresi erişim hattı.

    Erişim kesinliği için cümleler tek tek indekslenir; LLM'in yeterli bağlam
    alması için ±pencere bağlamı erişim anında genişletilir.
    """

    def __init__(
        self,
        chunker: SentenceWindowChunker,
        encoder: SentenceEncoder,
        store: VectorStore,
    ) -> None:
        self.chunker = chunker
        self.encoder = encoder
        self.store = store
        # Aynı süreçte alınan belgeler için hızlı yol; diğerleri depodan getirilir.
        self._by_doc: dict[str, list[Chunk]] = {}

    def ingest_document(self, text: str, metadata: dict[str, Any]) -> int:
        chunks = self.chunker.chunk(text, metadata)
        if not chunks:
            return 0
        self._by_doc[chunks[0].metadata["doc_id"]] = chunks
        vectors = self.encoder.encode([c.text for c in chunks])
        self.store.upsert(
            ids=[c.chunk_id for c in chunks],
            vectors=vectors,
            texts=[c.text for c in chunks],
            metadatas=[c.metadata for c in chunks],
        )
        return len(chunks)

    def retrieve(self, query: str, k: int = 10) -> list[RetrievedDocument]:
        qvec = self.encoder.encode_one(query)
        hits = self.store.query(qvec, k=k)
        out: list[RetrievedDocument] = []
        for hit in hits:
            doc_id = hit.metadata.get("doc_id")
            idx = hit.metadata.get("sentence_idx")
            if doc_id and doc_id in self._by_doc:
                window_chunk = Chunk(
                    chunk_id=hit.chunk_id, text=hit.text,
                    metadata={"doc_id": doc_id, "sentence_idx": idx or 0},
                )
                expanded = self.chunker.expand_window(window_chunk, self._by_doc)
            elif doc_id and isinstance(idx, int):
                expanded = self._expand_from_store(doc_id, idx, hit)
            else:
                expanded = hit.text
            out.append(
                RetrievedDocument(
                    chunk_id=hit.chunk_id, text=expanded,
                    score=hit.score, metadata=hit.metadata,
                )
            )
        return out

    def _expand_from_store(self, doc_id: str, idx: int, hit: ScoredHit) -> str:
        """Pencereyi vektör deposundan kurar (indeks başka bir süreçte oluşturulduysa).

        Cümle parçalarının kimlikleri deterministiktir (``{doc_id}::s{idx}``), bu
        yüzden komşu cümleler doğrudan kimlikleriyle getirilebilir.
        """
        total = hit.metadata.get("total_sentences", idx + self.chunker.window + 1)
        lo = max(0, idx - self.chunker.window)
        hi = min(int(total), idx + self.chunker.window + 1)
        parts: list[str] = []
        for i in range(lo, hi):
            if i == idx:
                parts.append(hit.text)
                continue
            neighbour = self.store.get_by_id(f"{doc_id}::s{i}")
            if neighbour is not None:
                parts.append(neighbour.text)
        return " ".join(parts)
