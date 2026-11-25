"""Vektör deposu bağdaştırıcıları: Chroma ve Milvus arka uçlarına sahip ortak bir Protocol.

Hat yalnızca :class:`VectorStore`'a bağımlıdır; somut arka uçlar yapılandırma
anında seçilir. Gömmeler önceden hesaplanmış olarak verilir, böylece kodlayıcı
hiçbir zaman arka uca ait olmaz — bu, gömme modelini değiştirilebilir tutar ve
arka uca özgü kodlayıcı bağımlılığını önler.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass(slots=True)
class ScoredHit:
    chunk_id: str
    score: float
    text: str
    metadata: dict[str, Any]


class VectorStore(Protocol):
    """Bir gömme deposu için arka uçtan bağımsız arayüz."""

    def upsert(
        self,
        ids: list[str],
        vectors: np.ndarray,
        texts: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None: ...

    def query(
        self, vector: np.ndarray, k: int, where: dict[str, Any] | None = None
    ) -> list[ScoredHit]: ...

    def get_by_id(self, chunk_id: str) -> ScoredHit | None: ...


def _sanitize_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Üst verileri Chroma'nın kabul ettiği skaler tiplere (str/int/float/bool) indirger.

    Chroma liste, sözlük veya None değerlerini reddeder; arXiv kayıtlarındaki
    ``authors`` ve ``categories`` gibi listeler ``"; "`` ile birleştirilir.
    """
    out: dict[str, Any] = {}
    for key, value in meta.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            out[key] = value
        elif isinstance(value, (list, tuple, set)):
            out[key] = "; ".join(str(v) for v in value)
        else:
            out[key] = str(value)
    return out


class ChromaStore:
    """Chroma arka ucu (yerel geliştirme için önerilir)."""

    def __init__(self, collection: str, persist_dir: str) -> None:
        import chromadb

        self._client = chromadb.PersistentClient(path=persist_dir)
        self._coll = self._client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(
        self,
        ids: list[str],
        vectors: np.ndarray,
        texts: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        self._coll.upsert(
            ids=ids,
            embeddings=vectors.tolist(),
            documents=texts,
            metadatas=[_sanitize_metadata(m) for m in metadatas],
        )

    def query(
        self, vector: np.ndarray, k: int, where: dict[str, Any] | None = None
    ) -> list[ScoredHit]:
        if k <= 0:
            return []
        count = self._coll.count()
        if count == 0:
            return []
        res = self._coll.query(
            query_embeddings=[vector.tolist()],
            n_results=min(k, count),  # koleksiyondan fazla sonuç istemek uyarı/hata üretir
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        ids = res["ids"][0]
        dists = (res["distances"] or [[]])[0]
        docs = (res["documents"] or [[]])[0]
        metas = (res["metadatas"] or [[]])[0]
        return [
            # Kosinüs uzaklığı d ∈ [0, 2] → benzerlik 1 - d ∈ [-1, 1].
            ScoredHit(chunk_id=i, score=1.0 - float(d), text=t or "", metadata=dict(m or {}))
            for i, d, t, m in zip(ids, dists, docs, metas, strict=True)
        ]

    def get_by_id(self, chunk_id: str) -> ScoredHit | None:
        res = self._coll.get(ids=[chunk_id], include=["documents", "metadatas"])
        if not res["ids"]:
            return None
        docs = res["documents"] or [""]
        metas = res["metadatas"] or [{}]
        return ScoredHit(
            chunk_id=res["ids"][0],
            score=1.0,
            text=docs[0] or "",
            metadata=dict(metas[0] or {}),
        )


class MilvusStore:
    """Milvus arka ucu (ölçek için önerilir).

    Uygulama notu: burada bilinçli olarak asgari tutulmuştur; üretim dağıtımı
    :class:`ChromaStore` kullanır. Bu bağdaştırıcı, takılıp çıkarılabilir arka
    uç deseninin kavram kanıtıdır.
    """

    def __init__(self, uri: str, token: str, collection: str, dim: int) -> None:
        from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

        connections.connect("default", uri=uri, token=token)
        if not utility.has_collection(collection):
            fields = [
                FieldSchema("chunk_id", dtype=DataType.VARCHAR, max_length=128, is_primary=True, auto_id=False),
                FieldSchema("vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
                FieldSchema("text", dtype=DataType.VARCHAR, max_length=8192),
                FieldSchema("metadata", dtype=DataType.JSON),
            ]
            schema = CollectionSchema(fields=fields, description="hiyerarşik RAG parçaları")
            Collection(name=collection, schema=schema)
        self._coll = Collection(collection)
        # Milvus, vektör alanında bir indeks olmadan koleksiyonu belleğe yükleyemez.
        if not self._coll.has_index():
            self._coll.create_index(
                field_name="vector",
                index_params={"index_type": "HNSW", "metric_type": "COSINE",
                              "params": {"M": 16, "efConstruction": 200}},
            )
        self._coll.load()

    def upsert(
        self,
        ids: list[str],
        vectors: np.ndarray,
        texts: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        self._coll.upsert([ids, vectors.tolist(), texts, metadatas])
        self._coll.flush()

    def query(
        self, vector: np.ndarray, k: int, where: dict[str, Any] | None = None
    ) -> list[ScoredHit]:
        res = self._coll.search(
            data=[vector.tolist()],
            anns_field="vector",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=k,
            expr=_milvus_filter(where),
            output_fields=["text", "metadata"],
            consistency_level="Strong",  # az önce yazılan kayıtlar da görünsün
        )[0]
        return [
            ScoredHit(
                chunk_id=str(hit.id),
                score=float(hit.score),
                text=hit.entity.get("text") or "",
                metadata=hit.entity.get("metadata") or {},
            )
            for hit in res
        ]

    def get_by_id(self, chunk_id: str) -> ScoredHit | None:
        escaped = chunk_id.replace("\\", "\\\\").replace('"', '\\"')
        rows = self._coll.query(
            expr=f'chunk_id == "{escaped}"',
            output_fields=["text", "metadata"],
            consistency_level="Strong",
        )
        if not rows:
            return None
        return ScoredHit(
            chunk_id=chunk_id,
            score=1.0,
            text=rows[0].get("text", ""),
            metadata=rows[0].get("metadata") or {},
        )


def _milvus_filter(where: dict[str, Any] | None) -> str | None:
    """Basit ``{alan: değer}`` eşitlik filtresini Milvus JSON-alanı ifadesine çevirir."""
    if not where:
        return None
    clauses = []
    for key, value in where.items():
        if isinstance(value, bool):
            literal = "true" if value else "false"
        elif isinstance(value, (int, float)):
            literal = str(value)
        else:
            escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
            literal = f'"{escaped}"'
        clauses.append(f'metadata["{key}"] == {literal}')
    return " and ".join(clauses)


def build_store(kind: str, **kwargs: Any) -> VectorStore:
    if kind == "chroma":
        return ChromaStore(**kwargs)
    if kind == "milvus":
        return MilvusStore(**kwargs)
    raise ValueError(f"Bilinmeyen vektör deposu: {kind!r}")
