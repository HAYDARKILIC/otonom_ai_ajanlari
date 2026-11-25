"""Sorgu yeniden yazıcı + erişimci + yeniden sıralayıcıyı birleştiren uçtan uca yeniden sıralama hattı."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.rag_engine.pipeline import RetrievedDocument
from src.reranking.cross_encoder import (
    CohereReranker,
    CrossEncoderReranker,
    RerankedDocument,
)
from src.reranking.query_rewriter import QueryRewriter
from src.utils.config import load_yaml_config
from src.utils.logging import get_logger

log = get_logger(__name__)


class Reranker(Protocol):
    def rerank(
        self,
        query: str,
        candidates: list[RetrievedDocument],
        k: int | None = None,
    ) -> list[RerankedDocument]: ...


@dataclass(slots=True)
class RerankingPipeline:
    """Şunları yönetir: (isteğe bağlı yeniden yazma) → erişim → yeniden sıralama → ilk k.

    Erişimci dışarıdan enjekte edilir; bu nesneye yalnızca yeniden sıralayıcı aittir.
    """

    reranker: Reranker
    query_rewriter: QueryRewriter | None = None
    candidate_k: int = 50

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedDocument],
        k: int = 5,
    ) -> list[RerankedDocument]:
        if self.query_rewriter is not None:
            query = self.query_rewriter.rewrite(query)
        # Cross-Encoder maliyeti aday sayısıyla doğrusal artar: en fazla candidate_k aday puanlanır.
        return self.reranker.rerank(query, candidates[: self.candidate_k], k=k)

    @classmethod
    def from_config(cls, path: str) -> RerankingPipeline:
        cfg = load_yaml_config(path)
        backend = cfg["reranker"]["backend"]
        if backend == "cross_encoder":
            rer: Reranker = CrossEncoderReranker(
                model_name=cfg["reranker"].get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
                device=cfg["reranker"].get("device"),
            )
        elif backend == "cohere":
            rer = CohereReranker(model=cfg["reranker"].get("model", "rerank-english-v3.0"))
        else:
            raise ValueError(f"Bilinmeyen yeniden sıralayıcı arka ucu: {backend!r}")
        return cls(
            reranker=rer,
            query_rewriter=None,
            candidate_k=cfg["reranker"].get("candidate_k", 50),
        )
