"""Eğitim hedefi denetimli Cross-Encoder yeniden sıralaması.

Bi-Encoder'larla matematiksel karşıtlık
---------------------------------------
Bir Bi-Encoder :math:`s(q, d) = \\phi(q)^\\top \\phi(d)` tanımlar. Kodlayıcı
ortak olduğundan ve bağımsız uygulandığından belge gömmeleri önceden
hesaplanıp önbelleğe alınabilir; bu da erişimi sorgu başına :math:`O(N d)`
yapar — milyonlarca belge için uygulanabilir.

Bir Cross-Encoder ise çifti tek bir girdide birleştirir,
:math:`[CLS]\\,q\\,[SEP]\\,d`, ve skaler bir :math:`s(q, d) \\in
\\mathbb{R}` tahmin eder. Belge gömmeleri önbelleğe alınamaz, bu nedenle
:math:`N` belgeyi yeniden sıralamak :math:`N` Transformer ileri geçişine mal
olur — ancak sorgu ve belge token'ları arasındaki ortak dikkat kesinliği
çarpıcı biçimde artırır.

Dolayısıyla standart üretim deseni şudur:

    Aşama 1: Bi-Encoder    →  ilk N aday           (N ~ 100)
    Aşama 2: Cross-Encoder →  yeniden sıralı ilk k (k ~ 5..10)

Burada uygulanan eğitim hedefleri
---------------------------------
- **Noktasal BCE** (ikili ilgililik): P(d ilgili | q) tahmin edilir,
  kayıp = BCE(σ(s(q, d)), y).
- **İkili marj**: (q, d⁺, d⁻) verildiğinde
  s(q, d⁺) ≥ s(q, d⁻) + marj zorlanır, kayıp = max(0, marj − Δs).

Bu modül her ikisini de eğitim yardımcıları olarak sunar, ancak çıkarım için
varsayılan olarak önceden eğitilmiş ``cross-encoder/ms-marco-MiniLM-L-6-v2``
modelini kullanır.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.rag_engine.pipeline import RetrievedDocument
from src.utils.logging import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class RerankedDocument:
    chunk_id: str
    text: str
    bi_encoder_score: float
    cross_encoder_score: float
    metadata: dict

    @property
    def score(self) -> float:
        return self.cross_encoder_score


class CrossEncoderReranker:
    """Çıkarım için bir sentence-transformers CrossEncoder'ını sarmalar."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name, device=device)
        self.model_name = model_name
        self.batch_size = batch_size

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedDocument],
        k: int | None = None,
    ) -> list[RerankedDocument]:
        if not candidates:
            return []
        pairs = [(query, c.text) for c in candidates]
        scores = self._model.predict(
            pairs, batch_size=self.batch_size, show_progress_bar=False
        )
        ranked = sorted(
            (
                RerankedDocument(
                    chunk_id=c.chunk_id,
                    text=c.text,
                    bi_encoder_score=c.score,
                    cross_encoder_score=float(s),
                    metadata=c.metadata,
                )
                for c, s in zip(candidates, scores, strict=True)
            ),
            key=lambda d: d.cross_encoder_score,
            reverse=True,
        )
        return ranked[:k] if k is not None else ranked


# ---------------------------------------------------------------------- #
# Eğitim hedefleri (Hafta 3 not defterinden başvurulur)                 #
# ---------------------------------------------------------------------- #


def pointwise_bce_loss(scores: np.ndarray, labels: np.ndarray) -> float:
    """Noktasal ilgililik etiketleri üzerinde ikili çapraz entropi.

    :math:`\\mathcal{L} = -\\frac{1}{N}\\sum_i\\bigl(y_i\\log\\sigma(s_i)
        + (1 - y_i)\\log(1 - \\sigma(s_i))\\bigr)`
    """
    sig = 1.0 / (1.0 + np.exp(-scores))
    eps = 1e-7
    return -float(np.mean(labels * np.log(sig + eps) + (1 - labels) * np.log(1 - sig + eps)))


def pairwise_margin_loss(
    pos_scores: np.ndarray, neg_scores: np.ndarray, margin: float = 1.0
) -> float:
    """(pozitif, negatif) skor çiftleri üzerinde menteşe-marj kaybı.

    :math:`\\mathcal{L} = \\frac{1}{N}\\sum_i \\max(0,\\, m - (s^+_i - s^-_i))`
    """
    return float(np.mean(np.maximum(0.0, margin - (pos_scores - neg_scores))))


class CohereReranker:
    """Cohere'un barındırılan yeniden sıralama API'si için isteğe bağlı bağdaştırıcı.

    :class:`CrossEncoderReranker` ile aynı arayüzü paylaşır; böylece hat
    sınırında doğrudan yerine takılabilir.
    """

    def __init__(self, model: str = "rerank-english-v3.0", api_key: str | None = None) -> None:
        import os

        import cohere

        self._client = cohere.Client(api_key or os.environ["COHERE_API_KEY"])
        self.model = model

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedDocument],
        k: int | None = None,
    ) -> list[RerankedDocument]:
        if not candidates:
            return []
        resp = self._client.rerank(
            model=self.model,
            query=query,
            documents=[c.text for c in candidates],
            top_n=k or len(candidates),
        )
        out: list[RerankedDocument] = []
        for r in resp.results:
            c = candidates[r.index]
            out.append(
                RerankedDocument(
                    chunk_id=c.chunk_id,
                    text=c.text,
                    bi_encoder_score=c.score,
                    cross_encoder_score=float(r.relevance_score),
                    metadata=c.metadata,
                )
            )
        return out
