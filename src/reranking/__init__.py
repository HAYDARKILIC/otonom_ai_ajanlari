"""Hafta 3 — Sorgu dönüşümü ve yeniden sıralama.

İki aşamalı erişim:

    Aşama 1: Bi-Encoder       (ucuz, paralel, geri çağırma odaklı)
                ↓
    Aşama 2: Cross-Encoder    (pahalı, sıralı, kesinlik odaklı)

Bi-Encoder sorguyu ve belgeyi *bağımsız olarak* gömer; benzerlik bir vektör
nokta çarpımıdır. Cross-Encoder ise (sorgu, belge) çiftini birlikte kodlar ve
skaler bir ilgililik skoru üretir. Cross-Encoder daha isabetlidir ancak
karesel olarak daha pahalıdır — bu da onu yalnızca Bi-Encoder'ın ürettiği
küçük bir aday kümesini yeniden sıralamaya uygun kılar.
"""

from src.reranking.cross_encoder import CrossEncoderReranker
from src.reranking.hyde import HyDE
from src.reranking.pipeline import RerankingPipeline
from src.reranking.query_rewriter import QueryRewriter

__all__ = [
    "CrossEncoderReranker",
    "HyDE",
    "QueryRewriter",
    "RerankingPipeline",
]
