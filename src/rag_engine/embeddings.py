"""Çıktıları L2-normalize edilmiş Sentence-Transformer kodlayıcısı.

RAG ve yeniden sıralama yığınının tamamında kullanılan kodlayıcı. Çıktılar
L2-normalize edilir; böylece iç çarpım, kosinüs ve Öklid benzerliği monoton
dönüşümlere kadar eşdeğer hâle gelir — bu da varsayılan metrik konusunda
anlaşamayan vektör deposu arka uçlarındaki ince hataların bütün bir sınıfını
ortadan kaldırır.
"""

from __future__ import annotations

import numpy as np


class SentenceEncoder:
    """``sentence-transformers`` etrafında ince bir sarmalayıcı."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer  # yerel içe aktarma soğuk başlatmayı ucuz tutar

        self._model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name
        self.dim = self._model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Kodlar ve L2-normalize eder. ``(n, dim)`` boyutlu float32 döndürür."""
        vectors = self._model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.astype(np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]
