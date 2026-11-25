"""HyDE — Varsayımsal Belge Gömmeleri (Hypothetical Document Embeddings; Gao vd., 2022).

Sezgi: Bir Bi-Encoder sorguyu *soru* manifolduna gömer, belgeler ise *yanıt*
manifoldunda yaşar. Bu manifoldların geometrisi farklıdır; yoğun (dense)
erişimin, sorgu kısa ve belgeler uzun açıklayıcı pasajlar olduğunda düşük
performans göstermesinin nedeni budur.

HyDE bu boşluğu, LLM'den sorgu için *makul bir yanıt belgesi halüsinasyonu
üretmesini* isteyip sorgu yerine bu uydurulmuş yanıtı gömerek kapatır.
Halüsinasyon olgusal hatalar içerse bile gömmesi, gerçek yanıt belgelerine
orijinal sorgunun gömmesinden daha yakındır.

Biçimsel olarak, :math:`\\phi : \\Sigma^* \\to \\mathbb{R}^d` Bi-Encoder
olsun. :math:`d` belgesi için erişim skoru şu hâle gelir

.. math::

    s(q, d) \\;=\\; \\phi(\\mathrm{LLM}(q))^\\top \\phi(d)

burada :math:`\\mathrm{LLM}(q)` sentetik yanıttır.
"""

from __future__ import annotations

import numpy as np

from src.rag_engine.embeddings import SentenceEncoder
from src.utils.llm_client import LLMClient, Message

_HYDE_SYS = """Uzman bir akademik yazarsın. Bir araştırma sorusu verildiğinde,
bu soruyu yanıtlayan bir makalede *yer alabilecek* tek ve öz bir paragraf
(3-5 cümle) yaz. Teknik terminoloji kullan. Kaçamak ya da çekinceli ifadeler
kullanma. Derlem ve gömme modeli İngilizce olduğundan paragrafı İngilizce yaz."""


class HyDE:
    """Bir sorgu dönüşüm katmanı olarak Varsayımsal Belge Gömmeleri."""

    def __init__(
        self,
        llm: LLMClient,
        encoder: SentenceEncoder,
        n_hypotheses: int = 1,
        temperature: float = 0.7,
    ) -> None:
        self.llm = llm
        self.encoder = encoder
        self.n_hypotheses = n_hypotheses
        self.temperature = temperature

    def hypothetical_documents(self, query: str) -> list[str]:
        return [
            self.llm.complete(
                [Message("system", _HYDE_SYS), Message("user", query)],
                max_tokens=300,
                temperature=self.temperature,
            )
            for _ in range(self.n_hypotheses)
        ]

    def embed_query(self, query: str) -> np.ndarray:
        """HyDE ile dönüştürülmüş sorgu gömmesini döndürür.

        Birden çok hipotez olduğunda gömmelerin ortalaması alınır — bu,
        beklenen varsayımsal gömmenin Monte-Carlo tahminidir ve
        halüsinasyon varyansını azaltır.
        """
        docs = self.hypothetical_documents(query)
        # Hipotez konudan saparsa olgusal geri çağırma yok olmasın diye
        # orijinal sorguyu da dâhil et.
        docs.append(query)
        vectors = self.encoder.encode(docs)
        mean = vectors.mean(axis=0)
        # Kosinüs anlamını korumak için yeniden normalize et.
        norm = np.linalg.norm(mean)
        return mean / norm if norm > 0 else mean
