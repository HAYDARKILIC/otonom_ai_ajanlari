"""Parçalama stratejileri: ebeveyn/çocuk hiyerarşisi ve cümle penceresi.

Matematiksel motivasyon
-----------------------
Erişimci (retriever) bir belgeyi yalnızca parçalar aracılığıyla görür. Parça
geometrisini birbirine zıt iki baskı şekillendirir:

1. **Gömme sadakati.** Sentence-Transformer'lar kısa girdilerle eğitilmiştir;
   parça uzunluğu arttıkça bi-encoder temsilleri bozulur (Reimers ve
   Gurevych, 2019). Daha küçük parçalar ⇒ daha sıkı anlam.
2. **Bağlamsal yeterlilik.** Erişilen bir parça, LLM'in yanıt verebilmesi için
   yeterli çevre metni içermelidir; aksi hâlde üretici eksik bağlamı
   halüsinasyonla uydurur.

İki çözüm uygulanmıştır:

- **Hiyerarşik (ebeveyn-çocuk) parçalama.** Belge büyük *ebeveyn* parçalara
  (≈1024 token), her ebeveyn de daha küçük *çocuk* parçalara (≈256 token)
  bölünür. Erişimci kesinlik için çocukları gömer ve arar, ancak bağlam için
  LLM'e ebeveyn metnini döndürür. Bu, iki hedefi birbirinden ayırır.

- **Cümle penceresi erişimi.** Her cümle ayrı ayrı gömülür, ancak erişim
  anında isabetin etrafındaki önceki ve sonraki *k* cümle genişletilir.
  Erişim indeksi cümle ayrıntı düzeyindedir; sunulan bağlam ise pencere
  ayrıntı düzeyindedir.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-ZÇĞİÖŞÜ])")


def _split_sentences(text: str) -> list[str]:
    """Hafif bir düzenli ifadeyle cümlelere böler (NLTK bağımlılığı yok)."""
    parts = [s.strip() for s in _SENTENCE_SPLIT.split(text)]
    return [p for p in parts if p]


def document_key(doc_text: str, doc_metadata: dict[str, Any]) -> str:
    """Belge için kararlı bir anahtar: ``doc_id`` / ``arxiv_id`` ya da içerik özeti.

    Parça kimlikleri bu anahtardan türetilir; böylece aynı belge yeniden
    indekslendiğinde vektör deposundaki kayıtlar çoğalmak yerine üzerine yazılır
    (upsert idempotent olur).
    """
    for key in ("doc_id", "arxiv_id", "id"):
        value = doc_metadata.get(key)
        if value:
            return str(value)
    return hashlib.sha256(doc_text.encode("utf-8")).hexdigest()[:16]


@dataclass(slots=True)
class Chunk:
    """Erişilebilir bir metin parçası ve ebeveynini gösteren üst veriler."""

    chunk_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    parent_id: str | None = None  # çocuklarda ayarlanır; ebeveynlerde None


class HierarchicalChunker:
    """Belgeleri ebeveyn ve çocuk parçalara böler.

    Parametreler
    ------------
    parent_chars
        Bir ebeveyn parça için yaklaşık karakter bütçesi. Doğal belge yapısına
        saygı göstermek için ebeveynler paragraf sınırlarından bölünür.
    child_chars
        Bir çocuk parça için yaklaşık karakter bütçesi. Çocuklar her ebeveynin
        içinde cümle sınırlarından bölünür.
    child_overlap_chars
        Komşu çocuk parçalar arasındaki örtüşme. Erişim anında sınır etkisinden
        kaynaklanan yanlış negatifleri azaltır.
    """

    def __init__(
        self,
        parent_chars: int = 4000,
        child_chars: int = 1000,
        child_overlap_chars: int = 100,
    ) -> None:
        if parent_chars < child_chars:
            raise ValueError("parent_chars, child_chars değerinden büyük veya eşit olmalıdır.")
        if not 0 <= child_overlap_chars < child_chars:
            raise ValueError("child_overlap_chars 0 ile child_chars arasında olmalıdır.")
        self.parent_chars = parent_chars
        self.child_chars = child_chars
        self.child_overlap = child_overlap_chars

    def chunk(self, doc_text: str, doc_metadata: dict[str, Any]) -> tuple[list[Chunk], list[Chunk]]:
        """(ebeveynler, çocuklar) döndürür. Ebeveynler indekslenMEZ; çocuklar indekslenİR."""
        parents: list[Chunk] = []
        children: list[Chunk] = []
        doc_key = document_key(doc_text, doc_metadata)
        for p_idx, parent_text in enumerate(self._split_parents(doc_text)):
            parent_id = f"{doc_key}::p{p_idx}"
            parents.append(
                Chunk(
                    chunk_id=parent_id,
                    text=parent_text,
                    metadata=dict(doc_metadata) | {"level": "parent"},
                )
            )
            for c_idx, child_text in enumerate(self._split_children(parent_text)):
                children.append(
                    Chunk(
                        chunk_id=f"{parent_id}::c{c_idx}",
                        text=child_text,
                        metadata=dict(doc_metadata) | {"level": "child"},
                        parent_id=parent_id,
                    )
                )
        return parents, children

    def _split_parents(self, text: str) -> list[str]:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        out, buf = [], ""
        for para in paragraphs:
            if len(buf) + len(para) > self.parent_chars and buf:
                out.append(buf.strip())
                buf = para
            else:
                buf = f"{buf}\n\n{para}" if buf else para
        if buf:
            out.append(buf.strip())
        return out

    def _split_children(self, parent_text: str) -> list[str]:
        sentences = _split_sentences(parent_text)
        out, buf = [], ""
        for sent in sentences:
            if len(buf) + len(sent) > self.child_chars and buf:
                out.append(buf.strip())
                # buf'ın sonundaki örtüşmeyi bir sonraki çocuğa taşı.
                buf = buf[-self.child_overlap:].strip() + " " + sent
            else:
                buf = f"{buf} {sent}".strip()
        if buf:
            out.append(buf.strip())
        return out


class SentenceWindowChunker:
    """Cümleleri tek tek indeksler; erişim anında bir pencere genişletir.

    Veri alımında her cümle, üst verisi belgedeki konumunu saklayan bir parça
    olur. Erişimde hat, çevredeki ±pencere cümlelerini getirmek için
    :meth:`expand_window` yöntemini çağırır.
    """

    def __init__(self, window: int = 3) -> None:
        if window < 0:
            raise ValueError("window negatif olmamalıdır.")
        self.window = window

    def chunk(self, doc_text: str, doc_metadata: dict[str, Any]) -> list[Chunk]:
        sentences = _split_sentences(doc_text)
        doc_id = document_key(doc_text, doc_metadata)
        chunks: list[Chunk] = []
        for idx, sent in enumerate(sentences):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}::s{idx}",
                    text=sent,
                    metadata=dict(doc_metadata) | {
                        "doc_id": doc_id,
                        "sentence_idx": idx,
                        "total_sentences": len(sentences),
                    },
                )
            )
        return chunks

    def expand_window(
        self, hit: Chunk, all_chunks_by_doc: dict[str, list[Chunk]]
    ) -> str:
        """İsabet eden cümleyi ±pencere komşularıyla birleştirilmiş olarak döndürür."""
        doc_id = hit.metadata["doc_id"]
        idx = hit.metadata["sentence_idx"]
        peers = all_chunks_by_doc[doc_id]
        lo = max(0, idx - self.window)
        hi = min(len(peers), idx + self.window + 1)
        return " ".join(c.text for c in peers[lo:hi])
