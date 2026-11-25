"""Uçtan uca Akademik Araştırma Asistanı.

Hat
---
    1. arXiv'den veri alma  → :class:`PaperStore`
    2. Özetleri indeksleme  → :class:`HierarchicalRAG`
    3. Rapor ana hattındaki her bölüm için:
       a. Bi-Encoder ile ilk K'yı getir
       b. Cross-Encoder ile ilk k'ya yeniden sırala
       c. LLM ile Markdown sentezle
    4. Bir Markdown literatür taramasında birleştir.

Not: Derlem ve gömme modeli İngilizce olduğundan bölüm başlıkları erişim
sorgusu olarak İngilizce üretilir; rapor metni ise Türkçe yazılır.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from src.academic_assistant.arxiv_ingest import Paper, fetch_papers
from src.academic_assistant.store import PaperStore
from src.rag_engine.chunking import HierarchicalChunker
from src.rag_engine.embeddings import SentenceEncoder
from src.rag_engine.parent_store import SQLiteParentStore
from src.rag_engine.pipeline import HierarchicalRAG
from src.rag_engine.vector_stores import build_store
from src.reranking.cross_encoder import CrossEncoderReranker, RerankedDocument
from src.reranking.pipeline import Reranker, RerankingPipeline
from src.utils.llm_client import LLMClient, Message, build_client
from src.utils.logging import get_logger

log = get_logger(__name__)


_OUTLINE_SYS = """Bir literatür taraması ana hattı tasarlıyorsun. Bir araştırma konusu
verildiğinde, tutarlı bir teknik hikâye anlatacak şekilde sıralanmış 4-6 bölüm
başlığı üret. Başlıklar İngilizce bir derlemde arama sorgusu olarak
kullanılacağından başlıkları İngilizce yaz. Her satıra bir bölüm başlığı yaz.
Numaralandırma. Giriş ya da kapanış metni ekleme."""

_SYNTHESIS_SYS = """Bir literatür taramasının bir bölümünü yazıyorsun.
Sana şunlar verilecek:
  - bölüm başlığı
  - her biri (başlık, özet) içeren, erişilmiş makale alıntıları
Alıntıları sentezleyen tek bir Markdown bölümü (## başlık + 2-4 paragraf) yaz.
Bölümü ve başlığını Türkçe yaz. Mümkün olduğunda makalelere metin içinde
[Yazar vd., Yıl] biçiminde atıf yap.
Atıf UYDURMA. Yalnızca alıntıların gerçekte söylediklerine bağlı kal."""


@dataclass(slots=True)
class AssistantConfig:
    topic: str
    max_papers: int = 25
    sections: int = 5
    candidate_k: int = 30
    rerank_k: int = 6
    persist_dir: str = "./data"


def _topic_key(topic: str) -> str:
    """Konudan süreçler arasında kararlı bir koleksiyon anahtarı türetir.

    Python'un yerleşik ``hash()`` fonksiyonu her süreçte rastgele tohumlanır
    (PYTHONHASHSEED); onu kullanmak aynı konu için her çalıştırmada yeni bir
    koleksiyon açılmasına ve eski koleksiyonların birikmesine yol açar.
    """
    return hashlib.sha256(topic.strip().lower().encode("utf-8")).hexdigest()[:12]


class AcademicAssistant:
    """Veri alma, RAG, yeniden sıralama ve LLM sentezini birleştirir."""

    def __init__(
        self,
        llm: LLMClient | None = None,
        *,
        encoder: SentenceEncoder | None = None,
        reranker: Reranker | None = None,
        paper_source: Callable[[str, int], Iterable[Paper]] | None = None,
    ) -> None:
        """Bileşenler isteğe bağlı olarak dışarıdan verilebilir (testler, özel modeller).

        Verilmeyenler ilk kullanımda varsayılanlarla oluşturulur: ``LLM_PROVIDER``
        ortam değişkenine göre bir LLM istemcisi, ``bge-small-en-v1.5`` kodlayıcısı,
        ``ms-marco-MiniLM`` cross-encoder'ı ve canlı arXiv API'si.
        """
        provider = os.environ.get("LLM_PROVIDER", "anthropic")
        self.llm = llm or build_client(provider)
        self.encoder = encoder
        self.reranker = reranker
        self.paper_source = paper_source or (lambda q, n: fetch_papers(q, max_results=n))

    def run(self, cfg: AssistantConfig) -> str:
        # 1. Veri alma
        papers = list(self.paper_source(cfg.topic, cfg.max_papers))
        log.info("assistant_ingest_done", count=len(papers))
        title = f"# {cfg.topic.strip()}: Bir Literatür Taraması\n"
        if not papers:
            return title + "\n_arXiv'de bu konu için makale bulunamadı; farklı bir konu deneyin._\n"
        store = PaperStore(Path(cfg.persist_dir) / "papers.sqlite")
        try:
            store.upsert_many(papers)
        finally:
            store.close()

        # 2. İndeksleme
        collection = f"assistant_{_topic_key(cfg.topic)}"
        vector_dir = Path(cfg.persist_dir) / "vector_store"
        with SQLiteParentStore(vector_dir / f"{collection}_parents.sqlite") as parents:
            rag = HierarchicalRAG(
                chunker=HierarchicalChunker(parent_chars=4000, child_chars=800),
                encoder=self.encoder or SentenceEncoder(),
                child_store=build_store(kind="chroma", collection=collection, persist_dir=str(vector_dir)),
                parent_lookup=parents,
            )
            return self._index_and_write(cfg, papers, rag, title)

    def _index_and_write(
        self, cfg: AssistantConfig, papers: list[Paper], rag: HierarchicalRAG, title: str,
    ) -> str:
        """Makaleleri indeksler, ana hattı kurar ve her bölümü sentezler (adım 2–4)."""
        for p in papers:
            text = f"# {p.title}\n\n{p.abstract}"
            rag.ingest_document(text, metadata={
                "arxiv_id": p.arxiv_id, "title": p.title,
                "authors": "; ".join(p.authors), "published": p.published,
            })

        reranker = RerankingPipeline(
            reranker=self.reranker or CrossEncoderReranker(), candidate_k=cfg.candidate_k,
        )

        # 3. Ana hat
        outline = self._build_outline(cfg.topic, cfg.sections)
        log.info("assistant_outline_built", sections=len(outline))

        # 4. Her bölümü sentezle
        md_sections: list[str] = [title]
        for section in outline:
            candidates = rag.retrieve(section, k=cfg.candidate_k)
            top = reranker.rerank(section, candidates, k=cfg.rerank_k)
            if not top:
                # Derlem bu bölümü kapsamıyor: LLM'i kanıtsız yazmaya zorlamak yerine işaretle.
                md_sections.append(f"## {section}\n\n_Bu bölüm için ilgili alıntı bulunamadı._")
                continue
            md_sections.append(self._synthesize_section(section, top))

        md_sections.append(self._references_section(papers))
        return "\n\n".join(md_sections)

    # ------------------------------------------------------------------ #
    # İç işleyiş                                                         #
    # ------------------------------------------------------------------ #

    def _build_outline(self, topic: str, n_sections: int) -> list[str]:
        prompt = f"Konu: {topic}\nTam olarak {n_sections} bölüm başlığı üret."
        text = self.llm.complete(
            [Message("system", _OUTLINE_SYS), Message("user", prompt)],
            max_tokens=300, temperature=0.4,
        )
        sections = []
        for line in text.splitlines():
            # LLM talimata rağmen numara, madde imi veya başlık işareti ekleyebilir.
            cleaned = re.sub(r"^\s*(?:#+|[-•*]|\d+[.)])\s*", "", line).strip().strip("*").strip()
            if cleaned:
                sections.append(cleaned)
        return sections[:n_sections] if sections else [topic]

    def _synthesize_section(self, section_title: str, reranked: Sequence[RerankedDocument]) -> str:
        excerpts = "\n\n".join(
            f"- {d.metadata.get('title', 'başlıksız')} "
            f"({d.metadata.get('published', '?')[:4]})\n  {d.text[:600]}"
            for d in reranked
        )
        user = f"BÖLÜM: {section_title}\n\nALINTILAR:\n{excerpts}"
        return self.llm.complete(
            [Message("system", _SYNTHESIS_SYS), Message("user", user)],
            max_tokens=1500, temperature=0.3,
        )

    @staticmethod
    def _references_section(papers: Sequence[Paper]) -> str:
        lines = ["## Kaynakça", ""]
        for p in papers:
            authors = "; ".join(p.authors[:3]) + (" vd." if len(p.authors) > 3 else "")
            lines.append(f"- **{p.title}** — {authors} ({p.published}). [{p.arxiv_id}]({p.pdf_url})")
        return "\n".join(lines)
