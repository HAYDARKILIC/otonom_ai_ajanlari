"""Gerçek bir Chroma deposuyla uçtan uca testler (model indirmesi gerektirmez).

Gömme modeli yerine ``conftest.HashingEncoder``, LLM yerine senaryolu
``FakeLLM`` kullanılır. Böylece RAG hattı, yeniden sıralama, betikler,
akademik asistan ve çoklu ajan orkestratörü gerçek bileşenlerle çalıştırılır.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

chromadb = pytest.importorskip("chromadb")

from typer.testing import CliRunner  # noqa: E402

from src.academic_assistant.arxiv_ingest import Paper  # noqa: E402
from src.academic_assistant.pipeline import AcademicAssistant, AssistantConfig  # noqa: E402
from src.multi_agent.orchestrator import SelfCorrectingCoder  # noqa: E402
from src.rag_engine.chunking import HierarchicalChunker, SentenceWindowChunker  # noqa: E402
from src.rag_engine.parent_store import SQLiteParentStore  # noqa: E402
from src.rag_engine.pipeline import HierarchicalRAG, RetrievedDocument, SentenceWindowRAG  # noqa: E402
from src.rag_engine.vector_stores import build_store  # noqa: E402
from src.reranking.cross_encoder import RerankedDocument  # noqa: E402
from src.reranking.hyde import HyDE  # noqa: E402
from src.reranking.pipeline import RerankingPipeline  # noqa: E402
from src.reranking.query_rewriter import QueryRewriter  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]

_DOCS = {
    "hyde": "HyDE generates a hypothetical answer document and embeds it for retrieval.\n\n"
            "The synthetic document lies closer to real answers than the original query does.",
    "rerank": "Cross encoders score query document pairs jointly with transformer attention.\n\n"
              "They are slower than bi encoders but substantially more accurate.",
    "hnsw": "HNSW is a graph based approximate nearest neighbour index.\n\n"
            "It maintains a hierarchy of small world graphs with logarithmic search time.",
}


class _WordOverlapReranker:
    """Cross-Encoder yerine: sorguyla ortak sözcük sayısına göre sıralar."""

    def __init__(self) -> None:
        self.seen_candidates: list[int] = []

    def rerank(self, query, candidates, k=None):
        self.seen_candidates.append(len(candidates))
        q = set(query.lower().split())
        scored = sorted(
            (RerankedDocument(c.chunk_id, c.text, c.score, float(len(q & set(c.text.lower().split()))),
                              c.metadata) for c in candidates),
            key=lambda d: d.cross_encoder_score, reverse=True,
        )
        return scored[:k] if k is not None else scored


def _hier_rag(tmp_path: Path, encoder, parents) -> HierarchicalRAG:
    return HierarchicalRAG(
        chunker=HierarchicalChunker(parent_chars=400, child_chars=120, child_overlap_chars=20),
        encoder=encoder,
        child_store=build_store("chroma", collection="test_coll", persist_dir=str(tmp_path / "chroma")),
        parent_lookup=parents,
    )


# ------------------------------------ RAG ------------------------------------- #

def test_hierarchical_rag_retrieves_and_expands_to_parent(tmp_path: Path, hashing_encoder) -> None:
    rag = _hier_rag(tmp_path, hashing_encoder, None)
    for doc_id, text in _DOCS.items():
        rag.ingest_document(text, {"doc_id": doc_id})
    hits = rag.retrieve("hierarchy of small world graphs", k=3)
    assert hits[0].metadata["doc_id"] == "hnsw"
    assert "logarithmic" in hits[0].text  # ebeveyn metni döndü


def test_reingesting_is_idempotent(tmp_path: Path, hashing_encoder) -> None:
    rag = _hier_rag(tmp_path, hashing_encoder, None)
    rag.ingest_document(_DOCS["hyde"], {"doc_id": "hyde"})
    coll = chromadb.PersistentClient(path=str(tmp_path / "chroma")).get_collection("test_coll")
    first = coll.count()
    rag.ingest_document(_DOCS["hyde"], {"doc_id": "hyde"})
    assert coll.count() == first  # kayıtlar çoğalmadı


def test_parents_survive_a_new_process(tmp_path: Path, hashing_encoder) -> None:
    """İndeksi bir örnekle kurup tamamen yeni bir örnekle (yeni süreç gibi) sorgula."""
    parents_path = tmp_path / "parents.sqlite"
    writer = _hier_rag(tmp_path, hashing_encoder, SQLiteParentStore(parents_path))
    for doc_id, text in _DOCS.items():
        writer.ingest_document(text, {"doc_id": doc_id})

    reader = _hier_rag(tmp_path, hashing_encoder, SQLiteParentStore(parents_path))
    top = reader.retrieve("synthetic document closer to real answers", k=1)[0]
    assert top.metadata["doc_id"] == "hyde"
    assert top.text.startswith("HyDE generates")  # tam ebeveyn, yalnızca çocuk değil


def test_from_config_uses_persistent_parent_store(tmp_path: Path, hashing_encoder, monkeypatch) -> None:
    import src.rag_engine.pipeline as pipeline_mod

    monkeypatch.setattr(pipeline_mod, "SentenceEncoder", lambda **_: hashing_encoder)
    cfg = tmp_path / "rag.yaml"
    cfg.write_text(
        "chunker: {parent_chars: 400, child_chars: 120, child_overlap_chars: 20}\n"
        "encoder: {model: x, device: null}\n"
        f"vector_store: {{kind: chroma, collection: cfg_coll, persist_dir: '{tmp_path / 'vs'}'}}\n",
        encoding="utf-8",
    )
    rag = HierarchicalRAG.from_config(str(cfg))
    assert isinstance(rag.parent_lookup, SQLiteParentStore)
    rag.ingest_document(_DOCS["rerank"], {"doc_id": "rerank"})
    assert (tmp_path / "vs" / "cfg_coll_parents.sqlite").is_file()
    again = HierarchicalRAG.from_config(str(cfg))
    assert "substantially more accurate" in again.retrieve("cross encoders attention", k=1)[0].text


def test_sentence_window_expands_from_store_in_new_instance(tmp_path: Path, hashing_encoder) -> None:
    def make() -> SentenceWindowRAG:
        return SentenceWindowRAG(
            chunker=SentenceWindowChunker(window=1), encoder=hashing_encoder,
            store=build_store("chroma", collection="sw_coll", persist_dir=str(tmp_path / "sw")),
        )

    text = "Alpha one starts here. Beta two follows next. Gamma three is unique. Delta four ends it."
    make().ingest_document(text, {"doc_id": "d"})
    fresh = make()  # bellek içi önbellek boş: komşular depodan gelmeli
    hit = fresh.retrieve("gamma three unique", k=1)[0]
    assert hit.text == "Beta two follows next. Gamma three is unique. Delta four ends it."
    assert make().ingest_document("", {"doc_id": "bos"}) == 0  # boş belge çökmez


def test_query_on_empty_collection_returns_empty(tmp_path: Path, hashing_encoder) -> None:
    assert _hier_rag(tmp_path, hashing_encoder, None).retrieve("anything", k=5) == []


# ----------------------------- Yeniden sıralama -------------------------------- #

def test_reranking_pipeline_respects_candidate_k_and_rewriter(fake_llm_factory) -> None:
    rer = _WordOverlapReranker()
    rewriter = QueryRewriter(fake_llm_factory(['{"rewritten": "graph index"}']))
    pipe = RerankingPipeline(reranker=rer, query_rewriter=rewriter, candidate_k=2)
    cands = [RetrievedDocument(str(i), t, 0.0, {}) for i, t in
             enumerate(["unrelated", "graph index here", "nothing", "more"])]
    out = pipe.rerank("orijinal sorgu", cands, k=1)
    assert rer.seen_candidates == [2]          # yalnızca ilk candidate_k aday puanlandı
    assert out[0].text == "graph index here"   # yeniden yazılmış sorguyla sıralandı


def test_query_rewriter_falls_back_on_bad_json(fake_llm_factory) -> None:
    rw = QueryRewriter(fake_llm_factory(["JSON yok", "yine yok"]))
    assert rw.rewrite("q") == "q"
    assert rw.decompose("q") == ["q"]


def test_hyde_returns_unit_vector(fake_llm_factory, hashing_encoder) -> None:
    import numpy as np

    hyde = HyDE(fake_llm_factory(["a b c", "c d e"]), hashing_encoder, n_hypotheses=2)
    vec = hyde.embed_query("soru")
    assert vec.shape == (hashing_encoder.dim,)
    assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-5


# ---------------------------------- Betikler ----------------------------------- #

def test_build_index_script_end_to_end(tmp_path: Path, hashing_encoder, monkeypatch) -> None:
    import scripts.build_index as build_index

    monkeypatch.setattr(build_index, "SentenceEncoder", lambda **_: hashing_encoder)
    corpus = tmp_path / "c.jsonl"
    corpus.write_text("\n".join(json.dumps({
        "arxiv_id": k, "title": k.upper(), "abstract": v, "authors": ["A", "B"], "categories": ["cs.IR"],
    }) for k, v in _DOCS.items()) + "\n", encoding="utf-8")
    args = ["--input", str(corpus), "--persist-dir", str(tmp_path / "vs"), "--collection", "idx_x"]
    runner = CliRunner()
    first = runner.invoke(build_index.app, args)
    assert first.exit_code == 0, first.output
    assert runner.invoke(build_index.app, args).exit_code == 0   # ikinci çalıştırma
    coll = chromadb.PersistentClient(path=str(tmp_path / "vs")).get_collection("idx_x")
    n_first = int(first.output.split()[0])
    assert coll.count() == n_first                               # kopya yok
    assert (tmp_path / "vs" / "idx_x_parents.sqlite").is_file()
    sw = runner.invoke(build_index.app, args + ["--strategy", "sentence_window", "--collection", "idx_y"])
    assert sw.exit_code == 0, sw.output


def test_ingest_script_writes_jsonl(tmp_path: Path, monkeypatch) -> None:
    import scripts.ingest_arxiv as ingest

    paper = Paper("1234.5678", "Başlık", ["A"], "Özet", "2024-01-01", "http://x", ["cs.CL"])
    monkeypatch.setattr(ingest, "fetch_papers", lambda q, max_results: iter([paper]))
    out = tmp_path / "o.jsonl"
    res = CliRunner().invoke(ingest.app, ["--query", "q", "--output", str(out)])
    assert res.exit_code == 0, res.output
    assert json.loads(out.read_text(encoding="utf-8"))["arxiv_id"] == "1234.5678"


def test_benchmark_script_runs_on_bundled_fixture(hashing_encoder, monkeypatch) -> None:
    import scripts.benchmark_retrieval as bench

    monkeypatch.setattr("src.rag_engine.embeddings.SentenceEncoder", lambda **_: hashing_encoder)
    fx = _REPO / "tests" / "fixtures" / "beir_toy"
    res = CliRunner().invoke(bench.app, ["--corpus", str(fx / "corpus.jsonl"),
                                         "--queries", str(fx / "queries.jsonl")])
    assert res.exit_code == 0, res.output
    assert "Sorgu sayısı = 50" in res.output
    recall = float(res.output.split("Recall@10 = ")[1].split()[0])
    assert recall > 0.5  # sözcük örtüşmesi bile çoğu ilgili belgeyi bulmalı


def test_ndcg_and_recall_math() -> None:
    from scripts.benchmark_retrieval import ndcg_at_k, recall_at_k

    assert ndcg_at_k(["a", "b"], {"a": 1}, 10) == 1.0
    assert ndcg_at_k(["b", "a"], {"a": 1}, 10) == pytest.approx(1 / 1.5849625, rel=1e-6)
    assert recall_at_k(["a", "x"], {"a", "b"}, 2) == 0.5
    assert recall_at_k([], set(), 5) == 0.0


# ------------------------------ Akademik asistan ------------------------------- #

def test_academic_assistant_end_to_end(tmp_path: Path, fake_llm_factory, hashing_encoder) -> None:
    papers = [
        Paper("2212.10496", "HyDE retrieval", ["Gao", "Ma", "Lin", "Callan"], _DOCS["hyde"],
              "2022-12-20", "https://arxiv.org/pdf/2212.10496", ["cs.IR"]),
        Paper("1908.10084", "Cross encoder study", ["Reimers"], _DOCS["rerank"],
              "2019-08-27", "https://arxiv.org/pdf/1908.10084", ["cs.CL"]),
    ]
    llm = fake_llm_factory([
        "1. Hypothetical document embeddings\n- Cross encoders attention",   # ana hat (numaralı!)
        "## HyDE\n\nMetin [Gao vd., 2022].",
        "## Cross-Encoder\n\nMetin [Reimers, 2019].",
    ])
    assistant = AcademicAssistant(
        llm=llm, encoder=hashing_encoder, reranker=_WordOverlapReranker(),
        paper_source=lambda q, n: papers[:n],
    )
    md = assistant.run(AssistantConfig(topic="yoğun erişim", max_papers=5, sections=2,
                                       persist_dir=str(tmp_path)))
    assert md.startswith("# yoğun erişim: Bir Literatür Taraması")
    assert "Metin [Gao vd., 2022]." in md and "## Kaynakça" in md
    assert "Gao; Ma; Lin vd." in md
    section_prompt = llm.calls[1][1].content
    assert section_prompt.startswith("BÖLÜM: Hypothetical document embeddings")  # numara temizlendi
    assert (tmp_path / "papers.sqlite").is_file()


def test_academic_assistant_handles_no_papers(tmp_path: Path, fake_llm_factory, hashing_encoder) -> None:
    assistant = AcademicAssistant(llm=fake_llm_factory([]), encoder=hashing_encoder,
                                  reranker=_WordOverlapReranker(), paper_source=lambda q, n: [])
    md = assistant.run(AssistantConfig(topic="yok", persist_dir=str(tmp_path)))
    assert "makale bulunamadı" in md


# -------------------------------- Çoklu ajan ----------------------------------- #

def test_self_correcting_coder_with_injected_llms(fake_llm_factory) -> None:
    buggy = "```python\nassert 1 + 1 == 3\n```"
    fixed = "```python\nassert 1 + 1 == 2\nprint('OK')\n```"
    coder = SelfCorrectingCoder(
        max_iterations=3,
        coder_llm=fake_llm_factory([buggy, fixed]),
        critic_llm=fake_llm_factory(["1 + 1 ikidir."]),
    )
    result = coder.solve("toplama")
    assert result.success and result.iterations == 2 and result.halted_reason == "success"


def test_last_iteration_does_not_waste_llm_calls(fake_llm_factory) -> None:
    coder_llm = fake_llm_factory(["```python\nraise SystemExit(1)\n```",
                                  "```python\nraise SystemExit(2)\n```"])
    critic_llm = fake_llm_factory(["düzelt"])
    coder = SelfCorrectingCoder(max_iterations=2, coder_llm=coder_llm, critic_llm=critic_llm)
    result = coder.solve("t")
    assert result.halted_reason == "max_iterations"
    assert len(coder_llm.calls) == 2 and len(critic_llm.calls) == 1  # son turda revizyon yok


def test_invalid_max_iterations(fake_llm_factory) -> None:
    with pytest.raises(ValueError):
        SelfCorrectingCoder(max_iterations=0, llm=fake_llm_factory([]))
