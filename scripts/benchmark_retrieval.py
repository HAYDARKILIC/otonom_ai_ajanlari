"""Küçük bir BEIR dilimi üzerinde erişim recall@k ve yeniden sıralama nDCG@k kıyaslaması.

Bu duman testi (smoke test) sürümüdür: ``tests/fixtures/beir_toy/`` altında 50
sorguluk bir oyuncak dilimle gelir. Tam BEIR değerlendirmesini çalıştırmak için
``--corpus`` ve ``--queries`` seçeneklerini resmî BEIR sürümüne yönlendirin.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import typer

app = typer.Typer(no_args_is_help=True, add_completion=False)


def ndcg_at_k(ranked_ids: list[str], relevance: dict[str, int], k: int) -> float:
    dcg = 0.0
    for i, doc_id in enumerate(ranked_ids[:k]):
        rel = relevance.get(doc_id, 0)
        dcg += (2 ** rel - 1) / math.log2(i + 2)
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant) / len(relevant)


@app.command()
def main(
    corpus: Path = typer.Option(..., help="{id, text} belgelerinden oluşan JSONL."),
    queries: Path = typer.Option(..., help="{qid, query, relevant: [ids]} kayıtlarından oluşan JSONL."),
    k: int = typer.Option(10, help="Değerlendirme kesme noktası."),
    encoder_model: str = typer.Option("BAAI/bge-small-en-v1.5", help="Gömme modeli."),
) -> None:
    from src.rag_engine.chunking import HierarchicalChunker
    from src.rag_engine.embeddings import SentenceEncoder
    from src.rag_engine.pipeline import HierarchicalRAG
    from src.rag_engine.vector_stores import build_store

    for path in (corpus, queries):
        if not path.is_file():
            raise typer.BadParameter(f"Dosya bulunamadı: {path}")

    encoder = SentenceEncoder(model_name=encoder_model)
    # Her çalıştırma temiz, geçici bir depoda yapılır; önceki çalıştırmaların
    # kayıtları sonuçlara karışmaz.
    with tempfile.TemporaryDirectory() as store_dir:
        rag = HierarchicalRAG(
            chunker=HierarchicalChunker(),
            encoder=encoder,
            child_store=build_store("chroma", collection="bench", persist_dir=store_dir),
        )
        with corpus.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rec = json.loads(line)
                    rag.ingest_document(rec["text"], metadata={"doc_id": rec["id"]})

        recall, ndcg, n = 0.0, 0.0, 0
        with queries.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                q = json.loads(line)
                # Bir belge birden çok ebeveyn üretebilir: belge düzeyinde tekilleştir.
                hits = rag.retrieve(q["query"], k=k * 3)
                ranked = list(dict.fromkeys(h.metadata.get("doc_id", h.chunk_id) for h in hits))[:k]
                recall += recall_at_k(ranked, set(q["relevant"]), k)
                ndcg += ndcg_at_k(ranked, {i: 1 for i in q["relevant"]}, k)
                n += 1

    typer.echo(f"Sorgu sayısı = {n}")
    typer.echo(f"Recall@{k} = {recall / max(n, 1):.4f}")
    typer.echo(f"nDCG@{k}   = {ndcg / max(n, 1):.4f}")


if __name__ == "__main__":
    app()
