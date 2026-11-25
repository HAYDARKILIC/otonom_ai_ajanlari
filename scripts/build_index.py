"""Bir JSONL derleminden hiyerarşik veya cümle penceresi vektör indeksi oluşturur.

Kullanım:
    python scripts/build_index.py \\
        --input data/processed/arxiv_rag.jsonl \\
        --strategy hierarchical \\
        --backend chroma \\
        --persist-dir data/vector_store/chroma
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from src.rag_engine.chunking import HierarchicalChunker, SentenceWindowChunker
from src.rag_engine.embeddings import SentenceEncoder
from src.rag_engine.parent_store import SQLiteParentStore
from src.rag_engine.pipeline import HierarchicalRAG, SentenceWindowRAG
from src.rag_engine.vector_stores import build_store

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command()
def main(
    input: Path = typer.Option(..., help="Belgeleri içeren JSONL dosyası."),
    strategy: str = typer.Option("hierarchical", help="'hierarchical' | 'sentence_window'"),
    backend: str = typer.Option("chroma", help="'chroma' | 'milvus'"),
    persist_dir: Path = typer.Option(Path("./data/vector_store/chroma"), help="Chroma veri dizini."),
    collection: str = typer.Option("arxiv_rag", help="Koleksiyon adı."),
    encoder_model: str = typer.Option("BAAI/bge-small-en-v1.5", help="Gömme modeli."),
    milvus_uri: str = typer.Option("http://localhost:19530", help="Milvus adresi (backend=milvus)."),
    milvus_token: str = typer.Option("", help="Milvus erişim belirteci (backend=milvus)."),
) -> None:
    if strategy not in ("hierarchical", "sentence_window"):
        raise typer.BadParameter(f"{strategy!r} stratejisi bilinmiyor.")
    if not input.is_file():
        raise typer.BadParameter(f"Girdi dosyası bulunamadı: {input}")

    encoder = SentenceEncoder(model_name=encoder_model)

    store_kwargs: dict[str, object]
    if backend == "chroma":
        store_kwargs = {"collection": collection, "persist_dir": str(persist_dir)}
    elif backend == "milvus":
        store_kwargs = {"uri": milvus_uri, "token": milvus_token,
                        "collection": collection, "dim": encoder.dim}
    else:
        raise typer.BadParameter(f"{backend!r} arka ucu bilinmiyor ('chroma' | 'milvus').")
    store = build_store(backend, **store_kwargs)
    parents: SQLiteParentStore | None = None

    if strategy == "hierarchical":
        # Ebeveynler kalıcı olarak saklanır; böylece HierarchicalRAG.from_config ile
        # başka bir süreçte yapılan sorgular da ebeveyn genişletmesinden yararlanır.
        parents = SQLiteParentStore(persist_dir / f"{collection}_parents.sqlite")
        rag = HierarchicalRAG(
            chunker=HierarchicalChunker(), encoder=encoder, child_store=store, parent_lookup=parents,
        )
        ingest = rag.ingest_document
    else:
        sw_rag = SentenceWindowRAG(chunker=SentenceWindowChunker(window=3), encoder=encoder, store=store)
        ingest = sw_rag.ingest_document

    total = 0
    try:
        with input.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise typer.BadParameter(f"{input}:{line_no} geçerli JSON değil: {exc}") from exc
                text = f"# {rec.get('title', '')}\n\n{rec.get('abstract', '')}"
                # Özet metin olarak zaten indekslenir; her parçanın üst verisine kopyalanmaz.
                meta = {k: v for k, v in rec.items() if k != "abstract"}
                total += ingest(text, metadata=meta)
    finally:
        if parents is not None:
            parents.close()
    typer.echo(f"{total} parça {backend}:{collection} içine indekslendi")


if __name__ == "__main__":
    app()
