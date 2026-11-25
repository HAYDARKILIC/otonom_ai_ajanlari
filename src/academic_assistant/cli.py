"""Bitirme projesi Akademik Asistan için komut satırı giriş noktası.

Kullanım:

    python -m src.academic_assistant.cli \\
        --topic "self-rewarding language models" \\
        --max-papers 25 \\
        --output reports/review.md
"""

from __future__ import annotations

from pathlib import Path

import typer

from src.academic_assistant.pipeline import AcademicAssistant, AssistantConfig

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command()
def run(
    topic: str = typer.Option(..., help="İncelenecek araştırma konusu."),
    max_papers: int = typer.Option(25, help="Alınacak arXiv makalesi sayısı."),
    sections: int = typer.Option(5, help="İnceleme ana hattındaki bölüm sayısı."),
    rerank_k: int = typer.Option(6, help="Bölüm başına yeniden sıralayıcı ilk-k değeri."),
    output: Path = typer.Option(Path("reports/review.md"), help="Çıktı Markdown dosyasının yolu."),
    persist_dir: Path = typer.Option(Path("./data"), help="Önbellekler için çalışma dizini."),
) -> None:
    cfg = AssistantConfig(
        topic=topic,
        max_papers=max_papers,
        sections=sections,
        rerank_k=rerank_k,
        persist_dir=str(persist_dir),
    )
    assistant = AcademicAssistant()
    markdown = assistant.run(cfg)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
    typer.echo(f"{output} dosyasına {len(markdown):,} karakter yazıldı")


if __name__ == "__main__":
    app()
