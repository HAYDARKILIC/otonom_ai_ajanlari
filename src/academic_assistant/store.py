"""Akademik Asistan'ın getirdiği makaleler için SQLite kalıcılık katmanı.

MCP sunucusu bu veritabanını LLM istemcilerine salt okunur olarak sunar
(bkz. :mod:`mcp_core.resources.SQLiteResources`); asistan ise bu modül
aracılığıyla doğrudan yazar.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from src.academic_assistant.arxiv_ingest import Paper

_SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    arxiv_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    authors TEXT NOT NULL,
    abstract TEXT NOT NULL,
    published TEXT NOT NULL,
    pdf_url TEXT NOT NULL,
    categories TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_papers_published ON papers(published);
"""


class PaperStore:
    """papers tablosu için ince bir SQLite sarmalayıcısı."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def upsert(self, paper: Paper) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO papers
               (arxiv_id, title, authors, abstract, published, pdf_url, categories)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                paper.arxiv_id, paper.title,
                "; ".join(paper.authors), paper.abstract,
                paper.published, paper.pdf_url,
                "; ".join(paper.categories),
            ),
        )
        self._conn.commit()

    def upsert_many(self, papers: list[Paper]) -> int:
        for p in papers:
            self.upsert(p)
        return len(papers)

    def all_papers(self) -> list[dict]:
        cur = self._conn.execute("SELECT * FROM papers ORDER BY published DESC")
        return [dict(row) for row in cur.fetchall()]

    def close(self) -> None:
        self._conn.close()
