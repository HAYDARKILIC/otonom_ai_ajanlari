"""arXiv'in herkese açık API'sinden üst verileri ve özetleri çeker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import arxiv


@dataclass(slots=True)
class Paper:
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    published: str
    pdf_url: str
    categories: list[str]


def fetch_papers(query: str, max_results: int = 25) -> Iterator[Paper]:
    """En üstteki ``max_results`` arXiv sonucu için :class:`Paper` nesneleri üretir."""
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    for result in arxiv.Client().results(search):
        yield Paper(
            arxiv_id=result.entry_id.rsplit("/", 1)[-1],
            title=result.title.strip().replace("\n", " "),
            authors=[a.name for a in result.authors],
            abstract=result.summary.strip().replace("\n", " "),
            published=result.published.date().isoformat(),
            pdf_url=result.pdf_url,
            categories=list(result.categories),
        )
