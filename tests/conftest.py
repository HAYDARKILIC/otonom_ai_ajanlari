"""Ortak pytest fikstürleri.

Birim testlerinde gerçek sağlayıcının yerini deterministik bir :class:`FakeLLM`
alır; böylece test paketi yalıtılmış, ücretsiz ve tekrarlanabilir olur.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from src.utils.llm_client import Message


class FakeLLM:
    """Senaryolu bir LLM: hazır yanıtları sırayla döndürür."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[list[Message]] = []

    def complete(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        stop: list[str] | None = None,
    ) -> str:
        self.calls.append(list(messages))
        if not self._responses:
            raise AssertionError("FakeLLM'in senaryolu yanıtları tükendi.")
        return self._responses.pop(0)


@pytest.fixture
def fake_llm_factory() -> Iterator[type[FakeLLM]]:
    yield FakeLLM


class HashingEncoder:
    """``SentenceEncoder`` ile aynı arayüze sahip, model gerektirmeyen deterministik kodlayıcı.

    Her sözcüğü sabit bir boyuta karma (hash) ile eşler ve vektörü L2-normalize
    eder. Anlamsal değildir ama sözcük örtüşmesini yakalar; bu, model indirmeden
    erişim hatlarını gerçek bir vektör deposuyla uçtan uca test etmeye yeter.
    """

    model_name = "hashing-test-encoder"

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def encode(self, texts: list[str], batch_size: int = 32):  # noqa: ARG002
        import hashlib
        import re

        import numpy as np

        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"\w+", text.lower()):
                h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
                out[row, h % self.dim] += 1.0
            norm = np.linalg.norm(out[row])
            if norm > 0:
                out[row] /= norm
        return out

    def encode_one(self, text: str):
        return self.encode([text])[0]


@pytest.fixture
def hashing_encoder() -> HashingEncoder:
    return HashingEncoder()
