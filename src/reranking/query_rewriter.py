"""LLM güdümlü sorgu yeniden yazma ve ayrıştırma.

Bir erişim sisteminin darboğazı, kullanıcı sorgusu ile indekslenmiş metin
arasındaki sözcüksel ve anlamsal boşluktur. İki dönüşüm bunu hafifletir:

1. **Yeniden yazma**: sorguyu derlemde geçme olasılığı daha yüksek
   terimlerle yeniden ifade et (ör. kısaltmaları aç, standart terminoloji ekle).
2. **Ayrıştırma**: çok adımlı (multi-hop) bir soruyu, birleşimleri daha kolay
   erişilebilen bağımsız tek adımlı alt sorulara böl.

Her ikisi de katı JSON çıktılı tek seferlik LLM çağrıları olarak uygulanmıştır.
"""

from __future__ import annotations

import json
import re

from src.utils.llm_client import LLMClient, Message

_REWRITE_SYS = """Akademik bir arama motoru için kullanıcı sorgularını yeniden yazıyorsun.
Şu özelliklere sahip tek bir yeniden ifade edilmiş sorgu üret:
- kısaltmaları tam biçimlerine açar
- gündelik ifadeleri teknik terminolojiyle değiştirir
- orijinal amacı değiştirmeden korur
- derlem İngilizce olduğundan İngilizce yazılır
YALNIZCA bir JSON nesnesi döndür: {"rewritten": "..."}"""

_DECOMPOSE_SYS = """Çok adımlı bir soruyu, her biri kendi başına yanıtlanabilen
bağımsız tek adımlı alt sorulara ayrıştırıyorsun.
YALNIZCA bir JSON nesnesi döndür: {"sub_questions": ["...", "...", ...]}
En fazla 5 alt soru kullan. Sorgu zaten tek adımlıysa, yalnızca
orijinal sorguyu içeren bir liste döndür."""


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict:
    match = _JSON_BLOCK.search(text)
    if not match:
        raise ValueError(f"LLM yanıtında JSON nesnesi bulunamadı: {text!r}")
    return json.loads(match.group(0))


class QueryRewriter:
    """Eşzamanlı, LLM destekli sorgu yeniden yazıcı / ayrıştırıcı."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def rewrite(self, query: str) -> str:
        resp = self.llm.complete(
            [Message("system", _REWRITE_SYS), Message("user", query)],
            max_tokens=256,
            temperature=0.0,
        )
        try:
            return _extract_json(resp)["rewritten"]
        except (KeyError, ValueError):
            return query  # yumuşak başarısızlık: orijinal sorgu

    def decompose(self, query: str) -> list[str]:
        resp = self.llm.complete(
            [Message("system", _DECOMPOSE_SYS), Message("user", query)],
            max_tokens=512,
            temperature=0.0,
        )
        try:
            sq = _extract_json(resp)["sub_questions"]
            if isinstance(sq, list) and all(isinstance(s, str) for s in sq):
                return sq
        except (KeyError, ValueError):
            pass
        return [query]
