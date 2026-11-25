"""Sağlayıcıdan bağımsız LLM istemcisi.

Alttaki sağlayıcının OpenAI mı yoksa Anthropic mi olduğunu gizleyen ince bir
soyutlama. Ajan ve yeniden sıralama kodu yalnızca :class:`LLMClient`
Protocol'üne bağımlıdır; testler deterministik bir sahte istemci enjekte eder.

Örnekleme parametreleri hakkında not
------------------------------------
Claude Opus 4.7 ve sonraki modeller ``temperature`` / ``top_p`` / ``top_k``
parametrelerini desteklemez; varsayılan olmayan bir değer gönderildiğinde API
400 hatası döndürür. :class:`AnthropicClient` bu nedenle ``temperature``
değerini yalnızca model kabul ediyorsa gönderir: ilk 400 hatasında parametreyi
düşürüp isteği yeniden dener ve bu kararı istemci ömrü boyunca hatırlar. Bu
modellerde çeşitlilik (ör. HyDE'nin birden çok hipotezi) yalnızca modelin
kendi örneklemesinden gelir.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol

DEFAULT_ANTHROPIC_MODEL = "claude-opus-5-5"
DEFAULT_OPENAI_MODEL = "gpt-4o"


@dataclass(slots=True)
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str


class LLMClient(Protocol):
    """Her somut LLM istemcisinin karşılaması gereken asgari arayüz."""

    def complete(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        stop: list[str] | None = None,
    ) -> str:
        """Asistanın yanıtını düz bir dize olarak döndürür."""
        ...


def _is_sampling_rejection(exc: Exception) -> bool:
    """400 hatası örnekleme parametresinin desteklenmemesinden mi kaynaklanıyor?"""
    text = str(exc).lower()
    return any(p in text for p in ("temperature", "top_p", "top_k", "sampling"))


class AnthropicClient:
    """`anthropic.Anthropic` etrafında ince bir sarmalayıcı."""

    def __init__(
        self,
        model: str = DEFAULT_ANTHROPIC_MODEL,
        api_key: str | None = None,
        **client_kwargs: Any,
    ) -> None:
        import anthropic

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY tanımlı değil; .env dosyanıza ekleyin veya api_key verin.")
        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=key, **client_kwargs)
        self._model = model
        self._send_temperature = True

    def complete(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        stop: list[str] | None = None,
    ) -> str:
        system_parts = [m.content for m in messages if m.role == "system"]
        api_msgs = [{"role": m.role, "content": m.content}
                    for m in messages if m.role != "system"]
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": api_msgs,
        }
        if system_parts:
            kwargs["system"] = "\n\n".join(system_parts)
        if stop:
            kwargs["stop_sequences"] = stop
        if self._send_temperature:
            # `extra_body` ile gönderilir: anthropic SDK 2.x `temperature` argümanını
            # imzadan kaldırdı, ancak `extra_body` tüm SDK sürümlerinde desteklenir.
            kwargs["extra_body"] = {"temperature": temperature}

        try:
            resp = self._client.messages.create(**kwargs)
        except self._anthropic.BadRequestError as exc:
            if "extra_body" not in kwargs or not _is_sampling_rejection(exc):
                raise
            # Model örnekleme parametrelerini desteklemiyor (Opus 4.7+): onsuz yeniden dene.
            self._send_temperature = False
            kwargs.pop("extra_body")
            resp = self._client.messages.create(**kwargs)

        # Yanıt metin dışı bloklar (ör. thinking) içerebilir; yalnızca metin bloklarını birleştir.
        return "".join(getattr(block, "text", "") for block in resp.content if block.type == "text")


class OpenAIClient:
    """`openai.OpenAI` etrafında ince bir sarmalayıcı."""

    def __init__(
        self,
        model: str = DEFAULT_OPENAI_MODEL,
        api_key: str | None = None,
        **client_kwargs: Any,
    ) -> None:
        import openai

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY tanımlı değil; .env dosyanıza ekleyin veya api_key verin.")
        self._client = openai.OpenAI(api_key=key, **client_kwargs)
        self._model = model

    def complete(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        stop: list[str] | None = None,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if stop:
            kwargs["stop"] = stop
        resp = self._client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""


def build_client(provider: str, model: str | None = None) -> LLMClient:
    """Fabrika: sağlayıcı adına göre yapılandırılmış bir istemci döndürür."""
    if provider == "anthropic":
        return AnthropicClient(model=model or DEFAULT_ANTHROPIC_MODEL)
    if provider == "openai":
        return OpenAIClient(model=model or DEFAULT_OPENAI_MODEL)
    raise ValueError(f"Bilinmeyen LLM sağlayıcısı: {provider!r} (geçerli: 'anthropic', 'openai')")
