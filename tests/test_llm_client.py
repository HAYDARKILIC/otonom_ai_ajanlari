"""LLM istemcilerinin gerçek SDK'larla, sahte bir HTTP katmanı üzerinden testleri.

Ağ çağrısı yapılmaz: ``httpx.MockTransport`` API'ye giden isteği yakalar ve
hazır bir yanıt döndürür. Böylece gönderilen gövde birebir doğrulanabilir.
"""

from __future__ import annotations

import json

import httpx
import pytest

from src.utils.llm_client import AnthropicClient, Message, OpenAIClient, build_client

anthropic = pytest.importorskip("anthropic")
openai = pytest.importorskip("openai")

# anthropic>=2 kendi HTTP kütüphanesi olarak `httpx2` kullanır; eski sürümler `httpx`.
try:
    import httpx2 as anthropic_httpx
except ImportError:  # pragma: no cover
    anthropic_httpx = httpx


def _anthropic_ok(text: str, with_thinking: bool = False) -> dict:
    content = [{"type": "text", "text": text}]
    if with_thinking:
        content.insert(0, {"type": "thinking", "thinking": "", "signature": "sig"})
    return {
        "id": "msg_1", "type": "message", "role": "assistant", "model": "m",
        "content": content, "stop_reason": "end_turn", "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


def _anthropic_client(handler) -> AnthropicClient:
    return AnthropicClient(
        model="claude-opus-5-5", api_key="test",
        http_client=anthropic_httpx.Client(transport=anthropic_httpx.MockTransport(handler)),
        max_retries=0,
    )


def test_anthropic_request_shape_and_text_extraction() -> None:
    seen: list[dict] = []

    def handler(request: anthropic_httpx.Request) -> anthropic_httpx.Response:
        seen.append(json.loads(request.content))
        return anthropic_httpx.Response(200, json=_anthropic_ok("merhaba", with_thinking=True))

    client = _anthropic_client(handler)
    out = client.complete(
        [Message("system", "sistem"), Message("user", "selam")], max_tokens=50, stop=["Observation:"]
    )
    assert out == "merhaba"  # thinking bloğu atlandı
    body = seen[0]
    assert body["system"] == "sistem"
    assert body["messages"] == [{"role": "user", "content": "selam"}]
    assert body["stop_sequences"] == ["Observation:"]


def test_anthropic_drops_temperature_after_400_and_remembers() -> None:
    seen: list[dict] = []

    def handler(request: anthropic_httpx.Request) -> anthropic_httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        if "temperature" in body:
            return anthropic_httpx.Response(400, json={
                "type": "error",
                "error": {"type": "invalid_request_error",
                          "message": "temperature is not supported for this model"},
            })
        return anthropic_httpx.Response(200, json=_anthropic_ok("tamam"))

    client = _anthropic_client(handler)
    assert client.complete([Message("user", "a")], temperature=0.7) == "tamam"
    assert client.complete([Message("user", "b")], temperature=0.7) == "tamam"
    # 1. çağrı: temperature ile (400) + onsuz yeniden deneme; 2. çağrı: doğrudan onsuz.
    assert ["temperature" in b for b in seen] == [True, False, False]


def test_anthropic_other_400_errors_are_not_swallowed() -> None:
    def handler(request: anthropic_httpx.Request) -> anthropic_httpx.Response:
        return anthropic_httpx.Response(400, json={
            "type": "error", "error": {"type": "invalid_request_error", "message": "max_tokens too large"},
        })

    with pytest.raises(anthropic.BadRequestError):
        _anthropic_client(handler).complete([Message("user", "x")])


def test_openai_request_shape() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "c", "object": "chat.completion", "created": 0, "model": "gpt-4o",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "selam"}}],
        })

    client = OpenAIClient(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert client.complete([Message("user", "x")]) == "selam"
    assert "stop" not in seen[0]  # stop=None gönderilmez


def test_missing_api_key_gives_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        build_client("anthropic")


def test_unknown_provider() -> None:
    with pytest.raises(ValueError):
        build_client("gemini")
