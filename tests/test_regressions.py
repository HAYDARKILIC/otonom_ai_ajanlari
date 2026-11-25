"""Düzeltilen hataların geri dönmemesi için regresyon testleri."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

from src.academic_assistant.arxiv_ingest import Paper
from src.academic_assistant.pipeline import _topic_key
from src.mcp_core.client import MCPStdioClient
from src.mcp_core.server import MCPServer
from src.mcp_core.tool_registry import Tool, ToolRegistry
from src.multi_agent.agents import ExecutorAgent
from src.multi_agent.state import AgentState
from src.rag_engine.chunking import SentenceWindowChunker
from src.rag_engine.vector_stores import _sanitize_metadata

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _rpc(server: MCPServer, method: str, params: dict | None = None, req_id: int | None = 1) -> dict | None:
    frame = {"jsonrpc": "2.0", "method": method, "params": params or {}}
    if req_id is not None:
        frame["id"] = req_id
    raw = server.handle(json.dumps(frame))
    return json.loads(raw) if raw else None


def _server_with_failing_tool() -> MCPServer:
    def boom() -> str:
        raise RuntimeError("araç patladı")

    tools = ToolRegistry()
    tools.register(Tool(
        name="boom", description="Her zaman hata verir.",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        handler=boom,
    ))
    server = MCPServer(name="t", version="0.1", tools=tools)
    _rpc(server, "initialize", {"protocolVersion": "2024-11-05"})
    return server


# --------------------------- MCP spesifikasyon uyumu --------------------------- #

def test_initialize_negotiates_version_and_uses_object_capabilities() -> None:
    server = _server_with_failing_tool()
    resp = _rpc(server, "initialize", {"protocolVersion": "2024-11-05"})
    assert resp is not None
    assert resp["result"]["protocolVersion"] == "2024-11-05"
    assert resp["result"]["capabilities"] == {"tools": {}, "logging": {}}

    unknown = _rpc(server, "initialize", {"protocolVersion": "1999-01-01"})
    assert unknown is not None and unknown["result"]["protocolVersion"] != "1999-01-01"


def test_tool_exception_returns_is_error_result() -> None:
    resp = _rpc(_server_with_failing_tool(), "tools/call", {"name": "boom", "arguments": {}})
    assert resp is not None and "error" not in resp
    assert resp["result"]["isError"] is True
    assert "araç patladı" in resp["result"]["content"][0]["text"]


def test_shutdown_and_ping_return_empty_result_instead_of_crashing() -> None:
    server = _server_with_failing_tool()
    assert _rpc(server, "ping") == {"jsonrpc": "2.0", "id": 1, "result": {}}
    assert _rpc(server, "shutdown", req_id=2) == {"jsonrpc": "2.0", "id": 2, "result": {}}


def test_failing_notification_gets_no_response() -> None:
    server = MCPServer(name="t", version="0.1")  # başlatılmamış → tools/list hata verir
    assert _rpc(server, "tools/list", req_id=None) is None


def test_non_object_payload_is_invalid_request() -> None:
    server = MCPServer(name="t", version="0.1")
    resp = json.loads(server.handle("[1, 2, 3]"))  # type: ignore[arg-type]
    assert resp["error"]["code"] == -32600


def test_stdio_server_end_to_end(tmp_path: Path) -> None:
    """Sunucuyu gerçek bir alt süreç olarak başlatıp stdio üzerinden konuşur."""
    cfg = tmp_path / "mcp.yaml"
    cfg.write_text(
        "server: {name: test-mcp, version: '0.1.0'}\n"
        f"filesystem: {{enabled: true, sandbox_root: '{tmp_path / 'files'}'}}\n"
        f"sqlite: {{enabled: true, db_path: '{tmp_path / 'db.sqlite'}'}}\n"
        "tools:\n  - {name: calculator, enabled: true}\n",
        encoding="utf-8",
    )
    (tmp_path / "files").mkdir()
    (tmp_path / "files" / "not.md").write_text("# Merhaba", encoding="utf-8")

    client = MCPStdioClient([sys.executable, "-m", "src.mcp_core", "--config", str(cfg)], cwd=_REPO_ROOT)
    try:
        info = client.initialize()
        assert info["serverInfo"]["name"] == "test-mcp"
        assert [t["name"] for t in client.list_tools()] == ["calculator"]
        result = client.call_tool("calculator", {"expression": "6 * 7"})
        assert result["content"][0]["text"] == "42"
    finally:
        client.close()


# ------------------------------ Çoklu ajan döngüsü ----------------------------- #

def test_executor_strips_random_temp_paths() -> None:
    result = ExecutorAgent(timeout_seconds=10).run("raise ValueError('x')")
    assert not result.passed
    assert "/tmp" not in result.stderr and "solution.py" in result.stderr


def test_identical_failures_are_detected_as_loop() -> None:
    """Aynı bozuk kod iki kez çalıştırıldığında döngü dedektörü tetiklenmelidir."""
    executor = ExecutorAgent(timeout_seconds=10)
    state = AgentState(task="t", code="assert 1 == 2")
    state.last_error = executor.run(state.code).stderr
    assert state.detect_loop() is False
    state.last_error = executor.run(state.code).stderr
    assert state.detect_loop() is True


# --------------------------------- Veri katmanı -------------------------------- #

def test_paper_serializes_with_asdict() -> None:
    paper = Paper("1", "Başlık", ["A", "B"], "Özet", "2024-01-01", "http://x", ["cs.CL"])
    assert json.loads(json.dumps(asdict(paper)))["authors"] == ["A", "B"]


def test_chroma_metadata_is_flattened_to_scalars() -> None:
    meta = _sanitize_metadata({"authors": ["A", "B"], "year": 2024, "x": None, "ok": True})
    assert meta == {"authors": "A; B", "year": 2024, "ok": True}


def test_topic_key_is_stable_and_normalized() -> None:
    assert _topic_key("HyDE ") == _topic_key("hyde")
    assert len(_topic_key("hyde")) == 12


def test_sentence_split_handles_turkish_capitals() -> None:
    chunks = SentenceWindowChunker(window=0).chunk(
        "Birinci cümle. Çok kısa ikinci. İşte üçüncü. Şimdi dördüncü.", doc_metadata={"doc_id": "d"}
    )
    assert len(chunks) == 4
