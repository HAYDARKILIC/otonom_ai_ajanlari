"""Model ya da ağ gerektiren bileşenlerin, sahte bağımlılıklarla birim testleri."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from src.mcp_core.server import MCPServer, build_server_from_config
from src.rag_engine.parent_store import SQLiteParentStore
from src.rag_engine.pipeline import RetrievedDocument
from src.react_agent.agent import ReActAgent
from src.react_agent.tools import calculator

_REPO = Path(__file__).resolve().parents[1]


def _rpc(server: MCPServer, method: str, params: dict | None = None, req_id: int = 1) -> dict:
    raw = server.handle(json.dumps({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params or {}}))
    assert raw is not None
    return json.loads(raw)


# --------------------------------- MCP sunucusu -------------------------------- #

def test_server_from_repo_config_serves_tools_and_resources(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)  # yapılandırmadaki göreli ./data yolları geçici dizine düşsün
    server = build_server_from_config(str(_REPO / "configs" / "mcp_server.yaml"))
    Path("data/processed/not.md").write_text("# Not", encoding="utf-8")

    init = _rpc(server, "initialize", {"protocolVersion": "2025-06-18"})["result"]
    assert init["serverInfo"]["name"] == "agentic-ce-mcp"
    assert set(init["capabilities"]) == {"tools", "resources", "logging"}

    names = [t["name"] for t in _rpc(server, "tools/list")["result"]["tools"]]
    assert names == ["calculator", "search_arxiv"]

    uris = [r["uri"] for r in _rpc(server, "resources/list")["result"]["resources"]]
    assert "file://not.md" in uris
    read = _rpc(server, "resources/read", {"uri": "file://not.md"})["result"]["contents"][0]
    assert read["text"] == "# Not" and read["mimeType"] == "text/markdown"

    assert _rpc(server, "resources/read", {"uri": "http://x"})["error"]["code"] == -32002
    assert _rpc(server, "resources/read", {"uri": 5})["error"]["code"] == -32600
    assert _rpc(server, "tools/call", {"name": 5})["error"]["code"] == -32600
    assert _rpc(server, "tools/call", {"name": "yok"})["error"]["code"] == -32001


def test_server_rejects_unknown_tool_in_config(tmp_path: Path) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text("tools:\n  - {name: hayalet, enabled: true}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hayalet"):
        build_server_from_config(str(cfg))


def test_server_read_file_tool_is_bound_to_sandbox(tmp_path: Path) -> None:
    root = tmp_path / "kum"
    root.mkdir()
    (root / "a.txt").write_text("merhaba", encoding="utf-8")
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"filesystem: {{enabled: false, sandbox_root: '{root}'}}\n"
                   "tools:\n  - {name: read_file, enabled: true}\n", encoding="utf-8")
    server = build_server_from_config(str(cfg))
    _rpc(server, "initialize")
    ok = _rpc(server, "tools/call", {"name": "read_file", "arguments": {"path": "a.txt"}})
    assert ok["result"]["content"][0]["text"] == "merhaba"
    bad = _rpc(server, "tools/call", {"name": "read_file", "arguments": {"path": "../c.yaml"}})
    assert bad["result"]["isError"] is True


def test_server_parse_and_positional_errors() -> None:
    server = MCPServer(name="t", version="1")
    assert json.loads(server.handle("{bozuk"))["error"]["code"] == -32700  # type: ignore[arg-type]
    pos = json.loads(server.handle(json.dumps(  # type: ignore[arg-type]
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": [1]})))
    assert pos["error"]["code"] == -32600
    bad = json.loads(server.handle(json.dumps({"jsonrpc": "1.0", "id": 7, "method": "x"})))  # type: ignore[arg-type]
    assert bad["error"]["code"] == -32600 and bad["id"] == 7


# ------------------------------------ CLI ------------------------------------- #

def test_assistant_cli_writes_report(tmp_path: Path, monkeypatch) -> None:
    import src.academic_assistant.cli as cli

    class _Stub:
        def run(self, cfg):
            assert cfg.topic == "konu" and cfg.max_papers == 3
            return "# rapor\n"

    monkeypatch.setattr(cli, "AcademicAssistant", _Stub)
    out = tmp_path / "r" / "rapor.md"
    res = CliRunner().invoke(cli.app, ["--topic", "konu", "--max-papers", "3", "--output", str(out)])
    assert res.exit_code == 0, res.output
    assert out.read_text(encoding="utf-8") == "# rapor\n"


# ------------------------------- Ebeveyn deposu -------------------------------- #

def test_parent_store_mapping_protocol(tmp_path: Path) -> None:
    store = SQLiteParentStore(tmp_path / "p.sqlite")
    store["a"] = "1"
    store.update_many({"b": "2", "c": "3"})
    assert len(store) == 3 and sorted(store) == ["a", "b", "c"] and store.get("z") is None
    del store["a"]
    with pytest.raises(KeyError):
        del store["a"]
    with pytest.raises(KeyError):
        _ = store["a"]
    store.close()
    assert dict(SQLiteParentStore(tmp_path / "p.sqlite")) == {"b": "2", "c": "3"}


# --------------------------- Sahte model kütüphaneleri -------------------------- #

@pytest.fixture
def fake_sentence_transformers(monkeypatch):
    """``sentence_transformers`` modülünü hafif bir sahtesiyle değiştirir."""
    mod = types.ModuleType("sentence_transformers")

    class SentenceTransformer:
        def __init__(self, name, device=None):
            self.name = name

        def get_sentence_embedding_dimension(self):
            return 4

        def encode(self, texts, **kw):
            assert kw["normalize_embeddings"] is True
            return np.ones((len(texts), 4), dtype=np.float64) / 2.0

    class CrossEncoder:
        def __init__(self, name, device=None):
            self.name = name

        def predict(self, pairs, **kw):
            return np.array([float(len(d)) for _, d in pairs])

    mod.SentenceTransformer = SentenceTransformer  # type: ignore[attr-defined]
    mod.CrossEncoder = CrossEncoder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", mod)
    return mod


def test_sentence_encoder_wrapper(fake_sentence_transformers) -> None:
    from src.rag_engine.embeddings import SentenceEncoder

    enc = SentenceEncoder("m")
    assert enc.dim == 4
    vec = enc.encode_one("x")
    assert vec.dtype == np.float32 and vec.shape == (4,)


def test_cross_encoder_reranker_orders_by_score(fake_sentence_transformers) -> None:
    from src.reranking.cross_encoder import CrossEncoderReranker

    rer = CrossEncoderReranker("m")
    cands = [RetrievedDocument("a", "kısa", 0.9, {}), RetrievedDocument("b", "çok daha uzun metin", 0.1, {})]
    out = rer.rerank("q", cands, k=1)
    assert [d.chunk_id for d in out] == ["b"] and out[0].bi_encoder_score == 0.1
    assert rer.rerank("q", []) == []


def test_reranking_pipeline_from_config(fake_sentence_transformers) -> None:
    from src.reranking.pipeline import RerankingPipeline

    pipe = RerankingPipeline.from_config(str(_REPO / "configs" / "reranker_cross_encoder.yaml"))
    assert pipe.candidate_k == 50


def test_reranking_pipeline_rejects_unknown_backend(tmp_path: Path) -> None:
    from src.reranking.pipeline import RerankingPipeline

    cfg = tmp_path / "r.yaml"
    cfg.write_text("reranker: {backend: yok}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        RerankingPipeline.from_config(str(cfg))


def test_cohere_reranker_maps_results(monkeypatch) -> None:
    mod = types.ModuleType("cohere")

    class Client:
        def __init__(self, key):
            self.key = key

        def rerank(self, **kw):
            assert kw["top_n"] == 1
            result = types.SimpleNamespace(index=1, relevance_score=0.8)
            return types.SimpleNamespace(results=[result])

    mod.Client = Client  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cohere", mod)
    from src.reranking.cross_encoder import CohereReranker

    rer = CohereReranker(api_key="k")
    cands = [RetrievedDocument("a", "x", 0.5, {}), RetrievedDocument("b", "y", 0.4, {})]
    out = rer.rerank("q", cands, k=1)
    assert out[0].chunk_id == "b" and out[0].cross_encoder_score == 0.8


def test_fetch_papers_maps_arxiv_results(monkeypatch) -> None:
    import datetime as dt

    import src.academic_assistant.arxiv_ingest as ing

    fake = types.SimpleNamespace(
        entry_id="http://arxiv.org/abs/2212.10496v1", title="Başlık\nikinci satır",
        authors=[types.SimpleNamespace(name="Gao")], summary=" Özet\nmetni ",
        published=dt.datetime(2022, 12, 20), pdf_url="http://pdf", categories=["cs.IR"],
    )

    class Client:
        def results(self, search):
            return iter([fake])

    monkeypatch.setattr(ing.arxiv, "Client", Client)
    paper = next(ing.fetch_papers("q", max_results=1))
    assert paper.arxiv_id == "2212.10496v1" and paper.title == "Başlık ikinci satır"
    assert paper.abstract == "Özet metni" and paper.published == "2022-12-20"


# ------------------------------------ ReAct ------------------------------------ #

def test_react_agent_halts_after_parse_retry_budget(fake_llm_factory) -> None:
    llm = fake_llm_factory(["anlamsız", "yine anlamsız", "hâlâ anlamsız"])
    run = ReActAgent(llm=llm, tools=[calculator], parse_retry_budget=2).run("x")
    assert run.halted_reason == "parse_failure" and len(llm.calls) == 3
    assert "Son yanıtın ayrıştırılamadı" in llm.calls[1][-1].content


def test_react_agent_reports_tool_errors_as_observations(fake_llm_factory) -> None:
    llm = fake_llm_factory([
        'Thought: böl.\nAction: calculator\nAction Input: {"expression": "1/0"}',
        "Thought: tamam.\nFinal Answer: bitti",
    ])
    run = ReActAgent(llm=llm, tools=[calculator]).run("x")
    assert run.trace[0].observation == "ERROR: ZeroDivisionError: division by zero"
    assert "Observation: ERROR" in llm.calls[1][-1].content
