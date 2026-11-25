"""LLM'e (ve MCP istemcilerine) açık araçların güvenlik testleri."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from src.mcp_core.protocol import RESOURCE_NOT_FOUND, MCPError
from src.mcp_core.resources import FilesystemResources, SQLiteResources
from src.react_agent.tools import CalculatorError, make_read_file_tool, safe_eval


@pytest.mark.parametrize(("expr", "expected"), [
    ("6 * 7", 42), ("2 ** 10", 1024), ("-3 + 5", 2), ("7 // 2", 3), ("7 % 4", 3),
    ("sqrt(16)", 4.0), ("max(1, 5, 3)", 5), ("round(pi, 2)", 3.14), ("factorial(5)", 120),
    ("(1 + 2) * (3 + 4)", 21),
])
def test_calculator_valid_expressions(expr: str, expected: float) -> None:
    assert safe_eval(expr) == expected


@pytest.mark.parametrize("expr", [
    "().__class__.__base__.__subclasses__()",   # klasik eval kaçışı
    "__import__('os').system('echo x')",
    "open('/etc/passwd').read()",
    "[x for x in range(10)]",
    "lambda: 1",
    "'abc' * 3",
    "9 ** 9 ** 9 ** 9",                           # DoS: devasa üs
    "factorial(100000)",
    "1" * 600,                                    # aşırı uzun ifade
])
def test_calculator_rejects_unsafe_expressions(expr: str) -> None:
    with pytest.raises(CalculatorError):
        safe_eval(expr)


def test_read_file_is_confined_to_root(tmp_path: Path) -> None:
    (tmp_path / "ok.txt").write_text("içerik", encoding="utf-8")
    outside = tmp_path.parent / "gizli.txt"
    outside.write_text("gizli", encoding="utf-8")
    tool = make_read_file_tool(tmp_path)
    assert tool.invoke({"path": "ok.txt"}) == "içerik"
    with pytest.raises(PermissionError):
        tool.invoke({"path": "../gizli.txt"})
    with pytest.raises(PermissionError):
        tool.invoke({"path": str(outside)})  # mutlak yol da reddedilir
    (tmp_path / "bag").symlink_to(outside)
    with pytest.raises(PermissionError):
        tool.invoke({"path": "bag"})


def test_filesystem_missing_file_is_resource_not_found(tmp_path: Path) -> None:
    with pytest.raises(MCPError) as info:
        FilesystemResources(tmp_path).read("file://yok.md")
    assert info.value.code == RESOURCE_NOT_FOUND


def test_sqlite_only_existing_tables_are_readable(tmp_path: Path) -> None:
    db = tmp_path / "x.sqlite"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE papers (id TEXT)")
    conn.execute("INSERT INTO papers VALUES ('1')")
    conn.commit()
    conn.close()
    res = SQLiteResources(db)
    assert res.read("sqlite://papers") == [{"id": "1"}]
    for bad in ("sqlite://sqlite_master", "sqlite://nope", "sqlite://papers; DROP TABLE papers"):
        with pytest.raises(MCPError) as info:
            res.read(bad)
        assert info.value.code == RESOURCE_NOT_FOUND
