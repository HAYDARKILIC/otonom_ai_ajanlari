"""Testler ve bitirme projesi tarafından kullanılan asgari bir MCP istemcisi.

İstemci, MCP sunucusu çalıştıran bir alt süreçle stdio üzerinden JSON-RPC 2.0
konuşur. Kablo düzeyindeki mekanikler görünür kalsın diye bilinçli olarak
küçük tutulmuştur — yaşam döngüsünü yürütmeye ve araçları çağırmaya ancak
yetecek kadar.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from src.mcp_core.protocol import JsonRpcRequest


class MCPStdioClient:
    """Bir MCP sunucusunu alt süreç olarak başlatır ve onunla stdio üzerinden konuşur."""

    def __init__(self, command: list[str], cwd: str | Path | None = None) -> None:
        self._proc = subprocess.Popen(
            command,
            cwd=cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._next_id = 0

    def _send(self, method: str, params: dict[str, Any] | None = None,
              *, notification: bool = False) -> dict[str, Any] | None:
        self._next_id += 1
        req_id: int | None = None if notification else self._next_id
        request = JsonRpcRequest(id=req_id, method=method, params=params)
        assert self._proc.stdin is not None
        self._proc.stdin.write(request.model_dump_json(exclude_none=True) + "\n")
        self._proc.stdin.flush()
        if notification:
            return None
        assert self._proc.stdout is not None
        line = self._proc.stdout.readline()
        if not line:
            stderr = self._proc.stderr.read() if self._proc.stderr else ""
            raise RuntimeError(f"MCP sunucusu bağlantıyı kapattı.\n{stderr}")
        return json.loads(line)

    def initialize(self) -> dict[str, Any]:
        resp = self._send("initialize", {"protocolVersion": "0.4.0"})
        assert resp is not None
        if "error" in resp and resp["error"] is not None:
            raise RuntimeError(f"initialize başarısız oldu: {resp['error']}")
        self._send("notifications/initialized", notification=True)
        return resp["result"]

    def list_tools(self) -> list[dict[str, Any]]:
        resp = self._send("tools/list")
        assert resp is not None
        return resp["result"]["tools"]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        resp = self._send("tools/call", {"name": name, "arguments": arguments})
        assert resp is not None
        if resp.get("error"):
            raise RuntimeError(resp["error"]["message"])
        return resp["result"]

    def close(self) -> None:
        try:
            self._send("shutdown", notification=True)
        except (BrokenPipeError, OSError):
            pass  # sunucu zaten kapanmış olabilir
        for stream in (self._proc.stdin, self._proc.stdout, self._proc.stderr):
            if stream is not None:
                stream.close()
        self._proc.terminate()
        self._proc.wait(timeout=5)
