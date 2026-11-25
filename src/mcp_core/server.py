"""MCP sunucusu: JSON-RPC yöntem çağrılarını araç ve kaynak işleyicilerine dağıtır.

Sunucu aktarımdan bağımsızdır: :class:`JsonRpcRequest` nesnelerini tüketir ve
:class:`JsonRpcResponse` nesneleri üretir. Eşlik eden
:func:`run_stdio_server` bağdaştırıcısı, sunucuyu Claude Desktop ve diğer MCP
istemcilerinin kullandığı standart stdin/stdout aktarımına bağlar.

Yaşam döngüsü (MCP spesifikasyonuna göre):

    1. İstemci ``initialize`` gönderir.
    2. Sunucu ``protocolVersion``, ``serverInfo`` ve yeteneklerle yanıt verir.
    3. İstemci ``notifications/initialized`` gönderir (yanıt yoktur).
    4. İstemci ``tools/list``, ``tools/call``, ``resources/list``,
       ``resources/read`` vb. çağırır.
    5. Taraflardan biri ``shutdown`` gönderir.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from src.mcp_core.protocol import (
    INTERNAL_ERROR,
    INVALID_REQUEST,
    LATEST_PROTOCOL_VERSION,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    RESOURCE_NOT_FOUND,
    SUPPORTED_PROTOCOL_VERSIONS,
    InitializeResult,
    JsonRpcError,
    JsonRpcRequest,
    JsonRpcResponse,
    MCPError,
    ServerCapabilities,
    ServerInfo,
)
from src.mcp_core.tool_registry import ToolRegistry
from src.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class MCPServer:
    """Asgari ama spesifikasyona uygun bir MCP sunucusu."""

    name: str
    version: str
    tools: ToolRegistry = field(default_factory=ToolRegistry)
    fs_resources: Any | None = None  # FilesystemResources
    db_resources: Any | None = None  # SQLiteResources

    _initialized: bool = field(default=False, init=False)

    # ------------------------------------------------------------------ #
    # Genel giriş noktası                                                #
    # ------------------------------------------------------------------ #

    def handle(self, raw: str) -> str | None:
        """Ham bir JSON-RPC çerçevesini işler; yanıt çerçevesini veya None döndürür."""
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            return self._error_frame(None, PARSE_ERROR, f"Ayrıştırma hatası: {exc}")

        if not isinstance(payload, dict):
            # Toplu (batch) istekler desteklenmez; dizi ya da skaler yük geçersiz istektir.
            return self._error_frame(None, INVALID_REQUEST, "İstek bir JSON nesnesi olmalıdır.")

        try:
            request = JsonRpcRequest.model_validate(payload)
        except Exception as exc:  # noqa: BLE001
            req_id = payload.get("id")
            if not isinstance(req_id, (int, str)):
                req_id = None
            return self._error_frame(req_id, INVALID_REQUEST, str(exc))

        try:
            result = self._dispatch(request)
        except MCPError as exc:
            log.warning("mcp_error", method=request.method, code=exc.code, message=exc.message)
            if request.is_notification:
                return None  # JSON-RPC: bildirimlere hata dâhil hiçbir yanıt gönderilmez.
            return self._error_frame(request.id, exc.code, exc.message, exc.data)
        except Exception as exc:  # noqa: BLE001
            log.exception("internal_error", method=request.method)
            if request.is_notification:
                return None
            return self._error_frame(request.id, INTERNAL_ERROR, f"Dahili hata: {exc}")

        if request.is_notification:
            return None

        # Sonucu olmayan istekler (ör. ``shutdown``, ``ping``) boş bir nesne döndürür;
        # JSON-RPC başarılı bir yanıtta ``result`` alanının bulunmasını şart koşar.
        response = JsonRpcResponse(id=request.id, result={} if result is None else result)
        return response.model_dump_json(by_alias=True, exclude_none=True)

    # ------------------------------------------------------------------ #
    # Yöntem dağıtımı                                                    #
    # ------------------------------------------------------------------ #

    def _dispatch(self, request: JsonRpcRequest) -> Any:
        handler = self._handlers.get(request.method)
        if handler is None:
            raise MCPError(METHOD_NOT_FOUND, f"Bilinmeyen yöntem: {request.method!r}")
        params = request.params or {}
        if isinstance(params, list):
            # Konumsal parametreler MCP tarafından kullanılmaz; katı olmak için reddedilir.
            raise MCPError(INVALID_REQUEST, "Konumsal parametreler desteklenmiyor.")
        return handler(self, params)

    # ------------------------------------------------------------------ #
    # Yöntem işleyicileri                                                #
    # ------------------------------------------------------------------ #

    def _handle_initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        self._initialized = True
        requested = params.get("protocolVersion")
        # Sürüm müzakeresi: desteklenen bir sürüm istendiyse aynısını döndür,
        # aksi hâlde en yenisini öner (istemci uyumsuzsa bağlantıyı kendisi keser).
        version = requested if requested in SUPPORTED_PROTOCOL_VERSIONS else LATEST_PROTOCOL_VERSION
        capabilities = ServerCapabilities(
            tools=len(self.tools) > 0,
            resources=self.fs_resources is not None or self.db_resources is not None,
        )
        return InitializeResult(
            protocolVersion=version,
            serverInfo=ServerInfo(name=self.name, version=self.version),
            capabilities=capabilities.to_wire(),
        ).model_dump(by_alias=True)

    def _handle_initialized(self, _params: dict[str, Any]) -> None:
        # Bildirim; yanıt gerekmez.
        return None

    def _handle_tools_list(self, _params: dict[str, Any]) -> dict[str, Any]:
        self._require_initialized()
        return {"tools": self.tools.list_all()}

    def _handle_tools_call(self, params: dict[str, Any]) -> dict[str, Any]:
        self._require_initialized()
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if not isinstance(name, str):
            raise MCPError(INVALID_REQUEST, "`name` bir dize (string) olmalıdır.")
        tool = self.tools.get(name)
        try:
            result = tool.invoke(arguments)
        except MCPError:
            raise  # şema doğrulama hataları protokol hatası olarak kalır
        except Exception as exc:  # noqa: BLE001
            # MCP spesifikasyonu: aracın kendi hatası JSON-RPC hatası değil,
            # ``isError: true`` taşıyan bir sonuç olarak döner; böylece LLM hatayı görüp toparlanabilir.
            log.warning("tool_execution_error", tool=name, error=str(exc))
            return {
                "content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}],
                "isError": True,
            }
        # MCP, araç sonuçlarını içerik blokları listesi olarak döndürür.
        return {
            "content": [{"type": "text", "text": _stringify(result)}],
            "isError": False,
        }

    def _handle_resources_list(self, _params: dict[str, Any]) -> dict[str, Any]:
        self._require_initialized()
        resources: list[dict[str, Any]] = []
        if self.fs_resources is not None:
            resources.extend(self.fs_resources.list_resources())
        if self.db_resources is not None:
            resources.extend(self.db_resources.list_resources())
        return {"resources": resources}

    def _handle_resources_read(self, params: dict[str, Any]) -> dict[str, Any]:
        self._require_initialized()
        uri = params.get("uri")
        if not isinstance(uri, str):
            raise MCPError(INVALID_REQUEST, "`uri` bir dize (string) olmalıdır.")
        if uri.startswith("file://") and self.fs_resources is not None:
            text = self.fs_resources.read(uri)
            from pathlib import Path

            from src.mcp_core.resources import _mime_for
            mime = _mime_for(Path(uri[len("file://"):]))
            return {"contents": [{"uri": uri, "text": text, "mimeType": mime}]}
        if uri.startswith("sqlite://") and self.db_resources is not None:
            rows = self.db_resources.read(uri)
            return {"contents": [{"uri": uri, "text": json.dumps(rows), "mimeType": "application/json"}]}
        raise MCPError(RESOURCE_NOT_FOUND, f"Bu URI için kaynak sağlayıcı yok: {uri!r}", data={"uri": uri})

    def _handle_shutdown(self, _params: dict[str, Any]) -> None:
        self._initialized = False
        return None

    def _handle_ping(self, _params: dict[str, Any]) -> dict[str, Any]:
        # MCP: her iki taraf da bağlantının canlı olduğunu ``ping`` ile denetleyebilir.
        return {}

    _handlers: dict[str, Callable[[MCPServer, dict[str, Any]], Any]] = field(
        init=False,
        repr=False,
        default_factory=lambda: {
            "initialize": MCPServer._handle_initialize,
            "notifications/initialized": MCPServer._handle_initialized,
            "tools/list": MCPServer._handle_tools_list,
            "tools/call": MCPServer._handle_tools_call,
            "resources/list": MCPServer._handle_resources_list,
            "resources/read": MCPServer._handle_resources_read,
            "shutdown": MCPServer._handle_shutdown,
            "ping": MCPServer._handle_ping,
        },
    )

    # ------------------------------------------------------------------ #
    # Yardımcılar                                                        #
    # ------------------------------------------------------------------ #

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise MCPError(INVALID_REQUEST, "Sunucu henüz başlatılmadı (initialize çağrılmadı).")

    @staticmethod
    def _error_frame(
        request_id: int | str | None,
        code: int,
        message: str,
        data: Any | None = None,
    ) -> str:
        resp = JsonRpcResponse(id=request_id, error=JsonRpcError(code=code, message=message, data=data))
        return resp.model_dump_json(by_alias=True, exclude_none=True)


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except TypeError:
        return str(value)


def run_stdio_server(server: MCPServer) -> None:
    """Bir :class:`MCPServer` örneğini stdin/stdout üzerinde satır ayrımlı JSON ile çalıştırır."""
    log.info("mcp_server_starting", name=server.name, version=server.version)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        response = server.handle(line)
        if response is not None:
            sys.stdout.write(response + "\n")
            sys.stdout.flush()


# ---------------------------------------------------------------------- #
# Komut satırı giriş noktası                                              #
# ---------------------------------------------------------------------- #

def build_server_from_config(config_path: str) -> MCPServer:
    """``configs/mcp_server.yaml`` biçimindeki bir yapılandırmadan sunucu kurar."""
    from pathlib import Path

    from src.mcp_core.resources import FilesystemResources, SQLiteResources
    from src.mcp_core.tool_registry import ToolRegistry
    from src.react_agent.tools import calculator, make_read_file_tool, search_arxiv
    from src.utils.config import load_yaml_config

    cfg = load_yaml_config(config_path)
    fs_cfg = cfg.get("filesystem", {})
    # read_file yalnızca dosya sistemi korumalı alanının içini okuyabilir.
    read_root = Path(fs_cfg.get("sandbox_root", "./data/processed"))
    read_root.mkdir(parents=True, exist_ok=True)
    available = {t.name: t for t in (calculator, search_arxiv, make_read_file_tool(read_root))}

    registry = ToolRegistry()
    for entry in cfg.get("tools", []):
        if not entry.get("enabled", True):
            continue
        name = entry["name"]
        if name not in available:
            raise ValueError(f"Yapılandırmada bilinmeyen araç: {name!r}. Mevcut: {sorted(available)}")
        registry.register(available[name])

    fs_resources = None
    if fs_cfg.get("enabled"):
        root = Path(fs_cfg["sandbox_root"])
        root.mkdir(parents=True, exist_ok=True)
        fs_resources = FilesystemResources(root)

    db_resources = None
    db_cfg = cfg.get("sqlite", {})
    if db_cfg.get("enabled"):
        db_path = Path(db_cfg["db_path"])
        if not db_path.exists():
            # Salt okunur açmadan önce boş bir veritabanı oluştur (ilk çalıştırma).
            db_path.parent.mkdir(parents=True, exist_ok=True)
            sqlite3.connect(db_path).close()
        db_resources = SQLiteResources(db_path)

    server_cfg = cfg.get("server", {})
    return MCPServer(
        name=server_cfg.get("name", "mcp-server"),
        version=str(server_cfg.get("version", "0.0.0")),
        tools=registry,
        fs_resources=fs_resources,
        db_resources=db_resources,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Özel MCP sunucusunu stdio üzerinden çalıştırır.")
    parser.add_argument("--config", default="configs/mcp_server.yaml", help="YAML yapılandırma dosyası.")
    args = parser.parse_args(argv)
    run_stdio_server(build_server_from_config(args.config))


if __name__ == "__main__":
    main()
