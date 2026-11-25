"""MCP spesifikasyonunun gerektirdiği JSON-RPC 2.0 çerçevelemesi.

Model Context Protocol, mesaj zarfı olarak JSON-RPC 2.0'ı zorunlu kılar.
Bu modül istek, yanıt ve hata nesnelerini Pydantic şemaları olarak
modeller; böylece kablodan geçen her mesaj doğrulanır. MCP katmanı bu
aktarımın *üzerinde* durur ve yönteme özgü yük şemalarını ekler
(``initialize``, ``tools/list``, ``tools/call``, ``resources/read``, …).

Kaynaklar
---------
- JSON-RPC 2.0 spesifikasyonu:  https://www.jsonrpc.org/specification
- Model Context Protocol:       https://modelcontextprotocol.io
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

# JSON-RPC 2.0'ın ayrılmış hata kodları (spesifikasyonun §5.1 bölümüne bakın).
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

# Uygulama hata kodları (-32000 ile -32099 arası uygulamaya ayrılmıştır).
# RESOURCE_NOT_FOUND (-32002) MCP spesifikasyonunda tanımlıdır; diğerleri bu uygulamaya özgüdür.
TOOL_NOT_FOUND = -32001
RESOURCE_NOT_FOUND = -32002
PERMISSION_DENIED = -32003
TOOL_EXECUTION_ERROR = -32004


class JsonRpcRequest(BaseModel):
    """Tek bir JSON-RPC 2.0 istek nesnesi."""

    jsonrpc: Literal["2.0"] = "2.0"
    id: int | str | None = None
    method: str
    params: dict[str, Any] | list[Any] | None = None

    @property
    def is_notification(self) -> bool:
        """Bir bildirim (notification) ``id`` taşımaz ve yanıt beklemez."""
        return self.id is None


class JsonRpcError(BaseModel):
    """Bir JSON-RPC 2.0 hata nesnesi."""

    code: int
    message: str
    data: Any | None = None


class JsonRpcResponse(BaseModel):
    """Tek bir JSON-RPC 2.0 yanıtı.

    ``result`` veya ``error`` alanlarından tam olarak biri bulunmalıdır.
    """

    jsonrpc: Literal["2.0"] = "2.0"
    id: int | str | None
    result: Any | None = None
    error: JsonRpcError | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> JsonRpcResponse:
        if (self.result is None) == (self.error is None):
            raise ValueError("`result` veya `error` alanlarından tam olarak biri ayarlanmalıdır.")
        return self


class MCPError(Exception):
    """İşleyiciler tarafından istemciye yapılandırılmış bir JSON-RPC hatası iletmek için fırlatılır."""

    def __init__(self, code: int, message: str, data: Any | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data

    def to_error(self) -> JsonRpcError:
        return JsonRpcError(code=self.code, message=self.message, data=self.data)


class ServerInfo(BaseModel):
    """``initialize`` el sıkışmasında döndürülen MCP sunucu kimliği."""

    name: str
    version: str


# Sunucunun desteklediği MCP protokol sürümleri (yeniden eskiye).
# İstemci bunlardan birini isterse aynısı döndürülür; aksi hâlde en yenisi önerilir.
SUPPORTED_PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
LATEST_PROTOCOL_VERSION = SUPPORTED_PROTOCOL_VERSIONS[0]


class ServerCapabilities(BaseModel):
    """``initialize`` sırasında döndürülen yetenek duyurusu.

    Burada MCP yeteneklerinin yalnızca bir alt kümesi uygulanmıştır. Dahili
    olarak her yetenek bir boole değeriyle tutulur; :meth:`to_wire` bunu
    spesifikasyonun beklediği biçime (``{"tools": {}, ...}``) çevirir —
    yalnızca etkin yetenekler, her biri bir nesne olarak duyurulur.
    """

    tools: bool = True
    resources: bool = True
    prompts: bool = False
    logging: bool = True

    def to_wire(self) -> dict[str, dict[str, Any]]:
        return {name: {} for name, enabled in self.model_dump().items() if enabled}


class InitializeResult(BaseModel):
    protocol_version: str = Field(default=LATEST_PROTOCOL_VERSION, alias="protocolVersion")
    server_info: ServerInfo = Field(alias="serverInfo")
    capabilities: dict[str, dict[str, Any]]

    model_config = {"populate_by_name": True}
