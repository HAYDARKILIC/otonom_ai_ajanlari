"""Hafta 1 — Model Context Protocol.

MCP'nin JSON-RPC 2.0 üzerinden sıfırdan yazılmış, spesifikasyona uygun bir uygulaması:

    İstemci ──▶  initialize          ──▶  Sunucu
            ◀──  yetenekler, bilgi   ◀──
            ──▶  tools/list          ──▶
            ◀──  [araç tanımları]    ◀──
            ──▶  tools/call          ──▶
            ◀──  sonuç | hata        ◀──

Buradaki uygulama stdio aktarımını, yaşam döngüsü işleyicilerini, araç
kaydını ve kaynakların dışa açılmasını gerçekleştirir. Bir SDK'nın etrafına
sarılmış ince bir katman olarak değil, bilinçli olarak bir *spesifikasyon
eşlikçisi* gibi okunabilecek şekilde yazılmıştır.
"""

from src.mcp_core.protocol import (
    JsonRpcError,
    JsonRpcRequest,
    JsonRpcResponse,
    MCPError,
)
from src.mcp_core.server import MCPServer
from src.mcp_core.tool_registry import Tool, ToolRegistry

__all__ = [
    "JsonRpcError",
    "JsonRpcRequest",
    "JsonRpcResponse",
    "MCPError",
    "MCPServer",
    "Tool",
    "ToolRegistry",
]
