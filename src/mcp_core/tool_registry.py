"""Araç kayıt defteri: MCP üzerinden sunulan, şemayla doğrulanan fonksiyonlar.

Her araç, girdilerinin JSON-Schema tanımını taşır; böylece LLM istemcisi
mevcut eylemleri çağırmadan önce inceleyebilir. Çağrı anında, Python
çağrılabilir nesnesi çalışmadan önce argümanlar şemaya göre doğrulanır.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from src.mcp_core.protocol import INVALID_PARAMS, TOOL_NOT_FOUND, MCPError


@dataclass(slots=True)
class Tool:
    """MCP ile çağrılabilir bir araç: ad + şema + işleyici."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., Any]
    _validator: Draft202012Validator = field(init=False)

    def __post_init__(self) -> None:
        # Çağrı başına ek yük asgari olsun diye doğrulayıcı bir kez önceden derlenir.
        Draft202012Validator.check_schema(self.input_schema)
        self._validator = Draft202012Validator(self.input_schema)

    def describe(self) -> dict[str, Any]:
        """``tools/list`` yanıtında gönderilen tanımlayıcıyı döndürür."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }

    def invoke(self, arguments: dict[str, Any]) -> Any:
        """Argümanları doğrular ve sarmalanan işleyiciyi çağırır."""
        try:
            self._validator.validate(arguments)
        except ValidationError as exc:
            raise MCPError(
                code=INVALID_PARAMS,
                message=f"{self.name!r} aracı için geçersiz argümanlar: {exc.message}",
                data={"path": list(exc.absolute_path)},
            ) from exc
        return self.handler(**arguments)


class ToolRegistry:
    """Ada göre indekslenmiş bir araç koleksiyonu."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"{tool.name!r} aracı zaten kayıtlı.")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise MCPError(
                code=TOOL_NOT_FOUND,
                message=f"{name!r} aracı kayıtlı değil.",
            ) from exc

    def list_all(self) -> list[dict[str, Any]]:
        return [t.describe() for t in self._tools.values()]

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._tools

    def __len__(self) -> int:
        return len(self._tools)
