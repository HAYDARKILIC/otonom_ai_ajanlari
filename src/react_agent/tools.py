"""ReAct ajanına kaydedilen araçlar.

Bir araç, girdilerinin JSON-Schema tanımını ve bir Python işleyicisini taşıyan
bir :class:`Tool` nesnesidir (:mod:`mcp_core.tool_registry` modülünden). Aynı
:class:`Tool` sınıfı burada yeniden kullanılır; çünkü araç şemaları, ister MCP
üzerinden sunulsun ister doğrudan süreç içinde çağrılsın, tamamen aynı
öğedir.

Güvenlik
--------
Bu araçlar bir LLM tarafından (ve MCP üzerinden herhangi bir istemci
tarafından) çağrılır; yani girdileri güvenilmezdir:

- ``calculator`` ``eval`` kullanmaz. İfade :mod:`ast` ile ayrıştırılır ve
  yalnızca beyaz listedeki sayısal düğümler değerlendirilir. ``eval`` ile
  ``__builtins__ = {}`` koruması ``().__class__.__base__.__subclasses__()``
  gibi ifadelerle aşılabilir. Üs ve sonuç büyüklüğü de sınırlandırılır
  (``9**9**9**9`` gibi ifadeler süreci kilitleyemez).
- ``read_file`` yalnızca bir kök dizinin altındaki dosyaları okur; ``../`` ve
  sembolik bağ kaçışları reddedilir.
"""

from __future__ import annotations

import ast
import math
import operator
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.mcp_core.tool_registry import Tool

# ---------------------------------------------------------------------- #
# calculator                                                              #
# ---------------------------------------------------------------------- #

_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "abs": abs, "min": min, "max": max, "round": round, "pow": pow,
    "sqrt": math.sqrt, "log": math.log, "log2": math.log2, "log10": math.log10,
    "exp": math.exp, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "floor": math.floor, "ceil": math.ceil, "factorial": math.factorial,
}
_CONSTANTS: dict[str, float] = {"pi": math.pi, "e": math.e, "tau": math.tau}

_BIN_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos, ast.USub: operator.neg,
}

_MAX_EXPRESSION_CHARS = 500
_MAX_EXPONENT = 1000
_MAX_INT_DIGITS = 4000
_MAX_FACTORIAL = 1000


class CalculatorError(ValueError):
    """İfade geçersiz ya da izin verilen sınırların dışında."""


def _check_size(value: Any) -> Any:
    if isinstance(value, int) and not isinstance(value, bool):
        if value.bit_length() > _MAX_INT_DIGITS * 3.33:
            raise CalculatorError(f"Sonuç çok büyük (>{_MAX_INT_DIGITS} basamak).")
    return value


def _eval_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
            and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise CalculatorError(f"Bilinmeyen ad: {node.id!r}")
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            raise CalculatorError(f"Üs çok büyük (en fazla {_MAX_EXPONENT}).")
        return _check_size(_BIN_OPS[type(node.op)](left, right))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
        func = _FUNCTIONS.get(node.func.id)
        if func is None:
            raise CalculatorError(f"Desteklenmeyen fonksiyon: {node.func.id!r}")
        args = [_eval_node(a) for a in node.args]
        if func is pow and len(args) >= 2 and abs(args[1]) > _MAX_EXPONENT:
            raise CalculatorError(f"Üs çok büyük (en fazla {_MAX_EXPONENT}).")
        if func is math.factorial and args and args[0] > _MAX_FACTORIAL:
            raise CalculatorError(f"factorial argümanı en fazla {_MAX_FACTORIAL} olabilir.")
        return _check_size(func(*args))
    raise CalculatorError(f"Desteklenmeyen ifade öğesi: {type(node).__name__}")


def safe_eval(expression: str) -> Any:
    """Bir aritmetik ifadeyi ``eval`` kullanmadan güvenli biçimde değerlendirir."""
    if len(expression) > _MAX_EXPRESSION_CHARS:
        raise CalculatorError(f"İfade çok uzun (en fazla {_MAX_EXPRESSION_CHARS} karakter).")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise CalculatorError(f"Geçersiz ifade: {exc.msg}") from exc
    return _eval_node(tree)


def _calculator_handler(expression: str) -> str:
    """Bir aritmetik ifadeyi değerlendirir ve sonucu dize olarak döndürür."""
    return str(safe_eval(expression))


calculator = Tool(
    name="calculator",
    description=(
        "Bir aritmetik ifadeyi değerlendirir. +, -, *, /, //, %, **, parantezler, "
        "sqrt, log, log2, log10, exp, sin, cos, tan, floor, ceil, factorial, abs, min, max, "
        "round, pow fonksiyonları ile pi, e ve tau sabitlerini destekler."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "Değerlendirilecek aritmetik ifade."}
        },
        "required": ["expression"],
        "additionalProperties": False,
    },
    handler=_calculator_handler,
)


# ---------------------------------------------------------------------- #
# search_arxiv                                                            #
# ---------------------------------------------------------------------- #

def _search_arxiv_handler(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """Herkese açık API aracılığıyla arXiv'de arama yapar. Başlık + özet + URL döndürür."""
    import arxiv

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    out = []
    for result in arxiv.Client().results(search):
        out.append({
            "title": result.title.strip(),
            "abstract": result.summary.strip().replace("\n", " "),
            "url": result.entry_id,
            "published": str(result.published.date()),
        })
    return out


search_arxiv = Tool(
    name="search_arxiv",
    description="arXiv'de akademik makale arar. Başlıkları, özetleri ve URL'leri döndürür.",
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 25, "default": 5},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    handler=_search_arxiv_handler,
)


# ---------------------------------------------------------------------- #
# read_file                                                               #
# ---------------------------------------------------------------------- #

def make_read_file_tool(root: str | Path) -> Tool:
    """Yalnızca ``root`` altındaki dosyaları okuyabilen bir ``read_file`` aracı üretir."""
    sandbox = Path(root).resolve()

    def _read_file_handler(path: str, max_chars: int = 20000) -> str:
        """Korumalı alandan UTF-8 bir metin dosyası okur (``max_chars`` uzunluğunda kırpılır)."""
        target = (sandbox / path).resolve()
        if os.path.commonpath([str(target), str(sandbox)]) != str(sandbox):
            raise PermissionError(f"Yol izin verilen dizinin dışında: {path!r}")
        if not target.is_file():
            raise FileNotFoundError(f"Dosya bulunamadı: {path!r}")
        return target.read_text(encoding="utf-8", errors="replace")[:max_chars]

    return Tool(
        name="read_file",
        description=(
            "İzin verilen çalışma dizininden UTF-8 bir metin dosyası okur. "
            "Yol bu dizine göre göreli olarak verilir."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Kök dizine göre göreli dosya yolu."},
                "max_chars": {"type": "integer", "minimum": 1, "default": 20000},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        handler=_read_file_handler,
    )


# Varsayılan örnek: READ_FILE_ROOT ortam değişkeni ya da geçerli çalışma dizini.
read_file = make_read_file_tool(os.environ.get("READ_FILE_ROOT", os.getcwd()))
