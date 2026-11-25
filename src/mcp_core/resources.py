"""MCP kaynak sağlayıcıları: yerel verilere salt okunur görünümler.

Bir MCP *kaynağı* (resource), istemcinin bağlama okuyabileceği, URI ile
adreslenebilen bir öğedir. İki sağlayıcı uygulanmıştır:

- :class:`FilesystemResources` — bir korumalı alan (sandbox) kökünün altındaki
  yollar; sembolik bağların korumalı alandan kaçamaması için
  :func:`os.path.realpath` ile denetlenir.
- :class:`SQLiteResources` — bir SQLite veritabanındaki tabloların
  ``sqlite://<tablo>`` URI'leri olarak sunulan salt okunur görünümleri.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

from src.mcp_core.protocol import INVALID_PARAMS, PERMISSION_DENIED, RESOURCE_NOT_FOUND, MCPError


class FilesystemResources:
    """Bir korumalı alan kökünün altındaki dosyaların salt okunur görünümünü sunar."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).resolve()
        if not self._root.is_dir():
            raise ValueError(f"Dosya sistemi kökü {self._root} bir dizin değil.")

    def list_resources(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for path in self._root.rglob("*"):
            if path.is_file():
                rel = path.relative_to(self._root)
                out.append(
                    {
                        "uri": f"file://{rel.as_posix()}",
                        "name": rel.name,
                        "mimeType": _mime_for(path),
                    }
                )
        return out

    def read(self, uri: str) -> str:
        if not uri.startswith("file://"):
            raise MCPError(INVALID_PARAMS, f"URI'de desteklenmeyen şema: {uri!r}")
        rel = uri[len("file://"):]
        target = (self._root / rel).resolve()
        # Korumalı alan denetimi: ../ kaçışlarını ve kökün dışını gösteren sembolik bağları reddeder.
        if os.path.commonpath([str(target), str(self._root)]) != str(self._root):
            raise MCPError(PERMISSION_DENIED, f"Yol korumalı alanın dışına çıkıyor: {uri!r}")
        if not target.is_file():
            raise MCPError(RESOURCE_NOT_FOUND, f"Kaynak bulunamadı: {uri!r}", data={"uri": uri})
        return target.read_text(encoding="utf-8", errors="replace")


class SQLiteResources:
    """Bir SQLite veritabanının tablolarını salt okunur ``sqlite://<tablo>`` URI'leri olarak sunar."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        # `mode=ro`, URI biçimi aracılığıyla veritabanını salt okunur açar.
        self._conn = sqlite3.connect(
            f"file:{self._db_path}?mode=ro", uri=True, check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row

    def list_resources(self) -> list[dict[str, Any]]:
        cur = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
        return [
            {
                "uri": f"sqlite://{row['name']}",
                "name": row["name"],
                "mimeType": "application/json",
            }
            for row in cur.fetchall()
        ]

    def close(self) -> None:
        conn = getattr(self, "_conn", None)
        if conn is not None:
            conn.close()
            self._conn = None  # type: ignore[assignment]

    def __del__(self) -> None:  # bağlantı açık unutulursa sızıntıyı önle
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

    def read(self, uri: str, limit: int = 100) -> list[dict[str, Any]]:
        if not uri.startswith("sqlite://"):
            raise MCPError(INVALID_PARAMS, f"URI'de desteklenmeyen şema: {uri!r}")
        table = uri[len("sqlite://"):]
        # Tablo adı SQL'e parametre olarak verilemez; bu yüzden yalnızca veritabanında
        # gerçekten var olan (ve sistem tablosu olmayan) adlara izin verilir.
        known = {r["name"] for r in self.list_resources()}
        if table not in known:
            raise MCPError(RESOURCE_NOT_FOUND, f"Kaynak bulunamadı: {uri!r}", data={"uri": uri})
        quoted = '"' + table.replace('"', '""') + '"'
        cur = self._conn.execute(f"SELECT * FROM {quoted} LIMIT ?", (limit,))  # noqa: S608
        return [dict(row) for row in cur.fetchall()]


_MIME_BY_EXT = {
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".py": "text/x-python",
    ".json": "application/json",
    ".yaml": "application/yaml",
    ".yml": "application/yaml",
}


def _mime_for(path: Path) -> str:
    return _MIME_BY_EXT.get(path.suffix.lower(), "application/octet-stream")
