"""Ebeveyn parça metinleri için kalıcı anahtar-değer deposu.

Hiyerarşik erişimde yalnızca çocuk parçalar vektör deposuna gömülür; ebeveyn
metinleri ise erişim anında ``parent_id`` ile geri getirilmek üzere ayrı tutulur.
Bu eşleme yalnızca bellekte tutulursa indeks bir süreçte (ör.
``scripts/build_index.py``) oluşturulup başka bir süreçte sorgulandığında
ebeveyn genişletmesi sessizce devre dışı kalır. :class:`SQLiteParentStore` bu
eşlemeyi diske yazar ve sıradan bir ``dict`` gibi kullanılabilir.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator, MutableMapping
from pathlib import Path


class SQLiteParentStore(MutableMapping[str, str]):
    """``parent_id -> parent_text`` eşlemesini SQLite'ta saklayan sözlük benzeri depo."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS parents (parent_id TEXT PRIMARY KEY, text TEXT NOT NULL)"
        )
        self._conn.commit()

    def __getitem__(self, key: str) -> str:
        row = self._conn.execute("SELECT text FROM parents WHERE parent_id = ?", (key,)).fetchone()
        if row is None:
            raise KeyError(key)
        return row[0]

    def __setitem__(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO parents (parent_id, text) VALUES (?, ?)", (key, value)
        )
        self._conn.commit()

    def update_many(self, items: dict[str, str]) -> None:
        """Birden çok ebeveyni tek bir işlemde yazar (toplu veri alımında hızlıdır)."""
        self._conn.executemany(
            "INSERT OR REPLACE INTO parents (parent_id, text) VALUES (?, ?)", list(items.items())
        )
        self._conn.commit()

    def __delitem__(self, key: str) -> None:
        cur = self._conn.execute("DELETE FROM parents WHERE parent_id = ?", (key,))
        self._conn.commit()
        if cur.rowcount == 0:
            raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        for (key,) in self._conn.execute("SELECT parent_id FROM parents"):
            yield key

    def __len__(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM parents").fetchone()[0])

    def close(self) -> None:
        conn = getattr(self, "_conn", None)
        if conn is not None:
            conn.close()
            self._conn = None  # type: ignore[assignment]

    def __enter__(self) -> SQLiteParentStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:  # bağlantı açık unutulursa sızıntıyı önle
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass
