"""Çoklu ajan iş birliği için ortak durum.

Durum, ajanlar arasında aktarılan ve her turdan sonra damgalanan
yapılandırılmış bir nesnedir. Her turda içeriğin SHA-256 özeti hesaplanır;
aynı özet tekrarlanırsa sistem verimsiz bir döngüye girmiştir ve orkestratör
işlemi iptal eder. Bu, en yaygın çoklu ajan hata türüne karşı basit ama etkili
bir korumadır: iki ajanın birbiriyle çelişen iki konum arasında salınması.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Turn:
    role: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentState:
    """Çoklu ajan döngüsü boyunca paylaşılan değiştirilebilir kara tahta."""

    task: str
    history: list[Turn] = field(default_factory=list)
    seen_hashes: set[str] = field(default_factory=set)
    code: str | None = None
    test_passed: bool = False
    last_error: str | None = None
    iteration: int = 0

    def append(self, role: str, content: str, **meta: Any) -> None:
        self.history.append(Turn(role=role, content=content, metadata=meta))

    def fingerprint(self) -> str:
        """Döngü tespiti için *verimli* içeriğin (kod + son hata) özeti."""
        material = f"{self.code or ''}|{self.last_error or ''}".encode("utf-8")
        return hashlib.sha256(material).hexdigest()

    def detect_loop(self) -> bool:
        """Geçerli parmak izi daha önce görüldüyse True döndürür."""
        fp = self.fingerprint()
        if fp in self.seen_hashes:
            return True
        self.seen_hashes.add(fp)
        return False

    def summary(self) -> str:
        return "\n".join(f"[{t.role}] {t.content[:200]}" for t in self.history[-5:])
