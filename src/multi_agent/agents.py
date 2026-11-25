"""Kendini düzelten kodlayıcı topolojisi için role özelleşmiş ajanlar.

Üç rol:

- :class:`CoderAgent`    — bir Python çözümü yazar / revize eder.
- :class:`ExecutorAgent` — kodu bir alt süreçte çalıştırır ve stdout / stderr /
                           çıkış kodunu yakalar. Bir LLM *değildir*.
- :class:`CriticAgent`   — bir hata izini inceler ve doğal dilde bir düzeltme
                           stratejisi önerir.

Alt süreçte yürütme
-------------------
Yürütücü, katı bir zaman aşımıyla ``subprocess.run`` kullanır. Bu,
sağlamlaştırılmış bir korumalı alan değildir — gerçek yalıtım için bir
konteyner çalışma zamanıyla (ör. ``docker``, ``firejail`` veya bir bulut
korumalı alanı) değiştirin. Arayüz, bu değişikliğin yerel kalacağı şekilde
tasarlanmıştır.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from src.utils.llm_client import LLMClient, Message

_CODE_FENCE = re.compile(r"```python\n(.*?)```", re.DOTALL)


def _extract_code(text: str) -> str:
    """Bir LLM yanıtından en büyük Python kod bloğunu çıkarır."""
    matches = _CODE_FENCE.findall(text)
    if not matches:
        return text.strip()
    return max(matches, key=len).strip()


# ---------------------------------------------------------------------- #
# Kodlayıcı (Coder)                                                       #
# ---------------------------------------------------------------------- #

_CODER_SYS_INITIAL = """Uzman bir Python mühendisisin. Kullanıcının tanımladığı görevi
çözen, eksiksiz ve kendi kendine yeten bir Python programı yaz.
Kısıtlar:
- Program, argümansız olarak `python solution.py` şeklinde çalıştırılabilmelidir.
- Başarı durumunda açık bir başarı göstergesi yazdırmalıdır (ör. `print("OK: ...")`).
- Doğruluğu teyit edecek yeterli öz-denetim / assertion içermelidir.
- NİHAİ programı tek bir ```python ... ``` kod bloğu içine al."""

_CODER_SYS_REVISE = """Başarısız olan bir programı düzelten uzman bir Python mühendisisin.
Şunları alacaksın:
  - önceki deneme (Python kaynak kodu)
  - yakalanan stderr / hata mesajı
  - bir eleştirmenin teşhisi
Hatayı gideren düzeltilmiş bir program üret.
NİHAİ programı tek bir ```python ... ``` kod bloğu içine al."""


@dataclass
class CoderAgent:
    llm: LLMClient

    def initial(self, task: str) -> str:
        text = self.llm.complete(
            [Message("system", _CODER_SYS_INITIAL), Message("user", task)],
            max_tokens=1500, temperature=0.2,
        )
        return _extract_code(text)

    def revise(self, task: str, code: str, error: str, critique: str) -> str:
        user = (
            f"GÖREV:\n{task}\n\n"
            f"ÖNCEKİ DENEME:\n```python\n{code}\n```\n\n"
            f"HATA:\n{error}\n\n"
            f"ELEŞTİRİ:\n{critique}\n"
        )
        text = self.llm.complete(
            [Message("system", _CODER_SYS_REVISE), Message("user", user)],
            max_tokens=1500, temperature=0.3,
        )
        return _extract_code(text)


# ---------------------------------------------------------------------- #
# Yürütücü (Executor; LLM değildir)                                      #
# ---------------------------------------------------------------------- #

@dataclass(slots=True)
class ExecutionResult:
    passed: bool
    stdout: str
    stderr: str
    returncode: int


class ExecutorAgent:
    """Python kodunu katı bir zaman aşımıyla bir alt süreçte çalıştırır."""

    def __init__(self, timeout_seconds: float = 10.0) -> None:
        self.timeout = timeout_seconds

    def run(self, code: str) -> ExecutionResult:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "solution.py"
            path.write_text(code, encoding="utf-8")
            try:
                proc = subprocess.run(
                    [sys.executable, str(path)],
                    capture_output=True, text=True,
                    timeout=self.timeout, check=False,
                )
            except subprocess.TimeoutExpired as exc:
                stdout = exc.stdout or ""
                if isinstance(stdout, bytes):
                    stdout = stdout.decode("utf-8", errors="replace")
                return ExecutionResult(
                    passed=False,
                    stdout=_normalize_paths(stdout, tmpdir),
                    stderr=f"TIMEOUT: süreç {self.timeout} saniyeyi aştı",
                    returncode=-1,
                )
            return ExecutionResult(
                passed=(proc.returncode == 0),
                stdout=_normalize_paths(proc.stdout, tmpdir),
                stderr=_normalize_paths(proc.stderr, tmpdir),
                returncode=proc.returncode,
            )


def _normalize_paths(text: str, tmpdir: str) -> str:
    """Her çalıştırmada değişen geçici dizin yolunu sabit bir yer tutucuyla değiştirir.

    Hata izleri (traceback) ``/tmp/tmpab12cd/solution.py`` gibi rastgele yollar
    içerir. Bunlar ayıklanmazsa aynı hata her turda farklı görünür ve
    :meth:`AgentState.detect_loop` tıkanmış bir döngüyü asla yakalayamaz.
    """
    import os

    for variant in {tmpdir, os.path.realpath(tmpdir)}:
        text = text.replace(variant + os.sep, "").replace(variant, "<tmp>")
    return text


# ---------------------------------------------------------------------- #
# Eleştirmen (Critic)                                                     #
# ---------------------------------------------------------------------- #

_CRITIC_SYS = """Kıdemli bir kod gözden geçiricisisin. Başarısız olan bir Python
programını ve hata çıktısını alırsın. (Yalnızca belirtiyi değil) *kök nedeni*
teşhis et ve 1-3 cümlelik öz bir düzeltme stratejisi öner. Kod yazma."""


@dataclass
class CriticAgent:
    llm: LLMClient

    def diagnose(self, code: str, error: str) -> str:
        user = (
            f"KOD:\n```python\n{code}\n```\n\nHATA:\n{error}\n"
        )
        return self.llm.complete(
            [Message("system", _CRITIC_SYS), Message("user", user)],
            max_tokens=400, temperature=0.2,
        )
