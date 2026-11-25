"""Ortam değişkeni enterpolasyonlu YAML yapılandırma yükleme."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

_ENV_PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)(?::-([^}]*))?\}")


def _interpolate(value: Any) -> Any:
    """${VAR} veya ${VAR:-varsayılan} ifadelerini os.environ'dan özyinelemeli olarak doldurur."""
    if isinstance(value, str):
        def _sub(match: re.Match[str]) -> str:
            var, default = match.group(1), match.group(2) or ""
            return os.environ.get(var, default)
        return _ENV_PATTERN.sub(_sub, value)
    if isinstance(value, dict):
        return {k: _interpolate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v) for v in value]
    return value


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Bir YAML dosyasını yükler ve ortam değişkenlerini yerine koyar."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Yapılandırma dosyası bulunamadı: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return _interpolate(raw)
