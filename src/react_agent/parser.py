"""ReAct metin protokolü için ayrıştırıcı.

Ajan, LLM'den şu biçimde yapılandırılmış bir iz üretmesini ister:

    Thought: <serbest metin>
    Action: <araç_adı>
    Action Input: <JSON nesnesi>

…ya da iş bittiğinde:

    Thought: <serbest metin>
    Final Answer: <serbest metin>

Sağlam bir ayrıştırıcı şarttır: küçük biçim sapmaları döngüyü bozmamalıdır.
Buradaki uygulama boşlukları tolere eder, eylem girdileri için gevşek JSON'u
kabul eder ve ajanın LLM bağlamına düzeltici bir mesaj geri enjekte etmek
için kullandığı yapılandırılmış bir :class:`ParseError` üretir.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass


class ParseError(ValueError):
    """Bir ReAct adımı ayrıştırılamadığında fırlatılır."""


@dataclass(slots=True)
class Step:
    thought: str
    is_final: bool
    action: str | None = None
    action_input: dict | None = None
    final_answer: str | None = None


_THOUGHT = re.compile(r"Thought:\s*(.+?)(?=\n(?:Action|Final Answer):|\Z)", re.DOTALL)
_ACTION = re.compile(r"Action:\s*(.+?)(?=\n)", re.DOTALL)
_ACTION_INPUT = re.compile(r"Action Input:\s*(.+?)(?=\nObservation:|\nThought:|\Z)", re.DOTALL)
_FINAL = re.compile(r"Final Answer:\s*(.+)\Z", re.DOTALL)


def parse_step(text: str) -> Step:
    """Tek bir LLM turunu bir :class:`Step` nesnesine ayrıştırır."""
    thought_match = _THOUGHT.search(text)
    if thought_match is None:
        raise ParseError("`Thought:` alanı eksik.")
    thought = thought_match.group(1).strip()

    final_match = _FINAL.search(text)
    if final_match is not None:
        return Step(thought=thought, is_final=True, final_answer=final_match.group(1).strip())

    action_match = _ACTION.search(text)
    if action_match is None:
        raise ParseError("`Action:` ve `Final Answer:` eksik — en az biri gereklidir.")
    action = action_match.group(1).strip()

    input_match = _ACTION_INPUT.search(text)
    if input_match is None:
        raise ParseError(f"{action!r} eylemi var ancak `Action Input:` verilmemiş.")

    raw_input = input_match.group(1).strip()
    raw_input = raw_input.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        action_input = json.loads(raw_input)
    except json.JSONDecodeError as exc:
        raise ParseError(f"Action Input geçerli bir JSON değil: {exc}") from exc
    if not isinstance(action_input, dict):
        raise ParseError("Action Input bir JSON nesnesi olmalıdır.")

    return Step(thought=thought, is_final=False, action=action, action_input=action_input)
