"""ReAct ajanı: sonlu döngülü bir Düşünce-Eylem-Gözlem yürütücüsü.

Bu saf Python bir uygulamadır. Ajan çatısı yoktur. Kontrol akışı şunları
yapan bir `while` döngüsüdür:

    1. LLM'den bir sonraki adımı ister,
    2. yanıtı bir :class:`Step` nesnesine ayrıştırır,
    3. adım nihai değilse aracı çalıştırır,
    4. gözlemi sohbete ekler,
    5. :class:`Step.is_final` olduğunda veya ``max_steps`` tükendiğinde
       sonlanır.

Hata türleri ve önlemler
------------------------
- **Ayrıştırılamayan LLM çıktısı** → :class:`ParseError` yakalanır; düzeltici
  bir kullanıcı mesajı enjekte edilir ve döngü küçük bir bütçeyle yeniden dener.
- **Araç hatası** → LLM'in toparlanabilmesi için istisna bir Gözleme
  dönüştürülür. Araç *çökmeleri* (programcı hataları) yukarı iletilir.
- **Sonsuz döngüler** → ``max_steps`` ile sınırlandırılır.
- **Eylem halüsinasyonu** (eylem adı kayıt defterinde yok) → araç hatası gibi
  ele alınır.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.mcp_core.tool_registry import Tool, ToolRegistry
from src.react_agent.parser import ParseError, Step, parse_step
from src.utils.llm_client import LLMClient, Message
from src.utils.logging import get_logger

log = get_logger(__name__)


_SYSTEM_PROMPT = """Problemleri Thought (Düşünce), Action (Eylem) ve Observation
(Gözlem) adımlarını iç içe geçirerek çözen, dikkatli bir akıl yürütme ajanısın.

Aşağıdaki araçlara erişimin var:
{tool_block}

Yanıtını aşağıdaki iki biçimden TAM OLARAK birinde vermelisin:

BİÇİM A (bir araç kullan):
Thought: <akıl yürütmen>
Action: <yukarıdaki listeden araç adı>
Action Input: <aracın girdi şemasına uyan tek bir JSON nesnesi>

BİÇİM B (işin bitti):
Thought: <akıl yürütmen>
Final Answer: <kullanıcıya nihai yanıtın>

Kesin kurallar:
- Bu iki biçimin dışında hiçbir şey yazma.
- Alan etiketlerini (Thought, Action, Action Input, Final Answer) çevirmeden aynen kullan.
- Action Input geçerli bir JSON olmalı, mümkünse tek satırda.
- Asla araç adı uydurma. Yalnızca yukarıda listelenen araçları kullan.
- Yanıt vermek için yeterli bilgiye sahip olduğunda BİÇİM B'ye geç.
"""


def _format_tool_block(tools: ToolRegistry) -> str:
    lines = []
    for descriptor in tools.list_all():
        lines.append(f"- {descriptor['name']}: {descriptor['description']}")
        lines.append(f"  girdi şeması: {descriptor['inputSchema']}")
    return "\n".join(lines)


@dataclass(slots=True)
class TraceEntry:
    """Ajanın yürütme izinin (trace) tek bir adımı."""

    thought: str
    action: str | None
    action_input: dict | None
    observation: str | None
    is_final: bool
    final_answer: str | None = None


@dataclass(slots=True)
class AgentRun:
    """Tek bir ``run`` çağrısının tam sonucu."""

    answer: str
    trace: list[TraceEntry] = field(default_factory=list)
    steps: int = 0
    halted_reason: str = "final_answer"  # final_answer | max_steps | parse_failure


class ReActAgent:
    """Saf Python bir ReAct ajanı."""

    def __init__(
        self,
        llm: LLMClient,
        tools: list[Tool],
        max_steps: int = 10,
        parse_retry_budget: int = 2,
    ) -> None:
        self.llm = llm
        self.tools = ToolRegistry()
        for tool in tools:
            self.tools.register(tool)
        self.max_steps = max_steps
        self.parse_retry_budget = parse_retry_budget

    def run(self, task: str) -> AgentRun:
        system = _SYSTEM_PROMPT.format(tool_block=_format_tool_block(self.tools))
        history: list[Message] = [Message("system", system), Message("user", task)]
        trace: list[TraceEntry] = []

        for step_idx in range(1, self.max_steps + 1):
            try:
                step = self._llm_step(history)
            except ParseError as exc:
                log.warning("react_parse_error", step=step_idx, error=str(exc))
                return AgentRun(
                    answer=f"[ajan durduruldu: {exc}]",
                    trace=trace,
                    steps=step_idx,
                    halted_reason="parse_failure",
                )

            log.info(
                "react_step",
                step=step_idx,
                action=step.action,
                is_final=step.is_final,
            )

            if step.is_final:
                trace.append(
                    TraceEntry(
                        thought=step.thought, action=None, action_input=None,
                        observation=None, is_final=True,
                        final_answer=step.final_answer,
                    )
                )
                return AgentRun(
                    answer=step.final_answer or "",
                    trace=trace,
                    steps=step_idx,
                    halted_reason="final_answer",
                )

            assert step.action is not None and step.action_input is not None
            observation = self._execute_action(step.action, step.action_input)
            trace.append(
                TraceEntry(
                    thought=step.thought, action=step.action,
                    action_input=step.action_input,
                    observation=observation, is_final=False,
                )
            )

            # Asistan turunu ve gözlemi sohbet geçmişine ekle.
            history.append(Message("assistant", self._render_step(step)))
            history.append(Message("user", f"Observation: {observation}"))

        return AgentRun(
            answer="[ajan durduruldu: max_steps aşıldı]",
            trace=trace,
            steps=self.max_steps,
            halted_reason="max_steps",
        )

    # ------------------------------------------------------------------ #
    # İç işleyiş                                                         #
    # ------------------------------------------------------------------ #

    def _llm_step(self, history: list[Message]) -> Step:
        last_error: ParseError | None = None
        for attempt in range(self.parse_retry_budget + 1):
            text = self.llm.complete(
                history if attempt == 0 else history + [
                    Message("user", f"Son yanıtın ayrıştırılamadı: {last_error}. "
                                    "Lütfen gerekli biçimde yeniden yanıt ver.")
                ],
                max_tokens=1024,
                temperature=0.2,
                stop=["Observation:"],
            )
            try:
                return parse_step(text)
            except ParseError as exc:
                last_error = exc
        assert last_error is not None
        raise last_error

    def _execute_action(self, action: str, action_input: dict) -> str:
        if action not in self.tools:
            return f"ERROR: bilinmeyen araç {action!r}. Mevcut araçlar: {list(self.tools._tools.keys())}"
        try:
            result = self.tools.get(action).invoke(action_input)
        except Exception as exc:  # noqa: BLE001
            return f"ERROR: {type(exc).__name__}: {exc}"
        if isinstance(result, str):
            return result
        import json
        return json.dumps(result, ensure_ascii=False, default=str)

    @staticmethod
    def _render_step(step: Step) -> str:
        import json
        return (
            f"Thought: {step.thought}\n"
            f"Action: {step.action}\n"
            f"Action Input: {json.dumps(step.action_input)}"
        )
