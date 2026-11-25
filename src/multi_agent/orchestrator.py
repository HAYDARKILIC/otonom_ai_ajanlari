"""Kendini düzelten kod üretimi orkestratörü.

Topoloji
--------

    Kullanıcı görevi
        │
        ▼
    ┌───────────┐
    │ Kodlayıcı │◀────────────────┐
    └────┬──────┘                 │
         │ kod                    │ revise(kod, hata, eleştiri)
         ▼                        │
    ┌──────────┐                  │
    │ Yürütücü │── sonuç (geçti/kaldı, stderr)
    └────┬─────┘                  │
         │ kaldı                  │
         ▼                        │
    ┌────────────┐                │
    │ Eleştirmen │── eleştiri ────┘
    └────────────┘
         │ geçti
         ▼
       BAŞARI

Sonlanma garantileri
--------------------
- ``max_iterations``, Kodlayıcı revizyonları üzerinde katı bir bütçedir.
- ``AgentState.detect_loop``, tekrarlanan (kod, hata) parmak izlerinde durur;
  Eleştirmen ile Kodlayıcı'nın salındığı hata türünü yakalar.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from src.multi_agent.agents import CoderAgent, CriticAgent, ExecutorAgent
from src.multi_agent.state import AgentState
from src.utils.llm_client import LLMClient, build_client
from src.utils.logging import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class CoderRunResult:
    final_code: str
    success: bool
    iterations: int
    halted_reason: str  # success | max_iterations | loop_detected


class SelfCorrectingCoder:
    """Kodlayıcı–Yürütücü–Eleştirmen döngüsünü koordine eder."""

    def __init__(
        self,
        max_iterations: int = 5,
        timeout_seconds: float = 10.0,
        llm_provider: str | None = None,
        llm_model: str | None = None,
        *,
        llm: LLMClient | None = None,
        coder_llm: LLMClient | None = None,
        critic_llm: LLMClient | None = None,
    ) -> None:
        """LLM'ler dışarıdan verilebilir; verilmezse sağlayıcıdan bir istemci kurulur.

        ``coder_llm`` / ``critic_llm`` her iki rol için farklı modeller (ya da
        testlerde farklı senaryolu sahte LLM'ler) kullanmayı sağlar; verilmeyen
        rol ``llm`` değerine düşer.
        """
        if max_iterations < 1:
            raise ValueError("max_iterations en az 1 olmalıdır.")
        coder_client = coder_llm or llm
        critic_client = critic_llm or llm
        if coder_client is None or critic_client is None:
            provider = llm_provider or os.environ.get("LLM_PROVIDER", "anthropic")
            default = build_client(provider, llm_model)
            coder_client = coder_client or default
            critic_client = critic_client or default
        self.coder = CoderAgent(coder_client)
        self.executor = ExecutorAgent(timeout_seconds=timeout_seconds)
        self.critic = CriticAgent(critic_client)
        self.max_iterations = max_iterations

    def solve(self, task: str) -> CoderRunResult:
        state = AgentState(task=task)
        code = self.coder.initial(task)
        state.code = code
        state.append("coder", code, role_event="initial")

        for iteration in range(1, self.max_iterations + 1):
            state.iteration = iteration
            log.info("multi_agent_iteration", iteration=iteration)

            result = self.executor.run(state.code or "")
            state.append("executor", f"passed={result.passed} rc={result.returncode}")

            if result.passed:
                return CoderRunResult(
                    final_code=state.code or "",
                    success=True,
                    iterations=iteration,
                    halted_reason="success",
                )

            error = result.stderr or result.stdout or "çıktı yok"
            state.last_error = error

            if state.detect_loop():
                log.warning("multi_agent_loop_detected", iteration=iteration)
                return CoderRunResult(
                    final_code=state.code or "",
                    success=False,
                    iterations=iteration,
                    halted_reason="loop_detected",
                )

            if iteration == self.max_iterations:
                break  # son turda revizyon yapılırsa hiç çalıştırılmadan atılırdı

            critique = self.critic.diagnose(state.code or "", error)
            state.append("critic", critique)

            revised = self.coder.revise(task, state.code or "", error, critique)
            state.code = revised
            state.append("coder", revised, role_event="revise")

        return CoderRunResult(
            final_code=state.code or "",
            success=False,
            iterations=self.max_iterations,
            halted_reason="max_iterations",
        )
