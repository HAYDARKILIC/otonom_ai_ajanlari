"""Hafta 5 — Çoklu ajan sistemleri ve durum yönetimi.

Üç temel öğe:

- :class:`AgentState` — döngü tespiti için özetlenen (hash) ortak kara tahta.
- :class:`Agent` — role özelleşmiş bir LLM sarmalayıcısı.
- :class:`Orchestrator` — ajanlar arasındaki konuşmayı yürütür ve sonlanmayı
  garanti eder (sınırlı yeniden denemeler, durum özeti tabanlı döngü
  tespiti, açık başarı ölçütleri).
"""

from src.multi_agent.agents import CoderAgent, CriticAgent, ExecutorAgent
from src.multi_agent.orchestrator import SelfCorrectingCoder
from src.multi_agent.state import AgentState

__all__ = ["CoderAgent", "CriticAgent", "ExecutorAgent", "SelfCorrectingCoder", "AgentState"]
