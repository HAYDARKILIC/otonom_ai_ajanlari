"""Hafta 6 — Bitirme Projesi: Otonom Akademik Araştırma Asistanı.

Uçtan uca hat:

    arXiv API ──► MCP server (sqlite store) ──► hierarchical RAG ──►
        cross-encoder ile yeniden sıralama ──► LLM sentezi ──► Markdown literatür taraması
"""

from src.academic_assistant.pipeline import AcademicAssistant

__all__ = ["AcademicAssistant"]
