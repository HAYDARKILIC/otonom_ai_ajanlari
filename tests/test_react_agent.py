"""Senaryolu bir FakeLLM kullanarak ReAct ajanının uçtan uca testi."""

from __future__ import annotations

from src.react_agent.agent import ReActAgent
from src.react_agent.tools import calculator


def test_agent_uses_calculator_then_answers(fake_llm_factory):
    """İki adımlı senaryo: hesap makinesini çağır → nihai yanıtı üret."""
    llm = fake_llm_factory([
        # Adım 1: hesapla
        "Thought: I will compute 6 * 7.\nAction: calculator\nAction Input: {\"expression\": \"6 * 7\"}",
        # Adım 2: teslim et
        "Thought: I have the answer.\nFinal Answer: 42",
    ])
    agent = ReActAgent(llm=llm, tools=[calculator], max_steps=5)
    run = agent.run("Compute 6 times 7.")
    assert run.answer == "42"
    assert run.steps == 2
    assert run.halted_reason == "final_answer"
    assert run.trace[0].action == "calculator"
    assert run.trace[0].observation == "42"


def test_agent_halts_on_max_steps(fake_llm_factory):
    # Her zaman hesap makinesini çağırır; asla nihai yanıt üretmez.
    looping = ("Thought: keep going.\n"
               "Action: calculator\n"
               "Action Input: {\"expression\": \"1+1\"}")
    llm = fake_llm_factory([looping] * 20)
    agent = ReActAgent(llm=llm, tools=[calculator], max_steps=3)
    run = agent.run("loop forever")
    assert run.halted_reason == "max_steps"
    assert run.steps == 3


def test_agent_recovers_from_unknown_tool(fake_llm_factory):
    llm = fake_llm_factory([
        # Hatalı araç adı
        "Thought: try ghost tool.\nAction: ghost\nAction Input: {}",
        # Toparlan ve bitir
        "Thought: ghost did not exist; I'll just answer.\nFinal Answer: ok",
    ])
    agent = ReActAgent(llm=llm, tools=[calculator], max_steps=5)
    run = agent.run("recover")
    assert run.answer == "ok"
    assert "ERROR" in (run.trace[0].observation or "")
