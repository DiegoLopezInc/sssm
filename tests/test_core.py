import pytest

from sssm.actions import ActionParseError, parse_action
from sssm.agents import Episode, extract_state, run_fha, run_sssm
from sssm.llm import CallStats, Usage
from sssm.tasks import grade, make_task, safe_eval


def test_parse_action_variants():
    assert parse_action('THOUGHT: x\nACTION: read_file("a.txt")') == ("read_file", ["a.txt"])
    assert parse_action("ACTION: `list_files()`") == ("list_files", [])
    assert parse_action('ACTION: submit({"apple": 2})') == ("submit", [{"apple": 2}])
    assert parse_action("ACTION: read_file(name='b.txt')") == ("read_file", ["b.txt"])
    with pytest.raises(ActionParseError):
        parse_action("I will read the file")


def test_safe_eval():
    assert safe_eval("12 + 7 - (3 * 2)") == 13
    with pytest.raises(ValueError):
        safe_eval("__import__('os')")


@pytest.mark.parametrize("family", ["ledger", "pointer", "cart"])
def test_tasks_deterministic_and_gradable(family):
    a, b = make_task(family, 6, 3), make_task(family, 6, 3)
    assert a.prompt == b.prompt and a.answer == b.answer
    assert grade(family, a.answer, a.answer)


def test_pointer_chain_solvable():
    t = make_task("pointer", 5, 0)
    start = t.prompt.split("reading ")[1].split()[0].rstrip(".")
    name, letters = start, ""
    while True:
        body = t.tool_read_file(name)
        letters += body.split("Code letter: ")[1][0]
        nxt = body.split("Next: ")[1].strip()
        if nxt == "END":
            break
        name = nxt
    assert letters == t.answer


def test_cart_grading_normalizes():
    t = make_task("cart", 6, 1)
    plural = {k + "s": v for k, v in t.answer.items()}
    assert grade("cart", plural, t.answer)
    assert grade("cart", repr(t.answer), t.answer)


def test_extract_state():
    assert extract_state("blah <state>\nGOAL: x\n</state> tail") == "GOAL: x"
    assert extract_state("<state>GOAL: y") == "GOAL: y"


class ScriptedLLM:
    """Oracle that solves a ledger task by reading files and doing arithmetic in its state."""

    model = "scripted"

    def __init__(self, task):
        self.task = task
        self.i = 0
        self.total = 0

    def chat(self, messages, max_tokens=512, usage=None):
        if usage is not None:
            usage.add(CallStats(sum(len(m["content"]) for m in messages) // 4, 10, 0.0))
        last = messages[-1]["content"]
        if "Now write the updated SEMANTIC STATE" in last:
            return f"<state>\nFACTS: processed {self.i} files, total {self.total}\n</state>"
        if last.startswith("OBSERVATION") and "Now write" not in last:
            pass
        n = self.task.horizon
        if self.i < n:
            self.i += 1
            name = f"ledger_{self.i:02d}.txt"
            target = self.task.prompt.split("account ")[1].split()[0]
            body = self.task.files[name]
            self.total += sum(int(l.split("|")[2]) for l in body.splitlines()[1:] if f"| {target} |" in l)
            return f'THOUGHT: read next\nACTION: read_file("{name}")'
        return f"THOUGHT: done\nACTION: submit({self.total})"


@pytest.mark.parametrize("runner", [run_fha, run_sssm])
def test_loops_with_oracle(runner):
    task = make_task("ledger", 4, 0)
    ep = runner(task, ScriptedLLM(task))
    assert isinstance(ep, Episode)
    assert task.is_correct()
    assert ep.steps == 5


def test_sssm_context_stays_bounded():
    task_long = make_task("ledger", 12, 0)
    ep_fha = run_fha(task_long, ScriptedLLM(task_long))
    task_long2 = make_task("ledger", 12, 0)
    ep_sssm = run_sssm(task_long2, ScriptedLLM(task_long2))
    assert ep_sssm.usage.peak_context < ep_fha.usage.peak_context


def test_usage_summary():
    u = Usage()
    u.add(CallStats(100, 10, 1.0))
    u.add(CallStats(50, 5, 0.5))
    assert u.summary()["peak_context"] == 110


@pytest.mark.parametrize("family", ["ledger", "pointer", "cart"])
@pytest.mark.parametrize("horizon", [3, 9])
def test_gold_episodes_solve_tasks(family, horizon):
    import random

    from sssm.train.gold import fha_example, gold_episode, sssm_examples

    task = make_task(family, horizon, 7)
    steps = gold_episode(task, random.Random(0))
    assert task.is_correct()
    ex = sssm_examples(task, steps)
    assert len(ex) == 2 * len(steps) - 1
    assert all(e["messages"][-1]["role"] == "assistant" for e in ex)
    assert parse_action(ex[0]["messages"][-1]["content"])[0] in {"read_file", "next_message"}
    assert fha_example(task, steps)["messages"][-1]["content"].count("ACTION: submit") == 1


def test_partial_score():
    from sssm.tasks import partial_score

    assert partial_score("pointer", "ABCD", "ABCD") == 1.0
    assert 0.5 < partial_score("pointer", "ABDC", "ABCD") < 1.0
    assert partial_score("ledger", None, 5) == 0.0
    assert 0 < partial_score("ledger", "-40", -44) < 1
    assert partial_score("cart", "{'tea': 1, 'rice': 2}", {"tea": 1, "rice": 1}) == pytest.approx(0.99 / 3)
