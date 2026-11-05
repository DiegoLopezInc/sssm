"""Context-management regimes from Belcak et al. (2026), arXiv:2611.05306.

- ``fha``  Full-History Accumulation: one ever-growing ReAct conversation (normal model use).
- ``eocc`` End-of-Conversation Compaction: FHA, but compact into a semantic state when the
           live context crosses a threshold.
- ``sssm`` Semantic State Space Model loop (single-action compaction): every action happens in
           a fresh conversation that sees only the task and the current natural-language state;
           after the action the same instance writes the updated state, then is discarded.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from .actions import ActionParseError, parse_action
from .llm import ChatClient, Usage
from .tasks import Task, partial_score

ACTION_FORMAT = """Respond in exactly this format and nothing else:
THOUGHT: <one or two sentences of reasoning about the next step>
ACTION: <exactly one tool call>

Example response:
THOUGHT: I have not read any file yet, so I start with the first one.
ACTION: read_file("notes_01.txt")"""

FHA_SYSTEM = """You are an autonomous agent that completes tasks by calling tools.
Tools:
{tools}

{action_format}"""

SSSM_SYSTEM = """You are a Semantic State Space Model agent that completes tasks by calling tools.
You have no memory of earlier turns. Everything known so far is in the SEMANTIC STATE, written
by a previous instance of you. Trust it, take exactly ONE action that makes progress, and
afterwards you will be asked to update the state for the next instance.
Tools:
{tools}

{action_format}"""

STATE_SECTIONS = """<state>
GOAL: <near-verbatim restatement of the task goal>
DONE WHEN: <exact completion criteria>
PLAN: <remaining steps>
FACTS: <every result needed later: values, running totals, collected items, items already processed>
LAST STEP: <the tool call just made> -> <its key outcome>
NEXT: <the exact next tool call, e.g. read_file("notes_02.txt")>
</state>"""

STATE_NUDGE = f"""Now write the updated SEMANTIC STATE. A fresh instance of you will see ONLY the task and
this state, so include every fact still needed (running totals, collected values, what has
been processed) and drop anything irrelevant. Keep it under 150 words. Use exactly this format:
{STATE_SECTIONS}"""

INITIAL_STATE = """GOAL: (see task)
DONE WHEN: (not yet determined)
PLAN: (none yet)
FACTS: (none yet)
LAST STEP: (none - this is the first step)
NEXT: (decide the first action)"""

_STATE_RE = re.compile(r"<state>(.*?)(?:</state>|$)", re.DOTALL | re.IGNORECASE)
MAX_STATE_CHARS = 2500


def extract_state(text: str) -> str:
    m = _STATE_RE.search(text)
    state = (m.group(1) if m else text).strip()
    lines = [ln for ln in state.splitlines() if not re.match(r"\s*(THOUGHT|ACTION)\s*:", ln, re.I)]
    return "\n".join(lines).strip()[:MAX_STATE_CHARS]


def sssm_user_message(task: Task, state: str) -> str:
    return (
        f"TASK:\n{task.prompt}\n\nSEMANTIC STATE (written by the previous instance):\n{state}\n\n"
        "Take exactly one action now."
    )


@dataclass
class Episode:
    mode: str
    model: str
    task: Task
    steps: int = 0
    parse_errors: int = 0
    compactions: int = 0
    usage: Usage = field(default_factory=Usage)
    trace: list[dict] = field(default_factory=list)
    wall_seconds: float = 0.0

    def record(self) -> dict:
        t = self.task
        return {
            "mode": self.mode,
            "model": self.model,
            "family": t.family,
            "horizon": t.horizon,
            "seed": t.seed,
            "correct": t.is_correct(),
            "score": partial_score(t.family, t.submitted, t.answer),
            "submitted": repr(t.submitted),
            "answer": repr(t.answer),
            "steps": self.steps,
            "parse_errors": self.parse_errors,
            "compactions": self.compactions,
            "wall_seconds": round(self.wall_seconds, 2),
            **self.usage.summary(),
            "trace": self.trace,
        }


def _act(task: Task, output: str, ep: Episode) -> tuple[str, bool]:
    try:
        name, args = parse_action(output)
    except ActionParseError as e:
        ep.parse_errors += 1
        return f"ERROR: {e}. {ACTION_FORMAT}", False
    res = task.step(name, args)
    return res.observation, res.done


def default_max_steps(task: Task) -> int:
    return 3 * task.horizon + 6


def run_fha(
    task: Task,
    llm: ChatClient,
    max_steps: int | None = None,
    compact_threshold: int | None = None,
) -> Episode:
    """Full-history ReAct loop; with ``compact_threshold`` it becomes EoCC."""
    ep = Episode("eocc" if compact_threshold else "fha", llm.model, task)
    t0 = time.perf_counter()
    system = {
        "role": "system",
        "content": FHA_SYSTEM.format(tools=task.tools, action_format=ACTION_FORMAT),
    }
    messages = [system, {"role": "user", "content": f"TASK:\n{task.prompt}"}]
    for _ in range(max_steps or default_max_steps(task)):
        output = llm.chat(messages, max_tokens=256, usage=ep.usage)
        obs, done = _act(task, output, ep)
        ep.steps += 1
        ep.trace.append({"output": output, "observation": obs})
        if done:
            break
        messages += [
            {"role": "assistant", "content": output},
            {"role": "user", "content": f"OBSERVATION:\n{obs}"},
        ]
        if compact_threshold and ep.usage.calls[-1].prompt_tokens > compact_threshold:
            messages.append({"role": "user", "content": STATE_NUDGE})
            state = extract_state(llm.chat(messages, max_tokens=400, usage=ep.usage))
            ep.compactions += 1
            ep.trace.append({"state": state})
            messages = [system, {"role": "user", "content": sssm_user_message(task, state)}]
    ep.wall_seconds = time.perf_counter() - t0
    return ep


def run_sssm(task: Task, llm: ChatClient, max_steps: int | None = None) -> Episode:
    """Single-action loop: (1) read state (2) reason (3) act (4) observe (5) rewrite state."""
    ep = Episode("sssm", llm.model, task)
    t0 = time.perf_counter()
    system = {
        "role": "system",
        "content": SSSM_SYSTEM.format(tools=task.tools, action_format=ACTION_FORMAT),
    }
    state = INITIAL_STATE
    for _ in range(max_steps or default_max_steps(task)):
        messages = [system, {"role": "user", "content": sssm_user_message(task, state)}]
        output = llm.chat(messages, max_tokens=256, usage=ep.usage)
        obs, done = _act(task, output, ep)
        ep.steps += 1
        step = {"state_in": state, "output": output, "observation": obs}
        ep.trace.append(step)
        if done:
            break
        messages += [
            {"role": "assistant", "content": output},
            {"role": "user", "content": f"OBSERVATION:\n{obs}\n\n{STATE_NUDGE}"},
        ]
        raw = llm.chat(messages, max_tokens=400, usage=ep.usage)
        state = extract_state(raw)
        step["state_raw"] = raw
        ep.compactions += 1
    ep.wall_seconds = time.perf_counter() - t0
    return ep
