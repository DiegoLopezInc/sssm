"""Generate gold (oracle) trajectories and turn them into SFT data for SSSM or FHA formats.

Both formats are built from the *same* oracle episodes, so the fine-tuned SSSM and the
fine-tuned full-history model see identical task knowledge - only the context regime differs.

    python -m sssm.train.gold --families ledger,pointer --horizons 3-10 --episodes 60 \
        --format sssm --out data/sssm_train.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

from ..agents import (
    ACTION_FORMAT,
    FHA_SYSTEM,
    INITIAL_STATE,
    SSSM_SYSTEM,
    STATE_NUDGE,
    sssm_user_message,
)
from ..tasks import Task, make_task


@dataclass
class GoldStep:
    sssm_thought: str
    fha_thought: str
    action: str
    observation: str = ""
    state: str = ""


def _state(goal: str, done: str, plan: str, facts: str, last: str, nxt: str) -> str:
    return (
        f"GOAL: {goal}\nDONE WHEN: {done}\nPLAN: {plan}\nFACTS: {facts}\n"
        f"LAST STEP: {last}\nNEXT: {nxt}"
    )


def _exec(task: Task, action: str) -> str:
    from ..actions import parse_action

    name, args = parse_action(f"ACTION: {action}")
    return task.step(name, args).observation


def gold_ledger(task: Task, rng: random.Random) -> list[GoldStep]:
    t, n = task.meta["target"], task.horizon
    goal = (
        f"Final balance of account {t} after all {t} transactions in ledger_01.txt..ledger_{n:02d}.txt, "
        "starting from 0; submit it as an integer."
    )
    done = f"All {n} files processed and the {t} balance submitted."
    steps, bal, carry = [], 0, ""
    for k in range(1, n + 1):
        name = f"ledger_{k:02d}.txt"
        action = f'read_file("{name}")'
        if k == 1:
            st = "Nothing has been processed yet, so I read the first ledger file."
        else:
            st = rng.choice([
                f"Files 01-{k - 1:02d} are done and the {t} balance is {bal}, so I read {name}.",
                f"The state says {k - 1} of {n} files are processed (balance {bal}); next is {name}.",
            ])
        ft = (carry + " " if carry else "") + f"Next I read {name}."
        step = GoldStep(st, ft.strip(), action)
        step.observation = _exec(task, action)
        amts = [int(ln.split("|")[2]) for ln in step.observation.splitlines()[1:] if f"| {t} |" in ln]
        expr = f"{bal} " + " ".join(f"{a:+d}" for a in amts)
        new = bal + sum(amts)
        upd = f"{t} amounts {', '.join(f'{a:+d}' for a in amts)}; {expr} = {new}"
        carry = f"{name} has {upd}."
        bal = new
        plan = (
            f"Read ledger_{k + 1:02d}.txt to ledger_{n:02d}.txt, add each {t} amount, then submit."
            if k < n
            else "Submit the balance."
        )
        nxt = f'read_file("ledger_{k + 1:02d}.txt")' if k < n else f"submit({bal})"
        facts = f"Processed {k} of {n} files (ledger_01..ledger_{k:02d}). Running {t} balance = {bal}."
        step.state = _state(goal, done, plan, facts, f"{action} -> {upd}", nxt)
        steps.append(step)
    final = f"submit({bal})"
    steps.append(GoldStep(
        f"All {n} files are processed and the {t} balance is {bal}, so I submit it.",
        f"{carry} All {n} files are processed, so I submit the balance {bal}.",
        final,
    ))
    return steps


def gold_pointer(task: Task, rng: random.Random) -> list[GoldStep]:
    start = task.meta["start"]
    goal = (
        f"Follow the node chain starting at {start}, collecting each code letter in order until a file "
        "says 'Next: END' (include its letter); submit the letters as one string."
    )
    done = "A file said 'Next: END' and the collected letters were submitted."
    steps, letters, current, carry = [], "", start, ""
    while True:
        action = f'read_file("{current}")'
        if not letters:
            st = f"Nothing has been read yet; the task says to start at {start}."
        else:
            st = rng.choice([
                f"I have collected {letters} so far and the last file pointed to {current}, so I read it.",
                f"Letters so far: {letters}. The next file in the chain is {current}.",
            ])
        ft = (carry + " " if carry else "") + f"Next I read {current}."
        step = GoldStep(st, ft.strip(), action)
        step.observation = _exec(task, action)
        letter = step.observation.split("Code letter: ")[1][0]
        nxt_file = step.observation.split("Next: ")[1].strip()
        letters += letter
        upd = f"letter {letter}, Next: {nxt_file}. Letters so far: {letters}"
        carry = f"{current} has {upd}."
        end = nxt_file == "END"
        plan = "Submit the collected letters." if end else f"Read {nxt_file} and keep following the chain."
        nxt = f'submit("{letters}")' if end else f'read_file("{nxt_file}")'
        facts = f"Collected letters in order: {letters} ({len(letters)} files read). " + (
            "The chain has ended." if end else f"{current} points to {nxt_file}."
        )
        step.state = _state(goal, done, plan, facts, f"{action} -> {upd}", nxt)
        steps.append(step)
        if end:
            break
        current = nxt_file
    steps.append(GoldStep(
        f"The chain has ended, so I submit the collected letters {letters}.",
        f"{carry} The chain has ended, so I submit {letters}.",
        f'submit("{letters}")',
    ))
    return steps


def gold_cart(task: Task, rng: random.Random) -> list[GoldStep]:
    n = task.horizon
    goal = (
        "Read customer messages one at a time with next_message(), track the cart, and when no messages "
        "remain submit the final cart as a dict of item -> quantity (omit zeros)."
    )
    done = f"All {n} messages read and the final cart submitted."
    steps, cart, carry = [], {}, ""

    def js(c: dict) -> str:
        return json.dumps({k: v for k, v in sorted(c.items()) if v > 0})

    for k in range(1, n + 1):
        action = "next_message()"
        if k == 1:
            st = "No messages have been read yet, so I read the first one."
        else:
            st = rng.choice([
                f"I have read {k - 1} of {n} messages and the cart is {js(cart)}, so I read the next one.",
                f"Cart so far: {js(cart)} after {k - 1} messages; more messages remain.",
            ])
        ft = (carry + " " if carry else "") + "Next I read the next message."
        step = GoldStep(st, ft.strip(), action)
        step.observation = _exec(task, action)
        msg = step.observation.split(": ", 1)[1]
        words = msg.replace(",", "").replace(".", "").split()
        if msg.startswith("Please add"):
            q, item = int(words[2]), words[4]
            before = cart.get(item, 0)
            cart[item] = before + q
            how = f"{item} {before} + {q} = {cart[item]}"
        elif msg.startswith("Remove"):
            q, item = int(words[1]), words[3]
            before = cart.get(item, 0)
            cart[item] = before - q
            how = f"{item} {before} - {q} = {cart[item]}"
        else:
            item, q = words[3], int(words[-1])
            cart[item] = q
            how = f"{item} set to {q}"
        upd = f'message {k}/{n} "{msg}"; {how}; cart now {js(cart)}'
        carry = f"Message {k} said \"{msg}\" so {how}; cart is {js(cart)}."
        last = k == n
        plan = "Submit the final cart." if last else f"Read the remaining {n - k} messages; update the cart."
        nxt = f"submit({js(cart)})" if last else "next_message()"
        facts = f"Read {k} of {n} messages. Cart: {js(cart)}."
        step.state = _state(goal, done, plan, facts, f"{action} -> {upd}", nxt)
        steps.append(step)
    steps.append(GoldStep(
        f"All {n} messages are read and the cart is {js(cart)}, so I submit it.",
        f"{carry} That was the last of {n} messages, so I submit the cart.",
        f"submit({js(cart)})",
    ))
    return steps


GOLD = {"ledger": gold_ledger, "pointer": gold_pointer, "cart": gold_cart}


def gold_episode(task: Task, rng: random.Random) -> list[GoldStep]:
    steps = GOLD[task.family](task, rng)
    _exec(task, steps[-1].action)
    assert task.is_correct(), (task.family, task.submitted, task.answer)
    return steps


def sssm_examples(task: Task, steps: list[GoldStep]) -> list[dict]:
    """Two examples per step: the action turn and the state-update turn."""
    system = {"role": "system", "content": SSSM_SYSTEM.format(tools=task.tools, action_format=ACTION_FORMAT)}
    out, state = [], INITIAL_STATE
    for s in steps:
        msgs = [system, {"role": "user", "content": sssm_user_message(task, state)}]
        act = {"role": "assistant", "content": f"THOUGHT: {s.sssm_thought}\nACTION: {s.action}"}
        out.append({"kind": "sssm_action", "messages": msgs + [act]})
        if s.state:
            nudge = {"role": "user", "content": f"OBSERVATION:\n{s.observation}\n\n{STATE_NUDGE}"}
            upd = {"role": "assistant", "content": f"<state>\n{s.state}\n</state>"}
            out.append({"kind": "sssm_state", "messages": msgs + [act, nudge, upd]})
            state = s.state
    return out


def fha_example(task: Task, steps: list[GoldStep]) -> dict:
    msgs = [
        {"role": "system", "content": FHA_SYSTEM.format(tools=task.tools, action_format=ACTION_FORMAT)},
        {"role": "user", "content": f"TASK:\n{task.prompt}"},
    ]
    for s in steps:
        msgs.append({"role": "assistant", "content": f"THOUGHT: {s.fha_thought}\nACTION: {s.action}"})
        if s.state:
            msgs.append({"role": "user", "content": f"OBSERVATION:\n{s.observation}"})
    return {"kind": "fha", "messages": msgs, "train_on": "all_assistant"}


def _range(spec: str) -> list[int]:
    if "-" in spec:
        lo, hi = map(int, spec.split("-"))
        return list(range(lo, hi + 1))
    return [int(x) for x in spec.split(",")]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--families", default="ledger,pointer")
    p.add_argument("--horizons", default="3-10", help="e.g. 3-10 or 4,8")
    p.add_argument("--episodes", type=int, default=60, help="episodes per family")
    p.add_argument("--seed-offset", type=int, default=10_000, help="keeps train seeds disjoint from eval")
    p.add_argument("--format", choices=["sssm", "fha"], required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    rng = random.Random(0)
    horizons = _range(a.horizons)
    rows = []
    for fam in a.families.split(","):
        for i in range(a.episodes):
            task = make_task(fam, rng.choice(horizons), a.seed_offset + i)
            steps = gold_episode(task, rng)
            fresh = make_task(fam, task.horizon, task.seed)
            for s in steps:
                if s.observation:
                    _exec(fresh, s.action)
            rows += sssm_examples(task, steps) if a.format == "sssm" else [fha_example(task, steps)]
    random.Random(1).shuffle(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} examples to {a.out}")


if __name__ == "__main__":
    main()
