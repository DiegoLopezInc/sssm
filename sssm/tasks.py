"""Procedurally generated long-horizon tasks with exact, automatic grading.

Every task exposes a tiny tool environment. The number of required actions grows with
``horizon`` so full-history context grows linearly while an SSSM state stays bounded.
"""

from __future__ import annotations

import ast
import difflib
import operator
import random
import string
from dataclasses import dataclass, field

MAX_OBS_CHARS = 1500


@dataclass
class StepResult:
    observation: str
    done: bool = False


@dataclass
class Task:
    family: str
    horizon: int
    seed: int
    prompt: str
    answer: object
    files: dict[str, str] = field(default_factory=dict)
    tools: str = ""
    meta: dict = field(default_factory=dict)
    submitted: object = None

    def step(self, name: str, args: list) -> StepResult:
        if name == "submit":
            if len(args) != 1:
                return StepResult("ERROR: submit takes exactly one argument.")
            self.submitted = args[0]
            return StepResult("Submitted.", done=True)
        handler = getattr(self, f"tool_{name}", None)
        if handler is None:
            return StepResult(f"ERROR: unknown tool '{name}'. Available tools:\n{self.tools}")
        try:
            obs = handler(*args)
        except TypeError as e:
            obs = f"ERROR: bad arguments for {name}: {e}"
        return StepResult(str(obs)[:MAX_OBS_CHARS])

    def tool_list_files(self) -> str:
        return "\n".join(sorted(self.files)) or "(no files)"

    def tool_read_file(self, name: str) -> str:
        name = str(name).strip().lstrip("./")
        if name not in self.files:
            return f"ERROR: no such file '{name}'."
        return self.files[name]

    def tool_calc(self, expression: str) -> str:
        try:
            return str(safe_eval(str(expression)))
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {e}"

    def is_correct(self) -> bool:
        return self.submitted is not None and grade(self.family, self.submitted, self.answer)


FILE_TOOLS = """- list_files(): list available files
- read_file(name): return the contents of a file
- calc(expression): evaluate arithmetic, e.g. calc("12 + 7 - 3")
- submit(answer): submit the final answer and end the task"""

CHAT_TOOLS = """- next_message(): read the next customer message
- calc(expression): evaluate arithmetic, e.g. calc("2 + 3")
- submit(answer): submit the final answer and end the task"""


def make_ledger(horizon: int, seed: int) -> Task:
    rng = random.Random(f"ledger-{horizon}-{seed}")
    accounts = rng.sample(["ACME", "BOLT", "CEDAR", "DELTA", "EMBER", "FJORD", "GLYPH"], 4)
    target = accounts[0]
    files, total = {}, 0
    for i in range(1, horizon + 1):
        rows = []
        for _ in range(rng.randint(1, 2)):
            rows.append((target, rng.choice([1, -1]) * rng.randint(5, 60)))
        for _ in range(rng.randint(2, 4)):
            rows.append((rng.choice(accounts[1:]), rng.choice([1, -1]) * rng.randint(5, 60)))
        rng.shuffle(rows)
        total += sum(a for acct, a in rows if acct == target)
        lines = [f"TXN-{rng.randint(1000, 9999)} | {acct} | {a:+d}" for acct, a in rows]
        files[f"ledger_{i:02d}.txt"] = "id | account | amount\n" + "\n".join(lines)
    prompt = (
        f"There are {horizon} ledger files named ledger_01.txt to ledger_{horizon:02d}.txt. "
        f"Starting from a balance of 0, compute the final balance of account {target} after "
        f"applying every {target} transaction in every ledger file (ignore other accounts). "
        "Submit the final balance as an integer."
    )
    return Task("ledger", horizon, seed, prompt, total, files, FILE_TOOLS, meta={"target": target})


def make_pointer(horizon: int, seed: int) -> Task:
    rng = random.Random(f"pointer-{horizon}-{seed}")
    ids = rng.sample(range(10, 100), 2 * horizon)
    chain, decoys = ids[:horizon], ids[horizon:]
    letters = "".join(rng.choice(string.ascii_uppercase) for _ in range(horizon))
    files = {}
    for i, node in enumerate(chain):
        nxt = f"node_{chain[i + 1]}.txt" if i + 1 < horizon else "END"
        files[f"node_{node}.txt"] = f"Code letter: {letters[i]}\nNext: {nxt}"
    for node in decoys:
        files[f"node_{node}.txt"] = (
            f"Code letter: {rng.choice(string.ascii_uppercase)}\nNext: node_{rng.choice(decoys)}.txt"
        )
    prompt = (
        f"Start by reading node_{chain[0]}.txt. Each node file contains a code letter and the "
        "name of the next file to read. Follow the chain, collecting the code letters in order, "
        "until a file says 'Next: END' (include that file's letter). Submit all collected "
        'letters as a single string, e.g. submit("QXTB").'
    )
    meta = {"start": f"node_{chain[0]}.txt"}
    return Task("pointer", horizon, seed, prompt, letters, files, FILE_TOOLS, meta=meta)


ITEMS = ["apple", "banana", "carrot", "milk", "bread", "egg", "rice", "tea"]


@dataclass
class CartTask(Task):
    messages: list[str] = field(default_factory=list)
    cursor: int = 0

    def tool_next_message(self) -> str:
        if self.cursor >= len(self.messages):
            return "No more messages. The customer is waiting for you to submit the order."
        self.cursor += 1
        return f"Customer message {self.cursor} of {len(self.messages)}: {self.messages[self.cursor - 1]}"


def make_cart(horizon: int, seed: int) -> Task:
    rng = random.Random(f"cart-{horizon}-{seed}")
    cart: dict[str, int] = {}
    msgs = []
    for _ in range(horizon):
        held = [k for k, v in cart.items() if v > 0]
        op = rng.choice(["add", "add", "remove", "set"]) if held else "add"
        if op == "add":
            item, k = rng.choice(ITEMS), rng.randint(1, 4)
            cart[item] = cart.get(item, 0) + k
            msgs.append(f"Please add {k} x {item} to my order.")
        elif op == "remove":
            item = rng.choice(held)
            k = rng.randint(1, cart[item])
            cart[item] -= k
            msgs.append(f"Remove {k} x {item}, please.")
        else:
            item, k = rng.choice(held), rng.randint(0, 5)
            cart[item] = k
            msgs.append(f"Actually, change the {item} quantity to exactly {k}.")
    answer = {k: v for k, v in cart.items() if v > 0}
    prompt = (
        "You are an order-taking assistant. Customer messages arrive one at a time; call "
        "next_message() to read the next one. Keep track of the cart as the customer changes it. "
        "When there are no more messages, submit the final cart as a dict mapping item name to "
        'quantity, e.g. submit({"apple": 2, "milk": 1}). Omit items with quantity 0.'
    )
    return CartTask("cart", horizon, seed, prompt, answer, {}, CHAT_TOOLS, messages=msgs)


FAMILIES = {"ledger": make_ledger, "pointer": make_pointer, "cart": make_cart}


def make_task(family: str, horizon: int, seed: int) -> Task:
    return FAMILIES[family](horizon, seed)


def _norm_item(name: str) -> str:
    name = str(name).strip().lower()
    return name[:-1] if name.endswith("s") and name[:-1] in ITEMS else name


def grade(family: str, submitted: object, answer: object) -> bool:
    if family == "ledger":
        try:
            return int(float(str(submitted).replace(",", "").strip())) == answer
        except ValueError:
            return False
    if family == "pointer":
        return "".join(ch for ch in str(submitted).upper() if ch.isalpha()) == answer
    if family == "cart":
        return _parse_cart(submitted) == answer
    raise ValueError(family)


def _parse_cart(submitted: object) -> dict | None:
    if isinstance(submitted, str):
        try:
            submitted = ast.literal_eval(submitted)
        except (ValueError, SyntaxError):
            return None
    if not isinstance(submitted, dict):
        return None
    try:
        return {_norm_item(k): int(v) for k, v in submitted.items() if int(v) != 0}
    except (TypeError, ValueError):
        return None


def partial_score(family: str, submitted: object, answer: object) -> float:
    """Partial credit in [0, 1], to tell near misses from total failures (1.0 iff grade() is True)."""
    if submitted is None:
        return 0.0
    if grade(family, submitted, answer):
        return 1.0
    if family == "ledger":
        try:
            err = abs(int(float(str(submitted).replace(",", "").strip())) - answer)
        except ValueError:
            return 0.0
        return max(0.0, 1 - err / max(abs(answer), 10)) * 0.99
    if family == "pointer":
        got = "".join(ch for ch in str(submitted).upper() if ch.isalpha())
        return difflib.SequenceMatcher(None, got, answer).ratio() * 0.99
    if family == "cart":
        got = _parse_cart(submitted)
        if got is None:
            return 0.0
        a, b = set(got.items()), set(answer.items())
        return len(a & b) / len(a | b) * 0.99
    raise ValueError(family)


_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def safe_eval(expr: str) -> float | int:
    def ev(node: ast.AST):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError("only + - * / and numbers are allowed")

    return ev(ast.parse(expr.strip(), mode="eval").body)
