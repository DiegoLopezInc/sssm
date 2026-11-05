"""Parse the ``ACTION: tool(args)`` line emitted by the model."""

from __future__ import annotations

import ast
import re

_ACTION_RE = re.compile(r"ACTION\s*:\s*(.+)", re.IGNORECASE)


class ActionParseError(ValueError):
    pass


def parse_action(text: str) -> tuple[str, list]:
    matches = _ACTION_RE.findall(text)
    if not matches:
        raise ActionParseError("no 'ACTION:' line found")
    call = matches[0].split(" -> ")[0].strip().strip("`").strip()
    if call.startswith("python"):
        call = call[len("python") :].strip()
    try:
        node = ast.parse(call, mode="eval").body
    except SyntaxError as e:
        raise ActionParseError(f"could not parse '{call}'") from e
    if isinstance(node, ast.Name):
        return node.id, []
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        raise ActionParseError(f"'{call}' is not a single tool call")
    try:
        args = [ast.literal_eval(a) for a in node.args]
        args += [ast.literal_eval(k.value) for k in node.keywords]
    except ValueError as e:
        raise ActionParseError(f"arguments of '{call}' must be literals") from e
    return node.func.id, args
