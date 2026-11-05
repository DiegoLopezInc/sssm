"""Aggregate result JSONL files into markdown tables and plots."""

from __future__ import annotations

import argparse
import ast
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from .tasks import partial_score


def load(paths: list[str]) -> list[dict]:
    rows = []
    for p in paths:
        rows += [json.loads(line) for line in Path(p).read_text().splitlines() if line.strip()]
    for r in rows:
        if "score" not in r:
            r["score"] = partial_score(r["family"], _unrepr(r["submitted"]), _unrepr(r["answer"]))
    return rows


def _unrepr(value: object) -> object:
    try:
        return ast.literal_eval(value) if isinstance(value, str) else value
    except (ValueError, SyntaxError):
        return value


def _agg(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "acc": 100 * mean(r["correct"] for r in rows),
        "score": 100 * mean(r["score"] for r in rows),
        "peak": mean(r["peak_context"] for r in rows),
        "peak_max": max(r["peak_context"] for r in rows),
        "tok_in": mean(r["prompt_tokens"] for r in rows),
        "tok_out": mean(r["completion_tokens"] for r in rows),
        "calls": mean(r["calls"] for r in rows),
        "secs": mean(r["wall_seconds"] for r in rows),
        "mem": max((r.get("resident_bytes") or 0) for r in rows) / 2**30,
    }


HEADER = (
    "| model | mode | n | accuracy % | partial credit % | peak ctx (mean/max) | prompt tok/episode | "
    "gen tok/episode | LLM calls | sec/episode | resident GiB |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|"
)


def _row(key: tuple, s: dict) -> str:
    return (
        f"| {' | '.join(map(str, key))} | {s['n']} | {s['acc']:.1f} | {s['score']:.1f} | "
        f"{s['peak']:.0f}/{s['peak_max']} | "
        f"{s['tok_in']:.0f} | {s['tok_out']:.0f} | {s['calls']:.1f} | {s['secs']:.1f} | {s['mem']:.2f} |"
    )


def tables(rows: list[dict]) -> str:
    overall, detail = defaultdict(list), defaultdict(list)
    for r in rows:
        overall[(r["model"], r["mode"])].append(r)
        detail[(r["family"], r["horizon"], r["model"], r["mode"])].append(r)
    out = ["## Overall", "", HEADER]
    out += [_row(k, _agg(v)) for k, v in sorted(overall.items())]
    out += [
        "",
        "## By task family and horizon",
        "",
        "| family | horizon | model | mode | n | accuracy % | partial % | peak ctx (mean/max) | "
        "prompt tok | gen tok | "
        "calls | sec | GiB |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    out += [_row(k, _agg(v)) for k, v in sorted(detail.items())]
    return "\n".join(out)


def plot(rows: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    families = sorted({r["family"] for r in rows})
    series = sorted({(r["model"], r["mode"]) for r in rows})
    fig, axes = plt.subplots(4, len(families), figsize=(5 * len(families), 14), squeeze=False)
    metrics = [("correct", "accuracy %", 100), ("score", "partial credit %", 100),
               ("peak_context", "peak context (tokens)", 1), ("wall_seconds", "seconds / episode", 1)]
    for j, fam in enumerate(families):
        for i, (key, label, scale) in enumerate(metrics):
            ax = axes[i][j]
            for model, mode in series:
                pts = defaultdict(list)
                for r in rows:
                    if r["family"] == fam and r["model"] == model and r["mode"] == mode:
                        pts[r["horizon"]].append(r[key] * scale)
                xs = sorted(pts)
                ax.plot(xs, [mean(pts[x]) for x in xs], marker="o", label=f"{model} / {mode}")
            ax.set_xlabel("horizon (required actions)")
            ax.set_ylabel(label)
            if i == 0:
                ax.set_title(fam)
            if i <= 1:
                ax.set_ylim(-5, 105)
            ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("files", nargs="+")
    p.add_argument("--md", default="results/summary.md")
    p.add_argument("--png", default="results/summary.png")
    a = p.parse_args(argv)
    rows = load(a.files)
    md = tables(rows)
    Path(a.md).write_text(md + "\n")
    print(md)
    plot(rows, Path(a.png))


if __name__ == "__main__":
    main()
