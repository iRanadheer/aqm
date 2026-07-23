"""Task adapters for the dynamic few-shot chapter.

This chapter deliberately reuses the *sibling* chapters' prompts and scorers
so the comparison is byte-identical to how the fine-tuned models were run:

  - system prompts come from cards/prompts.py and wind/prompts.py
  - scoring is done by cards/generate_report.py and wind/generate_report.py
    (see score.py) against the SAME held-out test/val splits.

The only thing this chapter adds is *how the prompt is assembled* — zero-shot,
static few-shot, or dynamic (retrieval-based) few-shot — none of which requires
fine-tuning. Each Task below knows:

  - where its few-shot CORPUS lives (= train + train_eval, the exact pool the
    LoRA models learned from) and how to turn a corpus row into a (text, gold,
    demo-answer) triple,
  - where its EVAL splits live (test / val, untouched) and how to read a row,
  - the norecot (YAML-only) system prompt to use,
  - how to render one labelled demo as an assistant YAML turn,
  - what a result row must contain so the sibling scorer can read it.

A "demo" is a prior chat turn pair: user `### Text:\n<text>` -> assistant
```yaml ...``` (**text + labels only, no <think> reasoning** — deliberately, to
keep cost down at higher k).

The system prompt and thinking-mode are IDENTICAL to each task's existing
base/zero-shot runs (slim RECoT prompt, `enable_thinking=True`). The only thing
this chapter adds is the demo turns, so few-shot results drop straight into the
same table as the base zero-shot results (no need to re-run zero-shot).
"""

from __future__ import annotations

import ast
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parent.parent  # /home/projects/aqm
CARDS_DIR = REPO_ROOT / "cards"
WIND_DIR = REPO_ROOT / "wind"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _clean_codes(tokens) -> list[str]:
    """Keep only [A-Za-z0-9_] per token (drops stray commas/quotes), like the
    sibling parsers. Some source labels are malformed, e.g. '1_1_1,'."""
    import re
    out = []
    for t in tokens:
        c = re.sub(r"[^A-Za-z0-9_]", "", str(t))
        if c:
            out.append(c)
    return out


def _as_list(val) -> list[str]:
    """Coerce a labels cell (list, or numpy-style repr string) into clean codes."""
    if isinstance(val, list):
        return _clean_codes(val)
    if val is None:
        return []
    s = str(val).strip()
    try:
        out = ast.literal_eval(s)
        if isinstance(out, list):
            return _clean_codes(out)
    except (ValueError, SyntaxError):
        pass
    import re
    return _clean_codes(re.findall(r"[A-Za-z0-9_]+", s))


def _yaml_list(codes: list[str], key: str) -> str:
    if not codes:
        return f"{key}: []"
    body = "\n".join(f"  - {c}" for c in codes)
    return f"{key}:\n{body}"


# ---------------------------------------------------------------------------
# Task definition
# ---------------------------------------------------------------------------
@dataclass
class Example:
    """A retrievable corpus item OR an eval item."""
    text: str                       # the passage to embed / classify
    gold: dict                      # task-specific gold fields (for eval rows)
    raw: dict = field(default_factory=dict)  # original row (kept verbatim in output)


@dataclass
class Task:
    name: str
    system_prompt: str
    corpus_paths: list[Path]
    split_paths: dict[str, Path]           # "test"/"val" -> file
    _load_corpus_row: Callable[[dict], Example]
    _load_eval_row: Callable[[dict], Example]
    _render_answer: Callable[[dict], str]  # gold dict -> assistant YAML string
    _strat_key: Callable[[dict], str]      # gold dict -> stratification bucket
    _result_row: Callable[[dict, str], dict]  # (raw eval row, response) -> output row

    # --- corpus / splits ---------------------------------------------------
    def load_corpus(self) -> list[Example]:
        out: list[Example] = []
        for p in self.corpus_paths:
            for row in _read_jsonl(p):
                out.append(self._load_corpus_row(row))
        return out

    def load_split(self, split: str) -> list[Example]:
        return [self._load_eval_row(r) for r in _read_jsonl(self.split_paths[split])]

    # --- prompt assembly ---------------------------------------------------
    def demo_turns(self, ex: Example) -> list[dict]:
        """A labelled corpus example as a (user, assistant) message pair."""
        return [
            {"role": "user", "content": f"### Text:\n{ex.text}"},
            {"role": "assistant", "content": self._render_answer(ex.gold)},
        ]

    def query_turn(self, text: str) -> dict:
        return {"role": "user", "content": f"### Text:\n{text}"}

    def result_row(self, raw: dict, response: str) -> dict:
        return self._result_row(raw, response)

    def strat_key(self, ex: Example) -> str:
        return self._strat_key(ex.gold)


# ---------------------------------------------------------------------------
# CARDS adapter
# ---------------------------------------------------------------------------
def _cards_task() -> Task:
    sys.path.insert(0, str(CARDS_DIR))
    import prompts as cp  # cards/prompts.py

    def load_corpus_row(row: dict) -> Example:
        codes = _as_list(row["true_claims"])
        return Example(text=row["text"], gold={"codes": codes}, raw=row)

    def load_eval_row(row: dict) -> Example:
        codes = _as_list(row["true_claims"])
        return Example(text=row["text"], gold={"codes": codes}, raw=row)

    def render_answer(gold: dict) -> str:
        codes = gold["codes"] or ["0_0_0"]
        return "```yaml\n" + _yaml_list(codes, "categories") + "\n```"

    def strat_key(gold: dict) -> str:
        codes = gold["codes"] or ["0_0_0"]
        return codes[0]

    def result_row(raw: dict, response: str) -> dict:
        out = dict(raw)
        out["response"] = response
        return out

    return Task(
        name="cards",
        # Same slim RECoT prompt (thinking) as the base zero-shot runs.
        system_prompt=cp.slim_system_instruction,
        corpus_paths=[CARDS_DIR / "data" / "training_recot_opus.jsonl"],
        split_paths={
            "test": CARDS_DIR / "data" / "cards_test.jsonl",
            "val": CARDS_DIR / "data" / "cards_val.jsonl",
        },
        _load_corpus_row=load_corpus_row,
        _load_eval_row=load_eval_row,
        _render_answer=render_answer,
        _strat_key=strat_key,
        _result_row=result_row,
    )


# ---------------------------------------------------------------------------
# WIND adapter
# ---------------------------------------------------------------------------
def _wind_task() -> Task:
    sys.path.insert(0, str(WIND_DIR))
    import prompts as wp  # wind/prompts.py

    def load_corpus_row(row: dict) -> Example:
        gold = {
            "opposition_detected": bool(row["true_opposition_detected"]),
            "frames": _as_list(row["true_frames"]),
            "claims": _as_list(row["true_claims"]),
        }
        return Example(text=row["content"], gold=gold, raw=row)

    load_eval_row = load_corpus_row  # same schema (val/test carry gold too)

    def render_answer(gold: dict) -> str:
        op = "true" if gold["opposition_detected"] else "false"
        frames = gold["frames"] if gold["opposition_detected"] else []
        claims = gold["claims"] if gold["opposition_detected"] else []
        return (
            "```yaml\n"
            f"opposition_detected: {op}\n"
            f"{_yaml_list(frames, 'frames')}\n"
            f"{_yaml_list(claims, 'claims')}\n"
            "```"
        )

    def strat_key(gold: dict) -> str:
        if not gold["opposition_detected"]:
            return "neg"
        return (gold["frames"] or ["N_0"])[0]

    def result_row(raw: dict, response: str) -> dict:
        out = dict(raw)
        out["response"] = response
        return out

    return Task(
        name="wind",
        # Same slim RECoT prompt (thinking) as the base zero-shot runs.
        system_prompt=wp.slim_system_instruction,
        corpus_paths=[
            WIND_DIR / "data" / "train" / "train_labels.jsonl",
            WIND_DIR / "data" / "train" / "train_eval_labels.jsonl",
        ],
        split_paths={
            "test": WIND_DIR / "data" / "test" / "test.jsonl",
            "val": WIND_DIR / "data" / "test" / "val.jsonl",
        },
        _load_corpus_row=load_corpus_row,
        _load_eval_row=load_eval_row,
        _render_answer=render_answer,
        _strat_key=strat_key,
        _result_row=result_row,
    )


_BUILDERS = {"cards": _cards_task, "wind": _wind_task}


def get_task(name: str) -> Task:
    if name not in _BUILDERS:
        raise SystemExit(f"Unknown task '{name}'. Choose from: {list(_BUILDERS)}")
    return _BUILDERS[name]()
