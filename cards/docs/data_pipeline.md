# Data pipeline

Two scripts.

```
training.csv  ──teacher.py──>  training_recot_opus.jsonl  ──prepare_splits.py──>  cards_train(_eval){,_norecot}.jsonl
```

## `teacher.py`

Calls a teacher LLM on each `(text, true_claims)` row to produce a
`<think>` reasoning trace plus YAML categories. Output is appended JSONL,
resume-safe (skips rows already in the output).

Backend is LiteLLM, so any provider works.

## `prepare_splits.py`

Wraps each `(text, response)` into the OpenAI chat format
(`{messages: [system, user, assistant]}`) and writes the train/eval JSONL
splits. The 90/10 split is stratified on the first category code and
deterministic; the same indices are reused for the no-RECoT mirror so that
the two variants share an identical row partition.

`_norecot` variants are produced by stripping `<think>...</think>` from the
assistant turn and dropping the CoT trigger from the user turn. Same rows,
same boundary, just no reasoning supervision.

## Eval splits are frozen

`cards_val.jsonl`, `cards_test.jsonl`, and `cards_twitter.jsonl` are
checked-in canonical artifacts. `prepare_splits.py` no longer regenerates
them — that logic was removed because re-running would silently drift away
from the published label set.
