# Prompts (`prompts.py`)

Single source of truth for the system prompt and triggers. All other
scripts import from here.

## Codebooks

Two are built at import time from `data/taxonomy.csv`:

- **verbose** — full XML-tagged labels with descriptions, used for big
  teacher/API models that benefit from rich context.
- **slim** — short labels keyed by code, used for trained small models
  that have already learned the taxonomy through SFT.

## System instruction

A single template is `.format`-ed against each codebook to produce
`system_instruction` (verbose) and `slim_system_instruction` (slim). The
template defines the hierarchical classification rules and pins the output
format: a `<think>` reasoning block with a fixed scaffold, then a YAML
`categories:` block. Strict.

## Triggers

- `recot_trigger` — used by the teacher script. Tells the teacher to
  generate reasoning that arrives at the given true labels without
  revealing them in the trace. The same trigger serves both the API and
  chat (`--chat`) teacher passes; only the system prompt differs.
- `cot_trigger` was removed: training and inference both use a bare
  `### Text:` user message; the system prompt alone pins the output
  format.

## Why centralize

The pre-refactor codebase had inline copies of the system prompt in
multiple training and eval scripts. Drift between copies caused subtle
training/inference mismatches. Now there is exactly one definition;
everything imports.
