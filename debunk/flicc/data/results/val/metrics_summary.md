# FLICC results — val split

Baseline: Zanartu et al. 2024 DeBERTa-v2-xlarge (887M), macro F1 **0.73**, accuracy 0.73 (published, not re-run).

Parse and vocabulary failures count as wrong. No hand-repair.

| Run | N | Macro F1 | vs 0.73 | Accuracy | Weighted F1 | Unparsed | OOV | Errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| openai-gpt-5-5-deconstruct-revised | 457 | **0.640** | -0.090 | 0.637 | 0.642 | 0 | 0 | 0 |
| anthropic-claude-opus-4-7-deconstruct-revised | 457 | **0.632** | -0.098 | 0.654 | 0.653 | 0 | 0 | 0 |
| openai-gpt-5-5-deconstruct-published | 457 | **0.471** | -0.259 | 0.473 | 0.482 | 0 | 0 | 0 |
| openai-gpt-5-5-bare | 457 | **0.464** | -0.266 | 0.473 | 0.460 | 0 | 0 | 0 |
| anthropic-claude-opus-4-7-deconstruct-published | 457 | **0.454** | -0.276 | 0.479 | 0.475 | 0 | 0 | 0 |
| anthropic-claude-opus-4-7-bare | 457 | **0.391** | -0.339 | 0.424 | 0.413 | 0 | 0 | 0 |
