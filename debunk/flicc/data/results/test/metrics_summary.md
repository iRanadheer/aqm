# FLICC results — test split

Baseline: Zanartu et al. 2024 DeBERTa-v2-xlarge (887M), macro F1 **0.73**, accuracy 0.74 (published, not re-run).

Parse and vocabulary failures count as wrong. No hand-repair.

| Run | N | Macro F1 | vs 0.73 | Accuracy | Weighted F1 | Unparsed | OOV | Errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| openai-gpt-5-5-deconstruct-revised | 256 | **0.654** | -0.076 | 0.656 | 0.647 | 0 | 0 | 0 |
| anthropic-claude-opus-4-7-deconstruct-revised | 256 | **0.635** | -0.095 | 0.656 | 0.653 | 0 | 0 | 0 |
| anthropic-claude-opus-4-7-deconstruct-published | 256 | **0.488** | -0.242 | 0.480 | 0.477 | 0 | 0 | 0 |
| openai-gpt-5-5-deconstruct-published | 256 | **0.463** | -0.267 | 0.457 | 0.458 | 0 | 0 | 0 |
| openai-gpt-5-5-bare | 256 | **0.454** | -0.276 | 0.473 | 0.458 | 0 | 0 | 0 |
| anthropic-claude-opus-4-7-bare | 256 | **0.402** | -0.328 | 0.426 | 0.409 | 0 | 0 | 0 |
