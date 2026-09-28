# FLICC results — val split

Baseline: Zanartu et al. 2024 DeBERTa-v2-xlarge (887M), macro F1 **0.73**, accuracy 0.73 (published, not re-run).

Parse and vocabulary failures count as wrong. No hand-repair.

| Run | N | Macro F1 | vs 0.73 | Accuracy | Weighted F1 | Unparsed | OOV | Errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| c3ds-flicc-qwen3-5-4b-deconstruct-revised | 457 | **0.655** | -0.075 | 0.670 | 0.672 | 0 | 0 | 0 |
| c3ds-flicc-qwen3-5-27b-deconstruct-revised | 457 | **0.645** | -0.085 | 0.667 | 0.664 | 0 | 0 | 0 |
| openai-gpt-5-5-deconstruct-revised | 457 | **0.640** | -0.090 | 0.637 | 0.642 | 0 | 0 | 0 |
| anthropic-claude-opus-4-7-deconstruct-revised | 457 | **0.632** | -0.098 | 0.654 | 0.653 | 0 | 0 | 0 |
| c3ds-flicc-qwen3-5-9b-deconstruct-revised | 457 | **0.623** | -0.107 | 0.635 | 0.635 | 0 | 0 | 0 |
| c3ds-flicc-qwen3-5-2b-deconstruct-revised | 457 | **0.555** | -0.175 | 0.565 | 0.568 | 0 | 0 | 0 |
| c3ds-flicc-qwen3-5-0-8b-deconstruct-revised | 457 | **0.500** | -0.230 | 0.519 | 0.527 | 0 | 1 | 0 |
| openai-gpt-5-5-deconstruct-published | 457 | **0.471** | -0.259 | 0.473 | 0.482 | 0 | 0 | 0 |
| qwen-qwen3-5-27b-deconstruct-revised-structured | 457 | **0.469** | -0.261 | 0.499 | 0.487 | 3 | 0 | 0 |
| openai-gpt-5-5-bare | 457 | **0.464** | -0.266 | 0.473 | 0.460 | 0 | 0 | 0 |
| anthropic-claude-opus-4-7-deconstruct-published | 457 | **0.454** | -0.276 | 0.479 | 0.475 | 0 | 0 | 0 |
| anthropic-claude-opus-4-7-bare | 457 | **0.391** | -0.339 | 0.424 | 0.413 | 0 | 0 | 0 |
| qwen-qwen3-5-9b-deconstruct-revised-structured | 457 | **0.362** | -0.368 | 0.383 | 0.381 | 23 | 0 | 0 |
| qwen-qwen3-5-4b-deconstruct-revised-structured | 457 | **0.323** | -0.407 | 0.359 | 0.349 | 9 | 0 | 0 |
| qwen-qwen3-5-2b-deconstruct-revised-structured | 457 | **0.127** | -0.603 | 0.155 | 0.158 | 0 | 0 | 0 |
| qwen-qwen3-5-0-8b-deconstruct-revised-structured | 457 | **0.058** | -0.672 | 0.068 | 0.072 | 156 | 0 | 0 |
