# FLICC results — test split

Baseline: Zanartu et al. 2024 DeBERTa-v2-xlarge (887M), macro F1 **0.73**, accuracy 0.74 (published, not re-run).

Parse and vocabulary failures count as wrong. No hand-repair.

| Run | N | Macro F1 | vs 0.73 | Accuracy | Weighted F1 | Unparsed | OOV | Errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| c3ds-flicc-qwen3-5-9b-deconstruct-revised | 256 | **0.702** | -0.028 | 0.695 | 0.693 | 0 | 0 | 0 |
| openai-gpt-5-5-deconstruct-revised | 256 | **0.654** | -0.076 | 0.656 | 0.647 | 0 | 0 | 0 |
| c3ds-flicc-qwen3-5-27b-deconstruct-revised | 256 | **0.636** | -0.094 | 0.637 | 0.630 | 0 | 0 | 0 |
| anthropic-claude-opus-4-7-deconstruct-revised | 256 | **0.635** | -0.095 | 0.656 | 0.653 | 0 | 0 | 0 |
| c3ds-flicc-qwen3-5-2b-deconstruct-revised | 256 | **0.586** | -0.144 | 0.609 | 0.608 | 0 | 1 | 0 |
| c3ds-flicc-qwen3-5-4b-deconstruct-revised | 256 | **0.571** | -0.159 | 0.574 | 0.574 | 0 | 0 | 0 |
| qwen-qwen3-5-27b-deconstruct-revised-structured | 256 | **0.495** | -0.235 | 0.520 | 0.494 | 0 | 0 | 0 |
| anthropic-claude-opus-4-7-deconstruct-published | 256 | **0.488** | -0.242 | 0.480 | 0.477 | 0 | 0 | 0 |
| openai-gpt-5-5-deconstruct-published | 256 | **0.463** | -0.267 | 0.457 | 0.458 | 0 | 0 | 0 |
| openai-gpt-5-5-bare | 256 | **0.454** | -0.276 | 0.473 | 0.458 | 0 | 0 | 0 |
| c3ds-flicc-qwen3-5-0-8b-deconstruct-revised | 256 | **0.415** | -0.315 | 0.453 | 0.452 | 0 | 0 | 0 |
| anthropic-claude-opus-4-7-bare | 256 | **0.402** | -0.328 | 0.426 | 0.409 | 0 | 0 | 0 |
| qwen-qwen3-5-9b-deconstruct-revised-structured | 256 | **0.351** | -0.379 | 0.371 | 0.354 | 8 | 0 | 0 |
| qwen-qwen3-5-4b-deconstruct-revised-structured | 256 | **0.314** | -0.416 | 0.348 | 0.323 | 8 | 0 | 0 |
| qwen-qwen3-5-2b-deconstruct-revised-structured | 256 | **0.149** | -0.581 | 0.160 | 0.172 | 0 | 0 | 0 |
| qwen-qwen3-5-0-8b-deconstruct-revised-structured | 256 | **0.096** | -0.634 | 0.105 | 0.106 | 88 | 0 | 0 |
