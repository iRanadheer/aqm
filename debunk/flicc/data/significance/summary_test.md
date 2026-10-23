# FLICC significance — test split

95% BCa bootstrap, 9999 resamples, seed 42. Parse failures and out-of-vocabulary labels count as wrong.

## Paired comparisons

| A | B | Delta (A-B) | 95% CI | verdict |
|---|---|---:|---|---|
| openai-gpt-5-5-deconstruct-revised | fzanartu-flicc-deberta-v2-xlarge | -0.079 | [-0.158, +0.002] | comparable |
| anthropic-claude-opus-4-7-deconstruct-revised | fzanartu-flicc-deberta-v2-xlarge | -0.098 | [-0.175, -0.017] | lower |
| c3ds-flicc-qwen3-5-27b-deconstruct-revised | fzanartu-flicc-deberta-v2-xlarge | -0.098 | [-0.176, -0.021] | lower |
| c3ds-flicc-qwen3-5-9b-deconstruct-revised | fzanartu-flicc-deberta-v2-xlarge | -0.032 | [-0.102, +0.041] | comparable |
| c3ds-flicc-qwen3-5-4b-deconstruct-revised | fzanartu-flicc-deberta-v2-xlarge | -0.162 | [-0.236, -0.092] | lower |
| qwen-qwen3-5-27b-deconstruct-revised-structured | fzanartu-flicc-deberta-v2-xlarge | -0.238 | [-0.317, -0.153] | lower |
