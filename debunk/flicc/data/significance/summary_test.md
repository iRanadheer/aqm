# FLICC significance — test split

95% BCa bootstrap, 9999 resamples, seed 42. Parse failures and out-of-vocabulary labels count as wrong.

## vs published baseline (one-sample)

Zanartu et al. DeBERTa-v2-xlarge macro F1 = **0.73**, a published point estimate. Their model is not re-run, so this is a one-sample comparison against a fixed value — weaker than a paired test.

| Run | n | Macro F1 | 95% CI | vs baseline |
|---|---:|---:|---|---|
| anthropic-claude-opus-4-7-bare | 256 | 0.402 | [0.351, 0.469] | lower |
| anthropic-claude-opus-4-7-deconstruct-published | 256 | 0.488 | [0.429, 0.558] | lower |
| anthropic-claude-opus-4-7-deconstruct-revised | 256 | 0.635 | [0.576, 0.702] | lower |
| openai-gpt-5-5-bare | 256 | 0.454 | [0.401, 0.527] | lower |
| openai-gpt-5-5-deconstruct-published | 256 | 0.463 | [0.407, 0.537] | lower |
| openai-gpt-5-5-deconstruct-revised | 256 | 0.654 | [0.598, 0.717] | lower |

## Paired comparison

| A | B | Delta (A-B) | 95% CI | verdict |
|---|---|---:|---|---|
| anthropic-claude-opus-4-7-deconstruct-revised | anthropic-claude-opus-4-7-deconstruct-published | +0.147 | [+0.070, +0.219] | **improves** |
