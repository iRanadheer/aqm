# Inference (`infer.py`)

Run a chat-completion model on a benchmark JSONL and save responses.
PEP-723. Replaces the old `workspace.ipynb`.

```
uv run infer.py [--backend {vllm|openai|openrouter}] --model <name> [--input <jsonl>]
```

## Backends

Three: local vLLM (default), OpenAI, OpenRouter. Each backend implies its
own base URL and API key env var — no `--base-url` flag, just pick the
backend.

## Prompt

Same shape as training-time inference: slim system instruction + CoT
trigger appended to the user message. The system prompt is always wrapped
as a cache-eligible content block (`cache_control: ephemeral`); providers
that don't support prompt caching ignore the field.

## Output

Saved to `data/results/<split>/<model-slug>.jsonl`, where `split` is
inferred from the input filename and `model-slug` is a normalized version
of `--model`. Each output row preserves the input row and adds a
`response` field with the raw model output (string).

Failed calls (after retry) get `"response": "ERROR: ..."` so the run
completes; the report counts these as parse failures.
