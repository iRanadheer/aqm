# Inference (`infer.py`)

Run any OpenAI-compatible chat model on the benchmark JSONL and save
responses. Optionally attach retrieval evidence (local RAG index) or Exa
search evidence. Resumable and concurrent.

```
python infer.py --backend <backend> --model <name> --prompt <variant> [--rag-index <dir>]
```

## Backends

| `--backend` | Base URL | API key env |
|---|---|---|
| `vllm` (default) | `http://localhost:8000/v1` | — |
| `openai` | `https://api.openai.com/v1` | `OPENAI_API_KEY` |
| `openrouter` | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` |
| `exa` | `https://api.exa.ai` | `EXA_API_KEY` |

Each backend implies its own base URL + key — pick the backend, no
`--base-url` flag. The `exa` backend uses the `/answer` endpoint with a JSON
output schema instead of the narrative prompt (see [prompts.md](prompts.md)).

## Key flags

| Flag | Default | Purpose |
|---|---|---|
| `--model` | (required) | e.g. `anthropic/claude-opus-4.7`, `gpt-4o-mini` |
| `--prompt` | `veracityV1` | one of `PROMPT_VARIANTS` — production: `veracityV2`, `climinator_v4` |
| `--input` | `data/test/test.jsonl` | benchmark JSONL; the filename stem becomes the `split` |
| `--output` | (auto) | default `data/results/<split>/<slug>-<prompt>.jsonl` |
| `--max-workers` | 20 | concurrent inference threads |
| `--max-tokens` | 4000 | high, to leave room for `<think>` reasoning |
| `--limit` | — | process only the first N rows (debugging) |
| `--max-input-tokens` | — | truncate user content to N tokens via the model tokenizer |
| `--no-think` | off | disable Qwen3 thinking mode (`enable_thinking=False`) |
| **RAG** | | |
| `--rag-index <dir>` | — | path to a `data/rag/<embedder>/` index; enables retrieval, tags the slug `-rag-*` |
| `--rag-k` | 5 | chunks retrieved + injected per claim |
| `--rag-no-rerank` | off | skip the cross-encoder rerank (dense+BM25+RRF only) |
| `--exa-evidence` | off | use Exa `/answer` evidence instead of a local index (mutex with `--rag-index`); tags `-exa` |
| **Audit** | | |
| `--dry-run N` | 0 | print the assembled system+user messages for the first N rows, no LLM call |
| `--dump-prompt-md <path>` | — | dump the fully assembled prompt for one claim as markdown |
| `--dump-itemId <id>` | — | which row to dump (default: first) |

## Output slug

The model name is slugified (`anthropic/claude-opus-4.7` →
`anthropic-claude-opus-4-7`), then an evidence tag is appended:

| Condition | Tag | Example |
|---|---|---|
| no evidence | — | `anthropic-claude-opus-4-7-climinator_v4.jsonl` |
| `--rag-index …pplx…ctx…` | `-rag-pplx-ctx` | `…-rag-pplx-ctx-climinator_v4.jsonl` |
| `--rag-index …qwen…` | `-rag-qwen` | `…-rag-qwen-veracityV2.jsonl` |
| `--exa-evidence` | `-exa` | `exa-veracityV1.jsonl` |

Final path: `data/results/<split>/<slug>-<prompt>.jsonl`. These slugs are
what `generate_report.py`'s `MODELS` list keys on — keep them stable.

## Prompt assembly

The system message is `PROMPT_VARIANTS[--prompt]`. The user message is the
claim, with evidence spliced in when an index / Exa is active:

- **no evidence** — `### Claim:\n{claim}`.
- **RAG / Exa** — `USER_TEMPLATE_RAG`: the claim, then a numbered evidence
  block (`[i] {source} · {title} · {date}` / `URL:` / chunk text), then a
  "how to use the evidence" rubric that forces the model to read it, ignore
  off-topic chunks, declare evidence use, and cite sources as clickable
  markdown links. Retrieval is via the `rag/retrieve.py` `HybridRetriever`
  (see [rag.md](rag.md)).

Calls run at `temperature=0`. For reasoning models that return a separate
`reasoning` field (DeepSeek, o-series), the reasoning is folded back into
`response` as a `<think>…</think>` block so the parser sees a uniform
shape.

## Resume & errors

Output is append-only; on rerun, rows whose `itemId` is already present are
skipped, so an interrupted run resumes cleanly. A call that still fails
after retries is written with `"response": "ERROR: …"` so the run completes
— `generate_report.py` counts those as parse failures.

## Examples

```bash
# Offline (built-in knowledge), production climinator prompt
python infer.py --backend openrouter --model anthropic/claude-opus-4.7 \
    --prompt climinator_v4

# + RAG (Perplexity context index)
python infer.py --backend openai --model gpt-4o-mini --prompt veracityV2 \
    --rag-index data/rag/perplexity-ai-pplx-embed-context-v1-0-6b --rag-k 5

# Exa search evidence
python infer.py --backend exa --model exa --prompt veracityV2 --exa-evidence

# Audit the assembled prompt without spending tokens
python infer.py --backend vllm --model Qwen/Qwen3.5-9B --prompt climinator_v4 \
    --rag-index data/rag/qwen-qwen3-embedding-0-6b --dry-run 3
```
