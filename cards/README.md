# cards

Hierarchical climate-discourse claim classifier. Per-stage docs live in
[docs/](docs/).

| Doc | Covers |
|---|---|
| [docs/data.md](docs/data.md) | All data: training, congressional benchmark, Twitter benchmark |
| [docs/training.md](docs/training.md) | `ft/train.py` — Unsloth + LoRA SFT |
| [docs/quantization.md](docs/quantization.md) | `ft/quantize.py` — FP8_DYNAMIC |
| [docs/inference.md](docs/inference.md) | `infer.py` — run a model on a benchmark JSONL |
| [docs/reports.md](docs/reports.md) | `generate_report.py` — strict-parsed metrics + ablations |
| [docs/prompts.md](docs/prompts.md) | `prompts.py` — codebooks, system prompt, triggers |
| [docs/serving.md](docs/serving.md) | `serve/start_cards_vllm.sh` — vLLM launcher |

## Pipeline at a glance

```
training.csv
    │ teacher.py
    ▼
training_recot_opus.jsonl
    │ prepare_splits.py
    ▼
SFT JSONL  ──ft/train.py──>  LoRA + merged BF16
                                │ ft/quantize.py
                                ▼
                              FP8 model
                                │ serve/start_cards_vllm.sh
                                ▼
                              vllm at :8000
                                │ infer.py
                                ▼
                              data/results/<split>/<slug>.jsonl
                                │ generate_report.py
                                ▼
                              metrics_summary + ablations
```
