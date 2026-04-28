---
license: apache-2.0
language:
- en
base_model:
- Qwen/Qwen3.6-27B
pipeline_tag: text-generation
library_name: transformers
tags:
- climate
- climate-change
- climate-discourse
- classification
- qwen3
- fine-tuned
- cards
datasets:
- C3DS/cards_sft_dataset
---

# CARDS-Qwen3.6-27B

Fine-tuned **Qwen3.6-27B** for classification of climate-contrarian claims using the **CARDS taxonomy** from Coan et al. (2026).

This is a **merged** checkpoint: a LoRA adapter (rank 16) trained on the CARDS SFT dataset has been merged back into the base weights for direct loading with `transformers`, vLLM, or any standard inference engine. The separate adapter is available at [`C3DS/CARDS-Qwen3.6-27B-lora`](https://huggingface.co/C3DS/CARDS-Qwen3.6-27B-lora).

## Results

Evaluated on the held-out CARDS test set (1,436 samples, Level 1, `min_support ≥ 3`):

| Metric | Qwen3.5-27B | Qwen3.5-27B FT | **Qwen3.6-27B FT** | Claude Opus 4.6 | Claude Opus 4.7 |
|---|---|---|---|---|---|
| Samples F1 | 0.844 | 0.884 | **0.893** | **0.893** | 0.882 |
| Macro F1 | 0.710 | 0.766 | 0.748 | 0.751 | **0.771** |
| Micro F1 | 0.854 | 0.877 | **0.885** | 0.881 | 0.874 |
| Precision | 0.870 | 0.879 | **0.893** | 0.863 | 0.868 |
| Recall | 0.838 | 0.874 | 0.876 | **0.900** | 0.880 |
| Parse failures | 86 / 1436 | 0 / 1436 | 2 / 1436 | 0 / 1436 | 0 / 1436 |

- Ties Claude Opus 4.6 on Samples F1 at L1.
- Wins Micro F1 and Precision at every hierarchy level.
- Trails Claude Opus on Recall and Macro F1 for rare labels.
- Parse failures drop from 6% (plain Qwen3.5-27B) to ~0 with fine-tuning.

## Usage

### With vLLM

```bash
vllm serve C3DS/CARDS-Qwen3.6-27B \
  --port 8000 \
  --max-model-len 4096 \
  --dtype bfloat16 \
  --enable-prefix-caching \
  --served-model-name CARDS-Qwen3.6-27B
```

Then query with any OpenAI-compatible client. The system prompt (`slim_system_instruction`) and the user-message suffix (`cot_trigger`) the model was trained with are bundled in this repo as [`cards_prompts.json`](./cards_prompts.json) — self-contained, with the CARDS taxonomy already inlined.

```python
import json
from huggingface_hub import hf_hub_download
from openai import OpenAI

prompts = json.load(open(hf_hub_download("C3DS/CARDS-Qwen3.6-27B", "cards_prompts.json")))
slim_system_instruction = prompts["slim_system_instruction"]
cot_trigger             = prompts["cot_trigger"]

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

def classify(text):
    resp = client.chat.completions.create(
        model="CARDS-Qwen3.6-27B",
        messages=[
            {"role": "system", "content": slim_system_instruction},
            {"role": "user",   "content": f"### Text:\n{text}\n\n{cot_trigger}"},
        ],
        temperature=0,
        max_tokens=4000,
    )
    return resp.choices[0].message.content

print(classify("These are only a few renewable energy technologies at work"))
```

The model produces a reasoning trace inside `<think>…</think>` followed by a YAML `categories:` block listing predicted CARDS codes. To parse: take the content after `</think>` and read the `categories:` list.

See the [project repository](https://github.com/project-c3ds/cards-2pO-paper) for training scripts, evaluation code, and dataset preparation.

## Training

- **Base model:** `Qwen/Qwen3.6-27B`
- **Method:** LoRA (rank 16, α 16, dropout 0) on `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`, then merged into base weights
- **Dataset:** [`C3DS/cards_sft_dataset`](https://huggingface.co/datasets/C3DS/cards_sft_dataset)
- **Framework:** Unsloth + TRL `SFTTrainer`
- **Hyperparameters:** 3 epochs, `per_device_train_batch_size=1`, `gradient_accumulation_steps=8`, `lr=2e-4`, cosine schedule, 10 warmup steps, `max_seq_length=4096`, `adamw_8bit`, `bf16`
- **Hardware:** 1× NVIDIA H200, ~2h 7min wall-clock
- **Checkpoint selection:** best via `load_best_model_at_end=True` (step 600, `eval_loss=0.0905`)
- **Final training loss:** 0.125

## Limitations

- **Macro F1 on rare labels.** The model trails Claude Opus at L3 macro-F1 (0.483 vs 0.531), reflecting the long-tailed label distribution of CARDS.
- **English only.**
- **Thinking tokens.** Training used `enable_thinking=True`. Either parse output after `</think>`, or disable thinking at inference via `chat_template_kwargs={"enable_thinking": false}`. Reserve token budget for the reasoning trace before the final YAML block.

## Citation

```bibtex
@article{cards2pO2025,
  title={Large language model reveals an increase in climate contrarian speech in the United States Congress},
  author={Travis G. Coan and Ranadheer Malla and Mirjam O. Nanko and William Kattrup and J. Timmons Roberts and John Cook and Constantine Boussalis},
  journal={Communications Sustainability},
  year={2025}
}
```

## License

Apache 2.0, inherited from Qwen3.6-27B.
