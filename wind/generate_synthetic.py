"""Generate synthetic training examples for any frame or claim code.

Positives (default): texts that SHOULD be labeled with the target code.
Negatives (--negatives): near-miss texts that should NOT.

Two calls per candidate:
  1. Opus generates N texts (JSON structured output, no regex parsing).
  2. Opus re-labels each text from scratch using the production classifier
     prompt; quality gate keeps rows where teacher confirms (positives) or
     denies (negatives) the target code.

Never touches val/test. Seeds come from data/train/train_labels.jsonl.

Usage:
    # Pre-registered augmentation plan (see data/results/augmentation_plan.md)
    python scripts/generate_synthetic.py --preset positives
    python scripts/generate_synthetic.py --preset negatives

    # Ad-hoc
    python scripts/generate_synthetic.py --codes C_28_0,C_0_1,N_0
    python scripts/generate_synthetic.py --codes C_2_0,N_5 --negatives
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from tqdm import tqdm


class GeneratedTexts(BaseModel):
    texts: list[str]

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from prompts import full_frames_codebook, full_claims_codebook, full_system_instruction  # noqa: E402
from generate_report import parse_response  # noqa: E402

MODEL = "anthropic/claude-opus-4.7"
SEEDS = REPO_ROOT / "data/train/train_labels.jsonl"
OUT = REPO_ROOT / "data/train/train_synthetic_labels.jsonl"

# Pre-registered augmentation plan (was run_augment.sh).
# See data/results/augmentation_plan.md for derivation.
PRESETS = {
    # Positives: train support < 10, plus three frames with val FN > 10.
    "positives": [
        # Claims with train support < 10 (42 codes)
        "C_25_5", "C_17_2", "C_28_0", "C_18_0",                     # support = 1
        "C_28_1", "C_28_2", "C_28_3", "C_33_0", "C_18_1", "C_2_1",  # support = 2
        "C_25_4", "C_19_0",                                          # support = 3
        "C_35_0", "C_7_0", "C_2_2", "C_4_0", "C_8_0",                # support = 4
        "C_37_0", "C_10_0", "C_17_1", "C_38_0", "C_9_0",             # support = 5
        "C_22_0", "C_31_0", "C_17_0", "C_25_0", "C_36_0",            # support = 5
        "C_28_4", "C_28_5", "C_29_0", "C_21_0",                      # support = 6
        "C_17_3", "C_3_0", "C_16_0", "C_25_2",                       # support = 7
        "C_14_0", "C_15_0", "C_25_3", "C_13_0",                      # support = 8
        "C_2_0", "C_6_0", "C_0_1",                                   # support = 9
        # Frames with train support < 10
        "N_0",                                                       # support = 9
        # Frames with val FN > 10 (boundary confusion)
        "N_1", "N_3", "N_5",
    ],
    # Negatives: codes with val FP >= 12 AND FP > FN (over-predicted on val).
    "negatives": [
        "N_5",    # FP=43, FN=14
        "C_2_0",  # FP=28, FN=6
        "N_6",    # FP=23, FN=5
        "N_1",    # FP=22, FN=16
        "N_3",    # FP=21, FN=19
        "C_32_0", # FP=17, FN=5
        "C_33_0", # FP=12, FN=9
    ],
}

CODEBOOK = f"### Frames codebook:\n{full_frames_codebook}\n\n### Claims codebook:\n{full_claims_codebook}"

POSITIVES_SYSTEM = f"""You are writing realistic texts about wind energy opposition in the style and register of a real-world training corpus (news articles, op-eds, council and regulatory proceedings, letters to the editor, research summaries). Your job is to GENERATE new training examples for a classifier — not classify. Match the register of the seed examples you will be shown; do not drift into other styles.

{CODEBOOK}"""

NEGATIVES_SYSTEM = f"""You are writing NEGATIVE (non-opposition) training examples for a wind-opposition classifier. These are texts that touch the same topics/keywords as a target opposition code — making them superficially confusable — but the speaker is NOT opposing wind energy. They train the classifier to distinguish "this topic is discussed" from "wind is being opposed."

Match the register of the seed examples you will be shown (typically news articles, op-eds, council proceedings, letters to the editor, research summaries). Do not drift into other styles.

{CODEBOOK}"""

POSITIVES_USER = """Target code: **{code}** (see codebook in system for its definition)

### Real training examples that include {code} (with full multi-label gold labels — use as style/register anchors, not scenario templates):
{exemplars}

Write {n} NEW texts that each satisfy {code}'s definition. Vary locations, speakers, organizations, scenarios widely; do NOT reuse the seeds' specifics. Match the corpus style of the seeds (register, length, sentence structure). Multi-label is expected — texts may naturally pick up other codes when they fit."""

NEGATIVES_USER = """Target code: **{code}** (see codebook in system for its definition)

### Real training examples that include {code} (with full multi-label gold labels):
{exemplars}

Write {n} NEW texts that a classifier might initially confuse with {code} — they should touch similar topics, keywords, or rhetorical register as the seeds — but the speaker is NOT opposing wind energy at all.

Each generated text must be a **non-opposition** text (`C_0_0` / opposition_detected = false). Valid angles:
- Neutral news reporting on the same topic (permitting decisions, project announcements, studies, market data).
- Defending wind or pushing back against opposition — advocates, industry, researchers, pro-wind community voices.
- Factual / explanatory texts: how projects work, what regulations exist, what studies have found.
- Personal anecdotes from people who like wind or are indifferent.

Keep the topical overlap with the seeds (same keywords, same issues, same register) so the near-miss is genuinely confusable at first glance. The speaker just happens to not be opposing wind.

Style should match the register of the seeds. No bullet points, no preambles."""


def main() -> None:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--codes", help="Comma- or space-separated target codes (e.g. C_28_0,N_0).")
    g.add_argument("--preset", choices=list(PRESETS), help="Pre-registered code list (positives/negatives).")
    p.add_argument("--negatives", action="store_true", help="Near-miss negatives instead of positives. Implied by --preset negatives.")
    p.add_argument("--per-code-target", type=int, default=30, help="Target total support (seeds + existing synth + this run).")
    p.add_argument("--concurrency", type=int, default=15)
    args = p.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("OPENROUTER_API_KEY not set.")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    if args.preset:
        codes = list(PRESETS[args.preset])
        neg = args.preset == "negatives"
    else:
        codes = [c for part in args.codes.replace(",", " ").split() for c in [part.strip()] if c]
        neg = args.negatives
    id_prefix = "neg" if neg else "pos"
    mode_tag = "negatives" if neg else "positives"

    all_seeds = [json.loads(l) for l in open(SEEDS)]

    # Scan existing output once: collect content set (for dedup) + per-code counts.
    done_content, counts = set(), {}
    if OUT.exists():
        for line in open(OUT):
            try:
                r = json.loads(line)
                done_content.add(r["content"].strip())
                for c in codes:
                    if f"synth_{id_prefix}_{c}_" in r.get("itemId", ""):
                        counts[c] = counts.get(c, 0) + 1
            except Exception:
                pass

    gen_system = [{"type": "text", "text": NEGATIVES_SYSTEM if neg else POSITIVES_SYSTEM, "cache_control": {"type": "ephemeral"}}]
    lbl_system = [{"type": "text", "text": full_system_instruction, "cache_control": {"type": "ephemeral"}}]
    user_template = NEGATIVES_USER if neg else POSITIVES_USER

    print(f"Mode: {mode_tag}  |  codes: {', '.join(codes)}  |  target: {args.per_code_target}")
    print(f"Output: {OUT}\n")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "a") as fout:
        for code in codes:
            seeds = [r for r in all_seeds if code in (r.get("true_frames") or []) or code in (r.get("true_claims") or [])]
            if not seeds:
                print(f"{code}: no seed rows in train — skipping\n")
                continue
            existing_synth = counts.get(code, 0)
            existing_total = existing_synth + (len(seeds) if not neg else 0)
            need = max(0, args.per_code_target - existing_total)
            print(f"=== {code} ({mode_tag}) ===  seeds={len(seeds)}  synth={existing_synth}  need={need}")
            if need == 0:
                print("  target met; skipping\n")
                continue

            exemplars = "\n\n".join(
                f'- frames={s.get("true_frames") or []} claims={s.get("true_claims") or []}\n  "{s["content"]}"'
                for s in seeds
            )
            @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=4, max=60),
                   retry=retry_if_exception_type(Exception))
            def _gen_call():
                return client.chat.completions.parse(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": gen_system},
                        {"role": "user", "content": user_template.format(code=code, exemplars=exemplars, n=need)},
                    ],
                    temperature=0,
                    max_tokens=8000,
                    response_format=GeneratedTexts,
                )

            try:
                gen_resp = _gen_call()
                msg = gen_resp.choices[0].message
                if msg.parsed is not None and msg.parsed.texts:
                    texts = msg.parsed.texts
                else:
                    # OpenRouter → Claude sometimes returns json_schema-compatible
                    # content but the SDK's strict-mode parse fails. Fallback to
                    # manual JSON parse of message.content.
                    raw = (msg.content or "").strip()
                    # Strip fenced code blocks if the model wrapped the JSON.
                    if raw.startswith("```"):
                        raw = raw.split("```", 2)[1]
                        if raw.startswith("json"):
                            raw = raw[4:]
                        raw = raw.strip()
                    texts = json.loads(raw).get("texts", [])
            except Exception as e:
                try:
                    raw_head = (gen_resp.choices[0].message.content or "")[:300]
                except Exception:
                    raw_head = "(no response)"
                print(f"  gen parse fail: {e}\n    raw head: {raw_head!r}\n")
                continue
            texts = [t for t in (s.strip() for s in texts) if len(t) >= 20 and t not in done_content]
            print(f"  generated: {len(texts)} unique new texts")
            if not texts:
                print()
                continue

            is_frame = code.startswith("N_")

            @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=4, max=60),
                   retry=retry_if_exception_type(Exception))
            def label(text: str):
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": lbl_system},
                        {"role": "user", "content": f"### Text:\n{text}"},
                    ],
                    temperature=0,
                    max_tokens=2000,
                )
                parsed = parse_response(resp.choices[0].message.content)
                if parsed["opposition_detected"] is None:
                    return None
                if neg:
                    # Negatives = C_0_0 non-opposition texts that could superficially
                    # look like the target code but teacher confirms no opposition.
                    if parsed["opposition_detected"]:
                        return None
                else:
                    # Positives: teacher must predict the target code.
                    labels = parsed["frames"] if is_frame else parsed["claims"]
                    if code not in labels:
                        return None
                return {
                    "content": text,
                    "true_opposition_detected": bool(parsed["opposition_detected"]),
                    "true_frames": list(parsed["frames"]),
                    "true_claims": list(parsed["claims"]),
                }

            kept = []
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futs = [pool.submit(label, t) for t in texts]
                for fut in tqdm(as_completed(futs), total=len(futs), desc=f"  label {code}"):
                    try:
                        rec = fut.result()
                    except Exception as e:
                        tqdm.write(f"  label error: {e}")
                        continue
                    if rec:
                        kept.append(rec)

            for i, rec in enumerate(kept, start=existing_synth):
                fout.write(json.dumps({
                    "itemId": f"synth_{id_prefix}_{code}_{i:03d}",
                    **rec,
                }, ensure_ascii=False) + "\n")
                done_content.add(rec["content"].strip())
            fout.flush()
            print(f"  kept: {len(kept)}/{len(texts)}\n")


if __name__ == "__main__":
    main()
