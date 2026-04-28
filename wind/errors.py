"""Extract error rows from a benchmark predictions jsonl for blinded expert review.

A row is an "error" if the parsed prediction disagrees with ground truth on
detection, frames, or claims (set equality). For each error, the gold and
predicted label sets are randomly assigned to columns A and B so the expert
can adjudicate without anchoring to either label as authoritative. A separate
key file unblinds the assignment per itemId.

Outputs:
    <out_prefix>.jsonl       — blinded review file
    <out_prefix>.key.jsonl   — itemId → which side (A/B) was gold vs. pred

If --sample-per-stratum N is given, also writes <out_prefix>.sample.jsonl
and <out_prefix>.sample.key.jsonl with up to N rows per error-type signature
(e.g. 'detection', 'frames+claims', 'detection+frames+claims').

Usage:
    python3 errors.py \
        --input data/results/test/iranadheer-windy-qwen3-5-27b.jsonl \
        --out-prefix data/results/error_analysis/27b-test-errors
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from generate_report import parse_response


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-prefix", required=True,
                    help="Writes <prefix>.jsonl and <prefix>.key.jsonl")
    ap.add_argument("--seed", type=int, default=42, help="Seed for A/B randomization and sampling.")
    ap.add_argument("--sample-per-stratum", type=int, default=0,
                    help="If >0, also write <prefix>.sample.jsonl with up to N rows per "
                         "error-type signature. Strata smaller than N keep all rows.")
    args = ap.parse_args()

    in_path = Path(args.input)
    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    n_total = 0
    n_errors = 0
    n_parse_fail = 0
    n_api_err = 0
    counts = {"detection": 0, "frames": 0, "claims": 0}
    stratum_counts: dict[str, int] = defaultdict(int)

    review_rows: list[dict] = []
    key_rows: list[dict] = []

    with open(in_path) as f:
        for line in f:
            n_total += 1
            r = json.loads(line)
            resp = r.get("response", "")

            if isinstance(resp, str) and resp.startswith("ERROR:"):
                n_api_err += 1
                continue

            pred = parse_response(resp)
            if pred["opposition_detected"] is None:
                n_parse_fail += 1
                continue

            true_opp = bool(r.get("true_opposition_detected", False))
            pred_opp = bool(pred["opposition_detected"])
            true_frames = sorted(r.get("true_frames") or [])
            true_claims = sorted(r.get("true_claims") or [])
            pred_frames = sorted(pred["frames"])
            pred_claims = sorted(pred["claims"])

            error_types: list[str] = []
            if true_opp != pred_opp:
                error_types.append("detection")
            if set(true_frames) != set(pred_frames):
                error_types.append("frames")
            if set(true_claims) != set(pred_claims):
                error_types.append("claims")
            if not error_types:
                continue

            for et in error_types:
                counts[et] += 1
            stratum_counts["+".join(error_types)] += 1
            n_errors += 1

            gold_payload = {
                "opposition_detected": true_opp,
                "frames": true_frames,
                "claims": true_claims,
            }
            pred_payload = {
                "opposition_detected": pred_opp,
                "frames": pred_frames,
                "claims": pred_claims,
            }
            if rng.random() < 0.5:
                a, b = gold_payload, pred_payload
                a_is, b_is = "gold", "pred"
            else:
                a, b = pred_payload, gold_payload
                a_is, b_is = "pred", "gold"

            review_rows.append({
                "itemId": r.get("itemId"),
                "content": r.get("content"),
                "a": a,
                "b": b,
                "expert_choice": "",
                "expert_annotation": {},
                "expert_notes": "",
            })
            key_rows.append({
                "itemId": r.get("itemId"),
                "a_is": a_is,
                "b_is": b_is,
                "error_types": error_types,
            })

    # Shuffle so file ordering doesn't leak signal.
    order = list(range(len(review_rows)))
    rng.shuffle(order)
    review_rows = [review_rows[i] for i in order]
    key_rows = [key_rows[i] for i in order]

    out_jsonl = Path(f"{prefix}.jsonl")
    out_key_jsonl = Path(f"{prefix}.key.jsonl")
    _write_jsonl(out_jsonl, review_rows)
    _write_jsonl(out_key_jsonl, key_rows)

    print(f"input:        {in_path}")
    print(f"review jsonl: {out_jsonl}")
    print(f"key jsonl:    {out_key_jsonl}")
    print(f"rows:           {n_total}")
    print(f"errors written: {n_errors}  ({n_errors/n_total:.1%})")
    print(f"  detection mismatches: {counts['detection']}")
    print(f"  frames    mismatches: {counts['frames']}")
    print(f"  claims    mismatches: {counts['claims']}")
    print(f"  parse failures:       {n_parse_fail}")
    print(f"  api errors:           {n_api_err}")
    print("  strata (error-type signature → count):")
    for sig in sorted(stratum_counts, key=lambda s: (-stratum_counts[s], s)):
        print(f"    {sig:<35} {stratum_counts[sig]}")

    if args.sample_per_stratum > 0:
        n_per = args.sample_per_stratum
        by_sig: dict[str, list[int]] = defaultdict(list)
        for i, k in enumerate(key_rows):
            by_sig["+".join(k["error_types"])].append(i)
        sample_idx: list[int] = []
        sample_rng = random.Random(args.seed + 1)
        for sig, idxs in by_sig.items():
            picked = idxs if len(idxs) <= n_per else sample_rng.sample(idxs, n_per)
            sample_idx.extend(picked)
        sample_rng.shuffle(sample_idx)

        s_jsonl = Path(f"{prefix}.sample.jsonl")
        s_key_jsonl = Path(f"{prefix}.sample.key.jsonl")
        _write_jsonl(s_jsonl, [review_rows[i] for i in sample_idx])
        _write_jsonl(s_key_jsonl, [key_rows[i] for i in sample_idx])

        print()
        print(f"stratified sample (≤{n_per} per stratum): {len(sample_idx)} rows")
        print(f"  {s_jsonl}")
        print(f"  {s_key_jsonl}")


if __name__ == "__main__":
    main()
