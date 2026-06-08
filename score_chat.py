"""Score chat-format eval results against text-level gold.

Chat responses are nested per-claim YAML:
  claims:
    - claim: "..."
      [frames: [N_4, ...]]        # wind only
      categories:
        - code: <code>
          reason: "..."

We aggregate to the text level (union of codes across claims) and reuse each
project's own metric functions, so the numbers line up with the API tables:
  cards  -> generate_report.compute_metrics (samples-F1 + macro-F1, min_support=3)
  wind   -> generate_report.{detection,multilabel}_metrics

Codes are pulled with regex, NOT yaml.safe_load — unquoted `4_1_1` parses as
the int 411 under YAML 1.1 digit-separator rules.

Usage:
  python score_chat.py cards <results.jsonl>
  python score_chat.py wind  <results.jsonl>
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _yaml_after_think(resp: str) -> str:
    if not isinstance(resp, str):
        return ""
    after = resp.split("</think>")[-1] if "</think>" in resp else resp
    m = re.search(r"```yaml\s*\n(.*?)```", after, re.DOTALL)
    return m.group(1) if m else after


def extract_cards_codes(resp: str):
    """Union of category codes; 0_0_0 only if nothing else fires. [] = parse fail."""
    block = _yaml_after_think(resp)
    codes = set(re.findall(r"code:\s*['\"]?([0-9_]+)", block))
    if not codes:
        return []                      # parse failure -> empty (scored as miss)
    real = sorted(c for c in codes if c != "0_0_0")
    return real if real else ["0_0_0"]


def extract_wind(resp: str):
    """Return (opposition_bool, frames_list, claims_list) or None on parse fail."""
    block = _yaml_after_think(resp)
    if "claims:" not in block:
        return None
    frames = set()
    for grp in re.findall(r"frames:\s*\[([^\]]*)\]", block):
        for tok in grp.split(","):
            t = tok.strip().strip("'\"")
            if re.fullmatch(r"N_\d+", t):
                frames.add(t)
    codes = set()
    for c in re.findall(r"code:\s*['\"]?([A-Za-z0-9_]+)", block):
        if c.lower() != "none" and re.fullmatch(r"C_[0-9_]+", c):
            codes.add(c)
    opp = bool(frames or codes)
    return opp, sorted(frames), sorted(codes)


def score_cards(path):
    sys.path.insert(0, str(ROOT / "cards"))
    import generate_report as cr
    import pandas as pd
    gold = cr.load_canonical_gold()
    df = pd.read_json(path, lines=True)
    df = df[df["text"].isin(gold)].copy()
    df["true_claims"] = df["text"].map(gold)
    df["pred_claims"] = df["response"].apply(extract_cards_codes)
    pf = int((df["pred_claims"].map(len) == 0).sum())
    print(f"cards: {len(df)} rows, {pf} parse failures")
    print(f"{'level':<6} {'samples_f1':>11} {'macro_f1':>9}")
    for lvl in (1, 2, 3):
        m = cr.compute_metrics(df, lvl, 3)   # min_support=3, matches API tables
        print(f"L{lvl:<5} {m['samples_f1']:>11.3f} {m['macro_f1']:>9.3f}")


def score_wind(path):
    sys.path.insert(0, str(ROOT / "wind"))
    import generate_report as wr
    rows = [json.loads(l) for l in open(path)]
    yt_op = yp_op = None
    yt_op, yp_op, yt_f, yp_f, yt_c, yp_c = [], [], [], [], [], []
    SENT = ["__PARSE_FAIL__"]
    pf = 0
    for r in rows:
        gold_op = bool(r.get("true_opposition_detected", False))
        gold_f = list(r.get("true_frames") or [])
        gold_c = list(r.get("true_claims") or [])
        resp = r.get("response", "")
        parsed = None if (isinstance(resp, str) and resp.startswith("ERROR:")) else extract_wind(resp)
        yt_op.append(gold_op); yt_f.append(gold_f); yt_c.append(gold_c)
        if parsed is None:
            pf += 1
            yp_op.append(not gold_op); yp_f.append(list(SENT)); yp_c.append(list(SENT))
        else:
            op, fr, cl = parsed
            yp_op.append(op); yp_f.append(fr); yp_c.append(cl)
    opp_idx = [i for i, t in enumerate(yt_op) if t]
    det = wr.detection_metrics(yt_op, yp_op)
    fr_all = wr.multilabel_metrics(yt_f, yp_f)
    cl_all = wr.multilabel_metrics(yt_c, yp_c)
    fr_opp = wr.multilabel_metrics([yt_f[i] for i in opp_idx], [yp_f[i] for i in opp_idx])
    cl_opp = wr.multilabel_metrics([yt_c[i] for i in opp_idx], [yp_c[i] for i in opp_idx])
    print(f"wind: {len(rows)} rows, {pf} parse failures, {len(opp_idx)} opposition rows")
    print(f"  detection      F1={det['f1']:.3f}  P={det['precision']:.3f}  R={det['recall']:.3f}")
    print(f"  frames (all)   sF1={fr_all['samples_f1']:.3f}  macro={fr_all['macro_f1']:.3f}")
    print(f"  claims (all)   sF1={cl_all['samples_f1']:.3f}  macro={cl_all['macro_f1']:.3f}")
    print(f"  frames (opp)   sF1={fr_opp['samples_f1']:.3f}  macro={fr_opp['macro_f1']:.3f}")
    print(f"  claims (opp)   sF1={cl_opp['samples_f1']:.3f}  macro={cl_opp['macro_f1']:.3f}")


if __name__ == "__main__":
    which, path = sys.argv[1], sys.argv[2]
    (score_cards if which == "cards" else score_wind)(path)
