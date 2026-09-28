import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from openai import (OpenAI, BadRequestError, RateLimitError,
                    APIConnectionError, APITimeoutError, InternalServerError)
from tenacity import (retry, stop_after_attempt, wait_exponential,
                      retry_if_exception_type)
from tqdm import tqdm

CSV = "/home/projects/overbeef/aqm/cards/application/speech_paragraphs_with_predictions.csv"
OUT = "/home/projects/overbeef/aqm/cards/application/speech_cards_predictions.jsonl"
ERR = "/home/projects/overbeef/aqm/cards/application/speech_cards_errors.log"
WORKERS = 16
MAX_TOKENS = 1024     # output is short (~300 tok); leaves the rest of the 8192 ctx for input

client = OpenAI(
    api_key="dl-cards-e992b7ae87d3b18935c7020df2ee94a7b612018bf622f9d1",
    base_url="https://api.discourselab.ai/v1",
)


# Retry only transient failures; a 400 (e.g. context length) is deterministic.
@retry(retry=retry_if_exception_type((RateLimitError, APIConnectionError,
                                      APITimeoutError, InternalServerError)),
       stop=stop_after_attempt(5), wait=wait_exponential(max=60))
def _call(text):
    return client.chat.completions.create(
        model="cards-latest",
        messages=[{"role": "user", "content": text}],
        temperature=0,
        max_tokens=MAX_TOKENS,
    )


def infer(row):
    try:
        m = _call(row["text"])
        return {**row,
                "reasoning_content": getattr(m.choices[0].message, "reasoning_content", None),
                "content": m.choices[0].message.content,
                "usage": m.usage.model_dump() if m.usage else None}
    except BadRequestError as e:  # deterministic (e.g. context too long) — skip for good
        return {"idx": row["idx"], "error": f"BadRequestError: {e}", "permanent": True}
    except Exception as e:        # other failures: recorded, retried on a future run
        return {"idx": row["idx"], "error": f"{type(e).__name__}: {e}"}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=WORKERS)
    args = ap.parse_args()

    # Skip successes and permanent (e.g. too-long) failures; only transient errors retry.
    done = {json.loads(l)["idx"] for l in open(OUT)} if os.path.exists(OUT) else set()
    if os.path.exists(ERR):
        done |= {r["idx"] for l in open(ERR) if (r := json.loads(l)).get("permanent")}
    df = pd.read_csv(CSV)
    rows = [{"idx": i, **r} for i, r in zip(df.index, df.to_dict("records"))
            if i not in done and isinstance(r["text"], str)]
    with ThreadPoolExecutor(args.workers) as ex, open(OUT, "a") as f, open(ERR, "a") as ef:
        futs = [ex.submit(infer, r) for r in rows]
        # as_completed -> write each row the instant it finishes (no head-of-line
        # wait on a slow row); miniters=1/smoothing=0 keep the bar honest.
        for fut in tqdm(as_completed(futs), total=len(futs), miniters=1, smoothing=0):
            out = fut.result()
            t = ef if "error" in out else f          # errors -> errors.log, not OUT
            t.write(json.dumps(out, ensure_ascii=False) + "\n")
            t.flush()
