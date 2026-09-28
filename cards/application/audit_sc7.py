"""SC-7 ('We need fossil fuels') precision audit with an independent Claude judge.

Samples SC-7-tagged speeches per party and asks the judge whether the speaker
genuinely advances a 'we need fossil fuels' claim (true positive) vs rebuts /
mentions neutrally / off-target (false positive). Writes sc7_audit.jsonl and
prints precision per party.
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from dotenv import load_dotenv
import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

from claims import parse_claims

load_dotenv("/home/projects/overbeef/aqm/.env")
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
JUDGE = "claude-sonnet-4-6"
N_PER_PARTY = 80
OUT = "sc7_audit.jsonl"

DEFINITION = (
    "CLAIM 'We need fossil fuels' (CARDS super-claim 7): the text argues that "
    "fossil fuels (coal, oil, gas) are plentiful, good, necessary, beneficial, "
    "or that we have a right to use them — e.g. for energy security, "
    "affordability, reliability, or jobs. The SPEAKER must be advancing/endorsing "
    "this claim, not rebutting it."
)
PROMPT = """You are auditing a climate-claim classifier. A classifier tagged the speech below as making the claim:

{definition}

Decide whether that tag is correct. Reply with ONLY compact JSON:
{{"genuine": true|false, "mode": "endorses"|"rebuts"|"neutral_mention"|"off_target", "reason": "<=15 words"}}

- genuine=true only if the speaker is genuinely advancing the 'we need fossil fuels' claim.
- "rebuts": speaker argues AGAINST fossil fuels / quotes the claim to criticize it.
- "neutral_mention": fossil fuels mentioned without advocating for them.
- "off_target": about something else (e.g. unrelated industry, procedure).

SPEECH:
{text}"""


@retry(stop=stop_after_attempt(4), wait=wait_exponential(max=30))
def judge(text):
    m = client.messages.create(
        model=JUDGE, max_tokens=120,
        messages=[{"role": "user", "content": PROMPT.format(definition=DEFINITION, text=text[:6000])}])
    s = m.content[0].text.strip()
    s = s[s.find("{"): s.rfind("}") + 1]
    return json.loads(s)


def main():
    df = pd.DataFrame(json.loads(l) for l in open("speech_cards_predictions.jsonl"))
    df["codes"] = df["content"].map(parse_claims)
    df["sc7"] = df["codes"].map(lambda cs: any(c.split("_")[0] == "7" for c in cs))
    pool = df[df.sc7 & df.party.isin(["Democrat", "Republican"])]

    sample = pd.concat([pool[pool.party == p].sample(min(N_PER_PARTY, (pool.party == p).sum()),
                        random_state=0) for p in ["Democrat", "Republican"]])
    recs = sample[["idx", "party", "speaker", "state", "text", "codes"]].to_dict("records")

    def run(r):
        try:
            r = {**r, **judge(r["text"])}
        except Exception as e:
            r = {**r, "genuine": None, "mode": "ERROR", "reason": str(e)[:80]}
        return r

    with ThreadPoolExecutor(12) as ex:
        results = list(ex.map(run, recs))

    with open(OUT, "w") as f:
        for r in results:
            f.write(json.dumps({k: r[k] for k in ("idx", "party", "speaker", "state",
                    "genuine", "mode", "reason")}, ensure_ascii=False) + "\n")

    print(f"\nSC-7 precision audit (judge={JUDGE}, n={len(results)})\n" + "=" * 50)
    for party in ["Democrat", "Republican"]:
        sub = [r for r in results if r["party"] == party and r["genuine"] is not None]
        tp = sum(1 for r in sub if r["genuine"])
        from collections import Counter
        modes = Counter(r["mode"] for r in sub)
        print(f"\n{party}: precision = {tp}/{len(sub)} = {tp/len(sub)*100:.0f}%")
        print("  modes:", dict(modes))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
