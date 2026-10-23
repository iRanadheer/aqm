"""Split documents into passages and classify each with the Windy model (vLLM, OpenAI API).
In: JSONL with url + text. Out: one JSONL row per passage (resumable).
Usage: python classify_passages.py docs.jsonl out.jsonl [--base http://localhost:8000/v1] [--limit N]
"""
import asyncio, json, re, sys, argparse, os
from huggingface_hub import hf_hub_download
from openai import AsyncOpenAI

MAXW = 120  # ponytail: test passages median 52 / p90 111 words; pack paragraphs up to ~120

def passages(text):
    out, cur = [], []
    for p in re.split(r"\n\s*\n", text):
        p = " ".join(p.split())
        if not p: continue
        words = p.split()
        for i in range(0, len(words), MAXW):  # hard-split overlong paragraphs
            chunk = " ".join(words[i:i + MAXW])
            if cur and len(" ".join(cur).split()) + len(chunk.split()) > MAXW:
                out.append(" ".join(cur)); cur = []
            cur.append(chunk)
    if cur: out.append(" ".join(cur))
    return [p for p in out if len(p.split()) >= 8]

async def main(a):
    sysp = json.load(open(hf_hub_download("C3DS/Windy-Qwen3.5-27B-FP8", "wind_prompts.json")))["slim_system_instruction"]
    cli = AsyncOpenAI(base_url=a.base, api_key="x", timeout=600)
    done = set()
    if os.path.exists(a.out):
        done = {(r["url"], r["pid"]) for r in map(json.loads, open(a.out))}
    jobs = []
    for i, l in enumerate(open(a.docs)):
        if a.limit and i >= a.limit: break
        d = json.loads(l)
        for pid, p in enumerate(passages(d["text"])):
            if (d["url"], pid) not in done: jobs.append((d["url"], pid, p))
    print(f"{len(jobs)} passages to classify", file=sys.stderr)
    sem, fo, n = asyncio.Semaphore(a.conc), open(a.out, "a"), 0

    async def one(url, pid, p):
        nonlocal n
        async with sem:
            try:
                r = await cli.chat.completions.create(model="windy", temperature=0, max_tokens=4000,
                    messages=[{"role": "system", "content": sysp}, {"role": "user", "content": p}])
                resp = r.choices[0].message.content
            except Exception as e:
                resp = None; print("ERR", url, pid, e, file=sys.stderr)
        if resp is not None:  # failures aren't written, so a rerun retries them
            fo.write(json.dumps({"url": url, "pid": pid, "passage": p, "response": resp}) + "\n"); fo.flush()
        n += 1
        if n % 200 == 0: print(n, "/", len(jobs), file=sys.stderr)

    await asyncio.gather(*(one(*j) for j in jobs))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("docs"); ap.add_argument("out")
    ap.add_argument("--base", default="http://localhost:8000/v1")
    ap.add_argument("--conc", type=int, default=64)  # matches --max-num-seqs
    ap.add_argument("--limit", type=int, default=0)
    asyncio.run(main(ap.parse_args()))
