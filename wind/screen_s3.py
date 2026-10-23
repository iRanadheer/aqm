"""Stream CTT crawls from S3 in parallel, dedupe by URL, keyword-screen for wind/renewable.
Out: docs.jsonl with url, spider, title, published_date, text."""
import gzip, io, json, re, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from dotenv import dotenv_values

E = dotenv_values(os.path.join(os.path.dirname(__file__), "..", ".env"))
ENV = {**os.environ, "RCLONE_CONFIG_S3_TYPE": "s3", "RCLONE_CONFIG_S3_PROVIDER": "Other",
       "RCLONE_CONFIG_S3_ACCESS_KEY_ID": E["S3_ACCESS_KEY"], "RCLONE_CONFIG_S3_SECRET_ACCESS_KEY": E["S3_SECRET_KEY"],
       "RCLONE_CONFIG_S3_ENDPOINT": E["S3_ENDPOINT"]}

WIND = re.compile(r"\bwind\b")  # ponytail: chapter says space-delimited; \b also catches wind, / wind.

def read(key):
    p = subprocess.Popen(["rclone", "cat", f"s3:{key}"], stdout=subprocess.PIPE, env=ENV, stderr=subprocess.DEVNULL)
    out = []
    src = gzip.GzipFile(fileobj=p.stdout) if key.endswith(".gz") else p.stdout  # key = bucket/path
    with io.TextIOWrapper(src, encoding="utf-8", errors="replace") as f:
        for l in f:
            try: d = json.loads(l)
            except ValueError: continue
            c = (d.get("content") or "").strip()
            if c and d.get("url"):
                out.append({"url": d["url"], "spider": d.get("spider_name"), "title": d.get("title"),
                            "published_date": d.get("published_date"), "text": c})
    p.wait()
    return key, out

if __name__ == "__main__":
    keys = [l.split(None, 1)[1].strip() for l in open(sys.argv[1])]
    seen, n_total, hits = set(), 0, []
    with ThreadPoolExecutor(int(sys.argv[3]) if len(sys.argv) > 3 else 16) as ex:
        for i, (k, docs) in enumerate(ex.map(read, keys), 1):
            for d in docs:
                if d["url"] in seen: continue
                seen.add(d["url"]); n_total += 1
                t = d["text"].lower()
                d["match"] = ("wind" if WIND.search(t) else "") + ("renewable" if "renewable" in t else "")
                if d["match"]: hits.append(d)
            print(f"{i}/{len(keys)} {k} total={n_total} hits={len(hits)}", file=sys.stderr)
    with open(sys.argv[2], "w") as f:
        for d in hits: f.write(json.dumps(d) + "\n")
    print("unique docs:", n_total, "screened:", len(hits))
