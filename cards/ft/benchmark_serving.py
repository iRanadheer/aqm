# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "vllm>=0.17.1",
#     "requests",
#     "pandas",
#     "huggingface_hub[hf_transfer]",
# ]
# ///

"""
Serving benchmark: CARDS Qwen3.5 bf16 vs FP8.

Industry-standard vLLM serving benchmark. For each (size, precision) pair:
  1. Start `vllm serve` in background
  2. Run `vllm bench serve` against it at several concurrency levels
  3. Collect JSON results, shutdown, move on.

Outputs:
  bench_results/{size}_{precision}_c{concurrency}.json   (raw vllm bench JSON)
  bench_results/summary.csv                              (wide comparison table)
  bench_results/summary.md                               (human-readable)

Uploaded to dataset repo `C3DS/cards-qwen35-benchmarks` by default.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

MODELS = {
    "4b":  ("C3DS/CARDS-Qwen3.5-4B",  "C3DS/CARDS-Qwen3.5-4B-FP8"),
    "9b":  ("C3DS/CARDS-Qwen3.5-9B",  "C3DS/CARDS-Qwen3.5-9B-FP8"),
    "27b": ("C3DS/CARDS-Qwen3.5-27B", "C3DS/CARDS-Qwen3.5-27B-FP8"),
}

parser = argparse.ArgumentParser()
parser.add_argument("--sizes", nargs="+", default=["4b", "9b", "27b"], choices=list(MODELS))
parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 16, 64])
parser.add_argument("--num-prompts", type=int, default=300)
parser.add_argument("--input-len", type=int, default=1024)
parser.add_argument("--output-len", type=int, default=256)
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--results-dir", default="bench_results")
parser.add_argument("--push-repo", default="C3DS/cards-qwen35-benchmarks")
parser.add_argument("--no-push", dest="push", action="store_false", default=True)
args = parser.parse_args()

RESULTS = Path(args.results_dir)
RESULTS.mkdir(exist_ok=True)

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")


def start_vllm(model: str) -> subprocess.Popen:
    print(f"  starting vllm serve {model} ...")
    proc = subprocess.Popen(
        [
            "vllm", "serve", model,
            "--port", str(args.port),
            "--max-model-len", "4096",
            "--disable-log-requests",
            "--gpu-memory-utilization", "0.9",
        ],
        preexec_fn=os.setsid,  # put in own group so we can SIGKILL the whole tree
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    deadline = time.time() + 900
    while time.time() < deadline:
        try:
            r = requests.get(f"http://localhost:{args.port}/v1/models", timeout=3)
            if r.status_code == 200:
                print(f"  server up after {int(time.time() - (deadline - 900))}s")
                return proc
        except requests.exceptions.RequestException:
            pass
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM exited early with code {proc.returncode}")
        time.sleep(5)
    proc.kill()
    raise TimeoutError(f"vLLM didn't come up in 15 min for {model}")


def stop_vllm(proc: subprocess.Popen) -> None:
    print("  stopping vllm ...")
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass
    time.sleep(10)  # let VRAM drain before next model


def run_bench(model: str, size: str, precision: str, concurrency: int) -> dict:
    out_name = f"{size}_{precision}_c{concurrency}.json"
    out_path = RESULTS / out_name
    print(f"\n  bench: {size}/{precision} concurrency={concurrency}")
    t = time.time()
    subprocess.run(
        [
            "vllm", "bench", "serve",
            "--model", model,
            "--host", "localhost",
            "--port", str(args.port),
            "--dataset-name", "random",
            "--random-input-len", str(args.input_len),
            "--random-output-len", str(args.output_len),
            "--num-prompts", str(args.num_prompts),
            "--max-concurrency", str(concurrency),
            "--save-result",
            "--result-dir", str(RESULTS),
            "--result-filename", out_name,
            "--seed", "42",
        ],
        check=True,
    )
    dt = time.time() - t
    print(f"  done in {dt:.1f}s")
    with open(out_path) as f:
        result = json.load(f)
    result["size"] = size
    result["precision"] = precision
    result["concurrency"] = concurrency
    return result


all_results: list[dict] = []
overall_start = time.time()

for size in args.sizes:
    bf16_repo, fp8_repo = MODELS[size]
    for precision, model in [("bf16", bf16_repo), ("fp8", fp8_repo)]:
        print(f"\n{'#'*60}")
        print(f"#  {size} / {precision}  ({model})")
        print(f"{'#'*60}")
        proc = start_vllm(model)
        try:
            for c in args.concurrency:
                all_results.append(run_bench(model, size, precision, c))
        finally:
            stop_vllm(proc)

# ---- Build summary ----
print(f"\n{'='*60}\n  Writing summary\n{'='*60}")

keep_cols = [
    "size", "precision", "concurrency",
    "request_throughput", "output_throughput",
    "mean_ttft_ms", "median_ttft_ms", "p99_ttft_ms",
    "mean_itl_ms", "median_itl_ms",
    "total_input_tokens", "total_output_tokens",
]
df = pd.DataFrame(all_results)
df = df[[c for c in keep_cols if c in df.columns]]
df = df.sort_values(["size", "concurrency", "precision"])
df.to_csv(RESULTS / "summary.csv", index=False)
print(df.to_string(index=False))

# Speedup table: fp8 vs bf16 at each concurrency
speedup_rows = []
for size in args.sizes:
    for c in args.concurrency:
        bf = df[(df["size"] == size) & (df["precision"] == "bf16") & (df["concurrency"] == c)]
        fp = df[(df["size"] == size) & (df["precision"] == "fp8")  & (df["concurrency"] == c)]
        if len(bf) and len(fp):
            speedup_rows.append({
                "size": size,
                "concurrency": c,
                "throughput_speedup_x": round(fp["output_throughput"].iloc[0] /
                                              bf["output_throughput"].iloc[0], 2),
                "ttft_ratio_fp8_over_bf16": round(fp["mean_ttft_ms"].iloc[0] /
                                                  bf["mean_ttft_ms"].iloc[0], 2),
                "bf16_tok_s": round(bf["output_throughput"].iloc[0], 1),
                "fp8_tok_s":  round(fp["output_throughput"].iloc[0], 1),
            })
speedup = pd.DataFrame(speedup_rows)
speedup.to_csv(RESULTS / "speedup.csv", index=False)
print("\nSpeedup (FP8 / bf16):")
print(speedup.to_string(index=False))

with open(RESULTS / "summary.md", "w") as f:
    f.write("# CARDS Qwen3.5 — bf16 vs FP8 Serving Benchmark\n\n")
    f.write(f"- Hardware: H200 (single GPU)\n")
    f.write(f"- Workload: `random`, input_len={args.input_len}, output_len={args.output_len}, "
            f"num_prompts={args.num_prompts}\n")
    f.write(f"- Concurrency levels: {args.concurrency}\n")
    f.write(f"- vLLM seed: 42\n\n")
    f.write("## Full results\n\n")
    f.write(df.to_markdown(index=False))
    f.write("\n\n## Speedup (FP8 / bf16)\n\n")
    f.write(speedup.to_markdown(index=False))
    f.write("\n")

print(f"\nTotal time: {(time.time() - overall_start) / 60:.1f} min")

# ---- Push to HF dataset repo ----
if args.push:
    from huggingface_hub import HfApi
    print(f"\nPushing results to {args.push_repo} ...")
    api = HfApi()
    api.create_repo(args.push_repo, repo_type="dataset", private=False, exist_ok=True)
    api.upload_folder(
        folder_path=str(RESULTS),
        repo_id=args.push_repo,
        repo_type="dataset",
        commit_message="Benchmark: bf16 vs FP8 serving",
    )
    print(f"  https://huggingface.co/datasets/{args.push_repo}")

print("\nDone.")
