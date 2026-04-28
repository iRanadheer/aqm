"""
Reproduces aerows_augcards_ctts_6kky59.jsonl — the annotation-tool input
for the wind annotation round (1,464 rows: 964 augmented-cards + 500 CTTs).

Run from this folder (data/sampling/) (or adjust paths in main()):
    python sample.py
"""
from __future__ import annotations
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

US_CTT_DOMAINS = [
    "www.heritage.org",
    "heartland.org",
    "www.instituteforenergyresearch.org",
    "cei.org",
    "www.texaspolicy.com",
]


def stratified_oversample(
    df: pd.DataFrame,
    target_col: str = "binary_claim",
    total_sample_size: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    """75% stratified slice (preserves natural class balance) + oversample
    class-1 to fill the remaining 25%, capped by available class-1 rows."""
    np.random.seed(random_state)
    proportions = df[target_col].value_counts(normalize=True)

    stratified_size = int(total_sample_size * 0.75)
    n0 = int(stratified_size * proportions[0])
    n1 = stratified_size - n0

    df_0 = df[df[target_col] == 0]
    df_1 = df[df[target_col] == 1]
    strat_0 = df_0.sample(n=n0, random_state=random_state).copy()
    strat_1 = df_1.sample(n=n1, random_state=random_state).copy()
    strat_0["id"] = [f"aug_sample_{i}" for i in range(len(strat_0))]
    strat_1["id"] = [f"aug_sample_{i}" for i in range(len(strat_0), len(strat_0) + len(strat_1))]

    remaining_1s = df_1.drop(strat_1.index)
    n_oversample = min(total_sample_size - stratified_size, len(remaining_1s))
    # Round-2 quirk: when the oversample target ≥ remaining pool, take all
    # remaining rows in pool order rather than shuffling via .sample().
    if n_oversample == len(remaining_1s):
        osample_1 = remaining_1s.copy()
    else:
        osample_1 = remaining_1s.sample(n=n_oversample, random_state=random_state).copy()
    osample_1["id"] = [f"aug_osample_{i}" for i in range(len(osample_1))]

    return pd.concat([strat_0, strat_1, osample_1], ignore_index=True)


def sample_aug_cards(path: str | Path) -> pd.DataFrame:
    """964 rows from augmented_cards.csv via 'wind|renewable' substring +
    stratified_oversample(total=1000, seed=42)."""
    df = pd.read_csv(path)
    df["text"] = df.text.str.lower().str.strip()
    pool = df[df.text.str.contains("wind|renewable", na=False)].reset_index(drop=True)
    sample = stratified_oversample(pool, "binary_claim", total_sample_size=1000, random_state=42)
    return sample[["id", "text"]]


def sample_ctts(path: str | Path) -> pd.DataFrame:
    """500 rows from ctts.parquet (5 conservative think tanks × 100), via
    'wind|renewable' substring + groupby('domain').sample(n=100, seed=42)."""
    df = pd.read_parquet(path)

    # Relative URLs (~15% of rows) belong to heritage.org
    rel_mask = ~df.url.str.startswith("http")
    df.loc[rel_mask, "url"] = "https://www.heritage.org" + df.loc[rel_mask, "url"]
    df["domain"] = df["url"].str.split("/").str[2]

    df = df[df.domain.isin(US_CTT_DOMAINS)].reset_index(drop=True)

    # Paragraph-level explode + normalize
    df.text = df.text.str.split("\n")
    df = df.explode("text").reset_index(drop=True)
    df.text = df.text.str.lower().str.strip()

    pool = df[df.text.str.contains("wind|renewable", na=False)].reset_index(drop=True)
    sample = pool.groupby("domain").sample(n=100, random_state=42).reset_index(drop=True)

    sample["domain_name"] = sample.domain.str.replace("www.", "").str.split(".").str[0]
    sample["id"] = sample.domain_name + "_" + sample.index.astype(str)
    return sample[["id", "text"]]


def main(
    aug_cards_csv: str | Path = "augmented_cards.csv",
    ctts_parquet: str | Path = "ctts.parquet",
    output_jsonl: str | Path = "aerows_augcards_ctts_6kky59.jsonl",
) -> None:
    ctts = sample_ctts(ctts_parquet)         # 500 rows
    aug = sample_aug_cards(aug_cards_csv)    # 964 rows
    out = pd.concat([ctts, aug], ignore_index=True)
    assert len(out) == 1464, f"Expected 1464 rows, got {len(out)}"
    out.to_json(output_jsonl, orient="records", lines=True)
    print(f"Wrote {output_jsonl} ({len(out)} rows)")


if __name__ == "__main__":
    main()
