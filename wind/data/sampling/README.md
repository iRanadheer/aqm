# Recipe — `aerows_augcards_ctts_6kky59.jsonl`

`sample.py` here produces **`aerows_augcards_ctts_6kky59.jsonl`** — the
annotation-tool **input** for the wind annotation round (1,464 rows).
Nothing in this folder produces `aerows_full.jsonl`; that file is the
**output** of the annotation tool, after annotators worked through this
input and dropped/merged rows.

```
this folder
   ├─ sample.py
   ├─ augmented_cards.csv
   └─ ctts.parquet
                       │
                       ▼  (sample.py)
   aerows_augcards_ctts_6kky59.jsonl  (1,464 rows)
                       │
                       ▼  (annotation tool, external — not in this repo)
   data/raw/aerows_full.jsonl         (1,116 rows: 5 ICR/benchmark
                                        + 1,111 survivors of the above)
```

`sample.py` reproduces the 1,464-row file deterministically (seed 42).
**Verified against `data/raw/aerows_full.jsonl`: every one of its 1,111
non-ICR rows matches a row produced here, by id AND text.**

```
python sample.py
```

## Composition

| ID prefix | Rows | Source |
|-----------|------|--------|
| `aug_sample_*`                 | 750 | `augmented_cards.csv`, stratified slice |
| `aug_osample_*`                | 214 | `augmented_cards.csv`, oversampled class-1 |
| `heritage_*`                   | 100 | `ctts.parquet`, www.heritage.org |
| `heartland_*`                  | 100 | `ctts.parquet`, heartland.org |
| `instituteforenergyresearch_*` | 100 | `ctts.parquet`, www.instituteforenergyresearch.org |
| `texaspolicy_*`                | 100 | `ctts.parquet`, www.texaspolicy.com |
| `cei_*`                        | 100 | `ctts.parquet`, cei.org |
| **Total**                      | **1,464** | |

Output schema (per line): `{"id": "...", "text": "..."}`.

Congress is intentionally **not** included — descoped after the September
prototype round.

---

## Part A — Augmented Cards (964 rows)

**Source:** `augmented_cards.csv` (75,653 rows).

**Filter:** `text.str.lower().str.strip()`, then substring match on
`'wind|renewable'` → **1,347-row pool** (862 class-0 + 485 class-1).

**Sample:** `stratified_oversample(pool, 'binary_claim', total_sample_size=1000, random_state=42)`.

The function takes a 75% stratified slice that mirrors the natural class
balance, then oversamples remaining class-1 rows to fill the rest:

| Step | Class 0 | Class 1 | Total |
|------|---------|---------|-------|
| Stratified slice (75% × 1000) | int(750 × 0.6399) = 479 | 750 − 479 = 271 | 750 |
| Oversample class-1 (target 250, capped by available) | 0 | min(250, 485 − 271) = 214 | 214 |
| **Final** | **479 (49.7%)** | **485 (50.3%)** | **964** |

The oversample step hits its cap (only 214 class-1 rows left in the pool
after the stratified slice). When the cap is hit, `sample.py` takes all
remaining rows in pool order rather than reshuffling — this is what the
reference output expects.

IDs:
- `aug_sample_<i>` for the 750 stratified rows
- `aug_osample_<i>` for the 214 oversampled rows

---

## Part B — Conservative Think Tanks (500 rows)

**Source:** `ctts.parquet` (1,182 rows). The parquet has no `domain` column
— it's derived from `url`.

**Preprocessing:**

```python
df = pd.read_parquet('ctts.parquet')

# Relative URLs (~15% of rows) are heritage.org with the host stripped.
df.loc[~df.url.str.startswith('http'), 'url'] = (
    'https://www.heritage.org' + df.loc[~df.url.str.startswith('http'), 'url']
)
df['domain'] = df['url'].str.split('/').str[2]

us_ctt = [
    'www.heritage.org', 'heartland.org',
    'www.instituteforenergyresearch.org', 'cei.org', 'www.texaspolicy.com',
]
df = df[df.domain.isin(us_ctt)].reset_index(drop=True)

# Paragraph-level explode + normalize
df.text = df.text.str.split('\n')
df = df.explode('text').reset_index(drop=True)
df.text = df.text.str.lower().str.strip()
```

**Filter:** substring match on `'wind|renewable'` → **8,467-row pool**.

| Domain                              | Pool rows |
|-------------------------------------|-----------|
| www.instituteforenergyresearch.org  | 4,063     |
| heartland.org                       | 2,515     |
| www.heritage.org                    | 1,038     |
| www.texaspolicy.com                 |   709     |
| cei.org                             |   142     |

**Sample:** 100 per domain via the direct grouped-sample API:

```python
sample = pool.groupby('domain').sample(n=100, random_state=42).reset_index(drop=True)
```

This is **not** equivalent to `groupby().apply(lambda g: g.sample(n=100, random_state=42))` — that pattern doesn't honor the seed consistently across groups. The direct `.sample()` API is what reproduces the reference output byte-for-byte.

IDs:

```python
sample['domain_name'] = sample.domain.str.replace('www.', '').str.split('.').str[0]
sample['id'] = sample.domain_name + '_' + sample.index.astype(str)
```

`cei.org` has 142 pool rows, so `n=100` works without underflow.

---

## Final concat

```python
out = pd.concat([ctts_sample, aug_sample], ignore_index=True)[['id', 'text']]
out.to_json('aerows_augcards_ctts_6kky59.jsonl', orient='records', lines=True)
# → 1,464 rows: 500 CTTs followed by 964 aug-cards
```

---

## Reproducibility — what the byte-stable output requires

| Knob | Value |
|---|---|
| Aug-cards filter | substring `'wind\|renewable'` (not `'wind'` token-membership) |
| Aug-cards `total_sample_size` | 1000 |
| Aug-cards seed | 42 |
| CTT keyword filter | substring `'wind\|renewable'` |
| CTT cap per domain | 100 |
| CTT sampling API | `groupby('domain').sample(n=100, random_state=42)` (direct, not via `.apply()`) |
| Oversample at-cap behavior | take all remaining class-1 in pool order; do not reshuffle |
