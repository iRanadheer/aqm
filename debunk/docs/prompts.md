# Prompts (`prompts.py`)

Two independent taxonomies, one shared reasoning chain, and a tolerant
parser. Structure mirrors `wind/prompts.py`: a slim **codebook** (short
label tags, the model's output target) plus separate **assessment
guidelines** (full definitions + decision rules) folded into one
instruction template.

```
LABEL_SETS / GOLD_FIELDS        ── which labels + which gold column per variant
PROMPT_VARIANTS                 ── variant id → system instruction string
ASSESSMENT_MODELS / output_schema ── pydantic schemas for structured-output backends (Exa)
extract_raw_label / normalise_gold ── parse a model response / a gold string to a canonical label
CLIMINATOR_LEVEL_SETS / climinator_rollup ── 12→5→3→2 credibility hierarchy
```

---

## Taxonomies

| Taxonomy | Labels | Gold column |
|---|---|---|
| **veracity** | 4: `TRUE` / `MISLEADING` / `FALSE` / `UNVERIFIABLE` | `true_veracity` |
| **climinator** | 12: `CORRECT`, `ACCURATE`, `MOSTLY CORRECT`, `MOSTLY ACCURATE`, `CORRECT BUT`, `IMPRECISE`, `LACKS CONTEXT`, `UNSUPPORTED`, `MISLEADING`, `INCORRECT`, `INACCURATE`, `FLAWED REASONING` | `true_cfb_label` |

The two are scored separately — there is no mapping from one to the other.
`LABEL_SETS` and `GOLD_FIELDS` are keyed by **variant id**, so each prompt
version below points at the right label space and gold column.

## Prompt variants

Seven variants are defined in `PROMPT_VARIANTS`. Only **`veracityV2`** and
**`climinator_v4`** are production (scored by `generate_report.py` and
`evals/significance.py`); the rest are kept for the methods narrative / the
ablation trail.

| Variant | Taxonomy | Status | What it is |
|---|---|---|---|
| `veracityV1` | veracity | early | shared `_instruction_template`, 4-label codebook |
| **`veracityV2`** | veracity | **production** | the `climinator_v4` method scaled to 4 labels — same `<think>` chain, anti-anchoring framing, RAG hooks. Pairs with `climinator_v4` so cross-taxonomy comparisons are "same method, different label space". |
| `climinator` (v1) | climinator | early | shared `_instruction_template`, 12-label codebook |
| `climinator_v2` | climinator | experimental | paper-faithful: Leippold 2025 SI tier-grouped codebook + verbatim per-label examples |
| `climinator_v3` | climinator | experimental | v2 codebook + a `SCAN`/`VERIFY` shortlist (borrowed from `cards/`) to stop models emitting reasoning codes as the final label |
| **`climinator_v4`** | climinator | **production** | single `SHORTLIST` step that walks **all 12 labels** with a forced `SHORTLIST`/`REMOVE` + 1–2 sentences each. Dense, un-skippable audit trail. |
| `climinator_v5` | climinator | experimental | hierarchical: commit to one of 5 L2 tiers first, then shortlist only within-tier. Trades tier-slip errors for error-propagation risk; **lost the v4-vs-v5 bake-off** (archived). |

The lineage in one line: `v1` (slim) → `v2` (paper-faithful examples) →
`v3` (SCAN/VERIFY) → `v4` (12-label CONSIDER, **chosen**) → `v5`
(tier-first, rejected). Each variant's source comment in `prompts.py`
records exactly what changed and why.

### Output contract (all variants)

Every prompt asks for `<think>…</think>` reasoning followed by a single
fenced YAML block:

```yaml
assessment: <LABEL>
```

The `<think>` chain shape is `CONTEXT → SUBCLAIMS → EVIDENCE → SHORTLIST →
DECISION` (v4/veracityV2). The `EVIDENCE` step requires **clickable
markdown citations** `[source](url)` when retrieval evidence is present,
and falls back to naming authoritative sources (IPCC / NASA / NOAA) when
it isn't — this is the hook that makes the RAG and non-RAG prompts the
same prompt.

## Climinator credibility hierarchy

The 12 Climate Feedback labels roll up through Leippold 2024's 4-level
credibility hierarchy (Fig. 3):

```
L1 (12) ── climinator_rollup ──> L2 (5) ──> L3 (3) ──> L4 (2)
CORRECT/ACCURATE              VERY HIGH CRED   HIGH CRED    CREDIBLE
…                             …                MODERATE     NOT CREDIBLE
FLAWED REASONING/INCORRECT    VERY LOW CRED     LOW CRED    NOT CREDIBLE
```

`CLIMINATOR_LEVEL_SETS[level]` gives the label set at each level;
`climinator_rollup(label, level)` maps an L1 label up. `generate_report.py`
scores all four levels so you can watch the metrics improve as the taxonomy
is coarsened (see [reports.md](reports.md)).

## Parsing

`extract_raw_label(response)` returns the canonical label or `None`,
handling three response shapes:

1. bare JSON `{"assessment": "FALSE"}` — structured-output backends (Exa,
   OpenAI strict JSON);
2. a fenced ` ```yaml ``` ` block with an `assessment:` key — the default
   chat-model path (the **last** such block wins);
3. a plain `assessment:` line anywhere — fallback.

The regex tolerates bold, quotes, and angle-bracket placeholders
(`assessment: <FALSE>`, which Opus emits because the template shows
`<label_code>`). Results are upper-cased with `_`/`-` collapsed to spaces.
No taxonomy validation here — the caller checks against `LABEL_SETS[variant]`
and flags out-of-vocabulary predictions as parse failures.

`normalise_gold(raw)` applies the same normalisation to a gold-column
string so `Mostly_Accurate` matches `MOSTLY ACCURATE`.

## Structured output (Exa)

Exa's `/answer` endpoint ignores narrative system prompts, so it's
constrained with a JSON schema instead. `VeracityAssessment` /
`ClimateFeedbackAssessment` are pydantic models whose `Literal` enum is the
source of truth (same labels as `LABEL_SETS`); `output_schema(variant)`
returns the JSON schema. `justification` is declared before `assessment` so
the model reasons before committing (property order = generation order for
strict-JSON endpoints).
