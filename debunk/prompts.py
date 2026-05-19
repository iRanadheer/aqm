"""Climate-claim fact-checking system prompts.

Structure mirrors `wind/prompts.py`:

  - Two TAXONOMIES (not slim/full as in wind, but two different scales):
      veracityV1 — 4-label taxonomy (TRUE / MISLEADING / FALSE / UNVERIFIABLE)
      climinator — 12-label Climate Feedback / Climinator scheme

  - ONE shared `_instruction_template`, parameterised by:
      {codebook}                — short label tags (model output target)
      {assessment_guidelines}   — full per-label definitions and decision rules
                                  (the substantive content that drives quality)

  - Output: `<think>...</think>` reasoning chain + a YAML block with a
    single `assessment:` key. Parsing is unchanged: `extract_raw_label`
    pulls the last fenced YAML block's assessment value.
"""

from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


# Per-taxonomy label space. Two independent benchmarks — no cross-mapping.
#   veracityV1 scores against the `true_veracity` column (4-class).
#   climinator scores against the `true_cfb_label` column (12-class Climate
#   Feedback scheme).
LABEL_SETS = {
    "veracityV1": ("TRUE", "MISLEADING", "FALSE", "UNVERIFIABLE"),
    "climinator": (
        "CORRECT", "ACCURATE", "MOSTLY CORRECT", "MOSTLY ACCURATE", "CORRECT BUT",
        "IMPRECISE", "LACKS CONTEXT", "UNSUPPORTED", "MISLEADING",
        "INCORRECT", "INACCURATE", "FLAWED REASONING",
    ),
}


# ---------------------------------------------------------------------------
# Climinator hierarchy — Climate Feedback / Climinator 4-level rollup
# (Leippold et al. 2024, Fig. 3). L1=12 → L2=5 → L3=3 → L4=2. L2 and L3 share
# the names "HIGH/LOW CREDIBILITY" so each level is namespaced by its level.
# ---------------------------------------------------------------------------

CLIMINATOR_LEVEL_SETS = {
    1: LABEL_SETS["climinator"],
    2: (
        "VERY HIGH CREDIBILITY", "HIGH CREDIBILITY", "NEUTRAL CREDIBILITY",
        "LOW CREDIBILITY", "VERY LOW CREDIBILITY",
    ),
    3: ("HIGH CREDIBILITY", "MODERATE CREDIBILITY", "LOW CREDIBILITY"),
    4: ("CREDIBLE", "NOT CREDIBLE"),
}

_CLIM_L1_TO_L2 = {
    "CORRECT":          "VERY HIGH CREDIBILITY",
    "ACCURATE":         "VERY HIGH CREDIBILITY",
    "MOSTLY CORRECT":   "HIGH CREDIBILITY",
    "MOSTLY ACCURATE":  "HIGH CREDIBILITY",
    "CORRECT BUT":      "HIGH CREDIBILITY",
    "IMPRECISE":        "NEUTRAL CREDIBILITY",
    "LACKS CONTEXT":    "NEUTRAL CREDIBILITY",
    "UNSUPPORTED":      "LOW CREDIBILITY",
    "MISLEADING":       "LOW CREDIBILITY",
    "FLAWED REASONING": "VERY LOW CREDIBILITY",
    "INACCURATE":       "VERY LOW CREDIBILITY",
    "INCORRECT":        "VERY LOW CREDIBILITY",
}

_CLIM_L2_TO_L3 = {
    "VERY HIGH CREDIBILITY": "HIGH CREDIBILITY",
    "HIGH CREDIBILITY":      "HIGH CREDIBILITY",
    "NEUTRAL CREDIBILITY":   "MODERATE CREDIBILITY",
    "LOW CREDIBILITY":       "LOW CREDIBILITY",
    "VERY LOW CREDIBILITY":  "LOW CREDIBILITY",
}

_CLIM_L3_TO_L4 = {
    "HIGH CREDIBILITY":     "CREDIBLE",
    "MODERATE CREDIBILITY": "NOT CREDIBLE",
    "LOW CREDIBILITY":      "NOT CREDIBLE",
}


def climinator_rollup(label: str, level: int) -> str | None:
    """Roll a canonical L1 climinator label up to the target level (1..4).
    Returns None if the label is not in the L1 vocabulary."""
    if label not in _CLIM_L1_TO_L2:
        return None
    if level == 1:
        return label
    l2 = _CLIM_L1_TO_L2[label]
    if level == 2:
        return l2
    l3 = _CLIM_L2_TO_L3[l2]
    if level == 3:
        return l3
    if level == 4:
        return _CLIM_L3_TO_L4[l3]
    raise ValueError(f"Unknown climinator level: {level}")

# Gold-column to use for scoring each prompt variant.
GOLD_FIELDS = {
    "veracityV1": "true_veracity",
    "climinator": "true_cfb_label",
}


# ---------------------------------------------------------------------------
# Codebooks — short label tags + one-line gist. The full per-label
# definitions and decision rules live in the ASSESSMENT GUIDELINES section
# of the template (mirrors wind's slim codebook + separate guidance).
# ---------------------------------------------------------------------------

veracity_codebook = """
<TRUE> None of the claim's substantive components are factually incorrect or materially misleading.
<MISLEADING> Contains correct (or not-directly-refutable) elements but is likely to lead a reasonable reader to an incorrect understanding — distortion rather than direct contradiction.
<FALSE> One or more substantive factual assertions are contradicted by the best-available evidence.
<UNVERIFIABLE> Truth value cannot be reliably determined at the time of assessment (vague, underspecified, or relies on inaccessible / empirically unresolved information).
""".strip()

climinator_codebook = """
<CORRECT> Aligns perfectly with established scientific consensus; factually accurate; no reasonable doubt.
<ACCURATE> Factually sound but may lack important context or nuance.
<MOSTLY CORRECT> Generally supported by scientific studies but slightly overstates the confidence or evidence.
<MOSTLY ACCURATE> Largely true; minor inaccuracies or missing context do not significantly affect overall validity.
<CORRECT BUT> Accurate but lacks critical caveats that could cause misunderstanding without additional context.
<IMPRECISE> Vague or under-detailed; conveys a general idea but leaves room for multiple interpretations.
<LACKS CONTEXT> Factually correct but omits crucial information that significantly alters its meaning or implications.
<UNSUPPORTED> Lacks support; may rely on speculation or unreliable sources.
<MISLEADING> Some elements may be true, but the claim distorts the facts (oversimplification, misrepresentation of data, selective use of evidence).
<INCORRECT> Demonstrably false; contradicts well-established scientific understanding.
<INACCURATE> Distorted or factually incorrect, often relying on cherry-picked evidence.
<FLAWED REASONING> Based on faulty logic, incorrect assumptions, or unsupported conclusions.
""".strip()


# ---------------------------------------------------------------------------
# Assessment guidelines — full definitions + decision rules. Lifted from
# the production .ts files (verbatim where the substance is well-written)
# and lightly adapted to drop chatbot UX language.
# ---------------------------------------------------------------------------

veracity_assessment_guidelines = """**Definitions**
<TRUE> A claim is labelled TRUE only if none of its substantive components are factually incorrect or materially misleading. Minor imprecision (e.g., rounding or informal phrasing) is acceptable provided it does not alter the substantive meaning or implication of the claim.
<MISLEADING> A claim is labelled MISLEADING if it contains factually correct elements, or elements that are not directly refutable as stated, but is likely to lead a reasonable reader to an incorrect understanding of the evidence or its implications. The defining feature is distortion rather than direct factual contradiction. MISLEADING claims may involve omission of critical context, selective presentation of evidence (e.g., cherry-picking specific time periods or data points), inappropriate generalization from limited cases, exaggeration or understatement of uncertainty, invalid inferences (e.g., inferring causation from correlation), or the presentation of technically correct facts in a way that implies unsupported conclusions.
<FALSE> A claim is labelled FALSE if one or more substantive factual assertions are contradicted by the best available evidence. This includes claims that rest on demonstrably false premises or that substantially misrepresent the magnitude, direction, or causal role of a phenomenon, even if other components are accurate or rhetorically persuasive. The presence of isolated true fragments does not prevent a FALSE label when the claim contains substantive factual contradictions.
<UNVERIFIABLE> A claim is labelled UNVERIFIABLE if its truth value cannot be reliably determined at the time of assessment. This includes claims that are too vague or underspecified to evaluate, claims relying on inaccessible or non-public information, and claims concerning empirically unresolved questions. UNVERIFIABLE does NOT indicate partial truth; it indicates that a reliable veracity judgement is not currently possible.

**Decision rules**
- Multi-assertion claims: evaluate every substantive component. Label TRUE only if none are incorrect or materially misleading. Label FALSE if one or more substantive components are contradicted by best-available evidence, even if other components are accurate. Label MISLEADING when no components are directly contradicted but the overall communicated message materially distorts the interpretation.
- Interpretation in context: assess claims by their ordinary communicative meaning, including what is implied as well as what is explicitly stated. Strategic vagueness, selective phrasing, or sarcasm does not exempt a claim from evaluation. Example: "scientists disagree about climate change" is assessed on its implied meaning (lack of consensus), not on the existence of isolated dissenters. Example: "the climate has always changed" is assessed in light of its implied relevance to current anthropogenic warming.
- Predictions: rely on (i) accuracy of stated premises and (ii) proportionality of certainty to evidence. Predictions on factually incorrect premises → FALSE. Predictions asserting categorical, highly certain, or sweeping outcomes not justified by current evidence → MISLEADING (even though the future outcome cannot yet be observed). Only when a prediction cannot be meaningfully evaluated given available knowledge → UNVERIFIABLE.
""".strip()

climinator_assessment_guidelines = """**Definitions**
<CORRECT> The claim aligns perfectly with the established scientific consensus and available evidence. It is factually accurate and leaves no room for reasonable doubt.
<ACCURATE> While factually sound, the claim might lack important context or nuance. Its description is consistent with data but may omit critical elements that could alter its implications.
<MOSTLY CORRECT> The claim is generally supported by scientific studies but may slightly overstate the confidence or evidence, requiring some clarification.
<MOSTLY ACCURATE> This claim is largely true, though minor inaccuracies or missing context do not significantly impact its overall validity.
<CORRECT BUT> The claim is accurate but lacks critical caveats, which could lead to misunderstanding without additional context.
<IMPRECISE> The claim lacks specific details or uses vague language, making it difficult to assess properly. While conveying a general idea, it leaves room for multiple interpretations.
<LACKS CONTEXT> The claim is factually correct but omits crucial information that significantly alters its meaning or implications.
<UNSUPPORTED> The claim lacks support and may rely on speculation or unreliable sources.
<MISLEADING> Though some elements may be true, the claim distorts the facts, leading to a false or exaggerated impression. Common techniques include oversimplification, misrepresentation of data, or selective use of evidence.
<INCORRECT> The claim is demonstrably false and contradicts well-established scientific understanding.
<INACCURATE> The claim presents distorted or factually incorrect information, often relying on cherry-picked evidence.
<FLAWED REASONING> The claim is based on faulty logic, incorrect assumptions, or unsupported conclusions.

**Decision rules**
- Pick the single most accurate label. The 12 labels form a gradient from CORRECT (perfect) through MOSTLY/CORRECT BUT/LACKS CONTEXT (small issues) to MISLEADING (distortion) to INCORRECT/INACCURATE/FLAWED REASONING (clearly wrong).
- Multi-assertion claims: evaluate every substantive component. If at least one substantive assertion is contradicted by best-available evidence → INCORRECT / INACCURATE / FLAWED REASONING (whichever fits the failure mode). If components are individually defensible but the claim's overall message distorts the interpretation → MISLEADING. If components are correct but framed without necessary caveats → CORRECT BUT / LACKS CONTEXT.
- Interpretation in context: assess claims by their ordinary communicative meaning, including what is implied as well as what is explicitly stated. Strategic vagueness, selective phrasing, or sarcasm does not exempt a claim from evaluation.
- Predictions: rely on accuracy of premises and proportionality of certainty to evidence. Predictions on incorrect premises → INCORRECT / FLAWED REASONING. Predictions over-stating certainty beyond what evidence supports → MISLEADING.
""".strip()


# ---------------------------------------------------------------------------
# Instruction template — shared across taxonomies.
# ---------------------------------------------------------------------------

_instruction_template = """You are an expert annotator classifying climate-related claims for factual accuracy. You will perform a single-level classification: assign the single most accurate label from the codebook to the claim. A claim may be a single assertion or a compound of multiple assertions joined by semicolons, quotes, or sentence breaks — in either case the output is one label that summarises the row.

### ASSESSMENT CODEBOOK:
{codebook}

### INSTRUCTIONS:

1. **Assessment**:
   - Read the claim carefully. If it contains multiple substantive assertions, evaluate each one before rolling up to a single row-level label per the decision rules.
   - Pick the single label from the codebook that best fits.

2. **Endorsement, not description**:
   - Classify what the speaker is asserting or endorsing — not what they are quoting, reporting, or criticising.
   - Meta-commentary about a debate is not endorsement of any position in it.

3. **Force-fit guard**:
   - If the claim is too vague, underspecified, or relies on inaccessible information to evaluate reliably, do not force-fit a concrete label. Use the UNVERIFIABLE / equivalent escape-hatch label from the codebook.
   - Predictions are NOT automatically UNVERIFIABLE — see the decision rules below.

### ASSESSMENT GUIDELINES:
{assessment_guidelines}

### OUTPUT FORMAT:
Reason inside <think> tags using the following chain. Every step is mandatory.

<think>
1. CONTEXT
   One-line summary of the claim: speaker/source (if known), tone, and the substantive thing being asserted.

2. ASSERTIONS
   Restate every substantive assertion in the claim, one per line. For compound claims (semicolons, multiple quotes, multiple sentences), enumerate each.

3. EVIDENCE
   For each assertion, one line summarising the established scientific position (IPCC / NASA / NOAA / peer-reviewed work) — what is established, what is contested, what is unknown.

4. ADJUDICATION
   For each assertion, one line — "[assertion]: ACCURATE | DISTORTED | CONTRADICTED | UNVERIFIABLE — [brief reason]".

5. DECISION
   Apply the decision rules to the per-assertion adjudication. State the single label from the codebook that summarises the row, and a one-line justification.
</think>

```yaml
assessment: <label_code>
```

STRICT RULES:
- All reasoning must be inside <think> tags. Nothing after </think> except the YAML block.
- `assessment` must be exactly one of the codes listed in the codebook, uppercase, spelled as shown (no underscores, no extra words).
- Be concise. Single pass. No second-guessing. Adjudication entries must be one line each.
"""


# ---------------------------------------------------------------------------
# Assembled per-taxonomy prompts
# ---------------------------------------------------------------------------

veracity_v1_system_instruction = _instruction_template.format(
    codebook=veracity_codebook,
    assessment_guidelines=veracity_assessment_guidelines,
)

climinator_system_instruction = _instruction_template.format(
    codebook=climinator_codebook,
    assessment_guidelines=climinator_assessment_guidelines,
)

PROMPT_VARIANTS = {
    "veracityV1": veracity_v1_system_instruction,
    "climinator": climinator_system_instruction,
}


# ---------------------------------------------------------------------------
# Pydantic models for structured output. Exa's `/answer` endpoint ignores
# system-prompt narrative instructions, so we constrain it with a JSON schema
# derived from these models. The label enum is the source of truth — same
# tuple as `LABEL_SETS` above, just lifted into the type system.
# ---------------------------------------------------------------------------


class VeracityAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # Justification first so the model reasons before committing to a label
    # (schema property order is the generation order for structured-output
    # endpoints like Exa / OpenAI strict JSON).
    justification: str = Field(
        ...,
        description="One-sentence justification grounded in retrieved evidence.",
    )
    assessment: Literal["TRUE", "MISLEADING", "FALSE", "UNVERIFIABLE"]


class ClimateFeedbackAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    justification: str = Field(
        ...,
        description="One-sentence justification grounded in retrieved evidence.",
    )
    assessment: Literal[
        "CORRECT", "ACCURATE", "MOSTLY CORRECT", "MOSTLY ACCURATE", "CORRECT BUT",
        "IMPRECISE", "LACKS CONTEXT", "UNSUPPORTED", "MISLEADING",
        "INCORRECT", "INACCURATE", "FLAWED REASONING",
    ]


ASSESSMENT_MODELS: dict[str, type[BaseModel]] = {
    "veracityV1": VeracityAssessment,
    "climinator": ClimateFeedbackAssessment,
}


def output_schema(variant: str) -> dict:
    """JSON schema for the assessment model, suitable for Exa's `output_schema`
    extra_body parameter (and any other endpoint that accepts a raw schema)."""
    return ASSESSMENT_MODELS[variant].model_json_schema()


# ---------------------------------------------------------------------------
# Parser — find the LAST fenced ```yaml``` block, pull the first
# `assessment:` line, normalise to the canonical raw label.
# ---------------------------------------------------------------------------

_YAML_FENCE_RE = re.compile(r"```(?:yaml|yml)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_ASSESS_RE = re.compile(
    # Skip any combination of bold (**), quotes, whitespace, and angle-bracket
    # placeholders before the label. Claude Opus emits `assessment: <FALSE>`
    # because the prompt template shows `assessment: <label_code>` — other
    # models drop the brackets, Opus keeps them.
    r"assessment\s*:[*'\"<\s]*([A-Za-z][A-Za-z _\-]*)",
    re.IGNORECASE,
)


def extract_raw_label(response: str) -> str | None:
    """Return the canonical raw label from the response, or None.

    Handles three shapes:
      1. Bare JSON object `{"assessment": "FALSE"}` — structured-output
         providers (Exa, OpenAI strict JSON mode).
      2. A fenced ```yaml``` block with an `assessment:` key — our default
         prompt template for chat models.
      3. A plain `assessment:` line anywhere in the response — fallback.

    Result is uppercase with `_`/`-` collapsed to spaces. No taxonomy
    validation here — the caller checks against `LABEL_SETS[variant]` if it
    wants to flag out-of-vocabulary predictions.
    """
    text = response or ""
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            obj = json.loads(stripped)
            if isinstance(obj, dict) and "assessment" in obj:
                return str(obj["assessment"]).strip().upper().replace("_", " ").replace("-", " ")
        except json.JSONDecodeError:
            pass
    fences = _YAML_FENCE_RE.findall(text)
    haystack = fences[-1] if fences else text
    m = _ASSESS_RE.search(haystack)
    if not m:
        return None
    return m.group(1).strip().upper().replace("_", " ").replace("-", " ")


def normalise_gold(raw: str) -> str:
    """Normalise a gold-column string (e.g. `Mostly_Accurate`, `Flawed_Reasoning`)
    to the canonical raw-label form used by the codebooks and the parser."""
    return str(raw or "").strip().upper().replace("_", " ").replace("-", " ")
