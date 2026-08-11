"""FLICC fallacy detection: three prompts and a parser.

Single-label classification of climate misinformation into the twelve FLICC
reasoning fallacies of Zanartu et al. 2024 (doi.org/10.1038/s41598-024-76139-w).

  bare                    Their own prompt — twelve label names, nothing else.
  deconstruct-published   Scaffold + Table 1's codebook, verbatim.
  deconstruct-revised     Scaffold + a codebook derived from the training split.

All three are zero-shot. The two deconstruct arms are identical apart from the
codebook, so the comparison between them isolates the definitions and nothing
else. None contains example phrases, keywords, or content cues; the prompts say
how to analyse an argument and what the labels mean, never which label to
prefer.
"""

from __future__ import annotations

import json
import re

FALLACY_LABELS: tuple[str, ...] = (
    "ad hominem",
    "anecdote",
    "cherry picking",
    "conspiracy theory",
    "fake experts",
    "false choice",
    "false equivalence",
    "impossible expectations",
    "misrepresentation",
    "oversimplification",
    "single cause",
    "slothful induction",
)

# ---------------------------------------------------------------------------
# Codebooks
# ---------------------------------------------------------------------------

# Zanartu et al. Table 1, verbatim, transcribed from the rendered page (PDF text
# extraction mis-orders the columns). Cherry picking has NO argument structure:
# Table 1 prints the conspiracy-theory structure in that cell.
CODEBOOK_PUBLISHED = """
<ad hominem> Attacking a person/group instead of addressing their arguments
    Argument structure: A has a negative trait. Therefore, A is not credible
<anecdote> Using personal experience or isolated examples instead of sound arguments or compelling evidence
    Argument structure: Y occurred once with X. Therefore, Y will occur every time with X
<cherry picking> Selecting data that appear to confirm one position while ignoring other data that contradicts that position
<conspiracy theory> Proposing that a secret plan exists to implement a nefarious scheme such as hiding a truth
    Argument structure: A is true. B is why the truth cannot be proven. Therefore, A is true
<fake experts> Presenting an unqualified person or institution as a source of credible information
    Argument structure: P has expertise in a non-climate topic. Therefore, P is an expert on climate
<false choice> Presenting two options as the only possibilities, when other possibilities exist
    Argument structure: P or Q. P. Therefore, not Q
<false equivalence> Incorrectly claiming that two things are equivalent, despite the fact that there are notable differences between them
    Argument structure: A and B both share characteristic C. Therefore, A and B share some other characteristic D
<impossible expectations> Demanding unrealistic standards of certainty before acting on the science
    Argument structure: There is not enough data or research about X to understand X properly
<misrepresentation> Misrepresenting a situation or an opponent's position in such a way as to distort understanding
<oversimplification> Simplifying a situation in such a way as to distort understanding, leading to erroneous conclusions
<single cause> Assuming a single cause or reason when there might be multiple causes or reasons
    Argument structure: X caused Y; therefore, X was the only cause of Y
<slothful induction> Ignoring relevant evidence when coming to a conclusion
""".strip()

# Ours. Written by reading the 1,796 TRAINING rows class by class and stating
# the pattern the annotations follow — never val, never test. Table 1 is a
# summary table and does not fully specify the annotation practice; three
# mismatches are large enough to name:
#
#   fake experts   Table 1 requires an UNQUALIFIED source, but the training
#                  labels include appeals to genuinely credentialled dissenters
#                  and to petition signatory counts. The operative idea is
#                  appeal to dissenting-expert authority, not lack of standing.
#   impossible     Table 1 describes only demands for more research. The
#   expectations   dominant training pattern is instead dismissing a measure
#                  because its individual effect is small.
#   single cause   Table 1's structure is "X caused Y; therefore X was the only
#                  cause". Many training texts assert no cause at all — a bare
#                  observation of past natural variation carries the implication.
#
# These state argument patterns at category level. No example phrases, no
# keyword lists, no content cues.
CODEBOOK_REVISED = """
<ad hominem> Rejecting a claim by attacking the people or institutions that make it — their character, motives, competence, hypocrisy, or bias — rather than answering the claim. The target is often a diffuse group rather than a named individual.
    Distinguish from conspiracy theory: that alleges deliberate, coordinated deception. Accusations of bias, alarmism, or incompetence alone are ad hominem.
<anecdote> Setting a single observation — one place, one season, one event — against a claim about long-term or large-scale trends, as though the instance refuted the trend.
    Distinguish from cherry picking: anecdote points to an individual occurrence; cherry picking points to selected data, trends, or studies.
<cherry picking> Citing real but selectively chosen evidence — a particular period, region, dataset, or study — that appears to support the position while the wider body of evidence does not.
    Distinguish from slothful induction: cherry picking argues from evidence it names; slothful induction dismisses the evidence without naming any.
<conspiracy theory> Alleging that researchers, institutions, or governments are deliberately fabricating, manipulating, or suppressing evidence, or that the issue itself is invented to serve a hidden agenda.
    Requires coordinated intent to deceive, not merely bias or shared interest.
<fake experts> Appealing to the authority of sources who dissent from the scientific consensus as though disagreement itself established the claim. Covers named dissenters, petitions, and counts of signatories, and applies whether or not those cited hold relevant credentials.
<false choice> Presenting two things as mutually exclusive when both may hold — two causes, two policies, or two priorities. Includes arguing that because one thing influences another the reverse cannot also be true, and arguing that effort spent on one problem must come at the expense of another. The exclusivity is usually implied rather than stated.
    Distinguish from single cause: false choice denies that two things can both be true; single cause assigns responsibility to one factor instead of another.
<false equivalence> Treating two things as comparable because they share a surface feature, when they differ in the respect the argument depends on. Frequently equates change occurring now with change that occurred under different conditions in the past.
<impossible expectations> Setting a standard that cannot be met and treating the failure to meet it as grounds for rejection or inaction. Two forms: requiring a degree of certainty or predictive precision that science cannot supply, and dismissing a measure because its individual contribution is small or partial.
<misrepresentation> Asserting something false — about the physical science, the evidence, or the position held by those being argued against — in a way that distorts what is at issue.
    Distinguish from oversimplification: here something stated is false; there everything stated is true but incomplete.
<oversimplification> Presenting a true but partial fact as though it settled the question, omitting the context that would change its significance.
    Distinguish from misrepresentation: every statement here is accurate; the distortion comes from what is left out.
<single cause> Attributing climate change to a single factor, typically a natural one, to the exclusion of others and of human activity. The causal claim is often unstated: a bare observation that the climate has varied before carries the implication on its own.
<slothful induction> Denying or dismissing a conclusion the evidence supports — asserting it is unproven, uncertain, or non-existent — without engaging the evidence.
    Distinguish from impossible expectations: this denies what is established; impossible expectations demands more than can be given.
""".strip()

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

BARE = """Please classify a piece of text into the following categories of \
logical fallacies: ad hominem, anecdote, cherry picking, conspiracy theory, \
fake experts, false choice, false equivalence, impossible expectations, \
misrepresentation, oversimplification, single cause, slothful induction.

### OUTPUT FORMAT:
Output only a YAML block naming the fallacy. No reasoning, no commentary, no other text.

```yaml
fallacy: <label>
```

STRICT RULES:
- `fallacy` must be exactly one of the twelve labels, lowercase, spelled as in the codebook.
- Nothing outside the YAML block.
"""

_DECONSTRUCT = """You are an expert in critical thinking and climate \
communication. Your task is to identify the reasoning fallacy in the given text \
using the provided codebook. This is a single-label classification task.

### CODEBOOK:
{codebook}

### INSTRUCTIONS:

1. **Single Label**:
   - Exactly one label. Climate misinformation often commits several fallacies at once; choose the dominant one — the fallacy the argument depends on to reach its conclusion.
   - Never answer "none", "other", or a label outside the codebook.

2. **Judge the Argument, Not the Subject Matter**:
   - Reconstruct the argument in the format below before naming any label. The flaw is usually in an assumption the argument leaves unstated, not in anything the text says outright.
   - Some fallacies are not visible in the form of the argument at all: the premises are well made, and the flaw is that one of them is false or incomplete against what is known. Check for that too.

3. **Decisiveness**:
   - Where two labels still survive the scan, commit to one and move on. Do not deliberate.

### OUTPUT FORMAT:
Reason inside <think> tags following this structure, then output YAML:

<think>
1. CONCLUSION: What the text wants you to believe or do. One line.
2. PREMISES: The stated premises, one line each: "P1: ...", "P2: ...". Quote or close-paraphrase; do not add reasoning of your own. One premise is fine.
3. HIDDEN PREMISE: The unstated assumption the argument needs to reach its conclusion. State it as a proposition, one line. If it needs no such assumption, write "HP: none".
4. SOURCE OF ERROR: Which single element is faulty — one of the premises, or HP? Name it, and say in one line what is wrong with it: either that it does not support the conclusion, or that it is false or incomplete against the evidence.
5. SCAN: Go through all twelve labels in codebook order. One line each, in that order: "[label]: KEEP/REMOVE — [max 8 words why]." Every label must appear exactly once. Judge each against the SOURCE OF ERROR, not against the topic of the text.
6. FINAL: one label.
</think>
```yaml
fallacy: <label>
```

STRICT RULES:
- All reasoning must be inside <think> tags. Nothing after </think> except the YAML block.
- `fallacy` must be exactly one of the twelve labels, lowercase, spelled as in the codebook.
- Be concise. SCAN entries must be one line each, twelve in total.
"""

DECONSTRUCT_PUBLISHED = _DECONSTRUCT.format(codebook=CODEBOOK_PUBLISHED)
DECONSTRUCT_REVISED = _DECONSTRUCT.format(codebook=CODEBOOK_REVISED)

PROMPT_VARIANTS: dict[str, str] = {
    "bare": BARE,
    "deconstruct-published": DECONSTRUCT_PUBLISHED,
    "deconstruct-revised": DECONSTRUCT_REVISED,
}
VARIANT_NAMES = tuple(PROMPT_VARIANTS)
USER_TEMPLATE = "### Text:\n{text}"

# ---------------------------------------------------------------------------
# RECoT — teacher-side only
# ---------------------------------------------------------------------------
# Used by teacher.py to generate training traces. The teacher is GIVEN the gold
# label and asked to produce reasoning that arrives at it, so its job is
# justification rather than classification. The system prompt is the ordinary
# `deconstruct-revised` one, unchanged — the student must learn the same prompt
# it will see at inference, so only the user turn differs.
#
# Never used at inference. Following cards/prompts.py::recot_trigger.

TEACHER_TEMPLATE = """### Text:
{text}

### True Label:
{label}

{trigger}"""

RECOT_TRIGGER = """The true label above is the correct classification. Generate \
expert-level reasoning that arrives at exactly that label.

Rules:
- Write as a confident expert who has never seen the true label. Do not mention it, refer to being given it, or hint that the answer was known in advance.
- Follow the OUTPUT FORMAT exactly: the seven-step <think> block, then the YAML.
- SCAN must still cover all twelve labels in codebook order. Remove the others for reasons that stand on their own; keep the true label.
- If the true label looks arguable, argue for it anyway — state the reading of the text on which it is the dominant fallacy. Do not hedge, and do not note the disagreement.
- Be decisive. Single pass, no second-guessing, no repetition.
- FINAL and the YAML must both be exactly the true label."""


def get_prompt(variant: str) -> str:
    return PROMPT_VARIANTS[variant]


def _check() -> None:
    """Codebooks are hand-written; guard against a typo'd or reordered label."""
    for name, book in (("published", CODEBOOK_PUBLISHED), ("revised", CODEBOOK_REVISED)):
        found = tuple(re.findall(r"^<([^>]+)>", book, re.MULTILINE))
        if found != FALLACY_LABELS:
            raise ValueError(f"{name} codebook drifted from FALLACY_LABELS: {found}")
    missing = [lab for lab in FALLACY_LABELS if lab not in BARE]
    if missing:
        raise ValueError(f"bare prompt is missing labels: {missing}")


_check()

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_YAML_FENCE_RE = re.compile(r"```(?:yaml|yml)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_FALLACY_RE = re.compile(r"^\s*fallacy\s*:\s*[\"']?([^\"'\n#]+)", re.IGNORECASE | re.MULTILINE)
# A SCAN line names the label it is REJECTING, so it must never be read as an answer.
_VERDICT_RE = re.compile(r":\s*(KEEP|REMOVE)\b", re.IGNORECASE)


def extract_raw_label(response: str) -> str | None:
    """Return the raw predicted label, or None if nothing parseable is present.

    Bare JSON, then the last fenced block, then a `fallacy:` line — enough for
    any model that obeys the output contract, which every API model here does.
    Small models do not: Qwen3.5-0.8B abandoned the contract on 406 of 457 rows
    while still naming a label, so the strict parser was scoring format failure
    as classification failure. Three fallbacks recover a committed answer, in
    descending order of certainty:

      1. angle-bracketed value — `fallacy: <ad hominem>`. Ours to fix: the
         template says `fallacy: <label>`, so copying the brackets is obedience.
      2. a fenced block whose entire content is exactly one vocabulary label.
      3. a first line that is exactly one vocabulary label, excluding SCAN
         verdict lines, where the named label is the one being rejected. The
         least airtight of the three.

    These are deterministic rules applied to every run alike — they changed 0
    predictions on all frontier runs, so they can only help a weak one. Not the
    paper's protocol: Zanartu et al. hand-picked "the most likely label" for 44
    of 256 GPT-4 outputs. Whatever the rules cannot reach counts as wrong.
    """
    text = response or ""
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            obj = json.loads(stripped)
            if isinstance(obj, dict) and "fallacy" in obj:
                return _norm(str(obj["fallacy"]))
        except json.JSONDecodeError:
            pass
    fences = _YAML_FENCE_RE.findall(text)
    haystack = fences[-1] if fences else text
    m = _FALLACY_RE.search(haystack)
    if m:
        return _norm(m.group(1))

    for fence in fences:
        if _norm(fence) in FALLACY_LABELS:
            return _norm(fence)

    first = stripped.split("\n")[0] if stripped else ""
    if not _VERDICT_RE.search(first) and _norm(first) in FALLACY_LABELS:
        return _norm(first)
    return None


def _norm(s: str) -> str:
    """Lowercase, collapse separators, and drop the codebook's angle brackets."""
    s = s.strip().lower().replace("_", " ").replace("-", " ")
    return " ".join(s.replace("<", " ").replace(">", " ").split())


def is_in_vocabulary(label: str | None) -> bool:
    return label in FALLACY_LABELS


if __name__ == "__main__":
    import sys
    print(get_prompt(sys.argv[1] if len(sys.argv) > 1 else "deconstruct-revised"))
