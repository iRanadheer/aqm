# Prompt dump · itemId `leippold_001`

- **Model:** `openai/gpt-4o-mini` (backend: `openrouter`)
- **Prompt variant:** `climinator`
- **Evidence source:** Exa Answer API
- **Gold `true_cfb_label`:** `Incorrect`
- **Gold `true_veracity`:** `FALSE`

## Claim

> Magnetic poles reversals involve the Earth flipping vertically and momentarily stopping its rotation, causing cataclysmic events during 6 days.

**Reported source:** Joe Rogan & Jimmy Corsetti, TikTok&Spotify

---

## System prompt (codebook + decision rules)

```text
You are an expert annotator classifying climate-related claims for factual accuracy. You will perform a single-level classification: assign the single most accurate label from the codebook to the claim. A claim may be a single assertion or a compound of multiple assertions joined by semicolons, quotes, or sentence breaks — in either case the output is one label that summarises the row.

### ASSESSMENT CODEBOOK:
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
**Definitions**
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

```

---

## Retrieved evidence

```text
Exa search summary:
The claim is false. Magnetic pole reversals do not involve the Earth flipping vertically, stopping its rotation, or causing 6-day cataclysmic events [1][2][3]. 

Geologic evidence confirms that reversals are gradual, taking hundreds to thousands of years to complete [4][3][5][6]. They occur periodically throughout Earth's history, and there is no evidence linking them to mass extinctions, climate change, or global disasters [1][2][4][7][6]. During a reversal, the magnetic field weakens and becomes complex rather than vanishing, leaving the atmosphere to continue protecting the planet from solar radiation [4][3][6].

Cited URLs (Exa retrieved these as the supporting sources):
- https://www.usgs.gov/faqs/do-any-mass-extinctions-correlate-magnetic-reversals?qt-news_science_products=0
- https://api.politifact.com/factchecks/2023/feb/15/instagram-posts/do-magnetic-pole-shifts-cause-modern-climate-chang/
- https://science.nasa.gov/earth/climate-change/flip-flop-why-variations-in-earths-magnetic-field-arent-causing-todays-climate-change
- https://www.usgs.gov/faqs/it-true-earths-magnetic-field-occasionally-reverses-its-polarity
- https://www.usgs.gov/faqs/are-we-about-have-a-magnetic-reversal?items_per_page=6&page=1
- https://www.nationalgeographic.com/science/article/earth-magnetic-field-flip-north-south-poles-science
- https://nationalgeographic.com/environment/article/earths-shifting-magnetic-fields-arent-causing-climate-change
- https://www.pnas.org/doi/10.1073/pnas.1722110115
```

---

## Assembled user message (verbatim, what the model sees)

```text
### Claim:
Magnetic poles reversals involve the Earth flipping vertically and momentarily stopping its rotation, causing cataclysmic events during 6 days.

### Source:
Joe Rogan & Jimmy Corsetti, TikTok&Spotify

### Evidence (retrieved from a vetted climate-science knowledge base):
Exa search summary:
The claim is false. Magnetic pole reversals do not involve the Earth flipping vertically, stopping its rotation, or causing 6-day cataclysmic events [1][2][3]. 

Geologic evidence confirms that reversals are gradual, taking hundreds to thousands of years to complete [4][3][5][6]. They occur periodically throughout Earth's history, and there is no evidence linking them to mass extinctions, climate change, or global disasters [1][2][4][7][6]. During a reversal, the magnetic field weakens and becomes complex rather than vanishing, leaving the atmosphere to continue protecting the planet from solar radiation [4][3][6].

Cited URLs (Exa retrieved these as the supporting sources):
- https://www.usgs.gov/faqs/do-any-mass-extinctions-correlate-magnetic-reversals?qt-news_science_products=0
- https://api.politifact.com/factchecks/2023/feb/15/instagram-posts/do-magnetic-pole-shifts-cause-modern-climate-chang/
- https://science.nasa.gov/earth/climate-change/flip-flop-why-variations-in-earths-magnetic-field-arent-causing-todays-climate-change
- https://www.usgs.gov/faqs/it-true-earths-magnetic-field-occasionally-reverses-its-polarity
- https://www.usgs.gov/faqs/are-we-about-have-a-magnetic-reversal?items_per_page=6&page=1
- https://www.nationalgeographic.com/science/article/earth-magnetic-field-flip-north-south-poles-science
- https://nationalgeographic.com/environment/article/earths-shifting-magnetic-fields-arent-causing-climate-change
- https://www.pnas.org/doi/10.1073/pnas.1722110115

Use the evidence above to ground your assessment. Cite chunks by their [id] when relevant. Evidence may be incomplete or off-topic — apply the force-fit guard from the codebook if so.
```
