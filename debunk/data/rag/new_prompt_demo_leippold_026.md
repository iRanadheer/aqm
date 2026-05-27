# RAG prompt rewrite · demo on `leippold_026`

This document demonstrates the new RAG user-template on a single Climate Feedback claim that **both RAG models got wrong under the previous prompt*, where at least one model offline (no RAG) had gotten it right.* The claim is the Amazon-rainforest one the paper itself devotes a full debate trace to in SI §3 — see Leippold 2025 npj Climate Action.

## Claim

> More than 75 percent of the rainforest is losing resilience; More than half of the rainforest could be converted into savanna in a matter of decades

- **Reported source:** Sarah Kaplan, Washington Post, 2022-03-15 00:00:00
- **Gold `true_cfb_label`:** `Mostly_Accurate`
- **Paper CLIM verdict (`true_climinator`):** `correct_but`

## Outcome summary

| Model | OLD-prompt RAG | OLD-prompt offline | **NEW-prompt RAG** | Gold |
|---|---|---|---|---|
| GPT-4o-mini | `CORRECT` ❌ | `CORRECT` ❌ | **`MOSTLY ACCURATE`** ✅ | `Mostly_Accurate` |
| DeepSeek V4 Flash (terminus) | `MISLEADING` ❌ | `MOSTLY ACCURATE` ✅ | **`MOSTLY CORRECT`** ✅ at L2+ | `Mostly_Accurate` |

*OLD prompt = `infer.py` `USER_TEMPLATE_RAG` with the 5 IRONCLAD GROUNDING RULES (corpus-only, force-fit to LACKS CONTEXT / IMPRECISE on retrieval miss). NEW prompt = the rewritten template with `EVIDENCE USE: relied-on | partial | ignored` declaration and explicit permission to fall back to internal knowledge when retrieval is poor.*

## EVIDENCE USE declarations (new prompt only)

- **GPT-4o-mini:** `EVIDENCE USE: relied-on — The retrieved evidence directly supports the claim regarding the loss of resilience in the Amazon rainforest and the potential for conversion to savanna.`
- **DeepSeek:**    `EVIDENCE USE: partial — The retrieved evidence directly supports the first subclaim and provides context for the second, but the specific "more than half" conversion figure is not explicitly stated in the provided chunks.`

Two different declarations from two different models on the same row and same retrieved chunks. `relied-on` vs `partial` is exactly the kind of post-hoc-greppable signal we want for diagnosing where RAG helps and where it doesn't.

## Run config

- **GPT-4o-mini:** `openai/gpt-4o-mini` (backend `openai`)
- **DeepSeek:** `deepseek/deepseek-v3.1-terminus` (backend `openrouter`)
- **Prompt variant:** `climinator_v4`
- **RAG index:** `data/rag/perplexity-ai-pplx-embed-context-v1-0-6b` (Perplexity pplx-embed-context-v1-0-6b dense + BM25, BGE cross-encoder rerank on)
- **Top-k retrieved chunks:** 5

---

## What the model saw (system + retrieved evidence + user message)

Both models received identical assembled prompts on this row (same RAG index → same top-5 chunks, same system prompt, same user template). The block below is a verbatim dump of what was sent.

- **Model:** `gpt-4o-mini` (backend: `openai`)
- **Prompt variant:** `climinator_v4`
- **Evidence source:** Local hybrid RAG (Qwen3 dense + BM25 + BGE rerank)
- **Gold `true_cfb_label`:** `Mostly_Accurate`
- **Gold `true_veracity`:** `TRUE`

## Claim

> More than 75 percent of the rainforest is losing resilience; More than half of the rainforest could be converted into savanna in a matter of decades

**Reported source:** Sarah Kaplan, Washington Post

---

## System prompt (codebook + decision rules)

```text
You are a fact-checking AI assistant specializing in climate-related scientific claims. Your task is to analyze a claim and categorize its accuracy using the following codebook, drawn from established fact-checking guidelines for evaluating scientific claims (Climate Feedback framework).

The codebook is organized into five credibility tiers (Very High → Very Low). Each tier contains one or more fine-grained labels with characteristics and worked examples to guide your analysis.

### ASSESSMENT CODEBOOK

**Very High Credibility**
Use this level for claims that are fully aligned with well-established scientific consensus and leave no room for reasonable doubt.

- **CORRECT**: The claim aligns perfectly with established scientific consensus and evidence, presenting no factual issues.
  *Example:* "Babies under six months should not drink water as it can result in health risks."
  *Explanation:* Babies under six months receive hydration through breast milk or formula, so drinking water can lead to health complications such as electrolyte and nutritional imbalances.

- **ACCURATE**: The claim is factually sound but lacks important nuances, context, or caveats that would significantly alter its implications.
  *Example:* "Eating a diet rich in whole foods like vegetables and grains is beneficial for long-term health."
  *Explanation:* This is accurate but lacks additional context on portion control and nutrient balance, which would provide a fuller understanding of healthy eating.

**High Credibility**
Use this level for claims that are mostly accurate but may contain minor inaccuracies or overstatements.

- **MOSTLY CORRECT**: The claim aligns generally with scientific studies but may overstate confidence or predictions slightly.
  *Example:* "Prioritizing plant-based foods reduces greenhouse gas emissions."
  *Explanation:* A predominantly plant-based diet reduces emissions but is most impactful if adopted globally, which the statement overstates.

- **MOSTLY ACCURATE**: The claim is mostly true, with minor inaccuracies that don't significantly alter the claim's validity.
  *Example:* "Prioritizing plant-based foods is healthier and reduces greenhouse gas emissions."
  *Explanation:* Research shows plant-based diets are beneficial, but reductions in emissions depend on widespread adoption.

- **CORRECT BUT**: The claim is largely accurate but requires additional details to avoid misinterpretation.
  *Example:* "Climate change will destroy all ecosystems."
  *Explanation:* Climate change threatens many ecosystems, but some may adapt or be resilient, which the claim overstates.

**Neutral Credibility**
Use this level for claims that may be factually correct but are vague or lack important context.

- **IMPRECISE**: The claim lacks specific details or clear definitions, leaving it open to multiple interpretations.
  *Example:* "Artificial intelligence is going to take over the world."
  *Explanation:* This vague claim lacks definitions and a timeframe, making it impossible to evaluate accurately.

- **LACKS CONTEXT**: The claim is potentially factually correct but omits critical information that could change its meaning.
  *Example:* "Wind turbine disposal has a large environmental footprint."
  *Explanation:* While true, studies show that waste from turbine blades is smaller than waste from coal energy sources, giving needed context.

**Low Credibility**
Use this level for claims that lack sufficient evidence or rely on misleading tactics.

- **UNSUPPORTED**: The claim relies on insufficient evidence, anecdotes, or speculation.
  *Example:* "Breast self-examinations are adequate mammogram substitutes."
  *Explanation:* Self-exams and thermograms are not FDA-approved mammogram replacements and lack adequate support.

- **MISLEADING**: The claim uses technically correct information to create a distorted impression.
  *Example:* "The Atlantic is cooling, and scientists don't know why."
  *Explanation:* This misleads by emphasizing a short-term observation, ignoring the long-term trends of climate change.

**Very Low Credibility**
Use this level for claims that contain fundamental errors, flawed logic, or unsupported theories.

- **FLAWED REASONING**: The claim is based on faulty logic or incorrect assumptions.
  *Example:* "How much of an impact do humans have on climate? It's likely a natural cycle."
  *Explanation:* Studies quantify human influence on modern warming, contradicting the idea of a natural cycle as the sole cause.

- **INCORRECT**: The claim is demonstrably false and directly contradicts scientific consensus.
  *Example:* "CO2 increases are mainly due to natural causes, not humans."
  *Explanation:* Data show fossil fuel emissions as the primary cause of modern CO2 levels, disproving this claim.

- **INACCURATE**: The claim contains inaccuracies that distort the scientific consensus or selectively present data.
  *Example:* "Greenland's ice cores show no significant warming, disproving climate change."
  *Explanation:* Greenland's warming rates are rising, contradicting this claim, while global studies also confirm warming.

### CRUCIAL INSTRUCTIONS

- **Evidence is essential**: ground your verdict in the evidence available to you (retrieved sources, scientific knowledge). Where evidence chunks are provided, cite them.
- **Endorsement, not description**: classify what the speaker is asserting or endorsing — not what they are quoting, reporting, or criticising. Meta-commentary about a debate is not endorsement of any position in it.
- **Stay within the codebook**: the output label must be exactly one of the 12 labels above (CORRECT, ACCURATE, MOSTLY CORRECT, MOSTLY ACCURATE, CORRECT BUT, IMPRECISE, LACKS CONTEXT, UNSUPPORTED, MISLEADING, FLAWED REASONING, INCORRECT, INACCURATE).
- **Force-fit guard**: if the claim is too vague, underspecified, or relies on inaccessible information to evaluate reliably, prefer IMPRECISE (vague language) or LACKS CONTEXT (missing critical information). Don't invent a strong verdict on weak evidence.
- **No speaker priming**: do NOT identify or comment on the speaker. The verdict must not depend on speaker identity.

### OUTPUT FORMAT

Reason inside <think> tags using the structure below. The flow is:
CONTEXT → SUBCLAIMS → EVIDENCE → SHORTLIST (wide enumeration) → DEBATE
(pairwise argument among shortlisted candidates) → DECISION. Every step is mandatory.

<think>
1. CONTEXT
   2–3 sentences summarising the claim's tone and substantive assertions. Do NOT identify the speaker.

2. SUBCLAIMS
   Restate every substantive assertion, one per line. For single-assertion claims, one line is fine.

3. EVIDENCE
   For each subclaim, 2–3 sentences summarising the established scientific position. **Cite sources as clickable markdown links `[<source-name>](<URL>)` using the URLs visible in the evidence block of the user message.** Do not use numbered references like `[1]`, `[3]`. If no retrieval evidence is provided, name the authoritative source (IPCC / NASA / NOAA / peer-reviewed work) and link to it where you can.

4. SHORTLIST (mandatory — one explicit line per label, all 12 labels, no skipping)
   Walk every label and mark each as either SHORTLIST (a plausible candidate
   for this claim, worth debating) or REMOVE (clearly does not fit).
   - Most labels will be REMOVE. Aim for 2–4 SHORTLIST entries — the labels
     that are genuinely plausible and ought to be compared.
   - For each line, write 1–2 sentences of reasoning. Telegraphic tags are
     not enough; you must engage with the codebook definition.

   - CORRECT:          SHORTLIST/REMOVE — …
   - ACCURATE:         SHORTLIST/REMOVE — …
   - MOSTLY CORRECT:   SHORTLIST/REMOVE — …
   - MOSTLY ACCURATE:  SHORTLIST/REMOVE — …
   - CORRECT BUT:      SHORTLIST/REMOVE — …
   - IMPRECISE:        SHORTLIST/REMOVE — …
   - LACKS CONTEXT:    SHORTLIST/REMOVE — …
   - UNSUPPORTED:      SHORTLIST/REMOVE — …
   - MISLEADING:       SHORTLIST/REMOVE — …
   - FLAWED REASONING: SHORTLIST/REMOVE — …
   - INCORRECT:        SHORTLIST/REMOVE — …
   - INACCURATE:       SHORTLIST/REMOVE — …

   STRICT RULES for SHORTLIST:
   - You MUST emit all 12 lines, in the order shown.
   - Each line MUST start with the label, colon, SHORTLIST/REMOVE, "—", then 1–2 sentences of reasoning.
   - No placeholder text ("...", "TBD", "see above"). Each line is a real decision.

5. DEBATE — pick the winner from the SHORTLIST, argue against every other shortlisted label.

   First, restate the shortlisted candidates:
   - SHORTLIST contains: [LABEL_1], [LABEL_2], [LABEL_3], …

   Then debate them pairwise against the candidate you ultimately favour.
   For EACH shortlisted label OTHER THAN your final pick, write a paragraph
   (2–4 sentences) of the form:

     **WHY NOT [LABEL_X]?**
     [Argument: this label is close, but my pick is a better fit because…
      cite the specific codebook *Example:* / *Explanation:* that makes the
      distinction, and tie it to the evidence shown in step 3.]

   Then justify your pick:

     **WHY [WINNING_LABEL]?**
     [3–4 sentences anchoring the pick to the codebook example most similar
      to the claim, and to the retrieved evidence.]

6. DECISION
   State the final L1 label and a 3–4 sentence verdict summary suitable for a journalist to quote, with at least one clickable markdown citation.
</think>

```yaml
assessment: <label_code>
```

STRICT OUTPUT RULES:
- All reasoning must be inside <think> tags. Nothing after </think> except the YAML block.
- The DEBATE section must address every shortlisted candidate other than the winner — do not silently drop any.
- `assessment` must be exactly one of the 12 codes from the codebook, uppercase, spelled with spaces (e.g. "MOSTLY CORRECT", "FLAWED REASONING").
- Single pass. No second-guessing.

```

---

## Retrieved evidence

```text
[1] carbonbrief_org · Declining ‘resilience’ pushing Amazon rainforest towards tipping point · 2022-03-07
URL: https://www.carbonbrief.org/declining-resilience-pushing-amazon-rainforest-towards-tipping-point/
[Source: carbonbrief_org | Declining ‘resilience’ pushing Amazon rainforest towards tipping point | 2022-03-07]
Three-quarters of the Amazon rainforest has lost “resilience” since 2003 – making it more vulnerable to extreme events such as droughts – according to new research.

Scientists have long warned that climate change and human-driven deforestation could push the Amazon rainforest past a “tipping point”, which would see the dieback of large sections of lush rainforest and a shift to dry savannah. However, modelling studies have been unable to agree on if and when this threshold might be crossed.

The new study uses observational data to explore how the forest’s “resilience” – a measure of how quickly the forest can recover from a drought or extreme weather event – has changed in recent decades.

The findings, published in Nature Climate Change, suggest that more than three-quarters of the Amazon rainforest has already lost resilience over the past two decades. The drier parts of the Amazon – and regions closer to human land use, such as roads and agricultural lands – are the least resilient, the study says.

These findings add further weight to concerns that the Amazon forest is approaching a “critical threshold” or “tipping point”, the authors say. However, in a press briefing, the lead author stressed that we cannot say when a tipping point might happen based on this analysis.

Amazon resilience

As the largest forest in the world, the Amazon is a huge store of carbon and houses at least 10% of the world’s biodiversity. However, humans are having a well-documented impact on the region through deforestation and climate change.

Last year, the deforestation rate reached their highest level in over a decade – with estimates suggesting that more than 10,000 square kilometres of forest have been lost between August 2020 and July 2021. And, as the climate warms, the moist atmosphere in the Amazon is drying out, leaving the forest more vulnerable to droughts and fires.

Both of these factors are reducing the Amazon’s “resilience” – by making the forest slower to recover from extreme events such as droughts.

The Amazon’s loss of resilience has long been a point of concern for scientists. Many modelling studies suggest that deforestation and climate change-driven drying could cause a further loss of resilience over the coming century. However, identifying how much resilience has already been lost can be tricky.

Rather than relying on climate models, this study uses satellite measurements of the Amazon to determine how much resilience the forest has already lost. The authors assess “vegetation optical depth” (VOD)

[2] carbonbrief_org · Declining ‘resilience’ pushing Amazon rainforest towards tipping point · 2022-03-07
URL: https://www.carbonbrief.org/declining-resilience-pushing-amazon-rainforest-towards-tipping-point/
[Source: carbonbrief_org | Declining ‘resilience’ pushing Amazon rainforest towards tipping point | 2022-03-07]
 extreme events such as droughts.

The Amazon’s loss of resilience has long been a point of concern for scientists. Many modelling studies suggest that deforestation and climate change-driven drying could cause a further loss of resilience over the coming century. However, identifying how much resilience has already been lost can be tricky.

Rather than relying on climate models, this study uses satellite measurements of the Amazon to determine how much resilience the forest has already lost. The authors assess “vegetation optical depth” (VOD) – a measure of how much biomass is in the plants, which is closely related to water content.

The authors divide the Amazon rainforest into grid cells, using data from the MODIS satellite to select only green areas of forest that have not already been cleared.

They then assess how VOD evolved in each grid cell over 1991-2016. The map below shows the overall change in VOD, where red shows a decrease in VOD (indicating a loss of biomass) and blue shows an increase in VOD.

Vegetation optical depth of the Amazon over 1991-2016. Source: Boulton et al (2022).

The map shows that overall, VOD decreased over the study period – with the most notable decrease seen in the south-eastern parts of the Amazon basin, along parts of the Amazon river and in some northern areas.

Equilibrium

By assessing incremental changes in VOD over 1991-2016, the authors were able to see how quickly the forest recovers from extreme events such as droughts – and from this determine how its resilience changed.

Dr Chris Boulton – a researcher at the University of Exeter and lead author on the paper – tells Carbon Brief that visualising a ball in a bowl provides a good analogy for the resilience of the Amazon

When the ball is sitting at the bottom of the bowl, he explains, the ball is in its “equilibrium state” – just as the Amazon has its own equilibrium. And the ball is pushed up the sides of the bowl, it will roll back to the centre. Similarly, if the Amazon experiences an extreme event such as a drought that pushes it away from its equilibrium, it will return to its equilibrium.

The faster the ball returns to the bottom of the bowl, the more stable or “resilient” the system is, Boulton explains. The Amazon losing resilience is like the walls of the bowl getting “shallower”, causing the ball to roll back to

[3] carbonbrief_org · Declining ‘resilience’ pushing Amazon rainforest towards tipping point · 2022-03-07
URL: https://www.carbonbrief.org/declining-resilience-pushing-amazon-rainforest-towards-tipping-point/
[Source: carbonbrief_org | Declining ‘resilience’ pushing Amazon rainforest towards tipping point | 2022-03-07]
 show that regions with less rainfall, and which are closer to human land use, such as agricultural lands, generally lose resilience more quickly.

The authors also included a separate piece of analysis showing the distance to roads in Brazil, again showing the areas of forest closer to the roads are likely to lose resilience more quickly.

Dr Chantelle Burton, who is from the UK Met Office and not involved in the paper, tells Carbon Brief that this study is “really important” and uses “robust statistical methods”.

However, Dr David Lapola – a researcher at the University of Campinas in Brazil, who was not involved in the study – adds a word of caution about its use of VOD for assessing resilience. He tells Carbon Brief:

“I think it might be an oversimplification to rely our understanding of the resilience of the forest on the amount of leaves or the greenness of the forest vegetation…We cannot forget that a tree is more than leaves, branches and a trunk. There is a whole, largely unknown world below ground, where roots thrive and where water, nutrients and other relations take place and might influence the resilience of the forest.”

Tipping points

Scientists have long worries that climate change and deforestation could push the Amazon rainforest past a “tipping point”. In a Carbon Brief special week of content in 2020 about tipping points, Prof Peter Cox from the University of Exeter wrote a guest post about Amazon dieback, in which he said:

“The Amazon rainforest sustains its own climate by recycling water to the atmosphere, which maintains rainfall and reduces the length of dry seasons. Deforestation undermines those regulatory mechanisms and may, ultimately, lead to a tipping point.

“This, in combination with the dry season becoming long enough to permit regular natural fires, could see the forest transition to a permanent savannah. This would be characterised by a mixed tree and grassland system with an open canopy that allows the soil to become much hotter and drier, as well as store much less carbon.”

In the press briefing, Boulton told journalists that the study shows we are “approaching a tipping point”. However, he stressed, the authors “can’t say when that’s going to happen” based on this analysis.

While many modelling studies have been conducted on the Amazon rainforest, Dr Chris Jones – a scientist at the UK Met Office who was not involved in the study – says that to the best of his knowledge, the critical slowing down method has not been used on observations of the Amazon

[4] carbonbrief_org · ‘Unprecedented’ stress in up to half of the Amazon may lead to tipping point by 2050 · 2024-02-14
URL: https://www.carbonbrief.org/unprecedented-stress-in-up-to-half-of-the-amazon-may-lead-to-tipping-point-by-2050/
[Source: carbonbrief_org | ‘Unprecedented’ stress in up to half of the Amazon may lead to tipping point by 2050 | 2024-02-14]
As much as half of the Amazon will face several “unprecedented” stressors that could push the forest towards a major tipping point by 2050, new research finds.

The largest rainforest in the world is already under pressure from climate change, deforestation, biodiversity loss and extreme weather.

Researchers analysed data on five key drivers of water stress in the Amazon and looked at how these pressures could lead to “local, regional or even biome-wide forest collapse”.

The new study, published in Nature, finds that by 2050, between 10 and 47% of the Amazon forest will be exposed to “compounding disturbances” that “may trigger unexpected ecosystem transitions”. This could result in large swathes of lush rainforest shifting to dry savannah.

One author of the study tells Carbon Brief that this prospect by mid-century is “very scary”.

The study notes that the complexity of the Amazon “adds uncertainty about future dynamics” and that there are still “opportunities for action”.

The findings highlight the likelihood that “climate change will continue to affect the forest in very unpredictable ways”, a scientist not involved in the study says.

Amazon pressures

The Amazon forest stores a huge amount of carbon and houses at least 10% of the world’s biodiversity. It faces an uncertain future largely due to the effects of deforestation and climate change.

Last year, the Amazon river basin experienced an “exceptional drought” that was 30 times more likely to occur due to climate change, a rapid attribution study found.

Around 20% of the Amazon has already been deforested and a further 6% is “highly degraded”.

According to several studies, the Brazilian section of the Amazon is now an overall net “source” of carbon, rather than a “sink”, due to a number of factors including deforestation.

Scientists have long warned that climate change and human-driven deforestation could push the Amazon rainforest past a “tipping point” – a threshold that, if crossed, would see the “dieback” of large amounts of dense Amazon rainforest and a shift into permanent, dry savannah.

This would be characterised by a mixed tree and grassland system with an open canopy that allows the soil to become much hotter and drier.

Previous studies suggest that the Amazon could be pushed beyond this tipping point if forest loss exceeds 40%. Other research published last October found that recent drying over the Amazon could be the “first warning signal” that the rainforest is approaching a tipping point.

The

[5] climatefactchecks_org · The viral post claims that trees do not help mitigate climate change; however, they are responsible for the global warming · 2023-06-21
URL: https://climatefactchecks.org/the-viral-post-claims-that-trees-do-not-help-mitigate-climate-change-however-they-are-responsible-for-the-global-warming/
[Source: climatefactchecks_org | The viral post claims that trees do not help mitigate climate change; however, they are responsible for the global warming | 2023-06-21]
 parasites, and predators. Earth’s forests continue to be a net carbon dioxide sink despite these perturbations and the slower disintegration process. There are currently 4 billion hectares of forest on the earth, which together generate 8.1 billion tonnes of carbon annually and absorb 16 billion tonnes. The most biodiverse biomes on the terrestrial world, tropical rainforests, might be expected to be the most substantial carbon sinks. But because of fires, land clearing for plantations, and peat soil drainage, Southeast Asia’s tropical rainforests, one of the three largest systems in the world, are now a net source of carbon emissions. Due to similar changes, the Amazon rainforest is on the verge of becoming a net source. The only rainforest in the top three that is still a substantial carbon sink is the Congo River Basin’s second-largest tropical rainforest . These sobering facts play a role in why conserving forests, particularly rainforests, has emerged as a major topic of discussion in the effort to reduce atmospheric carbon dioxide and halt global warming. References: https://climatefactchecks.org/the-post-falsely-claims-that-geothermal-activities-are-responsible-for-warming-the-oceans/ https://climatefactchecks.org/post-falsely-claims-that-arctic-ice-is-always-changing-emissions-are-irrelevant-and-climate-is-a-hoax/ https://www.iucn.org/resources/issues-brief/forests-and-climate-change#:~:text=Forests%20are%20also%20one%20of,absorbed%20by%20forests%20every%20year . https://www.ipcc.ch/report/ar6/wg1/ https://uplink.weforum.org/uplink/s/uplink-issue/a002o00000vOf09AAC/trillion-trees https://www.weforum.org/impact/investing-in-trees-restoring-forests/ https://trilliontrees.org/ https://www.epa.gov/sites/default/files/2021-04/documents/us-ghg-inventory-2021-chapter-6-land-use-land-use-change-and-forestry.pdf https://www.epa.gov/energy/greenhouse-gas-equivalencies-calculator https://www.theguardian.com/environment/2020/jun/02/football-pitch-area-tropical-rainforest-lost-every-six-seconds https://timesofindia.indiatimes.com/city/l
```

---

## Assembled user message (verbatim, what the model sees)

```text
### Claim:
More than 75 percent of the rainforest is losing resilience; More than half of the rainforest could be converted into savanna in a matter of decades

### Retrieved evidence (top-k chunks from a climate-science knowledge base; quality varies):
[1] carbonbrief_org · Declining ‘resilience’ pushing Amazon rainforest towards tipping point · 2022-03-07
URL: https://www.carbonbrief.org/declining-resilience-pushing-amazon-rainforest-towards-tipping-point/
[Source: carbonbrief_org | Declining ‘resilience’ pushing Amazon rainforest towards tipping point | 2022-03-07]
Three-quarters of the Amazon rainforest has lost “resilience” since 2003 – making it more vulnerable to extreme events such as droughts – according to new research.

Scientists have long warned that climate change and human-driven deforestation could push the Amazon rainforest past a “tipping point”, which would see the dieback of large sections of lush rainforest and a shift to dry savannah. However, modelling studies have been unable to agree on if and when this threshold might be crossed.

The new study uses observational data to explore how the forest’s “resilience” – a measure of how quickly the forest can recover from a drought or extreme weather event – has changed in recent decades.

The findings, published in Nature Climate Change, suggest that more than three-quarters of the Amazon rainforest has already lost resilience over the past two decades. The drier parts of the Amazon – and regions closer to human land use, such as roads and agricultural lands – are the least resilient, the study says.

These findings add further weight to concerns that the Amazon forest is approaching a “critical threshold” or “tipping point”, the authors say. However, in a press briefing, the lead author stressed that we cannot say when a tipping point might happen based on this analysis.

Amazon resilience

As the largest forest in the world, the Amazon is a huge store of carbon and houses at least 10% of the world’s biodiversity. However, humans are having a well-documented impact on the region through deforestation and climate change.

Last year, the deforestation rate reached their highest level in over a decade – with estimates suggesting that more than 10,000 square kilometres of forest have been lost between August 2020 and July 2021. And, as the climate warms, the moist atmosphere in the Amazon is drying out, leaving the forest more vulnerable to droughts and fires.

Both of these factors are reducing the Amazon’s “resilience” – by making the forest slower to recover from extreme events such as droughts.

The Amazon’s loss of resilience has long been a point of concern for scientists. Many modelling studies suggest that deforestation and climate change-driven drying could cause a further loss of resilience over the coming century. However, identifying how much resilience has already been lost can be tricky.

Rather than relying on climate models, this study uses satellite measurements of the Amazon to determine how much resilience the forest has already lost. The authors assess “vegetation optical depth” (VOD)

[2] carbonbrief_org · Declining ‘resilience’ pushing Amazon rainforest towards tipping point · 2022-03-07
URL: https://www.carbonbrief.org/declining-resilience-pushing-amazon-rainforest-towards-tipping-point/
[Source: carbonbrief_org | Declining ‘resilience’ pushing Amazon rainforest towards tipping point | 2022-03-07]
 extreme events such as droughts.

The Amazon’s loss of resilience has long been a point of concern for scientists. Many modelling studies suggest that deforestation and climate change-driven drying could cause a further loss of resilience over the coming century. However, identifying how much resilience has already been lost can be tricky.

Rather than relying on climate models, this study uses satellite measurements of the Amazon to determine how much resilience the forest has already lost. The authors assess “vegetation optical depth” (VOD) – a measure of how much biomass is in the plants, which is closely related to water content.

The authors divide the Amazon rainforest into grid cells, using data from the MODIS satellite to select only green areas of forest that have not already been cleared.

They then assess how VOD evolved in each grid cell over 1991-2016. The map below shows the overall change in VOD, where red shows a decrease in VOD (indicating a loss of biomass) and blue shows an increase in VOD.

Vegetation optical depth of the Amazon over 1991-2016. Source: Boulton et al (2022).

The map shows that overall, VOD decreased over the study period – with the most notable decrease seen in the south-eastern parts of the Amazon basin, along parts of the Amazon river and in some northern areas.

Equilibrium

By assessing incremental changes in VOD over 1991-2016, the authors were able to see how quickly the forest recovers from extreme events such as droughts – and from this determine how its resilience changed.

Dr Chris Boulton – a researcher at the University of Exeter and lead author on the paper – tells Carbon Brief that visualising a ball in a bowl provides a good analogy for the resilience of the Amazon

When the ball is sitting at the bottom of the bowl, he explains, the ball is in its “equilibrium state” – just as the Amazon has its own equilibrium. And the ball is pushed up the sides of the bowl, it will roll back to the centre. Similarly, if the Amazon experiences an extreme event such as a drought that pushes it away from its equilibrium, it will return to its equilibrium.

The faster the ball returns to the bottom of the bowl, the more stable or “resilient” the system is, Boulton explains. The Amazon losing resilience is like the walls of the bowl getting “shallower”, causing the ball to roll back to

[3] carbonbrief_org · Declining ‘resilience’ pushing Amazon rainforest towards tipping point · 2022-03-07
URL: https://www.carbonbrief.org/declining-resilience-pushing-amazon-rainforest-towards-tipping-point/
[Source: carbonbrief_org | Declining ‘resilience’ pushing Amazon rainforest towards tipping point | 2022-03-07]
 show that regions with less rainfall, and which are closer to human land use, such as agricultural lands, generally lose resilience more quickly.

The authors also included a separate piece of analysis showing the distance to roads in Brazil, again showing the areas of forest closer to the roads are likely to lose resilience more quickly.

Dr Chantelle Burton, who is from the UK Met Office and not involved in the paper, tells Carbon Brief that this study is “really important” and uses “robust statistical methods”.

However, Dr David Lapola – a researcher at the University of Campinas in Brazil, who was not involved in the study – adds a word of caution about its use of VOD for assessing resilience. He tells Carbon Brief:

“I think it might be an oversimplification to rely our understanding of the resilience of the forest on the amount of leaves or the greenness of the forest vegetation…We cannot forget that a tree is more than leaves, branches and a trunk. There is a whole, largely unknown world below ground, where roots thrive and where water, nutrients and other relations take place and might influence the resilience of the forest.”

Tipping points

Scientists have long worries that climate change and deforestation could push the Amazon rainforest past a “tipping point”. In a Carbon Brief special week of content in 2020 about tipping points, Prof Peter Cox from the University of Exeter wrote a guest post about Amazon dieback, in which he said:

“The Amazon rainforest sustains its own climate by recycling water to the atmosphere, which maintains rainfall and reduces the length of dry seasons. Deforestation undermines those regulatory mechanisms and may, ultimately, lead to a tipping point.

“This, in combination with the dry season becoming long enough to permit regular natural fires, could see the forest transition to a permanent savannah. This would be characterised by a mixed tree and grassland system with an open canopy that allows the soil to become much hotter and drier, as well as store much less carbon.”

In the press briefing, Boulton told journalists that the study shows we are “approaching a tipping point”. However, he stressed, the authors “can’t say when that’s going to happen” based on this analysis.

While many modelling studies have been conducted on the Amazon rainforest, Dr Chris Jones – a scientist at the UK Met Office who was not involved in the study – says that to the best of his knowledge, the critical slowing down method has not been used on observations of the Amazon

[4] carbonbrief_org · ‘Unprecedented’ stress in up to half of the Amazon may lead to tipping point by 2050 · 2024-02-14
URL: https://www.carbonbrief.org/unprecedented-stress-in-up-to-half-of-the-amazon-may-lead-to-tipping-point-by-2050/
[Source: carbonbrief_org | ‘Unprecedented’ stress in up to half of the Amazon may lead to tipping point by 2050 | 2024-02-14]
As much as half of the Amazon will face several “unprecedented” stressors that could push the forest towards a major tipping point by 2050, new research finds.

The largest rainforest in the world is already under pressure from climate change, deforestation, biodiversity loss and extreme weather.

Researchers analysed data on five key drivers of water stress in the Amazon and looked at how these pressures could lead to “local, regional or even biome-wide forest collapse”.

The new study, published in Nature, finds that by 2050, between 10 and 47% of the Amazon forest will be exposed to “compounding disturbances” that “may trigger unexpected ecosystem transitions”. This could result in large swathes of lush rainforest shifting to dry savannah.

One author of the study tells Carbon Brief that this prospect by mid-century is “very scary”.

The study notes that the complexity of the Amazon “adds uncertainty about future dynamics” and that there are still “opportunities for action”.

The findings highlight the likelihood that “climate change will continue to affect the forest in very unpredictable ways”, a scientist not involved in the study says.

Amazon pressures

The Amazon forest stores a huge amount of carbon and houses at least 10% of the world’s biodiversity. It faces an uncertain future largely due to the effects of deforestation and climate change.

Last year, the Amazon river basin experienced an “exceptional drought” that was 30 times more likely to occur due to climate change, a rapid attribution study found.

Around 20% of the Amazon has already been deforested and a further 6% is “highly degraded”.

According to several studies, the Brazilian section of the Amazon is now an overall net “source” of carbon, rather than a “sink”, due to a number of factors including deforestation.

Scientists have long warned that climate change and human-driven deforestation could push the Amazon rainforest past a “tipping point” – a threshold that, if crossed, would see the “dieback” of large amounts of dense Amazon rainforest and a shift into permanent, dry savannah.

This would be characterised by a mixed tree and grassland system with an open canopy that allows the soil to become much hotter and drier.

Previous studies suggest that the Amazon could be pushed beyond this tipping point if forest loss exceeds 40%. Other research published last October found that recent drying over the Amazon could be the “first warning signal” that the rainforest is approaching a tipping point.

The

[5] climatefactchecks_org · The viral post claims that trees do not help mitigate climate change; however, they are responsible for the global warming · 2023-06-21
URL: https://climatefactchecks.org/the-viral-post-claims-that-trees-do-not-help-mitigate-climate-change-however-they-are-responsible-for-the-global-warming/
[Source: climatefactchecks_org | The viral post claims that trees do not help mitigate climate change; however, they are responsible for the global warming | 2023-06-21]
 parasites, and predators. Earth’s forests continue to be a net carbon dioxide sink despite these perturbations and the slower disintegration process. There are currently 4 billion hectares of forest on the earth, which together generate 8.1 billion tonnes of carbon annually and absorb 16 billion tonnes. The most biodiverse biomes on the terrestrial world, tropical rainforests, might be expected to be the most substantial carbon sinks. But because of fires, land clearing for plantations, and peat soil drainage, Southeast Asia’s tropical rainforests, one of the three largest systems in the world, are now a net source of carbon emissions. Due to similar changes, the Amazon rainforest is on the verge of becoming a net source. The only rainforest in the top three that is still a substantial carbon sink is the Congo River Basin’s second-largest tropical rainforest . These sobering facts play a role in why conserving forests, particularly rainforests, has emerged as a major topic of discussion in the effort to reduce atmospheric carbon dioxide and halt global warming. References: https://climatefactchecks.org/the-post-falsely-claims-that-geothermal-activities-are-responsible-for-warming-the-oceans/ https://climatefactchecks.org/post-falsely-claims-that-arctic-ice-is-always-changing-emissions-are-irrelevant-and-climate-is-a-hoax/ https://www.iucn.org/resources/issues-brief/forests-and-climate-change#:~:text=Forests%20are%20also%20one%20of,absorbed%20by%20forests%20every%20year . https://www.ipcc.ch/report/ar6/wg1/ https://uplink.weforum.org/uplink/s/uplink-issue/a002o00000vOf09AAC/trillion-trees https://www.weforum.org/impact/investing-in-trees-restoring-forests/ https://trilliontrees.org/ https://www.epa.gov/sites/default/files/2021-04/documents/us-ghg-inventory-2021-chapter-6-land-use-land-use-change-and-forestry.pdf https://www.epa.gov/energy/greenhouse-gas-equivalencies-calculator https://www.theguardian.com/environment/2020/jun/02/football-pitch-area-tropical-rainforest-lost-every-six-seconds https://timesofindia.indiatimes.com/city/l

### How to use the evidence above
1. **Read the evidence first, then decide if it actually helps.** When a chunk directly addresses the claim's substantive assertions, prefer it over your training knowledge and cite it inline as a clickable markdown link `[<source-name>](<URL>)` using URLs taken verbatim from the chunks. Numbered references like `[1]`, `[3]`, `[1][2]` are not admissible — users cannot click them; only full markdown links count as citations.
2. **Ignore the evidence if it is off-topic, too sparse, or contradictory.** In that case, fall back on your internal knowledge of climate science (IPCC AR6, NASA, NOAA, peer-reviewed literature) and reason from there. Do NOT bail out to LACKS CONTEXT or IMPRECISE just because retrieval missed — those labels are reserved for claims that are *intrinsically* vague or under-specified, not for retrieval failures. A confident verdict from training knowledge is preferred over a hedged verdict from irrelevant evidence.
3. **Declare your source of truth.** As the FIRST line of the EVIDENCE step inside <think>, write exactly one of:
     EVIDENCE USE: relied-on — <one-line why>
     EVIDENCE USE: partial   — <one-line why>
     EVIDENCE USE: ignored   — <one-line why>
   `relied-on` = the verdict is anchored in the retrieved chunks. `partial` = some chunks helped, supplemented with training knowledge. `ignored` = retrieval was unhelpful, verdict comes from training. Be honest — `ignored` is a valid and correct choice when the chunks do not address the claim.
4. The label must still be exactly one of the 12 codebook labels — these rules constrain *how* you reach the label, not the label set.
```

---

## GPT-4o-mini response (NEW prompt, verbatim)

```text
<think>
1. CONTEXT
The claim asserts that over 75% of the rainforest is losing resilience and that more than half could transition to savanna within decades. This statement highlights concerns about the Amazon rainforest's stability and the potential for significant ecological shifts due to climate change and human activities.

2. SUBCLAIMS
- More than 75% of the rainforest is losing resilience.
- More than half of the rainforest could be converted into savanna in a matter of decades.

3. EVIDENCE
EVIDENCE USE: relied-on — The retrieved evidence directly supports the claim regarding the loss of resilience in the Amazon rainforest and the potential for conversion to savanna. 
- The first piece of evidence states that three-quarters of the Amazon rainforest has lost resilience since 2003, making it more vulnerable to extreme events like droughts, which aligns with the first subclaim about resilience loss ([source](https://www.carbonbrief.org/declining-resilience-pushing-amazon-rainforest-towards-tipping-point/)).
- The second piece of evidence discusses the potential for significant portions of the Amazon to face unprecedented stressors that could lead to a tipping point by 2050, suggesting that large areas could transition to savanna, supporting the second subclaim ([source](https://www.carbonbrief.org/declining-resilience-pushing-amazon-rainforest-towards-tipping-point/)).

4. SHORTLIST
- CORRECT:          REMOVE — The claim cannot be classified as correct because while it aligns with some evidence, it also lacks comprehensive support for the second subclaim about the timeline for conversion to savanna.
- ACCURATE:         SHORTLIST — The claim is factually sound regarding the loss of resilience and the potential for conversion to savanna, but it lacks nuances about the specific conditions and timelines.
- MOSTLY CORRECT:   SHORTLIST — The claim generally aligns with scientific findings but may overstate the immediacy of the savanna transition without clear timelines.
- MOSTLY ACCURATE:  SHORTLIST — The claim is mostly true, but the implications regarding the timeline for savanna conversion could be misleading without additional context.
- CORRECT BUT:      REMOVE — The claim does not require additional details to avoid misinterpretation; it is straightforward.
- IMPRECISE:        REMOVE — The claim is specific and not vague.
- LACKS CONTEXT:    REMOVE — The claim does not omit critical information that would change its meaning.
- UNSUPPORTED:      REMOVE — The claim is supported by evidence.
- MISLEADING:       REMOVE — The claim does not distort the scientific consensus.
- FLAWED REASONING: REMOVE — The claim is based on sound logic and scientific findings.
- INCORRECT:        REMOVE — The claim is not demonstrably false.
- INACCURATE:       REMOVE — The claim does not selectively present data in a misleading way.

5. DEBATE
- SHORTLIST contains: ACCURATE, MOSTLY CORRECT, MOSTLY ACCURATE.

**WHY NOT ACCURATE?**
While the claim is factually sound regarding the loss of resilience, it lacks important nuances about the timeline and conditions under which the rainforest could convert to savanna. The evidence suggests that while significant stressors are present, the exact timeline for such a transition is uncertain, which is a critical detail that the claim does not fully address.

**WHY NOT MOSTLY CORRECT?**
This label is close, but my pick is a better fit because the claim does not merely overstate confidence; it presents a significant concern about the potential for savanna conversion that is supported by evidence. The evidence indicates that while the transition is possible, it is not guaranteed to happen within the specified timeframe, which is a nuance that the claim does not capture.

**WHY MOSTLY ACCURATE?**
The claim is mostly true, as it accurately reflects the current understanding of the Amazon's resilience loss and the potential for ecological shifts. However, it could mislead readers into thinking that the transition to savanna is imminent without acknowledging the uncertainties and complexities involved in such predictions. The evidence supports the assertion of resilience loss and the risk of tipping points, but the timeline for conversion remains uncertain.

6. DECISION
The final label is MOSTLY ACCURATE. The claim about the Amazon rainforest losing resilience and the potential for conversion to savanna is largely true, but it lacks important context regarding the uncertainties in the timeline for such a transition. The evidence indicates significant ecological stressors, but the exact timing of a shift to savanna is not definitively established ([source](https://www.carbonbrief.org/declining-resilience-pushing-amazon-rainforest-towards-tipping-point/)).
</think>

```yaml
assessment: MOSTLY ACCURATE
```
```

## DeepSeek V4 Flash (terminus) response (NEW prompt, verbatim)

```text
<think>
1. CONTEXT
The claim asserts two related propositions about the Amazon rainforest's vulnerability. It states that a large majority of the forest is losing its ability to recover from stress and that a significant portion could transform into a savanna within a relatively short timeframe. The tone is declarative and alarming, presenting these as established or highly probable outcomes.

2. SUBCLAIMS
- Subclaim A: More than 75 percent of the rainforest is losing resilience.
- Subclaim B: More than half of the rainforest could be converted into savanna in a matter of decades.

3. EVIDENCE
EVIDENCE USE: partial — The retrieved evidence directly supports the first subclaim and provides context for the second, but the specific "more than half" conversion figure is not explicitly stated in the provided chunks.

- **Subclaim A**: The claim that "more than 75 percent of the rainforest is losing resilience" is directly supported by a 2022 study published in Nature Climate Change, which used satellite data to assess resilience. The research found that "three-quarters of the Amazon rainforest has lost 'resilience' since 2003" and that this loss is more pronounced in drier regions and areas closer to human activity [Declining ‘resilience’ pushing Amazon rainforest towards tipping point](https://www.carbonbrief.org/declining-resilience-pushing-amazon-rainforest-towards-tipping-point/). This aligns with the scientific consensus that deforestation and climate change are reducing the forest's ability to recover from disturbances.

- **Subclaim B**: The assertion that "more than half of the rainforest could be converted into savanna in a matter of decades" is consistent with the direction of scientific warnings but is a specific interpretation of projected outcomes. A 2024 study cited in the evidence notes that by 2050, "between 10 and 47% of the Amazon forest will be exposed to... disturbances that may trigger unexpected ecosystem transitions," potentially leading to a shift to savanna ['Unprecedented’ stress in up to half of the Amazon may lead to tipping point by 2050](https://www.carbonbrief.org/unprecedented-stress-in-up-to-half-of-the-amazon-may-lead-to-tipping-point-by-2050/). The upper bound of this range (47%) is close to "more than half," but the claim's phrasing ("could be converted") and the specific "more than half" figure represent a slight extrapolation beyond the study's stated upper limit, introducing a minor overstatement.

4. SHORTLIST
- CORRECT: REMOVE — The claim is not perfectly aligned with the evidence; Subclaim B slightly overstates the projected area at risk beyond the upper bound provided by the most recent study.
- ACCURATE: REMOVE — While factually sound in spirit, the claim's lack of nuance regarding the uncertainty and the specific quantification in Subclaim B prevents it from being merely an issue of missing context; it involves a slight overstatement.
- MOSTLY CORRECT: SHORTLIST — This label fits well as the claim aligns generally with scientific studies but may overstate confidence or predictions slightly, which matches the minor overstatement in Subclaim B.
- MOSTLY ACCURATE: SHORTLIST — The claim is mostly true, and the inaccuracy in the specific "more than half" figure is minor and doesn't significantly alter the claim's overall validity regarding the grave risk.
- CORRECT BUT: SHORTLIST — The claim is largely accurate but requires additional details about the uncertainty of the timeline and the precise percentage at risk to avoid potential misinterpretation.
- IMPRECISE: REMOVE — The claim is quite specific with numerical figures ("75 percent," "more than half") and a timeframe ("decades"), so it is not vague or open to multiple interpretations.
- LACKS CONTEXT: REMOVE — While context about uncertainty is always valuable, the claim's issue is not a critical omission that changes its meaning but a slight overstatement of a specific figure.
- UNSUPPORTED: REMOVE — The claim is supported by peer-reviewed research and scientific analysis, as shown in the evidence.
- MISLEADING: REMOVE — The claim does not use technically correct information to create a distorted impression; it is a reasonable, if slightly strong, interpretation of the evidence.
- FLAWED REASONING: REMOVE — The claim is not based on faulty logic or incorrect assumptions; it is grounded in observational and modeling studies.
- INCORRECT: REMOVE — The claim is not demonstrably false; its core message about declining resilience and a significant tipping point risk is supported.
- INACCURATE: REMOVE — The claim does not contain fundamental inaccuracies that distort the scientific consensus; the error is minor and quantitative.

5. DEBATE
- SHORTLIST contains: MOSTLY CORRECT, MOSTLY ACCURATE, CORRECT BUT

**WHY NOT MOSTLY ACCURATE?**
The codebook defines "MOSTLY ACCURATE" for claims that are "mostly true, with minor inaccuracies that don't significantly alter the claim's validity." While this is a close fit, the specific overstatement of "more than half" (vs. the study's "up to 47%") is better characterized as a slight overstatement of a prediction, which aligns more closely with the "MOSTLY CORRECT" example of overstating "confidence or predictions slightly." The distinction is subtle, but the predictive nature of the second subclaim makes "MOSTLY CORRECT" a more precise categorization.

**WHY NOT CORRECT BUT?**
The "CORRECT BUT" label is for claims that are "largely accurate but require additional details to avoid misinterpretation." The claim here does require context about uncertainty, but its primary deviation from the evidence is the quantitative overstatement in Subclaim B, not just a need for additional clarifying details. The codebook example for "CORRECT BUT" involves an overstatement of scope ("destroy all ecosystems"), whereas this claim's issue is a specific numerical overstatement of a prediction, making "MOSTLY CORRECT" a better fit.

**WHY MOSTLY CORRECT?**
The claim "aligns generally with scientific studies but may overstate confidence or predictions slightly," which is the definition of "MOSTLY CORRECT." The first subclaim is directly supported by a peer-reviewed study [Declining ‘resilience’ pushing Amazon rainforest towards tipping point](https://www.carbonbrief.org/declining-resilience-pushing-amazon-rainforest-towards-tipping-point/). The second subclaim is consistent with the dire warnings in the scientific literature, including a 2024 study projecting stress on up to 47% of the forest ['Unprecedented’ stress in up to half of the Amazon may lead to tipping point by 2050](https://www.carbonbrief.org/unprecedented-stress-in-up-to-half-of-the-amazon-may-lead-to-tipping-point-by-2050/). However, stating "more than half" slightly exceeds the study's upper estimate, and the phrase "in a matter of decades" slightly overstates the certainty of the timeline, as the authors caution that predicting the exact timing of a tipping point remains challenging. This minor overstatement is analogous to the "MOSTLY CORRECT" example about overstating the impact of a plant-based diet.

6. DECISION
The claim is **MOSTLY CORRECT**. It accurately reflects the findings of a major 2022 study showing that over three-quarters of the Amazon has lost resilience [Declining ‘resilience’ pushing Amazon rainforest towards tipping point](https://www.carbonbrief.org/declining-resilience-pushing-amazon-rainforest-towards-tipping-point/). The assertion that more than half the forest could become savanna is a reasonable interpretation of the risk, supported by a 2024 study projecting unprecedented stress on up to 47% of the biome by 2050 ['Unprecedented’ stress in up to half of the Amazon may lead to tipping point by 2050](https://www.carbonbrief.org/unprecedented-stress-in-up-to-half-of-the-amazon-may-lead-to-tipping-point-by-2050/). However, the claim slightly overstates the specific area at risk and the certainty of the timeline, moving it from a perfectly correct statement to one that is mostly correct.
</think>

```yaml
assessment: MOSTLY CORRECT
```


```

---

## Archive: what the old prompt produced on the same row

For reference, the responses logged in `.archives/v3/` under the previous `climinator_v2` prompt + IRONCLAD RULES user-template.

### GPT-4o-mini + RAG (OLD) — pred=`CORRECT`
```text
<think>
1. CONTEXT
   The claim asserts that more than 75 percent of the rainforest is losing resilience and that more than half could be converted into savanna in a matter of decades, attributed to Sarah Kaplan from the Washington Post.

2. SUBCLAIMS
   - More than 75 percent of the rainforest is losing resilience.
   - More than half of the rainforest could be converted into savanna in a matter of decades.

3. EVIDENCE
   - The claim that more than 75 percent of the Amazon rainforest has lost resilience is supported by research indicating that three-quarters of the Amazon has indeed lost resilience since 2003, making it more vulnerable to extreme events like droughts [1].
   - The assertion that more than half of the rainforest could be converted into savanna is consistent with findings that suggest significant portions of the Amazon may face unprecedented stressors that could lead to a tipping point by 2050, potentially resulting in large areas transitioning to savannah [4].

4. ADJUDICATION
   - More than 75 percent of the rainforest is losing resilience: ACCURATE — supported by research indicating significant loss of resilience in the Amazon.
   - More than half of the rainforest could be converted into savanna in a matter of decades: ACCURATE — aligns with studies suggesting that substantial portions of the Amazon could transition to savannah due to climate change and deforestation pressures.

5. TIER
   Very High Credibility

6. DECISION
   CORRECT — The claim aligns perfectly with established scientific consensus and evidence, presenting no factual issues.
</think>

```yaml
assessment: CORRECT
```
```

### GPT-4o-mini offline (OLD) — pred=`CORRECT`
```text
<think>
1. CONTEXT
   The claim asserts that a significant portion of the rainforest is losing resilience and could be converted into savanna within decades, presented by Sarah Kaplan from the Washington Post.

2. SUBCLAIMS
   - More than 75 percent of the rainforest is losing resilience.
   - More than half of the rainforest could be converted into savanna in a matter of decades.

3. EVIDENCE
   - The scientific consensus indicates that deforestation, climate change, and other factors are indeed leading to a loss of resilience in rainforests, with some studies suggesting significant areas are at risk of transitioning to savanna-like ecosystems.
   - Research supports the idea that large portions of the Amazon rainforest are at risk of conversion to savanna due to climate change and human activities, with projections indicating this could occur within decades if current trends continue.

4. ADJUDICATION
   - More than 75 percent of the rainforest is losing resilience: ACCURATE — Studies show significant loss of resilience in many rainforest areas due to various stressors.
   - More than half of the rainforest could be converted into savanna in a matter of decades: ACCURATE — Projections indicate that if deforestation and climate change continue, substantial areas of rainforest could transition to savanna.

5. TIER
   Very High Credibility

6. DECISION
   CORRECT — The claim aligns perfectly with established scientific consensus and evidence regarding rainforest resilience and potential conversion to savanna.
</think>

```yaml
assessment: CORRECT
```
```

### DeepSeek V4 Flash + RAG (OLD) — pred=`MISLEADING`
```text
```yaml
assessment: MISLEADING
```
```

### DeepSeek V4 Flash offline (OLD) — pred=`MOSTLY ACCURATE`
```text
```yaml
assessment: MOSTLY ACCURATE
```
```

