# Intercoder Reliability (ICR) Analysis

- **Items coded:** 50
- **Coders:** `travcoan`, `mirjamnanko`
- **Items with disagreement at Level 3:** 23/50

## Method

Two coders (travcoan, mirjamnanko) independently applied codes from the CARDS
contrarian-claims taxonomy to 50 items. Codes follow a three-level
hierarchical scheme (e.g. `2_1_4`), where each item may receive multiple codes.

**Code normalization.** Before scoring, we removed redundant parent codes from
each coder's set on a per-item basis. A code `A` is redundant when a strictly
more specific code `B` exists in the same set such that `A`'s non-zero
components match `B`'s at the corresponding positions (e.g. `2_1_0` is dropped
when `2_1_4` is also present, and `2_0_0` is dropped when `2_1_0` is present).
This avoids double-counting the same claim at parent and child levels.

**Hierarchy levels.** We evaluate agreement at three levels of granularity by
truncating each code to its first 1, 2, or 3 components. Level 1 reflects the
top-level category, Level 2 the sub-category, and Level 3 the most specific
claim.

**Krippendorff's Alpha.** Because items are multi-label, we transform each
(item, code) pair at a given level into a binary unit (1 if the coder applied
the code, 0 otherwise). Alpha is then computed on the resulting `2 x (n_items
* n_codes)` reliability matrix using the `nominal` level of measurement, via
the `krippendorff` Python package.

**Supporting statistics.** For each level we also report (i) percent exact
agreement on the full code set per item, (ii) mean Jaccard similarity of the
two coders' code sets per item, and (iii) the number of unique codes that
appear at that level.

## Reliability by Hierarchy Level

| Level | Krippendorff's Alpha | % Exact Agreement | Mean Jaccard | # Unique Codes |
|-------|----------------------|-------------------|--------------|----------------|
| 1 | 0.8105 | 68.0% | 0.8067 | 8 |
| 2 | 0.8187 | 62.0% | 0.7780 | 25 |
| 3 | 0.7888 | 54.0% | 0.7313 | 39 |

## Descriptive Statistics

| Stage | Coder | Mean codes / item | Total unique codes |
|-------|-------|-------------------|--------------------|
| Before normalization | `travcoan` | 1.62 | 38 |
| Before normalization | `mirjamnanko` | 1.78 | 41 |
| After normalization | `travcoan` | 1.60 | 38 |
| After normalization | `mirjamnanko` | 1.66 | 39 |

## Claim Frequency Distribution (Level 3)

Counts are the number of items (out of 50) on which each Level-3 claim was applied. *Either* counts items where at least one coder used the claim; *Both* counts items where the coders agreed on it; per-coder columns count items where that coder applied it.

| Code | Claim | travcoan | mirjamnanko | Either | Both |
|------|-------|----------|-------------|--------|------|
| `4_1_1` | Climate solutions will increase costs, harm the economy, and/or kill jobs | 8 | 6 | 9 | 5 |
| `2_1_4` | Climate has changed naturally and/or it's been warm in the past, so we shouldn't worry too much about recent climate change. | 7 | 7 | 7 | 7 |
| `6_2_0` | Climate change is a hoax or conspiracy. We have been deceived by climate scientists, politicians, bureaucrats, and environmental organizations on climate change. | 5 | 6 | 7 | 4 |
| `1_0_0` | Global warming is not happing. Climate change is NOT leading to melting ice (such as glaciers, sea ice, and permafrost), increased extreme weather, or rising sea levels. Cold weather also shows that climate change is not happening. | 3 | 6 | 6 | 3 |
| `4_1_5` | Climate regulation limits individual liberty, freedom, and undermines capitalism. This includes but not limited to arguments that climate solutions are a justification for government overreach and control. Note that claims of a "war on energy" would fall into this category. | 2 | 5 | 6 | 1 |
| `6_1_0` | Climate change proponents are alarmist, biased, wrong, hypocritical, and/or politically motivated. | 5 | 3 | 5 | 3 |
| `1_3_0` | We are experiencing cold weather, therefore climate change is not happening. | 4 | 4 | 4 | 4 |
| `1_7_0` | Climate change does not cause or worsen extreme weather events such as heatwaves, droughts, wildfires and floods. | 4 | 4 | 4 | 4 |
| `4_2_4` | Climate action is pointless because of the emissions of other countries such as China or India | 4 | 4 | 4 | 4 |
| `2_1_0` | Humans are not the causing change. Instead, climate change is due to natural variation. | 2 | 2 | 3 | 1 |
| `3_2_1` | Plants and animals will adapt to climate change and therefore the impacts will be minimal. | 3 | 3 | 3 | 3 |
| `4_2_7` | Climate-friendly technologies and practices are ineffective and won't work. | 3 | 2 | 3 | 2 |
| `7_0_0` | We need fossil fuels for economic growth, prosperity, and to maintain our standard of living. | 3 | 2 | 3 | 2 |
| `1_2_0` | We are heading into a period of global cooling or an ice age. | 1 | 2 | 2 | 1 |
| `1_6_0` | Sea level rise is exaggerated and not accelerating. | 2 | 1 | 2 | 1 |
| `2_1_1` | The sun, cosmic rays, or other astronomical phenomena are causing climate change. | 2 | 2 | 2 | 2 |
| `2_3_0` | There's no evidence for greenhouse effect or carbon dioxide driving climate change | 1 | 2 | 2 | 1 |
| `3_1_0` | Climate sensitivity is low and there are negative feedbacks that will reduce warming. | 2 | 1 | 2 | 1 |
| `3_3_1` | CO2 is plant food -- it helps plant growth. | 2 | 2 | 2 | 2 |
| `4_1_3` | Climate solutions will harm the environment, habitats, and/or species | 2 | 1 | 2 | 1 |
| `6_1_3` | Politicians, governments, and organizations such as the UN are alarmist, biased, and/or wrong on climate change. | 1 | 2 | 2 | 1 |
| `0_0_0` | No relevant claim detected | 1 | 1 | 1 | 1 |
| `1_1_1` | Antarctica is gaining ice. | 1 | 1 | 1 | 1 |
| `1_5_0` | Oceans are not warming and may even be cooling. | 1 | 1 | 1 | 1 |
| `1_8_0` | Climate advocates and alarmist changed the name from global warming to climate change so that cold weather as well as hot can be taken as evidence. | 1 | 1 | 1 | 1 |
| `2_0_0` | Greenhouse gases from humans are not the causing climate change. | 1 | 1 | 1 | 1 |
| `2_3_1` | CO2 is just a trace gas and so can't cause climate change. | 1 | 1 | 1 | 1 |
| `3_2_0` | Plants and animals are not showing harmful impacts from climate change and may be benefiting from climate change | 0 | 1 | 1 | 0 |
| `3_2_2` | Polar bears are not in danger from climate change. | 1 | 1 | 1 | 1 |
| `4_0_0` | Climate solutions are harmful or unnecessary | 1 | 0 | 1 | 0 |
| `4_2_11` | It's better to adapt to climate change and increase resiliency then to devote resources to mitigation. | 1 | 1 | 1 | 1 |
| `4_2_14` | There are more pressing problems than climate change and we should address those first | 1 | 1 | 1 | 1 |
| `4_2_3` | A single country or region only contributes a small percentage of global emissions | 0 | 1 | 1 | 0 |
| `5_1_0` | There is no scientific consensus on climate change. Scientists continue to disagree on many aspects of climate change and the science is not settled. This includes arguments that the science isn't settled or isn't there. | 1 | 0 | 1 | 0 |
| `5_4_0` | Climate models are flawed, unreliable, or uncertain. | 1 | 1 | 1 | 1 |
| `6_0_0` | Climate scientists and proponents of climate action are alarmist, biased, wrong, hypocritical, corrupt, and/or politically motivated. | 0 | 1 | 1 | 0 |
| `6_1_4` | Environmentalists are alarmist, biased, and/or wrong on climate change. | 1 | 1 | 1 | 1 |
| `6_1_5` | Scientists and academics are alarmist, biased, and/or wrong on climate change. | 0 | 1 | 1 | 0 |
| `7_3_0` | Fossil fuels are necessary to meet energy demand. This includes, but not limited to, arguments that we need all forms of energy, including fossil fuels. | 1 | 1 | 1 | 1 |
