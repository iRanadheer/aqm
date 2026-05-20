# Prompt dump · itemId `leippold_001`

- **Model:** `openai/gpt-4o-mini` (backend: `openrouter`)
- **Prompt variant:** `climinator`
- **Evidence source:** Local hybrid RAG (Qwen3 dense + BM25 + BGE rerank)
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
[1] factcheck_afp_com · TikTok clips revive debunked climate disaster theory · 2023-06-23
URL: https://factcheck.afp.com/doc.afp.com.33KK78D
[Source: factcheck_afp_com | TikTok clips revive debunked climate disaster theory | 2023-06-23]
"The Adam and Eve story, the theory of that is that it happens in cycles of 6,500 years and that it's a 90-degree flip. But six days later or on the seventh day, it corrects itself and the planet flips," says Jimmy Corsetti in a January 18, 2023 episode of the "Joe Rogan Experience," the most popular podcast on Spotify and a frequent source of misinformation.

"The theory is that when that event happens it's going to be cataclysmic."

Clips of Corsetti, who runs a YouTube channel about ancient mysteries and conspiracy theories, have since gained millions of interactions on TikTok and YouTube.

The theory in the posts is based on a book published in the 1960s called "The Adam and Eve Story -- the History of Cataclysms."

Chan Thomas, an engineer and self-proclaimed polymath, described how earthquakes, tsunamis and supersonic winds destroy civilizations when the poles switch places, thrusting the Arctic and Antarctic into the tropics. He detailed past pole reversals through a reinterpretation of the book of Genesis, pre-biblical legends and geological phenomena.

In another video published March 2, TikTok influencer Darris Watkins cited the book, calling it "a CIA document" and linking it to global warming.

"The Sahara Desert used to be green and Antarctica used to have forests," he says in the clip, viewed more than 19 million times. "So now like some places turn cold, some places turn hot."

Media Matters for America, a liberal media watchdog, said in May 2023 that posts promoting the Adam and Eve theory are part of a new trend in which skeptics promote "cataclysm narratives as climate sedatives."

Geoscientists say that although Earth's magnetic poles can reverse, no such switch is imminent. And when it does happen, it does not cause a cataclysm or climate change.

"There's no evidence that Earth's climate has been significantly impacted by the last three magnetic field excursions, nor by any excursion event within at least the last 2.8 million years," the National Aeronautics and Space Administration (NASA) says on its website (archived here).

Corsetti later told The Verge that the TikTok posts took his remarks out of context and that the Adam and Eve theory is "certainly not considered accepted science."

'No documented catastrophes'

NASA has refuted claims that a pole shift is imminent

[2] factcheck_afp_com · TikTok clips revive debunked climate disaster theory · 2023-06-23
URL: https://factcheck.afp.com/doc.afp.com.33KK78D
[Source: factcheck_afp_com | TikTok clips revive debunked climate disaster theory | 2023-06-23]
 climate has been significantly impacted by the last three magnetic field excursions, nor by any excursion event within at least the last 2.8 million years," the National Aeronautics and Space Administration (NASA) says on its website (archived here).

Corsetti later told The Verge that the TikTok posts took his remarks out of context and that the Adam and Eve theory is "certainly not considered accepted science."

'No documented catastrophes'

NASA has refuted claims that a pole shift is imminent and likely to cause the apocalypse, although it could make compasses point the other way and disorient birds and fish that use the magnetic field to navigate.

Earth's magnetic field is "in continual flux, its strength waxing and waning over time," NASA says on its website. The ongoing shift affects navigation but there is "little scientific evidence of any significant links between Earth's drifting magnetic poles and climate."

Changes in Earth's magnetic field -- generated by the planet's molten iron core -- apply only to its magnetic poles, not the geographical North and South poles. The former flip locations "every 300,000 years or so," according to NASA -- a process that usually takes place over "hundreds to thousands of years."

Paleomagnetic records, traces of metal magnetization in the ground, indicate the poles have fully reversed 183 times in the past 83 million years -- and several hundred times in the past 160 million years.

The last full flip was about 786,000 years ago, according to a 2014 study (archived here). The reversal may have taken less than a century but was preceded by "a period of instability that spanned more than 6,000 years," according to a University of California-Berkeley article about the research (archived here).

"There are no documented catastrophes associated with past reversals, despite much searching in the geologic and biologic record," the October 2014 article says, noting that a pole shift could disrupt the electrical grid.

NASA also says fossil records show no evidence of major changes such as extinctions or floods due to magnetic pole reversals.

The poles flipped about 41,500 years ago, only to change again 500 years later. However, ice cores from that period "don't show any major changes," NASA says on its website.

'Routine declassification'

The Central Intelligence Agency (CIA) in 2013 released 57 pages of a 1965 edition of Chan Thomas's book (archived

[3] snopes_com · NASA Reports Magnetosphere Collapse · 2016-04-26
URL: https://www.snopes.com/fact-check/nasa-reports-magnetosphere-collapse/
[Source: snopes_com | NASA Reports Magnetosphere Collapse | 2016-04-26]
. This process can also be modeled with supercomputers. Ours is, without hyperbole, a dynamic planet. The flow of liquid iron in Earth's core creates electric currents, which in turn create the magnetic field. So while parts of Earth's outer core are too deep for scientists to measure directly, we can infer movement in the core by observing changes in the magnetic field. The magnetic north pole has been creeping northward – by more than 600 miles (1,100 km) – since the early 19th century, when explorers first located it precisely. It is moving faster now, actually, as scientists estimate the pole is migrating northward about 40 miles per year, as opposed to about 10 miles per year in the early 20th century. Another doomsday hypothesis about a geomagnetic flip plays up fears about incoming solar activity. This suggestion mistakenly assumes that a pole reversal would momentarily leave Earth without the magnetic field that protects us from solar flares and coronal mass ejections from the sun. But, while Earth's magnetic field can indeed weaken and strengthen over time, there is no indication that it has ever disappeared completely. A weaker field would certainly lead to a small increase in solar radiation on Earth – as well as a beautiful display of aurora at lower latitudes - but nothing deadly. Moreover, even with a weakened magnetic field, Earth's thick atmosphere also offers protection against the sun's incoming particles. The science shows that magnetic pole reversal is – in terms of geologic time scales – a common occurrence that happens gradually over millennia. While the conditions that cause polarity reversals are not entirely predictable – the north pole's movement could subtly change direction, for instance – there is nothing in the millions of years of geologic record to suggest that any of the 2012 doomsday scenarios connected to a pole reversal should be taken seriously. A reversal might, however, be good business for magnetic compass manufacturers. However, NASA space weather analyst Leila Mays told us that the 23 April readings weren't the result of normal fluctuations in any case, but rather a software glitch that led to erroneous simulations: The images were taken from the integrated space weather analysis system (iswa.ccmc.gsfc.nasa.gov). iSWA is tool to display some of the real-time simulations of the models available at the CCMC - the Community Coordinated Modeling Center https://ccmc.gsfc.nasa.gov/about.php). The CCMC provides access to space science simulations

[4] snopes_com · NASA Reports Magnetosphere Collapse · 2016-04-26
URL: https://www.snopes.com/fact-check/nasa-reports-magnetosphere-collapse/
[Source: snopes_com | NASA Reports Magnetosphere Collapse | 2016-04-26]
Fact Check NASA Reports Magnetosphere Collapse A conspiracy-oriented web site claimed that that Earth's magnetosphere "COLLAPSED" for two hours, but NASA confirmed the incident was a software glitch — and that such fluctuations aren't doomsday events. Kim LaCapria Published April 26, 2016 Claim: Earth's magnetosphere collapsed for two hours on 23 April 2016. Rating: False About this rating On 23 April 2016, unreliable web site Superstation95 reported that data gleaned from NASA revealed that Earth's magnetosphere had "COLLAPSED" for a period of two hours: A stunning and terrifying event has taken place in space surrounding our planet; for two hours today, earth's "Magnetosphere" COLLAPSED around the entire planet! The magnetosphere is what protects earth from solar winds and some radiation. This morning at 01:37:05 eastern US Time, which is 05:37:05 UTC, satellites from the NASA Space Weather Prediction Center detected a complete collapse of earth's magnetosphere! It simply vanished for just over two hours, resuming as normal around 03:39:51 eastern US time, which is 07:39:51 UTC. Here is how NASA Space Weather Satellites recorded the event: This report was false, however. Heliophysics (the study of the effects of the sun on the solar system) is a complex topic, making bombastic claims easier to cloak in a veneer of plausibility. NASA's web site features a brief explainer which notes that doomsday scenarios tend to leverage unfamiliarity with the topic in order to exaggerate normal atmospheric fluctuations: Earth's polarity is not a constant. Unlike a classic bar magnet, or the decorative magnets on your refrigerator, the matter governing Earth's magnetic field moves around. Geophysicists are pretty sure that the reason Earth has a magnetic field is because its solid iron core is surrounded by a fluid ocean of hot, liquid metal. This process can also be modeled with supercomputers. Ours is, without hyperbole, a dynamic planet. The flow of liquid iron in Earth's core creates electric currents, which in turn create the magnetic field. So while parts of Earth's outer core are too deep for scientists to measure directly, we can infer movement in the core by observing changes in the magnetic field. The magnetic north pole has been creeping northward – by more than 600 miles (1,100 km) – since the

[5] science_nasa_gov · Flip Flop: Why Variations in Earth's Magnetic Field Aren't Causing Today's Climate Change - NASA Science · None
URL: https://science.nasa.gov/science-research/earth-science/flip-flop-why-variations-in-earths-magnetic-field-arent-causing-todays-climate-change/
[Source: science_nasa_gov | Flip Flop: Why Variations in Earth's Magnetic Field Aren't Causing Today's Climate Change - NASA Science | ?]
By Alan Buis, NASA's Jet Propulsion Laboratory Earth is surrounded by an immense magnetic field, called the magnetosphere. Generated by powerful, dynamic forces at the center of our world, our magnetosphere shields us from erosion of our atmosphere by the solar wind, particle radiation from coronal mass ejections (eruptions of large clouds of energetic, magnetized plasma from the Sun’s corona into space), and from cosmic rays from deep space. Our magnetosphere plays the role of gatekeeper, repelling these forms of energy that are harmful to life, trapping most of it safely away from Earth’s surface. You can learn more about Earth’s magnetosphere here . Earth is surrounded by a system of magnetic fields, called the magnetosphere. The magnetosphere shields our home planet from harmful solar and cosmic particle radiation, but it can change shape in response to incoming space weather from the Sun. NASA's Scientific Visualization Studio A constant outflow of solar material streams out from the Sun, depicted here in an artist's rendering. This solar wind is always passing by Earth. NASA Goddard's Conceptual Image Lab/Greg Shirah Since the forces that generate our magnetic field are constantly changing, the field itself is also in continual flux, its strength waxing and waning over time. This causes the location of Earth’s magnetic north and south poles to gradually shift, and to even completely flip locations every 300,000 years or so. That might be somewhat important if you use a compass, or for certain animals like birds, fish and sea turtles, whose internal compasses use the magnetic field to navigate . To view this video please enable JavaScript, and consider upgrading to a web browser that supports HTML5 video The Sun unleashed a series of four coronal mass ejections (CMEs) on May 22-24, 2010 as the STEREO Ahead spacecraft watched the action. In the coronagraph images, the Sun, blocked out by an occulting disk (seen as red), is represented by a white disk to show its relative size. CMEs are large solar storms that expel a billion tons of matter at a million miles per hour or more. Credit: NASA/European Space Agency Some people have claimed that variations in Earth’s magnetic field are contributing to current global warming and can cause catastrophic climate change. However, the science doesn’t support that argument. In this blog, we’ll examine a number of proposed hypotheses regarding the effects of changes in Earth’s magnetic field on
```

---

## Assembled user message (verbatim, what the model sees)

```text
### Claim:
Magnetic poles reversals involve the Earth flipping vertically and momentarily stopping its rotation, causing cataclysmic events during 6 days.

### Source:
Joe Rogan & Jimmy Corsetti, TikTok&Spotify

### Evidence (retrieved from a vetted climate-science knowledge base):
[1] factcheck_afp_com · TikTok clips revive debunked climate disaster theory · 2023-06-23
URL: https://factcheck.afp.com/doc.afp.com.33KK78D
[Source: factcheck_afp_com | TikTok clips revive debunked climate disaster theory | 2023-06-23]
"The Adam and Eve story, the theory of that is that it happens in cycles of 6,500 years and that it's a 90-degree flip. But six days later or on the seventh day, it corrects itself and the planet flips," says Jimmy Corsetti in a January 18, 2023 episode of the "Joe Rogan Experience," the most popular podcast on Spotify and a frequent source of misinformation.

"The theory is that when that event happens it's going to be cataclysmic."

Clips of Corsetti, who runs a YouTube channel about ancient mysteries and conspiracy theories, have since gained millions of interactions on TikTok and YouTube.

The theory in the posts is based on a book published in the 1960s called "The Adam and Eve Story -- the History of Cataclysms."

Chan Thomas, an engineer and self-proclaimed polymath, described how earthquakes, tsunamis and supersonic winds destroy civilizations when the poles switch places, thrusting the Arctic and Antarctic into the tropics. He detailed past pole reversals through a reinterpretation of the book of Genesis, pre-biblical legends and geological phenomena.

In another video published March 2, TikTok influencer Darris Watkins cited the book, calling it "a CIA document" and linking it to global warming.

"The Sahara Desert used to be green and Antarctica used to have forests," he says in the clip, viewed more than 19 million times. "So now like some places turn cold, some places turn hot."

Media Matters for America, a liberal media watchdog, said in May 2023 that posts promoting the Adam and Eve theory are part of a new trend in which skeptics promote "cataclysm narratives as climate sedatives."

Geoscientists say that although Earth's magnetic poles can reverse, no such switch is imminent. And when it does happen, it does not cause a cataclysm or climate change.

"There's no evidence that Earth's climate has been significantly impacted by the last three magnetic field excursions, nor by any excursion event within at least the last 2.8 million years," the National Aeronautics and Space Administration (NASA) says on its website (archived here).

Corsetti later told The Verge that the TikTok posts took his remarks out of context and that the Adam and Eve theory is "certainly not considered accepted science."

'No documented catastrophes'

NASA has refuted claims that a pole shift is imminent

[2] factcheck_afp_com · TikTok clips revive debunked climate disaster theory · 2023-06-23
URL: https://factcheck.afp.com/doc.afp.com.33KK78D
[Source: factcheck_afp_com | TikTok clips revive debunked climate disaster theory | 2023-06-23]
 climate has been significantly impacted by the last three magnetic field excursions, nor by any excursion event within at least the last 2.8 million years," the National Aeronautics and Space Administration (NASA) says on its website (archived here).

Corsetti later told The Verge that the TikTok posts took his remarks out of context and that the Adam and Eve theory is "certainly not considered accepted science."

'No documented catastrophes'

NASA has refuted claims that a pole shift is imminent and likely to cause the apocalypse, although it could make compasses point the other way and disorient birds and fish that use the magnetic field to navigate.

Earth's magnetic field is "in continual flux, its strength waxing and waning over time," NASA says on its website. The ongoing shift affects navigation but there is "little scientific evidence of any significant links between Earth's drifting magnetic poles and climate."

Changes in Earth's magnetic field -- generated by the planet's molten iron core -- apply only to its magnetic poles, not the geographical North and South poles. The former flip locations "every 300,000 years or so," according to NASA -- a process that usually takes place over "hundreds to thousands of years."

Paleomagnetic records, traces of metal magnetization in the ground, indicate the poles have fully reversed 183 times in the past 83 million years -- and several hundred times in the past 160 million years.

The last full flip was about 786,000 years ago, according to a 2014 study (archived here). The reversal may have taken less than a century but was preceded by "a period of instability that spanned more than 6,000 years," according to a University of California-Berkeley article about the research (archived here).

"There are no documented catastrophes associated with past reversals, despite much searching in the geologic and biologic record," the October 2014 article says, noting that a pole shift could disrupt the electrical grid.

NASA also says fossil records show no evidence of major changes such as extinctions or floods due to magnetic pole reversals.

The poles flipped about 41,500 years ago, only to change again 500 years later. However, ice cores from that period "don't show any major changes," NASA says on its website.

'Routine declassification'

The Central Intelligence Agency (CIA) in 2013 released 57 pages of a 1965 edition of Chan Thomas's book (archived

[3] snopes_com · NASA Reports Magnetosphere Collapse · 2016-04-26
URL: https://www.snopes.com/fact-check/nasa-reports-magnetosphere-collapse/
[Source: snopes_com | NASA Reports Magnetosphere Collapse | 2016-04-26]
. This process can also be modeled with supercomputers. Ours is, without hyperbole, a dynamic planet. The flow of liquid iron in Earth's core creates electric currents, which in turn create the magnetic field. So while parts of Earth's outer core are too deep for scientists to measure directly, we can infer movement in the core by observing changes in the magnetic field. The magnetic north pole has been creeping northward – by more than 600 miles (1,100 km) – since the early 19th century, when explorers first located it precisely. It is moving faster now, actually, as scientists estimate the pole is migrating northward about 40 miles per year, as opposed to about 10 miles per year in the early 20th century. Another doomsday hypothesis about a geomagnetic flip plays up fears about incoming solar activity. This suggestion mistakenly assumes that a pole reversal would momentarily leave Earth without the magnetic field that protects us from solar flares and coronal mass ejections from the sun. But, while Earth's magnetic field can indeed weaken and strengthen over time, there is no indication that it has ever disappeared completely. A weaker field would certainly lead to a small increase in solar radiation on Earth – as well as a beautiful display of aurora at lower latitudes - but nothing deadly. Moreover, even with a weakened magnetic field, Earth's thick atmosphere also offers protection against the sun's incoming particles. The science shows that magnetic pole reversal is – in terms of geologic time scales – a common occurrence that happens gradually over millennia. While the conditions that cause polarity reversals are not entirely predictable – the north pole's movement could subtly change direction, for instance – there is nothing in the millions of years of geologic record to suggest that any of the 2012 doomsday scenarios connected to a pole reversal should be taken seriously. A reversal might, however, be good business for magnetic compass manufacturers. However, NASA space weather analyst Leila Mays told us that the 23 April readings weren't the result of normal fluctuations in any case, but rather a software glitch that led to erroneous simulations: The images were taken from the integrated space weather analysis system (iswa.ccmc.gsfc.nasa.gov). iSWA is tool to display some of the real-time simulations of the models available at the CCMC - the Community Coordinated Modeling Center https://ccmc.gsfc.nasa.gov/about.php). The CCMC provides access to space science simulations

[4] snopes_com · NASA Reports Magnetosphere Collapse · 2016-04-26
URL: https://www.snopes.com/fact-check/nasa-reports-magnetosphere-collapse/
[Source: snopes_com | NASA Reports Magnetosphere Collapse | 2016-04-26]
Fact Check NASA Reports Magnetosphere Collapse A conspiracy-oriented web site claimed that that Earth's magnetosphere "COLLAPSED" for two hours, but NASA confirmed the incident was a software glitch — and that such fluctuations aren't doomsday events. Kim LaCapria Published April 26, 2016 Claim: Earth's magnetosphere collapsed for two hours on 23 April 2016. Rating: False About this rating On 23 April 2016, unreliable web site Superstation95 reported that data gleaned from NASA revealed that Earth's magnetosphere had "COLLAPSED" for a period of two hours: A stunning and terrifying event has taken place in space surrounding our planet; for two hours today, earth's "Magnetosphere" COLLAPSED around the entire planet! The magnetosphere is what protects earth from solar winds and some radiation. This morning at 01:37:05 eastern US Time, which is 05:37:05 UTC, satellites from the NASA Space Weather Prediction Center detected a complete collapse of earth's magnetosphere! It simply vanished for just over two hours, resuming as normal around 03:39:51 eastern US time, which is 07:39:51 UTC. Here is how NASA Space Weather Satellites recorded the event: This report was false, however. Heliophysics (the study of the effects of the sun on the solar system) is a complex topic, making bombastic claims easier to cloak in a veneer of plausibility. NASA's web site features a brief explainer which notes that doomsday scenarios tend to leverage unfamiliarity with the topic in order to exaggerate normal atmospheric fluctuations: Earth's polarity is not a constant. Unlike a classic bar magnet, or the decorative magnets on your refrigerator, the matter governing Earth's magnetic field moves around. Geophysicists are pretty sure that the reason Earth has a magnetic field is because its solid iron core is surrounded by a fluid ocean of hot, liquid metal. This process can also be modeled with supercomputers. Ours is, without hyperbole, a dynamic planet. The flow of liquid iron in Earth's core creates electric currents, which in turn create the magnetic field. So while parts of Earth's outer core are too deep for scientists to measure directly, we can infer movement in the core by observing changes in the magnetic field. The magnetic north pole has been creeping northward – by more than 600 miles (1,100 km) – since the

[5] science_nasa_gov · Flip Flop: Why Variations in Earth's Magnetic Field Aren't Causing Today's Climate Change - NASA Science · None
URL: https://science.nasa.gov/science-research/earth-science/flip-flop-why-variations-in-earths-magnetic-field-arent-causing-todays-climate-change/
[Source: science_nasa_gov | Flip Flop: Why Variations in Earth's Magnetic Field Aren't Causing Today's Climate Change - NASA Science | ?]
By Alan Buis, NASA's Jet Propulsion Laboratory Earth is surrounded by an immense magnetic field, called the magnetosphere. Generated by powerful, dynamic forces at the center of our world, our magnetosphere shields us from erosion of our atmosphere by the solar wind, particle radiation from coronal mass ejections (eruptions of large clouds of energetic, magnetized plasma from the Sun’s corona into space), and from cosmic rays from deep space. Our magnetosphere plays the role of gatekeeper, repelling these forms of energy that are harmful to life, trapping most of it safely away from Earth’s surface. You can learn more about Earth’s magnetosphere here . Earth is surrounded by a system of magnetic fields, called the magnetosphere. The magnetosphere shields our home planet from harmful solar and cosmic particle radiation, but it can change shape in response to incoming space weather from the Sun. NASA's Scientific Visualization Studio A constant outflow of solar material streams out from the Sun, depicted here in an artist's rendering. This solar wind is always passing by Earth. NASA Goddard's Conceptual Image Lab/Greg Shirah Since the forces that generate our magnetic field are constantly changing, the field itself is also in continual flux, its strength waxing and waning over time. This causes the location of Earth’s magnetic north and south poles to gradually shift, and to even completely flip locations every 300,000 years or so. That might be somewhat important if you use a compass, or for certain animals like birds, fish and sea turtles, whose internal compasses use the magnetic field to navigate . To view this video please enable JavaScript, and consider upgrading to a web browser that supports HTML5 video The Sun unleashed a series of four coronal mass ejections (CMEs) on May 22-24, 2010 as the STEREO Ahead spacecraft watched the action. In the coronagraph images, the Sun, blocked out by an occulting disk (seen as red), is represented by a white disk to show its relative size. CMEs are large solar storms that expel a billion tons of matter at a million miles per hour or more. Credit: NASA/European Space Agency Some people have claimed that variations in Earth’s magnetic field are contributing to current global warming and can cause catastrophic climate change. However, the science doesn’t support that argument. In this blog, we’ll examine a number of proposed hypotheses regarding the effects of changes in Earth’s magnetic field on

Use the evidence above to ground your assessment. Cite chunks by their [id] when relevant. Evidence may be incomplete or off-topic — apply the force-fit guard from the codebook if so.
```
