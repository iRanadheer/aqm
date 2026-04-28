"""Three-level classification prompt.

Level 1: DETECTION — does the text express opposition to, skepticism of, or
dislike for wind energy?
Level 2: FRAME    — which high-level frame(s) is the opposition using?
                    (Codes use the N_* prefix.)
Level 3: CLAIM    — which specific claim(s) are being made?

N_0 and C_0_1 act as "opposition is present but no specific frame / claim
fits" escape hatches under `opposition_detected: true`.
"""

# ---------------------------------------------------------------------------
# Codebooks — hard-coded here for easy editing.
# ---------------------------------------------------------------------------

slim_frames_codebook = """
<N_0> Opposition is present but does not match any specific frame below (vague dislike, sarcasm, or dismissiveness with no clear line of argument)
<N_1> Wind energy is bad for the environment
<N_2> Wind energy is bad for communities
<N_3> Wind energy is too expensive
<N_4> Wind energy is inefficient and unreliable
<N_5> Wind energy is poorly regulated and badly governed
<N_6> Wind energy relies on unethical practices and unreliable information
<N_7> Wind energy benefits foreign countries and reduces energy security
<N_8> Wind energy harms human health and quality of life
""".strip()

full_frames_codebook = """
<N_0>Opposition is present but does not clearly match any specific frame below. Use when the text expresses opposition (including through tone, sarcasm, or dismissiveness) without invoking a recognizable frame.</N_0>
<N_1>Wind energy is bad for the environment or won't help solve environmental problems.</N_1>
<N_2>Wind energy is bad for communities.</N_2>
<N_3>Wind energy is too expensive. It has negative economic impacts and costs too much to produce.</N_3>
<N_4>Wind energy is inefficient, unreliable, and can't meet demand.</N_4>
<N_5>Wind energy development is poorly regulated and badly governed. Poor governance often stems from inappropriate or harmful policies.</N_5>
<N_6>Wind energy development relies on unethical practices and/or unreliable information.</N_6>
<N_7>Wind energy benefits foreign countries, creates national security risks, reduces energy security, and leads to geopolitical challenges.</N_7>
<N_8>Wind energy negatively impacts human health and quality of life.</N_8>
""".strip()

slim_claims_codebook = """
<C_0_1> Opposition is present but does not match any specific claim. Use when the text's opposition is purely tonal (sarcasm, rhetorical question, dismissive joke, brief jab, cryptic fragment) and no concrete argument survives adjudication. If a specific claim is supported by the actual content (not just tone), pick that instead.
<C_1_0> Wind energy threatens wildlife, ecosystems, and biodiversity
<C_1_1> Wind turbines harm bird and bat populations
<C_1_2> Offshore wind harms marine life
<C_2_0> Wind energy will not help meet climate change goals
<C_2_1> Wind turbine manufacturing releases more CO2 than fossil fuels
<C_2_2> Wind increases fossil fuel use OR fails to displace fossils. Triggers: causal ("wind requires gas backup, raising CO2"); displacement failure ("wind hasn't replaced coal", "emissions unchanged despite wind"). If text says wind INCREASES or FAILS TO REDUCE emissions → C_2_2 (not C_2_0). Rescue framing ("coal fires up when wind fails") → C_28_5.
<C_2_3> Wind turbines cause CO2 emissions via habitat for CO2-releasing species
<C_3_0> Wind turbines are unsustainable, generate waste and pollution, and are not recyclable
<C_4_0> Wind turbines take up too much space/land — applies when wind's land footprint is framed as excessive (acres required, landscape blanketed, "energy sprawl"). NOT C_4_0: neutral cost accounting that lists land as one factor; "all energy infrastructure needs land" arguments that don't single out wind.
<C_5_0> Communities oppose wind energy development
<C_6_0> Wind turbine noise pollution harms residents' health and liveability
<C_7_0> Wind turbine shadow flicker harms residents and can trigger seizures
<C_8_0> Wind turbine vibration disturbances harm nearby residents
<C_9_0> Wind energy causes environmental hazards posing health risks to residents
<C_10_0> Electromagnetic radiation from wind turbines threatens health and animals
<C_11_0> Wind energy infrastructure degrades scenic landscapes and causes visual pollution
<C_12_0> Wind turbines pose safety hazards — debris, structural instability, and fire risks
<C_13_0> Wind turbines devalue nearby properties and inhibit residential development
<C_14_0> Wind energy development causes land dispossession and undermines landowner rights
<C_15_0> Wind energy developers fail to engage in fair community consultation
<C_16_0> Wind turbines negatively impact recreational activities
<C_17_0> Wind energy development threatens local culture, identity, and community character
<C_17_1> Wind farms damage cultural sites and historical artifacts
<C_17_2> Wind projects create divisions in communities
<C_17_3> Indigenous communities are negatively impacted by wind energy development
<C_18_0> Wind energy negatively impacts critical infrastructure
<C_18_1> Wind energy projects cause road disruptions, damage, and traffic increases
<C_19_0> Wind energy hurts the poor
<C_20_0> Wind/renewable policy raises costs or prices that impact consumers, households, or end-users. Core test: who bears the cost? Households/ratepayers/buyers → C_20_0. Project capital or LCOE → C_27_0. Taxpayer/municipal burden → C_21_0. Can co-occur.
<C_21_0> Wind energy places financial burdens on taxpayers and municipalities
<C_22_0> Wind energy benefits only a select few — developers, financiers, and landowners
<C_23_0> Wind energy is bad for jobs — causes job losses or fails to create employment. Includes outsourcing (green-jobs funding → foreign companies) and cases where renewable policy is cited as a cause of job losses elsewhere (e.g., coal/fossil-plant closures driven by renewable incentives).
<C_24_0> Wind depends on subsidies / government handouts. Applies whenever the text uses subsidies to frame wind negatively — explicit critique ("corporate welfare", "handouts"), subtle jabs ("subsidy-hungry backers"), or PTC-as-distortion language. NOT C_24_0: neutral listings of publicly-funded projects or factual PTC descriptions without negative framing.
<C_25_0> Wind energy disrupts industries
<C_25_1> Wind energy harms the tourism industry
<C_25_2> Wind energy harms agriculture and is bad for farmers
<C_25_3> Offshore wind hurts the fishing industry
<C_25_4> Wind energy disrupts maritime operations and poses navigational hazards
<C_25_5> Wind energy conflicts with mineral rights and oil/gas extraction
<C_26_0> Wind is unreliable or can't meet demand — intermittency, low capacity factor, grid instability, or scale-vs-demand comparisons framed as inadequate. NOT C_26_0: "wind hasn't replaced fossils" (→ C_2_2 if fossils named), cost (→ C_20_0/C_27_0), or vague dismissals without a concrete reliability/output claim.
<C_27_0> Wind projects cost too much to build — project-level capital/development costs. Triggers: levelized cost, capex, project payback period, development cost, commercial feasibility, construction cost. Not consumer bills (C_20_0) or subsidies (C_24_0). Can co-occur with C_20_0.
<C_28_0> Better energy alternatives exist — unspecified. Triggers: "conventional power", "reliable/existing power plants", "economic energy sources", or explicit "alternative forms of energy" contrasted with wind. The unnamed alternative must be an energy SOURCE, not a non-energy strategy/policy. Specific source named → C_28_1–C_28_5.
<C_28_1> Onshore wind is preferable to offshore wind
<C_28_2> Offshore wind is preferable to onshore wind
<C_28_3> Other renewables are preferable to wind energy
<C_28_4> Nuclear energy is preferable to wind. Triggers: explicit favorable comparison ("nuclear has half the capital cost of wind"), or advocacy to include nuclear instead of a wind-only mandate. NOT C_28_4: nuclear mentioned as one baseload option (→ C_26_0), or as a data point in a neutral cost/emissions comparison.
<C_28_5> Fossil fuels are preferable to wind. Triggers: explicit ("gas is the best backup", "coal is more reliable than wind"); rescue framing ("coal fired up when wind failed"); dismissal of the replacement premise ("delusional to think wind can replace fossils"). NOT C_28_5: fossils as one baseload option, "reliable generation" without naming fossils (→ C_26_0), or fossils named as a policy's target ("the plan bans fossils").
<C_29_0> Wind energy projects are irresponsibly or inappropriately sited
<C_30_0> Regulation is too LAX on wind. Two senses: (a) projects slip existing requirements/monitoring/enforcement; (b) the regulatory regime itself fails to set up appropriate requirements to prevent wind-related harms (bird kills, ecological damage, etc.). NOT C_30_0: complaints that regulation is too burdensome on wind developers or slows renewable deployment.
<C_31_0> Wind energy projects violate existing laws and zoning regulations
<C_32_0> Proponents use unreliable information, deception, unethical practices, or corruption to push wind/renewable energy. Covers: misleading or insufficient impact studies; understated costs/impacts; withholding information; over-promising benefits; ignoring unfavorable evidence; coercive tactics. Direction matters: accused = PROPONENTS (wind industry, advocates, sympathetic officials) — not the author being smeared.
<C_33_0> Wind/renewable policy used to restrict individual rights, freedoms, or market/private choice — via government overreach or proponents pushing the policy as a vehicle for excessive regulation. Key test: is the policy's LEGITIMACY attacked, or only its effects (cost/reliability/jobs)? Only the former is C_33_0. Triggers: rights/freedom/liberty language, "unconstitutional", "overreach", "arbitrary", "surreptitiously", "backdoor", "force us to accept", "regardless of cost(s)".
<C_34_0> Renewable energy mandates impose colonialist restrictions on developing nations
<C_35_0> Wind energy interferes with radar, aviation, and communication systems
<C_36_0> Wind energy projects harm military interests
<C_37_0> Wind energy increases foreign dependency and reduces energy security
<C_38_0> Wind energy benefits foreign companies and governments
""".strip()

full_claims_codebook = """
<C_0_1>Opposition is present but does not match any specific claim. Use when the text's opposition is purely tonal — sarcasm, rhetorical questions, dismissive jokes, brief jabs, cryptic fragments — and no concrete argument survives adjudication. Typical markers: sarcastic tone without a substantive claim, rhetorical questions that imply dismissal, snarky one-liners, incomplete fragments, and ironic or mock-serious phrasing. If a specific claim is actually supported by the content (not just by tone), pick that instead. Do not force-fit vague opposition into a specific code.</C_0_1>
<C_1_0>Wind energy threatens wildlife, ecosystems, and biodiversity (general — not birds/bats or marine life specifically).</C_1_0>
<C_1_1>Wind turbines harm bird and/or bat populations.</C_1_1>
<C_1_2>Offshore wind harms marine life.</C_1_2>
<C_2_0>Wind energy will not help meet climate change goals (general climate ineffectiveness).</C_2_0>
<C_2_1>Wind turbine manufacturing, transportation, and infrastructure release more CO2 than fossil fuels.</C_2_1>
<C_2_2>Wind energy increases fossil fuel use OR fails to displace fossils after deployment. Two valid triggers: (a) causal — "wind requires gas backup, which increases CO2 emissions"; (b) displacement failure — "wind has not replaced a single coal plant", "emissions stagnant despite wind rollout". Disambiguation: C_2_0 is abstract "won't help climate goals"; if text says wind INCREASES or FAILS TO REDUCE emissions, prefer C_2_2. Rescue framing ("coal fires up when wind fails") → C_28_5, not C_2_2. C_2_2 and C_28_5 can co-occur.</C_2_2>
<C_2_3>Wind turbines cause CO2 emissions by providing habitat or conditions for CO2-releasing species (e.g., peat disturbance).</C_2_3>
<C_3_0>Wind turbines are unsustainable — they generate waste, pollution, contaminate water, and/or are not recyclable.</C_3_0>
<C_4_0>Wind turbines take up too much space and land. Applies when the text frames wind's land footprint as excessive, wasteful, or consuming too many acres — e.g., "millions of acres required", "habitats blanketed by turbines", "energy sprawl", "forests sacrificed". NOT C_4_0: neutral cost accounting that lists "land" as one factor alongside others; "all energy infrastructure requires land" arguments that don't single out wind.</C_4_0>
<C_5_0>Communities oppose wind energy. The text states that communities, the public, or residents are against wind development.</C_5_0>
<C_6_0>Wind turbine noise pollution harms residents' health and liveability. Includes low-frequency noise, sleep disruption, "wind turbine syndrome."</C_6_0>
<C_7_0>Wind turbine shadow flicker harms residents. Can trigger seizures in people with epilepsy.</C_7_0>
<C_8_0>Wind turbine vibration disturbances harm nearby residents.</C_8_0>
<C_9_0>Wind energy causes environmental hazards posing health risks to nearby residents.</C_9_0>
<C_10_0>Electromagnetic radiation from wind turbines and cables threatens health and/or animals.</C_10_0>
<C_11_0>Wind energy infrastructure degrades scenic and natural landscapes — visual pollution.</C_11_0>
<C_12_0>Wind turbines pose safety hazards — debris, structural instability, fire risks to nearby communities.</C_12_0>
<C_13_0>Wind turbines devalue nearby properties and/or inhibit residential development.</C_13_0>
<C_14_0>Wind energy development causes land dispossession and undermines landowner rights.</C_14_0>
<C_15_0>Wind energy developers fail to engage in fair community consultation; community concerns are overlooked.</C_15_0>
<C_16_0>Wind turbines negatively impact recreational activities (boating, fishing, yachting, etc.).</C_16_0>
<C_17_0>Wind energy threatens local culture, identity, and/or community character (general).</C_17_0>
<C_17_1>Wind farms damage cultural sites and historical artifacts.</C_17_1>
<C_17_2>Wind projects create divisions in communities between beneficiaries and non-beneficiaries.</C_17_2>
<C_17_3>Indigenous communities are negatively impacted by wind energy development.</C_17_3>
<C_18_0>Wind energy negatively impacts critical infrastructure.</C_18_0>
<C_18_1>Wind energy projects cause road disruptions, damage, traffic increases due to oversized components or poor planning.</C_18_1>
<C_19_0>Wind energy hurts the poor. Includes energy poverty and starvation claims.</C_19_0>
<C_20_0>Wind/renewable policy raises costs or prices that impact consumers, households, or end-users. Core test: who bears the cost? If households / ratepayers / buyers / families / the poor → C_20_0. If project capital or LCOE (supply-side cost of producing electricity) → C_27_0. If taxpayers / municipalities → C_21_0. If the text describes subsidy dependence → C_24_0. C_20_0 can co-occur with C_27_0, C_21_0, and C_24_0 when a text cites multiple cost dimensions — always ask the "who bears it?" question per dimension.</C_20_0>
<C_21_0>Wind energy places financial burdens on taxpayers and municipalities, including reduced tax revenue.</C_21_0>
<C_22_0>Wind energy benefits only a select few — developers, financiers, and/or landowners profit disproportionately. Includes "producers get far more" or wealth transfer to elites.</C_22_0>
<C_23_0>Wind energy is bad for jobs — fails to create employment or causes job losses. Includes outsourcing ("green-jobs funding went to foreign companies", "subsidies outsourced american jobs") and cases where renewable policy is cited as a cause of job losses in other sectors (e.g., coal/fossil-plant closures driven by renewable mandates or incentives).</C_23_0>
<C_24_0>Wind energy depends on subsidies or government handouts. Applies whenever the text uses subsidies to frame wind negatively — explicit critique ("corporate welfare", "handouts"), subtle jabs that deploy subsidy language as a put-down ("subsidy-hungry backers"), or PTC-as-distortion framing ("PTCs make wind seem less expensive than it is"). NOT C_24_0: neutral listings of publicly-funded projects, factual PTC descriptions, or texts that argue subsidies are justified (pro-wind → detection=false).</C_24_0>
<C_25_0>Wind energy disrupts industries (general — not a specific industry below).</C_25_0>
<C_25_1>Wind energy harms the tourism industry.</C_25_1>
<C_25_2>Wind energy harms agriculture and farmers.</C_25_2>
<C_25_3>Offshore wind hurts the fishing industry.</C_25_3>
<C_25_4>Wind energy disrupts maritime operations and poses navigational hazards.</C_25_4>
<C_25_5>Wind energy conflicts with mineral rights and oil/gas extraction.</C_25_5>
<C_26_0>Wind energy is unreliable or cannot meet demand. Triggers: intermittency, low capacity factor, grid instability, blackouts attributed to wind, or scale-vs-demand comparisons framed as inadequate (e.g., "16.9 GW installed vs 80 GW peak demand"). NOT C_26_0: "wind hasn't displaced fossils" (→ C_2_2 if fossils named), cost (→ C_20_0/C_27_0), or vague dismissals ("unrealizable", "can't scale") without a concrete reliability claim.</C_26_0>
<C_27_0>Wind projects cost too much to build and are economically unviable. About project-level capital and development costs — not consumer bills (C_20_0) or subsidies (C_24_0). Triggers: levelized cost of energy (LCOE), capex, project payback period, construction/development cost, commercial feasibility, "unaffordable to build". C_27_0 and C_20_0 CAN co-occur when a text cites both project costs AND consumer prices — do not pick only one.</C_27_0>
<C_28_0>Better energy alternatives exist — used when no specific alternative is named. Triggers: "conventional power", "reliable/existing power plants", "economic energy sources", or explicit "alternative forms of energy" contrasted with wind. The unnamed alternative MUST refer to an energy source or generation method — "other strategies/approaches" referring to advocacy, policy, or communications is NOT C_28_0. If a specific source IS named (solar, nuclear, fossils, onshore, offshore), use C_28_1–C_28_5 instead.</C_28_0>
<C_28_1>Onshore wind is preferable to offshore wind.</C_28_1>
<C_28_2>Offshore wind is preferable to onshore wind.</C_28_2>
<C_28_3>Other renewables (solar, hydro, geothermal, bioenergy) are preferable to wind.</C_28_3>
<C_28_4>Nuclear energy is preferable to wind energy. Triggers: explicit favorable comparison ("nuclear is cheaper/cleaner/more reliable than wind", "reactor has half the capital cost of wind per MW"), or advocacy to include nuclear in the energy mix against a wind-only mandate ("nuclear will not be allowed… nuclear is safe and affordable"). NOT C_28_4: nuclear mentioned as one baseload option among others, or as a data point in a neutral cost/emissions comparison.</C_28_4>
<C_28_5>Fossil fuels are preferable to wind energy. Requires a favorable comparison: (a) explicit — "natural gas is the best backup", "coal is cheaper/more reliable than wind"; (b) rescue framing — "coal fired up when wind farms failed"; (c) dismissal of the replacement premise — "delusional to think wind can rid the world of fossil fuels". NOT C_28_5: fossils mentioned as one baseload option, "reliable generation" without naming fossils (→ C_26_0), or fossils named as the target of a policy ("the plan bans fossils"). Sector-neutral reliability talk → C_26_0.</C_28_5>
<C_29_0>Wind energy projects are irresponsibly or inappropriately sited.</C_29_0>
<C_30_0>Regulation is too lax on wind. Two senses: (a) wind projects slip or evade existing requirements, monitoring, or enforcement; (b) the regulatory regime itself has failed to set up appropriate requirements to prevent wind-related harms (e.g., "the policy question of how much wildlife killing is enough is not being asked"). Direction matters: C_30_0 is about regulation being INSUFFICIENT to constrain wind. NOT C_30_0: complaints that regulation is too burdensome on wind developers, or that bureaucratic friction slows renewable deployment — those are pro-industry/pro-deployment complaints, not wind opposition.</C_30_0>
<C_31_0>Wind energy projects violate existing laws and zoning regulations.</C_31_0>
<C_32_0>Proponents use unreliable information, deception, unethical practices, or corruption to push wind/renewable energy. Specific manifestations: (1) impact studies that are insufficient, inaccurate, or misleading — including critiques that a study understated or omitted costs/impacts, regardless of who conducted it; (2) withholding critical information from the public; (3) deliberately overstating project benefits or "over-promising"; (4) ignoring unfavorable evidence; (5) coercive tactics to suppress opposition or secure approvals. Direction matters: the accused must be PROPONENTS (wind industry, advocates, sympathetic officials/agencies) — descriptions of the author or critics being smeared do NOT trigger C_32_0. Mere disagreement with wind advocates is also not C_32_0.</C_32_0>
<C_33_0>Wind/renewable policy used to restrict individual rights, freedoms, or market/private choice — whether via government overreach (the state exceeding its legitimate authority) or via proponents pushing the policy as a vehicle for excessive regulation. Covers regulatory overreach too: agencies enacting rules beyond the original intent of the underlying legislation. Key test: does the text attack the LEGITIMACY of the policy (overstepping, arbitrary, coercive, circumventing process), or only its EFFECTS (cost, reliability, jobs)? Only legitimacy attacks are C_33_0. Triggers: rights/freedom/liberty language; "unconstitutional", "overreach", "arbitrary", "surreptitiously", "backdoor", "force us to accept", "regardless of cost(s)". Bare "mandate" or "forced" describing the mechanism behind a cost/reliability/jobs critique is NOT C_33_0.</C_33_0>
<C_34_0>Renewable energy mandates impose colonialist restrictions on developing nations.</C_34_0>
<C_35_0>Wind energy interferes with radar, aviation, communication systems, and/or emergency helicopter operations.</C_35_0>
<C_36_0>Wind energy projects harm military interests.</C_36_0>
<C_37_0>Wind energy increases foreign dependency and reduces energy security.</C_37_0>
<C_38_0>Wind energy benefits foreign companies and governments.</C_38_0>
""".strip()


# ---------------------------------------------------------------------------
# Instruction template — three-level cascade.
# ---------------------------------------------------------------------------

_instruction_template = """You are an expert annotator classifying texts for opposition to wind energy. You will perform a three-level classification: (1) detect whether the text expresses opposition, (2) if so, identify the frame(s) used, and (3) identify the specific claim(s) made. Frames and claims are both multi-label.

### FRAMES CODEBOOK:
{frames_codebook}

### CLAIMS CODEBOOK:
{claims_codebook}

### INSTRUCTIONS:

1. **Detection (level 1)**:
   - Set `opposition_detected: true` if the text expresses opposition to, skepticism of, or dislike for wind energy (onshore or offshore). Sarcasm, dismissiveness, and tonal opposition all count — the argument need not be explicit.
   - Set `opposition_detected: false` for neutral reporting, factual data, pro-wind advocacy, or meta-commentary about the debate.
   - When `opposition_detected` is `false`, `frames` and `claims` must be empty lists (`[]`).

IF `opposition_detected` is `true`, then perform the following steps:

2. **Frames (level 2)** — only if opposition is detected:
   - A frame is the high-level line of argument the speaker uses. Assign every frame (N_1–N_8) the text actively invokes.
   - Multiple frames are allowed. Do not invent a frame that the text does not support.
   - If opposition is present but no specific frame (N_1–N_8) clearly applies, use **N_0** alone. Do not force-fit.

3. **Claims (level 3)** — only if opposition is detected:
   - A claim is the specific assertion made within a frame. Assign every claim the text actively argues or endorses.
   - If opposition is present but no specific claim clearly applies (e.g., tonal-only opposition, vague skepticism), use **C_0_1** alone. Do not force-fit.
   - The claims codebook is hierarchical. C_X_0 is the parent; C_X_1+ are subclaims. Use the most specific subclaim that fits. Use the parent ONLY when no subclaim applies.
   - If the text says "alternatives are better" without naming one → parent C_28_0. If it names nuclear → C_28_4, fossil fuels → C_28_5, etc.
   - Parent→subclaim map (all other C_X_0 codes have NO subclaims):
     C_1_0 → C_1_1, C_1_2
     C_2_0 → C_2_1, C_2_2, C_2_3
     C_17_0 → C_17_1, C_17_2, C_17_3
     C_18_0 → C_18_1
     C_25_0 → C_25_1, C_25_2, C_25_3, C_25_4, C_25_5
     C_28_0 → C_28_1, C_28_2, C_28_3, C_28_4, C_28_5

4. **Endorsement, not description**:
   - Only classify frames/claims the text actively argues or endorses.
   - Meta-commentary or criticism of anti-wind arguments is not opposition (`opposition_detected: false`).

5. **Precision and recall**:
   - Do not leave any relevant frame or claim unassigned.
   - Do not assign any irrelevant frame or claim.

### OUTPUT FORMAT:
Reason inside <think> tags using the following chain. Every step is mandatory when reached — do not skip.

<think>
1. CONTEXT
   One-line summary of the text: type (news, comment, forum post, council minutes, etc.), tone, and overall intent.

2. DETECTION
   a. Quotes: Direct quotes from the text that carry the relevant signal. No paraphrasing.
   b. Reasoning: One or two lines classifying whether the text expresses opposition to, skepticism of, or dislike for wind energy.
   c. Decision: true / false.

   → If `false`, stop reasoning and emit the YAML with empty `frames: []` and `claims: []`.
   → If `true`, continue to step 3.

3. FRAMES
   a. Shortlist: List every frame (N_1–N_8) that could plausibly apply.
   b. Adjudicate: For each shortlisted frame, one line — "[N_X]: KEEP — [brief reason]" or "[N_X]: REMOVE — [brief reason]".
   c. Fallback: If no frame survives adjudication, the frame list is [N_0].

4. CLAIMS
   a. Shortlist: For each KEPT frame, list every claim that could plausibly apply under it.
   b. Adjudicate: For each shortlisted claim, one line — "[C_X_Y]: KEEP — [brief reason]" or "[C_X_Y]: REMOVE — [brief reason]".
   c. Granularity: For each KEPT parent claim (C_X_0 with subclaims), confirm whether a subclaim fits better. If so, replace the parent with the subclaim.
   d. Force-fit check: if every specific claim you are about to keep is supported ONLY by tone, sarcasm, a rhetorical question, or a cryptic fragment — with no substantive argument on the specific topic — drop them and use [C_0_1] alone. Opposition via tone is C_0_1, not a specific claim.

5. SPECIAL CONSIDERATIONS
   One to three lines walking through the relevant hints for this text:
   - Cost/price language — ask "who bears the cost?" for each dimension present:
     consumers/households/end-users/families/the poor → C_20_0; project capital or LCOE → C_27_0; taxpayers/municipalities → C_21_0; subsidy dependence (negative framing) → C_24_0; select few benefit → C_22_0. Multiple can co-occur.
   - Proponent-misconduct framing (C_32_0) — check whether the text accuses wind/renewable proponents of any of: (a) manipulated or insufficient studies, (b) financial corruption / rent-seeking / crony arrangements, (c) political favoritism or market manipulation, (d) withholding information, (e) over-promising benefits, (f) coercive tactics. Direction matters — the accused must be proponents (industry, advocates, sympathetic officials), not the author being smeared. Mere disagreement with wind advocates is not C_32_0.
   State any final adjustments, then list the final frame and claim codes.
</think>

```yaml
opposition_detected: <true|false>
frames:
  - <frame_code>
claims:
  - <claim_code>
```

STRICT RULES:
- All reasoning must be inside <think> tags. Nothing after </think> except the YAML block.
- When `opposition_detected` is `false`, `frames` and `claims` must be empty lists.
- When `opposition_detected` is `true`, both `frames` and `claims` must be non-empty. If no specific frame fits, use `[N_0]`; if no specific claim fits, use `[C_0_1]`.
- Be concise. No second-guessing. Single pass. Adjudication entries (KEEP/REMOVE) must be one line each.
"""


# Slim codebooks — for fine-tuned small models
slim_system_instruction = _instruction_template.format(
    frames_codebook=slim_frames_codebook,
    claims_codebook=slim_claims_codebook,
)

# Full codebooks — for big models (Opus, etc.)
full_system_instruction = _instruction_template.format(
    frames_codebook=full_frames_codebook,
    claims_codebook=full_claims_codebook,
)

# Default export
system_prompt = full_system_instruction
