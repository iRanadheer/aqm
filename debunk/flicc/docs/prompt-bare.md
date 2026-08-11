# Prompt: `bare`

System prompt as sent, 567 chars. Regenerate with `python prompts.py bare`.

```
Please classify a piece of text into the following categories of logical fallacies: ad hominem, anecdote, cherry picking, conspiracy theory, fake experts, false choice, false equivalence, impossible expectations, misrepresentation, oversimplification, single cause, slothful induction.

### OUTPUT FORMAT:
Output only a YAML block naming the fallacy. No reasoning, no commentary, no other text.

```yaml
fallacy: <label>
```

STRICT RULES:
- `fallacy` must be exactly one of the twelve labels, lowercase, spelled as in the codebook.
- Nothing outside the YAML block.

```

## User message

```
### Text:
{text}
```
