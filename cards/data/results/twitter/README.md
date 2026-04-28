---
name: Twitter Dataset Notes
description: How the merged Twitter CARDS dataset was assembled from annotool sources
---

# Twitter Dataset

## Source

Annotations were exported from **annotool**, combining two streams:

1. **CARDS chapter** examples
2. **AI-assisted** examples

These two sources were merged into a single dataset.

## Deduplication

Duplicate examples were found between annotators **Travis** and **Mirjam**.

For each duplicate:

- **Kept:** Travis' annotation
- **Dropped:** Mirjam's label

This ensures every example in the merged file appears exactly once, with Travis' label taking precedence on overlapping items.

## Files

- `twitter.jsonl` — source data
- `cards-twitter-merged.jsonl` — merged + deduplicated dataset used downstream
