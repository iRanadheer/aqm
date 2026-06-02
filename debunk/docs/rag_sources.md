# RAG sources

The retrieval knowledge base (`data/raw/kb_combined.jsonl`, 51,035 docs) is
a crawl of 22 climate-science and fact-checking domains. Counts below:

- **Docs** — records in `kb_combined.jsonl` for that `source`.
- **Chunks** — ~500-token windows in `data/rag/chunks.jsonl` (the indexed
  unit; see [rag.md](rag.md)). Two sources are crawled but **excluded from
  chunking** (chunks = 0) so their fact-check verdicts can't leak into the
  evidence a model is scored on.

| Source | Type | Docs | Chunks |
|---|---|---:|---:|
| `snopes_com` | Fact-checker | 31,680 | 0 — *excluded* |
| `wmo_int` | Science agency / IGO | 3,537 | 9,474 |
| `carbonbrief_org` | Journalism | 3,379 | 22,206 |
| `iea_org` | Science agency / IGO | 2,049 | 3,399 |
| `climate_copernicus_eu` | Science agency / IGO | 1,906 | 6,128 |
| `climatefactchecks_org` | Fact-checker | 1,291 | 4,393 |
| `science_org_au` | Science org | 1,176 | 2,352 |
| `theccc_org_uk` | Science agency / IGO | 1,020 | 4,156 |
| `science_nasa_gov` | Science agency / IGO | 833 | 2,400 |
| `politifact_com` | Fact-checker | 696 | 2,138 |
| `skepticalscience_com` | Fact-checker | 618 | 2,166 |
| `science_feedback_org` | Fact-checker | 520 | 0 — *excluded* |
| `futureclimateafrica_org` | Research org | 513 | 554 |
| `factcheck_org` | Fact-checker | 477 | 3,384 |
| `factcheck_afp_com` | Fact-checker | 317 | 722 |
| `eea_europa_eu` | Science agency / IGO | 261 | 368 |
| `berkeleyearth_org` | Science org | 235 | 822 |
| `yaleclimateconnections_org` | Journalism | 216 | 623 |
| `nationalacademies_org` | Science org | 127 | 178 |
| `fullfact_org` | Fact-checker | 97 | 189 |
| `hsph_harvard_edu` | Academic | 46 | 106 |
| `css_umich_edu` | Academic | 41 | 195 |
| **Total** | | **51,035** | **65,953** |

## Notes

- **Excluded from retrieval.** `snopes_com` (31,680 docs, 62% of the KB by
  document count — largely non-climate) and `science_feedback_org` (the
  Climate Feedback fact-checker behind the benchmark's gold labels) are kept
  in `kb_combined.jsonl` but dropped at chunk time. After exclusion the KB is
  carried by climate-science and journalism sources rather than dominated by
  a single fact-check archive.
- **Docs ≠ chunks.** Chunk count scales with document *length*, not just
  document count — long explainers (`carbonbrief_org`) yield ~6.6 chunks per
  doc, short posts far fewer. So `carbonbrief_org` is the largest source in
  the retrievable index (22,206 chunks) despite ranking third by doc count.
- **Source of these counts.** Document counts are `source` tallies over
  `kb_combined.jsonl`; chunk counts are `source` tallies over
  `data/rag/chunks.jsonl`. Re-derive both by grouping each file on its
  `source` field.

## Collection — scrapai-cli

The 22 domains were crawled with
[scrapai-cli](https://github.com/discourselab/scrapai-cli), an AI-assisted
Scrapy wrapper. Key points:

- **AI at build time, not run time.** You describe a target site in plain
  English; an agent analyses it and emits a JSON spider config (CSS/XPath
  selectors + extraction rules). Scrapy then runs that config
  deterministically — no LLM call per page, so re-crawls have no per-page
  model cost. Configs are database rows loaded by a generic
  `DatabaseSpider`, not per-site Python files.
- **Article extraction.** Built on `newspaper4k` + `trafilatura`, pulling
  `title` / `content` / `author` / `published_date` — exactly the fields
  carried in `kb_combined.jsonl`. Non-article pages use custom CSS/XPath
  callbacks.
- **Cloudflare / anti-bot.** This is what let us crawl the protected
  domains. scrapai-cli uses **CloakBrowser** (a patched Chromium) to clear
  Cloudflare Turnstile challenges, with **cookie-cached bypass** — the
  clearance cookie is reused instead of re-launching a browser per request
  (~0.1–0.5 s/page vs 5–10 s). On `403` / `429` it does **smart proxy
  escalation**: direct → datacenter → residential. It handles Cloudflare
  only — *not* DataDome / PerimeterX / Akamai — and does no auth or paywall
  bypass.
- **Incremental + resumable.** DeltaFetch skips already-scraped URLs, and
  crawls checkpoint so they can pause/resume — useful for the large
  multi-thousand-page sources (`snopes_com`, `wmo_int`, `carbonbrief_org`).
- **Output.** Exports CSV / JSON / JSONL / Parquet; we took **JSONL** and
  concatenated the per-source dumps into `kb_combined.jsonl`.

Stack: Scrapy + CloakBrowser/Playwright (rendering), SQLAlchemy/Alembic
(spider + item store), Click (CLI), Pydantic (config validation).
