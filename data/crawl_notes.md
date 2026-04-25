# Crawl Notes

Tracks scope decisions made for the initial crawl. Lives next to `manifest.json`
so any reviewer (and the PDF write-up) can audit *why* the index looks the way
it does.

## Scope

- **Domain**: `goldcard.nat.gov.tw` only.
- **Locale**: English (`/en/` path prefix). Chinese (`/zh/`) and any other
  localized siblings are filtered out at the link-discovery stage.
- **Caps**: 50 pages, depth 3, 1 request/sec, respects `robots.txt`.
- **Seed**: `https://goldcard.nat.gov.tw/en/`.

## Excluded: `talent.nat.gov.tw`

The original spec listed `talent.nat.gov.tw` as a second seed. It was dropped
after the first run for a concrete reason:

- The site is a **client-rendered React SPA**. Static `GET /` returns a 2 KB
  HTML shell with an empty `<body>` and the content is hydrated by JavaScript
  on load.
- `sitemap.xml` returns the same SPA shell instead of an XML sitemap, so even
  bypassing the homepage doesn't yield crawlable URLs.
- Indexing it would require a headless browser (Playwright/Chromium, ~200 MB
  dependency). That is heavy for a Streamlit Cloud demo and out of scope for
  the skill test deliverable.

`goldcard.nat.gov.tw` is the substantive content destination anyway — it is
where the Employment Gold Card eligibility, application steps, fees, tax
exemption, and FAQ pages all live. The Talent Taiwan portal is largely a
pointer/landing site for the same downstream pages.

## Excluded: `/zh/` Chinese mirror

A first crawl pulled in 12 Chinese-language pages that pollute English-only
retrieval. The crawler now enforces `crawl_path_prefix = "/en/"` (configurable
via `Settings.crawl_path_prefix`) at both URL-acceptance and link-enqueue time.

## What this means for the demo

- Retrieval answers questions framed in English, citing official Gold Card
  pages. That is exactly the user the chatbot is built for: foreign
  professionals reading English documentation.
- Production deployment that wants to cover Talent Taiwan's SPA content can
  either: (a) plug a Playwright-based crawler behind the same `Crawler`
  interface, or (b) ingest the underlying CMS feed directly when it becomes
  available. The rest of the pipeline (chunker, embedder, vector store, guards)
  is unchanged either way.
