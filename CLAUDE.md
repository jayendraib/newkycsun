# CLAUDE.md — newkycsummary

This project is a **Postgres-backed FIFO queue system** that processes two types of jobs:
- **KYC** — ZIP files containing identity documents (Aadhaar, PAN), classified via YOLO + LLM
- **Summary** — PDF URLs downloaded and summarised via LLM

Both jobs come in through a FastAPI API, which does nothing but insert the job into
Postgres (`queue_items` table) and return immediately — no blocking, no API timeout.
Each job type has its own dedicated worker thread that continuously pulls the oldest
pending row for that type (first in, first out) and runs it, so KYC and Summary
process concurrently and neither one starves the other.

---

## File Map

| File | Purpose |
|------|---------|
| `queue_system.py` | Pure FIFO queue engine (Postgres-backed) — knows nothing about KYC or Summary |
| `app.py` | Wiring — registers processors, creates FastAPI endpoints, webhook post-back |
| `kyc.py` | KYC processor: YOLO face detection + Ollama multimodal classification |
| `summary.py` | Summary processor: PDF download + LangChain map-reduce summarisation |
| `db.py` | PostgreSQL backend — `queue_items` (the job queue itself) + `kyc_logs`, `summary_logs`, `queue_logs` (logs) |
| `flowchart.html` | Visual flowchart of the full system pipeline |
| `requirements.txt` | All Python dependencies |

---

## Running

```bash
# Install dependencies
pip install -r requirements.txt

# Run the API server
uvicorn app:app --reload --port 8000
```

---

## API Endpoints

### Submit jobs

```
POST /api/kyc/document/processor/   { "bucket_name": "...", "aws_region": "ap-south-1", "zip_s3_path": "s3://bucket/key.zip" }
                                     header: auth-id: 2f8f114c-e61e-41dc-b158-f6f25a121006
POST /api/kyc/scan                  (no body, same auth-id header) — scan s3://ib-prod-ekyc/CVLKRA/ and enqueue all unprocessed ZIPs
POST /api/sum                       { "url": "https://example.com/file.pdf", "news_id": 123 }
GET  /status                        — current queue depth + failed count per processor
GET  /health
```

`/api/kyc/scan` uses `KYC_INPUT_BUCKET` / `KYC_INPUT_PREFIX` env vars to locate ZIPs. Already-processed ZIPs (detected via `_already_processed()`) are automatically skipped.

Summary results are posted back to a webhook (`SUMMARY_WEBHOOK_URL`) with:
```json
{ "NewsId": 123, "AiSummary": "...", "Error": null, "StatusCode": 200 }
```

---

## Queue Architecture

**Problem solved:** When one service receives continuous input, a single shared queue
causes the other service to never run (starvation).

**Solution:** every job type gets its own Postgres-backed FIFO queue and its own
dedicated worker thread. No slots, no priority, no round-robin — plain FIFO per queue.

```
queue_items (Postgres table)
    processor_name='kyc'     → KYC worker thread     → oldest pending first → kyc.py
    processor_name='summary' → Summary worker thread → oldest pending first → summary.py
```

Both threads run continuously and concurrently — 10,000 queued KYC jobs never block
Summary, because Summary has its own thread pulling from its own slice of the table.

Jobs are claimed atomically (`UPDATE ... WHERE item_id = (SELECT ... FOR UPDATE SKIP LOCKED)`),
so a job is only ever picked up by one worker. If the process crashes mid-job, any row
stuck in `processing` is reset to `pending` on the next startup — nothing is lost.

---

## Adding a New Processor

**Step 1** — Write a handler in any file:
```python
def ocr_handler(data: dict) -> dict:
    return run_ocr(data["file_path"])
```

**Step 2** — Register it in `app.py` (one line):
```python
queue.register_processor("ocr", ocr_handler)
```

**Step 3** — Add an endpoint in `app.py`:
```python
@app.post("/ocr")
def submit_ocr(req: OcrRequest):
    item_id = queue.enqueue("ocr", {"file_path": req.file_path})
    return {"item_id": item_id, "status": "queued"}
```

Nothing else changes. The queue engine picks it up automatically.

---

## Adding a New Input Source

Any code can push to the queue by calling:
```python
queue.enqueue("kyc",     {"zip_path": "s3://bucket/file.zip"})
queue.enqueue("summary", {"url": "https://example.com/doc.pdf"})
```

This is safe to call from any thread.

---

## Logs

Each processor still writes a colour-coded console stream, but persistent
logs and errors go to **PostgreSQL** (`db.py`) — no local `.log` files, no
Slack alerts. Each source has its own table:

| Stream | Colour | Postgres Table |
|--------|--------|-----------------|
| `[QUEUE]` | Yellow | `queue_logs` |
| `[KYC]` | Cyan | `kyc_logs` |
| `[SUMMARY]` | Magenta | `summary_logs` |

The Postgres connection is configured via env vars (see below) and is
optional — if the DB is unreachable, logging falls back to console-only so
the pipeline never crashes on its logging backend.

Console example:
```
[10:42:01][QUEUE][INFO] Registered 'kyc'  pending_on_disk=847
[10:42:01][KYC  ][INFO] START kyc_104201  [job #1]  waited 0.12s
[10:42:44][KYC  ][INFO] DONE  kyc_104201  |  42.3s
[10:43:28][SUMM ][INFO] START summary_104301  [job #1]  waited 3.10s
[10:43:44][SUMM ][INFO] DONE  summary_104301  |  15.2s
```

---

## KYC Pipeline (`kyc.py`)

```
ZIP input (local or S3)
    ↓
_already_processed() check — skip if CVLKRA_AI/<zip_name>/ already in S3
    ↓
Extract PDFs from ZIP (handles nested ZIPs)
    ↓
Select latest valid PDF (by date pattern in filename)
    ↓
Convert PDF pages to images (pdf2image, 300 DPI)
    ↓
YOLO face detection (yolov8n-face.pt) — finds best face crop
    ↓
LLM classification per page (qwen3-vl:8b via Ollama, num_ctx=32000)
    → aadhaar / pan / uncertain
    ↓
Keep highest-confidence image per document type
    ↓
Upload classified images to S3 + verify upload
    ↓
Failure recorded to Postgres (kyc_logs)
    ↓
3-second pause before next ZIP (inter-ZIP rate limiting)
```

Key classes:
- `SmartAadharDetector` — main processor class
- `ZipProcessingContext` — context manager tracking ZIP lifecycle stages (ENTRY → EXTRACTION → PROCESSING → S3_UPLOAD → VERIFICATION → CLEANUP)
- `record_failure()` — persists a structured failure to the `kyc_logs` table (replaces the old Slack alerting)

---

## Summary Pipeline (`summary.py`)

```
PDF URL
    ↓
download_pdf()   — async, retries 3×, streams to temp file
    ↓
load_and_chunk() — PyPDFLoader + RecursiveCharacterTextSplitter (2500 chars)
    ↓
summarize_docs() — LangChain map_reduce chain (qwen3-vl:8b via Ollama, num_ctx=32000)
    ↓
Returns clean summary string
    ↓
POST to SUMMARY_WEBHOOK_URL with { NewsId, AiSummary, Error, StatusCode }
```

**Model change:** was `gemma3:4b` with `num_ctx=100000`; now `qwen3-vl:8b` with `num_ctx=32000` — same model as KYC.

---

## LLM Models Required (Ollama)

```bash
ollama pull qwen3-vl:8b    # KYC classification + PDF summarisation (both use this model)
```

Both KYC and Summary now use `qwen3-vl:8b` with `num_ctx=32000`. `gemma3:4b` is no longer used.

---

## Environment Variables

| Variable | Used By | Notes |
|----------|---------|-------|
| `KYC_OUTPUT_FOLDER` | `app.py` | Local output path, default `/tmp/kyc_output` |
| `KYC_S3_BUCKET` | `app.py` | S3 bucket for KYC uploads, default `ib-prod-ekyc` |
| `KYC_INPUT_BUCKET` | `app.py` | S3 bucket to scan for input ZIPs (`/api/kyc/scan` endpoint) |
| `KYC_INPUT_PREFIX` | `app.py` | S3 prefix to scan for input ZIPs (e.g. `CVLKRA/`) |
| `AWS_ACCESS_KEY_ID` | `app.py` | AWS credentials |
| `AWS_SECRET_ACCESS_KEY` | `app.py` | AWS credentials |
| `AWS_REGION` | `app.py` | AWS region, default `ap-south-1` |
| `SUMMARY_WEBHOOK_URL` | `app.py` | Webhook to POST summary results back to India Bonds API |
| `SUMMARY_WEBHOOK_TOKEN` | `app.py` | Bearer token for the summary webhook |
| `OPENAI_API_KEY` | `kyc.py` | Optional, loaded but not used directly |
| `DATABASE_URL` | `db.py` | Full Postgres DSN (overrides the `PG_*` vars below). Required — this is where jobs *and* logs live |
| `PG_HOST` | `db.py` | Postgres host, default `localhost` |
| `PG_PORT` | `db.py` | Postgres port, default `5432` |
| `PG_DATABASE` | `db.py` | Database name, default `newkycsummary` |
| `PG_USER` | `db.py` | Postgres user, default `postgres` |
| `PG_PASSWORD` | `db.py` | Postgres password, default empty |

<!-- code-review-graph MCP tools -->
## MCP Tools: code-review-graph

**IMPORTANT: This project has a knowledge graph. ALWAYS use the
code-review-graph MCP tools BEFORE using Grep/Glob/Read to explore
the codebase.** The graph is faster, cheaper (fewer tokens), and gives
you structural context (callers, dependents, test coverage) that file
scanning cannot.

### When to use graph tools FIRST

- **Exploring code**: `semantic_search_nodes` or `query_graph` instead of Grep
- **Understanding impact**: `get_impact_radius` instead of manually tracing imports
- **Code review**: `detect_changes` + `get_review_context` instead of reading entire files
- **Finding relationships**: `query_graph` with callers_of/callees_of/imports_of/tests_for
- **Architecture questions**: `get_architecture_overview` + `list_communities`

Fall back to Grep/Glob/Read **only** when the graph doesn't cover what you need.

### Key Tools

| Tool | Use when |
| ------ | ---------- |
| `detect_changes` | Reviewing code changes — gives risk-scored analysis |
| `get_review_context` | Need source snippets for review — token-efficient |
| `get_impact_radius` | Understanding blast radius of a change |
| `get_affected_flows` | Finding which execution paths are impacted |
| `query_graph` | Tracing callers, callees, imports, tests, dependencies |
| `semantic_search_nodes` | Finding functions/classes by name or keyword |
| `get_architecture_overview` | Understanding high-level codebase structure |
| `refactor_tool` | Planning renames, finding dead code |

### Workflow

1. The graph auto-updates on file changes (via hooks).
2. Use `detect_changes` for code review.
3. Use `get_affected_flows` to understand impact.
4. Use `query_graph` pattern="tests_for" to check coverage.
