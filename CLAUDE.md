# CLAUDE.md — newkycsummary

This project is a **weighted round-robin queue system** that processes two types of jobs:
- **KYC** — ZIP files containing identity documents (Aadhaar, PAN), classified via YOLO + LLM
- **Summary** — PDF URLs downloaded and summarised via LLM

Both jobs come in through a FastAPI API and are processed by a single worker thread that guarantees neither service starves.

---

## File Map

| File | Purpose |
|------|---------|
| `queue_system.py` | Pure queue engine — knows nothing about KYC or Summary |
| `app.py` | Wiring — registers processors, creates FastAPI endpoints, webhook post-back |
| `kyc.py` | KYC processor: YOLO face detection + Ollama multimodal classification |
| `summary.py` | Summary processor: PDF download + LangChain map-reduce summarisation |
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
POST /kyc        { "zip_path": "s3://bucket/key.zip" }
POST /kyc/scan   (no body) — scan s3://ib-prod-ekyc/CVLKRA/ and enqueue all unprocessed ZIPs
POST /summary    { "url": "https://example.com/file.pdf", "news_id": 123 }
```

`/kyc/scan` uses `KYC_INPUT_BUCKET` / `KYC_INPUT_PREFIX` env vars to locate ZIPs. Already-processed ZIPs (detected via `_already_processed()`) are automatically skipped.

Summary results are posted back to a webhook (`SUMMARY_WEBHOOK_URL`) with:
```json
{ "NewsId": 123, "AiSummary": "...", "Error": null, "StatusCode": 200 }
```

### Control queue at runtime

```
PUT /slots/kyc         { "slots": 3 }    # KYC processes 3 files per cycle
PUT /slots/summary     { "slots": 1 }    # Summary processes 1 file per cycle
PUT /priority/kyc      { "priority": 1 } # KYC runs first in each cycle
PUT /priority/summary  { "priority": 2 } # Summary runs second in each cycle
GET /status                              # current queue depths + settings
```

---

## Queue Architecture

**Problem solved:** When one service receives continuous input, simple priority queues cause the other service to never run (starvation).

**Solution:** Weighted round-robin with `slots`.

```
One cycle = [KYC × slots_kyc → SUMMARY × slots_summary → repeat]
```

Default: `kyc slots=3, summary slots=1`
```
KYC  KYC  KYC  →  SUMMARY  →  KYC  KYC  KYC  →  SUMMARY  →  ...
```

Even with 10,000 KYC jobs queued, Summary always gets its turn every 3 KYC jobs.

**How `slots` controls the ratio:**

| kyc slots | summary slots | KYC share | Summary share |
|-----------|---------------|-----------|---------------|
| 3         | 1             | 75%       | 25%           |
| 1         | 1             | 50%       | 50%           |
| 10        | 1             | 91%       | 9%            |
| 3         | 5             | 37%       | 63%           |

---

## Adding a New Processor

**Step 1** — Write a handler in any file:
```python
def ocr_handler(data: dict) -> dict:
    return run_ocr(data["file_path"])
```

**Step 2** — Register it in `app.py` (one line):
```python
queue.register_processor("ocr", ocr_handler, priority=3, slots=2)
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

Each processor writes to its own colour-coded stream and log file:

| Stream | Colour | Log File |
|--------|--------|----------|
| `[QUEUE]` | Yellow | `queue_manager.log` |
| `[KYC]` | Cyan | `kyc_queue.log` |
| `[SUMMARY]` | Magenta | `summary_queue.log` |

Console example:
```
[10:42:01][QUEUE][INFO] Cycle start → kyc(p=1, s=3, q=847) | summary(p=2, s=1, q=23)
[10:42:01][KYC  ][INFO] START kyc_104201  [slot 1/3]  waited 0.12s
[10:42:44][KYC  ][INFO] DONE  kyc_104201  |  42.3s
[10:43:28][KYC  ][INFO] Cycle 1: processed 3/3 slots  |  844 remaining
[10:43:28][SUMM ][INFO] START summary_104301  [slot 1/1]  waited 127.5s
[10:43:44][SUMM ][INFO] DONE  summary_104301  |  15.2s
[10:43:44][QUEUE][INFO] Cycle 1 done → kyc(q=844) | summary(q=22)
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
Slack alert on failure
    ↓
3-second pause before next ZIP (inter-ZIP rate limiting)
```

Key classes:
- `SmartAadharDetector` — main processor class
- `ZipProcessingContext` — context manager tracking ZIP lifecycle stages (ENTRY → EXTRACTION → PROCESSING → S3_UPLOAD → VERIFICATION → CLEANUP)
- `SlackAlertManager` — sends Slack alerts only on failure, deduplicates alerts per hour

`SLACK_WEBHOOK_URL` is read from the environment variable (not hardcoded). Set it in `.env` to enable failure alerts.

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
| `KYC_INPUT_BUCKET` | `app.py` | S3 bucket to scan for input ZIPs (`/kyc/scan` endpoint) |
| `KYC_INPUT_PREFIX` | `app.py` | S3 prefix to scan for input ZIPs (e.g. `CVLKRA/`) |
| `AWS_ACCESS_KEY_ID` | `app.py` | AWS credentials |
| `AWS_SECRET_ACCESS_KEY` | `app.py` | AWS credentials |
| `AWS_REGION` | `app.py` | AWS region, default `ap-south-1` |
| `SLACK_WEBHOOK_URL` | `kyc.py` | Slack incoming webhook for failure alerts. Empty = alerts disabled |
| `SUMMARY_WEBHOOK_URL` | `app.py` | Webhook to POST summary results back to India Bonds API |
| `SUMMARY_WEBHOOK_TOKEN` | `app.py` | Bearer token for the summary webhook |
| `OPENAI_API_KEY` | `kyc.py` | Optional, loaded but not used directly |

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
