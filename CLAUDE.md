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
| `app.py` | Wiring — registers processors, creates FastAPI endpoints |
| `kyc.py` | KYC processor: YOLO face detection + Ollama multimodal classification |
| `summary.py` | Summary processor: PDF download + LangChain map-reduce summarisation |
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
POST /summary    { "url": "https://example.com/file.pdf" }
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
Extract PDFs from ZIP (handles nested ZIPs)
    ↓
Select latest valid PDF (by date pattern in filename)
    ↓
Convert PDF pages to images (pdf2image, 300 DPI)
    ↓
YOLO face detection (yolov8n-face.pt) — finds best face crop
    ↓
LLM classification per page (qwen3-vl:8b via Ollama)
    → aadhaar / pan / uncertain
    ↓
Keep highest-confidence image per document type
    ↓
Upload classified images to S3 + verify upload
    ↓
Slack alert on failure
```

Key classes:
- `SmartAadharDetector` — main processor class
- `ZipProcessingContext` — context manager tracking ZIP lifecycle stages
- `SlackAlertManager` — sends Slack alerts only on failure

---

## Summary Pipeline (`summary.py`)

```
PDF URL
    ↓
download_pdf()   — async, retries 3×, streams to temp file
    ↓
load_and_chunk() — PyPDFLoader + RecursiveCharacterTextSplitter (2500 chars)
    ↓
summarize_docs() — LangChain map_reduce chain (gemma3:4b via Ollama)
    ↓
Returns clean summary string
```

---

## LLM Models Required (Ollama)

```bash
ollama pull qwen3-vl:8b    # KYC classification
ollama pull gemma3:4b      # PDF summarisation
```

---

## Environment Variables

| Variable | Used By | Notes |
|----------|---------|-------|
| `KYC_OUTPUT_FOLDER` | `app.py` | Local output path, default `/tmp/kyc_output` |
| `KYC_S3_BUCKET` | `app.py` | S3 bucket for KYC uploads |
| `AWS_ACCESS_KEY_ID` | `app.py` | AWS credentials |
| `AWS_SECRET_ACCESS_KEY` | `app.py` | AWS credentials |
| `OPENAI_API_KEY` | `kyc.py` | Optional, loaded but not used directly |
