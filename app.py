"""
API + queue wiring.

Each incoming request just inserts the job into Postgres (queue_items table)
and returns immediately. KYC and Summary each have their own dedicated
worker thread that processes jobs one by one, oldest first (FIFO) — see
queue_system.py.

To add a new processor: add register_processor() below + a new endpoint.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import boto3
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

import db
from queue_system import QueueManager
from kyc import SmartAadharDetector
from summary import process_url

load_dotenv()

# Initialise the Postgres log tables (kyc_logs, summary_logs, queue_logs).
db.init_db()

logger = logging.getLogger(__name__)

kyc_logger = logging.getLogger("app.kyc")
kyc_logger.addHandler(db.PostgresLogHandler(table="kyc_logs", extra_fields=("zip_name",)))

summary_logger = logging.getLogger("app.summary")
summary_logger.addHandler(db.PostgresLogHandler(table="summary_logs", extra_fields=("url", "news_id")))

# ── S3 input config ───────────────────────────────────────────────────────────
# ZIPs are read from:  s3://ib-prod-ekyc/CVLKRA/
# Output is saved to:  s3://ib-prod-ekyc/CVLKRA_AI/<zip_name>/...

KYC_INPUT_BUCKET = os.getenv("KYC_INPUT_BUCKET", "")
KYC_INPUT_PREFIX = os.getenv("KYC_INPUT_PREFIX", "")

WEBHOOK_URL = os.getenv(
    "SUMMARY_WEBHOOK_URL",
    "",
)
WEBHOOK_TOKEN = os.getenv("SUMMARY_WEBHOOK_TOKEN", "")

KYC_AUTH_ID = os.getenv("KYC_AUTH_ID", "2f8f114c-e61e-41dc-b158-f6f25a121006")
KYC_WEBHOOK_URL = os.getenv(
    "KYC_WEBHOOK_URL",
    "https://kycapi.indiabonds.com/api/client/kra/documentfetchupdate",
)

# ── Queue setup ───────────────────────────────────────────────────────────────

queue = QueueManager()

# ── KYC processor ─────────────────────────────────────────────────────────────

detector = SmartAadharDetector(
    output_folder=os.getenv("KYC_OUTPUT_FOLDER", "/tmp/kyc_output"),
    bucket_name=os.getenv("KYC_S3_BUCKET", "ib-prod-ekyc"),
    aws_region=os.getenv("AWS_REGION", "ap-south-1"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)


def kyc_handler(data: dict) -> dict:
    zip_path = data["zip_path"]
    zip_name = Path(zip_path.split("/")[-1])

    # Skip if already processed (checks CVLKRA_AI/<zip_name>/ in output bucket)
    if detector._already_processed(zip_name):
        kyc_logger.info(f"Skipping already-processed ZIP: {zip_name}", extra={"zip_name": str(zip_name)})
        return {"status": "skipped", "zip": str(zip_name)}

    local_zip_path = None
    try:
        if zip_path.startswith("s3://"):
            local_zip_path = detector.download_from_s3(zip_path)
            result = detector.process_zip_as_one(str(local_zip_path))
        else:
            result = detector.process_zip_as_one(zip_path)
    finally:
        # Remove the downloaded zip from local disk regardless of success/failure —
        # results already live in S3 + the webhook, nothing local needs to remain.
        if local_zip_path is not None:
            Path(local_zip_path).unlink(missing_ok=True)

    # Only notify KYC completion webhook when valid documents were actually found
    # (matches ib_ai_news_summary/tasks/kyc_processor_task.py behaviour)
    has_valid_docs = (
        len(result.get("aadhaar", [])) > 0
        or len(result.get("pan", [])) > 0
        or len(result.get("userimage", [])) > 0
    )
    if has_valid_docs:
        file_name = zip_name.stem  # filename without .zip extension
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(KYC_WEBHOOK_URL, json={"fileName": file_name})
                resp.raise_for_status()
                kyc_logger.info(
                    f"KYC webhook notified: {file_name} → {resp.status_code}",
                    extra={"zip_name": str(zip_name)},
                )
        except Exception as exc:
            kyc_logger.error(f"KYC webhook failed for {file_name}: {exc}", extra={"zip_name": str(zip_name)})
    else:
        kyc_logger.info(f"No valid docs found for {zip_name}, skipping KYC webhook", extra={"zip_name": str(zip_name)})

    return result


def _list_zips_from_s3(bucket: str, prefix: str) -> list[str]:
    """List all .zip keys under s3://bucket/prefix/"""
    s3 = boto3.client(
        "s3",
        region_name=os.getenv("AWS_REGION", "ap-south-1"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith(".zip"):
                keys.append(f"s3://{bucket}/{key}")
    return keys


# ── Summary processor ──────────────────────────────────────────────────────────

async def summary_handler(data: dict) -> dict:
    result = await process_url(data["url"])

    has_error = "error" in result
    summary_text = result.get("summary", "")
    unable_to_summarise = "Sorry we are unable to summarise this pdf" in summary_text

    if has_error:
        payload = {"NewsId": data["news_id"], "AiSummary": None, "Error": result["error"], "StatusCode": 500}
    elif unable_to_summarise:
        payload = {"NewsId": data["news_id"], "AiSummary": None, "Error": summary_text, "StatusCode": 400}
    else:
        payload = {"NewsId": data["news_id"], "AiSummary": summary_text, "Error": None, "StatusCode": 200}

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                WEBHOOK_URL,
                json=payload,
                headers={"Authorization": f"Bearer {WEBHOOK_TOKEN}", "Content-Type": "application/json"},
            )
            resp.raise_for_status()
            summary_logger.info(
                f"Webhook posted for news_id={data['news_id']}: {resp.status_code}",
                extra={"url": data["url"], "news_id": data["news_id"]},
            )
    except Exception as exc:
        summary_logger.error(
            f"Webhook failed for news_id={data['news_id']}: {exc}",
            extra={"url": data["url"], "news_id": data["news_id"]},
        )

    return result


# ── Register processors ───────────────────────────────────────────────────────
#
# Each processor gets its own dedicated worker thread pulling FIFO from its
# own slice of the queue_items table — KYC and Summary never block each other.
#
# To add another processor (e.g. OCR):
#   def ocr_handler(data): ...
#   queue.register_processor("ocr", ocr_handler)

queue.register_processor("kyc", kyc_handler)
queue.register_processor("summary", summary_handler)


# ── FastAPI ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    queue.start()
    yield
    queue.stop()


app = FastAPI(
    title="IB Processing Queue",
    lifespan=lifespan,
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)


# ── Request/update models ─────────────────────────────────────────────────────

class KycRequest(BaseModel):
    bucket_name: str        # output S3 bucket, e.g. "CVLKRA_AI"
    aws_region: str         # e.g. "ap-south-1"
    zip_s3_path: str        # full S3 path, e.g. "s3://bucket/path/file.zip"


class SummaryRequest(BaseModel):
    url: str                # PDF URL to download and summarise
    news_id: int            # news record ID — echoed back in the webhook payload


# ── Input endpoints ───────────────────────────────────────────────────────────
#
# To add a new input source, copy the pattern below:
#   @app.post("/ocr")
#   def submit_ocr(req: OcrRequest):
#       item_id = queue.enqueue("ocr", {"file": req.file})
#       return {"item_id": item_id, "status": "queued"}

@app.post("/api/kyc/document/processor/") #here kyc job added to the queue
def submit_kyc(req: KycRequest, auth_id: str | None = Header(None, alias="auth-id")):
    if auth_id != KYC_AUTH_ID:
        raise HTTPException(status_code=401, detail="Invalid or missing auth-id header")
    item_id = queue.enqueue("kyc", {"zip_path": req.zip_s3_path})
    return {"item_id": item_id, "status": "queued"}


@app.post("/api/kyc/scan")
def scan_and_enqueue(auth_id: str | None = Header(None, alias="auth-id")):
    """Scan s3://ib-prod-ekyc/CVLKRA/ for all .zip files and enqueue any that have not been processed yet."""
    if auth_id != KYC_AUTH_ID:
        raise HTTPException(status_code=401, detail="Invalid or missing auth-id header")
    try:
        all_zips = _list_zips_from_s3(KYC_INPUT_BUCKET, KYC_INPUT_PREFIX)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to list S3 ZIPs: {exc}")

    item_ids = []
    for s3_path in all_zips:
        item_id = queue.enqueue("kyc", {"zip_path": s3_path})
        item_ids.append({"zip": s3_path.split("/")[-1], "item_id": item_id})

    return {
        "found": len(all_zips),
        "queued": len(item_ids),
        "items": item_ids,
    }


@app.post("/api/sum") #summary job added
def submit_summary(req: SummaryRequest):
    item_id = queue.enqueue("summary", {"url": req.url, "news_id": req.news_id})
    return {"item_id": item_id, "status": "queued"}


@app.get("/status")
def queue_status():
    """Returns current queue depth + failed count for each processor."""
    return queue.status()


@app.get("/health")
def health():
    return {"status": "ok"}
