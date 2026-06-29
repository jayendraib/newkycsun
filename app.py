"""
API + queue wiring.

Round-robin cycle currently set as:
    KYC     priority=1  slots=3   →  processes 3 items per cycle
    SUMMARY priority=2  slots=1   →  processes 1 item per cycle

So for every 3 KYC items processed, 1 Summary item runs — regardless
of how many items are queued. Neither service ever starves.

To add a new processor: add register_processor() below + a new endpoint.
To change the ratio:    PUT /slots/{processor}    {"slots": N}
To change cycle order:  PUT /priority/{processor} {"priority": N}
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import boto3
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from queue_system import QueueManager
from kyc import SmartAadharDetector
from summary import process_url

load_dotenv()

logger = logging.getLogger(__name__)

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

# ── Queue setup ───────────────────────────────────────────────────────────────

queue = QueueManager(log_dir=".")

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
        logger.info(f"Skipping already-processed ZIP: {zip_name}")
        return {"status": "skipped", "zip": str(zip_name)}

    if zip_path.startswith("s3://"):
        local_zip_path = detector.download_from_s3(zip_path)
        return detector.process_zip_as_one(str(local_zip_path))
    return detector.process_zip_as_one(zip_path)


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
            logger.info(f"Webhook posted for news_id={data['news_id']}: {resp.status_code}")
    except Exception as exc:
        logger.error(f"Webhook failed for news_id={data['news_id']}: {exc}")

    return result


# ── Register processors ───────────────────────────────────────────────────────
#
# slots  = items processed per cycle  (controls throughput share)
# priority = order within a cycle    (lower number runs first)
#
# Current config:  every cycle does [KYC×3, SUMMARY×1]
# KYC gets 75% of throughput, Summary gets 25% — Summary never starves.
#
# To add another processor (e.g. OCR):
#   def ocr_handler(data): ...
#   queue.register_processor("ocr", ocr_handler, priority=3, slots=2)

queue.register_processor("kyc",     kyc_handler,     priority=1, slots=3)
queue.register_processor("summary", summary_handler, priority=2, slots=1)


# ── FastAPI ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    queue.start()
    yield
    queue.stop()


app = FastAPI(title="IB Processing Queue", lifespan=lifespan)


# ── Request/update models ─────────────────────────────────────────────────────

class KycRequest(BaseModel):
    zip_path: str           # local path OR s3://bucket/key


class SummaryRequest(BaseModel):
    url: str                # PDF URL to download and summarise
    news_id: int            # news record ID — echoed back in the webhook payload


class PriorityUpdate(BaseModel):
    priority: int           # lower = earlier in each cycle


class SlotsUpdate(BaseModel):
    slots: int              # items this processor handles per cycle


# ── Input endpoints ───────────────────────────────────────────────────────────
#
# To add a new input source, copy the pattern below:
#   @app.post("/ocr")
#   def submit_ocr(req: OcrRequest):
#       item_id = queue.enqueue("ocr", {"file": req.file})
#       return {"item_id": item_id, "status": "queued"}

@app.post("/kyc")
def submit_kyc(req: KycRequest):
    item_id = queue.enqueue("kyc", {"zip_path": req.zip_path})
    return {"item_id": item_id, "status": "queued"}


@app.post("/kyc/scan")
def scan_and_enqueue():
    """
    Scan s3://ib-prod-ekyc/CVLKRA/ for all .zip files and enqueue
    any that have not been processed yet (no output in CVLKRA_AI/).
    Already-processed ZIPs are skipped automatically by kyc_handler.
    """
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


@app.post("/summary")
def submit_summary(req: SummaryRequest):
    item_id = queue.enqueue("summary", {"url": req.url, "news_id": req.news_id})
    return {"item_id": item_id, "status": "queued"}


# ── Control endpoints ─────────────────────────────────────────────────────────

@app.put("/slots/{processor}")
def set_slots(processor: str, req: SlotsUpdate):
    """
    Change how many items this processor handles per cycle.

    Example: PUT /slots/kyc  {"slots": 5}
    Now KYC processes 5 items before yielding to Summary.
    """
    try:
        queue.set_slots(processor, req.slots)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"processor": processor, "slots": req.slots}


@app.put("/priority/{processor}")
def set_priority(processor: str, req: PriorityUpdate):
    """
    Change a processor's position within each cycle.
    Lower number = runs first in the cycle.
    """
    try:
        queue.set_priority(processor, req.priority)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"processor": processor, "priority": req.priority}


@app.get("/status")
def queue_status():
    """
    Returns current queue depths, priorities, and slots for all processors.
    """
    return queue.status()
