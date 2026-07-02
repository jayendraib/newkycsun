from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

import boto3
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

load_dotenv()

import db  # noqa: E402
from queue_system import QueueManager  # noqa: E402
from kyc import SmartAadharDetector  # noqa: E402
from summary import process_url  # noqa: E402

db.init_db()

logger = logging.getLogger(__name__)

kyc_logger = logging.getLogger("app.kyc")
kyc_logger.addHandler(db.PostgresLogHandler(table="kyc_logs", extra_fields=("zip_name",)))

summary_logger = logging.getLogger("app.summary")
summary_logger.addHandler(db.PostgresLogHandler(table="summary_logs", extra_fields=("url", "news_id")))

# ── TEST MODE ─────────────────────────────────────────────────────────────────
TEST_MODE = os.getenv("TEST_MODE", "true").lower() == "true"

# ── TEST S3 CONFIG ────────────────────────────────────────────────────────────
KYC_INPUT_BUCKET = os.getenv("KYC_INPUT_BUCKET", "ib-ai-data")
KYC_INPUT_PREFIX = os.getenv("KYC_INPUT_PREFIX", "")
KYC_OUTPUT_BUCKET = os.getenv("KYC_OUTPUT_BUCKET", "ib-ai-data")

# ── PROD WEBHOOKS — COMMENTED OUT ─────────────────────────────────────────────
# WEBHOOK_URL = os.getenv(
#     "SUMMARY_WEBHOOK_URL",
#     "http://prod-api-new-internal.indiabonds.com:8080/marketnews/internapi/v1/news/ai-summary",
# )
# WEBHOOK_TOKEN = os.getenv("SUMMARY_WEBHOOK_TOKEN", "Ind1Ab0nd$AiService#4@128689")
# KYC_AUTH_ID = os.getenv("KYC_AUTH_ID", "2f8f114c-e61e-41dc-b158-f6f25a121006")
# KYC_WEBHOOK_URL = os.getenv(
#     "KYC_WEBHOOK_URL",
#     "https://kycapi.indiabonds.com/api/client/kra/documentfetchupdate",
# )

# ── TEST WEBHOOK — WEBHOOK.SITE ───────────────────────────────────────────────
WEBHOOK_URL = os.getenv("SUMMARY_WEBHOOK_URL", "https://webhook.site/601d8a53-a251-49b8-ad45-6ba047cbf203")
WEBHOOK_TOKEN = os.getenv("SUMMARY_WEBHOOK_TOKEN", "test-token-local")
KYC_AUTH_ID = os.getenv("KYC_AUTH_ID", "test-auth-local")
KYC_WEBHOOK_URL = os.getenv("KYC_WEBHOOK_URL", "https://webhook.site/601d8a53-a251-49b8-ad45-6ba047cbf203")

# ── Queue setup ───────────────────────────────────────────────────────────────
queue = QueueManager()

# ── KYC processor ─────────────────────────────────────────────────────────────
detector = SmartAadharDetector(
    output_folder=os.getenv("KYC_OUTPUT_FOLDER", "/tmp/kyc_output"),
    bucket_name=os.getenv("KYC_S3_BUCKET", KYC_OUTPUT_BUCKET),
    aws_region=os.getenv("AWS_REGION", "ap-south-1"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)


def kyc_handler(data: dict) -> dict:
    zip_path = data["zip_path"]
    zip_name = Path(zip_path.split("/")[-1])

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
        if local_zip_path is not None:
            Path(local_zip_path).unlink(missing_ok=True)

    has_valid_docs = (
        len(result.get("aadhaar", [])) > 0
        or len(result.get("pan", [])) > 0
        or len(result.get("userimage", [])) > 0
    )
    if has_valid_docs:
        file_name = zip_name.stem
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
            if key.lower().endswith(".zip") and not key.endswith("/"):
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


# ── REQUEST MODELS ────────────────────────────────────────────────────────────

class KycRequest(BaseModel):
    bucket_name: str
    aws_region: str
    zip_s3_path: str


class SummaryRequest(BaseModel):
    url: str
    news_id: int


class IngestRequest(BaseModel):
    """One endpoint for everything — pass any URL, system routes to correct processor."""
    url: str                    # s3://... for zips, http/https for PDFs
    news_id: int | None = None  # Required only for PDFs


# ── UNIFIED INGEST — ROUTES TO EXISTING ENDPOINTS ─────────────────────────────
@app.post("/ingest")
def ingest(req: IngestRequest, auth_id: str | None = Header(None, alias="auth-id")):
    """
    Random input comes here. System detects zip vs PDF and routes to:
      - /document/processor/  (for zips)
      - /sum                  (for PDFs)
    """
    if not TEST_MODE and auth_id != KYC_AUTH_ID:
        raise HTTPException(status_code=401, detail="Invalid or missing auth-id header")

    parsed = urlparse(req.url)
    path = parsed.path.lower()

    # ── ZIP → KYC (/document/processor/) ──────────────────────────────────────
    if path.endswith(".zip"):
        # Extract bucket from s3://bucket/key path
        if req.url.startswith("s3://"):
            parts = req.url.replace("s3://", "").split("/", 1)
            bucket_name = parts[0] if parts else KYC_INPUT_BUCKET
            # Route to existing KYC endpoint logic
            item_id = queue.enqueue("kyc", {"zip_path": req.url})
            return {
                "item_id": item_id,
                "status": "queued",
                "routed_to": "/document/processor/",
                "type": "kyc",
                "zip": req.url.split("/")[-1]
            }
        else:
            # Local path
            item_id = queue.enqueue("kyc", {"zip_path": req.url})
            return {
                "item_id": item_id,
                "status": "queued",
                "routed_to": "/document/processor/",
                "type": "kyc",
                "zip": req.url.split("/")[-1]
            }

    # ── PDF → SUMMARY (/sum) ─────────────────────────────────────────────────
    elif path.endswith(".pdf"):
        if req.news_id is None:
            raise HTTPException(status_code=400, detail="news_id is required for PDF summary jobs")
        item_id = queue.enqueue("summary", {"url": req.url, "news_id": req.news_id})
        return {
            "item_id": item_id,
            "status": "queued",
            "routed_to": "/sum",
            "type": "summary",
            "url": req.url
        }

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {path}. Only .zip and .pdf are supported.")


# ── EXISTING ENDPOINTS (unchanged, still work standalone) ────────────────────

@app.post("/document/processor/")
def submit_kyc(req: KycRequest, auth_id: str | None = Header(None, alias="auth-id")):
    if not TEST_MODE and auth_id != KYC_AUTH_ID:
        raise HTTPException(status_code=401, detail="Invalid or missing auth-id header")
    item_id = queue.enqueue("kyc", {"zip_path": req.zip_s3_path})
    return {"item_id": item_id, "status": "queued"}


@app.post("/api/kyc/scan")
def scan_and_enqueue(auth_id: str | None = Header(None, alias="auth-id")):
    if not TEST_MODE and auth_id != KYC_AUTH_ID:
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


@app.post("/sum")
def submit_summary(req: SummaryRequest):
    item_id = queue.enqueue("summary", {"url": req.url, "news_id": req.news_id})
    return {"item_id": item_id, "status": "queued"}


@app.get("/status")
def queue_status():
    return queue.status()


@app.get("/health")
def health():
    return {"status": "ok"}