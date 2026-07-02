import requests
import random
import time

BASE_URL = "http://localhost:8000"
HEADERS = {"Content-Type": "application/json", "auth-id": "test"}

# ── 10 ZIPS (at root of ib-ai-data) ───────────────────────────────────────────
ZIPS = [
    "s3://ib-ai-data/AAAPA0450E_75B3EAFB-B1A4-40D4-978E-7DDB74FBB08E.zip",
    "s3://ib-ai-data/AAAPC1721E_C6BF05B1-6D25-4F42-BFD8-293FF5E68939.zip",
    "s3://ib-ai-data/AAAPC7423K_A4E84366-F35E-44B4-9221-D9851FE5DDF0.zip",
    "s3://ib-ai-data/AAAPD3523D_D3AE377D-458D-4EC7-81C9-FEB382A8F397.zip",
    "s3://ib-ai-data/AAAPE4589Q_2DE0DA77-731A-40DF-B019-67236194F33E.zip",
    "s3://ib-ai-data/AAAPG5758K_69E2930D-584B-4148-AC51-2AD76E837335.zip",
    "s3://ib-ai-data/AAAPG6070A_104821AA-0340-4F57-A09D-609736A31946.zip",
    "s3://ib-ai-data/AAAPG6070A_57B75CB9-A659-4341-AED6-ED9248884C26.zip",
    "s3://ib-ai-data/AAAPG7380H_2E7B7DE6-6F36-407F-B7AD-AD770568AACD.zip",
    "s3://ib-ai-data/AAAPH3626H_6A61CBE1-FB9D-4338-9F03-FC19152549BF.zip",
]

# ── 10 PDFs (at root of ib-ai-data) ───────────────────────────────────────────
# These need presigned URLs. Generate them first with the script below.
PDF_KEYS = [
    "AR_29205_VIJIFIN_2024_2025_A_847563_08122025162828_202512301219526954662_FAEB03EB-657A-4AD5-8248-2F1E65FBF8F1.pdf",
    "AR_29368_POLYCAB_2025_2026_A_20879015_09062026002210_202606090303112686137_74D94719-CDC3-4454-AC65-E378108904E0.pdf",
    "AR_29463_ROSSARI_2025_2026_A_16880242_27062026002619_202606270303242357616_66301D25-2558-483C-B7A6-55020FD52EC5.pdf",
    "AR_29471_TANLA_2025_2026_A_13641614_28062026000048_202606280303062609290_06401B07-A9DE-48BC-9EA2-BDAC924233DF.pdf",
    "SME_AR_29170_CEDAAR_2024_2025_A_4037147_24102025194423_202512301220058923411_FAEB03EB-657A-4AD5-8248-2F1E65FBF8F1.pdf",
    "SME_AR_29173_CURRENT_2024_2025_A_7318724_28102025101902_202512301220037648979_FAEB03EB-657A-4AD5-8248-2F1E65FBF8F1.pdf",
    "SME_AR_29177_MGSL_2024_2025_A_1758946_05112025153107_202512301220023022950_FAEB03EB-657A-4AD5-8248-2F1E65FBF8F1.pdf",
    "SME_AR_29179_INSPIRE_2024_2025_A_1801853_06112025165123_202512301220010082137_FAEB03EB-657A-4AD5-8248-2F1E65FBF8F1.pdf",
    "SME_AR_29187_KAYTEX_2024_2025_A_1894808_21112025193812_202512301219568461647_FAEB03EB-657A-4AD5-8248-2F1E65FBF8F1.pdf",
    "SME_AR_29189_SAJHOTELS_2024_2025_A_2450169_29112025181937_202512301219555373647_FAEB03EB-657A-4AD5-8248-2F1E65FBF8F1.pdf",
]

# ── STEP 1: GENERATE PRESIGNED URLs FOR PDFs ───────────────────────────────────
# Run this first, then paste the URLs into PRESIGNED_PDFS below

import boto3

AWS_ACCESS_KEY_ID = "YOUR_ACCESS_KEY"       # <-- paste your key
AWS_SECRET_ACCESS_KEY = "YOUR_SECRET_KEY"   # <-- paste your secret
REGION = "ap-south-1"
BUCKET = "ib-ai-data"

s3 = boto3.client(
    "s3",
    region_name=REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

print("=== PRESIGNED PDF URLs (paste these into PRESIGNED_PDFS below) ===\n")
for i, key in enumerate(PDF_KEYS, 1):
    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": BUCKET, "Key": key},
        ExpiresIn=3600  # 1 hour
    )
    print(f'    "{url}",  # PDF {i}: {key[:50]}...')

print("\n=== COPY THE ABOVE URLs INTO PRESIGNED_PDFS LIST BELOW ===")
print("Then run the rest of the script.\n")

# ── STEP 2: PASTE PRESIGNED URLs HERE ───────────────────────────────────────
# Replace this list with the output from Step 1 above
PRESIGNED_PDFS = [
    # "https://ib-ai-data.s3.ap-south-1.amazonaws.com/AR_29205_VIJIFIN_...",
    # "https://ib-ai-data.s3.ap-south-1.amazonaws.com/AR_29368_POLYCAB_...",
    # ... paste all 10 URLs here
]

# ── STEP 3: FIRE RANDOM INGEST REQUESTS ──────────────────────────────────────
# Only runs if PRESIGNED_PDFS is filled
if len(PRESIGNED_PDFS) == 10:
    all_jobs = [{"type": "zip", "payload": {"url": z}} for z in ZIPS] + \
               [{"type": "pdf", "payload": {"url": p, "news_id": i+1}} for i, p in enumerate(PRESIGNED_PDFS)]

    random.shuffle(all_jobs)

    print(f"\n=== FIRING {len(all_jobs)} RANDOM JOBS ===\n")
    for job in all_jobs:
        url_preview = job["payload"]["url"][:70]
        print(f"[{job['type'].upper()}] {url_preview}...")
        resp = requests.post(f"{BASE_URL}/ingest", json=job["payload"], headers=HEADERS)
        print(f"  → {resp.status_code}: {resp.json()}")
        time.sleep(0.3)

    print("\n=== ALL JOBS SENT ===")
    print("Check: http://localhost:8000/status")
    print("Check: https://webhook.site/601d8a53-a251-49b8-ad45-6ba047cbf203")
else:
    print("\n!!! Fill PRESIGNED_PDFS list first (Step 2) !!!")