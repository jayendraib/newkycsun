import os
import boto3
from pathlib import Path

BUCKET = "ib-ai-data"
REGION = "ap-south-1"

ZIP_LOCAL_DIR = r"/home/jayendra/jayendraproject/clean-newkycsun/CVLKRA Files"      # <-- CHANGE THIS
PDF_LOCAL_DIR = r"/home/jayendra/jayendraproject/clean-newkycsun/pdf"      # <-- CHANGE THIS

s3 = boto3.client("s3", region_name=REGION)

# Upload ZIPS to bucket ROOT (no prefix)
zip_path = Path(ZIP_LOCAL_DIR)
for f in zip_path.iterdir():
    if f.is_file() and f.suffix.lower() == ".zip":
        print(f"Uploading zip {f.name} → s3://{BUCKET}/{f.name}")
        s3.upload_file(str(f), BUCKET, f.name)

# Upload PDFS to test-pdfs/ prefix
pdf_path = Path(PDF_LOCAL_DIR)
for f in pdf_path.iterdir():
    if f.is_file() and f.suffix.lower() == ".pdf":
        key = f"test-pdfs/{f.name}"
        print(f"Uploading pdf {f.name} → s3://{BUCKET}/{key}")
        s3.upload_file(str(f), BUCKET, key)

print("Done.")