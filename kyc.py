from langchain_core.messages import HumanMessage
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import shutil
import os
from PIL import Image
import json
from pdf2image import convert_from_path
import tempfile
import subprocess
import base64
import logging
from dotenv import load_dotenv
load_dotenv()
from langchain_ollama import ChatOllama
import re
import ast
import zipfile
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from datetime import datetime
import io
import urllib.request
import traceback
import sys
import threading
import psutil
import time


logger = logging.getLogger(__name__)

def upload_to_s3(local_zip_path, s3_bucket, s3_key):
    s3_client = boto3.client('s3')
    try:
        s3_client.upload_file(str(local_zip_path), s3_bucket, s3_key)
        logger.info(f"Uploaded {local_zip_path} to s3://{s3_bucket}/{s3_key}")
    except NoCredentialsError:
        logger.error("AWS credentials not available")
    except Exception as e:
        logger.error(f"S3 upload error: {e}")


# ============================================================
# CONFIGURATION
# ============================================================
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")

LOG_LEVEL = "DEBUG"

# ============================================================
# STRUCTURED LOGGER SETUP
# ============================================================
class StructuredLogFormatter(logging.Formatter):
    """Custom formatter that includes timestamps, trace IDs, and structured context."""
    
    def format(self, record):
        # Ensure extra fields exist
        if not hasattr(record, 'zip_name'):
            record.zip_name = 'N/A'
        if not hasattr(record, 'trace_id'):
            record.trace_id = 'N/A'
        if not hasattr(record, 'stage'):
            record.stage = 'GENERAL'
            
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        return (
            f"[{timestamp}] [{record.levelname:8s}] [ZIP:{record.zip_name}] "
            f"[TRACE:{record.trace_id}] [STAGE:{record.stage}] {record.getMessage()}"
        )


# Setup root logger
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Create structured logger for the detector
logger = logging.getLogger("SmartAadharDetector")
logger.setLevel(getattr(logging, LOG_LEVEL))

# Remove existing handlers to avoid duplicates
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Add structured handler
structured_handler = logging.StreamHandler(sys.stdout)
structured_handler.setFormatter(StructuredLogFormatter())
logger.addHandler(structured_handler)

# File handler for persistent logs
log_file_path = Path(tempfile.gettempdir()) / "smart_aadhar_detector.log"
file_handler = logging.FileHandler(log_file_path, mode='a')
file_handler.setFormatter(StructuredLogFormatter())
logger.addHandler(file_handler)

logger.info("Logger initialized", extra={'zip_name': 'SYSTEM', 'trace_id': 'INIT', 'stage': 'SETUP'})


# ============================================================
# SLACK ALERT MANAGER
# ============================================================
class SlackAlertManager:
    """
    Manages Slack alerts for S3 upload failures and system errors.
    Only sends alerts for FAILED operations, never for successes.
    """
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        self._alert_cache = set()  # Prevent duplicate alerts
        self._lock = threading.Lock()
        
    def _generate_alert_key(self, zip_name: str, error_type: str) -> str:
        """Generate unique key to prevent duplicate alerts."""
        return f"{zip_name}:{error_type}:{datetime.now().strftime('%Y-%m-%d %H')}"
    
    def send_alert(self, zip_name: str, error_details: dict, logs: list = None):
        """
        Send error alert to Slack webhook.
        Only alerts on failures, never on success.
        """
        if not self.webhook_url:
            logger.warning(
                f"Slack webhook not configured. Cannot send alert for {zip_name}",
                extra={'zip_name': zip_name, 'trace_id': error_details.get('trace_id', 'N/A'), 'stage': 'SLACK_ALERT'}
            )
            return False
            
        alert_key = self._generate_alert_key(zip_name, error_details.get('error_type', 'UNKNOWN'))
        
        with self._lock:
            if alert_key in self._alert_cache:
                logger.info(
                    f"Duplicate alert suppressed for {zip_name}",
                    extra={'zip_name': zip_name, 'trace_id': error_details.get('trace_id', 'N/A'), 'stage': 'SLACK_ALERT'}
                )
                return False
            self._alert_cache.add(alert_key)
        
        # Build comprehensive error message
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        trace_id = error_details.get('trace_id', 'N/A')
        
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "🚨 SmartAadharDetector S3 Upload Failure",
                    "emoji": True
                }
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*ZIP File:*\n`{zip_name}`"},
                    {"type": "mrkdwn", "text": f"*Timestamp:*\n{timestamp}"},
                    {"type": "mrkdwn", "text": f"*Trace ID:*\n`{trace_id}`"},
                    {"type": "mrkdwn", "text": f"*Error Type:*\n{error_details.get('error_type', 'UNKNOWN')}"}
                ]
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Error Message:*\n```{error_details.get('error_message', 'No details available')}```"
                }
            }
        ]
        
        # Add processing stage info
        if 'stage' in error_details:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Failed At Stage:*\n{error_details['stage']}"
                }
            })
        
        # Add S3 verification details
        if 's3_verification' in error_details:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*S3 Verification:*\n{error_details['s3_verification']}"
                }
            })
        
        # Add system info if available
        if 'system_info' in error_details:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*System Info:*\n```{json.dumps(error_details['system_info'], indent=2)}```"
                }
            })
        
        # Add recent logs
        if logs:
            log_text = "\n".join(logs[-20:])  # Last 20 log lines
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Recent Logs:*\n```{log_text[:2900]}```"  # Slack limit
                }
            })
        
        # Add traceback if available
        if 'traceback' in error_details:
            tb = error_details['traceback'][:2900]
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Traceback:*\n```{tb}```"
                }
            })
        
        payload = {
            "text": f"S3 Upload Failure: {zip_name}",
            "blocks": blocks,
            "attachments": [{
                "color": "danger",
                "footer": "SmartAadharDetector Alert System"
            }]
        }
        
        try:
            req = urllib.request.Request(
                self.webhook_url,
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'},
                method='POST'
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                if response.status == 200:
                    logger.info(
                        f"Slack alert sent successfully for {zip_name}",
                        extra={'zip_name': zip_name, 'trace_id': trace_id, 'stage': 'SLACK_ALERT'}
                    )
                    return True
        except Exception as e:
            logger.error(
                f"Failed to send Slack alert: {e}",
                extra={'zip_name': zip_name, 'trace_id': trace_id, 'stage': 'SLACK_ALERT'}
            )
            return False
        
        return False


# Initialize Slack alert manager
slack_alert = SlackAlertManager(SLACK_WEBHOOK_URL)


# ============================================================
# ZIP PROCESSING CONTEXT MANAGER
# ============================================================
class ZipProcessingContext:
    """
    Context manager that tracks the entire lifecycle of a ZIP file:
    ENTRY → EXTRACTION → PROCESSING → S3_UPLOAD → VERIFICATION → CLEANUP
    """
    
    STAGES = {
        'ENTRY': 'ZIP_ENTERED',
        'EXTRACTION': 'ZIP_EXTRACTED', 
        'PROCESSING': 'ZIP_PROCESSING',
        'S3_UPLOAD': 'S3_UPLOADING',
        'VERIFICATION': 'S3_VERIFYING',
        'CLEANUP': 'CLEANUP',
        'COMPLETE': 'COMPLETE',
        'FAILED': 'FAILED'
    }
    
    def __init__(self, zip_name: str, detector):
        self.zip_name = zip_name
        self.detector = detector
        self.trace_id = f"{zip_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{os.urandom(4).hex()}"
        self.current_stage = None
        self.log_buffer = []  # Capture all logs for this ZIP
        self.start_time = None
        self.error_info = None
        self.s3_paths = []  # Track what should be in S3
        
    def _log(self, level: str, message: str, stage: str = None):
        """Internal logging that captures to buffer and structured logger."""
        stage = stage or self.current_stage or 'UNKNOWN'
        extra = {
            'zip_name': self.zip_name,
            'trace_id': self.trace_id,
            'stage': stage
        }
        
        log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] [{level}] {message}"
        self.log_buffer.append(log_entry)
        
        getattr(logger, level.lower(), logger.info)(message, extra=extra)
        
    '''def __enter__(self):
        self.start_time = datetime.now()
        self.current_stage = self.STAGES['ENTRY']
        self._log('INFO', f"🚀 ZIP ENTERED processing pipeline: {self.zip_name}")
        self._log('INFO', f"Trace ID assigned: {self.trace_id}")
        self._log('INFO', f"Output folder: {self.detector.output_folder}")
        self._log('INFO', f"S3 Bucket: {self.detector.bucket_name or 'Not configured'}")
        return self'''
    def __enter__(self):
        self.start_time = datetime.now()
        self.current_stage = self.STAGES['ENTRY']
        self.trace_id = f"{self.zip_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{os.urandom(4).hex()}"
        
        # Safe logging - don't crash if detector attributes are missing
        try:
            output = self.detector.output_folder if self.detector else 'N/A'
            bucket = self.detector.bucket_name if self.detector else 'N/A'
            self._log('INFO', f"🚀 ZIP ENTERED: {self.zip_name}")
            self._log('INFO', f"Trace ID: {self.trace_id}")
            self._log('INFO', f"Output folder: {output}")
            self._log('INFO', f"S3 Bucket: {bucket or 'Not configured'}")
        except Exception as e:
            self._log('WARNING', f"Context setup logging failed: {e}")
            
        return self
        
    def transition_to(self, stage_name: str, details: str = ""):
        """Transition to a new processing stage."""
        self.current_stage = self.STAGES.get(stage_name, stage_name)
        self._log('INFO', f"➡️  STAGE TRANSITION: {stage_name} - {details}")
        
    def log_extraction(self, num_pdfs: int, nested: bool = False):
        """Log extraction details."""
        self.transition_to('EXTRACTION', f"{'Nested ' if nested else ''}ZIP extracted, {num_pdfs} PDFs found")
        
    def log_processing(self, pdf_name: str, page_num: int = None, details: str = ""):
        """Log processing details."""
        msg = f"Processing PDF: {pdf_name}"
        if page_num:
            msg += f" (Page {page_num})"
        if details:
            msg += f" | {details}"
        self._log('INFO', msg, self.STAGES['PROCESSING'])
        
    def log_classification(self, doctype: str, score: float, action: str):
        """Log classification result."""
        self._log('INFO', f"Classification: {doctype} (score={score:.2f}) → {action}", self.STAGES['PROCESSING'])
        
    def log_s3_upload(self, local_path: str, s3_key: str, folder_type: str):
        """Log S3 upload attempt."""
        self.s3_paths.append((s3_key, folder_type, local_path))
        self._log('INFO', f"S3 Upload attempt: {local_path} → s3://{self.detector.bucket_name}/{s3_key}", self.STAGES['S3_UPLOAD'])
        
    def log_s3_verification(self, s3_key: str, exists: bool, size: int = None):
        """Log S3 verification result."""
        status = "✅ VERIFIED" if exists else "❌ MISSING"
        size_info = f" ({size} bytes)" if size else ""
        self._log('INFO' if exists else 'ERROR', f"S3 Verification {status}: {s3_key}{size_info}", self.STAGES['VERIFICATION'])
        
    def log_error(self, error: Exception, stage: str = None):
        """Log an error with full context."""
        self.current_stage = self.STAGES['FAILED']
        error_type = type(error).__name__
        error_message = str(error)
        tb = traceback.format_exc()
        
        self.error_info = {
            'trace_id': self.trace_id,
            'error_type': error_type,
            'error_message': error_message,
            'stage': stage or self.current_stage,
            'traceback': tb,
            'system_info': self._get_system_info()
        }
        
        self._log('ERROR', f"❌ ERROR in stage {stage or self.current_stage}: {error_type}: {error_message}")
        self._log('ERROR', f"Traceback:\n{tb}")
        
    def _get_system_info(self):
        """Gather system information for debugging."""
        try:
            return {
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage_percent': psutil.disk_usage('/').percent,
                'cpu_percent': psutil.cpu_percent(interval=0.1),
                'temp_dir_size': self._get_dir_size(tempfile.gettempdir()),
                'output_folder_size': self._get_dir_size(str(self.detector.output_folder)) if self.detector.output_folder.exists() else 0
            }
        except Exception:
            return {'error': 'Could not gather system info'}
            
    def _get_dir_size(self, path):
        """Get directory size in MB."""
        try:
            total = 0
            for dirpath, dirnames, filenames in os.walk(path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    if os.path.exists(fp):
                        total += os.path.getsize(fp)
            return round(total / (1024 * 1024), 2)
        except Exception:
            return -1
        
    def send_failure_alert(self):
        """Send Slack alert for this failure."""
        if self.error_info:
            slack_alert.send_alert(
                zip_name=self.zip_name,
                error_details=self.error_info,
                logs=self.log_buffer
            )
            
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()
        
        if exc_type:
            self.log_error(exc_val, self.current_stage)
            self.send_failure_alert()
            self._log('ERROR', f"💥 ZIP PROCESSING FAILED after {duration:.2f}s: {self.zip_name}")
        else:
            self.current_stage = self.STAGES['COMPLETE']
            self._log('INFO', f"✅ ZIP PROCESSING COMPLETE in {duration:.2f}s: {self.zip_name}")
            
        return False  # Don't suppress exceptions


# ============================================================
# S3 VERIFICATION UTILITIES
# ============================================================
def verify_s3_object_exists(s3_client, bucket: str, key: str) -> tuple[bool, int]:
    """
    Verify that an object exists in S3 and return its size.
    Returns (exists, size_in_bytes)
    """
    try:
        response = s3_client.head_object(Bucket=bucket, Key=key)
        size = response.get('ContentLength', 0)
        return True, size
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            return False, 0
        logger.error(f"S3 head_object error: {e}", extra={'zip_name': 'S3_VERIFY', 'trace_id': 'VERIFY', 'stage': 'S3_VERIFY'})
        return False, 0


def verify_s3_upload(s3_client, bucket: str, key: str, local_path: Path, max_retries: int = 3) -> dict:
    """
    Comprehensive S3 upload verification with retries.
    Returns verification result dict.
    """
    result = {
        'verified': False,
        's3_key': key,
        'local_path': str(local_path),
        'local_size': local_path.stat().st_size if local_path.exists() else 0,
        's3_size': 0,
        'attempts': 0,
        'errors': []
    }
    
    for attempt in range(1, max_retries + 1):
        result['attempts'] = attempt
        try:
            exists, size = verify_s3_object_exists(s3_client, bucket, key)
            if exists:
                result['verified'] = True
                result['s3_size'] = size
                return result
            else:
                result['errors'].append(f"Attempt {attempt}: Object not found in S3")
        except Exception as e:
            result['errors'].append(f"Attempt {attempt}: {str(e)}")
            
    return result

# ============================================================
# EXISTING FUNCTIONS (with logging enhancements)
# ============================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
logger.info(f"OPENAI_API_KEY loaded: {'Yes' if OPENAI_API_KEY else 'No'}", extra={'zip_name': 'SYSTEM', 'trace_id': 'INIT', 'stage': 'SETUP'})

# Load YOLO model
model = YOLO("yolov8n-face.pt")
logger.info("YOLO model loaded: yolov8n-face.pt", extra={'zip_name': 'SYSTEM', 'trace_id': 'INIT', 'stage': 'SETUP'})


def pil_to_cv2(img: Image.Image):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# Converting image to base64 string
def image_to_base64(image_path):
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except FileNotFoundError:
        logger.error(f"Image not found: {image_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading image: {e}")
        raise


def extract_json_from_code_fence(text):
    text = text.strip()
    # Try code-fence block
    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if match:
        text = match.group(1)
    # Try JSON loads
    try:
        return json.loads(text)
    except Exception:
        try:
            # Relaxed fallback for python-dict style
            return ast.literal_eval(text)
        except Exception:
            return {}




def classify(image_path):
    try:
        image_base64 = image_to_base64(image_path)

        llm = ChatOllama(
            model="qwen3-vl:8b",
            temperature=0.0,
            num_ctx=32000,
            timeout=120
        )

        # Image-only prompt (no extracted_text)
        classify_prompt = """
            You are a strict multimodal document classifier.

            You receive:
            - An IMAGE of a document page

            RULE 0 (Most Important):
            Decide using the IMAGE ONLY. Do not rely on OCR text.

            Your task:
            Return ONLY one of:
            {"document_type": "aadhaar" | "pan" | "uncertain", "confidence": 0–200}

            You must classify ONLY real Aadhaar or PAN CERTIFICATE pages.

            ==========================
            STEP 1 — IMAGE DECISION (Primary and Only)

            Does this page visually contain a *standalone identity certificate block*?

            A valid certificate block looks like:
            - A boxed or bordered area
            - Structured rows (Name, DOB, Gender, Address, ID)
            - ID-card or certificate layout
            - Photo (for Aadhaar)
            - Government / DigiLocker / Income Tax style
            - QR code or “verified” badge

            If the page mainly looks like:
            - An account opening form
            - A KYC application
            - A declaration / consent page
            - A page full of fields and paragraphs

            Then it is NOT a certificate page.

            If NO clear certificate block is visible in the IMAGE:
            → Return {"document_type": "uncertain", "confidence": 20–50}

            ==========================
            STEP 2 — AADHAAR CLASSIFICATION

            Classify as "aadhaar" if the IMAGE shows:
            - A photo of the person
            - A QR code
            - Government of India / UIDAI / DigiLocker branding
            - A boxed layout with address and Aadhaar number
            - Structured fields like Name, DOB, Gender

            ==========================
            STEP 3 — PAN CLASSIFICATION

            Classify as "pan" if the IMAGE shows:
            - Income Tax Department or NSDL/UTIITSL branding
            - PAN format like AAAAA9999A
            - Name, DOB, Father’s Name
            - Photo and signature
            - No address block

            ==========================
            STEP 4 — SPECIAL AADHAAR XML OVERRIDE

            If the IMAGE shows a boxed identity layout AND any of the following phrases are visually present:

            - “generated from UIDAI XML”
            - “e-aadhaar generated from DigiLocker verified Aadhaar XML”
            - “paperless offline e-KYC”
            - “Downloaded from DigiLocker”
            - “Verified Aadhaar XML”

            → Then classify as:
            {"document_type": "aadhaar", "confidence": 130–200}

            Even if the page looks like a form or belongs to a KYC provider (e.g., Zerodha, Groww, CAMS), this is a valid Aadhaar certificate.

            ==========================
            STEP 5 — SPECIAL PAN VERIFICATION OVERRIDE

            If the IMAGE shows a bordered or structured identity block AND any of the following phrases are visually present:

            - “PAN Verification Record”
            - “Verified PAN”
            - “Digitally signed PAN”
            - “Downloaded from DigiLocker”
            - “Issued by NSDL” or “Issued by Income Tax Department”
            - PAN format like AAAAA9999A

            → Then classify as:
            {"document_type": "pan", "confidence": 130–200}

            Even if the page looks like a KYC form, account opening document, or belongs to a third-party provider (e.g., CAMS, Zerodha, KRA), this is a valid PAN certificate.

            ==========================
            CONFIDENCE SCALE

            0–40   : Form / not an ID
            41–79  : Mixed / weak
            80–129 : Likely ID
            130–200: Strong certificate

        """

        message = HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_base64}"
                    }
                },
                {
                    "type": "text",
                    "text": classify_prompt
                }
            ]
        )

        response = llm.invoke([message])

        try:
            parsed = extract_json_from_code_fence(response.content)
            logger.info(parsed)
            return parsed.get("document_type", "uncertain"), parsed.get("confidence", 0)
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON from model: {response.content}")
            return "uncertain", 0

    except Exception as e:
        logger.error(f"Classification error: {e}")
        return "uncertain", 0
    

class SmartAadharDetector:
    def __init__(self, output_folder=None, bucket_name: str = None, aws_region: str = "ap-south-1",
                 aws_access_key_id: str = None, aws_secret_access_key: str = None):
        # Generate instance ID for tracking
        self.instance_id = f"detector_{datetime.now().strftime('%Y%m%d%H%M%S')}_{os.urandom(4).hex()}"
        logger.info(f"Initializing SmartAadharDetector instance: {self.instance_id}",
                   extra={'zip_name': 'SYSTEM', 'trace_id': self.instance_id, 'stage': 'INIT'})
        
        self.bucket_name = bucket_name  # None = local-only mode
        self.aws_region = aws_region
        self.base_folder = None
        self.s3_client = None
        self.temp_dir = None

        if bucket_name:
            self.temp_dir = Path(tempfile.mkdtemp(prefix="kyc_processing_"))
            if output_folder is None:
                self.output_folder = self.temp_dir
            else:
                self.output_folder = Path(output_folder)
            if aws_access_key_id and aws_secret_access_key:
                self.s3_client = boto3.client(
                    "s3",
                    region_name=aws_region,
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                )
                logger.info("S3 client initialized with explicit credentials",
                           extra={'zip_name': 'SYSTEM', 'trace_id': self.instance_id, 'stage': 'INIT'})
            else:
                self.s3_client = boto3.client("s3", region_name=aws_region)
                logger.info("S3 client initialized with default credentials",
                           extra={'zip_name': 'SYSTEM', 'trace_id': self.instance_id, 'stage': 'INIT'})
        else:
            self.output_folder = Path(output_folder)
            self.output_folder.mkdir(parents=True, exist_ok=True)
            logger.info(f"Local-only mode. Output folder: {self.output_folder}",
                       extra={'zip_name': 'SYSTEM', 'trace_id': self.instance_id, 'stage': 'INIT'})

        # Default output folders
        self.aadhar_folder = self.output_folder / "aadhar_cards"
        self.pan_folder = self.output_folder / "pan_cards"
        self.uncertain_folder = self.output_folder / "uncertain"
        self.user_image = self.output_folder / "user_image"
        for folder in [self.aadhar_folder, self.pan_folder, self.uncertain_folder, self.user_image]:
            folder.mkdir(parents=True, exist_ok=True)
            
        logger.info(f"Detector initialized. Folders ready: aadhar, pan, uncertain, user_image",
                   extra={'zip_name': 'SYSTEM', 'trace_id': self.instance_id, 'stage': 'INIT'})

    def _get_s3_key(self, folder_type: str, filename: str) -> str:
        return f"{self.base_folder}/{folder_type}/{filename}"

    def upload_file_to_s3(self, local_path: Path, folder_type: str, filename: str = None, 
                         zip_context: ZipProcessingContext = None) -> str:
        """
        Upload a local file to S3 with verification and context logging.
        """
        if filename is None:
            filename = Path(local_path).name
        s3_key = f"CVLKRA_AI/{self._get_s3_key(folder_type, filename)}"
        
        # Log upload attempt
        if zip_context:
            zip_context.log_s3_upload(str(local_path), s3_key, folder_type)
        else:
            logger.info(f"S3 Upload attempt: {local_path} → s3://{self.bucket_name}/{s3_key}",
                       extra={'zip_name': 'UNKNOWN', 'trace_id': 'UNKNOWN', 'stage': 'S3_UPLOAD'})
        
        try:
            self.s3_client.upload_file(str(local_path), self.bucket_name, s3_key)
            
            # Verify upload
            if zip_context:
                zip_context.transition_to('VERIFICATION', f"Verifying upload: {s3_key}")
            
            verification = verify_s3_upload(self.s3_client, self.bucket_name, s3_key, local_path)
            
            if verification['verified']:
                if zip_context:
                    zip_context.log_s3_verification(s3_key, True, verification['s3_size'])
                logger.info(f"✅ Uploaded and verified {filename} to s3://{self.bucket_name}/{s3_key}",
                           extra={'zip_name': zip_context.zip_name if zip_context else 'UNKNOWN', 
                                  'trace_id': zip_context.trace_id if zip_context else 'UNKNOWN', 
                                  'stage': 'S3_VERIFY'})
                return s3_key
            else:
                error_msg = f"S3 verification failed after {verification['attempts']} attempts: {verification['errors']}"
                if zip_context:
                    zip_context.log_s3_verification(s3_key, False)
                    zip_context.error_info = {
                        'trace_id': zip_context.trace_id,
                        'error_type': 'S3_VERIFICATION_FAILED',
                        'error_message': error_msg,
                        'stage': 'S3_VERIFY',
                        's3_verification': verification,
                        'system_info': zip_context._get_system_info()
                    }
                raise ClientError({'Error': {'Code': 'VerificationFailed', 'Message': error_msg}}, 'upload_file')
                
        except ClientError as e:
            zip_label = zip_context.zip_name if zip_context else 'UNKNOWN'
            trace_label = zip_context.trace_id if zip_context else 'UNKNOWN'
            logger.error(f"Upload error for {filename}: {e}",
                        extra={'zip_name': zip_label, 'trace_id': trace_label, 'stage': 'S3_UPLOAD'})

            # Before alerting, confirm whether the file actually landed in S3
            logger.info(f"Bucket-check: verifying s3://{self.bucket_name}/{s3_key}",
                       extra={'zip_name': zip_label, 'trace_id': trace_label, 'stage': 'S3_VERIFY'})
            exists, size = verify_s3_object_exists(self.s3_client, self.bucket_name, s3_key)

            if exists:
                # File is in the bucket — upload succeeded despite the exception
                logger.info(f"File CONFIRMED in bucket (size={size}B) — treating as success: {s3_key}",
                           extra={'zip_name': zip_label, 'trace_id': trace_label, 'stage': 'S3_VERIFY'})
                if zip_context:
                    zip_context.log_s3_verification(s3_key, True, size)
                return s3_key

            # File is genuinely missing — send Slack alert with full logs
            logger.error(f"File CONFIRMED MISSING from s3://{self.bucket_name}/{s3_key} — sending Slack alert",
                        extra={'zip_name': zip_label, 'trace_id': trace_label, 'stage': 'S3_VERIFY'})
            if zip_context:
                zip_context.error_info = {
                    'trace_id': trace_label,
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'stage': 'S3_UPLOAD',
                    's3_verification': f'MISSING from s3://{self.bucket_name}/{s3_key}',
                    'system_info': zip_context._get_system_info()
                }
                zip_context.send_failure_alert()
            raise

    def upload_bytes_to_s3(self, data: bytes, folder_type: str, filename: str, 
                          content_type: str = "image/jpeg", zip_context: ZipProcessingContext = None) -> str:
        s3_key = self._get_s3_key(folder_type, filename)
        
        if zip_context:
            zip_context.log_s3_upload(f"bytes({len(data)})", s3_key, folder_type)
        
        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=data,
                ContentType=content_type,
            )
            
            exists, size = verify_s3_object_exists(self.s3_client, self.bucket_name, s3_key)
            if not exists:
                raise ClientError({'Error': {'Code': 'VerificationFailed', 'Message': 'Bytes upload not verified'}}, 'put_object')
                
            logger.info(f"Uploaded bytes {filename} to s3://{self.bucket_name}/{s3_key}",
                       extra={'zip_name': zip_context.zip_name if zip_context else 'UNKNOWN',
                              'trace_id': zip_context.trace_id if zip_context else 'UNKNOWN',
                              'stage': 'S3_UPLOAD'})
            return s3_key
        except ClientError as e:
            logger.error(f"Failed to upload bytes {filename} to S3: {e}",
                        extra={'zip_name': zip_context.zip_name if zip_context else 'UNKNOWN',
                               'trace_id': zip_context.trace_id if zip_context else 'UNKNOWN',
                               'stage': 'S3_UPLOAD'})
            raise

    def upload_json_to_s3(self, data: dict, filename: str = "detection_results.json",
                         zip_context: ZipProcessingContext = None) -> str:
        s3_key = f"CVLKRA_AI/{self.base_folder}/{filename}"
        json_bytes = json.dumps(data, indent=2).encode('utf-8')
        
        if zip_context:
            zip_context.log_s3_upload(f"JSON({len(json_bytes)} bytes)", s3_key, "json")
        
        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=json_bytes,
                ContentType="application/json",
            )
            logger.info(f"Uploaded JSON {filename} to s3://{self.bucket_name}/{s3_key}",
                       extra={'zip_name': zip_context.zip_name if zip_context else 'UNKNOWN',
                              'trace_id': zip_context.trace_id if zip_context else 'UNKNOWN',
                              'stage': 'S3_UPLOAD'})
            return s3_key
        except ClientError as e:
            logger.error(f"Failed to upload JSON {filename} to S3: {e}",
                        extra={'zip_name': zip_context.zip_name if zip_context else 'UNKNOWN',
                               'trace_id': zip_context.trace_id if zip_context else 'UNKNOWN',
                               'stage': 'S3_UPLOAD'})
            raise

    def download_from_s3(self, s3_path: str, zip_context: ZipProcessingContext = None) -> Path:
        if not self.s3_client or not self.temp_dir:
            raise ValueError("download_from_s3 requires bucket_name in __init__")
        if not s3_path.startswith("s3://"):
            raise ValueError(f"Invalid S3 path format: {s3_path}. Expected: s3://bucket-name/key")
            
        s3_path_parts = s3_path[5:].split("/", 1)
        if len(s3_path_parts) != 2:
            raise ValueError(f"Invalid S3 path format: {s3_path}")
            
        source_bucket = s3_path_parts[0]
        s3_key = s3_path_parts[1]
        filename = Path(s3_key).name
        local_path = self.temp_dir / filename
        
        if zip_context:
            zip_context._log('INFO', f"Downloading from s3://{source_bucket}/{s3_key}")
        
        logger.info(f"Downloading from s3://{source_bucket}/{s3_key} to {local_path}",
                   extra={'zip_name': zip_context.zip_name if zip_context else 'UNKNOWN',
                          'trace_id': zip_context.trace_id if zip_context else 'UNKNOWN',
                          'stage': 'S3_DOWNLOAD'})
        
        try:
            self.s3_client.download_file(source_bucket, s3_key, str(local_path))
            logger.info(f"Successfully downloaded {filename}",
                       extra={'zip_name': zip_context.zip_name if zip_context else 'UNKNOWN',
                              'trace_id': zip_context.trace_id if zip_context else 'UNKNOWN',
                              'stage': 'S3_DOWNLOAD'})
            return local_path
        except ClientError as e:
            logger.error(f"Failed to download from S3: {e}",
                        extra={'zip_name': zip_context.zip_name if zip_context else 'UNKNOWN',
                               'trace_id': zip_context.trace_id if zip_context else 'UNKNOWN',
                               'stage': 'S3_DOWNLOAD'})
            raise

    def cleanup_temp_dir(self):
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            logger.info(f"Cleaned up temp directory: {self.temp_dir}",
                       extra={'zip_name': 'SYSTEM', 'trace_id': self.instance_id, 'stage': 'CLEANUP'})

    def process_multiple(self, file_paths, results=None):
        if results is None:
            results = {"aadhaar": [], "pan": [], "uncertain": [], "user_image": []}

        if isinstance(file_paths, (str, Path)):
            file_paths = [file_paths]

        all_files = []

        for path_item in file_paths:
            path_item = Path(path_item)

            if not path_item.exists():
                logger.warning(f"Path not found: {path_item}",
                              extra={'zip_name': 'BATCH', 'trace_id': self.instance_id, 'stage': 'BATCH'})
                continue

            if path_item.is_file():
                all_files.append(path_item)
                logger.info(f"Added file: {path_item.name}",
                           extra={'zip_name': 'BATCH', 'trace_id': self.instance_id, 'stage': 'BATCH'})

            elif path_item.is_dir():
                supported_extensions = {".pdf", ".png", ".jpg", ".jpeg", ".zip"}
                folder_files = list(path_item.rglob("*"))
                pdf_images = [f for f in folder_files if f.suffix.lower() in supported_extensions]

                if pdf_images:
                    all_files.extend(pdf_images)
                    logger.info(f"Added {len(pdf_images)} files from folder: {path_item.name}",
                               extra={'zip_name': 'BATCH', 'trace_id': self.instance_id, 'stage': 'BATCH'})
                else:
                    logger.warning(f"No supported files found in: {path_item}",
                                  extra={'zip_name': 'BATCH', 'trace_id': self.instance_id, 'stage': 'BATCH'})
            else:
                logger.warning(f"Skipping invalid path: {path_item}",
                              extra={'zip_name': 'BATCH', 'trace_id': self.instance_id, 'stage': 'BATCH'})

        if not all_files:
            logger.error("No valid files found to process!",
                        extra={'zip_name': 'BATCH', 'trace_id': self.instance_id, 'stage': 'BATCH'})
            return results

        logger.info(f"Processing {len(all_files)} total files...",
                   extra={'zip_name': 'BATCH', 'trace_id': self.instance_id, 'stage': 'BATCH'})

        for i, file_path in enumerate(all_files, 1):
            logger.info(f"[{i}/{len(all_files)}] Processing: {file_path.name}",
                       extra={'zip_name': file_path.name, 'trace_id': self.instance_id, 'stage': 'BATCH'})

            if file_path.suffix.lower() == ".zip":
                try:
                    zip_results = self.process_zip_as_one(str(file_path))
                    if zip_results:
                        for key, val in zip_results.items():
                            results.setdefault(key, []).extend(val)
                except Exception as e:
                    logger.error(f"ZIP processing failed for {file_path.name}: {e}",
                                extra={'zip_name': file_path.name, 'trace_id': self.instance_id, 'stage': 'BATCH'})
            else:
                try:
                    self.process_file(file_path, results)
                except Exception as e:
                    logger.error(f"File processing failed for {file_path.name}: {e}",
                                extra={'zip_name': file_path.name, 'trace_id': self.instance_id, 'stage': 'BATCH'})

        # Save batch results
        results_path = self.output_folder / "detection_results.json"
        try:
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Batch results saved: {results_path}",
                       extra={'zip_name': 'BATCH', 'trace_id': self.instance_id, 'stage': 'BATCH'})
        except Exception as e:
            logger.error(f"Failed to save batch results: {e}",
                        extra={'zip_name': 'BATCH', 'trace_id': self.instance_id, 'stage': 'BATCH'})

        logger.info("=" * 60,
                   extra={'zip_name': 'BATCH', 'trace_id': self.instance_id, 'stage': 'BATCH'})
        logger.info("PROCESSING COMPLETE!",
                   extra={'zip_name': 'BATCH', 'trace_id': self.instance_id, 'stage': 'BATCH'})
        logger.info(f"Aadhaar: {len(results['aadhaar']):>3}",
                   extra={'zip_name': 'BATCH', 'trace_id': self.instance_id, 'stage': 'BATCH'})
        logger.info(f"PAN:    {len(results['pan']):>3}",
                   extra={'zip_name': 'BATCH', 'trace_id': self.instance_id, 'stage': 'BATCH'})
        logger.info(f"Uncertain: {len(results['uncertain']):>3}",
                   extra={'zip_name': 'BATCH', 'trace_id': self.instance_id, 'stage': 'BATCH'})
        logger.info(f"UserImage: {len(results['user_image']):>3}",
                   extra={'zip_name': 'BATCH', 'trace_id': self.instance_id, 'stage': 'BATCH'})
        total = sum(len(results[k]) for k in results)
        logger.info(f"TOTAL:  {total:>3}",
                   extra={'zip_name': 'BATCH', 'trace_id': self.instance_id, 'stage': 'BATCH'})
        logger.info("=" * 60,
                   extra={'zip_name': 'BATCH', 'trace_id': self.instance_id, 'stage': 'BATCH'})

        return results

    def _process_image_file(self, img_file, results, zip_context: ZipProcessingContext = None):
        if zip_context:
            zip_context._log('INFO', f"Processing image: {img_file.name}")
        else:
            print(f"\n Processing image: {img_file.name}")
            
        doc_type, score = self.classify_image(img_file)

        dest_folder = {
            "aadhaar": self.aadhar_folder,
            "pan": self.pan_folder,
            "uncertain": self.uncertain_folder,
            "user_image": self.user_image
        }[doc_type]

        dest_path = dest_folder / img_file.name
        shutil.copy2(img_file, dest_path)
        results.setdefault(doc_type, []).append({"file": img_file.name, "score": score})
        
        msg = f"{img_file.name} → {doc_type.upper()} (score={score})"
        if zip_context:
            zip_context._log('INFO', msg)
        else:
            print(f" {msg}")

    def _extract_zip(self, zip_path: str, zip_context: ZipProcessingContext = None) -> list[Path]:
        zip_path = Path(zip_path)

        try:
            result = subprocess.run(
                ["unzip", "-t", str(zip_path)],
                capture_output=True,
                text=True
            )
        except FileNotFoundError:
            msg = "unzip tool not found on system — please install unzip"
            if zip_context:
                zip_context._log('ERROR', msg)
            else:
                print(f" {msg}")
            return []

        if result.returncode != 0:
            msg = f"Not a valid or extractable ZIP: {zip_path}"
            if zip_context:
                zip_context._log('ERROR', msg)
                zip_context._log('ERROR', result.stderr.strip() or result.stdout.strip())
            else:
                print(f" {msg}")
                print(result.stderr.strip() or result.stdout.strip())
            return []

        msg = f"ZIP validation passed: {zip_path}"
        if zip_context:
            zip_context._log('INFO', msg)
        else:
            print(f" {msg}")

        temp_dir = Path(tempfile.mkdtemp(prefix="zip_extract_"))
        msg = f"Extracting ZIP to {temp_dir}"
        if zip_context:
            zip_context._log('INFO', msg)
        else:
            print(f" {msg}")

        try:
            subprocess.run(
                ["unzip", "-o", str(zip_path), "-d", str(temp_dir)],
                check=True
            )
        except subprocess.CalledProcessError as e:
            msg = f"Failed to extract ZIP: {e}"
            if zip_context:
                zip_context._log('ERROR', msg)
            else:
                print(f" {msg}")
            return []

        pdf_files = []
        for file_path in temp_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() == ".pdf":
                pdf_files.append(file_path)

        msg = f"{len(pdf_files)} PDF(s) found in the ZIP"
        if zip_context:
            zip_context.log_extraction(len(pdf_files))
        else:
            print(f" {msg}")
        return pdf_files

    def get_latest_valid_pdf(self, pdffiles, zip_context: ZipProcessingContext = None):
        import re
        from datetime import datetime

        always_ignore = ['SIG']
        conditional_ignore = ['APP', 'OTH']
        date_pattern = r'(\d{2})(\d{2})(\d{4})'
        
        aadhar_oth_pdfs = [pdf for pdf in pdffiles if 'AADHAR_OTH' in pdf.stem.upper()]
        pan_oth_pdfs = [pdf for pdf in pdffiles if 'PAN_OTH' in pdf.stem.upper()]
        all_oth_pdfs = aadhar_oth_pdfs + pan_oth_pdfs
        
        if zip_context:
            zip_context._log('INFO', f"PDF Selection: {len(pdffiles)} total, {len(aadhar_oth_pdfs)} AADHAR_OTH, {len(pan_oth_pdfs)} PAN_OTH")
        
        if all_oth_pdfs:
            oth_dates = []
            for pdf in all_oth_pdfs:
                m = re.search(date_pattern, pdf.stem)
                if m:
                    dd, mm, yyyy = m.groups()
                    try:
                        oth_dates.append(datetime(int(yyyy), int(mm), int(dd)))
                    except ValueError:
                        pass

            if oth_dates:
                latest_oth_date = max(oth_dates)
                plain_pdfs = [
                    pdf for pdf in pdffiles
                    if not any(kw in pdf.stem.upper() for kw in always_ignore + conditional_ignore)
                ]
                newer_plain = []
                for pdf in plain_pdfs:
                    m = re.search(date_pattern, pdf.stem)
                    if m:
                        dd, mm, yyyy = m.groups()
                        try:
                            dt = datetime(int(yyyy), int(mm), int(dd))
                            if dt > latest_oth_date:
                                newer_plain.append((dt, pdf))
                        except ValueError:
                            pass
                if newer_plain:
                    latest_newer = max(newer_plain, key=lambda x: x[0])[1]
                    msg = f"Newer plain PDF found → Returning: {latest_newer.name}"
                    if zip_context:
                        zip_context._log('INFO', msg)
                    else:
                        print(f"📄 {msg}")
                    return [latest_newer]
            
            if aadhar_oth_pdfs and pan_oth_pdfs:
                latest_aadhar = max(aadhar_oth_pdfs, key=lambda p: p.stat().st_mtime)
                latest_pan = max(pan_oth_pdfs, key=lambda p: p.stat().st_mtime)
                msg = f"AADHAR_OTH + PAN_OTH found → Returning: {latest_pan.name}, {latest_aadhar.name}"
                if zip_context:
                    zip_context._log('INFO', msg)
                else:
                    print(f"📄 {msg}")
                return [latest_pan, latest_aadhar]
        
        potential_pdfs = [pdf for pdf in pdffiles if not any(kw in pdf.stem.upper() for kw in always_ignore)]
        if not potential_pdfs:
            latest = max(pdffiles, key=lambda p: p.stat().st_mtime)
            msg = f"Only SIG PDFs; using newest: {latest.name}"
            if zip_context:
                zip_context._log('WARNING', msg)
            else:
                print(f" {msg}")
            return [latest]

        other_pdfs = [pdf for pdf in potential_pdfs if not any(kw in pdf.stem.upper() for kw in conditional_ignore)]
        if other_pdfs:
            date_candidates = []
            for pdf in other_pdfs:
                match = re.search(date_pattern, pdf.stem)
                if match:
                    dd, mm, yyyy = match.groups()
                    try:
                        dt = datetime(int(yyyy), int(mm), int(dd))
                        date_candidates.append((dt, pdf))
                    except ValueError:
                        pass
            if date_candidates:
                latest = max(date_candidates, key=lambda x: x[0])[1]
                msg = f"Selected LATEST OTHER by date: {latest.name}"
            else:
                latest = max(other_pdfs, key=lambda p: p.stat().st_mtime)
                msg = f"Selected LATEST OTHER by mod time: {latest.name}"
            if zip_context:
                zip_context._log('INFO', msg)
            else:
                print(f"✅ {msg}")
            return [latest]
        else:
            oth_pdfs = [pdf for pdf in potential_pdfs if 'OTH' in pdf.stem.upper()]
            app_pdfs = [pdf for pdf in potential_pdfs if 'APP' in pdf.stem.upper()]

            result = []
            if oth_pdfs:
                latest_oth = max(oth_pdfs, key=lambda p: p.stat().st_mtime)
                result.append(latest_oth)
                msg = f"Added LATEST OTH: {latest_oth.name}"
                if zip_context:
                    zip_context._log('INFO', msg)
                else:
                    print(f"📄 {msg}")

            if app_pdfs:
                latest_app = max(app_pdfs, key=lambda p: p.stat().st_mtime)
                result.append(latest_app)
                msg = f"Added LATEST APP: {latest_app.name}"
                if zip_context:
                    zip_context._log('INFO', msg)
                else:
                    print(f"📄 {msg}")

            if not result:
                latest = max(potential_pdfs, key=lambda p: p.stat().st_mtime)
                msg = f"No APP/OTH → Using: {latest.name}"
                if zip_context:
                    zip_context._log('INFO', msg)
                else:
                    print(f"📄 {msg}")
                return [latest]

            msg = f"Returning BOTH: {[p.name for p in result]}"
            if zip_context:
                zip_context._log('INFO', msg)
            else:
                print(f"✅ {msg}")
            return result

    def _already_processed(self, zippath: Path) -> bool:
        """
        Return True if this ZIP's output already exists, so we can skip re-processing.

        S3 mode  : checks for any object under CVLKRA_AI/<zip_name>/
        Local mode: checks whether the output subfolder exists and is non-empty.

        The zip name is stripped of leading/trailing whitespace before the
        comparison so a filename with accidental spaces never causes a false
        positive (skipping a ZIP that hasn't actually been processed).
        """
        zip_name = zippath.name.strip()

        if self.s3_client and self.bucket_name:
            prefix = f"CVLKRA_AI/{zip_name}/"
            try:
                resp = self.s3_client.list_objects_v2(
                    Bucket=self.bucket_name,
                    Prefix=prefix,
                    MaxKeys=1,
                )
                if resp.get("KeyCount", 0) > 0:
                    logger.info(
                        f"SKIP (already in S3): {prefix}",
                        extra={"zip_name": zip_name, "trace_id": "SKIP_CHECK", "stage": "DUPLICATE_CHECK"},
                    )
                    return True
            except Exception as e:
                logger.warning(
                    f"S3 duplicate-check failed for {zip_name}, will process anyway: {e}",
                    extra={"zip_name": zip_name, "trace_id": "SKIP_CHECK", "stage": "DUPLICATE_CHECK"},
                )
        else:
            zip_root = self.output_folder / zippath.stem.strip()
            if zip_root.exists() and any(zip_root.rglob("*")):
                logger.info(
                    f"SKIP (local output exists): {zip_root}",
                    extra={"zip_name": zip_name, "trace_id": "SKIP_CHECK", "stage": "DUPLICATE_CHECK"},
                )
                return True

        return False

    def process_zip_as_one(self, zip_paths: str, dpi=300, s3_bucket=None, s3_base_prefix=""):
        if isinstance(zip_paths, (str, Path)):
            zip_paths = [zip_paths]

        overall_results = {'aadhaar': [], 'pan': [], 'uncertain': [], 'userimage': []}

        for zippath in zip_paths:
            zippath = Path(zippath)
            zip_name = zippath.name.strip()  # normalise once; use everywhere below

            # Warn if the filename had stray whitespace (would cause a wrong S3 key)
            if zip_name != zippath.name:
                logger.warning(
                    f"ZIP filename had whitespace — normalised: '{zippath.name}' → '{zip_name}'",
                    extra={"zip_name": zip_name, "trace_id": "NORMALISE", "stage": "INPUT_CHECK"},
                )
                zippath = zippath.parent / zip_name


            ctx = None  # ← Track if context started
            try:
                # Use context manager for full lifecycle tracking
                with ZipProcessingContext(zip_name, self) as ctx:
                    try:
                        ctx._log('INFO', f"Starting ZIP: {zippath.name}")
                        
                        # Root folder for this ZIP
                        zip_root = self.output_folder / zippath.stem
                        if zip_root.exists():
                            shutil.rmtree(zip_root, ignore_errors=True)
                            ctx._log('INFO', f"Cleaned existing zip_root: {zip_root}")
                        zip_root.mkdir(parents=True, exist_ok=True)

                        # Per-ZIP folders
                        self.aadhar_folder   = zip_root / "aadhar_cards"
                        self.pan_folder      = zip_root / "pan_cards"
                        self.uncertain_folder = zip_root / "uncertain"
                        self.user_image      = zip_root / "user_image"

                        for folder in [
                            self.aadhar_folder,
                            self.pan_folder,
                            self.uncertain_folder,
                            self.user_image,
                        ]:
                            folder.mkdir(parents=True, exist_ok=True)
                        ctx._log('INFO', f"Created per-ZIP folders in {zip_root}")

                        # ====== NESTED ZIP CHECK ======
                        ctx.transition_to('EXTRACTION', "Checking for nested ZIPs...")
                        with zipfile.ZipFile(zippath, 'r') as main_zip:
                            nested_zips = [f for f in main_zip.namelist() if f.lower().endswith('.zip')]

                            if nested_zips:
                                ctx._log('INFO', f"Found {len(nested_zips)} nested ZIPs")
                                for nz in nested_zips:
                                    ctx._log('INFO', f"  Nested: {nz}")

                                date_pattern = r'(\d{2})(\d{2})(\d{4})'
                                nested_zip_dates = []
                                for nz_name in nested_zips:
                                    match = re.search(date_pattern, nz_name)
                                    if match:
                                        dd, mm, yyyy = match.groups()
                                        try:
                                            dt = datetime(int(yyyy), int(mm), int(dd))
                                            nested_zip_dates.append((dt, nz_name))
                                        except ValueError:
                                            pass

                                if nested_zip_dates:
                                    latest_nested_zip = max(nested_zip_dates, key=lambda x: x[0])[1]
                                    ctx._log('INFO', f"Selected LATEST nested ZIP by date: {latest_nested_zip}")
                                else:
                                    latest_nested_zip = nested_zips[-1]
                                    ctx._log('INFO', f"Selected nested ZIP (no date): {latest_nested_zip}")

                                temp_dir = self.output_folder / f"temp_nested_{zippath.stem}"
                                temp_dir.mkdir(exist_ok=True)
                                nested_zip_data = main_zip.read(latest_nested_zip)
                                with zipfile.ZipFile(io.BytesIO(nested_zip_data)) as latest_nested:
                                    latest_nested.extractall(temp_dir)
                                pdffiles = list(temp_dir.glob("*.pdf"))
                                ctx.log_extraction(len(pdffiles), nested=True)
                            
                            else:
                                pdffiles = self._extract_zip(zippath, ctx)

                            ctx._log('INFO', f"ALL PDFs ({len(pdffiles)}):")
                            for i, pdf in enumerate(pdffiles, 1):
                                ctx._log('INFO', f"  {i}. {pdf.name}")

                            selected_pdfs = self.get_latest_valid_pdf(pdffiles, ctx)
                            if 'temp_dir' in locals() and temp_dir.exists():
                                shutil.rmtree(temp_dir, ignore_errors=True)
                                ctx._log('INFO', "Cleaned temp nested dir")

                            if not selected_pdfs:
                                ctx._log('WARNING', f"No valid PDF in {zippath.name}, skipping.")
                                ctx.error_info = {
                                    'trace_id': ctx.trace_id,
                                    'error_type': 'NO_VALID_PDF',
                                    'error_message': f"No valid PDF selected from {zippath.name}",
                                    'stage': 'PDF_SELECTION',
                                }
                                ctx.send_failure_alert()
                                continue

                        ctx._log('INFO', f"Selected {len(selected_pdfs)} PDF(s): {[p.name for p in selected_pdfs]}")
                        existing_scores = {}
                        globalbestface = None
                        
                        ctx.transition_to('PROCESSING', f"Processing {len(selected_pdfs)} PDFs...")
                        
                        for latest_pdf in selected_pdfs:
                            ctx.log_processing(latest_pdf.name)
                            pagenum = 1

                            while True:
                                try:
                                    singlepage = convert_from_path(str(latest_pdf), dpi=dpi, first_page=pagenum, last_page=pagenum)
                                except Exception as e:
                                    ctx._log('ERROR', f"convert_from_path failed on page {pagenum}: {e}")
                                    ctx.error_info = {
                                        'trace_id': ctx.trace_id,
                                        'error_type': type(e).__name__,
                                        'error_message': f"PDF conversion failed at page {pagenum} of {latest_pdf.name}: {e}",
                                        'stage': 'PDF_CONVERSION',
                                        'traceback': traceback.format_exc(),
                                    }
                                    ctx.send_failure_alert()
                                    break
                                if not singlepage:
                                    break

                                page = singlepage[0]
                                imgbgr = pil_to_cv2(page)

                                try:
                                    yoloresults = model(imgbgr)
                                except Exception as e:
                                    ctx._log('ERROR', f"YOLO inference failed: {e}")
                                    ctx.error_info = {
                                        'trace_id': ctx.trace_id,
                                        'error_type': type(e).__name__,
                                        'error_message': f"YOLO failed on page {pagenum} of {latest_pdf.name}: {e}",
                                        'stage': 'YOLO_INFERENCE',
                                        'traceback': traceback.format_exc(),
                                    }
                                    ctx.send_failure_alert()
                                    break
                                if len(yoloresults[0].boxes) > 0:
                                    pagebest = max(yoloresults[0].boxes, key=lambda b: float(b.conf[0]))
                                
                                    conf = float(pagebest.conf[0])
                                    x1, y1, x2, y2 = map(int, pagebest.xyxy[0])
                                    ctx._log('INFO', f"{latest_pdf.name} P{pagenum} Conf:{conf:.2f}")
                                    if globalbestface is None or conf > globalbestface[0]:
                                        globalbestface = (conf, latest_pdf, pagenum, x1, y1, x2, y2, imgbgr)

                                imgname = f"{latest_pdf.stem}_page{pagenum}.jpg"
                                imgpath = self.output_folder / imgname
                                page.save(imgpath, "JPEG")

                                try:
                                    doctype, score = classify(imgpath)
                                except Exception as e:
                                    ctx._log('ERROR', f"Classification failed for {imgname}: {e}")
                                    ctx.error_info = {
                                        'trace_id': ctx.trace_id,
                                        'error_type': type(e).__name__,
                                        'error_message': f"LLM classify() failed for {imgname}: {e}",
                                        'stage': 'LLM_CLASSIFICATION',
                                        'traceback': traceback.format_exc(),
                                    }
                                    ctx.send_failure_alert()
                                    doctype, score = 'uncertain', 0

                                ctx.log_classification(doctype, score, "evaluating...")

                                if doctype == 'userimage':
                                    ctx._log('INFO', "Face skip - YOLO handles face.")
                                    imgpath.unlink(missing_ok=True)
                                elif score < 90:
                                    ctx._log('INFO', f"Skip score <90")
                                    imgpath.unlink(missing_ok=True)
                                else:
                                    if doctype not in existing_scores or score > existing_scores[doctype][1]:
                                        if doctype in existing_scores:
                                            old_path = existing_scores[doctype][0]
                                            if old_path.exists():
                                                old_path.unlink()
                                                ctx._log('INFO', f"Replaced old {doctype}: {existing_scores[doctype][1]}")

                                        existing_scores[doctype] = (imgpath, score)
                                        destfolder = {
                                            'aadhaar': self.aadhar_folder,
                                            'pan': self.pan_folder,
                                            'uncertain': self.uncertain_folder,
                                            'userimage': self.user_image
                                        }[doctype]
                                        destpath = destfolder / imgname
                                        shutil.move(str(imgpath), destpath)
                                        overall_results.setdefault(doctype, []).append((imgname, score))
                                        ctx._log('INFO', f"{doctype.upper()} {score} - KEPT")
                                    else:
                                        ctx._log('INFO', f"{doctype.upper()} {score} < BEST {existing_scores[doctype][1]} - DISCARDED")
                                        imgpath.unlink(missing_ok=True)

                                pagenum += 1

                            # Save best face for this PDF
                            if globalbestface:
                                conf, bestpdf, bestpage, x1, y1, x2, y2, imgbgr = globalbestface
                                crop = imgbgr[y1:y2, x1:x2]
                                outname = f"{zippath.stem}_bestface_{bestpdf.stem}_p{bestpage:03d}_conf{conf:.2f}.jpg"
                                outpath = self.user_image / outname
                                cv2.imwrite(str(outpath), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                                overall_results['userimage'].append((outname, 'bestface_yolo'))
                                ctx._log('INFO', f"Saved global best face: {outname}")
                            else:
                                ctx._log('WARNING', "No faces found in PDF.")

                        # ====== S3 UPLOAD PHASE ======
                        ctx.transition_to('S3_UPLOAD', "Starting S3 upload phase...")
                        valid_card_files = list(self.aadhar_folder.glob("*")) + list(self.pan_folder.glob("*"))
                        ctx._log('INFO', f"Valid card files to upload: {len(valid_card_files)}")

                        if len(valid_card_files) > 0:
                            if self.s3_client:
                                self.base_folder = zip_name  # already stripped
                                uploaded_files = []
                                failed_uploads = []
                                
                                for root, _, files in os.walk(zip_root):
                                    for file in files:
                                        local_path = Path(root) / file
                                        rel_path = local_path.relative_to(zip_root)
                                        folder_type = rel_path.parts[0] if len(rel_path.parts) > 1 else ""
                                        filename = rel_path.name
                                        
                                        try:
                                            s3_key = self.upload_file_to_s3(local_path, folder_type, filename, zip_context=ctx)
                                            uploaded_files.append((str(local_path), s3_key))
                                        except Exception as e:
                                            failed_uploads.append((str(local_path), str(e)))
                                            ctx._log('ERROR', f"Upload failed for {filename}: {e}")


                                # Verify all uploads
                                ctx.transition_to('VERIFICATION', "Verifying all S3 uploads...")
                                all_verified = True
                                for local_path_str, s3_key in uploaded_files:
                                    local_p = Path(local_path_str)
                                    verification = verify_s3_upload(self.s3_client, self.bucket_name, s3_key, local_p)
                                    if not verification['verified']:
                                        all_verified = False
                                        ctx.log_s3_verification(s3_key, False)
                                        ctx._log('ERROR', f"S3 verification FAILED for {s3_key}: {verification['errors']}")
                                        failed_uploads.append((local_path_str, f"Verification failed: {verification['errors']}"))
                                    else:
                                        ctx.log_s3_verification(s3_key, True, verification['s3_size'])

                                if failed_uploads:
                                    error_details = {
                                        'trace_id': ctx.trace_id,
                                        'error_type': 'S3_PARTIAL_UPLOAD_FAILURE',
                                        'error_message': f"{len(failed_uploads)} of {len(uploaded_files) + 1} uploads failed",
                                        'stage': 'S3_VERIFY',
                                        'failed_uploads': failed_uploads,
                                        'successful_uploads': len(uploaded_files),
                                        'system_info': ctx._get_system_info()
                                    }
                                    ctx.error_info = error_details
                                    ctx.send_failure_alert()
                                    raise Exception(f"S3 upload verification failed for {len(failed_uploads)} files")
                                    
                                ctx._log('INFO', f"All {len(uploaded_files)} files verified in S3")
                                
                            else:
                                # Legacy path Uploads to S3 without using self.s3_client Why is it there?	For cases where detector has no S3 config at initialization
                                if not s3_bucket:
                                    raise ValueError("s3_bucket must be provided when detector has no bucket_name")
                                s3_prefix = f"{s3_base_prefix.rstrip('/')}/{zippath.stem}/"
                                failed_uploads = []  # Track legacy failures too
                                for root, _, files in os.walk(zip_root):
                                    for file in files:
                                        local_path = Path(root) / file
                                        rel_path = local_path.relative_to(zip_root)
                                        s3_key = f"CVLKRA_AI/{s3_prefix}{rel_path.as_posix()}"
                                        try:
                                            upload_to_s3(local_path, s3_bucket, s3_key)
                                        except Exception as e:
                                            ctx._log('ERROR', f"Legacy upload failed for {file}: {e}")
                                            failed_uploads.append((str(local_path), str(e)))
                                if failed_uploads:
                                    ctx.error_info = {
                                        'trace_id': ctx.trace_id,
                                        'error_type': 'S3_LEGACY_UPLOAD_FAILURE',
                                        'error_message': f"{len(failed_uploads)} legacy uploads failed",
                                        'stage': 'S3_UPLOAD',
                                        'failed_uploads': failed_uploads,
                                    }
                                    ctx.send_failure_alert()
                                    raise Exception(f"Legacy S3 upload failed for {len(failed_uploads)} files")
                                ctx._log('INFO', f"Uploaded folder {zip_root.name} to S3 (legacy mode)")
                        else:
                            msg = f"No valid Aadhaar/PAN card images found for {zippath.name}"
                            ctx._log('WARNING', msg)
                            ctx.error_info = {
                                'trace_id': ctx.trace_id,
                                'error_type': 'NO_VALID_CARDS',
                                'error_message': msg,
                                'stage': 'CLASSIFICATION',
                            }
                            ctx.send_failure_alert()
                        
                        # ====== CLEANUP ======
                        ctx.transition_to('CLEANUP', "Cleaning up local files...")
                        if zip_root.exists():
                            shutil.rmtree(zip_root, ignore_errors=True)
                            ctx._log('INFO', f"Local folder removed: {zip_root}")
                            
                    except Exception as e:
                        ctx._log('ERROR', f"Exception in process_zip_as_one: {type(e).__name__}: {e}")
                        ctx.log_error(e, ctx.current_stage)
                        ctx.send_failure_alert()
                        raise  # Re-raise to propagate
            except Exception as e:
                if ctx is None:
                    # ✅ __enter__ FAILED or exception before context started
                    slack_alert.send_alert(
                        zip_name=zip_name,
                        error_details={
                            'trace_id': 'NO_CTX',
                            'error_type': type(e).__name__,
                            'error_message': str(e),
                            'stage': 'CONTEXT_ENTER',
                            'traceback': traceback.format_exc(),
                        }
                    )
                # If ctx exists, alert was already sent inside. Just re-raise.
                raise

            logger.info(
                "Pausing 3s before next ZIP...",
                extra={"zip_name": zip_name, "trace_id": "PAUSE", "stage": "INTER_ZIP_PAUSE"},
            )
            time.sleep(3)

        return overall_results





