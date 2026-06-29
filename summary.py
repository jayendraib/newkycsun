"""PDF summarization service."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import tempfile
import traceback
import unicodedata
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiofiles
import httpx
from dotenv import load_dotenv
from langchain.chains.summarize import load_summarize_chain
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

if TYPE_CHECKING:
    from langchain_core.documents import Document

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)

# LLM setup
llm = ChatOllama(model="gemma3:4b", temperature=0.0, num_ctx=100000)

# Prompt template
reduce_template = ChatPromptTemplate.from_template(
    """You are a helpful assistant.
If no chunks or text are available, respond with:
Sorry we are unable to summarise this pdf.
Otherwise, combine the provided chunk summaries into one final summary without any headers, labels, or extra formatting. Output only the pure summary text.
{text}"""
)

# Headers for PDF download requests
PDF_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,image/apng,*/*;q=0.8,application/pdf;q=0.9"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://nsearchives.nseindia.com/",
}


async def download_pdf(
    url: str,
    out_dir: str | None = None,
    retries: int = 3,
    backoff_factor: float = 1.5,
    connect_timeout: float = 30.0,
    read_timeout: float = 120.0,
    write_timeout: float = 30.0,
    pool_timeout: float = 60.0,
) -> str:
    """Download a PDF from a URL with retry logic.

    Args:
        url: The URL to download the PDF from.
        out_dir: Optional directory to save the file.
        retries: Number of retry attempts.
        backoff_factor: Exponential backoff multiplier.
        connect_timeout: Connection timeout in seconds.
        read_timeout: Read timeout in seconds.
        write_timeout: Write timeout in seconds.
        pool_timeout: Pool timeout in seconds.

    Returns:
        Path to the downloaded PDF file.

    Raises:
        Exception: If all retry attempts fail.
    """
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf",
        dir=out_dir,
    ) as tmp:
        tmp_path = tmp.name

    timeout = httpx.Timeout(
        connect=connect_timeout,
        read=read_timeout,
        write=write_timeout,
        pool=pool_timeout,
    )

    for attempt in range(1, retries + 1):
        try:
            async with (
                httpx.AsyncClient(headers=PDF_HEADERS, timeout=timeout) as client,
                client.stream("GET", url) as resp,
            ):
                resp.raise_for_status()
                async with aiofiles.open(tmp_path, "wb") as f:
                    async for chunk in resp.aiter_bytes():
                        await f.write(chunk)

            size = Path(tmp_path).stat().st_size
            if size < 200:
                logger.warning(
                    f"Downloaded file likely incomplete ({size} bytes): {tmp_path}",
                )

            return tmp_path

        except Exception as e:
            logger.warning(f"Attempt {attempt}/{retries} failed: {e}")
            if attempt == retries:
                with contextlib.suppress(Exception):
                    Path(tmp_path).unlink()
                raise
            delay = backoff_factor**attempt
            logger.info(f"Retrying in {delay} seconds...")
            await asyncio.sleep(delay)

    return tmp_path


async def load_and_chunk(
    pdf_path: str,
    chunk_size: int = 2500,
    chunk_overlap: int = 250,
) -> list[Document]:
    """Load a PDF and split it into chunks.

    Args:
        pdf_path: Path to the PDF file.
        chunk_size: Maximum size of each chunk.
        chunk_overlap: Overlap between chunks.

    Returns:
        List of document chunks.
    """
    logger.info("Loading and chunking PDF...")
    loop = asyncio.get_event_loop()
    loader = PyPDFLoader(pdf_path)
    docs = await loop.run_in_executor(None, loader.load)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunked = splitter.split_documents(docs)
    logger.info(f"Split into {len(chunked)} chunks")
    return chunked


async def summarize_docs(docs: list[Document]) -> str:
    """Summarize a list of documents using LLM.

    Args:
        docs: List of document chunks to summarize.

    Returns:
        The summarized text.
    """
    logger.info("Running summarization chain...")
    loop = asyncio.get_event_loop()
    chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        verbose=True,
        combine_prompt=reduce_template,
    )
    return await loop.run_in_executor(None, chain.run, docs)


async def process_url(url: str) -> dict[str, Any]:
    """Process a URL to download and summarize the PDF.

    Args:
        url: The URL of the PDF to process.

    Returns:
        Dictionary containing URL, summary, or error information.
    """
    rec: dict[str, Any] = {"URL": url}
    pdf_path = None

    try:
        logger.info("Starting PDF summarization pipeline...")
        pdf_path = await download_pdf(url)
        rec["file_path"] = pdf_path

        docs = await load_and_chunk(pdf_path)
        summary = await summarize_docs(docs)
        clean_summary = unicodedata.normalize("NFKC", summary).replace("\n", " ")

        rec["summary"] = clean_summary
        logger.info(f"Summary length: {len(clean_summary)} characters")

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Error processing {url}: {e}\n{tb}")
        rec["error"] = f"{e}\n{tb}"

    finally:
        if pdf_path:
            try:
                Path(pdf_path).unlink()
                logger.info(f"Deleted {pdf_path}")
            except Exception as e:
                logger.warning(f"Could not delete {pdf_path}: {e}")

    return rec