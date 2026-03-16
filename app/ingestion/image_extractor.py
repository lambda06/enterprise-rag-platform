"""
Image extraction for the Enterprise RAG Platform.

Extracts embedded images from PDF pages using PyMuPDF (fitz) and produces
768-dim Gemini embeddings for each image directly from the PIL Image object.

Architectural note — why there is NO captioning step
------------------------------------------------------
In a traditional multimodal RAG pipeline, images must be converted to text
(via a vision LLM like GPT-4o) before embedding, because text-only models
cannot represent images.  The caption then becomes the proxy for the image
in the vector store, losing visual detail and adding LLM latency/cost.

``gemini-embedding-2-preview`` is a *natively multimodal* model: it maps
both images and text into the **same** unified 3072-dim vector space (MRL-
truncated to 768 dims here).  This means:
  - A PIL Image can be passed *directly* to ``EmbeddingService.embed_image()``.
  - The resulting image vector sits in the same space as text query vectors.
  - A text query such as "bar chart of quarterly revenue" will correctly
    retrieve a stored image of that chart without any caption acting as an
    intermediary.

This eliminates the vision-LLM captioning step entirely, cutting one extra
API round-trip per image and avoiding quality loss from imperfect captions.

Requires: pip install pymupdf Pillow
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import numpy as np
from PIL import Image

from app.rag.embeddings import get_embedding_service

logger = logging.getLogger(__name__)


class ImageExtractor:
    """Extract and embed images from PDF files.

    Iterates through every page of the PDF, extracts each embedded image
    larger than ``MIN_WIDTH × MIN_HEIGHT`` pixels, embeds it directly with
    Gemini (no caption intermediary), and returns a list of image records
    ready for upsert into Qdrant alongside text chunk records.

    Usage::

        extractor = ImageExtractor()
        records = extractor.extract("report.pdf")
        # Each record has: page_number, image_index, image_base64,
        #                   embedding (np.ndarray), content_type, metadata.

    Attributes:
        MIN_WIDTH:  Minimum image width in pixels to include (default 100).
        MIN_HEIGHT: Minimum image height in pixels to include (default 100).
    """

    MIN_WIDTH: int = 100
    MIN_HEIGHT: int = 100

    def __init__(self, embed_service=None) -> None:
        """Initialise the extractor.

        Args:
            embed_service: Optional ``EmbeddingService`` instance.  If None,
                the module-level singleton (``get_embedding_service()``) is
                used.  Pass a mock here in unit tests to avoid real API calls.
        """
        self._embed = embed_service or get_embedding_service()

    # ── Public API ────────────────────────────────────────────────────────────

    def extract(
        self,
        file_path: str | Path,
        source_filename: str | None = None,
    ) -> list[dict[str, Any]]:
        """Extract and embed all qualifying images from a PDF.

        For each page, ``page.get_images(full=True)`` returns a list of
        (xref, …) tuples referencing embedded image objects.  The raw bytes
        are retrieved via ``doc.extract_image(xref)``, decoded into a PIL
        Image, filtered by minimum size, and passed directly to
        ``EmbeddingService.embed_image()`` to obtain a 768-dim vector.

        A corrupt or unreadable image on one page never aborts the rest —
        errors are caught per-image and logged at WARNING level.

        Args:
            file_path:       Path to the PDF file (str or Path).
            source_filename: Human-readable name stored in metadata.
                             Defaults to the file's basename.

        Returns:
            List of dicts, one per qualifying image, with keys:

            - ``page_number``  (int)      — 1-based page index.
            - ``image_index``  (int)      — 0-based image index on that page.
            - ``image_base64`` (str)      — Base64-encoded PNG bytes (for
                                            display or re-embedding later).
            - ``embedding``    (np.ndarray) — 768-dim L2-normalised float32
                                              vector from ``embed_image()``.
            - ``content_type`` (str)      — Always ``"image"``.
            - ``metadata``     (dict)     — source_filename, page_number,
                                            image_index, content_type.

        Raises:
            FileNotFoundError: If ``file_path`` does not exist.
            ValueError:        If the file cannot be opened as a PDF.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")

        display_name = source_filename or path.name
        records: list[dict[str, Any]] = []

        try:
            doc = fitz.open(path)
        except Exception as exc:
            raise ValueError(f"Could not open PDF '{path}': {exc}") from exc

        try:
            total_pages = len(doc)
            logger.info(
                "ImageExtractor: scanning %d pages of '%s' for images",
                total_pages,
                display_name,
            )

            for page_index in range(total_pages):
                page = doc[page_index]
                page_number = page_index + 1  # 1-based for human readability

                # get_images(full=True) returns a list of tuples:
                # (xref, smask, width, height, bpc, colorspace, alt_cs, name, filter, referencer)
                image_list = page.get_images(full=True)

                for image_index, img_info in enumerate(image_list):
                    try:
                        record = self._process_image(
                            doc=doc,
                            img_info=img_info,
                            page_number=page_number,
                            image_index=image_index,
                            source_filename=display_name,
                        )
                        if record is not None:
                            records.append(record)
                    except Exception as exc:
                        # Per-image error — log and continue with the next
                        logger.warning(
                            "ImageExtractor: skipping image %d on page %d of '%s': %s",
                            image_index,
                            page_number,
                            display_name,
                            exc,
                        )

        finally:
            doc.close()

        logger.info(
            "ImageExtractor: extracted %d qualifying images from '%s'",
            len(records),
            display_name,
        )
        return records

    # ── Private helpers ───────────────────────────────────────────────────────

    def _process_image(
        self,
        doc: fitz.Document,
        img_info: tuple,
        page_number: int,
        image_index: int,
        source_filename: str,
    ) -> dict[str, Any] | None:
        """Process a single image reference and return a record dict or None.

        Steps:
        1. Extract raw image bytes from the PDF xref table.
        2. Decode bytes into a PIL Image.
        3. Apply the minimum-size filter (100×100 px).
        4. Embed the PIL Image directly via ``EmbeddingService.embed_image()``.
        5. Base64-encode the image (PNG) for storage.

        Args:
            doc:             Open PyMuPDF document.
            img_info:        Tuple from ``page.get_images(full=True)``.
            page_number:     1-based page number.
            image_index:     0-based index of this image within the page.
            source_filename: Display name of the source PDF.

        Returns:
            Dict with image record fields, or ``None`` if the image is below
            the minimum size threshold.
        """
        xref = img_info[0]  # xref is always the first element

        # Extract raw image data dict: keys include 'image', 'width', 'height', etc.
        raw = doc.extract_image(xref)
        width: int = raw["width"]
        height: int = raw["height"]

        # ── Size filter: discard decorative or icon-sized images ─────────────
        if width < self.MIN_WIDTH or height < self.MIN_HEIGHT:
            logger.debug(
                "Skipping image %d on page %d (%dx%d < %dx%d threshold)",
                image_index,
                page_number,
                width,
                height,
                self.MIN_WIDTH,
                self.MIN_HEIGHT,
            )
            return None

        # ── Decode raw bytes → PIL Image ──────────────────────────────────────
        image_bytes: bytes = raw["image"]
        pil_image: Image.Image = Image.open(io.BytesIO(image_bytes))

        # ── Embed directly via Gemini — no captioning step needed ─────────────
        # Because gemini-embedding-2-preview maps images and text into the same
        # shared vector space, we can pass the PIL Image object directly to
        # embed_image(). The resulting 768-dim vector is directly comparable to
        # RETRIEVAL_QUERY text vectors at search time — cross-modal retrieval
        # works out of the box without a vision LLM generating an intermediate
        # text caption first.
        embedding: np.ndarray = self._embed.embed_image(pil_image)

        # ── Base64-encode as PNG for storage ──────────────────────────────────
        buf = io.BytesIO()
        # Convert to RGB first to ensure consistent PNG export (RGBA, P mode, etc.)
        pil_image.convert("RGB").save(buf, format="PNG")
        image_base64: str = base64.b64encode(buf.getvalue()).decode("utf-8")

        metadata = {
            "source_filename": source_filename,
            "page_number": page_number,
            "image_index": image_index,
            "content_type": "image",
        }

        return {
            "page_number": page_number,
            "image_index": image_index,
            "image_base64": image_base64,
            "embedding": embedding,
            "content_type": "image",
            "metadata": metadata,
        }
