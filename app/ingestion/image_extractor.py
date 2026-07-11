"""
Image extraction and captioning for the Enterprise RAG Platform.

Extracts embedded images from document pages, generates high-fidelity visual
captions using Gemini (Vision-to-Text), and embeds those captions as standard
text vectors. The raw base64 image bytes are stored directly inside the Qdrant
payload (`metadata.image_base64`) so multimodal LLMs can inspect the visual data
at answer generation time.

Architectural advantage of Vision-to-Text + Payload Storage
-----------------------------------------------------------
1. **Cross-Encoder Reranker Compatibility**: Standard cross-encoders (Jina)
   require (query_text, document_text) pairs. Captions enable accurate Stage 2
   reranking where raw image vectors (`embed_image()`) would fail or score 0.
2. **BM25 Sparse Keyword Indexing**: Generating a rich text summary of charts,
   axis labels, legends, and numbers makes every visual detail searchable via
   exact keyword matching.
3. **Payload Efficiency**: Single text embedding vector per image chunk keeps
   vector search fast, while preserving full visual fidelity via base64 payload.

Requires: pip install pymupdf Pillow google-genai
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

        # ── Base64-encode as PNG for storage in payload ───────────────────────
        buf = io.BytesIO()
        pil_image.convert("RGB").save(buf, format="PNG")
        image_base64: str = base64.b64encode(buf.getvalue()).decode("utf-8")

        # ── Generate Vision Caption via Gemini 2.5 Flash Lite ─────────────────
        caption: str = self._generate_image_caption(pil_image)

        # ── Embed the caption directly as standard text vector ────────────────
        # This ensures BM25 sparse keyword search and Jina Cross-Encoder reranking
        # work natively over the rich text description, while the raw base64
        # bytes remain inside the payload for multimodal LLM generation.
        embedding: np.ndarray = self._embed.embed_chunks([caption])[0]

        metadata = {
            "source_filename": source_filename,
            "page_number": page_number,
            "image_index": image_index,
            "content_type": "image",
            "image_base64": image_base64,
            "text": caption,
        }

        return {
            "page_number": page_number,
            "image_index": image_index,
            "image_base64": image_base64,
            "text": caption,
            "embedding": embedding,
            "content_type": "image",
            "metadata": metadata,
        }

    def _generate_image_caption(self, pil_image: Image.Image) -> str:
        """Generate a natural language description/caption for the image using Gemini 2.5 Flash Lite.

        Args:
            pil_image: The decoded PIL Image object.

        Returns:
            A descriptive text string for embedding, keyword search, and reranking.
        """
        try:
            from google import genai
            from app.core.config import get_settings

            settings = get_settings()
            if not settings.gemini.api_key:
                logger.debug("ImageExtractor: No Gemini API key configured; returning structural fallback caption.")
                return f"Embedded diagram or chart ({pil_image.width}x{pil_image.height} px)."

            client = genai.Client(api_key=settings.gemini.api_key)
            prompt = (
                "Describe this image, chart, or diagram in 3-4 detailed sentences. "
                "Include all visible numbers, axis labels, legends, titles, and key takeaways."
            )
            response = client.models.generate_content(
                model=settings.gemini.generation_model or "gemini-3.1-flash-lite-preview",
                contents=[prompt, pil_image],
            )
            caption = (response.text or "").strip()
            return caption if caption else f"Embedded visual element ({pil_image.width}x{pil_image.height} px)."
        except Exception as exc:
            logger.warning("ImageExtractor: caption generation failed (%s). Using structural fallback.", exc)
            return f"Embedded diagram or chart ({pil_image.width}x{pil_image.height} px)."
