"""
Unit tests for ImageExtractor.

All file I/O (fitz.open) and Gemini API calls (EmbeddingService.embed_image)
are fully mocked — no real PDF file or API key is required.

Tests verify:
  - Images ≥ 100×100 px are extracted and returned as records.
  - Images < 100×100 px are silently filtered out.
  - Returned records contain all required keys with correct types.
  - Embedding shape is (768,) float32.
  - image_base64 decodes to valid PNG bytes.
  - A bad image (embed_image raises) is skipped without crashing.
  - A PDF with no images returns an empty list.
"""

from __future__ import annotations

import base64
import io
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image


# ─── Helpers ──────────────────────────────────────────────────────────────────

_DIM = 768


def _fake_embedding() -> np.ndarray:
    """Return a synthetic unit-norm embedding vector."""
    v = np.ones(_DIM, dtype=np.float32)
    return v / np.linalg.norm(v)


def _pil_to_png_bytes(width: int = 120, height: int = 120) -> bytes:
    """Build raw PNG bytes for a synthetic image of the given size."""
    img = Image.new("RGB", (width, height), color=(200, 100, 50))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_extract_image_payload(width: int, height: int) -> dict:
    """Fake return value for ``doc.extract_image(xref)``."""
    return {
        "image": _pil_to_png_bytes(width, height),
        "width": width,
        "height": height,
        "ext": "png",
    }


def _make_fitz_doc(images_per_page: list[list[tuple]]) -> MagicMock:
    """
    Build a minimal fitz.Document mock.

    Args:
        images_per_page: A list (one entry per page) of lists of
            ``(xref, …)`` tuples as returned by ``page.get_images(full=True)``.
            Each tuple's first element (xref) is used as the key to look up
            the extract_image payload; pass a dict with matching xrefs via
            extract_image_payloads configured on the doc mock directly.
    """
    pages = []
    for page_images in images_per_page:
        page = MagicMock()
        page.get_images.return_value = page_images
        pages.append(page)

    doc = MagicMock()
    doc.__len__ = MagicMock(return_value=len(pages))
    doc.__getitem__ = MagicMock(side_effect=lambda i: pages[i])
    return doc


# ─── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def mock_embed_service():
    """Mock EmbeddingService that returns a deterministic unit-norm vector."""
    svc = MagicMock()
    svc.embed_image.return_value = _fake_embedding()
    return svc


@pytest.fixture()
def extractor(mock_embed_service):
    """ImageExtractor with mocked embed service (no API calls)."""
    from app.ingestion.image_extractor import ImageExtractor
    return ImageExtractor(embed_service=mock_embed_service)


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestSizeFilter:
    def test_large_image_included(self, extractor):
        """An image of 120×120 px meets the 100×100 threshold and is returned."""
        xref = 10
        doc = _make_fitz_doc(images_per_page=[[(xref, 0, 120, 120, 8, "rgb", "", "img", "DCTDecode", 0)]])
        doc.extract_image.return_value = _make_extract_image_payload(120, 120)

        with patch("app.ingestion.image_extractor.Path.exists", return_value=True), patch("fitz.open", return_value=doc):
            from app.ingestion.image_extractor import ImageExtractor
            ex = ImageExtractor(embed_service=extractor._embed)
            records = ex.extract("fake.pdf")

        assert len(records) == 1

    def test_small_image_skipped(self, extractor):
        """An image of 50×50 px is below the threshold and must be omitted."""
        xref = 20
        doc = _make_fitz_doc(images_per_page=[[(xref, 0, 50, 50, 8, "rgb", "", "img", "FlateDecode", 0)]])
        doc.extract_image.return_value = _make_extract_image_payload(50, 50)

        with patch("app.ingestion.image_extractor.Path.exists", return_value=True), patch("fitz.open", return_value=doc):
            from app.ingestion.image_extractor import ImageExtractor
            ex = ImageExtractor(embed_service=extractor._embed)
            records = ex.extract("fake.pdf")

        assert len(records) == 0

    def test_exactly_minimum_size_included(self, extractor):
        """An image that is exactly 100×100 px must pass the filter."""
        xref = 30
        doc = _make_fitz_doc(images_per_page=[[(xref, 0, 100, 100, 8, "rgb", "", "img", "FlateDecode", 0)]])
        doc.extract_image.return_value = _make_extract_image_payload(100, 100)

        with patch("app.ingestion.image_extractor.Path.exists", return_value=True), patch("fitz.open", return_value=doc):
            from app.ingestion.image_extractor import ImageExtractor
            ex = ImageExtractor(embed_service=extractor._embed)
            records = ex.extract("fake.pdf")

        assert len(records) == 1


class TestRecordSchema:
    def _extract_one(self, extractor):
        xref = 10
        doc = _make_fitz_doc(images_per_page=[[(xref, 0, 150, 150, 8, "rgb", "", "img", "DCTDecode", 0)]])
        doc.extract_image.return_value = _make_extract_image_payload(150, 150)

        with patch("app.ingestion.image_extractor.Path.exists", return_value=True), patch("fitz.open", return_value=doc):
            from app.ingestion.image_extractor import ImageExtractor
            ex = ImageExtractor(embed_service=extractor._embed)
            return ex.extract("report.pdf")

    def test_record_has_all_required_keys(self, extractor):
        records = self._extract_one(extractor)
        assert len(records) == 1
        record = records[0]
        for key in ("page_number", "image_index", "image_base64", "embedding", "content_type", "metadata"):
            assert key in record, f"Missing key: {key}"

    def test_page_number_is_one_based(self, extractor):
        records = self._extract_one(extractor)
        assert records[0]["page_number"] == 1

    def test_image_index_starts_at_zero(self, extractor):
        records = self._extract_one(extractor)
        assert records[0]["image_index"] == 0

    def test_content_type_is_image(self, extractor):
        records = self._extract_one(extractor)
        assert records[0]["content_type"] == "image"

    def test_metadata_has_expected_keys(self, extractor):
        records = self._extract_one(extractor)
        meta = records[0]["metadata"]
        assert meta["content_type"] == "image"
        assert meta["page_number"] == 1
        assert meta["image_index"] == 0
        assert "source_filename" in meta


class TestEmbeddingOutput:
    def test_embedding_shape(self, extractor):
        """Embedding returned in the record must be a (768,) array."""
        xref = 10
        doc = _make_fitz_doc(images_per_page=[[(xref, 0, 150, 150, 8, "rgb", "", "img", "DCTDecode", 0)]])
        doc.extract_image.return_value = _make_extract_image_payload(150, 150)

        with patch("app.ingestion.image_extractor.Path.exists", return_value=True), patch("fitz.open", return_value=doc):
            from app.ingestion.image_extractor import ImageExtractor
            ex = ImageExtractor(embed_service=extractor._embed)
            records = ex.extract("fake.pdf")

        assert records[0]["embedding"].shape == (_DIM,)

    def test_embedding_is_float32(self, extractor):
        xref = 10
        doc = _make_fitz_doc(images_per_page=[[(xref, 0, 150, 150, 8, "rgb", "", "img", "DCTDecode", 0)]])
        doc.extract_image.return_value = _make_extract_image_payload(150, 150)

        with patch("app.ingestion.image_extractor.Path.exists", return_value=True), patch("fitz.open", return_value=doc):
            from app.ingestion.image_extractor import ImageExtractor
            ex = ImageExtractor(embed_service=extractor._embed)
            records = ex.extract("fake.pdf")

        assert records[0]["embedding"].dtype == np.float32


class TestBase64Storage:
    def test_base64_decodes_to_valid_png(self, extractor):
        """image_base64 must round-trip to valid PNG bytes."""
        xref = 10
        doc = _make_fitz_doc(images_per_page=[[(xref, 0, 150, 150, 8, "rgb", "", "img", "DCTDecode", 0)]])
        doc.extract_image.return_value = _make_extract_image_payload(150, 150)

        with patch("app.ingestion.image_extractor.Path.exists", return_value=True), patch("fitz.open", return_value=doc):
            from app.ingestion.image_extractor import ImageExtractor
            ex = ImageExtractor(embed_service=extractor._embed)
            records = ex.extract("fake.pdf")

        raw = base64.b64decode(records[0]["image_base64"])
        # PNG files start with the 8-byte PNG signature
        assert raw[:8] == b"\x89PNG\r\n\x1a\n", "Expected a valid PNG header"


class TestErrorHandling:
    def test_corrupt_image_skipped_gracefully(self, extractor):
        """If embed_image raises for one image, the extractor skips it and continues."""
        xref1, xref2 = 10, 20
        doc = _make_fitz_doc(images_per_page=[[
            (xref1, 0, 150, 150, 8, "rgb", "", "img1", "DCTDecode", 0),
            (xref2, 0, 150, 150, 8, "rgb", "", "img2", "DCTDecode", 0),
        ]])

        call_count = {"n": 0}

        def extract_side(xref):
            call_count["n"] += 1
            return _make_extract_image_payload(150, 150)

        doc.extract_image.side_effect = extract_side

        # First embed_image call raises; second succeeds
        extractor._embed.embed_image.side_effect = [
            RuntimeError("corrupt image"),
            _fake_embedding(),
        ]

        with patch("app.ingestion.image_extractor.Path.exists", return_value=True), patch("fitz.open", return_value=doc):
            from app.ingestion.image_extractor import ImageExtractor
            ex = ImageExtractor(embed_service=extractor._embed)
            records = ex.extract("fake.pdf")

        # Only the second image succeeds
        assert len(records) == 1
        assert records[0]["image_index"] == 1

    def test_empty_pdf_returns_empty_list(self, extractor):
        """A PDF whose pages have no images returns an empty list."""
        doc = _make_fitz_doc(images_per_page=[[]])  # one page, zero images

        with patch("app.ingestion.image_extractor.Path.exists", return_value=True), patch("fitz.open", return_value=doc):
            from app.ingestion.image_extractor import ImageExtractor
            ex = ImageExtractor(embed_service=extractor._embed)
            records = ex.extract("empty.pdf")

        assert records == []

    def test_file_not_found_raises(self, extractor):
        """A non-existent path raises FileNotFoundError."""
        from app.ingestion.image_extractor import ImageExtractor
        ex = ImageExtractor(embed_service=extractor._embed)
        with pytest.raises(FileNotFoundError):
            ex.extract("does_not_exist.pdf")
