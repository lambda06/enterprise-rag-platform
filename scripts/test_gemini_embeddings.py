"""
Smoke test for EmbeddingService (Gemini Embedding 2 Preview).

Tests three things against the live Gemini API:
  1. embed_query   — dimension and sample values
  2. embed_chunks  — batch of 3 strings, all 768-dim
  3. L2 norm check — each vector norm ≈ 1.0
  4. embed_image   — image downloading, embedding, and dimension matching

Run from the project root with:
    d:\\Projects\\AI\\enterprise-rag-platform\\venv_312\\python.exe scripts/test_gemini_embeddings.py
"""

from __future__ import annotations

import io
import sys
import traceback

import numpy as np
import requests
from PIL import Image

# ── Make sure app/ is importable when run as a script ─────────────────────────
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.embeddings import EmbeddingService

EXPECTED_DIM = 768
L2_TOLERANCE = 1e-4  # norm must be within this of 1.0

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def run_tests() -> None:
    svc = EmbeddingService()
    failures: list[str] = []

    # ── Test 1: embed_query ──────────────────────────────────────────────────
    print("\n--- Test 1: embed_query ---")
    try:
        query = "What is the company's refund policy?"
        vec = svc.embed_query(query)

        dim_ok = vec.shape == (EXPECTED_DIM,)
        print(f"  Query   : '{query}'")
        print(f"  Shape   : {vec.shape}  (expected ({EXPECTED_DIM},))")
        print(f"  First 5 : {vec[:5].tolist()}")

        if dim_ok:
            print(f"  Dimension check -> {PASS}")
        else:
            print(f"  Dimension check -> {FAIL}  (got {vec.shape})")
            failures.append("embed_query: wrong dimension")

    except Exception:
        print(f"  -> {FAIL} (exception)")
        traceback.print_exc()
        failures.append("embed_query: raised exception")

    # ── Test 2: embed_chunks — all 768-dim ───────────────────────────────────
    print("\n--- Test 2: embed_chunks (3 strings) ---")
    texts = [
        "Invoice number 1042 for September 2024.",
        "The product warranty extends to 24 months from purchase date.",
        "Please contact support at help@company.com for assistance.",
    ]
    chunk_vecs: list[np.ndarray] = []
    try:
        chunk_vecs = svc.embed_chunks(texts)

        count_ok = len(chunk_vecs) == len(texts)
        print(f"  Chunks embedded : {len(chunk_vecs)} / {len(texts)}")

        all_dim_ok = True
        for i, (text, vec) in enumerate(zip(texts, chunk_vecs)):
            dim_ok = vec.shape == (EXPECTED_DIM,)
            all_dim_ok = all_dim_ok and dim_ok
            status = PASS if dim_ok else FAIL
            print(f"  [{i}] shape={vec.shape}  first_val={vec[0]:.6f}  -> {status}")

        if count_ok and all_dim_ok:
            print(f"  All dimension checks -> {PASS}")
        else:
            print(f"  Dimension checks -> {FAIL}")
            failures.append("embed_chunks: wrong count or dimension")

    except Exception:
        print(f"  -> {FAIL} (exception)")
        traceback.print_exc()
        failures.append("embed_chunks: raised exception")

    # ── Test 3: L2 norm ≈ 1.0 ────────────────────────────────────────────────
    print("\n--- Test 3: L2 norm ~= 1.0 ---")
    try:
        # Include the query vector from Test 1 and chunk vectors from Test 2
        all_vecs = {"query": vec} if "vec" in dir() else {}
        for i, cv in enumerate(chunk_vecs):
            all_vecs[f"chunk[{i}]"] = cv

        all_norm_ok = True
        for label, v in all_vecs.items():
            norm = float(np.linalg.norm(v))
            ok = abs(norm - 1.0) < L2_TOLERANCE
            all_norm_ok = all_norm_ok and ok
            status = PASS if ok else FAIL
            print(f"  {label:12s}  norm={norm:.8f}  -> {status}")

        if all_norm_ok:
            print(f"  All L2 norm checks -> {PASS}")
        else:
            print(f"  L2 norm checks -> {FAIL}")
            failures.append("norm: one or more vectors not unit-length")

    except Exception:
        print(f"  -> {FAIL} (exception)")
        traceback.print_exc()
        failures.append("norm check: raised exception")

    # ── Test 4: embed_image ──────────────────────────────────────────────────
    print("\n--- Test 4: embed_image ---")
    try:
        url = "https://storage.googleapis.com/generativeai-downloads/images/scones.jpg"
        print(f"  Downloading     : {url}")
        resp = requests.get(url)
        resp.raise_for_status()
        
        img = Image.open(io.BytesIO(resp.content))
        print(f"  Image loaded    : {img.format} {img.mode} {img.size}")
        
        img_vec = svc.embed_image(img)
        
        dim_ok = img_vec.shape == (EXPECTED_DIM,)
        norm = float(np.linalg.norm(img_vec))
        norm_ok = abs(norm - 1.0) < L2_TOLERANCE
        
        print(f"  Shape           : {img_vec.shape} (Matches text dim: {'Yes' if dim_ok else 'No'})")
        print(f"  L2 Norm         : {norm:.8f} (Normalized: {'Yes' if norm_ok else 'No'})")
        
        if dim_ok and norm_ok:
            print(f"  Image embed check -> {PASS}")
        else:
            print(f"  Image embed check -> {FAIL}")
            failures.append("embed_image: wrong dimension or unnormalized")
            
    except Exception:
        print(f"  -> {FAIL} (exception)")
        traceback.print_exc()
        failures.append("embed_image: raised exception")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    if not failures:
        print(f"  All tests {PASS}")
    else:
        print(f"  {FAIL}  --  {len(failures)} failure(s):")
        for f in failures:
            print(f"    - {f}")
    print("=" * 50)

    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    run_tests()
