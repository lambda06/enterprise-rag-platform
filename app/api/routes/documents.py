"""
Document upload and ingestion API routes.
"""

import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, Body

from app.ingestion.pipeline import ingest

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)) -> dict:
    """
    Upload a PDF file for ingestion into the vector store.

    Saves the file temporarily, runs the ingestion pipeline (parse → chunk →
    embed → upsert), then deletes the temp file. Returns the ingestion summary.
    """
    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted",
        )

    # Create temp file with .pdf suffix so parser accepts it
    suffix = Path(file.filename).suffix or ".pdf"
    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as tmp:
            # Stream file content to disk
            content = await file.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save uploaded file: {e}",
        ) from e

    try:
        summary = await ingest(tmp_path, filename=file.filename)

        if summary["status"] == "failed":
            raise HTTPException(
                status_code=500,
                detail=summary.get("error", "Ingestion failed"),
            )

        return summary

    finally:
        # Always delete temp file, even on error
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


@router.post("/chat")
async def chat(question: dict = Body(...)) -> dict:
    """Simple chat endpoint that answers a question using the RAG pipeline.

    Expects JSON body: {"question": "..."}
    Returns: {"answer": str, "source_chunks": [...], "chunk_count": int}
    """
    q = question.get("question") if isinstance(question, dict) else None
    if not q or not isinstance(q, str):
        raise HTTPException(status_code=400, detail="Request body must include a 'question' string")

    from app.rag.pipeline import rag_pipeline

    try:
        result = await rag_pipeline.query(q)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return result
