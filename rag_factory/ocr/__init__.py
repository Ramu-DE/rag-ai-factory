from .extractor    import extract_document, ExtractedDocument, ExtractedPage
from .textract     import ocr_file, ocr_bytes, TextractResult
from .local_engine import (
    LocalOCREngine, get_local_engine,
    ocr_file_local, ocr_bytes_local,
)

__all__ = [
    # unified extractor (auto Textract → local fallback)
    "extract_document", "ExtractedDocument", "ExtractedPage",
    # Textract adapter (direct AWS calls)
    "ocr_file", "ocr_bytes", "TextractResult",
    # local engine (no AWS required)
    "LocalOCREngine", "get_local_engine",
    "ocr_file_local", "ocr_bytes_local",
]
