from .extractor    import extract_document, ExtractedDocument, ExtractedPage
from .textract     import ocr_file, ocr_bytes, TextractResult
from .local_engine import (
    LocalOCREngine, get_local_engine,
    ocr_file_local, ocr_bytes_local,
)
from .ml_layout import (
    classify_layout, LayoutResult, WordBox,
    words_from_tesseract,
    SINGLE_COLUMN, MULTI_COLUMN, FORM, TABLE_HEAVY, MIXED,
)
from .ner_normalizer import (
    NERNormalizer, NERResult, Entity,
    normalize_ocr_text, get_normalizer,
)

__all__ = [
    # unified extractor (auto Textract → local fallback)
    "extract_document", "ExtractedDocument", "ExtractedPage",
    # Textract adapter (direct AWS calls)
    "ocr_file", "ocr_bytes", "TextractResult",
    # local engine (no AWS required)
    "LocalOCREngine", "get_local_engine",
    "ocr_file_local", "ocr_bytes_local",
    # ML layout classifier
    "classify_layout", "LayoutResult", "WordBox",
    "words_from_tesseract",
    "SINGLE_COLUMN", "MULTI_COLUMN", "FORM", "TABLE_HEAVY", "MIXED",
    # Post-OCR NER + field normalizer
    "NERNormalizer", "NERResult", "Entity",
    "normalize_ocr_text", "get_normalizer",
]
