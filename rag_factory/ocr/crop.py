# -*- coding: utf-8 -*-
"""
Auto-Crop — Isolate multiple documents on a single scanned page
===============================================================
Mirrors Mindee's "Crop" feature: detects individual document regions
within a single page image (batch scan / clustered photo), extracts each
as a separate cropped image ready for downstream OCR/extraction.

Algorithm
---------
1. Pre-process: grayscale → blur → adaptive threshold → morphological close
2. Contour detection: findContours on the binarised image
3. Filter: keep contours with area >= min_area_ratio (default 3% of page)
   and aspect ratio in a plausible document range (0.25 – 4.0)
4. Perspective correction: if corner-point detection succeeds, apply
   4-point warpPerspective to deskew the crop
5. Fallback: if no distinct regions found, return the full image as one crop

Output
------
List[CropResult] sorted by area descending (largest region first).
Each CropResult carries:
  region_idx     : int        (0-based)
  bbox           : dict       (x, y, w, h — normalised 0-1)
  confidence     : float      (area / convexHull area ratio ≈ "rectangularity")
  image_bytes    : bytes      (PNG of the cropped region)
  area_ratio     : float      (fraction of page area this region occupies)
  pixel_bbox     : dict       (x, y, w, h in pixels — for debugging)

Usage
-----
  from rag_factory.ocr.crop import AutoCropper
  cropper = AutoCropper()
  crops   = cropper.crop_page(pil_image)          # PIL Image input
  crops   = cropper.crop_file("scan.pdf", page=0) # PDF path input
  for crop in crops:
      print(crop.region_idx, crop.confidence, crop.area_ratio)
      open(f"crop_{crop.region_idx}.png", "wb").write(crop.image_bytes)
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ─── result type ─────────────────────────────────────────────────────────────

@dataclass
class CropResult:
    region_idx:  int
    bbox:        dict          # normalised {x, y, w, h}
    confidence:  float         # rectangularity score 0-1
    image_bytes: bytes         # PNG of cropped region
    area_ratio:  float         # fraction of total page area
    pixel_bbox:  dict = field(default_factory=dict)   # pixel {x, y, w, h}
    angle_deg:   float = 0.0   # detected skew angle (degrees)

    @property
    def aspect_ratio(self) -> float:
        w = self.bbox.get("w", 1)
        h = self.bbox.get("h", 1)
        return w / max(h, 1e-6)

    def to_dict(self) -> dict:
        return {
            "region_idx":  self.region_idx,
            "bbox":        self.bbox,
            "confidence":  round(self.confidence, 3),
            "area_ratio":  round(self.area_ratio, 4),
            "aspect_ratio": round(self.aspect_ratio, 3),
            "angle_deg":   round(self.angle_deg, 2),
            "image_bytes_len": len(self.image_bytes),
        }


# ─── main class ──────────────────────────────────────────────────────────────

class AutoCropper:
    """
    Detects and isolates individual document regions on a single page image.

    Parameters
    ----------
    min_area_ratio   : float  — minimum document area as fraction of page (default 0.03)
    max_regions      : int    — cap on number of documents per page (default 12)
    perspective_fix  : bool   — attempt 4-point warpPerspective on detected docs
    padding          : int    — pixel padding added around each crop (default 8)
    """

    def __init__(
        self,
        min_area_ratio:  float = 0.03,
        max_regions:     int   = 12,
        perspective_fix: bool  = True,
        padding:         int   = 8,
    ):
        self.min_area_ratio  = min_area_ratio
        self.max_regions     = max_regions
        self.perspective_fix = perspective_fix
        self.padding         = padding

    # ── public interface ──────────────────────────────────────────────────────

    def crop_page(self, pil_image) -> List[CropResult]:
        """
        Detect and crop document regions from a PIL Image.

        Returns list of CropResult sorted by area descending.
        Always returns at least one result (full page fallback).
        """
        try:
            import cv2
            import numpy as np
        except ImportError:
            return [self._full_page_fallback(pil_image)]

        img_np = _pil_to_bgr(pil_image)
        H, W   = img_np.shape[:2]
        page_area = H * W

        # ── pre-process ───────────────────────────────────────────────────────
        gray   = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        blur   = cv2.GaussianBlur(gray, (5, 5), 0)
        # Adaptive threshold — handles uneven lighting in scans
        thresh = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=15, C=4,
        )
        # Close gaps between edge fragments
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        # ── contour detection ─────────────────────────────────────────────────
        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        candidates = []
        min_area   = page_area * self.min_area_ratio

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            # Filter wild aspect ratios
            aspect = w / max(h, 1)
            if not (0.20 <= aspect <= 5.0):
                continue

            # Rectangularity = contour area / bounding-rect area (1.0 = perfect rect)
            rect_area = w * h
            rectangularity = area / max(rect_area, 1)
            if rectangularity < 0.30:
                continue

            # Confidence = rectangularity weighted by relative area
            confidence = min(rectangularity * (1 + area / page_area * 0.5), 1.0)

            candidates.append((area, x, y, w, h, confidence, cnt))

        # Sort by area descending, cap to max_regions
        candidates.sort(key=lambda c: c[0], reverse=True)
        candidates = candidates[: self.max_regions]

        if not candidates:
            return [self._full_page_fallback(pil_image)]

        results = []
        for idx, (area, x, y, w, h, conf, cnt) in enumerate(candidates):
            # Add padding (clamped to image bounds)
            px1 = max(x - self.padding, 0)
            py1 = max(y - self.padding, 0)
            px2 = min(x + w + self.padding, W)
            py2 = min(y + h + self.padding, H)

            # Perspective correction attempt
            angle_deg = 0.0
            if self.perspective_fix:
                crop_bgr, angle_deg = _perspective_crop(img_np, cnt, (px1, py1, px2, py2))
            else:
                crop_bgr = img_np[py1:py2, px1:px2]

            image_bytes = _bgr_to_png_bytes(crop_bgr)

            results.append(CropResult(
                region_idx  = idx,
                bbox        = {"x": px1/W, "y": py1/H, "w": (px2-px1)/W, "h": (py2-py1)/H},
                confidence  = round(conf, 3),
                image_bytes = image_bytes,
                area_ratio  = round(area / page_area, 4),
                pixel_bbox  = {"x": px1, "y": py1, "w": px2-px1, "h": py2-py1},
                angle_deg   = angle_deg,
            ))

        return results

    def crop_file(self, file_path: str, page: int = 0,
                  dpi: int = 150) -> List[CropResult]:
        """
        Detect and crop regions from a page of a PDF or image file.

        Parameters
        ----------
        file_path : PDF or image path
        page      : 0-based page number (ignored for image files)
        dpi       : render resolution for PDF pages
        """
        from PIL import Image

        ext = file_path.lower().rsplit(".", 1)[-1]
        if ext == "pdf":
            try:
                import fitz
                doc  = fitz.open(file_path)
                pno  = min(page, len(doc) - 1)
                mat  = fitz.Matrix(dpi / 72, dpi / 72)
                pix  = doc[pno].get_pixmap(matrix=mat)
                pil  = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
                doc.close()
            except ImportError:
                raise RuntimeError("PyMuPDF (fitz) required for PDF crop — pip install pymupdf")
        else:
            pil = Image.open(file_path).convert("RGB")

        return self.crop_page(pil)

    def crop_pdf_all_pages(self, pdf_path: str,
                           dpi: int = 150) -> List[List[CropResult]]:
        """
        Crop all pages of a PDF, returning one list per page.
        Useful for batch scans where each page may contain multiple documents.
        """
        try:
            import fitz
        except ImportError:
            raise RuntimeError("PyMuPDF required — pip install pymupdf")

        from PIL import Image

        all_results = []
        doc = fitz.open(pdf_path)
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        for pno in range(len(doc)):
            pix = doc[pno].get_pixmap(matrix=mat)
            pil = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            all_results.append(self.crop_page(pil))
        doc.close()
        return all_results

    # ── helpers ───────────────────────────────────────────────────────────────

    def _full_page_fallback(self, pil_image) -> CropResult:
        """Return the whole page as a single crop (confidence=0.5)."""
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        W, H = pil_image.size
        return CropResult(
            region_idx  = 0,
            bbox        = {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0},
            confidence  = 0.50,
            image_bytes = buf.getvalue(),
            area_ratio  = 1.0,
            pixel_bbox  = {"x": 0, "y": 0, "w": W, "h": H},
        )


# ─── internal helpers ────────────────────────────────────────────────────────

def _pil_to_bgr(pil_image):
    """Convert PIL RGB image to numpy BGR array for OpenCV."""
    import numpy as np
    rgb = np.array(pil_image.convert("RGB"))
    return rgb[:, :, ::-1].copy()   # RGB → BGR


def _bgr_to_png_bytes(bgr_array) -> bytes:
    """Encode an OpenCV BGR numpy array to PNG bytes."""
    import cv2
    ok, buf = cv2.imencode(".png", bgr_array)
    if ok:
        return bytes(buf)
    # Fallback via PIL
    from PIL import Image
    import numpy as np
    rgb = bgr_array[:, :, ::-1]
    pil = Image.fromarray(rgb.astype(np.uint8))
    out = io.BytesIO()
    pil.save(out, format="PNG")
    return out.getvalue()


def _perspective_crop(img_bgr, contour, bbox_xywh: Tuple):
    """
    Attempt 4-point perspective warp on the detected region.
    Falls back to simple bounding-rect crop if warp fails.
    Returns (cropped_bgr, angle_deg).
    """
    import cv2
    import numpy as np

    px1, py1, px2, py2 = bbox_xywh

    # Approximate contour to polygon
    peri   = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

    if len(approx) == 4:
        pts = approx.reshape(4, 2).astype(np.float32)

        # Order: top-left, top-right, bottom-right, bottom-left
        rect = _order_points(pts)
        tl, tr, br, bl = rect

        # Compute destination width and height
        wA = np.linalg.norm(br - bl)
        wB = np.linalg.norm(tr - tl)
        hA = np.linalg.norm(tr - br)
        hB = np.linalg.norm(tl - bl)
        maxW = max(int(wA), int(wB))
        maxH = max(int(hA), int(hB))

        if maxW > 10 and maxH > 10:
            dst = np.array([
                [0,        0       ],
                [maxW - 1, 0       ],
                [maxW - 1, maxH - 1],
                [0,        maxH - 1],
            ], dtype=np.float32)
            M     = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(img_bgr, M, (maxW, maxH))
            # Estimate skew angle from top edge
            angle = float(np.degrees(np.arctan2(tr[1] - tl[1], tr[0] - tl[0])))
            return warped, angle

    # Fallback: simple crop
    cropped = img_bgr[py1:py2, px1:px2]
    return cropped, 0.0


def _order_points(pts):
    """Order 4 corner points: top-left, top-right, bottom-right, bottom-left."""
    import numpy as np
    rect   = np.zeros((4, 2), dtype=np.float32)
    s      = pts.sum(axis=1)
    diff   = np.diff(pts, axis=1)
    rect[0] = pts[s.argmin()]      # top-left     (smallest sum)
    rect[2] = pts[s.argmax()]      # bottom-right (largest sum)
    rect[1] = pts[diff.argmin()]   # top-right    (smallest diff)
    rect[3] = pts[diff.argmax()]   # bottom-left  (largest diff)
    return rect


# ─── convenience wrapper ─────────────────────────────────────────────────────

_CROPPER: Optional[AutoCropper] = None


def get_cropper(**kwargs) -> AutoCropper:
    global _CROPPER
    if _CROPPER is None:
        _CROPPER = AutoCropper(**kwargs)
    return _CROPPER


def crop_page(pil_image, **kwargs) -> List[CropResult]:
    """Top-level convenience: crop documents from a PIL Image."""
    return AutoCropper(**kwargs).crop_page(pil_image)


def crop_file(file_path: str, page: int = 0, **kwargs) -> List[CropResult]:
    """Top-level convenience: crop from a file (PDF or image)."""
    return AutoCropper(**kwargs).crop_file(file_path, page=page)
