"""
ocr_reader.py - OCR and File Parsing Module
Extracts blood parameter values from uploaded PDFs, images, and CSV files.
Uses Pytesseract for OCR and regex for parameter detection.
"""

import re
import io
import pandas as pd
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# ── Parameter regex patterns ─────────────────────────────────────────────────
# Each pattern tries to match the parameter name followed by its numeric value.
PARAMETER_PATTERNS = {
    "hemoglobin":    r"hemoglobin[\s\S]{0,30}?(\d{1,2}\.?\d*)",
    "glucose":       r"(?:glucose|blood sugar|fasting glucose)[\s\S]{0,30}?(\d{2,3}\.?\d*)",
    "cholesterol":   r"(?:total cholesterol|cholesterol)[\s\S]{0,30}?(\d{2,3}\.?\d*)",
    "vitamin_d":     r"(?:vitamin d|vit\.?\s*d|25-oh)[\s\S]{0,30}?(\d{1,3}\.?\d*)",
    "iron":          r"(?:serum iron|iron)[\s\S]{0,30}?(\d{2,3}\.?\d*)",
    "calcium":       r"calcium[\s\S]{0,30}?(\d{1,2}\.?\d*)",
    "vitamin_b12":   r"(?:vitamin b12|b12|cobalamin)[\s\S]{0,30}?(\d{2,4}\.?\d*)",
    "triglycerides": r"triglycerides?[\s\S]{0,30}?(\d{2,3}\.?\d*)",
}


def extract_from_text(text: str) -> dict:
    """
    Applies regex patterns on raw OCR/PDF text to extract blood parameter values.
    Returns a dict of {parameter: float_value}.
    """
    text = text.lower()
    results = {}
    for param, pattern in PARAMETER_PATTERNS.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                results[param] = float(match.group(1))
            except ValueError:
                pass
    return results


def read_image_file(file_bytes: bytes):
    try:
        from PIL import Image
        import cv2
        import numpy as np
        import pytesseract

        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

        # Convert bytes → numpy array
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {}, "Unable to read image."

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        text = pytesseract.image_to_string(gray, config="--psm 6")

        return extract_from_text(text), text

    except Exception as e:
        return {}, f"OCR ERROR: {type(e).__name__}: {e}"

def read_pdf_file(file_bytes: bytes) -> dict:
    """
    Extracts text from a PDF using PyPDF2, then runs regex extraction.
    Falls back to pdf2image + OCR if text extraction fails.
    """
    full_text = ""

    # ── Method 1: Direct text extraction ─────────────────────────────────────
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        for page in reader.pages:
            page_text = page.extract_text() or ""
            full_text += page_text + "\n"
    except Exception as e:
        full_text = ""

    if full_text.strip():
        return extract_from_text(full_text), full_text

    # ── Method 2: Rasterize PDF pages → OCR ─────────────────────────────────
    try:
        from pdf2image import convert_from_bytes
        import pytesseract

        images = convert_from_bytes(file_bytes, dpi=200)
        for img in images:
            full_text += pytesseract.image_to_string(img) + "\n"
        return extract_from_text(full_text), full_text
    except Exception as e:
        return {}, f"[PDF OCR Error] {str(e)}"


def read_csv_file(file_bytes: bytes) -> dict:
    """
    Reads a CSV blood report. Expected columns (case-insensitive):
      Parameter, Value   OR   hemoglobin, glucose, ... as column headers.
    Returns extracted values dict.
    """
    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        results = {}

        # ── Format A: columns = parameter, value ──────────────────────────────
        if "parameter" in df.columns and "value" in df.columns:
            for _, row in df.iterrows():
                param_raw = str(row["parameter"]).lower().strip()
                for key in PARAMETER_PATTERNS:
                    if key.replace("_", " ") in param_raw or key in param_raw:
                        try:
                            results[key] = float(str(row["value"]).strip())
                        except ValueError:
                            pass

        # ── Format B: parameters as column names ──────────────────────────────
        else:
            for key in PARAMETER_PATTERNS:
                if key in df.columns:
                    val = df[key].iloc[0]
                    try:
                        results[key] = float(val)
                    except (ValueError, TypeError):
                        pass

        return results, df.to_string()
    except Exception as e:
        return {}, f"[CSV Error] {str(e)}"


def parse_uploaded_file(file_bytes: bytes, filename: str):
    """
    Master dispatcher — routes uploaded file to the correct reader.
    Returns: (extracted_values: dict, raw_text: str)
    """
    ext = filename.lower().split(".")[-1]

    if ext in ("jpg", "jpeg", "png", "bmp", "tiff", "tif"):
        return read_image_file(file_bytes)
    elif ext == "pdf":
        return read_pdf_file(file_bytes)
    elif ext == "csv":
        return read_csv_file(file_bytes)
    else:
        return {}, f"[Error] Unsupported file type: .{ext}"


# ── Demo CSV template generator ──────────────────────────────────────────────

def generate_sample_csv() -> str:
    """Returns a CSV string that users can use as a template."""
    data = {
        "Parameter": [
            "Hemoglobin", "Glucose", "Cholesterol", "Vitamin D",
            "Iron", "Calcium", "Vitamin B12", "Triglycerides"
        ],
        "Value": [13.5, 95, 185, 22.0, 80, 9.2, 350, 140],
        "Unit": [
            "g/dL", "mg/dL", "mg/dL", "ng/mL",
            "µg/dL", "mg/dL", "pg/mL", "mg/dL"
        ]
    }
    return pd.DataFrame(data).to_csv(index=False)
