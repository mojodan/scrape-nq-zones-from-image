#!/usr/bin/env python3
"""Extract white text zone labels from a stock chart PNG image."""

import re
import sys

import cv2
import numpy as np
import pytesseract
from PIL import Image


def extract_zones(image_path: str) -> list[str]:
    """Extract white text lines from a stock chart image.

    Preprocesses the image to isolate white/near-white text, then uses
    Tesseract OCR to read the text.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    h, w = img.shape[:2]

    # Isolate white text: high brightness, low saturation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    val = hsv[:, :, 2]
    sat = hsv[:, :, 1]

    # Use high brightness threshold for clean text extraction
    white_mask = ((val > 220) & (sat < 40)).astype(np.uint8) * 255

    # Remove horizontal grid lines
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 1))
    horiz_lines = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, horiz_kernel)
    white_mask = cv2.subtract(white_mask, horiz_lines)

    # Remove vertical lines
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    vert_lines = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, vert_kernel)
    white_mask = cv2.subtract(white_mask, vert_lines)

    # Crop to the text region (right portion, skip toolbar)
    x_start = int(w * 0.50)
    x_end = int(w * 0.96)
    y_start = int(h * 0.01)
    y_end = int(h * 0.96)
    region = white_mask[y_start:y_end, x_start:x_end]
    rh, rw = region.shape

    # Scale up 4x
    scale = 4
    scaled = cv2.resize(region, (rw * scale, rh * scale), interpolation=cv2.INTER_CUBIC)
    _, scaled = cv2.threshold(scaled, 127, 255, cv2.THRESH_BINARY)

    # Add padding
    pad = 50
    padded = np.zeros(
        (scaled.shape[0] + 2 * pad, scaled.shape[1] + 2 * pad), dtype=np.uint8
    )
    padded[pad : pad + scaled.shape[0], pad : pad + scaled.shape[1]] = scaled

    # Invert for Tesseract
    inverted = cv2.bitwise_not(padded)

    # PSM 6 = uniform block of text
    config = r"--oem 3 --psm 6"
    text = pytesseract.image_to_string(Image.fromarray(inverted), config=config)

    # Process all non-empty lines through cleanup, then filter
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Apply cleanup first (before keyword matching)
        cleaned = _clean_line(line)
        if not cleaned:
            continue
        lower = cleaned.lower()
        if any(kw in lower for kw in [
            "zone", "range", "support", "resistance", "hvn", "gap",
            "initial", "aggressive", "pre-market",
        ]):
            lines.append(cleaned)

    return lines


def _clean_line(line: str) -> str:
    """Fix common OCR misreadings in zone label text."""
    # Remove leading noise characters (but preserve digits like "24H")
    line = re.sub(r'^[|!"\'\s,.;:]+', "", line)

    # Fix bracket variants to parentheses
    line = line.replace("[", "(").replace("{", "(")
    line = line.replace("]", ")").replace("}", ")")

    # Fix common OCR character confusions in numbers
    line = re.sub(r"(\d)UU", r"\g<1>00", line)
    line = re.sub(r"(\d)0C", r"\g<1>00", line)

    # Remove double opening parens
    line = line.replace("((", "(")

    # Fix garbled "Range" variants (do this early so keyword matching works)
    line = re.sub(r"\bRanne\b", "Range", line)
    line = re.sub(r"\bRanae\b", "Range", line)
    line = re.sub(r"\bKange\b", "Range", line)
    line = re.sub(r"\bRanaea\b", "Range", line)
    line = re.sub(r"\bRanges?\s+E", "Range E", line)  # extra s

    # Fix garbled "Exhaustion" variants
    for pattern in [
        r"Evhauetinn", r"Exnaustion", r"Frhaustian", r"CAnausuon",
        r"CAnaustion", r"CANausiCn", r"CANaUuSIVN", r"Fehaustion",
        r"Frhaustinn", r"Exnausuon", r"Fxhaustion",
    ]:
        line = re.sub(rf"\b{pattern}\b", "Exhaustion", line, flags=re.IGNORECASE)

    # Fix garbled "High/Low" at end of range lines
    line = re.sub(r"\bHinh\b", "High", line)
    line = re.sub(r"\bHiah\b", "High", line)
    line = re.sub(r"\bnigh\b", "High", line)
    line = re.sub(r"\bMign\b", "High", line)

    # Fix common OCR word misreads for zone types
    line = line.replace("Contirming", "Confirming")
    line = line.replace("contirming", "Confirming")
    line = line.replace("Contimming", "Confirming")
    line = line.replace("Cnanging", "Changing")
    line = line.replace("Chanaina", "Changing")
    line = line.replace("Channing", "Changing")

    # Fix "Zone)" ending variants (only at end of parenthetical expression)
    line = re.sub(
        r"\(([^)]*(?:Confirming|Changing|Weakness|Strength))\s+[Zz]on[aegkno]\)?\s*$",
        r"(\1 Zone)",
        line,
    )
    line = re.sub(
        r"\(([^)]*(?:Confirming|Changing|Weakness|Strength))\s+[Zz]cne\)?\s*$",
        r"(\1 Zone)",
        line,
    )

    # Ensure closing paren on lines ending with Zone/Weakness/Strength keywords
    line = re.sub(r"\(([^)]*(?:Zone|Weakness|Strength))\s*$", r"(\1)", line)

    # Remove double closing parens
    line = re.sub(r"\)\)+", ")", line)

    # Strip trailing noise after closing paren
    m = re.match(r"(.*\))", line)
    if m and "(" in m.group(1):
        line = m.group(1)

    # Remove extra spaces in numbers (e.g., "25092 .25" -> "25092.25")
    line = re.sub(r"(\d)\s+\.", r"\1.", line)
    line = re.sub(r"\.\s+(\d)", r".\1", line)
    # Fix spaces in the middle of numbers
    line = re.sub(r"(\d)\s+(\d)", r"\1\2", line)

    # Fix O'N -> O/N
    line = re.sub(r"O[''`/]?N\b", "O/N", line)

    # Fix lowercase "zone" after keywords to "Zone"
    line = re.sub(r"(Confirming|Changing)\s+zone\)", r"\1 Zone)", line)

    return line.strip()


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_zones.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    lines = extract_zones(image_path)
    for line in lines:
        print(line)


if __name__ == "__main__":
    main()
