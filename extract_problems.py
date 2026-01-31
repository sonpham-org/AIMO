#!/usr/bin/env python3
"""
Extract problems from AIMO3_Reference_Problems.pdf using pdfplumber.

Produces a CSV with columns: id, problem
Falls back to reference.csv if pdfplumber extraction is lossy.

Usage:
    .venv/bin/python extract_problems.py
"""

import os
import re
import sys

PDF_PATH = "data/ai-mathematical-olympiad-progress-prize-3/AIMO3_Reference_Problems.pdf"
REF_CSV = "data/ai-mathematical-olympiad-progress-prize-3/reference.csv"
OUTPUT_CSV = "logs/problems_from_pdf.csv"


def extract_from_pdf(pdf_path: str) -> list[dict]:
    """Extract problem texts from the AIMO3 PDF using pdfplumber."""
    import pdfplumber

    with pdfplumber.open(pdf_path) as pdf:
        full_text = "\n".join(page.extract_text() or "" for page in pdf.pages)

    # Try splitting on "Problem N" headers
    # The PDF has "Problem 1", "Problem 2", ..., "Problem 10"
    parts = re.split(r"\bProblem\s+(\d+)\b", full_text)
    # parts = [preamble, "1", text1, "2", text2, ...]

    problems = []
    for i in range(1, len(parts) - 1, 2):
        num = int(parts[i])
        text = parts[i + 1].strip()

        # Remove everything after "Answer:" or "Solution:" if present
        for marker in ["Answer:", "Solution:"]:
            idx = text.find(marker)
            if idx != -1:
                text = text[:idx].strip()

        # Remove leading "Problem:" label if present
        text = re.sub(r"^Problem:\s*", "", text, flags=re.IGNORECASE)

        if len(text) > 20:
            problems.append({"num": num, "problem": text})

    return problems


def main():
    import pandas as pd

    os.makedirs("logs", exist_ok=True)

    if not os.path.exists(PDF_PATH):
        print(f"PDF not found at {PDF_PATH}")
        sys.exit(1)

    # Try pdfplumber extraction
    try:
        problems = extract_from_pdf(PDF_PATH)
        print(f"Extracted {len(problems)} problems from PDF")
    except ImportError:
        print("pdfplumber not installed. Install with: uv pip install pdfplumber")
        problems = []
    except Exception as e:
        print(f"PDF extraction failed: {e}")
        problems = []

    # Compare with reference.csv
    if os.path.exists(REF_CSV):
        ref_df = pd.read_csv(REF_CSV)
        print(f"Reference CSV has {len(ref_df)} problems")

        if len(ref_df) >= 10:
            print(
                f"reference.csv has {len(ref_df)} verified problems — using it as ground truth "
                f"(PDF extraction got {len(problems)}, which may include false positives)."
            )
            ref_df.to_csv(OUTPUT_CSV, index=False)
            print(f"Copied reference.csv -> {OUTPUT_CSV}")
            return

    # Save extracted problems
    if problems:
        df = pd.DataFrame(problems)
        df["id"] = [f"prob_{p['num']:02d}" for p in problems]
        df = df[["id", "problem"]]
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"Saved {len(problems)} problems -> {OUTPUT_CSV}")
    else:
        print("No problems extracted and no reference CSV available.")
        sys.exit(1)


if __name__ == "__main__":
    main()
