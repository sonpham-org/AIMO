"""Load AIMO problems from CSV files for both AIMO2 and AIMO3 competitions."""

import os
from typing import Dict, List, Optional

import pandas as pd


# Paths relative to repo root
AIMO3_REFERENCE = "data/aimo3/reference.csv"
AIMO2_REFERENCE = "data/aimo2/reference.csv"


def load_problems(
    csv_path: Optional[str] = None,
    competition: str = "aimo3",
) -> pd.DataFrame:
    """Load problems from a CSV file.

    Args:
        csv_path: Explicit path to a CSV. If None, uses the default for the competition.
        competition: "aimo2" or "aimo3" — selects default CSV path.

    Returns:
        DataFrame with columns: id, problem, and optionally answer.
    """
    if csv_path is None:
        repo_root = _find_repo_root()
        if competition == "aimo3":
            csv_path = os.path.join(repo_root, AIMO3_REFERENCE)
        elif competition == "aimo2":
            csv_path = os.path.join(repo_root, AIMO2_REFERENCE)
        else:
            raise ValueError(f"Unknown competition: {competition}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Problem CSV not found at {csv_path}. "
            f"Download it from the Kaggle competition data page."
        )

    df = pd.read_csv(csv_path)

    # Normalize column names
    col_map = {}
    for col in df.columns:
        cl = col.lower().strip()
        if cl in ("question", "problem_text"):
            col_map[col] = "problem"
        elif cl == "prob_id":
            col_map[col] = "id"
    if col_map:
        df = df.rename(columns=col_map)

    required = {"id", "problem"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    return df


def get_known_answers(df: pd.DataFrame) -> Dict[str, int]:
    """Extract known answers from a DataFrame if the 'answer' column exists."""
    if "answer" not in df.columns:
        return {}
    return {str(row["id"]): int(row["answer"]) for _, row in df.iterrows()}


def _find_repo_root() -> str:
    """Walk up from this file to find the repo root (contains .git)."""
    path = os.path.dirname(os.path.abspath(__file__))
    for _ in range(10):
        if os.path.isdir(os.path.join(path, ".git")):
            return path
        path = os.path.dirname(path)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
