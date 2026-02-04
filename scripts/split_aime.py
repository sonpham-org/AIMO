#!/usr/bin/env python3
"""Download and split AIME dataset for hyperparameter tuning.

Downloads from HuggingFace (gneubig/aime-1983-2024) and splits by year into
train/val/test sets for selection strategy optimization.

Usage:
    # Download and split with defaults
    python scripts/split_aime.py

    # Custom split years
    python scripts/split_aime.py \
        --train-end 2018 \
        --val-end 2022 \
        --output-dir data/aime

    # Split for two machines
    python scripts/split_aime.py --split-for-machines
"""

import argparse
import os
import sys

import pandas as pd


def download_aime_dataset(cache_dir: str = None) -> pd.DataFrame:
    """Download AIME dataset from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing datasets library...")
        os.system(f"{sys.executable} -m pip install datasets")
        from datasets import load_dataset

    print("Downloading AIME dataset from HuggingFace...")
    ds = load_dataset("gneubig/aime-1983-2024", cache_dir=cache_dir)

    # Convert to DataFrame
    df = ds["train"].to_pandas()
    print(f"Downloaded {len(df)} problems")

    return df


def normalize_aime_format(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize AIME dataset to match our trace generator format.

    Input columns: Year, Problem Number, Problem, Answer
    Output columns: id, problem, answer (ground truth)
    """
    # Create unique ID from year and problem number
    df = df.copy()
    df["id"] = df.apply(
        lambda r: f"aime_{r['Year']}_{r['Problem Number']}", axis=1
    )
    # HuggingFace dataset uses "Question", some sources use "Problem"
    problem_col = "Question" if "Question" in df.columns else "Problem"
    df["problem"] = df[problem_col]
    # Handle answers like "080 or 081 (both were accepted)" - extract first number
    def parse_answer(x):
        import re
        if pd.isna(x):
            return None
        s = str(x).strip()
        # Extract first integer
        match = re.match(r'^(\d+)', s)
        if match:
            return int(match.group(1))
        return None

    df["answer"] = df["Answer"].apply(parse_answer)
    # Drop rows with unparseable answers
    df = df.dropna(subset=["answer"])
    df["answer"] = df["answer"].astype(int)

    # Keep year for splitting
    df["year"] = df["Year"].astype(int)

    return df[["id", "problem", "answer", "year"]]


def split_by_year(
    df: pd.DataFrame,
    train_end: int = 2018,
    val_end: int = 2022,
) -> tuple:
    """Split dataset by year into train/val/test."""
    train = df[df["year"] <= train_end].copy()
    val = df[(df["year"] > train_end) & (df["year"] <= val_end)].copy()
    test = df[df["year"] > val_end].copy()

    return train, val, test


def split_for_machines(
    df: pd.DataFrame,
    train_end: int = 2018,
    split_year: int = 2004,
) -> tuple:
    """Split train set for two machines by year.

    Returns (machine1_df, machine2_df) where:
    - Machine 1 (RTX 4090, faster): 1983-split_year
    - Machine 2 (AMD Strix, parallel): split_year+1 to train_end
    """
    train = df[df["year"] <= train_end].copy()

    machine1 = train[train["year"] <= split_year].copy()
    machine2 = train[train["year"] > split_year].copy()

    return machine1, machine2


def main():
    parser = argparse.ArgumentParser(
        description="Download and split AIME dataset for hyperparameter tuning"
    )
    parser.add_argument(
        "--output-dir", default="data/aime",
        help="Output directory for CSV files"
    )
    parser.add_argument(
        "--train-end", type=int, default=2018,
        help="Last year to include in train set (default: 2018)"
    )
    parser.add_argument(
        "--val-end", type=int, default=2022,
        help="Last year to include in val set (default: 2022)"
    )
    parser.add_argument(
        "--machine-split-year", type=int, default=2004,
        help="Year to split train set between two machines (default: 2004)"
    )
    parser.add_argument(
        "--split-for-machines", action="store_true",
        help="Also create machine-specific splits for parallel generation"
    )
    parser.add_argument(
        "--cache-dir", default=None,
        help="HuggingFace cache directory"
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Download and normalize
    raw_df = download_aime_dataset(cache_dir=args.cache_dir)
    df = normalize_aime_format(raw_df)

    # Save full dataset
    full_path = os.path.join(args.output_dir, "aime_1983_2024_full.csv")
    df.to_csv(full_path, index=False)
    print(f"Saved full dataset: {full_path} ({len(df)} problems)")

    # Split by year
    train, val, test = split_by_year(df, args.train_end, args.val_end)

    # Save splits
    train_path = os.path.join(args.output_dir, "aime_train.csv")
    val_path = os.path.join(args.output_dir, "aime_val.csv")
    test_path = os.path.join(args.output_dir, "aime_test.csv")

    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)
    test.to_csv(test_path, index=False)

    print(f"\nSplit by year (train <= {args.train_end}, "
          f"val {args.train_end+1}-{args.val_end}, test > {args.val_end}):")
    print(f"  Train: {train_path} ({len(train)} problems, "
          f"years {train['year'].min()}-{train['year'].max()})")
    print(f"  Val:   {val_path} ({len(val)} problems, "
          f"years {val['year'].min()}-{val['year'].max()})")
    print(f"  Test:  {test_path} ({len(test)} problems, "
          f"years {test['year'].min()}-{test['year'].max()})")

    # Machine-specific splits
    if args.split_for_machines:
        m1, m2 = split_for_machines(df, args.train_end, args.machine_split_year)

        m1_path = os.path.join(args.output_dir, "aime_train_machine1_4090.csv")
        m2_path = os.path.join(args.output_dir, "aime_train_machine2_amd.csv")

        m1.to_csv(m1_path, index=False)
        m2.to_csv(m2_path, index=False)

        print(f"\nMachine splits (for parallel generation):")
        print(f"  RTX 4090: {m1_path} ({len(m1)} problems, "
              f"years {m1['year'].min()}-{m1['year'].max()})")
        print(f"  AMD Strix: {m2_path} ({len(m2)} problems, "
              f"years {m2['year'].min()}-{m2['year'].max()})")

    # Summary statistics
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"  Total problems: {len(df)}")
    print(f"  Year range: {df['year'].min()}-{df['year'].max()}")
    print(f"  Problems per year: ~{len(df) // (df['year'].max() - df['year'].min() + 1)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
