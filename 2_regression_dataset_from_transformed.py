from pathlib import Path
import re

import numpy as np
import pandas as pd


def clean_text(text):
    """Convert text into simple column-friendly format."""
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if text == "":
        text = "unknown"
    return text


def load_and_pivot(csv_path):
    """Load one parsed file and pivot it to wide format by age."""
    df = pd.read_csv(csv_path)

    # Prefer Total gender rows for a single national table by age.
    gender_text = df["gender"].astype(str).str.strip().str.lower()
    total_df = df[gender_text == "total"].copy()

    # If a file has no Total rows, use all rows as fallback.
    if total_df.empty:
        total_df = df.copy()

    wide = total_df.pivot_table(
        index="age",
        columns="level",
        values="value",
        aggfunc="sum",
    ).reset_index()

    source_name = csv_path.stem.replace("_parsed", "")

    rename_map = {"age": "age"}
    for col in wide.columns:
        if col == "age":
            continue
        rename_map[col] = f"{source_name}__{clean_text(col)}"
    wide = wide.rename(columns=rename_map)

    return wide


def build_merged_table(input_dir):
    """Merge all parsed files into one wide table."""
    csv_files = sorted(input_dir.glob("*_parsed.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No parsed files found in {input_dir}")

    merged = None

    for csv_path in csv_files:
        one_wide = load_and_pivot(csv_path)

        if merged is None:
            merged = one_wide
        else:
            merged = merged.merge(one_wide, on="age", how="outer")

    merged = merged.sort_values("age").reset_index(drop=True)
    return merged


def add_rate_columns(df):
    """Create rate columns by dividing each count by population total."""
    if "nhs_population__total" not in df.columns:
        raise ValueError("Could not find denominator column: nhs_population__total")

    out = df.copy()
    population = out["nhs_population__total"]

    count_cols = [c for c in out.columns if c not in ["age"]]

    for col in count_cols:
        if col.startswith("nhs_population__"):
            continue

        rate_col = col + "_rate"
        out[rate_col] = np.where(population > 0, out[col] / population, np.nan)

    return out


def main():
    # Step 1: locate folders
    root = Path(__file__).parent
    input_dir = root / "outputs_transformed_data"
    out_dir = root / "outputs_regression_dataset"
    out_dir.mkdir(exist_ok=True)

    # Step 2: build merged count dataset
    merged_counts = build_merged_table(input_dir)

    # Step 3: build rate dataset for regression
    merged_rates = add_rate_columns(merged_counts)

    # Step 4: keep adult ages only (common for health regression)
    adult_rates = merged_rates[merged_rates["age"] >= 16].copy()

    # Step 5: save only age greather than 16
    adult_rates.to_csv(out_dir / "regression_dataset.csv", index=False)

    # Step 6: save quick summary for the single output file
    summary = pd.DataFrame(
        [
            {"metric": "rows_rates", "value": len(adult_rates)},
            {"metric": "cols_rates", "value": len(adult_rates.columns)},
        ]
    )
    summary.to_csv(out_dir / "regression_dataset_summary.csv", index=False)

    print("Created regression dataset")
    print("Output folder:", out_dir)


if __name__ == "__main__":
    main()
