"""
This script consolidates the preprocessing logic that was previously split
across exploratory notebooks.

1. Construct interval-limited predictor summaries from raw observation files.
2. Merge those summaries with patient-level labels and timing metadata to
   create the final modeling dataframe.

Inputs expected from the original project structure
---------------------------------------------------
- A directory of raw observation CSV files.
- `general_table.csv` containing at least:
    patientid, admissiontime, sex, age, discharge_status
- A variable reference file listing the variable IDs of interest.
- A patient-category file created during the ventilation change labeling step.
  This file should contain:
    patientid, interval_time, output
- Optionally, a patient timing file containing:
    patientid, ventilation type, ventilation time, change group,
    interval time on change, interval group, Failure timing
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


OBSERVATION_COLUMNS = ["patientid", "variableid", "entertime", "value", "status"]
GENERAL_COLUMNS = ["patientid", "admissiontime", "sex", "age", "discharge_status"]
CATEGORY_COLUMNS = ["patientid", "interval_time", "output"]
TIMING_COLUMNS = [
    "patientid",
    "ventilation type",
    "ventilation time",
    "change group",
    "interval time on change",
    "interval group",
    "Failure timing",
]
SUMMARY_AGGS = ["mean", "std", "min", "max", "first", "last"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess NIV interval data.")
    parser.add_argument("--observations-dir", type=Path, help="Directory containing raw observation CSV files.")
    parser.add_argument("--general-table", type=Path, required=True, help="Path to general_table.csv.")
    parser.add_argument(
        "--variables-file",
        type=Path,
        required=True,
        help="CSV containing variable IDs of interest. Expected column: id or variableid.",
    )
    parser.add_argument(
        "--patient-categories",
        type=Path,
        required=True,
        help="CSV containing patientid, interval_time, and output.",
    )
    parser.add_argument(
        "--timing-file",
        type=Path,
        help="Optional CSV containing patient-level timing labels such as Failure timing.",
    )
    parser.add_argument(
        "--interval-end",
        type=int,
        default=2,
        help="End of the observation window in hours. Default: 2.",
    )
    parser.add_argument(
        "--interval-start",
        type=int,
        default=0,
        help="Start of the observation window in hours. Default: 0.",
    )
    parser.add_argument(
        "--precomputed-summary",
        type=Path,
        help="Optional precomputed interval summary CSV. If provided, raw observation processing is skipped.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="0_2",
        help="Prefix used when writing output files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory for output files.",
    )
    return parser.parse_args()


def load_variable_ids(path: Path) -> list[int]:
    df = pd.read_csv(path)
    for column in ("id", "variableid", "Variable_id", "variable_id"):
        if column in df.columns:
            values = pd.to_numeric(df[column], errors="coerce").dropna().astype(int).unique().tolist()
            return values
    raise ValueError(f"Could not find a variable ID column in {path}")


def list_csv_files(path: Path) -> list[Path]:
    return sorted([p for p in path.iterdir() if p.suffix.lower() == ".csv"])


def build_interval_summary(
    observations_dir: Path,
    general_table_path: Path,
    variable_ids: Iterable[int],
    interval_start: int,
    interval_end: int,
) -> pd.DataFrame:
    general = pd.read_csv(general_table_path, usecols=["patientid", "admissiontime"])
    general["admissiontime"] = pd.to_datetime(general["admissiontime"])

    frames: list[pd.DataFrame] = []
    files = list_csv_files(observations_dir)

    for index, file in enumerate(files, 1):
        print(f"Processing observation file {index}/{len(files)}: {file.name}")
        df = pd.read_csv(file, usecols=lambda c: c in OBSERVATION_COLUMNS)
        df = df[df["variableid"].isin(variable_ids)].copy()
        if df.empty:
            continue

        df = df.merge(general, on="patientid", how="inner")
        df["entertime"] = pd.to_datetime(df["entertime"])
        df["intervaltime"] = (
            (df["entertime"] - df["admissiontime"]).dt.total_seconds() // 3600
        ).astype("Int64")
        df = df[(df["intervaltime"] >= interval_start) & (df["intervaltime"] <= interval_end)]
        if df.empty:
            continue

        result = (
            df.groupby(["patientid", "variableid"])
            .agg({"value": SUMMARY_AGGS})
            .reset_index()
        )
        result.columns = ["_".join(col).rstrip("_") for col in result.columns]
        frames.append(result)

    if not frames:
        raise ValueError("No interval-limited observations were produced.")

    return pd.concat(frames, ignore_index=True)


def add_variable_names(summary_df: pd.DataFrame, variables_path: Path) -> pd.DataFrame:
    variables = pd.read_csv(variables_path)
    rename_map = {}

    if "Variable_description" in variables.columns:
        name_col = "Variable_description"
    elif "variable_name" in variables.columns:
        name_col = "variable_name"
    elif "name" in variables.columns:
        name_col = "name"
    else:
        return summary_df

    id_col = None
    for candidate in ("id", "variableid", "Variable_id", "variable_id"):
        if candidate in variables.columns:
            id_col = candidate
            break

    if id_col is None:
        return summary_df

    id_to_name = (
        variables[[id_col, name_col]]
        .dropna()
        .assign(**{id_col: lambda x: pd.to_numeric(x[id_col], errors="coerce")})
        .dropna(subset=[id_col])
    )
    id_to_name[id_col] = id_to_name[id_col].astype(int)
    rename_map = dict(zip(id_to_name[id_col], id_to_name[name_col]))

    df = summary_df.copy()
    if "variableid" in df.columns:
        df["variable_name"] = df["variableid"].map(rename_map)
    elif "variableid_" in df.columns:
        df["variable_name"] = df["variableid_"].map(rename_map)
    return df


def pivot_interval_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    value_columns = [c for c in summary_df.columns if c.startswith("value_")]
    variable_col = "variable_name" if "variable_name" in summary_df.columns else "variableid"

    long_frames = []
    for metric_col in value_columns:
        metric_name = metric_col.replace("value_", "")
        temp = summary_df[["patientid", variable_col, metric_col]].copy()
        temp["feature_name"] = temp[variable_col].astype(str) + "_" + metric_name
        temp = temp.rename(columns={metric_col: "feature_value"})
        long_frames.append(temp[["patientid", "feature_name", "feature_value"]])

    long_df = pd.concat(long_frames, ignore_index=True)
    wide_df = (
        long_df.pivot_table(index="patientid", columns="feature_name", values="feature_value", aggfunc="first")
        .reset_index()
    )
    wide_df.columns.name = None
    return wide_df


def merge_final_dataset(
    interval_features: pd.DataFrame,
    general_table_path: Path,
    patient_categories_path: Path,
    timing_file_path: Path | None,
) -> pd.DataFrame:
    general = pd.read_csv(general_table_path, usecols=lambda c: c in GENERAL_COLUMNS)
    categories = pd.read_csv(patient_categories_path, usecols=lambda c: c in CATEGORY_COLUMNS)

    merged = general.merge(interval_features, on="patientid", how="inner")
    merged = merged.merge(categories, on="patientid", how="left")

    if timing_file_path is not None:
        timing = pd.read_csv(timing_file_path, usecols=lambda c: c in TIMING_COLUMNS)
        timing = timing.drop_duplicates(subset=["patientid"])
        merged = merged.merge(timing, on="patientid", how="left")

    return merged


def remove_immediate_failures(df: pd.DataFrame) -> pd.DataFrame:
    if "Failure timing" not in df.columns:
        return df
    return df[df["Failure timing"].fillna("") != "Immediate"].copy()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    variable_ids = load_variable_ids(args.variables_file)

    if args.precomputed_summary:
        print(f"Reading precomputed interval summary: {args.precomputed_summary}")
        summary = pd.read_csv(args.precomputed_summary)
    else:
        if not args.observations_dir:
            raise ValueError("--observations-dir is required when --precomputed-summary is not provided.")
        summary = build_interval_summary(
            observations_dir=args.observations_dir,
            general_table_path=args.general_table,
            variable_ids=variable_ids,
            interval_start=args.interval_start,
            interval_end=args.interval_end,
        )

    summary_named = add_variable_names(summary, args.variables_file)
    interval_features = pivot_interval_summary(summary_named)

    final_df = merge_final_dataset(
        interval_features=interval_features,
        general_table_path=args.general_table,
        patient_categories_path=args.patient_categories,
        timing_file_path=args.timing_file,
    )
    final_df = remove_immediate_failures(final_df)

    summary_path = args.output_dir / f"{args.output_prefix}_interval_summary.csv"
    features_path = args.output_dir / f"{args.output_prefix}_interval_features.csv"
    final_path = args.output_dir / f"{args.output_prefix}_merged_respiratory_final.csv"

    summary_named.to_csv(summary_path, index=False)
    interval_features.to_csv(features_path, index=False)
    final_df.to_csv(final_path, index=False)

    print("\nSaved files:")
    print(f"- {summary_path}")
    print(f"- {features_path}")
    print(f"- {final_path}")
    print(f"- Final rows: {len(final_df)}")
    print(f"- Final patients: {final_df['patientid'].nunique() if 'patientid' in final_df.columns else 'n/a'}")


if __name__ == "__main__":
    main()
