"""
Feature extraction pipeline. Run after cohort extraction.

Examples:
  python -m flab_features.extract_features --extractor LIT --cohort neutropenic_fever
  python -m flab_features.extract_features --extractor DTB --cohort A08-A41
  python -m flab_features.extract_features --extractor LIT --cohort all
"""
import argparse
from pathlib import Path
import pandas as pd
from flab_cohorts.config.constants import get_data_path
from flab_cohorts.utils.io import set_all_paths
from flab_features.feature_extractor import FeatureExtractor


def _resolve_cohort_csvs(cohort_path: Path, cohort: str):
    if cohort == "all":
        return list(cohort_path.glob("cohort_*.csv")) + list(cohort_path.glob("cohort_*.csv.gz"))
    for suffix in [".csv", ".csv.gz"]:
        p = cohort_path / f"cohort_{cohort}{suffix}"
        if p.exists():
            return [p]
    raise FileNotFoundError(
        f"Cohort file not found: {cohort_path / f'cohort_{cohort}.csv'}\n"
        "Run cohort extraction first."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Extract lab features from MIMIC-IV for a given cohort."
    )
    parser.add_argument("--extractor", type=str, choices=["DTB", "LIT"], default="LIT")
    parser.add_argument("--cohort", type=str, default="neutropenic_fever",
                        help="Cohort name (e.g. neutropenic_fever, aki) or 'all'.")
    parser.add_argument("--dataset", type=str, default="MIMIC_IV")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Override MIMIC root path. Default: from MIMIC_IV_PATH env.")
    parser.add_argument("--days", type=int, default=14,
                        help="Days before discharge to extract labs. Default: 14.")
    parser.add_argument("--agg-interval", type=int, default=24,
                        help="Aggregation interval in hours for bin index. Default: 24.")
    args = parser.parse_args()

    mimic_dir = Path(args.data_path) if args.data_path else get_data_path(args.dataset)
    paths = set_all_paths(args, out=False)

    cohort_files = _resolve_cohort_csvs(paths["cohort_path"], args.cohort)

    extractor = FeatureExtractor(
        mimic_dir=mimic_dir,
        features_base_path=paths["features_path"],
        days_before_discharge=args.days,
        agg_interval=args.agg_interval,
    )

    for cohort_file in cohort_files:
        cohort_df = pd.read_csv(cohort_file)
        cohort_name = cohort_file.stem.removeprefix("cohort_").removesuffix(".csv")
        print(f"\n--- Extracting features for: {cohort_name} ---")
        extractor.extract(cohort_df, cohort_name)
        print(f"Saved to {paths['features_path'] / cohort_name}/")


if __name__ == "__main__":
    main()
