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
from config.constants import get_data_path
from io_utils import set_all_paths
from flab_features.feature_extractor import FeatureExtractor


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
    parser.add_argument("--feature-selection", type=lambda x: x.lower() == "true", default=True,
                        help="Filter to top features before saving.")
    args = parser.parse_args()

    mimic_dir = Path(args.data_path) if args.data_path else get_data_path(args.dataset)
    paths = set_all_paths(args, out=False)

    if args.cohort == "all":
        cohort_files = list(paths["cohort_path"].glob("cohort_*.csv.gz"))
    else:
        cohort_files = [paths["cohort_path"] / f"cohort_{args.cohort}.csv.gz"]

    extractor = FeatureExtractor(
        mimic_dir=mimic_dir,
        features_base_path=paths["features_path"],
        top_features_path=paths["top_features_path"] if args.feature_selection else None,
        days_before_discharge=args.days,
    )

    for cohort_file in cohort_files:
        cohort_df = pd.read_csv(cohort_file, compression="gzip")
        cohort_name = cohort_file.name.removeprefix("cohort_").removesuffix(".csv.gz")
        print(f"\n--- Extracting features for: {cohort_name} ---")
        extractor.extract(cohort_df, cohort_name)
        print(f"Saved to {paths['features_path'] / cohort_name}/")


if __name__ == "__main__":
    main()
