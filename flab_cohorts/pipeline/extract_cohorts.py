"""
Cohort extraction pipeline. Run with --extractor and --cohort.

Examples:
  python extract_cohorts.py --extractor LIT --cohort NF
  python extract_cohorts.py --extractor LIT --cohort aplasia
  python extract_cohorts.py --extractor DTB --cohort A08-A41
  python extract_cohorts.py --extractor DTB --cohort DTB_all
  python extract_cohorts.py --extractor DTB --cohort A08-A41,B12-C34
"""
import argparse
from pathlib import Path

from flab_cohorts.extractors.DTB_extractor import DTBExtractor
from flab_cohorts.extractors.LIT_extractor import LITExtractor
from flab_cohorts.utils.logger import setup_logging


def main():
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Extract cohorts from MIMIC-IV (LIT: literature-based; DTB: disease trajectory).",
    )
    parser.add_argument("--extractor",type=str,choices=["DTB", "LIT"],default="LIT",help="Extractor type: DTB (disease trajectory) or LIT (literature-based).")
    parser.add_argument("--cohort", type=str, default="NF",
    help="LIT: 'NF', 'neutropenic_fever', 'aplasia'. DTB: 'DTB_all', one 'D1-D2', or comma-separated list (e.g. A08-A41,B12-C34).")
    parser.add_argument("--dataset",type=str,choices=["MIMIC_IV", "MIMIC_III"],default="MIMIC_IV",help="Dataset key for data path lookup (e.g. MIMIC_IV).")
    parser.add_argument("--data-path",type=str,default=None,help="Override MIMIC root path. Default: from config / MIMIC_IV_PATH env.")
    parser.add_argument("--prefix",type=str,default=None,help="Optional output prefix.")
    
    
    
    args = parser.parse_args()

    if args.data_path is not None:
        args.data_path = Path(args.data_path)

    if args.extractor == "DTB":
        e = DTBExtractor(args)
        cohort_arg = args.cohort
        if "," in cohort_arg:
            cohorts = [c.strip() for c in cohort_arg.split(",") if c.strip()]
            e.extract_full_cohort(cohorts)
        else:
            e.extract_full_cohort(cohort_arg)
    elif args.extractor == "LIT":
        e = LITExtractor(args)
        e.extract_full_cohort(args.cohort)


if __name__ == "__main__":
    main()