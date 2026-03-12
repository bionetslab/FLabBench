"""
Cohort extraction pipeline. Run with --extractor and --cohort.

Examples:
  python extract_cohorts.py --extractor LIT --cohort NF
  python extract_cohorts.py --extractor LIT --cohort aplasia
  python extract_cohorts.py --extractor DTB --cohort A08-A41
  python extract_cohorts.py --extractor DTB --cohort DTB_all
"""
import argparse
from pathlib import Path

from flab_cohorts.extractors.DTB_extractor import DTBExtractor
from flab_cohorts.extractors.LIT_extractor import LITExtractor


def main():
    parser = argparse.ArgumentParser(
        description="Extract cohorts from MIMIC-IV (LIT: literature-based; DTB: disease trajectory).",
    )
    parser.add_argument("--extractor",type=str,choices=["DTB", "LIT"],default="LIT",help="Extractor type: DTB (disease trajectory) or LIT (literature-based).")
    parser.add_argument("--cohort", type=str, default="NF",
    help="LIT: 'NF', 'neutropenic_fever', 'aplasia'. DTB: 'DTB_all' or 'D1-D2' (e.g. A08-A41).")
    parser.add_argument("--dataset",type=str,choices=["MIMIC_IV", "MIMIC_III"],default="MIMIC_IV",help="Dataset key for data path lookup (e.g. MIMIC_IV).")
    parser.add_argument("--data-path",type=str,default=None,help="Override MIMIC root path. Default: from config / MIMIC_IV_PATH env.")
    parser.add_argument("--prefix",type=str,default=None,help="Optional output prefix.")
    
    
    
    args = parser.parse_args()

    if args.data_path is not None:
        args.data_path = Path(args.data_path)

    if args.extractor == "DTB":
        e = DTBExtractor(args)
        e.extract_full_cohort(args.cohort)
    elif args.extractor == "LIT":
        e = LITExtractor(args)
        e.extract_full_cohort(args.cohort)


if __name__ == "__main__":
    main()