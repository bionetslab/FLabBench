import pandas as pd
from pathlib import Path
from config.constants import get_data_path
from flab_features.feature_extractor import FeatureExtractor


class Extractor:
    def __init__(self, args):
        self.args = args
        features_file = Path(args.paths["features_path"]) / args.cohort / "features.csv.gz"
        if not features_file.exists():
            args.logger.write(f"Features not found for {args.cohort}, extracting...")
            self._extract()
            args.logger.write(f"Extraction complete: {features_file}")
        else:
            args.logger.write(f"Features file used: {features_file}")

    def _extract(self):
        cohort_file = Path(self.args.paths["cohort_path"]) / f"cohort_{self.args.cohort}.csv.gz"
        if not cohort_file.exists():
            raise FileNotFoundError(
                f"Cohort file not found: {cohort_file}\n"
                f"Run cohort extraction first."
            )
        cohort_df = pd.read_csv(cohort_file, compression="gzip")
        mimic_dir = get_data_path(self.args.dataset)
        extractor = FeatureExtractor(
            mimic_dir=mimic_dir,
            features_base_path=self.args.paths["features_path"],
            top_features_path=self.args.paths["top_features_path"],
            days_before_discharge=getattr(self.args, "days_before_discharge", 14),
        )
        extractor.extract(cohort_df, self.args.cohort)


class ExtractorTrain(Extractor):
    def __init__(self, args):
        super().__init__(args)


class ExtractorPretrain(Extractor):
    def __init__(self, args):
        super().__init__(args)