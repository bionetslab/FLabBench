import pandas as pd
from flab_cohorts.extractors.base import BaseExtractor, HOSP_COHORT_COLUMNS
from flab_cohorts.utils.logger import get_logger

logger = get_logger("MIMIC_ALL")

EXCLUDE_COHORTS = [
    "neutropenic_fever",
    "aplasia",
]
#check mimic all later
class MimicAllExtractor(BaseExtractor):
    COHORT_NAME = "mimic_all"
    COHORT_COLUMNS = HOSP_COHORT_COLUMNS

    def extract_cohort(self):
        cohort = self.adms.copy()

        supervised_subjects = set()
        for name in EXCLUDE_COHORTS:
            f = self.paths["cohort_path"] / f"cohort_{name}.csv.gz"
            if f.exists():
                df = pd.read_csv(f, compression="gzip", usecols=["subject_id"])
                supervised_subjects.update(df["subject_id"].tolist())
                logger.info("Excluded %d subjects from %s", df["subject_id"].nunique(), name)

        cohort = cohort[~cohort["subject_id"].isin(supervised_subjects)].copy()
        cohort["label"] = 0

        self.save_cohort(cohort)
        logger.info("mimic_all cohort saved: %d admissions", len(cohort))
        return cohort