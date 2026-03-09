import pandas as pd
from pathlib import Path

from flab_cohorts.cohort_extractor.base import BaseExtractor
from flab_cohorts.config.constants import MIMIC_IV_PATH
from flab_cohorts.utils.cohort_utils import (
    extract_disease_cohort,
    extract_chemo_cohort,
    current_NF_occurance,
    split_neutropenic_fever_cases,
)


class NeutropenicFeverExtractor(BaseExtractor):
    def __init__(self, args):
        super().__init__(args)

    def extract_full_cohort(
        self,
        days: int = 30,
        splitting_approach: str = "both readmissions and no admission",
    ) -> pd.DataFrame:
        path = Path(MIMIC_IV_PATH)
        dataset = getattr(self.args, "dataset", "MIMIC_IV_HOSP")
        cancer_cohort = extract_disease_cohort(path, dataset=dataset, disease_label="C")
        cancer_chemo_cohort = extract_chemo_cohort(cancer_cohort, path, dataset=dataset)
        target_cohort = current_NF_occurance(cancer_chemo_cohort, path, dataset=dataset)
        target_cohort = split_neutropenic_fever_cases(
            target_cohort, days=days, splitting_approach=splitting_approach
        )
        pos_case = target_cohort[target_cohort["NF_in_30_days"] == 2].assign(label=1)
        neg_case = target_cohort[target_cohort["NF_in_30_days"] == 1].assign(label=0)
        cohort = pd.concat([pos_case, neg_case], axis=0)
        drop_cols = ["hospital_expire_flag", "chemo", "fever", "neutropenia", "NF", "NF_in_30_days"]
        cohort = cohort.drop(columns=[c for c in drop_cols if c in cohort.columns])
        out_path = self.paths["cohort_path"] / f"mimic_cohort_NF_{days}_days.csv.gz"
        cohort.to_csv(out_path, index=False, compression="gzip")
        return cohort
