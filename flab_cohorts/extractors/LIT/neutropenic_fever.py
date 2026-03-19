"""
This class extracts the neutropenic fever cohort from the MIMIC-IV dataset.
Reference:  https://www.medrxiv.org/content/10.64898/2025.12.12.25342142v1
"""

# non-ICU

import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from datetime import timedelta
from tqdm import tqdm
tqdm.pandas()


from flab_cohorts.extractors.base import BaseExtractor
from flab_cohorts.extractors.LIT.cohort_utils import extract_diag_pts, extract_chemo_cohort
from flab_cohorts.utils.logger import get_logger

logger = get_logger("NEUTROPENIC_FEVER")


@dataclass
class NeutropenicFeverConfig:
    follow_up_days: int = 30
    splitting_approach: str = "both readmissions and no admission"
    cancer_icd_code: str = "C"
    fever_icd_code: str = "R50"
    neutropenia_icd_code: str = "D70"


class NeutropenicFeverExtractor(BaseExtractor):
    def __init__(self, args, config: NeutropenicFeverConfig = NeutropenicFeverConfig()):
        super().__init__(args)
        self.config = config

    def build_cancer_chemo_cohort(self) -> pd.DataFrame:
 
        cancer_pts = extract_diag_pts(self.data_path, icd_code=self.config.cancer_icd_code)
        cancer_cohort = self.adms[self.adms["subject_id"].isin(cancer_pts["subject_id"])]
        return extract_chemo_cohort(cancer_cohort, self.data_path)

    def add_nf_flags(self, cohort: pd.DataFrame) -> pd.DataFrame:
        """Add fever, neutropenia, and combined NF flags."""
        
        fever_pts = extract_diag_pts(self.data_path, icd_code=self.config.fever_icd_code)
        cohort['fever'] = cohort['hadm_id'].isin(fever_pts['hadm_id']).astype(int)

        neutropenia_pts = extract_diag_pts(self.data_path, icd_code=self.config.neutropenia_icd_code)
        cohort['neutropenia'] = cohort['hadm_id'].isin(neutropenia_pts['hadm_id']).astype(int)

        cohort['NF'] = ((cohort['fever'] == 1) & (cohort['neutropenia'] == 1)).astype(int)
        return cohort

    def split_neutropenic_fever_cases(self, x: pd.Series, target_cohort: pd.DataFrame) -> int:
        x["dod"] = pd.to_datetime(x["dod"])
        days = self.config.follow_up_days

        if x.chemo == 1 and x.NF == 0 and x.hospital_expire_flag == 0:
            sub = target_cohort[
                (target_cohort["subject_id"] == x.subject_id)
                & (target_cohort["admittime"] > x.dischtime)
                & (target_cohort["admittime"] <= (x.dischtime + timedelta(days=days)))
            ].sort_values("admittime")

            if sub.empty and x.dod <= (x.dischtime + timedelta(days=days)):
                return 0

            if not sub.empty and (sub["chemo"] == 1).any():
                positive_chemo_index = (sub["chemo"] == 1).argmax()
                readmissions_after_next_chemo = sub[positive_chemo_index:]
                sub = sub[:positive_chemo_index]
                if (sub["NF"] == 0).all() or sub.empty:
                    if (readmissions_after_next_chemo["NF"] == 0).all():
                        return 1
                    else:
                        return 0

            if self.config.splitting_approach == "only readmissions":
                if not sub.empty and (sub["NF"] == 0).all():
                    return 1
                if not sub.empty and (sub["NF"] == 1).any():
                    return 2

            if self.config.splitting_approach == "both readmissions and no admission":
                if sub.empty or (sub["NF"] == 0).all():
                    return 1
                if not sub.empty and (sub["NF"] == 1).any():
                    return 2
                else:
                    return 0

    def extract_cohort(self):
        """Run neutropenic fever cohort extraction."""
        cohort = self.build_cancer_chemo_cohort()

        logger.info("Extracting NF cohort...")
        cohort = self.add_nf_flags(cohort)
        cohort["NF_case"] = cohort.progress_apply(
            lambda x: self.split_neutropenic_fever_cases(x, cohort), axis=1,
        )

        cohort = cohort[cohort["NF_case"].isin([1, 2])]
        cohort["NF_case"] = cohort["NF_case"].replace({1: 0, 2: 1}).astype(int)

        self.save_cohort(cohort)
        return cohort

    def save_cohort(self, cohort: pd.DataFrame) -> None:
        """Save final cohort and report summary stats."""
        cohort = cohort.rename(columns={'NF_case': 'label'})
        cohort = cohort.drop(columns=['hospital_expire_flag', 'chemo', 'fever', 'neutropenia', 'NF'])

        pct = 100 * cohort["label"].mean()
        logger.info("Number of admissions in NF cohort: %s", cohort.hadm_id.nunique())
        logger.info("Number of patients in NF cohort: %s", cohort.subject_id.nunique())
        logger.info("Number of admissions with NF: %s", cohort[cohort["label"] == 1].hadm_id.nunique())
        logger.info("Neutropenic fever positive in %d days: %.2f%%", self.config.follow_up_days, pct)

        cohort.to_csv(self.paths["cohort_path"] / "cohort_neutropenic_fever.csv", index=False)
        logger.info("Neutropenic fever cohort saved.")
