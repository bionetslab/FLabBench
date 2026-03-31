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


from flab_cohorts.extractors.base import BaseExtractor, HOSP_COHORT_COLUMNS
from flab_cohorts.utils.dataset_loader import load_procedures, load_chemo_procedure_codes
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
    COHORT_NAME = "neutropenic_fever"
    COHORT_COLUMNS = HOSP_COHORT_COLUMNS

    def __init__(self, args, config: NeutropenicFeverConfig = NeutropenicFeverConfig()):
        super().__init__(args)
        self.config = config

    def build_cancer_chemo_cohort(self) -> pd.DataFrame:
        cohort = self.adms.copy()
        cohort = self.add_diagnosis_flags(cohort, icd_codes=self.config.cancer_icd_code, column="has_cancer", match="startswith", level="subject")
        cohort = cohort[cohort["has_cancer"]].drop(columns=["has_cancer"])

        proc_codes = load_chemo_procedure_codes(self.data_path)
        proc = load_procedures(self.data_path)
        pattern = "|".join(proc_codes)
        chemo_hadms = proc.loc[proc["icd_code"].str.contains(pattern, na=False), "hadm_id"].unique()
        cohort["chemo"] = cohort["hadm_id"].isin(chemo_hadms).astype(int)

        cohort = self.add_diagnosis_flags(cohort, icd_codes="Z5111", column="has_chemo_diag", match="startswith", level="hadm")
        cohort["chemo"] = cohort["chemo"] | cohort["has_chemo_diag"].astype(int)
        cohort = cohort.drop(columns=["has_chemo_diag"])

        chemo_subjects = cohort.loc[cohort["chemo"] == 1, "subject_id"].unique()
        return cohort[cohort["subject_id"].isin(chemo_subjects)].copy()

    def add_nf_flags(self, cohort: pd.DataFrame) -> pd.DataFrame:
        """Add fever, neutropenia, and combined NF flags."""
        cohort = self.add_diagnosis_flags(cohort, icd_codes=self.config.fever_icd_code, column="fever", match="startswith", level="hadm")
        cohort = self.add_diagnosis_flags(cohort, icd_codes=self.config.neutropenia_icd_code, column="neutropenia", match="startswith", level="hadm")
        cohort['NF'] = (cohort['fever'] & cohort['neutropenia']).astype(int)
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

        cohort = self.build_cancer_chemo_cohort()
        cohort = self.add_nf_flags(cohort)
        cohort["NF_case"] = cohort.progress_apply(lambda x: self.split_neutropenic_fever_cases(x, cohort), axis=1)

        cohort = cohort[cohort["NF_case"].isin([1, 2])]
        cohort["label"] = cohort["NF_case"].replace({1: 0, 2: 1}).astype(int)
        cohort = cohort.drop(columns=["NF_case", "hospital_expire_flag", "chemo", "fever", "neutropenia", "NF"])

        self.save_cohort(cohort)
        return cohort
