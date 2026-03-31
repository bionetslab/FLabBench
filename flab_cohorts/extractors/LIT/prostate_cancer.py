"""
This class extracts the prostate cancercohort from MIMIC-IV.
Reference: https://www.sciencedirect.com/science/article/pii/S2949866X23000114
"""
#HOSP not ICU
#ADDED excluding other malignancies

import pandas as pd
from dataclasses import dataclass

from flab_cohorts.extractors.base import BaseExtractor, HOSP_COHORT_COLUMNS
from flab_cohorts.utils.logger import get_logger

logger = get_logger("PROSTATE_CANCER")


@dataclass
class ProstateCancerConfig:
    age_min: float = 18.0
    age_max: float = 120.0
    PC_ICD10_CODE: tuple[str, ...] = ("C61",)   # Prostate cancer
    BPH_ICD10_CODE: tuple[str, ...] = ("N40",)   # Benign prostatic hyperplasia


class ProstateCancerExtractor(BaseExtractor):
    COHORT_NAME = "prostate_cancer"
    COHORT_COLUMNS = HOSP_COHORT_COLUMNS

    def __init__(self, args, config: ProstateCancerConfig = ProstateCancerConfig()):
        super().__init__(args)
        self.config = config



    def add_diagnosis_labels(self, adms: pd.DataFrame) -> pd.DataFrame:
        diags = self.diags.copy()
        diags["icd_code"] = diags["icd_code"].astype(str).str.replace(".", "", regex=False)

        pc_mask = (diags["icd_version"] == 10) & diags["icd_code"].str.startswith(self.config.PC_ICD10_CODE)
        bph_mask = (diags["icd_version"] == 10) & diags["icd_code"].str.startswith(self.config.BPH_ICD10_CODE)
        other_primary_malig_mask = (
            (diags["icd_version"] == 10)
            & diags["icd_code"].str.startswith("C")
            & (~diags["icd_code"].str.startswith(self.config.PC_ICD10_CODE))
        )

        # Paper flow is patient-level for PC/BPH membership and overlap exclusion.
        pc_subjects = set(diags.loc[pc_mask, "subject_id"].dropna().unique())
        bph_subjects = set(diags.loc[bph_mask, "subject_id"].dropna().unique())
        overlap_subjects = pc_subjects & bph_subjects
        other_malig_subjects = set(diags.loc[other_primary_malig_mask, "subject_id"].dropna().unique())

        # Keep hadm-level tags for "first diagnosis-admission" selection.
        pc_hadm_ids = set(diags.loc[pc_mask, "hadm_id"].dropna().unique())
        bph_hadm_ids = set(diags.loc[bph_mask, "hadm_id"].dropna().unique())

        adms["has_pc"] = adms["subject_id"].isin(pc_subjects)
        adms["has_bph"] = adms["subject_id"].isin(bph_subjects)
        adms["has_pc_hadm"] = adms["hadm_id"].isin(pc_hadm_ids)
        adms["has_bph_hadm"] = adms["hadm_id"].isin(bph_hadm_ids)
        adms["has_pc_and_bph"] = adms["subject_id"].isin(overlap_subjects)
        adms["has_other_primary_malignancy"] = adms["subject_id"].isin(other_malig_subjects)


        return adms



    def extract_cohort(self) -> None:
        logger.info("Extracting prostate cancer cohort")

        adms = self.adms.copy()
        adms["is_age_eligible"] = (adms["age"] >= self.config.age_min) & (adms["age"] <= self.config.age_max)
        adms["is_male"] = adms["gender"].eq("M")
        adms = self.add_diagnosis_labels(adms)


        
        cohort = adms[(adms["has_pc"] | adms["has_bph"]) & (~adms["has_pc_and_bph"] & ~adms["has_other_primary_malignancy"])].copy()
        cohort = cohort.sort_values(["subject_id", "admittime"])
        cohort["is_first_pc_hadm"]= (cohort.groupby("subject_id")["admittime"].transform("min") == cohort["admittime"])
        cohort["is_first_bph_hadm"]= (cohort.groupby("subject_id")["admittime"].transform("min") == cohort["admittime"])
        logger.info("Subjects after first-admission filter: %s", cohort.subject_id.nunique())
        
        cohort = cohort[cohort["is_first_pc_hadm"] | cohort["is_first_bph_hadm"]]
        cohort = cohort[cohort["is_age_eligible"] & cohort["is_male"]]  


        cohort["label"] = cohort["has_pc"].astype(int)

        logger.info(
            "Final cohort subjects: total=%s, PC=%s, BPH=%s",
            cohort["subject_id"].nunique(),
            cohort.loc[cohort["label"] == 1, "subject_id"].nunique(),
            cohort.loc[cohort["label"] == 0, "subject_id"].nunique(),
        )

        self.save_cohort(cohort)
