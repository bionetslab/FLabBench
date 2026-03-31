"""
This class extracts the heart failure + atrial fibrillation mortality cohort from the MIMIC dataset.
Reference: https://pmc.ncbi.nlm.nih.gov/articles/PMC11667998/
"""

import pandas as pd
from dataclasses import dataclass

from flab_cohorts.extractors.base import ICUBaseExtractor
from flab_cohorts.utils.dataset_loader import load_diagnoses
from flab_cohorts.utils.logger import get_logger

logger = get_logger("HF_AND_AF_MORTALITY")

#HF Heart Failure
#AF Atrial Fibrillation

@dataclass
class HFAndAFConfig:
    age_min: float = 18.0
    age_max: float = 120.0
    min_los_days: float = 1.0
    HF_ICD10_codes: tuple[str, ...] = (
        "I5021", "I5023", "I5031", "I5033", "I5041", "I5043",
        "I50811", "I50813", "I5022", "I5032", "I5042", "I50812",
    )
    HF_ICD9_codes: tuple[str, ...] = (
        "42821", "42823", "42831", "42833", "42841", "42843",
        "42822", "42832", "42842",
    )
    AF_ICD9_codes: tuple[str, ...] = ("42731",)
    AF_ICD10_codes: tuple[str, ...] = ("I480", "I481", "I482", "I4891")


class HFAndAFExtractor(ICUBaseExtractor):
    COHORT_NAME = "hf_af"
    def __init__(self, args, config: HFAndAFConfig = HFAndAFConfig()):
        super().__init__(args)
        self.config = config

    def add_hf_and_af_diagnosis(self, stays: pd.DataFrame) -> pd.DataFrame:
        
        hf_ids = self.diags[self.diags["icd_code"].isin(self.config.HF_ICD10_codes + self.config.HF_ICD9_codes)]
        af_ids = self.diags[self.diags["icd_code"].str.startswith(self.config.AF_ICD10_codes) | self.diags["icd_code"].isin(self.config.AF_ICD9_codes)]
        
        stays["has_HF_diagnosis"] = stays["hadm_id"].isin(hf_ids["hadm_id"])
        stays["has_AF_diagnosis"] = stays["hadm_id"].isin(af_ids["hadm_id"])
        return stays

    def extract_cohort(self):

        stays = self.initialize_icu_stays()
        stays = self.add_hf_and_af_diagnosis(stays)
        stays = self.add_inhospital_mortality(stays)

        cohort = stays[stays["has_HF_diagnosis"] & stays["has_AF_diagnosis"]].copy()
        cohort = self.first_stay_per_patient(cohort)
        cohort = cohort[cohort["has_min_icu_los"] & cohort["is_age_eligible"]]
        cohort["label"] = cohort["in_hospital_mortality"].astype(int)

        self.save_cohort(cohort)
