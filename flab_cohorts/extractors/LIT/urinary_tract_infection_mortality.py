"""
This class extracts the catheter-associated urinary tract infection (CAUTI) mortality cohort from MIMIC-IV.
Reference: https://pubmed.ncbi.nlm.nih.gov/40995080/
"""
# ICU
# In hospital mortality

import pandas as pd
from dataclasses import dataclass

from flab_cohorts.extractors.base import BaseExtractor
from flab_cohorts.extractors.LIT.cohort_utils import save_cohort
from flab_cohorts.utils.dataset_loader import load_icu_stays, load_diagnoses
from flab_cohorts.utils.logger import get_logger

logger = get_logger("URINARY_TRACT_INFECTION_MORTALITY")


@dataclass
class UrinaryTractInfectionConfig:
    age_min: float = 18.0
    age_max: float = 120.0
    min_los_days: float = 1.0
    CAUTI_ICD9_CODES: tuple[str, ...] = ("99664",)
    CAUTI_ICD10_CODES: tuple[str, ...] = ("T83511", "T8351X")
    ALL_CODES: tuple[str, ...] = CAUTI_ICD9_CODES + CAUTI_ICD10_CODES

class UrinaryTractInfectionExtractor(BaseExtractor):
    def __init__(self, args, config: UrinaryTractInfectionConfig = UrinaryTractInfectionConfig()):
        super().__init__(args)
        self.config = config



    def prepare_stays(self) -> pd.DataFrame:
 
        stays = load_icu_stays(self.data_path)
        stays = stays.merge(self.patients, on="subject_id", how="left")
        stays["is_age_eligible"] = (stays["age"] >= self.config.age_min) & (stays["age"] <= self.config.age_max)
        stays["has_min_icu_los"] = stays["los"] >= self.config.min_los_days
        
        stays = stays.merge(self.adms[["hadm_id", "admittime", "dischtime", "deathtime", "race"]], on="hadm_id",how="left")
        stays = stays.sort_values(["subject_id", "intime"])
        stays["is_first_icustay"] = (stays.groupby("subject_id")["intime"].transform("min") == stays["intime"])
        return stays

    def add_diagnosis_labels(self, stays: pd.DataFrame) -> pd.DataFrame:
        
        diags = load_diagnoses(self.data_path)
        diags["icd_code"] = diags["icd_code"].str.replace(".", "", regex=False)
        
        ids = diags[diags["icd_code"].str.startswith(self.config.ALL_CODES)]
        stays["has_diagnosis"] = stays["hadm_id"].isin(ids["hadm_id"])
        return stays

    def add_mortality_label(self, stays: pd.DataFrame) -> pd.DataFrame:
        stays["in_hospital_mortality"] = stays["deathtime"].notna() 
        
        return stays

    def extract_cohort(self) -> None:
        logger.info("Extracting urinary tract infection in-hospital mortality cohort")

        stays = self.prepare_stays()
        stays = self.add_diagnosis_labels(stays)
        stays = self.add_mortality_label(stays)
        
        cohort = stays[stays["has_diagnosis"] & stays["is_age_eligible"]]
        cohort["is_first_icustay_with_diagnosis"] = (cohort.groupby("subject_id")["intime"].transform("min") == cohort["intime"])
        
        cohort = cohort[cohort["is_first_icustay_with_diagnosis"]]
        cohort["label"] = cohort["in_hospital_mortality"].astype(int)

        save_cohort(cohort, self.paths, "urinary_tract_infection_mortality")
