"""
This class extracts the pneumonia in-hospital mortality cohort from MIMIC-IV.
Reference: https://pmc.ncbi.nlm.nih.gov/articles/PMC12486837/
"""

import pandas as pd
from dataclasses import dataclass

from flab_cohorts.extractors.base import BaseExtractor
from flab_cohorts.extractors.LIT.cohort_utils import save_cohort
from flab_cohorts.utils.dataset_loader import load_icu_stays, load_diagnoses
from flab_cohorts.utils.logger import get_logger

logger = get_logger("PNEUMONIA_MORTALITY")


@dataclass
class PneumoniaConfig:
    age_min: float = 18.0
    age_max: float = 120.0
    min_los_days: float = 1.0
    pneumonia_icd9_codes: tuple[str, ...] = ("480", "481", "482", "483", "484", "485", "486")
    pneumonia_icd10_codes: tuple[str, ...] = ("J12", "J13", "J14", "J15", "J16", "J17", "J18")
    ALL_CODES = pneumonia_icd9_codes + pneumonia_icd10_codes

class PneumoniaExtractor(BaseExtractor):
    def __init__(self, args, config: PneumoniaConfig = PneumoniaConfig()):
        super().__init__(args)
        self.config = config

    def _first_hospital_admissions(self) -> pd.DataFrame:
        first_hadm_ids = set(
            self.adms.sort_values(["subject_id", "admittime"])
            .groupby("subject_id", as_index=False)
            .head(1)["hadm_id"]
            .dropna()
            .astype(int)
        )
        return first_hadm_ids

    def prepare_stays(self) -> pd.DataFrame:
        first_hadm_ids = self._first_hospital_admissions()
        
        stays = load_icu_stays(self.data_path)
        stays = stays.merge(self.patients, on="subject_id", how="left")
        stays["is_age_eligible"] = (stays["age"] >= self.config.age_min) & (stays["age"] <= self.config.age_max)
        stays["has_min_icu_los"] = stays["los"] >= self.config.min_los_days
        
        stays = stays.merge(self.adms[["hadm_id", "admittime", "dischtime", "deathtime", "race"]], on="hadm_id",how="left")
        stays = stays.sort_values(["subject_id", "intime"])
        stays["is_first_icustay"] = (stays.groupby("subject_id")["intime"].transform("min") == stays["intime"])
        stays["is_first_hadm"] = stays["hadm_id"].isin(first_hadm_ids)
        return stays

    def add_diagnosis_labels(self, stays: pd.DataFrame) -> pd.DataFrame:
        
        diags = load_diagnoses(self.data_path)
        diags["icd_code"] = diags["icd_code"].str.replace(".", "", regex=False)
        
        ids = diags[diags["icd_code"].str.startswith(self.config.ALL_CODES)]
        stays["has_diagnosis"] = stays["hadm_id"].isin(ids["hadm_id"])

        return stays

    def add_mortality_label(self, stays: pd.DataFrame) -> pd.DataFrame:
        stays['in_hospital_mortality'] = stays["deathtime"] .notna() 
        return stays

    def extract_cohort(self) -> None:
        logger.info("Extracting pneumonia in-hospital mortality cohort")

        stays = self.prepare_stays()
        stays = self.add_diagnosis_labels(stays)
        stays = self.add_mortality_label(stays)

        cohort = stays[stays["is_first_hadm"] & stays["has_diagnosis"]]
        cohort = cohort[cohort["is_first_icustay"] & cohort["is_age_eligible"] & cohort["has_min_icu_los"]]
        cohort["label"] = cohort["in_hospital_mortality"].astype(int)

        save_cohort(cohort, self.paths, "pneumonia_mortality")
