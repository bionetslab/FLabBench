"""
This class extracts the pneumonia in-hospital mortality cohort from MIMIC-IV.
Reference: https://pmc.ncbi.nlm.nih.gov/articles/PMC12486837/
"""

import pandas as pd
from dataclasses import dataclass

from flab_cohorts.extractors.base import ICUBaseExtractor
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


class PneumoniaExtractor(ICUBaseExtractor):
    def __init__(self, args, config: PneumoniaConfig = PneumoniaConfig()):
        super().__init__(args)
        self.config = config

    def extract_cohort(self) -> None:


        stays = self.initialize_stays()

        first_hadm_ids = set(
            self.adms.sort_values(["subject_id", "admittime"])
            .groupby("subject_id", as_index=False)
            .head(1)["hadm_id"]
            .dropna()
            .astype(int)
        )
        stays["is_first_hadm"] = stays["hadm_id"].isin(first_hadm_ids)
        stays = self.add_diagnosis_flags(stays)
        stays = self.add_inhospital_mortality(stays)

        cohort = stays[
            stays["is_first_hadm"]
            & stays["has_diagnosis"]
            & stays["is_first_icustay"]
            & stays["is_age_eligible"]
            & stays["has_min_icu_los"]
        ]
        cohort["label"] = cohort["in_hospital_mortality"].astype(int)

        self.save_cohort(cohort, "pneumonia_mortality")
