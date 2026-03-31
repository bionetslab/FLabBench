"""
This class extracts the immunocompromised 28-day mortality cohort from the MIMIC dataset.
Reference: https://link.springer.com/article/10.1186/s40001-025-02622-3
"""

#TODO: check the ICD codes again

import pandas as pd
from dataclasses import dataclass

from flab_cohorts.extractors.base import ICUBaseExtractor
from flab_cohorts.utils.logger import get_logger

logger = get_logger("IMMUNOCOMPROMISED_MORTALITY")


@dataclass
class ImmunocompromisedConfig:
    age_min: float = 18.0
    age_max: float = 120.0
    min_los_hours: float = 6.0
    # Primary immunodeficiency
    PID_ICD10_codes: tuple[str, ...] = ("D80", "D81", "D82", "D83", "D84", "D89")
    PID_ICD9_codes: tuple[str, ...] = ("279",)
    # HIV
    HIV_ICD10_codes: tuple[str, ...] = ("B20", "B21", "B22", "B23", "B24")
    HIV_ICD9_codes: tuple[str, ...] = ("042", "043", "044")
    # Hematologic malignancy (C81–C96 / 200–208)
    HEME_ICD10_codes: tuple[str, ...] = (
        "C81", "C82", "C83", "C84", "C85", "C86", "C87", "C88",
        "C89", "C90", "C91", "C92", "C93", "C94", "C95", "C96",
    )
    HEME_ICD9_codes: tuple[str, ...] = ("200", "201", "202", "203", "204", "205", "206", "207", "208")
    # Metastatic cancer
    META_ICD10_codes: tuple[str, ...] = ("C77", "C78", "C79", "C80")
    META_ICD9_codes: tuple[str, ...] = ("196", "197", "198", "199")
    # Solid organ transplant / HSCT
    TX_ICD10_codes: tuple[str, ...] = ("Z94", "T86")
    TX_ICD9_codes: tuple[str, ...] = ("V42", "9968")
    # Immunosuppressive therapy / chemo / long-term steroids
    THERAPY_ICD10_codes: tuple[str, ...] = ("Z7952", "Z7989", "Z9221", "Z5111", "Z5112")
    THERAPY_ICD9_codes: tuple[str, ...] = ("V580", "V5811", "V5869")

    ALL_CODES: tuple[str, ...] = (
        PID_ICD10_codes + PID_ICD9_codes
        + HIV_ICD10_codes + HIV_ICD9_codes
        + HEME_ICD10_codes + HEME_ICD9_codes
        + META_ICD10_codes + META_ICD9_codes
        + TX_ICD10_codes + TX_ICD9_codes
        + THERAPY_ICD10_codes + THERAPY_ICD9_codes
    )


class ImmunocompromisedExtractor(ICUBaseExtractor):
    def __init__(self, args, config: ImmunocompromisedConfig = ImmunocompromisedConfig()):
        super().__init__(args)
        self.config = config

    def extract_cohort(self):

        stays = self.initialize_icu_stays()
        stays["icu_los_hours"] = ((stays["outtime"] - stays["intime"]).dt.total_seconds() / 3600.0)
        stays["has_min_icu_los"] = stays["icu_los_hours"] >= self.config.min_los_hours
        stays = self.add_diagnosis_flags(stays)
        stays = self.add_timed_mortality(stays, days=28, col="mortality_28d")

        cohort = stays[
            stays["has_diagnosis"]
            & stays["is_age_eligible"]
            & stays["has_min_icu_los"]
        ].copy()
        cohort = self.first_stay_per_patient(cohort)
        cohort["label"] = cohort["mortality_28d"]

        self.save_cohort(cohort, "immunocompromised_mortality")
