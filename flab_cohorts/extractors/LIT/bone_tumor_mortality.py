"""
This class extracts the bone tumor mortality cohort from the MIMIC dataset.
Reference: https://pmc.ncbi.nlm.nih.gov/articles/PMC10978606/
"""

import pandas as pd
from dataclasses import dataclass
from flab_cohorts.extractors.base import ICUBaseExtractor
from flab_cohorts.utils.logger import get_logger

logger = get_logger("BONE_TUMOR_MORTALITY")


@dataclass
class BoneTumorConfig:
    age_min: float = 18.0
    age_max: float = 120.0
    min_los_days: float = 1.0
    BT_ICD9_codes = ("1985",)
    BT_ICD10_codes = ("C7B03", "C795")
    ALL_CODES = BT_ICD9_codes + BT_ICD10_codes


class BoneTumorExtractor(ICUBaseExtractor):
    def __init__(self, args, config: BoneTumorConfig = BoneTumorConfig()):
        super().__init__(args)
        self.config = config

    def add_survival_label(self, stays: pd.DataFrame) -> pd.DataFrame:
        
        stays["days_to_death_from_icu"] = ((stays["dod"] - stays["intime"]).dt.total_seconds() / 86400.0)
        not_valid = stays[stays["days_to_death_from_icu"] < -1]["hadm_id"].unique()
        stays = stays[~stays["hadm_id"].isin(not_valid)]
        stays["days_to_death_from_icu"] = stays["days_to_death_from_icu"].clip(lower=0)

        died = stays["dod"].notna() & stays["intime"].notna() & (stays["dod"] >= stays["intime"])
        stays["survival_1m"] = (died & (stays["days_to_death_from_icu"] >= 31)).astype(int)
        stays["survival_3m"] = (died & (stays["days_to_death_from_icu"] >= 90)).astype(int)
        stays["survival_1y"] = (died & (stays["days_to_death_from_icu"] >= 365)).astype(int)
        stays["survival_3y"] = (died & (stays["days_to_death_from_icu"] >= 3 * 365)).astype(int)
        stays["long_term_mortality"] = (stays["days_to_death_from_icu"] >= 31).astype(int)
        return stays

    def extract_cohort(self):
   

        stays = self.initialize_icu_stays()
        stays = self.add_diagnosis_flags(stays)
        stays = self.add_survival_label(stays)

        cohort = stays[stays["has_diagnosis"]].copy()
        cohort = self.first_stay_per_patient(cohort)
        cohort = cohort[cohort["has_min_icu_los"] & cohort["is_age_eligible"]]
        cohort["label"] = cohort["long_term_mortality"]

        self.save_cohort(cohort, "bone_tumor_mortality")
