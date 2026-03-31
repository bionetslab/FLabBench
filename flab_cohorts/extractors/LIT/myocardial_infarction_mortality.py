"""
This class extracts the myocardial infarction mortality cohort from the MIMIC dataset.
Reference: https://www.frontiersin.org/journals/cardiovascular-medicine/articles/10.3389/fcvm.2024.1368022/full
"""

from dataclasses import dataclass

from flab_cohorts.extractors.base import ICUBaseExtractor
from flab_cohorts.utils.logger import get_logger

logger = get_logger("MYOCARDIAL_INFARCTION_MORTALITY")


@dataclass
class MyocardialInfarctionConfig:
    age_min: float = 18.0
    age_max: float = 120.0
    mortality_days: int = 30
    ICD9_CODES: tuple[str, ...] = ("410",)
    ICD10_CODES: tuple[str, ...] = ("I210", "I211", "I212", "I213", "I214")
    ALL_CODES = ICD9_CODES + ICD10_CODES


class MyocardialInfarctionExtractor(ICUBaseExtractor):
    COHORT_NAME = "mi"
    def __init__(self, args, config: MyocardialInfarctionConfig = MyocardialInfarctionConfig()):
        super().__init__(args)
        self.config = config

    def extract_cohort(self):


        stays = self.initialize_icu_stays()
        stays = self.add_diagnosis_flags(stays)
        stays["is_ccu_stay"] = stays["first_careunit"].str.contains("CCU|CORONARY", na=False)
        stays = self.add_timed_mortality(stays, days=self.config.mortality_days, col="mortality_days")

        cohort = stays[stays["has_diagnosis"] & stays["is_ccu_stay"]].copy()
        cohort = self.first_stay_per_patient(cohort)
        cohort = cohort[cohort["is_age_eligible"]]
        cohort["label"] = cohort["mortality_days"]

        self.save_cohort(cohort)
