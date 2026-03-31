"""
This class extracts the alcoholic cirrhosis mortality cohort from the MIMIC dataset.
Reference: ??
"""

from dataclasses import dataclass
from flab_cohorts.extractors.base import ICUBaseExtractor
from flab_cohorts.utils.logger import get_logger

logger = get_logger("ALCOHOLIC_CIRRHOSIS_MORTALITY")


@dataclass
class AlcoholicCirrhosisConfig:
    age_min: float = 18.0
    age_max: float = 120.0
    ac_icd_codes: tuple[str, ...] = ("5712", "K7030", "K7031")


class AlcoholicCirrhosisExtractor(ICUBaseExtractor):
    COHORT_NAME = "alc_cirrhosis"
    def __init__(self, args, config: AlcoholicCirrhosisConfig = AlcoholicCirrhosisConfig()):
        super().__init__(args)
        self.config = config

    def extract_cohort(self):


        stays = self.initialize_icu_stays()
        stays = self.add_diagnosis_flags(stays, self.config.ac_icd_codes, "has_ac_diagnosis", match="exact")
        stays = self.add_timed_mortality(stays, days=28, col="mortality_28d")

        cohort = stays[stays["has_ac_diagnosis"] & stays["is_age_eligible"]].copy()
        cohort = self.first_stay_per_patient(cohort)
        cohort["label"] = cohort["mortality_28d"]

        self.save_cohort(cohort)
