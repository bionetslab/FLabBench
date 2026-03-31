"""
This class extracts the catheter-associated urinary tract infection (CAUTI) mortality cohort from MIMIC-IV.
Reference: https://pubmed.ncbi.nlm.nih.gov/40995080/
"""

from dataclasses import dataclass

from flab_cohorts.extractors.base import ICUBaseExtractor
from flab_cohorts.utils.logger import get_logger

logger = get_logger("URINARY_TRACT_INFECTION_MORTALITY")


@dataclass
class UrinaryTractInfectionConfig:
    age_min: float = 18.0
    age_max: float = 120.0
    CAUTI_ICD9_CODES: tuple[str, ...] = ("99664",)
    CAUTI_ICD10_CODES: tuple[str, ...] = ("T83511", "T8351X")
    ALL_CODES: tuple[str, ...] = CAUTI_ICD9_CODES + CAUTI_ICD10_CODES


class UrinaryTractInfectionExtractor(ICUBaseExtractor):
    def __init__(self, args, config: UrinaryTractInfectionConfig = UrinaryTractInfectionConfig()):
        super().__init__(args)
        self.config = config

    def extract_cohort(self) -> None:
        logger.info("Extracting urinary tract infection in-hospital mortality cohort")

        stays = self.initialize_stays()
        stays = self.add_diagnosis_flags(stays)
        stays = self.add_inhospital_mortality(stays)

        cohort = stays[stays["has_diagnosis"] & stays["is_age_eligible"]].copy()
        cohort = self.first_stay_per_patient(cohort)
        cohort["label"] = cohort["in_hospital_mortality"].astype(int)

        self.save_cohort(cohort, "urinary_tract_infection_mortality")
