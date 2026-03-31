"""
This class extracts the obesity pneumonia cohort from the MIMIC dataset.
Reference: https://pubmed.ncbi.nlm.nih.gov/40972486/
"""

import pandas as pd
from dataclasses import dataclass

from flab_cohorts.extractors.base import ICUBaseExtractor
from flab_cohorts.utils.dataset_loader import load_icu_chartevents_for_itemid
from flab_cohorts.utils.logger import get_logger

logger = get_logger("OBESITY_PNEUMONIA")


@dataclass
class ObesityPneumoniaConfig:
    age_min: float = 18.0
    age_max: float = 120.0
    mortality_days: int = 90
    CAP_ICD9_CODES = ("480", "481", "482", "483", "484", "485", "486")
    CAP_ICD10_CODES = ("J12", "J13", "J14", "J15", "J16", "J17", "J18")
    ALL_CODES = CAP_ICD9_CODES + CAP_ICD10_CODES
    HEIGHT_IDS = {226707, 1394}
    WEIGHT_IDS = {226512, 763}


class ObesityPneumoniaExtractor(ICUBaseExtractor):
    COHORT_NAME = "obesity_pneumonia"
    def __init__(self, args, config: ObesityPneumoniaConfig = ObesityPneumoniaConfig()):
        super().__init__(args)
        self.config = config

    def add_obesity_labels(self, stays: pd.DataFrame) -> pd.DataFrame:
        chartevents = load_icu_chartevents_for_itemid(self.data_path,list(self.config.HEIGHT_IDS) + list(self.config.WEIGHT_IDS))
        
        height = chartevents[chartevents.itemid.isin(self.config.HEIGHT_IDS)].groupby("hadm_id")["valuenum"].median()
        weight = chartevents[chartevents.itemid.isin(self.config.WEIGHT_IDS)].groupby("hadm_id")["valuenum"].median()
        stays["height_cm"] = stays["hadm_id"].map(height)
        stays["weight_kg"] = stays["hadm_id"].map(weight)
        stays["bmi"] = stays["weight_kg"] / (stays["height_cm"] / 100) ** 2
        stays["has_bmi"] = stays["bmi"].notna()
        return stays

    def extract_cohort(self):

        stays = self.initialize_icu_stays()
        stays = self.add_diagnosis_flags(stays)
        stays = self.add_obesity_labels(stays)
        stays = self.add_timed_mortality(stays, days=self.config.mortality_days, col="mortality_days")

        cohort = stays[stays["has_diagnosis"]].copy()
        cohort = self.first_stay_per_patient(cohort)
        cohort = cohort[cohort["has_bmi"]]
        cohort["label"] = cohort["mortality_days"]

        self.save_cohort(cohort)
