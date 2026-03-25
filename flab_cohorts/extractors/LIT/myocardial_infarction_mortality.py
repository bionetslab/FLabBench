"""
This class extracts the myocardial infarction mortality cohort from the MIMIC dataset.
Reference: https://www.frontiersin.org/journals/cardiovascular-medicine/articles/10.3389/fcvm.2024.1368022/full
"""
# ICU

import pandas as pd
from dataclasses import dataclass
from tqdm import tqdm
tqdm.pandas()


from flab_cohorts.extractors.base import BaseExtractor
from flab_cohorts.utils.dataset_loader import load_admissions, load_diagnoses, load_patients, load_icu_stays
from flab_cohorts.utils.logger import get_logger
from flab_cohorts.extractors.LIT.cohort_utils import save_cohort

logger = get_logger("MYOCARDIAL_INFARCTION_MORTALITY")

#HF Heart Failure
#AF Atrial Fibrillation

@dataclass
class MyocardialInfarctionConfig:
    age_min: float = 18.0
    age_max: float = 120.0
    min_los_days: float = 1.0
    observation_window_hours: float = 24.0
    mortality_days: int = 30
    
    ICD9_CODES: tuple[str, ...] = ("410",)
    ICD10_CODES: tuple[str, ...] = ("I210", "I211", "I212", "I213", "I214")

    
    ALL_CODES = ICD9_CODES + ICD10_CODES

class MyocardialInfarctionExtractor(BaseExtractor):
    def __init__(self, args, config: MyocardialInfarctionConfig = MyocardialInfarctionConfig()):
        super().__init__(args)
        self.config = config
        
    def prepare_stays(self) -> pd.DataFrame:
        
        stays = load_icu_stays(self.data_path)
        stays = stays.merge(self.patients, on="subject_id", how="left")
        stays["is_age_eligible"] = (stays["age"] >= self.config.age_min) & (stays["age"] <= self.config.age_max)
        stays = stays.merge(self.adms[["hadm_id", "deathtime", "race"]], on="hadm_id", how="left")
        stays["has_min_icu_los"] = stays["los"] > self.config.min_los_days
        stays = stays.sort_values(["subject_id", "intime"])
        stays["is_first_icustay"] = (stays.groupby("subject_id")["intime"].transform("min") == stays["intime"])
        return stays
        
    def add_diagnosis_labels(self, stays: pd.DataFrame) -> pd.DataFrame:
        
        diags = load_diagnoses(self.data_path)
        diags["icd_code"] = diags["icd_code"].str.replace(".", "")
        ids = diags[diags["icd_code"].str.startswith(self.config.ALL_CODES)]
        stays["has_diagnosis"] = stays["hadm_id"].isin(ids["hadm_id"])
        return stays

    
    def add_ccu_labels(self, stays: pd.DataFrame) -> pd.DataFrame:
        
        stays["is_ccu_stay"] = stays["first_careunit"].str.contains("CCU", na=False) | stays["first_careunit"].str.contains("CORONARY", na=False)
        return stays
        
        
    def add_mortality_label(self, stays: pd.DataFrame, days: int) -> pd.DataFrame:

        stays["death_time"] = stays["deathtime"].fillna(stays["dod"])

        stays["days_to_death_from_icu"] = (
            (stays["death_time"] - stays["intime"]).dt.total_seconds() / 86400.0
        )
        stays["mortality_days"] = (
            stays["death_time"].notna()
            & stays["intime"].notna()
            & (stays["death_time"] >= stays["intime"])
            & (stays["death_time"] <= stays["intime"] + pd.Timedelta(days=days))
        ).astype(int)
    
        return stays
   
    
    def extract_cohort(self):
        """Run atrial fibrillation mortality cohort extraction."""
        
        stays = self.prepare_stays()
        stays = self.add_diagnosis_labels(stays)
        stays = self.add_ccu_labels(stays)
        stays = self.add_mortality_label(stays,days=self.config.mortality_days)
        
        
        cohort = stays[stays["has_diagnosis"] & stays["is_ccu_stay"]]
        cohort = cohort.sort_values(["subject_id", "intime"])
        cohort["is_first_icustay_with_diagnosis"] = (cohort.groupby("subject_id")["intime"].transform("min") == cohort["intime"])
        
        cohort = cohort[cohort["is_first_icustay_with_diagnosis"] & cohort['is_age_eligible']]
        
        cohort["label"] = cohort["mortality_days"].astype(int)
            
        save_cohort(cohort, self.paths, "myocardial_infarction_mortality")

