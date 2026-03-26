"""
This class extracts the obesity pneumonia cohort from the MIMIC dataset.
Reference:https://pubmed.ncbi.nlm.nih.gov/40972486/
"""
# ICU

import pandas as pd
from dataclasses import dataclass
from tqdm import tqdm
tqdm.pandas()


from flab_cohorts.extractors.base import BaseExtractor
from flab_cohorts.utils.dataset_loader import load_admissions, load_diagnoses, load_patients, load_icu_stays, load_icu_chartevents_for_itemid
from flab_cohorts.utils.logger import get_logger
from flab_cohorts.extractors.LIT.cohort_utils import save_cohort

logger = get_logger("OBESITY_PNEUMONIA")

#Obesity
#Pneumonia

@dataclass
class ObesityPneumoniaConfig:
    age_min: float = 18.0
    age_max: float = 120.0
    min_los_days: float = 1.0
    observation_window_hours: float = 24.0
    mortality_days: int = 90
    
    # community-acquired pneumonia (CAP)
    CAP_ICD9_CODES = ("480", "481", "482", "483", "484", "485", "486")
    CAP_ICD10_CODES = ("J12", "J13", "J14", "J15", "J16", "J17", "J18")
        
    ALL_CODES = CAP_ICD9_CODES + CAP_ICD10_CODES
    
    HEIGHT_IDS = {226707, 1394}     # cm
    WEIGHT_IDS = {226512, 763}      # kg

class ObesityPneumoniaExtractor(BaseExtractor):
    def __init__(self, args, config: ObesityPneumoniaConfig = ObesityPneumoniaConfig()):
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

    def add_obesity_labels(self, stays: pd.DataFrame) -> pd.DataFrame:


        chartevents = load_icu_chartevents_for_itemid(self.data_path, list(self.config.HEIGHT_IDS) + list(self.config.WEIGHT_IDS))
        height = chartevents[chartevents.itemid.isin(self.config.HEIGHT_IDS)].groupby("hadm_id")["valuenum"].median()
        weight = chartevents[chartevents.itemid.isin(self.config.WEIGHT_IDS)].groupby("hadm_id")["valuenum"].median()
        stays["height_cm"] = stays["hadm_id"].map(height)
        stays["weight_kg"] = stays["hadm_id"].map(weight)
        #stays = stays.dropna(subset=["height_cm", "weight_kg"])
        stays["bmi"] = stays["weight_kg"] / (stays["height_cm"] / 100) ** 2
        stays['has_bmi'] = stays["bmi"].notna()
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
        stays = self.add_obesity_labels(stays)
        stays = self.add_mortality_label(stays,days=self.config.mortality_days)
        
        cohort = stays[stays["has_diagnosis"]]
        cohort = cohort.sort_values(["subject_id", "intime"])
        cohort["is_first_icustay_with_diagnosis"] = (cohort.groupby("subject_id")["intime"].transform("min") == cohort["intime"])
        
        cohort = cohort[cohort["is_first_icustay_with_diagnosis"] & cohort["has_bmi"]]
 
        cohort["label"] = cohort["mortality_days"].astype(int)
        
        save_cohort(cohort, self.paths, "obesity_pneumonia")
        


