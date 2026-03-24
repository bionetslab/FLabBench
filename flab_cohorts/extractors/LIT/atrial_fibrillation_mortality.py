"""
This class extracts the atrial fibrillation mortality cohort from the MIMIC dataset.
Reference: https://pmc.ncbi.nlm.nih.gov/articles/PMC11667998/
"""
# ICU
#ULCER: Stage II 

import pandas as pd
from dataclasses import dataclass
from tqdm import tqdm
tqdm.pandas()


from flab_cohorts.extractors.base import BaseExtractor
from flab_cohorts.utils.dataset_loader import load_admissions, load_diagnoses, load_patients, load_icu_stays
from flab_cohorts.utils.logger import get_logger

logger = get_logger("ATRIAL_FIBRILLATION_MORTALITY")

#HF Heart Failure
#AF Atrial Fibrillation

@dataclass
class AtrialFibrillationConfig:
    age_min: float = 18.0
    age_max: float = 120.0
    min_los_days: float = 1.0
    observation_window_hours: float = 24.0

    # Heart failure – HF
    HF_ICD10_codes: tuple[str, ...] = ("I5021", "I5023", "I5031", "I5033", "I5041", "I5043", "I50811", "I50813","I5022", "I5032", "I5042", "I50812")
    HF_ICD9_codes: tuple[str, ...] = ("42821", "42823", "42831", "42833", "42841", "42843", "42822", "42832", "42842")
    # Atrial fibrillation – AF
    AF_ICD9_codes: tuple[str, ...] = ("42731",)
    AF_ICD10_codes: tuple[str, ...] = ("I480", "I481", "I482", "I4891")
    
    


class AtrialFibrillationExtractor(BaseExtractor):
    def __init__(self, args, config: AtrialFibrillationConfig = AtrialFibrillationConfig()):
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
        
    def add_HF_and_AF_diagnosis(self, stays: pd.DataFrame) -> pd.DataFrame:
        
        diags = load_diagnoses(self.data_path)
        HF_ids = diags[diags["icd_code"].isin(self.config.HF_ICD10_codes + self.config.HF_ICD9_codes)]
        AF_ids = diags[diags["icd_code"].str.startswith(self.config.AF_ICD10_codes) | (diags["icd_code"].isin(self.config.AF_ICD9_codes))]
        stays["has_HF_diagnosis"] = stays["hadm_id"].isin(HF_ids["hadm_id"])
        stays["has_AF_diagnosis"] = stays["hadm_id"].isin(AF_ids["hadm_id"])

        return stays

    
   
    def add_inhospital_mortality_label(self, stays: pd.DataFrame) -> pd.DataFrame:

        stays["inhospital_mortality"] = stays["deathtime"].notna()
        return stays 
    
    def extract_cohort(self):
        """Run atrial fibrillation mortality cohort extraction."""
        
        stays = self.prepare_stays()
        stays = self.add_HF_and_AF_diagnosis(stays)
        stays = self.add_inhospital_mortality_label(stays)
        

        
        cohort = stays[stays["has_HF_diagnosis"] & stays["has_AF_diagnosis"]].copy()
        cohort = cohort.sort_values(["subject_id", "intime"])
        cohort["is_first_icustay_with_HF_and_AF"] = (cohort.groupby("subject_id")["intime"].transform("min") == cohort["intime"])

        cohort = cohort[cohort["is_first_icustay_with_HF_and_AF"] & cohort["has_min_icu_los"] & cohort["is_age_eligible"]]

        
            
        self.save_cohort(cohort)


    def save_cohort(self, cohort: pd.DataFrame) -> None:
        """Save final atrial fibrillation mortality cohort and report summary stats."""
        
        cohort = cohort.rename(columns={"inhospital_mortality": "label"})
        cols = ["subject_id", "hadm_id", "stay_id", "intime", "outtime", "race", "los", "gender", "age", "dod", "label"]
        cohort = cohort[cols]

        pct = 100 * cohort["label"].mean()
        logger.info("Number of ICU stays in atrial fibrillation mortality cohort: %s", cohort.stay_id.nunique())
        logger.info("Number of patients in atrial fibrillation mortality cohort: %s", cohort.subject_id.nunique())
        logger.info("Number of stays with atrial fibrillation mortality: %s", cohort[cohort["label"] == 1].stay_id.nunique())
        logger.info("Atrial fibrillation mortality positive rate: %.2f%%", pct)

        cohort.to_csv(self.paths["cohort_path"] / "cohort_atrial_fibrillation_mortality.csv", index=False)
        logger.info("Atrial fibrillation mortality cohort saved.")
