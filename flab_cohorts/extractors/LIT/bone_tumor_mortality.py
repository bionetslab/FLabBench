"""
This class extracts the bone tumor mortality cohort from the MIMIC dataset.
Reference: https://pmc.ncbi.nlm.nih.gov/articles/PMC10978606/
"""
# ICU
# Secondary bone tumor
# long term mortality (1month to 3 years)


import pandas as pd
from dataclasses import dataclass
from tqdm import tqdm
tqdm.pandas()


from flab_cohorts.extractors.base import BaseExtractor
from flab_cohorts.utils.dataset_loader import load_admissions, load_diagnoses, load_patients, load_icu_stays
from flab_cohorts.utils.logger import get_logger

logger = get_logger("BONE_TUMOR_MORTALITY")


@dataclass
class BoneTumorConfig:
    age_min: float = 18.0
    age_max: float = 120.0
    min_los_days: float = 1.0
    observation_window_hours: float = 24.0

    '''BT_ICD9_codes: tuple[str, ...] = ("170", "1985", "V1081")
    BT_ICD10_codes: tuple[str, ...] = ("C40", "C41", "C795", "Z8583", "D16")'''


    BT_ICD9_codes = ("1985",)
    BT_ICD10_codes = ("C7B03", "C795")

    
    
class BoneTumorExtractor(BaseExtractor):
    def __init__(self, args, config: BoneTumorConfig = BoneTumorConfig()):
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
        bt_ids = diags[diags["icd_code"].str.startswith(self.config.BT_ICD10_codes + self.config.BT_ICD9_codes)]
        stays["has_diagnosis"] = stays["hadm_id"].isin(bt_ids["hadm_id"])
    
        return stays

    
   
    def add_survival_label(self, stays: pd.DataFrame) -> pd.DataFrame:


        stays["days_to_death_from_icu"] = ((stays["dod"] - stays["intime"]).dt.total_seconds() / 86400.0)   
        not_valid = stays[stays["days_to_death_from_icu"] < -1]["hadm_id"].unique() # only 6 admission wrong dod
        stays = stays[~stays["hadm_id"].isin(not_valid)]
        stays["days_to_death_from_icu"] = stays["days_to_death_from_icu"].clip(lower=0) # dod is on the same day of the intime


        died = stays["dod"].notna() & stays["intime"].notna() & (stays["dod"] >= stays["intime"])

        stays["survival_1m"] = (died & (stays["days_to_death_from_icu"] >= 31)).astype(int)
        stays["survival_3m"] = (died & (stays["days_to_death_from_icu"] >= 90)).astype(int)
        stays["survival_1y"] = (died & (stays["days_to_death_from_icu"] >= 365)).astype(int)
        stays["survival_3y"] = (died & (stays["days_to_death_from_icu"] >= 3 * 365)).astype(int)
        stays["long_term_mortality"] = (stays["days_to_death_from_icu"] >= 31).astype(int) #& (stays["days_to_death_from_icu"] <= 3 * 365).astype(int)
        
        return stays 
    
    def extract_cohort(self):
        """Run atrial fibrillation mortality cohort extraction."""
        
        logger.info("Extracting bone tumor long termmortality cohort")
        
        stays = self.prepare_stays()
        stays = self.add_diagnosis_labels(stays)
        stays = self.add_survival_label(stays)
        
        cohort = stays[stays["has_diagnosis"]].copy()    
        cohort["is_first_icustay_with_BT"] = (cohort.groupby("subject_id")["intime"].transform("min") == cohort["intime"])

          
        cohort = cohort[cohort["is_first_icustay_with_BT"] & cohort["has_min_icu_los"] & cohort["is_age_eligible"]]


            
        self.save_cohort(cohort)


    def save_cohort(self, cohort: pd.DataFrame) -> None:
        
        cohort = cohort.rename(columns={"long_term_mortality": "label"})
        cols = ["subject_id", "hadm_id", "stay_id", "intime", "outtime", "race", "los", "gender", "age", "dod", "label"]
        cohort = cohort[cols]

        pct = 100 * cohort["label"].mean()
        logger.info("Number of ICU stays cohort: %s", cohort.stay_id.nunique())
        logger.info("Number of patients in cohort: %s", cohort.subject_id.nunique())
        logger.info("Number of stays: %s", cohort[cohort["label"] == 1].stay_id.nunique())
        logger.info("Positive stays rate: %.2f%%", pct)

        cohort.to_csv(self.paths["cohort_path"] / "cohort_bone_tumor_mortality.csv", index=False)
        logger.info("Cohort saved.")
