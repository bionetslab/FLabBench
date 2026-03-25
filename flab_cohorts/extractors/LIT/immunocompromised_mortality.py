"""
This class extracts the immunocompromised 28-day mortality cohort from the MIMIC dataset.
Reference: https://link.springer.com/article/10.1186/s40001-025-02622-3
"""
# ICU
# Immunocompromised
# 28-day all-cause mortality from ICU intime

#TODO: check the ICD codes again

import pandas as pd
from dataclasses import dataclass
from tqdm import tqdm
tqdm.pandas()

from flab_cohorts.extractors.base import BaseExtractor
from flab_cohorts.utils.dataset_loader import load_admissions, load_diagnoses, load_patients, load_icu_stays
from flab_cohorts.utils.logger import get_logger
from flab_cohorts.extractors.LIT.cohort_utils import save_cohort

logger = get_logger("IMMUNOCOMPROMISED_MORTALITY")


@dataclass
class ImmunocompromisedConfig:
    age_min: float = 18.0
    age_max: float = 120.0
    min_los_hours: float = 6.0
    mortality_days: float = 28.0

    # Primary immunodeficiency
    PID_ICD10_codes: tuple[str, ...] = ("D80", "D81", "D82", "D83", "D84", "D89")
    PID_ICD9_codes: tuple[str, ...] = ("279",)
    # HIV
    HIV_ICD10_codes: tuple[str, ...] = ("B20", "B21", "B22", "B23", "B24")
    HIV_ICD9_codes: tuple[str, ...] = ("042", "043", "044")
    # Hematologic malignancy (C81–C96 / 200–208)
    HEME_ICD10_codes: tuple[str, ...] = ("C81","C82","C83","C84","C85","C86","C87","C88","C89","C90","C91","C92","C93","C94","C95","C96")
    HEME_ICD9_codes: tuple[str, ...] = ("200","201","202","203","204","205","206","207","208")
    # Metastatic cancer
    META_ICD10_codes: tuple[str, ...] = ("C77", "C78", "C79", "C80")
    META_ICD9_codes: tuple[str, ...] = ("196", "197", "198", "199")
    # Solid organ transplant / HSCT
    TX_ICD10_codes: tuple[str, ...] = ("Z94", "T86")
    TX_ICD9_codes: tuple[str, ...] = ("V42", "9968")
    # Immunosuppressive therapy / chemo / long-term steroids
    THERAPY_ICD10_codes: tuple[str, ...] = ("Z7952", "Z7989", "Z9221", "Z5111", "Z5112")
    THERAPY_ICD9_codes: tuple[str, ...] = ("V580", "V5811", "V5869")


    ALL_ICD_CODES: tuple[str, ...] = (PID_ICD10_codes +
                                      PID_ICD9_codes + 
                                      HIV_ICD10_codes +
                                      HIV_ICD9_codes +
                                      HEME_ICD10_codes +
                                      HEME_ICD9_codes +
                                      META_ICD10_codes +
                                      META_ICD9_codes +
                                      TX_ICD10_codes +
                                      TX_ICD9_codes +
                                      THERAPY_ICD10_codes +
                                      THERAPY_ICD9_codes)

class ImmunocompromisedExtractor(BaseExtractor):
    def __init__(self, args, config: ImmunocompromisedConfig = ImmunocompromisedConfig()):
        super().__init__(args)
        self.config = config

    def prepare_stays(self) -> pd.DataFrame:
        stays = load_icu_stays(self.data_path)
        stays = stays.merge(self.patients, on="subject_id", how="left")
        stays["is_age_eligible"] = (stays["age"] >= self.config.age_min) & (stays["age"] <= self.config.age_max)
        stays = stays.merge(self.adms[["hadm_id", "deathtime", "race"]], on="hadm_id", how="left")
        stays["icu_los_hours"] = (stays["outtime"] - stays["intime"]).dt.total_seconds() / 3600.0
        stays["has_min_icu_los"] = stays["icu_los_hours"] >= self.config.min_los_hours
        # first ICU stay per hospital admission (subject_id + hadm_id)
        stays = stays.sort_values(["subject_id", "intime"])
        stays["is_first_icustay"] = (stays.groupby("subject_id")["intime"].transform("min") == stays["intime"])
        return stays

    def add_diagnosis_labels(self, stays: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        diags = load_diagnoses(self.data_path)
        diags["icd_code"] = diags["icd_code"].str.replace(".", "", regex=False)

        immuno_mask = diags["icd_code"].str.startswith(cfg.ALL_ICD_CODES)

        stays["has_diagnosis"] = stays["hadm_id"].isin(diags.loc[immuno_mask, "hadm_id"])
                       
        return stays

    def add_28d_mortality_label(self, stays: pd.DataFrame) -> pd.DataFrame:

        stays["death_time"] = stays["deathtime"].fillna(stays["dod"])

        stays["days_to_death_from_icu"] = (
            (stays["death_time"] - stays["intime"]).dt.total_seconds() / 86400.0
        )
        stays["mortality_28d"] = (
            stays["death_time"].notna()
            & stays["intime"].notna()
            & (stays["death_time"] >= stays["intime"])
            & (stays["death_time"] <= stays["intime"] + pd.Timedelta(days=28))
        ).astype(int)
    
        return stays

    def extract_cohort(self):
        logger.info("Extracting immunocompromised 28-day mortality cohort")

        stays = self.prepare_stays()
        stays = self.add_diagnosis_labels(stays)
        stays = self.add_28d_mortality_label(stays)
        
        
        cohort = stays[stays["has_diagnosis"] & stays["is_age_eligible"] &stays["has_min_icu_los"]].copy()
        cohort = cohort.sort_values(["subject_id", "intime"])
        cohort["is_first_icustay_with_immuno"] = (cohort.groupby("subject_id")["intime"].transform("min") == cohort["intime"])
        cohort = cohort[cohort["is_first_icustay_with_immuno"]]
        
        cohort["label"] = cohort["mortality_28d"].astype(int)
    
        
        save_cohort(cohort, self.paths, "immunocompromised_mortality")

