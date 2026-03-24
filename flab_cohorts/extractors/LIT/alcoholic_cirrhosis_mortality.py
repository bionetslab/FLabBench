"""
This class extracts the alcoholic cirrhosis mortality cohort from the MIMIC dataset.
Reference: ??
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

logger = get_logger("ALCOHOLIC_CIRRHOSIS_MORTALITY")


@dataclass
class AlcoholicCirrhosisConfig:
    age_min: float = 18.0
    age_max: float = 120.0
    min_los_days: float = 1.0
    observation_window_hours: float = 24.0
    ac_icd_codes: tuple[str, ...] = ("5712","K7030", "K7031") 


class AlcoholicCirrhosisExtractor(BaseExtractor):
    def __init__(self, args, config: AlcoholicCirrhosisConfig = AlcoholicCirrhosisConfig()):
        super().__init__(args)
        self.config = config
        
    def prepare_stays(self) -> pd.DataFrame:
        
        stays = load_icu_stays(self.data_path)
        stays = stays.merge(self.patients, on="subject_id", how="left")
        stays["is_age_eligible"] = (stays["age"] >= self.config.age_min) & (stays["age"] <= self.config.age_max)
        stays = stays.merge(self.adms[["hadm_id", "deathtime", "race"]], on="hadm_id", how="left")
        stays = stays.sort_values(["subject_id", "intime"])
        stays["is_first_icustay"] = (stays.groupby("subject_id")["intime"].transform("min") == stays["intime"])
        return stays
        
    def add_ac_diagnosis(self, stays: pd.DataFrame) -> pd.DataFrame:
        
        diags = load_diagnoses(self.data_path)
        ac_ids = diags[diags["icd_code"].isin(self.config.ac_icd_codes)]
        stays["has_ac_diagnosis"] = stays["hadm_id"].isin(ac_ids["hadm_id"])
        return stays
    
    def add_mortality_label(self, stays: pd.DataFrame) -> pd.DataFrame:

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
        """Run alcoholic cirrhosis mortality cohort extraction."""

        stays = self.prepare_stays()
        stays = self.add_ac_diagnosis(stays)
        stays = self.add_mortality_label(stays)

        # Match paper script logic:
        # 1) keep AC stays, 2) then keep first ICU stay per subject within AC stays.
        cohort = stays[stays["has_ac_diagnosis"] & stays["is_age_eligible"]].copy()
        cohort = cohort.sort_values(["subject_id", "intime"])
        cohort["is_first_ac_icustay"] = (cohort.groupby("subject_id")["intime"].transform("min") == cohort["intime"])
        #cohort = cohort.groupby("subject_id", as_index=False).head(1)
        cohort = cohort[cohort["is_first_ac_icustay"]]

        self.save_cohort(cohort)

    def save_cohort(self, cohort: pd.DataFrame) -> None:
        """Save final alcoholic cirrhosis mortality cohort and report summary stats."""
        
        cohort = cohort.rename(columns={"mortality_28d": "label"})
        cols = ["subject_id", "hadm_id", "stay_id", "intime", "outtime", "race", "los", "gender", "age", "dod", "label"]
        cohort = cohort[cols]

        pct = 100 * cohort["label"].mean()
        logger.info("Number of ICU stays in alcoholic cirrhosis mortality cohort: %s", cohort.stay_id.nunique())
        logger.info("Number of patients in alcoholic cirrhosis mortality cohort: %s", cohort.subject_id.nunique())
        logger.info("Number of stays with alcoholic cirrhosis mortality: %s", cohort[cohort["label"] == 1].stay_id.nunique())
        logger.info("Alcoholic cirrhosis mortality positive rate: %.2f%%", pct)

        cohort.to_csv(self.paths["cohort_path"] / "cohort_alcoholic_cirrhosis_mortality.csv", index=False)
        logger.info("Alcoholic cirrhosis mortality cohort saved.")
