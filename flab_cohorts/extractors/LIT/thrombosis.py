"""
This class extracts the thrombosis cohort from the MIMIC dataset.
Reference: https://journals.sagepub.com/doi/full/10.1177/10760296251408357
"""

import pandas as pd
from dataclasses import dataclass, field

from flab_cohorts.extractors.base import ICUBaseExtractor
from flab_cohorts.utils.dataset_loader import load_labevents_for_cohort
from flab_cohorts.utils.logger import get_logger

logger = get_logger("THROMBOSIS")


@dataclass
class ThrombosisConfig:
    age_min: int = 18
    age_max: int = 300
    min_los_days: float = 1.0
    window_before_hours: int = 6
    window_after_hours: int = 24
    include_arterial: bool = False # only include Venous Thromboembolism

    CANCER_ICD_INCL: tuple = (
        ("C",)
        + tuple(str(i) for i in range(140, 209))
        + ("2090", "2091", "2092", "2093")
    )
    CANCER_ICD_EXCL: tuple = ("Z85",)
    
    # Venous Thromboembolism
    VTE_CODES: tuple = ("I26", "I80", "I81", "I82", "4151", "451", "452", "453")
    # Arterial Thromboembolism
    ARTERIAL_CODES: tuple = ("I74", "444")
    PREGNANCY_CODES: tuple = (
        ("O",)
        + tuple(str(i) for i in range(630, 680))
        + ("V22", "V23", "V24", "V27", "V28")
    )

    REQUIRED_LAB_ITEMIDS: dict = field(default_factory=lambda: {
        "platelet": [51265, 53189], # platelet count
        "hemoglobin": [50811, 50852, 50855, 51212, 51222, 51223, 51224, 51225, 51285, 51631, 51640, 51641, 51642, 51643, 51644, 51645, 51646, 51647, 52032, 52128, 52129, 52157],
        "wbc": [51300, 51301, 51516, 51755, 51756, 52407],
        "ddimer": [50915, 51196, 52551],
    })

class ThrombosisExtractor(ICUBaseExtractor):
    COHORT_NAME = "thrombosis"

    def __init__(self, args, config: ThrombosisConfig = ThrombosisConfig()):
        super().__init__(args)
        self.config = config

    def add_cancer_flag(self, stays):

        stays = self.add_diagnosis_flags(stays, icd_codes=self.config.CANCER_ICD_INCL, column="has_cancer")
        stays = self.add_diagnosis_flags(stays, icd_codes=self.config.CANCER_ICD_EXCL, column="_cancer_excl")
        stays["has_cancer"] = stays["has_cancer"] & ~stays["_cancer_excl"]
        return stays.drop(columns=["_cancer_excl"])

    def add_pregnancy_flag(self, stays):
        
        return self.add_diagnosis_flags(stays, icd_codes=self.config.PREGNANCY_CODES, column="is_pregnant")

    def add_thrombosis_flag(self, stays):
        
        codes = list(self.config.VTE_CODES)
        if self.config.include_arterial:
            codes += list(self.config.ARTERIAL_CODES)
        return self.add_diagnosis_flags(stays, icd_codes=tuple(codes), column="has_thrombosis")

    def filter_by_first_day_labs(self, stays):
        """Keep patients who have all required labs within the admission window."""
        
        all_ids = [iid for ids in self.config.REQUIRED_LAB_ITEMIDS.values() for iid in ids]

        labs = load_labevents_for_cohort(self.data_path, stays)
        labs = labs[labs["itemid"].isin(all_ids)]
        labs = labs.merge(stays[["subject_id", "hadm_id", "intime"]], on=["subject_id", "hadm_id"], how="inner")

        hours = (labs["charttime"] - labs["intime"]).dt.total_seconds() / 3600.0
        labs = labs[(hours >= -self.config.window_before_hours) & (hours <= self.config.window_after_hours)]

        patient_keys = set(stays[["subject_id", "hadm_id"]].itertuples(index=False, name=None))
        for lab_name, itemids in self.config.REQUIRED_LAB_ITEMIDS.items():
            has_lab = set(labs.loc[labs["itemid"].isin(itemids), ["subject_id", "hadm_id"]].itertuples(index=False, name=None))
            patient_keys &= has_lab

        keep = pd.DataFrame(list(patient_keys), columns=["subject_id", "hadm_id"])
        return stays.merge(keep, on=["subject_id", "hadm_id"], how="inner")

    def extract_cohort(self):

        stays = self.initialize_icu_stays()
        stays = self.add_cancer_flag(stays)
        stays = self.add_pregnancy_flag(stays)
        stays = self.add_thrombosis_flag(stays)



        cohort = stays[
            stays["is_first_icustay"]
            & stays["is_age_eligible"]
            & stays["has_min_icu_los"]
            & stays["has_cancer"]
            & ~stays["is_pregnant"]
        ].copy()


        #cohort = self.filter_by_first_day_labs(cohort) # only include patients with all required labs within the admission window

        cohort["label"] = cohort["has_thrombosis"].astype(int)

        self.save_cohort(cohort)
        return cohort
