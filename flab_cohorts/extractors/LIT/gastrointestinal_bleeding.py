"""
This class extracts the gastrointestinal bleeding cohort from the MIMIC dataset.
Reference:  https://pmc.ncbi.nlm.nih.gov/articles/PMC7813389/pdf/bmjhci-2020-100245.pdf
"""

import pandas as pd
from dataclasses import dataclass

from flab_cohorts.extractors.base import ICUBaseExtractor
from flab_cohorts.utils.dataset_loader import load_icu_items, load_icu_inputevents
from flab_cohorts.utils.logger import get_logger

logger = get_logger("GI_BLEED")


@dataclass
class GIBleedingConfig:
    age_min: float = 18.0
    age_max: float = 120.0
    transfusion_after_hours: float = 5.0
    transfusion_labels: tuple[str, ...] = ("PRBC", "PACKED RBC")
    gi_bleeding_codes: tuple[str, ...] = (
        "5307", "5693", "5780", "5781", "5789",
        "53021", "53082", "53100", "53101", "53120", "53121", "53140",
        "53141", "53160", "53161", "53200", "53201", "53220", "53221",
        "53240", "53241", "53260", "53261", "53300", "53301", "53320",
        "53321", "53340", "53341", "53360", "53361", "53400", "53401",
        "53420", "53421", "53440", "53441", "53460", "53461", "53501",
        "53511", "53521", "53531", "53541", "53551", "53561", "53571",
        "53784", "56202", "56203", "56212", "56213", "56985",
    )
    ALL_CODES = gi_bleeding_codes


class GastrointestinalBleedingExtractor(ICUBaseExtractor):
    def __init__(self, args, config: GIBleedingConfig = GIBleedingConfig()):
        super().__init__(args)
        self.config = config

    def add_transfusions(self, stays: pd.DataFrame) -> pd.DataFrame:
        items = load_icu_items(self.data_path)
        input_events = load_icu_inputevents(self.data_path)

        transfusion_ids = items[
            (items["label"].str.upper().str.contains(self.config.transfusion_labels[0]))
            | (items["label"].str.upper().str.contains(self.config.transfusion_labels[1]))
        ]["itemid"].tolist()
        input_trans = input_events[input_events["itemid"].isin(transfusion_ids)]

        input_trans_hadm = input_trans.merge(stays[["stay_id", "intime", "outtime", "los"]], on="stay_id", how="inner")
        input_trans_hadm = input_trans_hadm[
            (input_trans_hadm["starttime"] <= input_trans_hadm["outtime"])
            & (input_trans_hadm["starttime"] >= input_trans_hadm["intime"])
        ]
        input_trans_hadm["hours_since_icu_admit"] = (
            (input_trans_hadm["starttime"] - input_trans_hadm["intime"]).dt.total_seconds() / 3600
        )

        stays["has_transf"] = stays["stay_id"].isin(input_trans_hadm["stay_id"])
        stays["has_transf_after5h"] = stays["stay_id"].isin(
            input_trans_hadm[
                input_trans_hadm["hours_since_icu_admit"] > self.config.transfusion_after_hours
            ]["stay_id"]
        )
        return stays

    def extract_cohort(self):
        
        stays = self.initialize_icu_stays()
        stays["is_first_icustay_hadm"] = (stays.groupby("hadm_id")["intime"].transform("min") == stays["intime"])
        stays = self.add_diagnosis_flags(stays, match="exact")
        stays = self.add_transfusions(stays)

        cohort = stays[
            stays["is_first_icustay_hadm"]
            & stays["is_age_eligible"]
            & stays["has_diagnosis"]
        ].copy()
        
        cohort["label"] = cohort["has_transf_after5h"].astype(int)

        self.save_cohort(cohort, "gi_bleeding")
