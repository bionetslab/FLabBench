"""
This class extracts the Pressure Ulcers cohort from the MIMIC dataset.
Reference:  https://onlinelibrary.wiley.com/doi/abs/10.1111/jocn.17860
"""

import pandas as pd
from dataclasses import dataclass

from flab_cohorts.extractors.base import ICUBaseExtractor
from flab_cohorts.utils.dataset_loader import load_icu_chartevents_for_itemid
from flab_cohorts.utils.logger import get_logger

logger = get_logger("PRESSURE_ULCER")


@dataclass
class PressureUlcerConfig:
    age_min: float = 18.0
    age_max: float = 120.0
    min_los_days: float = 1.0
    observation_window_hours: float = 24.0
    pu_chart_itemids: tuple[int, ...] = (551, 552, 553, 224631, 224965, 224966)
    pu_stage2_itemids: tuple[int, ...] = (552, 553, 224965, 224966)


class PressureUlcerExtractor(ICUBaseExtractor):
    def __init__(self, args, config: PressureUlcerConfig = PressureUlcerConfig()):
        super().__init__(args)
        self.config = config

    def add_pu_stage2_labels(self, stays: pd.DataFrame) -> pd.DataFrame:
        pu_events = load_icu_chartevents_for_itemid(self.data_path, list(self.config.pu_chart_itemids))
        pu_merged = pu_events.merge(stays[["stay_id", "intime", "outtime"]], on="stay_id", how="inner")
        pu_merged = pu_merged[
            (pu_merged["charttime"] >= pu_merged["intime"])
            & (pu_merged["charttime"] <= pu_merged["outtime"])
        ]
        pu_merged = pu_merged[pu_merged["itemid"].isin(self.config.pu_stage2_itemids)]
        pu_merged["hours_since_icu_admit"] = (
            (pu_merged["charttime"] - pu_merged["intime"]).dt.total_seconds() / 3600
        )

        pu_ow = pu_merged[
            pu_merged["hours_since_icu_admit"] < self.config.observation_window_hours
        ]
        pu_pw = pu_merged[
            pu_merged["hours_since_icu_admit"] >= self.config.observation_window_hours
        ]
        stays["has_pu_in_observation_window"] = stays["stay_id"].isin(pu_ow["stay_id"])
        stays["has_pu_in_prediction_window"] = stays["stay_id"].isin(pu_pw["stay_id"])
        return stays

    def extract_cohort(self):
        stays = self.initialize_icu_stays()
        stays = self.add_pu_stage2_labels(stays)

        cohort = stays[
            stays["is_age_eligible"]
            & stays["has_min_icu_los"]
            & ~stays["has_pu_in_observation_window"]
        ].copy()
        cohort["label"] = cohort["has_pu_in_prediction_window"].astype(int)

        self.save_cohort(cohort, "pressure_ulcer")
