"""
This class extracts the Pressure Ulcers cohort from the MIMIC dataset.
Reference:  https://onlinelibrary.wiley.com/doi/abs/10.1111/jocn.17860

"""
# ICU
#ULCER: Stage II 

import pandas as pd
from dataclasses import dataclass
from tqdm import tqdm
tqdm.pandas()


from flab_cohorts.extractors.base import BaseExtractor
from flab_cohorts.utils.dataset_loader import load_icu_stays, load_icu_chartevents_for_itemid
from flab_cohorts.utils.logger import get_logger

logger = get_logger("PRESSURE_ULCER")


@dataclass
class PressureUlcerConfig:
    age_min: float = 18.0
    min_los_days: float = 1.0
    observation_window_hours: float = 24.0
    pu_chart_itemids: tuple[int, ...] = (551, 552, 553, 224631, 224965, 224966)
    pu_stage2_itemids: tuple[int, ...] = (552, 553, 224965, 224966)


class PressureUlcerExtractor(BaseExtractor):

    def __init__(self, args, config: PressureUlcerConfig = PressureUlcerConfig()):
        super().__init__(args)
        self.config = config

    def prepare_stays(self) -> pd.DataFrame:
        
        stays = load_icu_stays(self.data_path)
        stays = stays.merge(self.patients, on="subject_id", how="left")
        stays = stays.merge(self.adms[["hadm_id", "race"]], on="hadm_id", how="left")

        stays["is_age_eligible"] = stays["age"] >= self.config.age_min
        stays["has_min_los"] = stays["los"] >= self.config.min_los_days
        return stays


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

        pu_ow = pu_merged[pu_merged["hours_since_icu_admit"] < self.config.observation_window_hours]
        pu_pw = pu_merged[pu_merged["hours_since_icu_admit"] >= self.config.observation_window_hours]

        stays["has_pu_in_observation_window"] = stays["stay_id"].isin(pu_ow["stay_id"])
        stays["has_pu_in_prediction_window"] = stays["stay_id"].isin(pu_pw["stay_id"])
        
        return stays

    def extract_cohort(self):
        """Run pressure ulcer cohort extraction."""
        stays = self.prepare_stays()
        stays = self.add_pu_stage2_labels(stays)

        inclusion_mask = (
            stays["is_age_eligible"]
            & stays["has_min_los"]
            & ~stays["has_pu_in_observation_window"]
        )
        cohort = stays.loc[inclusion_mask].copy()
        self.save_cohort(cohort)
        

    def save_cohort(self, cohort: pd.DataFrame) -> None:
        """Save final pressure ulcer cohort and report summary stats."""
        
        cohort = cohort.rename(columns={"has_pu_in_prediction_window": "label"})
        cols = ["subject_id", "hadm_id", "stay_id", "intime", "outtime", "race", "los", "gender", "age", "dod", "label"]
        cohort = cohort[cols]

        pct = 100 * cohort["label"].mean()
        logger.info("Number of ICU stays in PU cohort: %s", cohort.stay_id.nunique())
        logger.info("Number of patients in PU cohort: %s", cohort.subject_id.nunique())
        logger.info("Number of stays with PU: %s", cohort[cohort["label"] == 1].stay_id.nunique())
        logger.info("PU positive rate: %.2f%%", pct)

        cohort.to_csv(self.paths["cohort_path"] / "cohort_pressure_ulcer.csv", index=False)
        logger.info("Pressure ulcer cohort saved.")
