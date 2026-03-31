"""
This class extracts the acute kidney injury cohort from the MIMIC dataset.
Reference:  https://www.sciencedirect.com/science/article/pii/S1532046420302811?
possibly compare with https://github.com/ExaScience/Aki-Predictor
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm
tqdm.pandas()

from flab_cohorts.extractors.base import ICUBaseExtractor
from flab_cohorts.utils.dataset_loader import load_icu_procedures, load_labevents_for_itemid
from flab_cohorts.utils.logger import get_logger

logger = get_logger("AKI")


@dataclass
class AKIConfig:
    age_min: float = 18.0
    age_max: float = 89.0
    min_los_days: float = 1.0
    creatinine_itemid: int = 50912
    dialysis_codes: tuple[int, ...] = (225441, 225802, 225803, 225809, 225955, 225805)
    rrt_codes: tuple[int, ...] = (225802,)
    eGFR_normal: float = 75.0
    ckd_icd_codes: str = "585"
    observation_window_hours: float = 24.0
    prediction_window_hours: float = 72.0


class AcuteKidneyInjuryExtractor(ICUBaseExtractor):
    def __init__(self, args, config: AKIConfig = AKIConfig()):
        super().__init__(args)
        self.config = config

    def add_ckd_diagnosis(self, stays: pd.DataFrame) -> pd.DataFrame:
        
        diag_ckd = self.diags[self.diags["icd_code"].str.startswith(self.config.ckd_icd_codes)]
        diag_ckd = diag_ckd.merge(self.adms[["hadm_id", "admittime", "dischtime"]], on="hadm_id", how="left")
        diag_ckd_first = (
            diag_ckd.sort_values(["subject_id", "admittime"])
            .groupby("subject_id").first().reset_index()
            .rename(columns={"admittime": "cdk_admittime"})
        )

        stays = stays.merge(diag_ckd_first[["subject_id", "icd_code", "cdk_admittime"]], on="subject_id", how="left")
        stays["ckd"] = stays["icd_code"].notna() & (stays["intime"] >= stays["cdk_admittime"])
        return stays

    def add_dialysis_procedure(self, stays: pd.DataFrame) -> pd.DataFrame:
        
        procedures = load_icu_procedures(self.data_path)
        dial_proc = procedures[procedures["itemid"].isin(self.config.rrt_codes)]
        dial_proc = dial_proc.merge(stays[["stay_id", "intime"]], on="stay_id", how="inner")
        dial_proc["hours_since_icu_admit"] = (
            (dial_proc["starttime"] - dial_proc["intime"]).dt.total_seconds() / 3600
        )
        ow = self.config.observation_window_hours
        pw = self.config.prediction_window_hours
        dialysis_ow = dial_proc[
            (dial_proc["hours_since_icu_admit"] >= 0)
            & (dial_proc["hours_since_icu_admit"] < ow)
        ]
        dialysis_pw = dial_proc[
            (dial_proc["hours_since_icu_admit"] >= ow)
            & (dial_proc["hours_since_icu_admit"] <= pw)
        ]
        stays["dialysis_OW"] = stays["stay_id"].isin(dialysis_ow["stay_id"])
        stays["dialysis_PW"] = stays["stay_id"].isin(dialysis_pw["stay_id"])
        return stays

    def add_serum_creatinine_features(self, stays: pd.DataFrame) -> pd.DataFrame:
        scr_lab = load_labevents_for_itemid(self.data_path, self.config.creatinine_itemid)
        scr_hadm = scr_lab.merge(
            stays[["hadm_id", "stay_id", "intime", "los"]], on="hadm_id", how="inner",
        )
        scr_hadm["hours_since_icu_admit"] = (
            (scr_hadm["charttime"] - scr_hadm["intime"]).dt.total_seconds() / 3600
        )
        scr_hadm = scr_hadm.sort_values(["stay_id", "charttime"])

        ow = self.config.observation_window_hours
        pw = self.config.prediction_window_hours
        scr_ow = scr_hadm[
            (scr_hadm["hours_since_icu_admit"] >= 0)
            & (scr_hadm["hours_since_icu_admit"] < ow)
        ]
        scr_pw = scr_hadm[
            (scr_hadm["hours_since_icu_admit"] >= ow)
            & (scr_hadm["hours_since_icu_admit"] <= pw)
        ]

        scr_first = (
            scr_ow.sort_values(["stay_id", "charttime"])
            .groupby("stay_id").first().reset_index()[["stay_id", "valuenum"]]
            .rename(columns={"valuenum": "first_scr"})
        )
        scr_ow_max = (
            scr_ow.groupby("stay_id").valuenum.max().reset_index()
            .rename(columns={"valuenum": "max_has_src_in_obs_window"})
        )
        scr_pw_max = (
            scr_pw.groupby("stay_id").valuenum.max().reset_index()
            .rename(columns={"valuenum": "max_scr_pw"})
        )

        stays = stays.merge(scr_pw_max, on="stay_id", how="left")
        stays = stays.merge(scr_ow_max, on="stay_id", how="left")
        stays = stays.merge(scr_first, on="stay_id", how="left")
        stays["has_src_in_obs_window"] = stays["first_scr"].notna()
        stays["high_scr"] = stays["first_scr"] > 4

        aki_abs = scr_pw.groupby("stay_id").apply(self.aki_absolute).rename("AKI_abs_48h")
        aki_abs = aki_abs.reindex(stays["stay_id"]).fillna(False).reset_index()
        stays["AKI_abs_48h"] = stays["stay_id"].isin(aki_abs[aki_abs["AKI_abs_48h"]]["stay_id"])
        return stays

    def aki_absolute(self, group):
        times = group["charttime"].values
        scr = group["valuenum"].values
        n = len(scr)
        for i in range(n):
            for j in range(i + 1, n):
                if times[j] - times[i] > pd.Timedelta(hours=48):
                    break
                if scr[j] - scr[i] >= 0.3:
                    return True
        return False

    def compute_baseline_scr(self, row):
        if row["is_age_eligible"]:
            if row['ckd']:
                return row['first_scr']
            else:
                sex_factor = 0.742 if row['gender'] == 'F' else 1
                race_factor = 1.21 if row['is_black'] else 1
                age_factor = row['age'] ** 0.203
                scr_est = (
                    (175 * sex_factor * race_factor)
                    / (self.config.eGFR_normal * age_factor)
                ) ** (1 / 1.154)
                return scr_est
        else:
            return np.nan

    def extract_cohort(self):
        
        logger.info("Extracting acute kidney injury cohort")
        
        stays = self.initialize_stays()
        stays["is_black"] = stays["race"].isin(['BLACK/AFRICAN AMERICAN', 'BLACK/AFRICAN', 'BLACK/CAPE VERDEAN', 'BLACK/HAITIAN'])
        stays = self.add_ckd_diagnosis(stays)
        stays = self.add_dialysis_procedure(stays)
        stays = self.add_serum_creatinine_features(stays)

        stays['baseline_scr'] = stays.apply(self.compute_baseline_scr, axis=1)
        stays["AKI_base_50%_24h"] = stays["max_has_src_in_obs_window"] > stays['baseline_scr'] * 1.5
        stays["AKI_base_50%"] = stays["max_scr_pw"] > stays['baseline_scr'] * 1.5
        stays["AKI_24h"] = stays['high_scr']
        stays["AKI"] = stays['dialysis_PW'] | stays['AKI_abs_48h'] | stays['AKI_base_50%']

        cohort = stays[
            stays["is_age_eligible"]
            & stays["is_first_icustay"]
            & stays["has_min_icu_los"]
            & stays["has_src_in_obs_window"]
            & ~stays["AKI_24h"]
        ].copy()
        cohort["label"] = cohort["AKI"].astype(int)

        self.save_cohort(cohort, "aki")
