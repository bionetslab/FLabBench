"""
This class extracts a NOAF (new-onset atrial fibrillation) prediction cohort from MIMIC-IV.
Reference: Interpretable machine learning model for new-onset atrial fibrillation prediction in critically ill patients.
"""

import re
import pandas as pd
from dataclasses import dataclass

from flab_cohorts.extractors.base import BaseExtractor
from flab_cohorts.utils.dataset_loader import (
    load_diagnoses,
    load_icu_chartevents_for_itemid,
    load_icu_stays,
    load_procedures,
)
from flab_cohorts.utils.logger import get_logger
from flab_cohorts.extractors.LIT.cohort_utils import save_cohort

logger = get_logger("ATRIAL_FIBRILLATION_NOAF")


@dataclass
class AtrialFibrillationConfig:
    age_min: float = 18.0
    age_max: float = 120.0
    min_los_days: float = 2.0
    observation_window_hours: float = 24.0

    AF_ICD9_EXACT: tuple[str, ...] = ("42731", "42732")
    AF_ICD10_PREFIXES: tuple[str, ...] = ("I48",)
    AF_ALL_PREFIXES: tuple[str, ...] = AF_ICD10_PREFIXES

    CABG_ICD9_PREFIXES: tuple[str, ...] = ("361",)
    VALVE_ICD9_PREFIXES: tuple[str, ...] = ("35",)
    CABG_ICD10_PCS_PREFIXES: tuple[str, ...] = ("0210",)
    VALVE_ICD10_PCS_PREFIXES: tuple[str, ...] = ("02Q", "02R")
    CARDIAC_SURG_ALL_PREFIXES: tuple[str, ...] = ("361", "35", "0210", "02Q", "02R")

    AF_VALUE_PATTERN: str = (
        r"(ATRIAL\s*FIB|A[\s\-]?FIB|AFIB|\bAF\b|ATRIAL\s*FLUTTER|"
        r"A[\s\-]?FLUTTER|AFLUTTER|FIB\/FLUT|FIB-?FLUT)"
    )

    RYTHM_ITEMIDS =  [220048]


class AtrialFibrillationExtractor(BaseExtractor):
    def __init__(self, args, config: AtrialFibrillationConfig = AtrialFibrillationConfig()):
        super().__init__(args)
        self.config = config
        self.af_value_re = re.compile(self.config.AF_VALUE_PATTERN, re.IGNORECASE)

    def prepare_stays(self) -> pd.DataFrame:
        stays = load_icu_stays(self.data_path)
        stays = stays.merge(self.patients, on="subject_id", how="left")
        stays = stays.merge(self.adms[["hadm_id", "race", "deathtime"]], on="hadm_id", how="left")

        stays["is_age_eligible"] = (stays["age"] > self.config.age_min) & (stays["age"] <= self.config.age_max)
        stays = stays.sort_values(["subject_id", "intime"])
        stays["is_first_icustay"] = stays.groupby("subject_id")["intime"].transform("min") == stays["intime"]
        stays["has_min_icu_los"] = stays["los"] > self.config.min_los_days
        return stays

    def add_diagnosis_labels(self, stays: pd.DataFrame) -> pd.DataFrame:
        diags = load_diagnoses(self.data_path)
        diags["icd_code"] = diags["icd_code"].astype(str).str.replace(".", "", regex=False).str.upper()

        af_mask = diags["icd_code"].isin(self.config.AF_ICD9_EXACT) | diags["icd_code"].str.startswith(self.config.AF_ALL_PREFIXES)
        af_history_subjects = set(diags.loc[af_mask, "subject_id"].dropna().astype(int).unique().tolist())
        stays["has_diagnosis"] = stays["subject_id"].isin(af_history_subjects)
        
        return stays

    def add_cardiac_surgery_labels(self, stays: pd.DataFrame) -> pd.DataFrame:
        
        proc = load_procedures(self.data_path)
        proc["icd_code"] = proc["icd_code"].astype(str).str.replace(".", "", regex=False).str.upper()
        surg_mask = proc["icd_code"].str.startswith(self.config.CARDIAC_SURG_ALL_PREFIXES)
        stays["has_cardiac_surgery"] = stays["hadm_id"].isin(proc.loc[surg_mask, "hadm_id"])
        return stays


    def add_noaf_label(self, stays: pd.DataFrame) -> pd.DataFrame:
        stay_window = {int(r.stay_id): (r.intime, r.outtime) for r in stays[["stay_id", "intime", "outtime"]].itertuples(index=False)}

        af_first24h = set()
        af_after24h = set()

        chart = load_icu_chartevents_for_itemid(self.data_path,list(self.config.RYTHM_ITEMIDS),chunksize=100_000)
        chart = chart[chart["stay_id"].isin(stays["stay_id"])].copy()
        chart["value"] = chart["value"].astype(str)

        for row in chart.itertuples(index=False):
            if pd.isna(row.stay_id) or pd.isna(row.charttime):
                continue

            sid = int(row.stay_id)
            intime, outtime = stay_window.get(sid, (pd.NaT, pd.NaT))
            if pd.isna(intime) or pd.isna(outtime) or row.charttime < intime or row.charttime > outtime:
                continue
            if not self.af_value_re.search(row.value):
                continue

            hours = (row.charttime - intime).total_seconds() / 3600.0
            if 0.0 <= hours <= self.config.observation_window_hours:
                af_first24h.add(sid)
            elif hours > self.config.observation_window_hours:
                af_after24h.add(sid)

        cohort = stays[~stays["stay_id"].isin(af_first24h)].copy()
        cohort["noaf"] = cohort["stay_id"].isin(af_after24h).astype(int)

        logger.info("Stays with AF in first 24h excluded: %s", len(stays) - len(cohort))
        logger.info("NOAF positive stays (AF after 24h): %s", int(cohort["noaf"].sum()))
        return cohort

    def extract_cohort(self):
        """Run NOAF cohort extraction with paper-aligned criteria."""
        stays = self.prepare_stays()
        stays = self.add_diagnosis_labels(stays)
        stays = self.add_cardiac_surgery_labels(stays)
        cohort = stays[
            stays["is_first_icustay"]
            & stays["is_age_eligible"]
            & stays["has_min_icu_los"]
            & ~stays["has_diagnosis"]
            & ~stays["has_cardiac_surgery"]
        ].copy()
        cohort = self.add_noaf_label(cohort)


        cohort["label"] = cohort["noaf"]
        logger.info("After first-ICU + age + LOS filters: %s", len(cohort))
        save_cohort(cohort, self.paths, "atrial_fibrillation_mortality")



