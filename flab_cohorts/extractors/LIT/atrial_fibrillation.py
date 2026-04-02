"""
This class extracts a NOAF (new-onset atrial fibrillation) prediction cohort from MIMIC-IV.
Reference: https://link.springer.com/article/10.1186/s13054-024-05138-0
"""

import re
import pandas as pd
from dataclasses import dataclass

from flab_cohorts.extractors.base import ICUBaseExtractor
from flab_cohorts.utils.dataset_loader import load_icu_chartevents_for_itemid, load_procedures
from flab_cohorts.utils.logger import get_logger

logger = get_logger("ATRIAL_FIBRILLATION_NOAF")


@dataclass
class AtrialFibrillationConfig:
    age_min: float = 18.0
    age_max: float = 120.0
    min_los_days: float = 2.0
    observation_window_hours: float = 24.0

    AF_ICD9_CODES: tuple[str, ...] = ("42731", "42732")
    AF_ICD10_CODES: tuple[str, ...] = ("I48",)
    ALL_AF_CODES: tuple[str, ...] = AF_ICD9_CODES + AF_ICD10_CODES

    CABG_ICD9_PREFIXES: tuple[str, ...] = ("361",)
    VALVE_ICD9_PREFIXES: tuple[str, ...] = ("35",)
    CABG_ICD10_PCS_PREFIXES: tuple[str, ...] = ("0210",)
    VALVE_ICD10_PCS_PREFIXES: tuple[str, ...] = ("02Q", "02R")
    CARDIAC_SURG_ALL_PREFIXES: tuple[str, ...] = ("361", "35", "0210", "02Q", "02R")

    AF_VALUE_PATTERN: str = (
        r"(ATRIAL\s*FIB|A[\s\-]?FIB|AFIB|\bAF\b|ATRIAL\s*FLUTTER|"
        r"A[\s\-]?FLUTTER|AFLUTTER|FIB\/FLUT|FIB-?FLUT)"
    )
    RYTHM_ITEMIDS = [220048]

class AtrialFibrillationExtractor(ICUBaseExtractor):
    COHORT_NAME = "af"
    def __init__(self, args, config: AtrialFibrillationConfig = AtrialFibrillationConfig()):
        super().__init__(args)
        self.config = config
        self.af_value_re = re.compile(self.config.AF_VALUE_PATTERN, re.IGNORECASE)

    def add_cardiac_surgery_labels(self, stays: pd.DataFrame) -> pd.DataFrame:
        proc = load_procedures(self.data_path)
        proc["icd_code"] = proc["icd_code"].astype(str).str.replace(".", "", regex=False).str.upper()
        
        surg_mask = proc["icd_code"].str.startswith(self.config.CARDIAC_SURG_ALL_PREFIXES)
        stays["has_cardiac_surgery"] = stays["hadm_id"].isin(proc.loc[surg_mask, "hadm_id"])
        
        return stays

    def add_noaf_label(self, stays: pd.DataFrame) -> pd.DataFrame:
        stay_window = {
            int(r.stay_id): (r.intime, r.outtime)
            for r in stays[["stay_id", "intime", "outtime"]].itertuples(index=False)
        }
        af_first24h = set()
        af_after24h = set()

        chart = load_icu_chartevents_for_itemid(self.data_path, list(self.config.RYTHM_ITEMIDS), chunksize=100_000)
        chart = chart[chart["stay_id"].isin(stays["stay_id"])].copy()
        chart["value"] = chart["value"].astype(str)

        for row in chart.itertuples(index=False):
            if pd.isna(row.stay_id) or pd.isna(row.charttime):
                continue
            sid = int(row.stay_id)
            intime, outtime = stay_window.get(sid, (pd.NaT, pd.NaT))
            if (
                pd.isna(intime) or pd.isna(outtime)
                or row.charttime < intime or row.charttime > outtime
            ):
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

        return cohort

    def extract_cohort(self):
        
        
        stays = self.initialize_icu_stays()
        stays = self.add_diagnosis_flags(stays, self.config.ALL_AF_CODES, level="subject")
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

        self.save_cohort(cohort)
