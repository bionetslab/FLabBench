"""
This class extracts the AKI + liver cirrhosis 28-day mortality cohort from the MIMIC dataset.
Reference: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0328662
"""
# ICU
# AKI (first-hospital dx) + optional LC flag
# 28-day all-cause mortality from ICU intime

from dataclasses import dataclass

import pandas as pd

from flab_cohorts.extractors.base import BaseExtractor
from flab_cohorts.extractors.LIT.cohort_utils import save_cohort
from flab_cohorts.utils.dataset_loader import load_diagnoses, load_icu_stays, load_procedures
from flab_cohorts.utils.logger import get_logger

logger = get_logger("LIVER_CIRRHOSIS_MORTALITY")


@dataclass
class LiverCirrhosisConfig:
    age_min: float = 18.0
    age_max: float = 120.0
    min_los_hours: float = 24.0
    min_hosp_los_hours: float = 24.0
    mortality_days: float = 28.0

    # AKI (classic codes only; ICD-10 N17*)
    AKI_ICD9_codes: tuple[str, ...] = ("5845", "5846", "5847", "5848", "5849")
    AKI_ICD10_codes: tuple[str, ...] = ("N17",)
    # Liver cirrhosis (K70.3*, K71.7*, K74*)
    LC_ICD9_codes: tuple[str, ...] = ("5712", "5715", "5716")
    LC_ICD10_codes: tuple[str, ...] = ("K703", "K717", "K74")
    ESRD_ICD9_codes: tuple[str, ...] = ("5856",)
    ESRD_ICD10_codes: tuple[str, ...] = ("N186",)
    SEPSIS_ICD9_codes: tuple[str, ...] = ("99591", "99592", "78552")
    SEPSIS_ICD10_codes: tuple[str, ...] = ("A40", "A41", "R652")
    RRT_ICD9_codes: tuple[str, ...] = ("3995",)
    RRT_ICD10_codes: tuple[str, ...] = ("5A1D",)


class LiverCirrhosisExtractor(BaseExtractor):
    def __init__(self, args, config: LiverCirrhosisConfig = LiverCirrhosisConfig()):
        super().__init__(args)
        self.config = config

    def _first_hospital_admissions(self) -> pd.DataFrame:
        adms = self.adms.sort_values(["subject_id", "admittime"])
        return adms.groupby("subject_id", as_index=False).head(1)[
            ["subject_id", "hadm_id", "admittime", "dischtime", "deathtime"]
        ].copy()

    def prepare_stays(self) -> pd.DataFrame:
        self._first_adm_df = self._first_hospital_admissions()
        first_hadm_by_subject = self._first_adm_df.set_index("subject_id")["hadm_id"]

        stays = load_icu_stays(self.data_path)
        stays = stays.merge(self.patients, on="subject_id", how="left")
        stays["is_age_eligible"] = (stays["age"] >= self.config.age_min) & (stays["age"] <= self.config.age_max)
        stays = stays.merge(
            self.adms[["hadm_id", "admittime", "dischtime", "deathtime", "race"]],
            on="hadm_id",
            how="left",
        )
        # Same idea as is_first_icustay: boolean flag per row (first hospitalization hadm_id from admissions).
        stays["is_first_hadm"] = stays["hadm_id"].eq(stays["subject_id"].map(first_hadm_by_subject))

        stays["icu_los_hours"] = pd.to_numeric(stays["los"], errors="coerce") * 24.0
        stays.loc[stays["icu_los_hours"].isna(), "icu_los_hours"] = (
            stays["outtime"] - stays["intime"]
        ).dt.total_seconds() / 3600.0

        return stays.sort_values(["subject_id", "intime"])

    def add_diagnosis_labels(self, stays: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        first_hadm_set = set(self._first_adm_df["hadm_id"].dropna().unique())

        diags = load_diagnoses(self.data_path)
        diags = diags[diags["hadm_id"].isin(first_hadm_set)].copy()
        diags["icd_code"] = diags["icd_code"].str.replace(".", "", regex=False)
        c = diags["icd_code"]
        v9 = diags["icd_version"] == 9
        v10 = diags["icd_version"] == 10

        aki_mask = (v10 & c.str.startswith(cfg.AKI_ICD10_codes)) | (v9 & c.isin(cfg.AKI_ICD9_codes))
        lc_mask = (v10 & c.str.startswith(cfg.LC_ICD10_codes)) | (v9 & c.isin(cfg.LC_ICD9_codes))
        esrd_mask = (v10 & c.isin(cfg.ESRD_ICD10_codes)) | (v9 & c.isin(cfg.ESRD_ICD9_codes))
        sepsis_mask = (v10 & c.str.startswith(cfg.SEPSIS_ICD10_codes)) | (v9 & c.isin(cfg.SEPSIS_ICD9_codes))

        aki_hadm = set(diags.loc[aki_mask, "hadm_id"].dropna().unique())
        lc_hadm = set(diags.loc[lc_mask, "hadm_id"].dropna().unique())
        esrd_hadm = set(diags.loc[esrd_mask, "hadm_id"].dropna().unique())
        sepsis_hadm = set(diags.loc[sepsis_mask, "hadm_id"].dropna().unique())

        stays["has_aki"] = stays["hadm_id"].isin(aki_hadm)
        stays["has_lc"] = stays["hadm_id"].isin(lc_hadm)
        stays["has_esrd"] = stays["hadm_id"].isin(esrd_hadm)
        stays["has_sepsis"] = stays["hadm_id"].isin(sepsis_hadm)
        return stays

    def exclude_rrt_procedures(self, cohort: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        proc_path = self.data_path / "hosp" / "procedures_icd.csv.gz"
        cohort_hadm_set = set(cohort["hadm_id"].dropna().unique().tolist())
        if not proc_path.exists() or not cohort_hadm_set:
            logger.info("Skipping RRT exclusion (no procedures file or empty cohort).")
            return cohort

        proc = load_procedures(self.data_path)
        proc = proc[proc["hadm_id"].isin(cohort_hadm_set)].copy()
        proc["icd_code"] = proc["icd_code"].str.replace(".", "", regex=False)
        c = proc["icd_code"]
        v9 = proc["icd_version"] == 9
        v10 = proc["icd_version"] == 10
        mask = (v9 & c.isin(cfg.RRT_ICD9_codes)) | (v10 & c.str.startswith(cfg.RRT_ICD10_codes))
        rrt_hadm = set(proc.loc[mask, "hadm_id"].dropna().unique())
        if rrt_hadm:
            cohort = cohort[~cohort["hadm_id"].isin(rrt_hadm)].copy()
        logger.info("RRT hadm_id excluded: %s", len(rrt_hadm))
        return cohort

    def add_28d_mortality_label(self, stays: pd.DataFrame) -> pd.DataFrame:
        stays["death_time"] = stays["deathtime"].fillna(stays["dod"])
        stays["days_to_death_from_icu"] = (
            (stays["death_time"] - stays["intime"]).dt.total_seconds() / 86400.0
        )
        stays["mortality_28d"] = (
            stays["death_time"].notna()
            & stays["intime"].notna()
            & (stays["death_time"] >= stays["intime"])
            & (stays["death_time"] <= stays["intime"] + pd.Timedelta(days=self.config.mortality_days))
        ).astype(int)
        return stays

    def extract_cohort(self) -> None:
        logger.info("Extracting AKI + liver cirrhosis 28-day mortality cohort")

        stays = self.prepare_stays()
        stays = self.add_diagnosis_labels(stays)

        cohort = stays[stays["has_aki"] & stays["is_first_hadm"]].copy()
        cohort = cohort.sort_values(["subject_id", "intime"])
        cohort = cohort.groupby("subject_id", as_index=False).head(1).copy()
        cohort = cohort[cohort["icu_los_hours"] >= self.config.min_los_hours].copy()

        cohort = cohort[cohort["is_age_eligible"]].copy()
        cohort["hosp_los_hours"] = (cohort["dischtime"] - cohort["admittime"]).dt.total_seconds() / 3600.0
        cohort = cohort[cohort["hosp_los_hours"] >= self.config.min_hosp_los_hours].copy()
        cohort = cohort[(~cohort["has_esrd"]) & (~cohort["has_sepsis"])].copy()

        cohort = self.exclude_rrt_procedures(cohort)
        cohort["lc_flag"] = cohort["has_lc"].astype(int)

        cohort = self.add_28d_mortality_label(cohort)
        cohort["label"] = cohort["mortality_28d"].astype(int)

        save_cohort(cohort, self.paths, "aki_lc_mortality")
        save_cohort(cohort[cohort["lc_flag"] == 1].copy(), self.paths, "aki_lc_only_mortality")
