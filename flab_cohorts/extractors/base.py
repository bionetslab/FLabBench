from pathlib import Path
import pandas as pd

from flab_cohorts.utils.dataset_loader import load_admissions, load_diagnoses, load_icu_stays, load_patients
from flab_cohorts.utils.io import set_all_paths
from config.constants import get_data_path
from flab_cohorts.utils.logger import get_logger

logger = get_logger("EXTRACTOR")

HOSP_COHORT_COLUMNS = [
    "subject_id", "hadm_id", "admittime", "dischtime",
    "race", "los", "gender", "age", "dod", "label",
]

ICU_COHORT_COLUMNS = [
    "subject_id", "hadm_id", "stay_id", "intime", "outtime",
    "race", "los", "gender", "age", "dod", "label",
]


class BaseExtractor:
    def __init__(self, args):
        self.args = args
        self.data_path = (
            Path(args.data_path)
            if getattr(args, "data_path", None)
            else get_data_path(getattr(args, "dataset", "MIMIC_IV"))
        )
        self.paths = set_all_paths(args, out=False)
        self._load_mimic(self.data_path)

    def _load_mimic(self, path: Path):
        self.adms = load_admissions(path)
        self.diags = load_diagnoses(path)
        self.patients = load_patients(path)
        self.adms = self.adms.merge(self.patients, on="subject_id", how="left")
        self.stays = load_icu_stays(path)
        cols = ["hadm_id", "admittime", "dischtime", "deathtime","hospital_expire_flag", "race", "age", "gender", "dod"]
        self.stays = self.stays.merge(self.adms[cols], on="hadm_id", how="left")

    def add_diagnosis_flags(self, df: pd.DataFrame, icd_codes: list = None, column: str = "has_diagnosis", match: str = "startswith", level: str = "hadm") -> pd.DataFrame:
        
        if icd_codes is None:
            icd_codes = self.config.ALL_CODES
        diags = self.diags.copy()
        diags["icd_code"] = diags["icd_code"].str.replace(".", "", regex=False)
        if match == "startswith":
            mask = diags["icd_code"].str.startswith(icd_codes)
        else:
            mask = diags["icd_code"].isin(icd_codes)
            
        if level == "subject":
            matched = set(diags.loc[mask, "subject_id"].dropna().unique())
            df[column] = df["subject_id"].isin(matched)
        else:
            df[column] = df["hadm_id"].isin(diags.loc[mask, "hadm_id"])
        return df

    def save_cohort(self, cohort, cohort_name=None):
        cohort_name = cohort_name or self.COHORT_NAME

        columns = getattr(self, "COHORT_COLUMNS", None)
        if columns is not None:
            cols = [c for c in columns if c in cohort.columns]
            cohort = cohort[cols]
            
        pct = 100 * cohort["label"].mean()
        id_col = "stay_id" if "stay_id" in cohort.columns else "hadm_id"
        logger.info(
            "%s — %s: %s | patients: %s | positive: %s (%.2f%%)",
            cohort_name,
            id_col,
            cohort[id_col].nunique(),
            cohort["subject_id"].nunique(),
            cohort.loc[cohort["label"] == 1, id_col].nunique(),
            pct,
        )
        cohort.to_csv(self.paths["cohort_path"] / f"cohort_{cohort_name}.csv", index=False)


class ICUBaseExtractor(BaseExtractor):
    COHORT_COLUMNS = ICU_COHORT_COLUMNS

    def initialize_icu_stays(self) -> pd.DataFrame:

        stays = self.stays.copy()
        stays["is_age_eligible"] = ((stays["age"] >= self.config.age_min) & (stays["age"] <= self.config.age_max))
        min_los = getattr(self.config, "min_los_days", None)
        if min_los is not None:
            stays["has_min_icu_los"] = stays["los"] >= min_los
        stays = stays.sort_values(["subject_id", "intime"])
        stays["is_first_icustay"] = (stays.groupby("subject_id")["intime"].transform("min") == stays["intime"])
        return stays


    def add_inhospital_mortality(self, stays):
        stays["in_hospital_mortality"] = stays["deathtime"].notna()
        return stays

    def add_timed_mortality(self, stays, days, col="mortality"):
        """Add N-day mortality from ICU admission."""
        stays["death_time"] = stays["deathtime"].fillna(stays["dod"])
        stays["days_to_death_from_icu"] = ((stays["death_time"] - stays["intime"]).dt.total_seconds() / 86400.0)
        stays[col] = (
            stays["death_time"].notna()
            & stays["intime"].notna()
            & (stays["death_time"] >= stays["intime"])
            & (stays["death_time"]
               <= stays["intime"] + pd.Timedelta(days=days))
        ).astype(int)
        return stays


    @staticmethod
    def first_stay_per_patient(cohort):
        """Keep only the earliest ICU stay per patient."""
        cohort = cohort.sort_values(["subject_id", "intime"])
        mask = (cohort.groupby("subject_id")["intime"].transform("min") == cohort["intime"])

        return cohort[mask].copy()

