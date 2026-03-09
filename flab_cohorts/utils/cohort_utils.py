import os
from pathlib import Path
from datetime import timedelta

import pandas as pd
from tqdm import tqdm

from flab_cohorts.config.constants import MIMIC_IV_PATH

DATASET_CONFIG = {
    "MIMIC_IV_HOSP": {"subfolder": "hosp", "case": "lower"},
    "MIMIC_IV_ICU": {"subfolder": "icu", "case": "lower"},
    "MIMIC_III": {"subfolder": "", "case": "upper"},
}

_TABLE_NAMES = {
    "lower": {
        "admissions": "admissions.csv.gz",
        "diagnoses": "diagnoses_icd.csv.gz",
        "patients": "patients.csv.gz",
        "procedures": "procedures_icd.csv.gz",
    },
    "upper": {
        "admissions": "ADMISSIONS.csv.gz",
        "diagnoses": "DIAGNOSES_ICD.csv.gz",
        "patients": "PATIENTS.csv.gz",
        "procedures": "PROCEDURES_ICD.csv.gz",
    },
}

_ADM_COLS_LOWER = ["subject_id", "hadm_id", "admittime", "dischtime", "deathtime", "hospital_expire_flag", "insurance", "race"]
_ADM_COLS_UPPER = [c.upper() for c in _ADM_COLS_LOWER]
_PT_COLS_LOWER = ["subject_id", "anchor_age", "gender", "dod"]
_PT_COLS_UPPER = [c.upper() for c in _PT_COLS_LOWER]


def _data_path(data_path: Path, dataset: str, table_key: str) -> str:
    cfg = DATASET_CONFIG.get(dataset, DATASET_CONFIG["MIMIC_IV_HOSP"])
    case = cfg["case"]
    subfolder = cfg.get("subfolder", "")
    fname = _TABLE_NAMES[case][table_key]
    if subfolder:
        return os.path.join(data_path, subfolder, fname)
    return os.path.join(data_path, fname)


def _normalize_columns(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    if DATASET_CONFIG.get(dataset, {}).get("case") == "upper":
        df = df.rename(columns=str.lower)
    return df


def load_admissions(data_path: Path, dataset: str = "MIMIC_IV_HOSP") -> pd.DataFrame:
    p = _data_path(data_path, dataset, "admissions")
    if not os.path.exists(p):
        raise FileNotFoundError(f"No admissions file at {p}")
    cols = _ADM_COLS_UPPER if DATASET_CONFIG.get(dataset, {}).get("case") == "upper" else _ADM_COLS_LOWER
    parse_dates = ["admittime", "dischtime", "deathtime"] if dataset != "MIMIC_III" else ["ADMITTIME", "DISCHTIME", "DEATHTIME"]
    admissions = pd.read_csv(p, compression="gzip", header=0, index_col=None, usecols=cols, parse_dates=parse_dates)
    admissions = _normalize_columns(admissions, dataset)
    admissions["los"] = (admissions["dischtime"] - admissions["admittime"]).dt.total_seconds() / 86400
    return admissions


def load_diagnoses(data_path: Path, dataset: str = "MIMIC_IV_HOSP") -> pd.DataFrame:
    p = _data_path(data_path, dataset, "diagnoses")
    if not os.path.exists(p):
        raise FileNotFoundError(f"No diagnoses file at {p}")
    diags = pd.read_csv(p, compression="gzip", header=0, index_col=None)
    return _normalize_columns(diags, dataset)


def load_patients(data_path: Path, dataset: str = "MIMIC_IV_HOSP") -> pd.DataFrame:
    p = _data_path(data_path, dataset, "patients")
    if not os.path.exists(p):
        raise FileNotFoundError(f"No patients file at {p}")
    cols = _PT_COLS_UPPER if DATASET_CONFIG.get(dataset, {}).get("case") == "upper" else _PT_COLS_LOWER
    patients = pd.read_csv(p, compression="gzip", header=0, index_col=None, usecols=cols)
    patients = _normalize_columns(patients, dataset)
    if "anchor_age" in patients.columns:
        patients = patients.rename(columns={"anchor_age": "age"})
    patients = patients.loc[patients["age"] >= 18]
    return patients


def load_procedures(data_path: Path, dataset: str = "MIMIC_IV_HOSP") -> pd.DataFrame:
    p = _data_path(data_path, dataset, "procedures")
    if not os.path.exists(p):
        raise FileNotFoundError(f"No procedures file at {p}")
    proc = pd.read_csv(p, compression="gzip", header=0, index_col=None)
    return _normalize_columns(proc, dataset)


def load_chemo_procedure_codes(data_path: Path, dataset: str = "MIMIC_IV_HOSP", limit: int = 47) -> list:
    cfg = DATASET_CONFIG.get(dataset, DATASET_CONFIG["MIMIC_IV_HOSP"])
    subfolder = cfg.get("subfolder", "hosp")
    p = os.path.join(data_path, subfolder, "chemo_procedures.csv") if subfolder else os.path.join(data_path, "chemo_procedures.csv")
    if not os.path.exists(p):
        raise FileNotFoundError(f"chemo_procedures.csv not found at {p}")
    df = pd.read_csv(p, header=0, delimiter=";")
    col = "icd_code" if "icd_code" in df.columns else "ICD_CODE"
    if col not in df.columns:
        col = df.columns[0]
    return df[col].tolist()[:limit]


def extract_diag_pts(module_path: Path, icd_code: str, dataset: str = "MIMIC_IV_HOSP") -> pd.DataFrame:
    diag = load_diagnoses(module_path, dataset=dataset)
    return diag.loc[diag["icd_code"].str.startswith(icd_code, na=False)][["subject_id", "hadm_id"]].drop_duplicates()


def extract_procedure_pts(module_path: Path, icd_codes: list, dataset: str = "MIMIC_IV_HOSP") -> pd.DataFrame:
    if isinstance(icd_codes, str):
        icd_codes = [icd_codes]
    pattern = "|".join(icd_codes)
    proc = load_procedures(module_path, dataset=dataset)
    return proc.loc[proc["icd_code"].str.contains(pattern, na=False)][["subject_id", "hadm_id"]].drop_duplicates()


def extract_disease_cohort(
    mimic_path: Path,
    dataset: str = "MIMIC_IV_HOSP",
    subject_col: str = "subject_id",
    visit_col: str = "hadm_id",
    admit_col: str = "admittime",
    disch_col: str = "dischtime",
    disease_label: str = "",
) -> pd.DataFrame:
    visit = load_admissions(mimic_path, dataset=dataset)
    visit = visit.rename(columns={"admittime": admit_col, "dischtime": disch_col})
    visit[admit_col] = pd.to_datetime(visit[admit_col])
    visit[disch_col] = pd.to_datetime(visit[disch_col])
    visit["los"] = (visit[disch_col] - visit[admit_col]).dt.total_seconds() / 86400

    if disease_label:
        disease_pts = extract_diag_pts(mimic_path, disease_label, dataset=dataset)
        visit = visit[visit[subject_col].isin(disease_pts[subject_col])]

    cfg = DATASET_CONFIG.get(dataset, DATASET_CONFIG["MIMIC_IV_HOSP"])
    subfolder = cfg.get("subfolder", "hosp")
    pts_path = os.path.join(mimic_path, subfolder, "patients.csv.gz") if subfolder else os.path.join(mimic_path, "PATIENTS.csv.gz")
    pt_cols = [subject_col, "anchor_year", "anchor_age", "anchor_year_group", "dod", "gender"]
    if DATASET_CONFIG.get(dataset, {}).get("case") == "upper":
        pt_cols = [c.upper() for c in pt_cols]
    pts = pd.read_csv(pts_path, compression="gzip", header=0, usecols=pt_cols)
    pts = _normalize_columns(pts, dataset)
    pts["dod"] = pd.to_datetime(pts["dod"])
    pts["yob"] = pts["anchor_year"] - pts["anchor_age"]
    pts["min_valid_year"] = pts["anchor_year"] + (2019 - pts["anchor_year_group"].str.slice(start=-4).astype(int))
    pts["age"] = pts["anchor_age"]

    visit_pts = visit[[subject_col, visit_col, admit_col, disch_col, "los", "hospital_expire_flag"]].merge(
        pts[[subject_col, "anchor_year", "anchor_age", "yob", "min_valid_year", "dod", "gender", "age"]],
        on=subject_col,
        how="inner",
    )
    visit_pts = visit_pts.loc[visit_pts["age"] >= 18]
    eth = visit[["hadm_id", "insurance", "race"]].drop_duplicates()
    visit_pts = visit_pts.merge(eth, on="hadm_id", how="inner")
    return visit_pts.dropna(subset=["min_valid_year"])[
        [subject_col, visit_col, admit_col, disch_col, "los", "dod", "hospital_expire_flag", "age", "gender", "race", "insurance"]
    ]


def extract_chemo_cohort(df: pd.DataFrame, mimic_path: Path, dataset: str = "MIMIC_IV_HOSP") -> pd.DataFrame:
    icd_codes = load_chemo_procedure_codes(mimic_path, dataset=dataset)
    chemo_ids = extract_procedure_pts(mimic_path, icd_codes, dataset=dataset)
    df = df.copy()
    df["chemo"] = df["hadm_id"].isin(chemo_ids["hadm_id"]).astype(int)
    z5111_ids = extract_diag_pts(mimic_path, "Z5111", dataset=dataset)
    df["chemo"] = (df["chemo"] | df["hadm_id"].isin(z5111_ids["hadm_id"])).astype(int)
    return df[df["subject_id"].isin(df[df["chemo"] == 1]["subject_id"])].copy()


def current_NF_occurance(df: pd.DataFrame, mimic_path: Path, dataset: str = "MIMIC_IV_HOSP") -> pd.DataFrame:
    df = df.copy()
    fever_ids = extract_diag_pts(mimic_path, "R50", dataset=dataset)
    df["fever"] = df["hadm_id"].isin(fever_ids["hadm_id"]).astype(int)
    neutropenia_ids = extract_diag_pts(mimic_path, "D70", dataset=dataset)
    df["neutropenia"] = df["hadm_id"].isin(neutropenia_ids["hadm_id"]).astype(int)
    df["NF"] = ((df["fever"] == 1) & (df["neutropenia"] == 1)).astype(int)
    return df


def _split_nf_row(x, days: int, target_cohort: pd.DataFrame, splitting_approach: str):
    if not (x["chemo"] == 1 and x["NF"] == 0 and x["hospital_expire_flag"] == 0):
        return 0
    sub = target_cohort[
        (target_cohort["subject_id"] == x["subject_id"])
        & (target_cohort["admittime"] > x["dischtime"])
        & (target_cohort["admittime"] <= x["dischtime"] + timedelta(days=days))
    ].sort_values("admittime")

    if sub.empty and pd.notna(x["dod"]) and x["dod"] <= x["dischtime"] + timedelta(days=days):
        return 0
    if not sub.empty and (sub["chemo"] == 1).any():
        idx_pos = (sub["chemo"] == 1).values.argmax()
        readmissions_after_next_chemo = sub.iloc[idx_pos:]
        sub = sub.iloc[:idx_pos]
        if (sub["NF"] == 0).all() or sub.empty:
            if (readmissions_after_next_chemo["NF"] == 0).all():
                return 1
            return 0

    if splitting_approach == "only readmissions":
        if not sub.empty and (sub["NF"] == 0).all():
            return 1
        if not sub.empty and (sub["NF"] == 1).any():
            return 2
    if splitting_approach == "both readmissions and no admission":
        if sub.empty or (sub["NF"] == 0).all():
            return 1
        if not sub.empty and (sub["NF"] == 1).any():
            return 2
    return 0


def split_neutropenic_fever_cases(
    target_cohort: pd.DataFrame,
    days: int = 30,
    splitting_approach: str = "both readmissions and no admission",
) -> pd.DataFrame:
    tqdm.pandas()
    target_cohort = target_cohort.copy()
    target_cohort["NF_in_30_days"] = target_cohort.progress_apply(
        lambda x: _split_nf_row(x, days, target_cohort, splitting_approach),
        axis=1,
    )
    return target_cohort
