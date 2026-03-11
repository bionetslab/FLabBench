import os
from pathlib import Path
from datetime import timedelta

import pandas as pd
from tqdm import tqdm

from flab_cohorts.utils.io import load_admissions, load_diagnoses, load_procedures, load_patients




def extract_diag_pts(data_path: Path,  icd_code: str) -> pd.DataFrame:
    diag = load_diagnoses(data_path)
    diag = diag.loc[diag["icd_code"].str.startswith(icd_code, na=False)]
    print('#disease code', icd_code, 'admissions', diag['hadm_id'].nunique())
    print('#disease code', icd_code, 'patients', diag['subject_id'].nunique())
    return diag

def extract_procedure_pts(data_path: Path, icd_codes: list) -> pd.DataFrame:
    if isinstance(icd_codes, str): icd_codes = [icd_codes]
    pattern = "|".join(icd_codes)
    proc = load_procedures(data_path)
    return proc.loc[proc["icd_code"].str.contains(pattern, na=False)][["subject_id", "hadm_id"]].drop_duplicates()



def load_chemo_procedure_codes(data_path: Path) -> list:
    p = os.path.join(data_path, "hosp/chemo_procedures.csv")
    if not os.path.exists(p):
        raise FileNotFoundError(f"chemo_procedures.csv not found under {data_path}/hosp/")
    df = pd.read_csv(p, header=0, delimiter=';')
    return df["icd_code"].tolist()[:47]



def extract_chemo_cohort(df:pd.DataFrame, data_path:Path):

    proc_codes = load_chemo_procedure_codes(data_path)
    chemo_pts= extract_procedure_pts(data_path,proc_codes) # chemo based on procedure codes
    df['chemo'] = df['hadm_id'].isin(chemo_pts['hadm_id']).astype(int)

    icd_code = 'Z5111'
    chemo_pts= extract_diag_pts(data_path, icd_code) #chemo based on diagnosis codes
    df['chemo'] = df['chemo'] |df['hadm_id'].isin(chemo_pts['hadm_id']).astype(int)


    chemo_df  = df[df['subject_id'].isin(set(df[df['chemo'] == 1]['subject_id']))] 
    print('#cancer chemo all admissions',chemo_df['hadm_id'].nunique())
    print('#chemo patients', chemo_df[chemo_df.chemo == 1]['subject_id'].nunique())
    print('#chemo admissions',chemo_df[chemo_df.chemo == 1]['hadm_id'].nunique())
    
    return chemo_df








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
