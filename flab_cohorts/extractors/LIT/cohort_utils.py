from pathlib import Path
import pandas as pd

from flab_cohorts.utils.dataset_loader import *



def extract_diag_pts(data_path: Path, icd_code: str) -> pd.DataFrame:
    
    diag = load_diagnoses(data_path)
    diag = diag.loc[diag["icd_code"].str.startswith(icd_code, na=False)]
    return diag

def extract_procedure_pts(data_path: Path, icd_codes: list) -> pd.DataFrame:
    if isinstance(icd_codes, str): icd_codes = [icd_codes]
    pattern = "|".join(icd_codes)
    proc = load_procedures(data_path)
    return proc.loc[proc["icd_code"].str.contains(pattern, na=False)][["subject_id", "hadm_id"]].drop_duplicates()


def find_itemid_by_label(data_path: Path, label: str) -> list:
    lab_item_labels = load_d_labitems_labels(data_path)
    lab_item_labels["label"] = lab_item_labels["label"].str.lower()
    ids = lab_item_labels[lab_item_labels["label"] == label]['itemid']
    return ids.tolist()

def extract_chemo_cohort(df: pd.DataFrame, data_path: Path):

    proc_codes = load_chemo_procedure_codes(data_path)
    chemo_pts= extract_procedure_pts(data_path,proc_codes) # chemo based on procedure codes
    df['chemo'] = df['hadm_id'].isin(chemo_pts['hadm_id']).astype(int)

    icd_code = 'Z5111'
    chemo_pts= extract_diag_pts(data_path, icd_code) #chemo based on diagnosis codes
    df['chemo'] = df['chemo'] |df['hadm_id'].isin(chemo_pts['hadm_id']).astype(int)


    chemo_df = df[df["subject_id"].isin(set(df[df["chemo"] == 1]["subject_id"]))]
    return chemo_df

