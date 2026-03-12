import os
from pathlib import Path
from datetime import timedelta

import pandas as pd
from tqdm import tqdm

from flab_cohorts.utils.io import load_admissions, load_diagnoses, load_procedures, load_patients




def extract_diag_pts(data_path: Path,  icd_code: str) -> pd.DataFrame:
    diag = load_diagnoses(data_path)
    diag = diag.loc[diag["icd_code"].str.startswith(icd_code, na=False)]
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
    
    return chemo_df

def get_procedures(data_path: Path) -> pd.DataFrame:
    p = os.path.join(data_path, "hosp/d_icd_procedures.csv.gz")
    if not os.path.exists(p):
        raise FileNotFoundError(f"d_icd_procedures.csv.gz not found under {data_path}/hosp/")
    procedures_definition_df = pd.read_csv(p, compression="gzip", header=0)
    
    p = os.path.join(data_path, "hosp/procedures_icd.csv.gz")
    if not os.path.exists(p):
        raise FileNotFoundError(f"procedures_icd.csv.gz not found under {data_path}/hosp/")
    procedures_df = pd.read_csv(p, compression="gzip", header=0)

    return procedures_definition_df, procedures_df


### LABS UTILS ###

def extract_cohort_labs(data_path: Path, cohort: pd.DataFrame) -> pd.DataFrame:
    
    usecols = ['subject_id','hadm_id','itemid','charttime','valuenum']
    dtypes = {
        'itemid': 'int64',
        'subject_id': 'int64',
        }

    p = os.path.join(data_path, "hosp/labevents.csv.gz")
    if not os.path.exists(p):
        raise FileNotFoundError(f"labevents.csv.gz not found under {data_path}/hosp/")
    
    lab_df_cohort=pd.DataFrame()

    chunksize = 10000000
    for chunk in tqdm(pd.read_csv(p, compression='gzip', usecols=usecols,chunksize=chunksize)): #, nrows=10000000)):

        chunk=chunk.dropna(subset=['valuenum'])
        chunk=chunk[chunk['subject_id'].isin(cohort['subject_id'].unique())]
        chunk['charttime']=pd.to_datetime(chunk['charttime'])
        chunk['hadm_id']=chunk['hadm_id'].fillna(0)
        chunk=chunk.dropna()

        if lab_df_cohort.empty:
            lab_df_cohort=chunk
        else:
            lab_df_cohort = pd.concat([lab_df_cohort, chunk], ignore_index=True) #return all the lab results for the subjects
        
    #print("# Itemid: ", lab_df_cohort.itemid.nunique())

    return lab_df_cohort   


def find_itemid_by_label (data_path: Path,label)-> list:
    p = os.path.join(data_path, "hosp/d_labitems.csv.gz")
    lab_item_labels = pd.read_csv(p, compression="gzip", header=0)
    lab_item_labels["label"] = lab_item_labels["label"].str.lower()
    ids = lab_item_labels[lab_item_labels["label"] == label]['itemid']
    return ids.tolist()
