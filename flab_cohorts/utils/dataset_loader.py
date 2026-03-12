from pathlib import Path
import pandas as pd
import os
from flab_cohorts.config.constants import PROJECT_ROOT
from tqdm import tqdm


def load_admissions(data_path: Path) -> pd.DataFrame:
    p = os.path.join(data_path, "hosp/admissions.csv.gz")
    if not os.path.exists(p):
        raise FileNotFoundError(f"No admissions file under {data_path}")
    
    admissions= pd.read_csv(p, compression='gzip', header=0, index_col=None, usecols=["subject_id", "hadm_id", "admittime", "dischtime", "deathtime","hospital_expire_flag", "insurance", "race"], parse_dates=["admittime", "dischtime","deathtime"])
    admissions['los'] = (admissions["dischtime"] - admissions["admittime"]).dt.total_seconds() / 86400
    
    return admissions

def load_diagnoses(data_path: Path) -> pd.DataFrame:
    p = os.path.join(data_path, "hosp/diagnoses_icd.csv.gz")
    if not os.path.exists(p):
        raise FileNotFoundError(f"No diagnoses file under {data_path}")
    diags = pd.read_csv(p, compression='gzip', header=0, index_col=None)#, usecols=["subject_id", "hadm_id", "icd_code", "icd_version"])
    
    return diags

def load_patients(data_path: Path) -> pd.DataFrame:
    p = os.path.join(data_path, "hosp/patients.csv.gz")
    if not os.path.exists(p):
        raise FileNotFoundError(f"No patients file under {data_path}")
    patients = pd.read_csv(p, compression='gzip', header=0, index_col=None, usecols=["subject_id","anchor_age", "gender", "dod"])
    patients = patients.rename(columns={"anchor_age": "age"})
    patients = patients.loc[patients['age'] >= 18]
    
    return patients


def load_procedures(data_path: Path) -> pd.DataFrame:
    p = os.path.join(data_path, "hosp/procedures_icd.csv.gz")
    if not os.path.exists(p):
        raise FileNotFoundError(f"No procedures file under {data_path}")
    return pd.read_csv(p, compression='gzip', header=0, index_col=None)



def load_chemo_procedure_codes(data_path: Path, limit: int = 47) -> list:
    p = os.path.join(data_path, "hosp/chemo_procedures.csv")
    if not os.path.exists(p):
        raise FileNotFoundError(f"chemo_procedures.csv not found under {data_path}/hosp/")
    df = pd.read_csv(p, header=0, delimiter=';')
    return df["icd_code"].tolist()[:limit]


def load_d_icd_procedures(data_path: Path) -> pd.DataFrame:
    p = os.path.join(data_path, "hosp/d_icd_procedures.csv.gz")
    if not os.path.exists(p):
        raise FileNotFoundError(f"d_icd_procedures.csv.gz not found under {data_path}/hosp/")
    return pd.read_csv(p, compression="gzip", header=0)



def load_d_labitems_labels(data_path: Path) -> list:
    p = os.path.join(data_path, "hosp/d_labitems.csv.gz")
    if not os.path.exists(p):
        raise FileNotFoundError(f"d_labitems.csv.gz not found under {data_path}/hosp/")
    return pd.read_csv(p, compression="gzip", header=0)


def load_labevents_for_cohort(data_path: Path, cohort: pd.DataFrame, usecols=None, chunksize=10000000):

    usecols = ["subject_id", "hadm_id", "itemid", "charttime", "valuenum"]
    dtypes = {"itemid": "int64", "subject_id": "int64"}
    
    p = os.path.join(data_path, "hosp/labevents.csv.gz")
    if not os.path.exists(p):
        raise FileNotFoundError(f"labevents.csv.gz not found under {data_path}/hosp/")

    chunks_list = []
    for chunk in tqdm(pd.read_csv(p, compression="gzip", usecols=usecols, dtype=dtypes, chunksize=chunksize)):
        chunk = chunk.dropna(subset=["valuenum"])
        chunk = chunk[chunk["subject_id"].isin(cohort["subject_id"])]
        if chunk.empty:
            continue
        chunk["charttime"] = pd.to_datetime(chunk["charttime"])
        chunk["hadm_id"] = chunk["hadm_id"].fillna(0)
        chunk = chunk.dropna()
        chunks_list.append(chunk)
    if not chunks_list:
        return pd.DataFrame(columns=usecols)
    return pd.concat(chunks_list, ignore_index=True)

