from pathlib import Path
import pandas as pd
from flab_cohorts.config.constants import PROJECT_ROOT
from tqdm import tqdm


def load_admissions(data_path: Path) -> pd.DataFrame:
    p = data_path / "hosp" / "admissions.csv.gz"
    if not p.exists():
        raise FileNotFoundError(f"No admissions file under {data_path}")
    
    admissions= pd.read_csv(p, compression='gzip', header=0, index_col=None, usecols=["subject_id", "hadm_id", "admittime", "dischtime", "deathtime","hospital_expire_flag", "insurance", "race"], parse_dates=["admittime", "dischtime","deathtime"])
    admissions['los'] = (admissions["dischtime"] - admissions["admittime"]).dt.total_seconds() / 86400
    
    return admissions

def load_diagnoses(data_path: Path) -> pd.DataFrame:
    p = data_path / "hosp" / "diagnoses_icd.csv.gz"
    if not p.exists():
        raise FileNotFoundError(f"No diagnoses file under {data_path}")
    diags = pd.read_csv(p, compression='gzip', header=0, index_col=None)#, usecols=["subject_id", "hadm_id", "icd_code", "icd_version"])
    
    return diags

def load_patients(data_path: Path) -> pd.DataFrame:
    p = data_path / "hosp" / "patients.csv.gz"
    if not p.exists():
        raise FileNotFoundError(f"No patients file under {data_path}")
    patients = pd.read_csv(p, compression='gzip', header=0, index_col=None, usecols=["subject_id","anchor_age", "gender", "dod"], parse_dates=["dod"])
    patients = patients.rename(columns={"anchor_age": "age"})
    patients = patients.loc[patients['age'] >= 18]
    
    return patients


def load_procedures(data_path: Path) -> pd.DataFrame:
    p = data_path / "hosp" / "procedures_icd.csv.gz"
    if not p.exists():
        raise FileNotFoundError(f"No procedures file under {data_path}")
    df = pd.read_csv(p, compression='gzip', header=0, index_col=None)
    return df



def load_chemo_procedure_codes(data_path: Path, limit: int = 47) -> list:
    p = data_path / "hosp" / "chemo_procedures.csv"
    if not p.exists():
        raise FileNotFoundError(f"chemo_procedures.csv not found under {data_path}/hosp/")
    df = pd.read_csv(p, header=0, delimiter=';')
    return df["icd_code"].tolist()[:limit]


def load_d_icd_procedures(data_path: Path) -> pd.DataFrame:
    p = data_path / "hosp" / "d_icd_procedures.csv.gz"
    if not p.exists():
        raise FileNotFoundError(f"d_icd_procedures.csv.gz not found under {data_path}/hosp/")
    return pd.read_csv(p, compression="gzip", header=0)



def load_d_labitems_labels(data_path: Path) -> pd.DataFrame:
    p = data_path / "hosp" / "d_labitems.csv.gz"
    if not p.exists():
        raise FileNotFoundError(f"d_labitems.csv.gz not found under {data_path}/hosp/")
    return pd.read_csv(p, compression="gzip", header=0)


def find_itemid_by_label(data_path: Path, label: str) -> list:
    lab_items = load_d_labitems_labels(data_path)
    lab_items["label"] = lab_items["label"].str.lower()
    return lab_items[lab_items["label"] == label]["itemid"].tolist()


def load_labevents_for_cohort(data_path: Path, cohort: pd.DataFrame, usecols=None, chunksize=10000000):

    usecols = ["subject_id", "hadm_id", "itemid", "charttime", "valuenum"]
    dtypes = {"itemid": "int64", "subject_id": "int64"}
    
    p = data_path / "hosp" / "labevents.csv.gz"
    if not p.exists():
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


def load_icu_stays(data_path: Path) -> pd.DataFrame:
    p = data_path / "icu" / "icustays.csv.gz"
    if not p.exists():
        raise FileNotFoundError(f"icustays.csv.gz not found under {data_path}/icu/")
    df = pd.read_csv(p, compression="gzip", header=0, usecols=["subject_id", "hadm_id", "stay_id","intime", "outtime","los","first_careunit", "last_careunit"], parse_dates=["intime", "outtime"])
    return df

def load_icu_procedures(data_path: Path) -> pd.DataFrame:
    p = data_path / "icu" / "procedureevents.csv.gz"
    if not p.exists():
        raise FileNotFoundError(f"procedureevents.csv.gz not found under {data_path}/icu/")
    df = pd.read_csv(p, compression="gzip", header=0, usecols=['subject_id', 'hadm_id', 'stay_id', 'starttime', 'endtime', 'itemid', 'value', 'valueuom'],
                         parse_dates=['starttime', 'endtime'])
    return df

def load_icu_items(data_path: Path) -> pd.DataFrame:
    p = data_path / "icu" / "d_items.csv.gz"
    if not p.exists():
        raise FileNotFoundError(f"d_items.csv.gz not found under {data_path}/icu/")
    df = pd.read_csv(p, compression="gzip", header=0)
    return df


def load_labevents_for_itemid(data_path: Path, itemid: int, usecols=None, chunksize=1_000_000):

    usecols = ["subject_id", "hadm_id", "itemid", "charttime", "valuenum"]
    p = data_path / "hosp" / "labevents.csv.gz"

    chunks_list = []
    for chunk in tqdm(pd.read_csv(p, compression="gzip", usecols=usecols, chunksize=chunksize, parse_dates=["charttime"])):

        chunk = chunk[chunk["itemid"] == itemid]
        chunks_list.append(chunk)

    return pd.concat(chunks_list, ignore_index=True)


def load_icu_chartevents_for_itemid(data_path: Path, itemids: list, usecols=None, chunksize=1_000_000) -> pd.DataFrame:
    usecols=["subject_id", "hadm_id", "stay_id", "itemid", "charttime", "valuenum","value"]
    p = data_path / "icu" / "chartevents.csv.gz"
    chunks = []
    for chunk in tqdm(pd.read_csv(p, compression="gzip", usecols=usecols, chunksize=chunksize, parse_dates=["charttime"])):

        chunk = chunk[chunk["itemid"].isin(itemids)]
        if len(chunk) > 0:
            chunks.append(chunk)
            
    return pd.concat(chunks, ignore_index=True)

def load_icu_inputevents(data_path: Path) -> pd.DataFrame:
    p = data_path / "icu" / "inputevents.csv.gz"
    if not p.exists():
        raise FileNotFoundError(f"inputevents.csv.gz not found under {data_path}/icu/")
    df = pd.read_csv(p, compression="gzip", header=0, parse_dates=["starttime"])
    return df