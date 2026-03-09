from pathlib import Path
import pandas as pd
import os
from flab_cohorts.config.constants import PROJECT_ROOT



def load_admissions(data_path: Path) -> pd.DataFrame:
    p = os.path.join(data_path, "hosp/admissions.csv.gz")
    if p is None:
        raise FileNotFoundError(f"No admissions file under {data_path}")
    
    admissions= pd.read_csv(p, compression='gzip', header=0, index_col=None, usecols=["subject_id", "hadm_id", "admittime", "dischtime", "deathtime","hospital_expire_flag", "insurance", "race"], parse_dates=["admittime", "dischtime","deathtime"])
    admissions['los'] = (admissions["dischtime"] - admissions["admittime"]).dt.total_seconds() / 86400
    
    return admissions

def load_diagnoses(data_path: Path) -> pd.DataFrame:
    p = os.path.join(data_path, "hosp/diagnoses_icd.csv.gz")
    if p is None:
        raise FileNotFoundError(f"No diagnoses file under {data_path}")
    diags = pd.read_csv(p, compression='gzip', header=0, index_col=None)#, usecols=["subject_id", "hadm_id", "icd_code", "icd_version"])
    
    return diags

def load_patients(data_path: Path) -> pd.DataFrame:
    p = os.path.join(data_path, "hosp/patients.csv.gz")
    if p is None:
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




def set_all_paths(args, out=True):
    dataset = args.dataset
    cohort = getattr(args, "cohort", "")
    extractor = args.extractor


    saved_data_path = Path(PROJECT_ROOT) / "data" / dataset

    feature_path = saved_data_path / "top_features" # / cohort

    #CV_folds_path = saved_data_path / extractor / "folds" / cohort
    
    cohort_path = saved_data_path / "cohorts"/ extractor
    

    # save this as dict
    path_dict = {
        "saved_data_path": saved_data_path,
        #"CV_folds_path": CV_folds_path,
        "feature_path": feature_path,
        "cohort_path": cohort_path,
    }
    
    for p in path_dict.values():
        Path(p).mkdir(parents=True, exist_ok=True)
        

    return path_dict