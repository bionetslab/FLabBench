"""
This class extracts the aki cohort from the MIMIC dataset.
Reference:  https://www.sciencedirect.com/science/article/pii/S1532046420302811?
possibly compare with https://github.com/ExaScience/Aki-Predictor

"""


import pandas as pd
from tqdm import tqdm
import numpy as np
tqdm.pandas()  


from flab_cohorts.extractors.base import BaseExtractor
from flab_cohorts.utils.dataset_loader import load_icu_stays, load_icu_procedures, load_labevents_for_itemid, load_diagnoses
from flab_cohorts.utils.logger import get_logger

logger = get_logger("AKI")

class AcuteKidneyInjuryExtractor(BaseExtractor):
    def __init__(self, args):
        super().__init__(args)
        
        self.creatinine_itemid = 50912 # serum creatinine
        self.dialysis_codes = [225441, 225802, 225803, 225809, 225955, 225805]
        self.rrt_codes = [225802]
        self.eGFR_normal = 75
        self.ckd_icd_codes = "585"
        
        self.stays =  load_icu_stays(self.data_path)
        self.diags = load_diagnoses(self.data_path)
        self.procedures = load_icu_procedures(self.data_path)

    
    
    def extract_cohort(self):
        """Run AKI cohort extraction and persist the cohort."""

        self.prepare_stays()
        self.find_first_CDK_diagnosis()
        self.add_dialysis_procedure()
        self.add_serum_creatinine_features()
        
        
        self.stays['baseline_scr'] = self.stays.apply(self.compute_baseline_scr, axis=1)
        self.stays["AKI_base_50%_24h"] = self.stays["max_has_src_in_obs_window"] > self.stays['baseline_scr']*1.5 
        self.stays["AKI_base_50%"] = self.stays["max_scr_pw"] > self.stays['baseline_scr']*1.5 # THIRD TARGET CRITERION
        self.stays["AKI_24h"] = self.stays['high_scr'] 
        self.stays["AKI"] =  self.stays['dialysis_PW'] | self.stays['AKI_abs_48h'] | self.stays['AKI_base_50%']
        inclusion_mask = (
            self.stays["is_age_eligible"]
            & self.stays["is_first_icustay"]
            & self.stays["has_min_icu_los"]
            & self.stays["has_src_in_obs_window"]
            & ~self.stays["AKI_24h"]
        )
        aki_cohort = self.stays.loc[inclusion_mask].copy()
        
        self.save_cohort(aki_cohort)
        
    def prepare_stays(self) -> None:
        """Prepare stay-level features; mutates self.stays in place."""


        # add age & gender
        self.stays = self.stays.merge(self.patients, on="subject_id", how="left")
        # in predefined age window
        self.stays["is_age_eligible"] = (self.stays["age"] >= 18) & (self.stays["age"] <= 89)
        # add ethnicity
        self.stays = self.stays.merge(self.adms[["hadm_id","race"]], on="hadm_id", how="left")
        # of black ethnicity
        self.stays["is_black"] = self.stays["race"].isin(['BLACK/AFRICAN AMERICAN', 'BLACK/AFRICAN', 'BLACK/CAPE VERDEAN', 'BLACK/HAITIAN'])
        # ICU stay > 24 h
        self.stays["has_min_icu_los"] = self.stays["los"] > 1
        # label first ICU stay
        self.stays = self.stays.sort_values(["subject_id", "intime"])
        self.stays["is_first_icustay"] = self.stays.groupby("subject_id")["intime"].transform("min") == self.stays["intime"]
    
    # CDK Chronic Kidney Disease
    def find_first_CDK_diagnosis(self) -> None:
        """Add CKD history features; mutates self.stays in place."""

        diag_ckd = self.diags[self.diags["icd_code"].str.startswith(self.ckd_icd_codes)]
        diag_ckd = diag_ckd.merge(self.adms[["hadm_id","admittime","dischtime"]], on="hadm_id",how="left")

        diag_ckd_first = (diag_ckd.sort_values(["subject_id","admittime"]).groupby("subject_id").first().reset_index().rename(columns={"admittime": "cdk_admittime"}))

        self.stays = self.stays.merge(diag_ckd_first[["subject_id","icd_code","cdk_admittime"]], on="subject_id", how="left")
        self.stays["ckd"] = self.stays["icd_code"].notna() & (self.stays["intime"] >= self.stays["cdk_admittime"])
        

        
    def add_dialysis_procedure(self) -> None:
        """Add dialysis timing flags; mutates self.stays in place."""

        dial_proc = self.procedures[self.procedures["itemid"].isin(self.rrt_codes)]

        dial_proc = dial_proc.merge(self.stays[["stay_id", "intime"]], on="stay_id", how="inner")
        dial_proc["hours_since_icu_admit"] = (dial_proc["starttime"] - dial_proc["intime"]).dt.total_seconds() / 3600

        dialysis_ow = dial_proc[(dial_proc["hours_since_icu_admit"] >= 0) & (dial_proc["hours_since_icu_admit"] < 24)]
        dialysis_pw = dial_proc[(dial_proc["hours_since_icu_admit"] >= 24) & (dial_proc["hours_since_icu_admit"] <= 72)]

        self.stays["dialysis_OW"] = self.stays["stay_id"].isin(dialysis_ow["stay_id"])
        self.stays["dialysis_PW"] = self.stays["stay_id"].isin(dialysis_pw["stay_id"]) # FIRST TARGET CRITERION
        
        
        
    def add_serum_creatinine_features(self) -> None:
        """Add creatinine-derived AKI features; mutates self.stays in place."""
        
        self.scr_lab = load_labevents_for_itemid(self.data_path, self.creatinine_itemid)

        #serum creatinine in observation and prediction window
        scr_hadm = self.scr_lab.merge(self.stays[["hadm_id", "stay_id", "intime", "los"]], on="hadm_id", how="inner")
        scr_hadm["hours_since_icu_admit"] = (scr_hadm["charttime"] - scr_hadm["intime"]).dt.total_seconds() / 3600
        scr_hadm = scr_hadm.sort_values(["stay_id","charttime"])

        # serum creatinine in the first 24h of ICU admission (observation window)
        has_src_in_obs_window = scr_hadm[(scr_hadm["hours_since_icu_admit"] >= 0) & (scr_hadm["hours_since_icu_admit"] < 24)]
        scr_pw = scr_hadm[(scr_hadm["hours_since_icu_admit"] >= 24) & (scr_hadm["hours_since_icu_admit"] <= 72)]

        # first measurement of serum creatinine
        scr_first = (has_src_in_obs_window.sort_values(["stay_id","charttime"]).groupby("stay_id").first().reset_index()[["stay_id","valuenum"]].rename(columns={"valuenum":"first_scr"}))

        has_src_in_obs_window_max = has_src_in_obs_window.groupby("stay_id").valuenum.max().reset_index().rename(columns={"valuenum":"max_has_src_in_obs_window"})
        scr_pw_max = scr_pw.groupby("stay_id").valuenum.max().reset_index().rename(columns={"valuenum":"max_scr_pw"})

        self.stays = self.stays.merge(scr_pw_max, on="stay_id", how="left")
        self.stays = self.stays.merge(has_src_in_obs_window_max, on="stay_id", how="left")
        self.stays = self.stays.merge(scr_first, on="stay_id", how="left")

        # serum creatinine in first 24h
        self.stays["has_src_in_obs_window"] = self.stays["first_scr"].notna()
        # high serum creatinine
        self.stays["high_scr"] = self.stays["first_scr"]>4
        
        aki_abs = scr_pw.groupby("stay_id").apply(self.aki_absolute).rename("AKI_abs_48h")
        aki_abs = aki_abs.reindex(self.stays["stay_id"]).fillna(False).reset_index()

        self.stays["AKI_abs_48h"] = self.stays["stay_id"].isin(aki_abs[aki_abs["AKI_abs_48h"]]["stay_id"]) # SECOND TARGET CRITERION
  
        
    def aki_absolute(self, group):
        times = group["charttime"].values
        scr   = group["valuenum"].values

        n = len(scr)
        for i in range(n):
            for j in range(i + 1, n):
                # stop if >48h
                if times[j] - times[i] > pd.Timedelta(hours=48):
                    break
                if scr[j] - scr[i] >= 0.3:
                    return True
        return False




    def compute_baseline_scr(self, row):
        if row["is_age_eligible"]:
            if row['ckd']:
                # CKD: use first SCr at ICU admission

                return row['first_scr']
            else:
                # Non-CKD: estimate baseline using MDRD inversion
                sex_factor = 0.742 if row['gender'] == 'F' else 1
                race_factor = 1.21 if row['is_black'] else 1
                age_factor = row['age'] ** 0.203
                scr_est = ((175 * sex_factor * race_factor) / (self.eGFR_normal * age_factor)) ** (1 / 1.154)         
                return scr_est
        else:
            return np.nan


    def save_cohort(self, cohort: pd.DataFrame) -> None:
        """Save final AKI cohort and report summary stats."""
        
        cohort = cohort.rename(columns={'AKI': 'label'})
        cols = ['subject_id', 'hadm_id', 'stay_id','intime', 'outtime', 'race', 'los', 'gender', 'age', 'dod', 'label']
        cohort = cohort[cols]

        
        pct = 100 * cohort["label"].mean()
        logger.info("Number of ICU stays in AKI cohort: %s", cohort.stay_id.nunique())
        logger.info("Number of patients in AKI cohort: %s", cohort.subject_id.nunique())
        logger.info("Number of ICU stays with AKI: %s", cohort[cohort["label"] == 1].stay_id.nunique())
        logger.info("AKI positive rate: %.2f%%", pct)
        
        cohort.to_csv(self.paths["cohort_path"] / f"cohort_aki.csv", index=False)
        logger.info("AKI cohort saved.")