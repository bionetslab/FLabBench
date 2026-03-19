"""
This class extracts the gastrointestinal bleeding cohort from the MIMIC dataset.
Reference:  https://pmc.ncbi.nlm.nih.gov/articles/PMC7813389/pdf/bmjhci-2020-100245.pdf

"""
# ICU

import pandas as pd
from tqdm import tqdm
tqdm.pandas()  


from flab_cohorts.extractors.base import BaseExtractor
from flab_cohorts.utils.dataset_loader import load_icu_stays, load_icu_procedures, load_diagnoses, load_icu_items, load_icu_inputevents
from flab_cohorts.utils.logger import get_logger

logger = get_logger("GI_BLEED")


class GastrointestinalBleedingExtractor(BaseExtractor):
    

    def __init__(self, args):   
        super().__init__(args)
        
        self.age_min = 18.0
        self.transfusion_after_hours = 5.0
        self.transfusion_labels = ["PRBC", "PACKED RBC"]
        
        self.gi_bleeding_codes = [
            "5307",    "5693",    "5780",    "5781",    "5789",    
            "53021",    "53082",    "53100",    "53101",    "53120",    "53121",    "53140",
            "53141",    "53160",    "53161",    "53200",    "53201",    "53220",    "53221",    
            "53240",    "53241",    "53260",    "53261",    "53300",    "53301",    "53320",    
            "53321",    "53340",    "53341",    "53360",    "53361",    "53400",    "53401",    
            "53420",    "53421",    "53440",    "53441",    "53460",    "53461",    "53501",    
            "53511",    "53521",    "53531",    "53541",    "53551",    "53561",    "53571",    
            "53784",    "56202",    "56203",    "56212",    "56213",    "56985",
        ]
        

        
        self.stays =  load_icu_stays(self.data_path)
        self.diags = load_diagnoses(self.data_path)
        self.procedures = load_icu_procedures(self.data_path)
        self.items = load_icu_items(self.data_path)
        self.input_events = load_icu_inputevents(self.data_path)

    
    
    def extract_cohort(self):
        """Run GI bleeding cohort extraction and persist the cohort."""

        self.prepare_stays()
        self.add_gi_bleeding_diagnosis()
        self.add_transfusions()

        inclusion_mask = (self.stays["first_icustay_hadm"] & self.stays["is_age_eligible"] & self.stays["gi_bleed"])
        gi_bleeding_cohort = self.stays.loc[inclusion_mask].copy()
        
        self.save_cohort(gi_bleeding_cohort)
        


    def prepare_stays(self) -> None:
        """Prepare stay-level demographics/features; mutates self.stays in place."""

        self.stays = self.stays.merge(self.patients, on="subject_id", how="left")
        self.stays["is_age_eligible"] = (self.stays["age"] >= self.age_min) 
        self.stays = self.stays.merge(self.adms[["hadm_id","race"]], on="hadm_id", how="left")
        # label first ICU stay
        self.stays = self.stays.sort_values(["subject_id", "intime"])
        self.stays["first_icustay"] = self.stays.groupby("subject_id")["intime"].transform("min") == self.stays["intime"]
        self.stays["first_icustay_hadm"] = self.stays.groupby("hadm_id")["intime"].transform("min") == self.stays["intime"]
    
    
    
    def add_gi_bleeding_diagnosis(self) -> None:
        """Add diagnosis-based GI bleeding flag; mutates self.stays in place."""
        # Based on diagnosis codes
        gi_ids = self.diags[self.diags["icd_code"].isin(self.gi_bleeding_codes)]
        self.stays["gi_bleed"] = self.stays["hadm_id"].isin(gi_ids["hadm_id"])

        
    def add_transfusions(self) -> None:
        """Add transfusion timing flags; mutates self.stays in place."""
        # Based on inputevents in ICU
        transfusion_ids = self.items[
            (self.items["label"].str.upper().str.contains(self.transfusion_labels[0]))
            | (self.items["label"].str.upper().str.contains(self.transfusion_labels[1]))
        ]["itemid"].tolist()
        input_trans = self.input_events[self.input_events["itemid"].isin(transfusion_ids)]

        
        input_trans_hadm = input_trans.merge(self.stays[["stay_id", "intime", "outtime", "los"]], on="stay_id", how="inner")
        input_trans_hadm = input_trans_hadm[(input_trans_hadm["starttime"] <= input_trans_hadm["outtime"]) & (input_trans_hadm["starttime"] >= input_trans_hadm["intime"])]
        input_trans_hadm["hours_since_icu_admit"] = (input_trans_hadm["starttime"] - input_trans_hadm["intime"]).dt.total_seconds() / 3600
        
        self.stays["has_transf"] = self.stays["stay_id"].isin(input_trans_hadm["stay_id"])
        self.stays["has_transf_after5h"] = self.stays["stay_id"].isin(
            input_trans_hadm[input_trans_hadm["hours_since_icu_admit"] > self.transfusion_after_hours]["stay_id"]
        )


        


        

    def save_cohort(self, cohort: pd.DataFrame) -> None:
        """Save final cohort and report summary stats."""
        
        cohort = cohort.rename(columns={"has_transf_after5h": "label"})
        cols = ["subject_id", "hadm_id", "stay_id", "intime", "outtime", "race", "los", "gender", "age", "dod", "label"]
        cohort = cohort[cols]

        
        pct = 100 * cohort["label"].mean()
        
        logger.info("Number of admissions in GI bleeding cohort: %s", cohort.hadm_id.nunique())
        logger.info("Number of patients in GI bleeding cohort: %s", cohort.subject_id.nunique())
        logger.info("Number of admissions with GI bleeding: %s",cohort[cohort["label"] == 1].hadm_id.nunique())
        logger.info("GI bleeding positive rate: %.2f%%", pct)
        
        cohort.to_csv(self.paths["cohort_path"] / f"cohort_gi_bleeding.csv", index=False)
        logger.info("GI bleeding cohort saved.")