import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union
from flab_cohorts.config.constants import DTB_DATA_PATH
from flab_cohorts.cohort_extractor.base import BaseExtractor
from tqdm import tqdm


class DTBExtractor(BaseExtractor):
    def __init__(self, args):
        super().__init__(args)
        self._prepare_data()

    def _prepare_data(self):
        
        self.adms = self.adms.sort_values("admittime")
        self.adms["next_admittime"] = self.adms.groupby("subject_id")["admittime"].shift(-1)
        self.adms["next_hadm_id"] = self.adms.groupby("subject_id")["hadm_id"].shift(-1)
        
        # Merge patients data to admissions data
        self.adms = self.adms.merge(self.patients, on="subject_id", how="left")
        
        # ONLY KEEP ICD-10 DIAGNOSES
        self.diags_10 = self.diags[self.diags["icd_version"] == 10].copy()
        self.diags_10["icd_group"] = self.diags_10["icd_code"].str[:3]
        self.diags_10 = self.diags_10.merge(self.adms[["hadm_id", "admittime", "dischtime"]], on="hadm_id", how="left")
        self.diags_10 = self.diags_10.sort_values(["subject_id", "admittime"])
        
        return self.adms, self.diags_10
    
    
    def select_edges(self, cohort: str) -> pd.DataFrame:

        try: 
            edges_df = pd.read_csv(DTB_DATA_PATH / "41467_2020_18682_MOESM1_ESM.tsv", sep="\t")
            
            if cohort == "DTB_all":
                sel_edges = edges_df[(edges_df["RR"] > 1) & (edges_df["direction_yes_no"] == 1) & (edges_df["D2"] != "D99")].dropna()
                sel_edges = sel_edges[:10]
            else:
                try:
                    D1, D2 = cohort.split("-")
                    sel_edges = edges_df[(edges_df["D1"] == D1) & (edges_df["D2"] == D2)]
                except (ValueError, KeyError):
                    raise ValueError(f"Invalid cohort format: {cohort}. Expected format: D1-D2. Edges not found in trajectory file.")
            
            return sel_edges
            
        except FileNotFoundError:
            raise ValueError(f"No edges file found at {DTB_DATA_PATH}")
       


    def extract_cohort_from_edge(self, row: pd.Series):
        
        D1 = row["D1"]
        D2 = row["D2"]
        W = row["CODE_DIFF_DAYS"]
        # D1, D2, W are method parameters
        D1_first = self.diags_10[self.diags_10["icd_group"] == D1].groupby("subject_id").first().reset_index()
        D2_first = self.diags_10[self.diags_10["icd_group"] == D2].groupby("subject_id").first().reset_index()
        sub_adms = self.adms[self.adms["subject_id"].isin(D1_first["subject_id"])].merge(
            D1_first[["subject_id", "admittime"]].rename(columns={"admittime": "D1_date"}),
            on="subject_id", how="left"
        )
        sub_adms["D1_W"] = sub_adms["D1_date"] + pd.to_timedelta(W, unit="D")
        sub_adms["D1_5y"] = sub_adms["D1_date"] + pd.to_timedelta(5 * 365, unit="D")
        sub_adms = sub_adms.merge(
            D2_first[["subject_id", "admittime"]].rename(columns={"admittime": "D2_date"}),
            on="subject_id", how="left"
        )
        
        #admissions after D1 diagnosis
        #admissions within W days of D1 diagnosis
        # no D2 diagnosis OR admissions before D2 diagnosis
        # remove admissions where, patient died within 5 years of D1 diagnosis and no D2 diagnosis
        # patient must have a next admission within 5 years of D1 diagnosis
        
        cohort = sub_adms[
            (sub_adms["admittime"] >= sub_adms["D1_date"]) 
            & (sub_adms["admittime"] <= sub_adms["D1_W"]) 
            & ((sub_adms["D2_date"].isna()) | (sub_adms["admittime"] < sub_adms["D2_date"]))
            & ~((sub_adms["dod"] <= sub_adms["D1_5y"]) & sub_adms["D2_date"].isna()) 
            #& ~(sub_adms["deathtime"].notna() & (sub_adms["deathtime"] <= sub_adms["D1_5y"]) & sub_adms["D2_date"].isna())
            & (sub_adms["next_admittime"].notna() & (sub_adms["next_admittime"] <= sub_adms["D1_5y"]))
        ]
        cohort["target_D2_5y"] = ((cohort["D2_date"].notna()) & (cohort["D2_date"] <= cohort["D1_5y"])).astype(int)
        
        
        cohort_counts = cohort["target_D2_5y"].value_counts().to_dict()
        n_pos = cohort_counts.get(1, 0)
        n_neg = cohort_counts.get(0, 0)
        if n_pos > 10 and n_neg > 50:
            cohort.to_csv(self.paths["cohort_path"] / f"cohort_{D1}_{D2}_{W}.csv", index=False)
            #print(f"cohort {D1}-{D2}-{W} has {n_pos} positive and {n_neg} negative. Saved.")
        else:
            print(f"cohort {D1}-{D2}-{W} has {n_pos} positive and {n_neg} negative. Skipping...")

        
    
    def extract_full_cohort(self, cohort: str) -> pd.DataFrame:
        sel_edges = self.select_edges(cohort)
        tqdm.pandas()
        sel_edges.progress_apply(self.extract_cohort_from_edge, axis=1)

