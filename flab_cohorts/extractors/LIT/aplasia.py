"""
This class extracts the aplasia cohort from the MIMIC-IV dataset.
Reference: https://www.medrxiv.org/content/10.64898/2025.12.12.25342142v1
"""



import pandas as pd
from pathlib import Path
from datetime import timedelta
from tqdm import tqdm 
tqdm.pandas()  


from flab_cohorts.extractors.base import BaseExtractor
from flab_cohorts.extractors.LIT.cohort_utils import extract_diag_pts, extract_chemo_cohort, find_itemid_by_label
from flab_cohorts.utils.dataset_loader import load_labevents_for_cohort, load_d_icd_procedures, load_procedures

class AplasiaExtractor(BaseExtractor):
    def __init__(self, args):
        super().__init__(args)
        
        self.days = 45
        
    
    def extract_cohort (self):  
          
        cancer_pts= extract_diag_pts(self.data_path, icd_code="C")
        cancer_cohort = self.adms[self.adms["subject_id"].isin(cancer_pts["subject_id"])]
        cancer_chemo_cohort  = extract_chemo_cohort(cancer_cohort, self.data_path)
        cancer_chemo_cohort_labs = load_labevents_for_cohort(self.data_path, cancer_chemo_cohort)
        
        # Find ANC values 
        ANC_lab_df = self.extract_ANC_cohort(cancer_chemo_cohort_labs)
        # Find transfusions
        target_cohort = self.extract_transfusion_cohort(cancer_chemo_cohort)
        
        print("Extracting aplasia cohort ... ")
        target_cohort["current_aplasia"]  = target_cohort.progress_apply(lambda x: self.current_aplasia_occurrence(x, labs=ANC_lab_df), axis=1)
        target_cohort[["next_aplasia", "next_aplasia_time"]]  = target_cohort.progress_apply(lambda x: pd.Series(self.after_admission_aplasia_occurrence(x, target_cohort,days=self.days, labs=ANC_lab_df)), axis=1)
        target_cohort["aplasia_case"] = target_cohort.progress_apply(lambda x: self.split_aplasia_cases(x, days=self.days,target_cohort=target_cohort), axis=1)
        
        #remove admissions where NF was not determined
        target_cohort = target_cohort[target_cohort["aplasia_case"].isin([1, 2])]
        target_cohort["aplasia_case"] = target_cohort["aplasia_case"].replace({1: 0, 2: 1}).astype(int)
        

        self.save_cohort(target_cohort)
        
        return target_cohort
    
    

    def find_ANC_labels(self):
        ANC_label='absolute neutrophil count'
        ANC_itemids = find_itemid_by_label(self.data_path, ANC_label)
        print('ANC itemids are: ',ANC_itemids)
        return ANC_itemids
    
    def extract_ANC_cohort(self, labs_df):
        ANC_itemids = self.find_ANC_labels()
        labs_df = labs_df.groupby(['subject_id', 'hadm_id', 'itemid', 'charttime'])["valuenum"].max().reset_index()
        ANC_lab_df = labs_df[labs_df['itemid'].isin(ANC_itemids)]
        mask = (ANC_lab_df['valuenum'] < 0.5)
        ANC_lab_df['ANC<0.5'] = 0
        ANC_lab_df.loc[mask, 'ANC<0.5'] = 1
        return ANC_lab_df
    
    def find_transfusions(self) -> pd.DataFrame:
        proc_icd_definition_df = load_d_icd_procedures(self.data_path)
        proc_icd_df = load_procedures(self.data_path)

        transfusion_codes= proc_icd_definition_df[
            (proc_icd_definition_df['icd_version'] == 10) &
            (proc_icd_definition_df['long_title'].str.contains('transfusion', case=False, na=False)) &
            (proc_icd_definition_df['long_title'].str.contains('platelet|red blood cell|RBC', case=False, na=False))
        ]
        transfusion_codes = transfusion_codes['icd_code']
        transfusion_procedures= proc_icd_df[proc_icd_df['icd_code'].isin(transfusion_codes)][['hadm_id', 'chartdate']].rename(columns={'chartdate': 'transfusion_date'})

        return transfusion_procedures
    
    def extract_transfusion_cohort(self, cohort: pd.DataFrame) -> pd.DataFrame:
        transfusion_procedures = self.find_transfusions()
        cohort = cohort.merge(transfusion_procedures,on='hadm_id', how='left')
        
        cohort['transfusion'] = cohort['transfusion_date'].notna().astype(int)
        cohort['transfusion_date'] = pd.to_datetime(cohort['transfusion_date'])
        
        return cohort
    
    
    def current_aplasia_occurrence(self, x:pd.Series, labs:pd.DataFrame) -> int:

        sub_labs = labs[
            (labs["subject_id"] == x.subject_id) & 
            (labs["charttime"]  >= x.admittime)  & 
            (labs["charttime"]  <= x.dischtime) 
            ].sort_values("charttime") 
        
        if x.transfusion == 1:
            return 1
        if sub_labs.empty: 
            return 0
        if sub_labs["ANC<0.5"].any():
            return 1
        else:
            return 0
        
    def after_admission_aplasia_occurrence(self, x:pd.Series, target_cohort:pd.DataFrame, days:int, labs:pd.DataFrame) -> tuple[int, pd.Timestamp]:
    
        #based on Labs
        sub_labs = labs[
            (labs["subject_id"] == x.subject_id) & 
            (labs["charttime"]  >= x.dischtime)  & 
            (labs["charttime"]  <= (x.dischtime + timedelta(days=days))) 
            ].sort_values("charttime") 
        
        sub_admissions = target_cohort[
                (target_cohort["subject_id"] == x.subject_id) & 
                (target_cohort["admittime"] >= x.dischtime) & 
                (target_cohort["admittime"] <= (x.dischtime + timedelta(days=days)))
            ].sort_values("admittime") 


        
        ANC_low_rows = sub_labs[sub_labs["ANC<0.5"] == True]
        transfusion_rows = sub_admissions[sub_admissions["transfusion"] == 1]


        
        if ANC_low_rows.empty and transfusion_rows.empty: # if no aplasia in 45 days after discharge
            return 0, None

        else:
            times = []
        
            if not ANC_low_rows.empty:
                times.append(ANC_low_rows.iloc[0]["charttime"]) # first time for ANC low level
            
            if not transfusion_rows.empty:
                times.append(transfusion_rows.iloc[0]["admittime"]) # first time for transfusion
            
            min_time = min(times)
            return 1, min_time # return the earliest aplasia occurrence time
            

    
    def split_aplasia_cases(self, x:pd.Series, days:int, target_cohort:pd.DataFrame) -> int:
        
        
        x["dod"] = pd.to_datetime(x["dod"])
        if x.chemo == 0 or (x.chemo == 1 and x.current_aplasia==1): # if there is no chemo or if both chemo and aplasia are present the admission is not considered!
            return 0
        if x.chemo ==1 and x.current_aplasia == 0 and x.hospital_expire_flag ==0:
            sub = target_cohort[
                (target_cohort["subject_id"] == x.subject_id) & 
                (target_cohort["hadm_id"]  != x.hadm_id) &  # avoid selecting the same admission as next where the los for admission is 0
                (target_cohort["admittime"]  >= x.dischtime) &
                (target_cohort["admittime"] <= (x.dischtime + timedelta(days=days)))
                #(target_cohort["admittime"] <= (x.dischtime + days))
            ].sort_values("admittime")  

            #remove admissions where patient died within 30 days of discharge
            if sub.empty and x.dod <= (x.dischtime + timedelta(days=days)): 
            #if sub.empty and x.dod <= (x.dischtime + days):
                return 0
            
            if sub.empty:  # no readmission
                if (x["next_aplasia"] == 1): return 2
                if (x["next_aplasia"] == 0): return 1
                
            if not sub.empty: # if there is readmission
                
                if (sub["chemo"] == 0).all(): # if no chemo in readmissions
                    if (x["next_aplasia"] == 1): return 2
                    if (x["next_aplasia"] == 0): return 1
                    
                if (sub["chemo"] == 1).any(): # if chemo in readmissions
                    first_chemo_time = sub.loc[sub["chemo"] == 1, "admittime"].min()
                    
                    
                    if (x["next_aplasia"] == 0):  # no aplasia in 45 days 
                        return 1

                    if (x["next_aplasia"] == 1): 
                        if x.next_aplasia_time < first_chemo_time:  # if aplasia before next chemo positive else not considered
                            return 2
                    
                        else: 
                            return 0
                        
    def save_cohort(self, cohort: pd.DataFrame):
        
        cohort = cohort.rename(columns={'aplasia_case': 'label'})
        cohort = cohort.drop(columns =['hospital_expire_flag','chemo','current_aplasia','next_aplasia','next_aplasia_time','transfusion','transfusion_date'])

        
        pct = 100 * cohort["label"].mean()
        print("Number of admissions in Aplasia cohort: ", cohort.hadm_id.nunique())
        print("Number of patients in Aplasia cohort: ", cohort.subject_id.nunique())
        print("Number of admissions with Aplasia: ", cohort[cohort["label"] == 1].hadm_id.nunique())
        print(f"Aplasia positive in 45 days: {pct:.2f}%")
        
        cohort.to_csv(self.paths["cohort_path"] / f"cohort_aplasia.csv", index=False)
        print("Aplasia cohort saved.")
        
                

