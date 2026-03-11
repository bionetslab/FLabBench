import pandas as pd
from pathlib import Path
from datetime import timedelta
from tqdm import tqdm 
tqdm.pandas()  


from flab_cohorts.cohort_extractor.base import BaseExtractor
from flab_cohorts.cohort_extractor.LIT.cohort_utils import extract_diag_pts, extract_chemo_cohort, split_neutropenic_fever_cases

class NeutropenicFeverExtractor(BaseExtractor):
    def __init__(self, args):
        super().__init__(args)
        
    
    def extract_cohort (self):  
          
        cancer_pts= extract_diag_pts(self.data_path, icd_code="C")
        cancer_cohort = self.adms[self.adms["subject_id"].isin(cancer_pts["subject_id"])]
        cancer_chemo_cohort  = extract_chemo_cohort(cancer_cohort, self.data_path)
        target_cohort = self.current_NF_occurance(cancer_chemo_cohort, self.data_path)
        print("Extracting NF cohort ... ")
        target_cohort["NF_in_30_days"] = target_cohort.progress_apply(lambda x: self.split_neutropenic_fever_cases(x, 30,target_cohort,"both readmissions and no admission"), axis=1)
        
        #remove admissions where NF was not determined
        target_cohort = target_cohort[target_cohort["NF_in_30_days"].isin([1, 2])]
        target_cohort["NF_in_30_days"] = target_cohort["NF_in_30_days"].replace({1: 0, 2: 1}).astype(int)
        
        
        pct = 100 * target_cohort["NF_in_30_days"].mean()
        print(f"Neutropenic fever positive in 30 days: {pct:.2f}%")
        
        target_cohort.to_csv(self.paths["cohort_path"] / f"cohort_neutropenic_fever.csv", index=False)
        print("Neutropenic fever cohort saved.")
        
        
        return target_cohort
    
    
    def current_NF_occurance(self, cohort:pd.DataFrame, data_path:Path):
        
        
        icd_code= 'R50'
        fever_pts= extract_diag_pts(data_path, icd_code = icd_code)
        cohort['fever'] = cohort['hadm_id'].isin(fever_pts['hadm_id']).astype(int)

        icd_code= 'D70'
        neutropenia_pts = extract_diag_pts(data_path, icd_code = icd_code)
        cohort['neutropenia'] = cohort['hadm_id'].isin(neutropenia_pts['hadm_id']).astype(int)
        
        
        cohort['NF'] =((cohort['fever'] == 1) & (cohort['neutropenia'] == 1)).astype(int)
        return cohort


    def split_neutropenic_fever_cases(self, x:pd.Series, days:int, target_cohort:pd.DataFrame, spliting_approach:str):
        x["dod"] = pd.to_datetime(x["dod"])
        # extract all readmissions
        if x.chemo ==1 and x.NF ==0 and x.hospital_expire_flag ==0:
            sub = target_cohort[
                (target_cohort["subject_id"] == x.subject_id) & 
                (target_cohort["admittime"] > x.dischtime) & 
                (target_cohort["admittime"] <= (x.dischtime + timedelta(days=days)))
            ].sort_values("admittime")  
            
            #remove admissions where patient died within 30 days of discharge
            if sub.empty and x.dod <= (x.dischtime + timedelta(days=days)): 
                return 0
            #check for other chemotherapy within 30 days
            if not sub.empty and (sub["chemo"] == 1).any(): # if there is another chemo in 30 days
                positive_chemo_index = (sub["chemo"] == 1).argmax()
                readmissions_after_next_chemo = sub[positive_chemo_index:] 
                sub = sub[:positive_chemo_index]
                if ((sub["NF"] == 0).all() or sub.empty):# no NF before second chemo
                    if (readmissions_after_next_chemo ["NF"] == 0).all():
                        return 1 # all readmissions after first chemo have negative NF
                    else:
                        return 0 # second chemo or admissions after that have at least on positive NF

            
            # cohort 1: check only readmissions
            if spliting_approach == "only readmissions":
                if not sub.empty and (sub["NF"] == 0).all():
                    return 1
                if not sub.empty and (sub["NF"] == 1).any():
                    return 2
            
            #cohort 2: check both readmissions and no admissions
            if spliting_approach == "both readmissions and no admission":
                if sub.empty or (sub["NF"] == 0).all():
                    return 1
                if not sub.empty and (sub["NF"]== 1).any():
                    return 2
                else: 
                    return 0
                