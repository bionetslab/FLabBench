"""
This class extracts the aplasia cohort from the MIMIC-IV dataset.
Reference: https://www.medrxiv.org/content/10.64898/2025.12.12.25342142v1
"""

# non-ICU

import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from datetime import timedelta
from tqdm import tqdm
tqdm.pandas()


from flab_cohorts.extractors.base import BaseExtractor
from flab_cohorts.extractors.LIT.cohort_utils import extract_diag_pts, extract_chemo_cohort, find_itemid_by_label
from flab_cohorts.utils.dataset_loader import load_labevents_for_cohort, load_d_icd_procedures, load_procedures
from flab_cohorts.utils.logger import get_logger

logger = get_logger("APLASIA")


@dataclass
class AplasiaConfig:
    follow_up_days: int = 45
    cancer_icd_code: str = "C"
    anc_label: str = "absolute neutrophil count"
    anc_threshold: float = 0.5


class AplasiaExtractor(BaseExtractor):
    COHORT_COLUMNS = [
        "subject_id", "hadm_id", "admittime", "dischtime",
        "race", "los", "gender", "age", "dod", "label",
    ]

    def __init__(self, args, config: AplasiaConfig = AplasiaConfig()):
        super().__init__(args)
        self.config = config

    def build_cancer_chemo_cohort(self) -> pd.DataFrame:
        
        cancer_pts = extract_diag_pts(self.data_path, icd_code=self.config.cancer_icd_code)
        cancer_cohort = self.adms[self.adms["subject_id"].isin(cancer_pts["subject_id"])]
        return extract_chemo_cohort(cancer_cohort, self.data_path)


    def extract_anc_labs(self, labs_df: pd.DataFrame) -> pd.DataFrame:
        
        anc_itemids = find_itemid_by_label(self.data_path, self.config.anc_label)
        labs_df = labs_df.groupby(['subject_id', 'hadm_id', 'itemid', 'charttime'])["valuenum"].max().reset_index()
        anc_df = labs_df[labs_df['itemid'].isin(anc_itemids)]
        anc_df['ANC<0.5'] = (anc_df['valuenum'] < self.config.anc_threshold).astype(int)
        return anc_df

    def find_transfusions(self) -> pd.DataFrame:
        
        proc_icd_definition_df = load_d_icd_procedures(self.data_path)
        proc_icd_df = load_procedures(self.data_path)

        transfusion_codes = proc_icd_definition_df[
            (proc_icd_definition_df['icd_version'] == 10)
            & (proc_icd_definition_df['long_title'].str.contains('transfusion', case=False, na=False))
            & (proc_icd_definition_df['long_title'].str.contains('platelet|red blood cell|RBC', case=False, na=False))
        ]['icd_code']

        return proc_icd_df[proc_icd_df['icd_code'].isin(transfusion_codes)][['hadm_id', 'chartdate']].rename(columns={'chartdate': 'transfusion_date'})

    def add_transfusion_flag(self, cohort: pd.DataFrame) -> pd.DataFrame:
        
        transfusion_procedures = self.find_transfusions()
        cohort = cohort.merge(transfusion_procedures, on='hadm_id', how='left')
        cohort['transfusion'] = cohort['transfusion_date'].notna().astype(int)
        cohort['transfusion_date'] = pd.to_datetime(cohort['transfusion_date'])
        return cohort

    def current_aplasia_occurrence(self, x: pd.Series, labs: pd.DataFrame) -> int:
        
        sub_labs = labs[
            (labs["subject_id"] == x.subject_id)
            & (labs["charttime"] >= x.admittime)
            & (labs["charttime"] <= x.dischtime)
        ].sort_values("charttime")

        if x.transfusion == 1:
            return 1
        if sub_labs.empty:
            return 0
        if sub_labs["ANC<0.5"].any():
            return 1
        return 0

    def after_admission_aplasia_occurrence(
        self, x: pd.Series, target_cohort: pd.DataFrame, labs: pd.DataFrame,) -> tuple[int, pd.Timestamp]:
        days = self.config.follow_up_days

        sub_labs = labs[
            (labs["subject_id"] == x.subject_id)
            & (labs["charttime"] >= x.dischtime)
            & (labs["charttime"] <= (x.dischtime + timedelta(days=days)))
        ].sort_values("charttime")

        sub_admissions = target_cohort[
            (target_cohort["subject_id"] == x.subject_id)
            & (target_cohort["admittime"] >= x.dischtime)
            & (target_cohort["admittime"] <= (x.dischtime + timedelta(days=days)))
        ].sort_values("admittime")

        anc_low_rows = sub_labs[sub_labs["ANC<0.5"] == True]
        transfusion_rows = sub_admissions[sub_admissions["transfusion"] == 1]

        if anc_low_rows.empty and transfusion_rows.empty:
            return 0, None

        times = []
        if not anc_low_rows.empty:
            times.append(anc_low_rows.iloc[0]["charttime"])
        if not transfusion_rows.empty:
            times.append(transfusion_rows.iloc[0]["admittime"])

        return 1, min(times)

    def split_aplasia_cases(self, x: pd.Series, target_cohort: pd.DataFrame) -> int:
        days = self.config.follow_up_days
        x["dod"] = pd.to_datetime(x["dod"])

        if x.chemo == 0 or (x.chemo == 1 and x.current_aplasia == 1):
            return 0

        if x.chemo == 1 and x.current_aplasia == 0 and x.hospital_expire_flag == 0:
            sub = target_cohort[
                (target_cohort["subject_id"] == x.subject_id)
                & (target_cohort["hadm_id"] != x.hadm_id)
                & (target_cohort["admittime"] >= x.dischtime)
                & (target_cohort["admittime"] <= (x.dischtime + timedelta(days=days)))
            ].sort_values("admittime")

            if sub.empty and x.dod <= (x.dischtime + timedelta(days=days)):
                return 0

            if sub.empty:
                if x["next_aplasia"] == 1: return 2
                if x["next_aplasia"] == 0: return 1

            if not sub.empty:
                if (sub["chemo"] == 0).all():
                    if x["next_aplasia"] == 1: return 2
                    if x["next_aplasia"] == 0: return 1

                if (sub["chemo"] == 1).any():
                    first_chemo_time = sub.loc[sub["chemo"] == 1, "admittime"].min()

                    if x["next_aplasia"] == 0:
                        return 1

                    if x["next_aplasia"] == 1:
                        if x.next_aplasia_time < first_chemo_time:
                            return 2
                        else:
                            return 0

    def extract_cohort(self):
        """Run aplasia cohort extraction."""
        
        cohort = self.build_cancer_chemo_cohort()
        labs = load_labevents_for_cohort(self.data_path, cohort)
        anc_labs = self.extract_anc_labs(labs)

        cohort = self.add_transfusion_flag(cohort)

        logger.info("Extracting aplasia cohort...")
        cohort["current_aplasia"] = cohort.progress_apply(lambda x: self.current_aplasia_occurrence(x, labs=anc_labs), axis=1)
        cohort[["next_aplasia", "next_aplasia_time"]] = cohort.progress_apply(lambda x: pd.Series(self.after_admission_aplasia_occurrence(x, cohort, labs=anc_labs)),axis=1)
        cohort["aplasia_case"] = cohort.progress_apply(lambda x: self.split_aplasia_cases(x, target_cohort=cohort), axis=1)

        cohort = cohort[cohort["aplasia_case"].isin([1, 2])]
        cohort["label"] = cohort["aplasia_case"].replace({1: 0, 2: 1}).astype(int)
        cohort = cohort.drop(columns=[
            "aplasia_case", "hospital_expire_flag", "chemo", "current_aplasia",
            "next_aplasia", "next_aplasia_time", "transfusion", "transfusion_date",
        ])

        self.save_cohort(cohort, "aplasia")
        return cohort
