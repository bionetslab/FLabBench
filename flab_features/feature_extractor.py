import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm


ITEMIDS_TO_REMOVE = [50934, 50947, 51678]

class FeatureExtractor:
    def __init__(self, mimic_dir, features_base_path, top_features_path=None, days_before_discharge=14):
        self.mimic_dir = Path(mimic_dir)
        self.output_dir = Path(features_base_path)
        self.days = days_before_discharge
        self.top_features = None
        if top_features_path is not None:
            with open(Path(top_features_path) / "mimic_top100_features.pkl", "rb") as f:
                self.top_features = set(int(x) for x in pickle.load(f))
            print(f"Top features filter applied: {len(self.top_features)} itemids")

    def extract(self, cohort_df, cohort_name):
        cohort_df = cohort_df.copy()
        cohort_df["dischtime"] = pd.to_datetime(cohort_df["dischtime"])

        adms = cohort_df[["subject_id", "hadm_id", "dischtime"]].drop_duplicates("hadm_id")
        adms = adms.copy()
        adms["starttime"] = (adms["dischtime"] - pd.DateOffset(days=self.days)).apply(lambda x: x.replace(hour=0, minute=0, second=0))
        adms["time_before_disch"] = adms["dischtime"] - pd.Timedelta(days=self.days)

        subject_ids = set(adms["subject_id"])
        collected = []

        for chunk in tqdm(pd.read_csv(
            self.mimic_dir / "hosp/labevents.csv.gz",
            compression="gzip",
            usecols=["subject_id", "itemid", "charttime", "valuenum", "valueuom"],
            dtype={"subject_id": "int64", "itemid": "int64", "valuenum": "float64", "valueuom": "object"},
            parse_dates=["charttime"],
            chunksize=1_000_000,
        )):
            chunk = chunk.dropna(subset=["valuenum"])
            chunk = chunk[chunk["subject_id"].isin(subject_ids)]
            chunk = chunk[~chunk["itemid"].isin(ITEMIDS_TO_REMOVE)]
            if self.top_features is not None:
                chunk = chunk[chunk["itemid"].isin(self.top_features)]
            if chunk.empty:
                continue
            sub = chunk.merge(
                adms[["subject_id", "hadm_id", "dischtime", "starttime", "time_before_disch"]],
                on="subject_id",
            )
            sub = sub[
                (sub["charttime"] >= sub["time_before_disch"]) &
                (sub["charttime"] <= sub["dischtime"])
            ]
            if sub.empty:
                continue
            collected.append(
                sub[["subject_id", "hadm_id", "itemid", "valuenum", "valueuom", "charttime", "starttime"]]
            )

        if not collected:
            return

        labs = pd.concat(collected, ignore_index=True)

        labs["minute"] = (labs["charttime"] - labs["starttime"]).dt.total_seconds() / 60
        labs = labs[labs["minute"] >= 0]
        labs = labs.rename(columns={"valuenum": "value"})

        out_dir = self.output_dir / cohort_name
        out_dir.mkdir(parents=True, exist_ok=True)

        labs[["subject_id", "hadm_id", "itemid", "value", "minute"]].to_csv(out_dir / "features.csv.gz",compression="gzip",index=False)
