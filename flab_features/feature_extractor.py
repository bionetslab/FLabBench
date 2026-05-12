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
        self.labs_parquet = self.output_dir / "labs.parquet"
        if not self.labs_parquet.exists():
            print("Cleaning raw lab_events")
            self._clean_lab_events()

    def _clean_lab_events(self):
        chunks = []
        for chunk in tqdm(pd.read_csv(
            self.mimic_dir / "hosp/labevents.csv.gz",
            compression="gzip",
            usecols=["subject_id", "itemid", "charttime", "valuenum"],
            dtype={"subject_id": "int64", "itemid": "int64", "valuenum": "float64"},
            parse_dates=["charttime"],
            chunksize=1_000_000,
        )):
            chunk = chunk.dropna(subset=["valuenum"])
            chunk = chunk[~chunk["itemid"].isin(ITEMIDS_TO_REMOVE)]
            chunks.append(chunk)
        pd.concat(chunks, ignore_index=True).sort_values("subject_id").to_parquet(self.labs_parquet, index=False)

    def extract(self, cohort_df, cohort_name):
        cohort_df = cohort_df.copy()
        cohort_df["dischtime"] = pd.to_datetime(cohort_df["dischtime"])

        adms = cohort_df[["subject_id", "hadm_id", "dischtime"]].drop_duplicates("hadm_id")
        adms = adms.copy()
        adms["starttime"] = (adms["dischtime"] - pd.DateOffset(days=self.days)).apply(lambda x: x.replace(hour=0, minute=0, second=0))
        adms["time_before_disch"] = adms["dischtime"] - pd.Timedelta(days=self.days)

        subject_ids = list(adms["subject_id"].unique())

        labs = pd.read_parquet(self.labs_parquet, filters=[("subject_id", "in", subject_ids)])

        if self.top_features is not None:
            labs = labs[labs["itemid"].isin(self.top_features)]

        if labs.empty:
            return

        sub = labs.merge(
            adms[["subject_id", "hadm_id", "dischtime", "starttime", "time_before_disch"]],
            on="subject_id",
        )
        sub = sub[
            (sub["charttime"] >= sub["time_before_disch"]) &
            (sub["charttime"] <= sub["dischtime"])
        ]

        if sub.empty:
            return

        sub["minute"] = (sub["charttime"] - sub["starttime"]).dt.total_seconds() / 60
        sub = sub[sub["minute"] >= 0]
        sub = sub.rename(columns={"valuenum": "value"})

        out_dir = self.output_dir / cohort_name
        out_dir.mkdir(parents=True, exist_ok=True)

        sub[["subject_id", "hadm_id", "itemid", "value", "minute"]].to_csv(out_dir / "features.csv.gz",compression="gzip",index=False)
