from pathlib import Path
import pandas as pd

from config.constants import PROJECT_ROOT, ICD_CHAPTERS

COHORTS_DIR = PROJECT_ROOT / "saved_data" / "cohorts" / "DTB"
EDGES_PATH = COHORTS_DIR / "selected_edges_DTB_all.csv"


class CohortCatalog:
    def __init__(self, edges_path: Path = EDGES_PATH):
        edges = pd.read_csv(edges_path)
        edges["D1_chapter"] = edges["D1"].str[0].map(ICD_CHAPTERS)
        edges["D2_chapter"] = edges["D2"].str[0].map(ICD_CHAPTERS)
        edges["cohort"] = edges["D1"] + "-" + edges["D2"]
        edges["n_total"] = edges["n_neg"] + edges["n_pos"]
        edges["target_rate"] = edges["n_pos"] / edges["n_total"]
        self.edges = edges

    def query(
        self,
        d1_chapter: str = None, d2_chapter: str = None,
        min_rr: float = None, max_rr: float = None,
        min_n_pos: int = None, min_n_total: int = None,
        min_target_rate: float = None, max_target_rate: float = None,
        min_age: float = None, max_age: float = None,
        min_window_days: float = None, max_window_days: float = None,
        extracted_only: bool = False,
    ) -> pd.DataFrame:
        df = self.edges
        if d1_chapter is not None:
            df = df[df["D1_chapter"] == d1_chapter]
        if d2_chapter is not None:
            df = df[df["D2_chapter"] == d2_chapter]
        if min_rr is not None:
            df = df[df["RR"] >= min_rr]
        if max_rr is not None:
            df = df[df["RR"] <= max_rr]
        if min_n_pos is not None:
            df = df[df["n_pos"] >= min_n_pos]
        if min_n_total is not None:
            df = df[df["n_total"] >= min_n_total]
        if min_target_rate is not None:
            df = df[df["target_rate"] >= min_target_rate]
        if max_target_rate is not None:
            df = df[df["target_rate"] <= max_target_rate]
        if min_age is not None:
            df = df[df["AGE_AT_DISEASE"] >= min_age]
        if max_age is not None:
            df = df[df["AGE_AT_DISEASE"] <= max_age]
        if min_window_days is not None:
            df = df[df["CODE_DIFF_DAYS"] >= min_window_days]
        if max_window_days is not None:
            df = df[df["CODE_DIFF_DAYS"] <= max_window_days]
        if extracted_only:
            df = df[df["cohort"].isin(self.extracted_cohorts())]
        return df.sort_values("RR", ascending=False)

    def extracted_cohorts(self) -> set:
        return {
            p.name.removeprefix("cohort_").removesuffix(".csv.gz")
            for p in COHORTS_DIR.glob("cohort_*.csv.gz")
        }


def interactive_query(catalog: "CohortCatalog"):
    import ipywidgets as widgets
    from IPython.display import display, clear_output

    edges = catalog.edges
    style = {"description_width": "150px"}
    layout = widgets.Layout(width="500px")

    chapters = sorted(set(edges["D1_chapter"].dropna()) | set(edges["D2_chapter"].dropna()))
    chapter_options = ["Any"] + chapters

    d1_dd = widgets.Dropdown(options=chapter_options, description="D1 disease group:", style=style)
    d2_dd = widgets.Dropdown(options=chapter_options, description="D2 disease group:", style=style)

    rr_slider = widgets.FloatRangeSlider(
        value=[1, edges["RR"].max()], min=0, max=edges["RR"].max(), step=0.1,
        description="RR range:", style=style, layout=layout,
    )
    n_pos_slider = widgets.IntRangeSlider(
        value=[10, edges["n_pos"].max()], min=0, max=int(edges["n_pos"].max()),
        description="n_pos range:", style=style, layout=layout,
    )
    target_rate_slider = widgets.FloatRangeSlider(
        value=[0, 1], min=0, max=1, step=0.01, readout_format=".2f",
        description="target rate:", style=style, layout=layout,
    )
    age_slider = widgets.FloatRangeSlider(
        value=[edges["AGE_AT_DISEASE"].min(), edges["AGE_AT_DISEASE"].max()],
        min=edges["AGE_AT_DISEASE"].min(), max=edges["AGE_AT_DISEASE"].max(),
        description="age at disease:", style=style, layout=layout,
    )
    window_slider = widgets.FloatRangeSlider(
        value=[0, edges["CODE_DIFF_DAYS"].max()], min=0, max=edges["CODE_DIFF_DAYS"].max(),
        description="window (days):", style=style, layout=layout,
    )
    extracted_only_cb = widgets.Checkbox(value=True, description="only extracted cohorts")
    run_button = widgets.Button(description="Search cohorts", button_style="primary")
    out = widgets.Output()

    def on_click(_):
        with out:
            clear_output()
            result = catalog.query(
                d1_chapter=None if d1_dd.value == "Any" else d1_dd.value,
                d2_chapter=None if d2_dd.value == "Any" else d2_dd.value,
                min_rr=rr_slider.value[0], max_rr=rr_slider.value[1],
                min_n_pos=n_pos_slider.value[0],
                min_target_rate=target_rate_slider.value[0], max_target_rate=target_rate_slider.value[1],
                min_age=age_slider.value[0], max_age=age_slider.value[1],
                min_window_days=window_slider.value[0], max_window_days=window_slider.value[1],
                extracted_only=extracted_only_cb.value,
            )
            print(f"{len(result)} matching cohorts")
            display(result[[
                "cohort", "D1_chapter", "D2_chapter", "RR",
                "n_pos", "n_neg", "target_rate", "AGE_AT_DISEASE",
            ]].head(50))

    run_button.on_click(on_click)

    return widgets.VBox([
        d1_dd, d2_dd, rr_slider, n_pos_slider, target_rate_slider,
        age_slider, window_slider, extracted_only_cb, run_button, out,
    ])
