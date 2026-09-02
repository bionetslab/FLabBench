from pathlib import Path
import datetime
import hashlib
import io
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COHORTS_DIR = PROJECT_ROOT / "saved_data" / "cohorts" / "DTB" / "new"
SAVED_DATA = PROJECT_ROOT / "saved_data"

# `streamlit run` puts dashboard/ on sys.path, not the project root, so the
# pipeline packages (config, flab_features) are not importable without this.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Must be the first Streamlit call in the script.
st.set_page_config(page_title="My cohorts", layout="wide")

COHORT_FILES = {
    "First admissions only": "first_adms_cohorts_summary.csv",
    "All admissions": "initial_cohorts_summary.csv",
}

# Replaces the notebook's `feat_labels`: column -> axis label.
FEAT_LABELS = {
    "n_cohort": "Cohort size",
    "log_n_cohort": "Log cohort size",
    "target_rate": "Positive rate",
    "log_target_rate": "Log positive rate",
    "RR": "Relative risk",
    "log_RR": "Log relative risk",
    "imbalance_ratio": "Imbalance ratio",
    "log_imb": "Log imbalance ratio",
    "AGE_AT_DISEASE_cohort": "Age at disease (years)",
    "female_rate_cohort": "Female rate",
    "death_rate_cohort": "Death rate",
    "n_pos_test_estimate": "Est. test positives",
    "n_neg_test_estimate": "Est. test negatives",
    "CODE_DIFF_DAYS_cohort": "D1 to D2 gap (days)",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


@st.cache_data
def load_cohorts(filename: str) -> pd.DataFrame:
    """Read one cohort summary CSV.

    Streamlit re-runs the whole script on every widget click, so @st.cache_data
    keeps the file from being read again each time.
    """
    return pd.read_csv(COHORTS_DIR / filename)


st.title("FLabBench Cohorts")


choice = st.selectbox("Which cohort table?", list(COHORT_FILES))
df = load_cohorts(COHORT_FILES[choice])

st.write(f"Loaded **{len(df):,}** cohorts with **{df.shape[1]}** columns.")
st.dataframe(df, width="stretch", height=500)
with st.expander("What columns are in here?"):
    st.write(list(df.columns))
# --------------------------------------------------------------------------- #
# Distribution of one characteristic, for the selected cohort table
# --------------------------------------------------------------------------- #

st.subheader("Distribution of one characteristic")

# This widget replaces the notebook's hard-coded `feat_to_plot_1`.
feat_to_plot_1 = st.selectbox(
    "Feature", list(FEAT_LABELS), format_func=lambda c: FEAT_LABELS[c],
)


def feature_histogram(data, feature, edges, y_max, title, color):
    """One histogram. Called twice below with the SAME `edges` and `y_max`, so
    both panels are directly comparable - bar for bar and height for height."""
    fig = px.histogram(data, x=feature,
                       color_discrete_sequence=[color], opacity=0.8)
    fig.update_traces(xbins=dict(start=edges[0], end=edges[-1],
                                 size=edges[1] - edges[0]))
    fig.update_layout(
        title=title,
        xaxis_title=FEAT_LABELS[feature],
        yaxis_title="Nr. of cohorts",
        xaxis=dict(range=[edges[0], edges[-1]]),   # same x span in both panels
        yaxis=dict(range=[0, y_max]),              # same y scale in both panels
        width=500,
        height=350,
        margin=dict(l=60, r=20, t=45, b=50),
        font=dict(size=12),
    )
    return fig


# Bins come from the FULL table, then get reused for the reliable subset.
# Recomputing them per subset would give each panel different bars.
edges = np.histogram_bin_edges(df[feat_to_plot_1].dropna(), bins=40)
reliable_df = df[df["reliable"]]

# Tallest bar across BOTH panels, so one shared y scale fits both.
counts_all, _ = np.histogram(df[feat_to_plot_1].dropna(), bins=edges)
counts_rel, _ = np.histogram(reliable_df[feat_to_plot_1].dropna(), bins=edges)
y_max = max(counts_all.max(), counts_rel.max(), 1) * 1.05

left, right, _spacer = st.columns([1, 1, 0.35], gap="small")
with left:
    st.plotly_chart(
        feature_histogram(df, feat_to_plot_1, edges, y_max,
                          f"{choice} — all ({len(df):,})", "#431C3A"),
        width="content",
    )
with right:
    st.plotly_chart(
        feature_histogram(reliable_df, feat_to_plot_1, edges, y_max,
                          f"{choice} — reliable only ({len(reliable_df):,})", "#1C7A8A"),
        width="content",
    )


# --------------------------------------------------------------------------- #
# Cohort size vs positive rate, coloured by D1 chapter
# --------------------------------------------------------------------------- #

st.subheader("Cohort characteristics by D1 chapter")

ring_reliable = st.checkbox(
    "Ring reliable cohorts",
    help="Draws an open circle around every cohort flagged reliable, in both "
         "plots below. Colour keeps showing the D1 chapter.",
)

# Same husl colours as the notebook, so the two figures look like siblings.
hue_order = sorted(df["D1type"].dropna().unique())
palette_dict = dict(zip(hue_order, sns.color_palette("husl", len(hue_order)).as_hex()))

scatter_df = df.dropna(subset=["log_n_cohort", "log_target_rate", "D1type"])

fig2 = px.scatter(
    scatter_df,
    x="log_n_cohort",
    y="log_target_rate",
    color="D1type",
    color_discrete_map=palette_dict,
    category_orders={"D1type": hue_order},
    opacity=0.5,
    render_mode="webgl",          # 4,600 points stay responsive
    hover_name="cohort_name",
    hover_data={
        "D1type": True,
        "D2type": True,
        "n_cohort": ":,",
        "target_rate": ":.1%",
        "log_n_cohort": False,    # hidden: already on the axis
        "log_target_rate": False,
    },
)
fig2.update_traces(marker=dict(size=6, line=dict(width=0)))


def add_reliable_rings(fig, data, x, y):
    """Open circles around the reliable cohorts, drawn on top of the coloured
    points. Used by both scatters so they stay consistent."""
    rel = data[data["reliable"]]
    if rel.empty:
        return
    fig.add_trace(go.Scattergl(
        x=rel[x], y=rel[y], mode="markers",
        name=f"Reliable ({len(rel):,})",
        marker=dict(size=11, symbol="circle-open",
                    line=dict(width=1.4, color="#52514e")),
        hoverinfo="skip",   # the coloured point underneath already has a tooltip
    ))


# Added after the coloured traces, so the rings sit on top.
if ring_reliable:
    add_reliable_rings(fig2, scatter_df, "log_n_cohort", "log_target_rate")

fig2.update_layout(
    title=choice,
    xaxis_title="Log Cohort Size",
    yaxis_title="Log Positive Rate",
    width=560,
    height=430,
    margin=dict(l=60, r=20, t=45, b=50),
    font=dict(size=12),
    legend=dict(title="D1 Group", itemsizing="constant"),
)


# --- Positives vs negatives, against the reliability threshold --------------- #

# Same definition as the notebook: reliable needs at least `min_test_n` positives
# AND negatives in the test split, so n_pos and n_neg must each clear
# min_test_n / test_frac.
TEST_FRAC = 0.2
MIN_TEST_N = 10
threshold = MIN_TEST_N / TEST_FRAC          # = 50

thr_df = df.dropna(subset=["n_pos", "n_neg", "D1type"])

# The notebook used sharex/sharey so both panels had identical axes. Here the two
# universes are shown one at a time, so we fix the axes to the range spanning
# BOTH tables - otherwise switching the dropdown rescales the plot and the two
# are no longer comparable. Both loads are cached, so this is free.
_both = pd.concat([load_cohorts(f) for f in COHORT_FILES.values()])
# On a log axis, `range` is given in log10 units.
x_range = [np.log10(_both["n_pos"].min()) - 0.15, np.log10(_both["n_pos"].max()) + 0.15]
y_range = [np.log10(_both["n_neg"].min()) - 0.15, np.log10(_both["n_neg"].max()) + 0.15]

# Coloured by D1 chapter with the same palette as the plot on the left, so a
# chapter is the same colour in both. Reliability is shown by the rings and by
# the threshold lines instead of by colour.
fig3 = px.scatter(
    thr_df,
    x="n_pos",
    y="n_neg",
    color="D1type",
    color_discrete_map=palette_dict,
    category_orders={"D1type": hue_order},
    opacity=0.5,
    log_x=True,
    log_y=True,
    render_mode="webgl",
    hover_name="cohort_name",
    hover_data={
        "D1type": True,
        "D2type": True,
        "n_pos": ":,",
        "n_neg": ":,",
        "target_rate": ":.1%",
        "reliable": True,
    },
)
fig3.update_traces(marker=dict(size=6, line=dict(width=0)))

# The chapter colours are already spelled out by the legend on the left-hand
# plot, so hide those 15 entries here and give the width back to the plot area.
# Traces added below (threshold lines, rings) keep their own legend entries.
fig3.update_traces(showlegend=False)

# Drawn as traces rather than add_vline/add_hline: shapes on a log axis are
# interpreted in log10 units, traces in plain data units. Traces are the
# unambiguous option. Grey rather than black so the lines show up in both the
# light and the dark Streamlit theme.
# They span the full axis range - like matplotlib's axvline/axhline - not just
# the extent of the data.
x_span = [10 ** x_range[0], 10 ** x_range[1]]
y_span = [10 ** y_range[0], 10 ** y_range[1]]
line_style = dict(dash="dash", color="#7b7874", width=1.2)

fig3.add_trace(go.Scatter(                      # vertical: n_pos = 50
    x=[threshold, threshold], y=y_span, mode="lines", line=line_style,
    name=f"threshold = {threshold:.0f}", legendgroup="thr",
    hovertemplate=f"n_pos = {threshold:.0f}<extra></extra>",
))
fig3.add_trace(go.Scatter(                      # horizontal: n_neg = 50
    x=x_span, y=[threshold, threshold], mode="lines", line=line_style,
    legendgroup="thr", showlegend=False,
    hovertemplate=f"n_neg = {threshold:.0f}<extra></extra>",
))

if ring_reliable:
    add_reliable_rings(fig3, thr_df, "n_pos", "n_neg")

n_reliable = int(thr_df["reliable"].sum())
fig3.add_annotation(
    xref="paper", yref="paper", x=0.02, y=0.98, xanchor="left", yanchor="top",
    align="left", showarrow=False,
    text=(f"total: {len(thr_df):,}<br>"
          f"reliable: {n_reliable:,}<br>"
          f"not reliable: {len(thr_df) - n_reliable:,}"),
    font=dict(size=10),
    bgcolor="rgba(255,255,255,0.8)", bordercolor="gray", borderwidth=1, borderpad=4,
)

fig3.update_layout(
    title=f"{choice} — threshold at {threshold:.0f}",
    xaxis_title="n_pos",
    yaxis_title="n_neg",
    xaxis=dict(range=x_range),   # fixed, so both universes share the same axes
    yaxis=dict(range=y_range),
    width=560,
    height=430,
    margin=dict(l=60, r=20, t=45, b=50),
    font=dict(size=12),
    legend=dict(title="D1 Group", itemsizing="constant"),
)


left2, right2, _spacer2 = st.columns([1, 1, 0.2], gap="small")
with left2:
    st.plotly_chart(fig2, width="content")
with right2:
    st.plotly_chart(fig3, width="content")


# --------------------------------------------------------------------------- #
# Which ICD chapters the cohorts are built from (D1 and D2 shares)
# --------------------------------------------------------------------------- #

st.subheader("Chapter composition")

d1 = df["D1type"].value_counts()
d2 = df["D2type"].value_counts()

# Chapter order: largest first, by whichever of D1/D2 is bigger - so both donuts
# use one shared order and a chapter sits in the same place in each.
types = sorted(set(d1.index) | set(d2.index))
order = (
    pd.DataFrame({"D1": d1.reindex(types).fillna(0), "D2": d2.reindex(types).fillna(0)})
    .assign(max_=lambda x: x[["D1", "D2"]].max(axis=1))
    .sort_values("max_", ascending=False)
    .index.tolist()
)

# specs type "domain" is what lets a pie/donut live in a subplot cell.
fig4 = make_subplots(rows=1, cols=2,
                     specs=[[{"type": "domain"}, {"type": "domain"}]],
                     subplot_titles=("D1", "D2"))

for col, series in enumerate([d1, d2], start=1):
    values = series.reindex(order).fillna(0).values
    percent = 100 * values / values.sum()
    fig4.add_trace(
        go.Pie(
            labels=order,
            values=values,
            hole=0.5,             # matplotlib's wedgeprops width=0.5
            sort=False,           # keep `order`; Plotly would re-sort by value
            direction="clockwise",
            marker=dict(colors=[palette_dict[c] for c in order]),
            opacity=0.5,          # matplotlib's alpha=0.5
            # Reproduces autopct=lambda p: f"{p:.0f}%" if p > 5 else ""
            text=[f"{p:.0f}%" if p > 5 else "" for p in percent],
            textinfo="text",
            textposition="inside",
            insidetextfont=dict(size=12),
            hovertemplate="%{label}<br>%{value:,} cohorts (%{percent})<extra></extra>",
            showlegend=(col == 1),   # one shared legend, not two identical ones
        ),
        row=1, col=col,
    )

fig4.update_layout(
    title=choice,
    width=820,
    height=420,
    margin=dict(l=20, r=20, t=70, b=20),
    font=dict(size=12),
    legend=dict(title="ICD Group", itemsizing="constant"),
)

st.plotly_chart(fig4, width="content")


# --------------------------------------------------------------------------- #
# UMAP of the cohort characteristics
# --------------------------------------------------------------------------- #

st.subheader("UMAP of cohort characteristics")

# Cohort-specific features only: everything is computed on the patients actually
# extracted, not on the DTB paper's aggregate statistics. The paper columns
# (female_rate, death_rate, AGE_AT_DISEASE_years, CODE_DIFF_DAYS) are
# deliberately not offered - they describe a different population (paper death
# rate 47% vs cohort 18%, correlation 0.59).
#
# log_RR is the one exception, and it is kept because it has no cohort
# equivalent: RR is a property of the D1->D2 edge in the source table, not a
# population statistic, so there is nothing to recompute it from. Deselect it in
# the picker if you want a strictly cohort-only embedding.
UMAP_FEATURES = ["n_cohort", "target_rate", "log_RR", "AGE_AT_DISEASE_cohort",
                 "female_rate_cohort", "death_rate_cohort",
                 "D1type_enc", "D2type_enc"]

UMAP_CANDIDATES = [
    "log_n_cohort", "n_cohort", "target_rate", "log_target_rate", "log_imb",
    "imbalance_ratio", "D1type_enc", "D2type_enc",
    "female_rate_cohort", "death_rate_cohort", "AGE_AT_DISEASE_cohort",
    "CODE_DIFF_DAYS_cohort",
    "n_pos_test_estimate", "n_neg_test_estimate",
    "log_RR",   # edge property from the source table; no cohort equivalent
]

UMAP_LABELS = {
    "n_cohort":"cohort size",
    "log_n_cohort": "Log cohort size",
    "target_rate": "Positive rate",
    "log_target_rate": "Log positive rate",
    "log_imb": "Log imbalance ratio",
    "imbalance_ratio": "Imbalance ratio",
    "D1type_enc": "D1 chapter (encoded)",
    "D2type_enc": "D2 chapter (encoded)",
    "female_rate_cohort": "Female rate",
    "death_rate_cohort": "Death rate",
    "AGE_AT_DISEASE_cohort": "Mean age (years)",
    "CODE_DIFF_DAYS_cohort": "D1 to D2 gap (days)",
    "n_pos_test_estimate": "Est. test positives",
    "n_neg_test_estimate": "Est. test negatives",
    "log_RR": "Log relative risk (edge)",
}


@st.cache_data
def feature_frame(filename: str, reliable_only: bool) -> pd.DataFrame:
    """Cohort table with the chapter encodings added, optionally restricted to
    the reliable cohorts. Used by both the UMAP and the clustering, so the two
    always see the same rows in the same order.

    D1type_enc / D2type_enc are a plain alphabetical LabelEncoder over the
    chapter names - verified to match the notebook's values exactly. Built from
    the full table, so a chapter keeps the same code when filtering.
    """
    data = load_cohorts(filename).copy()

    chapters = sorted(set(data["D1type"]) | set(data["D2type"]))
    enc = {c: i for i, c in enumerate(chapters)}
    data["D1type_enc"] = data["D1type"].map(enc)
    data["D2type_enc"] = data["D2type"].map(enc)

    if reliable_only:
        data = data[data["reliable"]]

    return data.reset_index(drop=True)


@st.cache_data(show_spinner="Clustering…")
def cluster_representatives(filename: str, features: tuple, k: int,
                            reliable_only: bool) -> pd.DataFrame:
    """K-means over the chosen features, then the medoid of each cluster.

    Same procedure as initial_analysis.ipynb: a cluster's representative is the
    cohort closest to that cluster's centre.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import pairwise_distances

    data = feature_frame(filename, reliable_only)
    X = StandardScaler().fit_transform(data[list(features)])

    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(X)

    rep_indices = []
    for c in range(k):
        idx = np.where(labels == c)[0]
        center = km.cluster_centers_[c].reshape(1, -1)
        dists = pairwise_distances(X[idx], center).ravel()
        rep_indices.append(idx[np.argmin(dists)])

    out = data[["cohort_name"]].copy()
    out["cluster"] = labels
    out["is_representative"] = False
    out.loc[rep_indices, "is_representative"] = True
    return out


@st.cache_data(show_spinner="Fitting UMAP (~20 s the first time, then cached)…",
               persist="disk")
def umap_coords(filename: str, features: tuple, n_neighbors: int,
                min_dist: float, reliable_only: bool) -> pd.DataFrame:
    """2-D UMAP embedding of the chosen cohort features.

    Cached to disk because a fit takes ~17 s - far too slow to repeat on every
    widget click. The cache key is every argument, so each combination is
    computed once, ever. `features` is a tuple rather than a list so it is
    hashable and order-stable.

    `reliable_only` filters BEFORE the fit, so the embedding is built from the
    reliable cohorts alone. Filtering afterwards would instead just hide points
    from a layout that the excluded cohorts had already shaped.
    """
    # Imported here rather than at the top: `import umap` alone costs ~3.5 s, and
    # it is only needed when this function actually runs.
    import umap
    from sklearn.preprocessing import StandardScaler

    data = feature_frame(filename, reliable_only)
    X = StandardScaler().fit_transform(data[list(features)])
    coords = umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                       min_dist=min_dist, random_state=42).fit_transform(X)

    return pd.DataFrame({"cohort_name": data["cohort_name"].values,
                         "U1": coords[:, 0], "U2": coords[:, 1]})


umap_features = st.multiselect(
    "Features fed into UMAP",
    UMAP_CANDIDATES,
    default=UMAP_FEATURES,
    format_func=lambda c: UMAP_LABELS.get(c, c),
    help="All cohort-specific, computed on the extracted patients. "
         "Every change refits the embedding (~20 s), then caches.",
)

c1, c2, c3, c4 = st.columns([1.3, 1, 1, 1])
umap_color = c1.selectbox(
    "Colour by",
    ["log_n_cohort", "target_rate", "log_RR", "log_imb", "n_cohort",
     "female_rate_cohort", "death_rate_cohort", "AGE_AT_DISEASE_cohort",
     "CODE_DIFF_DAYS_cohort", "D1type", "reliable", "is_representative"],
    format_func=lambda c: UMAP_LABELS.get(c, c),
)
k_clusters = c1.slider(
    "K (clusters)", 10, 200, 100, step=10,
    help="K-means on the features above; the cohort nearest each cluster centre "
         "becomes that cluster's representative.",
)
n_neighbors = c2.slider("n_neighbors", 5, 100, 15, step=5,
                        help="Changing this refits the UMAP (~20 s), then caches.")
min_dist = c3.slider("min_dist", 0.0, 0.99, 0.1, step=0.05,
                     help="Changing this refits the UMAP (~20 s), then caches.")
with c4:
    st.write("")
    umap_reliable_only = st.checkbox(
        "Reliable only",
        help="Embed only the reliable cohorts. The filter is applied before the "
             "fit, so this is a different embedding - not a zoom of the full one.",
    )

if len(umap_features) < 2:
    st.warning("Pick at least two features to embed.")
    st.stop()

# sorted() so {a, b} and {b, a} hit the same cache entry instead of refitting.
emb = umap_coords(COHORT_FILES[choice], tuple(sorted(umap_features)),
                  n_neighbors, min_dist, umap_reliable_only)
umap_df = emb.merge(df, on="cohort_name", how="left")

# Cluster labels + medoid representatives, on the same features and rows.
clusters = cluster_representatives(COHORT_FILES[choice],
                                   tuple(sorted(umap_features)),
                                   k_clusters, umap_reliable_only)
umap_df = umap_df.merge(clusters, on="cohort_name", how="left")

theme_note = f"{len(umap_df):,} cohorts embedded"
if umap_reliable_only:
    theme_note += " (reliable only)"
st.caption(theme_note + f" · {len(umap_features)} features · "
                        f"n_neighbors={n_neighbors}, min_dist={min_dist}")

hover = dict(D1type=True, D2type=True, n_cohort=":,", target_rate=":.1%",
             reliable=True, U1=False, U2=False)

if umap_color == "D1type":
    fig5 = px.scatter(
        umap_df, x="U1", y="U2", color="D1type",
        color_discrete_map=palette_dict, category_orders={"D1type": hue_order},
        opacity=0.5, render_mode="webgl",
        hover_name="cohort_name", hover_data=hover,
    )
    fig5.update_layout(legend=dict(title="D1 Group", itemsizing="constant"))
elif umap_color == "is_representative":
    fig5 = px.scatter(
        umap_df, x="U1", y="U2", color="is_representative",
        color_discrete_map={True: "#2a78d6", False: "#a9a7a1"},
        category_orders={"is_representative": [False, True]},
        opacity=0.5, render_mode="webgl",
        hover_name="cohort_name", hover_data=hover,
    )
    fig5.update_layout(legend=dict(title="representative", itemsizing="constant"))
elif umap_color == "reliable":
    fig5 = px.scatter(
        umap_df, x="U1", y="U2", color="reliable",
        color_discrete_map={True: "#2E7D32", False: "#B23B3B"},
        category_orders={"reliable": [True, False]},
        opacity=0.5, render_mode="webgl",
        hover_name="cohort_name", hover_data=hover,
    )
    fig5.update_layout(legend=dict(title="reliable", itemsizing="constant"))
else:
    # Continuous feature: one perceptual ramp plus a colourbar, as in the notebook.
    fig5 = px.scatter(
        umap_df, x="U1", y="U2", color=umap_color,
        color_continuous_scale="viridis",
        opacity=0.5, render_mode="webgl",
        hover_name="cohort_name", hover_data=hover,
    )
    fig5.update_layout(coloraxis_colorbar=dict(
        title=dict(text=FEAT_LABELS.get(umap_color, umap_color), side="right"),
        thickness=12, len=0.75, outlinewidth=0,
    ))

fig5.update_traces(marker=dict(size=5, line=dict(width=0)))

# 100 representatives among ~4,600 points vanish at the default size, so give
# that trace bigger, opaque markers. px names boolean traces "True"/"False",
# and category_orders already draws True last, i.e. on top.
if umap_color == "is_representative":
    fig5.update_traces(
        selector=dict(name="True"),
        marker=dict(size=10, opacity=1.0,
                    line=dict(width=1.2, color="#fcfcfb")),
    )
    fig5.update_traces(selector=dict(name="False"), marker=dict(opacity=0.3))

fig5.update_layout(
    title=f"{choice} — coloured by {FEAT_LABELS.get(umap_color, umap_color)}",
    xaxis_title="UMAP 1",
    yaxis_title="UMAP 2",
    width=760,
    height=520,
    margin=dict(l=60, r=20, t=45, b=50),
    font=dict(size=12),
)

st.plotly_chart(fig5, width="content")

# Same two outputs the notebook wrote to disk, offered as downloads instead so
# nothing in saved_data/ gets overwritten by a slider nudge.
reps = clusters[clusters["is_representative"]]

# The selection depends entirely on the feature space, K and the reliable filter
# - changing the two demographic features alone swaps 92 of 100 representatives.
# So every download carries the config that produced it, and the filenames carry
# a short hash of it so two selections can never be confused for each other.
selection_config = {
    "cohort_file": COHORT_FILES[choice],
    "features": sorted(umap_features),
    "K": int(k_clusters),
    "reliable_only": bool(umap_reliable_only),
}
config_hash = hashlib.md5(
    json.dumps(selection_config, sort_keys=True).encode()
).hexdigest()[:6]

selection_meta = {
    **selection_config,
    "config_hash": config_hash,
    "universe": choice,
    "n_features": len(umap_features),
    "n_cohorts_clustered": int(len(clusters)),
    "n_representatives": int(len(reps)),
    "method": "KMeans(n_init=20, random_state=42) on StandardScaler'd features; "
              "representative = medoid (cohort nearest each cluster centre)",
    "generated": datetime.datetime.now().isoformat(timespec="seconds"),
}

stem = f"representative_cohorts_K{k_clusters}_{config_hash}"
st.caption(f"Selection `{config_hash}` · {len(umap_features)} features · K={k_clusters}"
           f"{' · reliable only' if umap_reliable_only else ''}")

# Its own folder, so nothing the pipeline reads can be overwritten. Filenames
# carry K and the config hash, so two selections never collide.
SELECTIONS_DIR = PROJECT_ROOT / "saved_data" / "rep_selections"

if st.button(f"Save {len(reps)} representatives to folder", type="primary"):
    SELECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    reps["cohort_name"].to_csv(SELECTIONS_DIR / f"{stem}.txt",
                               index=False, header=False)
    (SELECTIONS_DIR / f"{stem}.json").write_text(
        json.dumps(selection_meta, indent=2))
    clusters.to_csv(SELECTIONS_DIR / f"cohorts_clusters_K{k_clusters}_{config_hash}.csv",
                    index=False)
    st.success(f"Saved to `{SELECTIONS_DIR}/` as `{stem}.*`")

with st.expander("Selection provenance"):
    st.json(selection_meta)


# --------------------------------------------------------------------------- #
# Extract lab features for one cohort
# --------------------------------------------------------------------------- #

st.subheader("Extract features for one cohort")


def cohort_csv(cohort_name: str) -> Path | None:
    """Resolve one cohort's CSV, or None if it does not exist."""
    candidate = COHORTS_DIR / f"{cohort_name}.csv.gz"
    return candidate if candidate.exists() else None


def features_dir_for(days: int, first_adm_only: bool) -> Path:
    return SAVED_DATA / (f"features_{days}d" + ("_firstadm" if first_adm_only else ""))


@st.cache_data(show_spinner=False)
def extract_cohort_features(cohort_path: str, cohort_name: str, days: int,
                            first_adm_only: bool) -> pd.DataFrame:
    """Run the real FeatureExtractor for one cohort and return what it wrote.

    Same call the pipeline makes, with top_features_path=None - i.e. the
    `--feature-selection False` the DTB job uses, so the output matches
    saved_data/features_<days>d/ rather than being a filtered variant of it.

    Cached on the arguments because Streamlit re-runs the script on every widget
    click and a button press does not survive the next run. A repeat click
    therefore returns the cached frame instead of re-extracting; the file on
    disk is already written, so nothing is lost.

    The caller must check labs.parquet exists first: FeatureExtractor builds it
    in __init__ when missing, which is the hours-long pass over labevents.csv.gz.
    """
    from config.constants import get_data_path
    from flab_features.feature_extractor import FeatureExtractor

    features_dir = features_dir_for(days, first_adm_only)
    extractor = FeatureExtractor(
        mimic_dir=get_data_path("MIMIC_IV"),
        features_base_path=features_dir,
        top_features_path=None,
        days_before_discharge=days,
        first_adm_only=first_adm_only,
    )
    extractor.extract(pd.read_csv(cohort_path, compression="gzip"), cohort_name)

    written = features_dir / cohort_name / "features.csv.gz"
    # extract() returns early without writing when no labs fall in the window.
    return pd.read_csv(written) if written.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_features(path: str, mtime: float) -> pd.DataFrame:
    """Read an already-extracted features.csv.gz. `mtime` is part of the cache
    key so a re-extraction is picked up instead of serving the stale frame."""
    return pd.read_csv(path)


def plot_subject_labs(feats: pd.DataFrame, cohort_name: str, days: int):
    """Lab trajectories for one patient, on days before discharge."""
    key = f"{cohort_name}_{days}"
    subjects = sorted(feats["subject_id"].unique())

    p1, p2 = st.columns([1, 3])
    subject = p1.selectbox(f"Subject ({len(subjects):,} in cohort)", subjects,
                           key=f"subject_{key}")

    one = feats[feats["subject_id"] == subject]
    counts = one["itemid"].value_counts()
    items = p2.multiselect(
        "Lab itemids", counts.index.tolist(), default=counts.index[:5].tolist(),
        format_func=lambda i: f"{i} ({counts[i]} values)", key=f"itemids_{key}",
        help="Ordered by how often each lab was measured for this patient.",
    )
    if not items:
        st.info("Pick at least one itemid to plot.")
        return

    one = one[one["itemid"].isin(items)].sort_values("minute")

    fig = px.line(
        one.astype({"itemid": str}).assign(day=one["minute"] / 1440 - days),
        x="day", y="value", color="itemid", markers=True,
    )
    fig.update_layout(
        title=f"{cohort_name} — subject {subject}",
        xaxis_title="Days before discharge (0 = discharge)",
        yaxis_title="Value",
        xaxis=dict(range=[-days, 1]),
        width=760,
        height=420,
        margin=dict(l=60, r=20, t=45, b=50),
        font=dict(size=12),
        legend=dict(title="itemid", itemsizing="constant"),
    )
    st.plotly_chart(fig, width="content")
    st.caption(f"{len(one):,} measurements · units differ per itemid, so the "
               "y axis is shared but not comparable across labs")


def show_features(feats: pd.DataFrame, out_file: Path, cohort_name: str, days: int):
    """Summary, per-subject lab plot and download for one extracted features table."""
    if feats.empty:
        st.warning(
            f"No labs for **{cohort_name}** in the {days}-day window before "
            "discharge, so nothing was written."
        )
        return

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Lab measurements", f"{len(feats):,}")
    m2.metric("Admissions", f"{feats['hadm_id'].nunique():,}")
    m3.metric("Distinct itemids", f"{feats['itemid'].nunique():,}")
    m4.metric("File size", f"{out_file.stat().st_size / 1e6:.1f} MB")

    plot_subject_labs(feats, cohort_name, days)

    st.download_button(
        "Download features.csv.gz",
        data=out_file.read_bytes(),
        file_name=f"{cohort_name}_features_{days}d.csv.gz",
        mime="application/gzip",
    )


all_cohorts = sorted(df["cohort_name"].dropna().unique())

e1, e2, e3 = st.columns([2, 1, 1])
cohort_to_extract = e1.selectbox(
    "Cohort", all_cohorts,
    help="Type to search. Lists the cohorts in the table selected above.",
)
days_before_discharge = e2.number_input(
    "Days before discharge", min_value=1, max_value=365, value=14, step=1,
    help="Labs are taken from this many days before discharge up to discharge. "
         "Writes to saved_data/features_<days>d/.",
)
with e3:
    st.write("")
    first_adm_only = st.checkbox(
        "First admissions only",
        help="Keep each patient's earliest admission only. Writes to "
             "features_<days>d_firstadm/, so it never overwrites the "
             "all-admissions features.",
    )

days_before_discharge = int(days_before_discharge)
features_dir = features_dir_for(days_before_discharge, first_adm_only)
labs_parquet = SAVED_DATA / "labs.parquet"
out_file = features_dir / cohort_to_extract / "features.csv.gz"
src_csv = cohort_csv(cohort_to_extract)

if src_csv is None:
    st.error(f"No cohort file for **{cohort_to_extract}** in `{COHORTS_DIR}`.")
    st.stop()

st.caption(f"Source `{src_csv.relative_to(PROJECT_ROOT)}` → target "
           f"`{out_file.relative_to(PROJECT_ROOT)}`")

if not labs_parquet.exists():
    # FeatureExtractor.__init__ would build it here: a full pass over the
    # 2.5 GB labevents.csv.gz that the batch job asks 200 GB of memory for.
    # Not something to trigger from a dashboard on a login node.
    st.warning(
        "`labs.parquet` does not exist yet. Extracting would first build it "
        "from the 2.5 GB `labevents.csv.gz` - hours of work, and the reason "
        "the batch job requests 200 GB."
    )
    st.stop()

if out_file.exists():
    stamp = datetime.datetime.fromtimestamp(out_file.stat().st_mtime)
    st.info(f"Already extracted on {stamp:%Y-%m-%d %H:%M}. Extracting again "
            "overwrites it.")
    button_label = f"Re-extract {cohort_to_extract} ({features_dir.name})"
else:
    button_label = f"Extract {cohort_to_extract} ({features_dir.name})"

if st.button(button_label, type="primary"):
    with st.spinner(f"Reading labs for {cohort_to_extract}…"):
        feats = extract_cohort_features(
            str(src_csv), cohort_to_extract, days_before_discharge, first_adm_only)
    st.success(f"Wrote `{out_file.relative_to(PROJECT_ROOT)}`")
    show_features(feats, out_file, cohort_to_extract, days_before_discharge)
elif out_file.exists():
    show_features(load_features(str(out_file), out_file.stat().st_mtime),
                  out_file, cohort_to_extract, days_before_discharge)

with st.expander("Run this as a batch job instead"):
    st.write("For many cohorts, or a `days` value whose labs.parquet still has "
             "to be built, submit it rather than waiting on the dashboard:")
    st.code(
        f"python -m flab_features.extract_features \\\n"
        f"    --extractor DTB --cohort {cohort_to_extract} \\\n"
        f"    --days {days_before_discharge} --feature-selection False"
        + (" \\\n    --first-adm-only" if first_adm_only else ""),
        language="bash",
    )
    st.caption("Use `--cohort all` for every cohort. See "
               "jobs/job_features_dtb.sh for the batch version.")


st.subheader("The raw table")

