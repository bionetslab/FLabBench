"""Palette tokens and plotly styling for the cohort dashboard.

Colors are taken verbatim from the validated reference palette. Only the first
three categorical slots are used for real multi-series charts, because those are
the slots certified for all-pairs (scatter/dot) use in both light and dark mode.
Anything with more than three groups is encoded by position + facet, or by a
single-hue sequential ramp, never by a fourth-plus hue.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# Categorical slots 1-3 (all-pairs certified), plus a neutral for "de-emphasised".
LIGHT = {
    "mode": "light",
    "surface": "#fcfcfb",
    "surface_alt": "#f2f1ee",
    "ink": "#0b0b0b",
    "ink_secondary": "#52514e",
    "ink_muted": "#7b7874",
    "grid": "#e8e7e3",
    "axis": "#d5d3ce",
    "series": ("#2a78d6", "#eb6834", "#1baf7a"),
    "neutral": "#a9a7a1",
    "reference": "#c7c2bc",
    # Sequential blue, light -> dark (magnitude increases away from the surface).
    "seq": (
        "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
        "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
    ),
    "status": {
        "good": "#0ca30c",
        "warning": "#fab219",
        "serious": "#ec835a",
        "critical": "#d03b3b",
    },
    "diverging": ("#2a78d6", "#f0efec", "#d03b3b"),
}

DARK = {
    "mode": "dark",
    "surface": "#1a1a19",
    "surface_alt": "#242423",
    "ink": "#ffffff",
    "ink_secondary": "#c3c2b7",
    "ink_muted": "#8f8e85",
    "grid": "#2e2e2c",
    "axis": "#3d3d3a",
    "series": ("#3987e5", "#d95926", "#199e70"),
    "neutral": "#6b6a64",
    "reference": "#4a4a46",
    # Same ramp, reversed: on a dark surface magnitude must increase toward light.
    "seq": (
        "#0d366b", "#104281", "#184f95", "#1c5cab", "#256abf", "#2a78d6",
        "#3987e5", "#5598e7", "#6da7ec", "#86b6ef", "#9ec5f4", "#b7d3f6", "#cde2fb",
    ),
    "status": {
        "good": "#0ca30c",
        "warning": "#fab219",
        "serious": "#ec835a",
        "critical": "#d03b3b",
    },
    "diverging": ("#3987e5", "#383835", "#d03b3b"),
}

FONT = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"


def palette() -> dict:
    """Palette matching the viewer's current Streamlit theme."""
    try:
        return DARK if st.context.theme.type == "dark" else LIGHT
    except Exception:
        return LIGHT


def register_template(p: dict) -> str:
    """Register (once per palette) a plotly template and return its name."""
    name = f"flab_{p['mode']}"
    if name in pio.templates:
        return name

    axis = dict(
        gridcolor=p["grid"],
        griddash="solid",
        linecolor=p["axis"],
        linewidth=1,
        zeroline=False,
        ticks="outside",
        ticklen=4,
        tickcolor=p["axis"],
        tickfont=dict(size=12, color=p["ink_secondary"]),
        title_font=dict(size=13, color=p["ink_secondary"]),
        automargin=True,
    )

    pio.templates[name] = go.layout.Template(
        layout=dict(
            paper_bgcolor=p["surface"],
            plot_bgcolor=p["surface"],
            colorway=list(p["series"]),
            colorscale=dict(sequential=[[i / (len(p["seq"]) - 1), c]
                                        for i, c in enumerate(p["seq"])]),
            font=dict(family=FONT, size=13, color=p["ink_secondary"]),
            title=dict(font=dict(family=FONT, size=16, color=p["ink"]), x=0, xanchor="left"),
            xaxis=axis,
            yaxis=axis,
            legend=dict(
                bgcolor="rgba(0,0,0,0)",
                borderwidth=0,
                font=dict(size=12, color=p["ink_secondary"]),
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
            ),
            hoverlabel=dict(
                bgcolor=p["surface_alt"],
                bordercolor=p["axis"],
                font=dict(family=FONT, size=12, color=p["ink"]),
            ),
            margin=dict(l=8, r=8, t=56, b=8),
            boxgap=0.45,
            boxgroupgap=0.12,
            bargap=0.28,
        )
    )
    return name


def seq_scale(p: dict) -> list:
    """Sequential colorscale in plotly's [[pos, color], ...] form."""
    n = len(p["seq"]) - 1
    return [[i / n, c] for i, c in enumerate(p["seq"])]


def styled(fig: go.Figure, p: dict, height: int | None = None, **layout) -> go.Figure:
    fig.update_layout(template=register_template(p), **layout)
    if height is not None:
        fig.update_layout(height=height)
    return fig


def inject_css(p: dict) -> None:
    """Small amount of chrome so metric tiles and tables sit on the same surface."""
    st.markdown(
        f"""
        <style>
          .stMetric {{
              background: {p['surface_alt']};
              border-radius: 10px;
              padding: 0.7rem 0.9rem;
          }}
          .stMetric label p {{
              font-size: 0.78rem !important;
              color: {p['ink_muted']} !important;
              letter-spacing: 0.02em;
          }}
          div[data-testid="stMetricValue"] {{
              font-size: 1.45rem;
              color: {p['ink']};
          }}
          .flab-note {{
              color: {p['ink_muted']};
              font-size: 0.82rem;
              line-height: 1.5;
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def note(text: str) -> None:
    st.markdown(f'<p class="flab-note">{text}</p>', unsafe_allow_html=True)
