from __future__ import annotations

import html
from io import BytesIO
import math
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pydeck as pdk
import streamlit as st
from pyproj import Transformer
from typing import Any

from ags_app.atterberg import build_atterberg_table
from ags_app.bre import build_bre_sulphate_table, calculate_bre_summary, classify_acec, sulphate_class
from ags_app.geolmodel import build_geological_model
from ags_app.groundwater import build_groundwater_table
from ags_app.mapviewer import build_geology_intervals, build_map_locations
from ags_app.parser import parse_uploaded_file
from ags_app.ivan import build_ivan_table
from ags_app.pointload import build_pointload_table
from ags_app.psd import build_psd_table
from ags_app.rqd import build_rqd_table
from ags_app.spt import build_spt_table
from ags_app.ucs import build_ucs_table


st.set_page_config(page_title="AGS Geotechnical Analysis", layout="wide")
SCIENTIFIC_PALETTE = [
    "#0F4C81",
    "#A23E48",
    "#4C956C",
    "#E0A458",
    "#6B6D76",
    "#2D6A8A",
]
DEFAULT_POINT_COLOR = SCIENTIFIC_PALETTE[0]
DESIGN_LINE_COLOR = "#1F1F1F"
DESIGN_LINE_OPTIONS = [
    "Off",
    "Lower cautious estimate",
    "Upper cautious estimate",
    "Mean trend",
]
UI_NAVY = "#002b5b"
UI_WHITE = "#ffffff"
UI_PANEL = "#f6f7f9"
UI_LINE = "#e0e0e0"
UI_INK = "#2b2d42"
UI_RED = "#d90429"


@st.cache_data(show_spinner=False)
def load_analysis_data(file_name: str, content: bytes):
    parsed = parse_uploaded_file(file_name, content)
    if not parsed.tables:
        raise ValueError(
            "No AGS groups were detected. Check the file is a text AGS transfer file or an AGS-style Excel export."
        )
    spt, spt_error = build_optional_table(parsed.tables, build_spt_table)
    ivan, ivan_error = build_optional_table(parsed.tables, build_ivan_table)
    ucs, ucs_error = build_optional_table(parsed.tables, build_ucs_table)
    rqd, rqd_error = build_optional_table(parsed.tables, build_rqd_table)
    atterberg, atterberg_error = build_optional_table(parsed.tables, build_atterberg_table)
    pointload, pointload_error = build_optional_table(parsed.tables, build_pointload_table)
    psd, psd_error = build_optional_table(parsed.tables, build_psd_table)
    groundwater, groundwater_error = build_optional_table(parsed.tables, build_groundwater_table)
    return (
        parsed,
        spt,
        ivan,
        ucs,
        rqd,
        atterberg,
        pointload,
        psd,
        groundwater,
        spt_error,
        ivan_error,
        ucs_error,
        rqd_error,
        atterberg_error,
        pointload_error,
        psd_error,
        groundwater_error,
    )


def main() -> None:
    inject_custom_css()
    st.session_state.setdefault("screen", "home")

    if st.session_state["screen"] == "spt":
        render_spt_screen()
    elif st.session_state["screen"] == "ivan":
        render_ivan_screen()
    elif st.session_state["screen"] == "ucs":
        render_ucs_screen()
    elif st.session_state["screen"] == "rqd":
        render_rqd_screen()
    elif st.session_state["screen"] == "atterberg":
        render_atterberg_screen()
    elif st.session_state["screen"] == "pointload":
        render_pointload_screen()
    elif st.session_state["screen"] == "psd":
        render_psd_screen()
    elif st.session_state["screen"] == "groundwater":
        render_groundwater_screen()
    elif st.session_state["screen"] == "map":
        render_map_screen()
    elif st.session_state["screen"] == "geological_model":
        render_geological_model_screen()
    elif st.session_state["screen"] == "summary_stats":
        render_summary_stats_screen()
    elif st.session_state["screen"] == "bre_sulphate":
        render_bre_sulphate_screen()
    else:
        render_home_screen()


def inject_custom_css() -> None:
    st.markdown(
        f"""
        <style>
        :root {{
            --ags-navy: {UI_NAVY};
            --ags-white: {UI_WHITE};
            --ags-panel: {UI_PANEL};
            --ags-line: {UI_LINE};
            --ags-ink: {UI_INK};
            --ags-red: {UI_RED};
        }}
        .stApp {{
            background:
                linear-gradient(135deg, rgba(0, 43, 91, 0.045), rgba(255, 255, 255, 0) 36%),
                linear-gradient(180deg, #ffffff 0%, #fafafa 100%);
            color: var(--ags-ink);
        }}
        .block-container {{
            max-width: 1180px;
            padding-top: 2.2rem;
            padding-bottom: 3rem;
        }}
        h1, h2, h3 {{
            color: var(--ags-ink);
            letter-spacing: 0;
        }}
        .ags-hero {{
            border: 1px solid var(--ags-line);
            border-left: 7px solid var(--ags-navy);
            border-radius: 8px;
            padding: 1.35rem 1.45rem;
            background: rgba(255, 255, 255, 0.92);
            box-shadow: 0 10px 28px rgba(43, 45, 66, 0.08);
            margin-bottom: 1.25rem;
        }}
        .ags-kicker {{
            margin: 0 0 0.4rem 0;
            color: var(--ags-red);
            font-size: 0.78rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }}
        .ags-title {{
            margin: 0;
            font-size: clamp(2.1rem, 4vw, 4rem);
            line-height: 1.02;
            font-weight: 850;
            color: var(--ags-navy);
        }}
        .ags-subtitle {{
            margin: 0.7rem 0 0 0;
            color: #4f5264;
            max-width: 780px;
            font-size: 1.02rem;
        }}
        .ags-status {{
            border: 1px solid var(--ags-line);
            border-radius: 8px;
            padding: 0.85rem 1rem;
            background: var(--ags-panel);
            display: flex;
            gap: 0.75rem;
            align-items: center;
            margin: 1rem 0 1.25rem 0;
        }}
        .ags-status-dot {{
            width: 0.75rem;
            height: 0.75rem;
            border-radius: 999px;
            background: var(--ags-red);
            box-shadow: 0 0 0 4px rgba(217, 4, 41, 0.12);
            flex: 0 0 auto;
        }}
        .ags-status-title {{
            margin: 0;
            color: var(--ags-ink);
            font-weight: 800;
            line-height: 1.15;
        }}
        .ags-status-path {{
            margin: 0.15rem 0 0 0;
            color: #6d7080;
            font-size: 0.88rem;
            overflow-wrap: anywhere;
        }}
        .ags-section-title {{
            color: var(--ags-navy);
            font-weight: 850;
            margin: 1.6rem 0 0.2rem 0;
            font-size: 1.45rem;
        }}
        .ags-section-copy {{
            color: #606372;
            margin: 0 0 0.9rem 0;
        }}
        .ags-card {{
            border: 1px solid var(--ags-line);
            border-top: 5px solid var(--ags-navy);
            border-radius: 8px;
            padding: 1rem 1rem 0.85rem 1rem;
            background: #ffffff;
            min-height: 142px;
            box-shadow: 0 8px 20px rgba(43, 45, 66, 0.06);
            margin-bottom: 0.55rem;
        }}
        .ags-card.accent {{
            border-top-color: var(--ags-red);
        }}
        .ags-card h3 {{
            margin: 0;
            font-size: 1.02rem;
            color: var(--ags-ink);
            font-weight: 850;
        }}
        .ags-card p {{
            margin: 0.45rem 0 0 0;
            color: #606372;
            font-size: 0.88rem;
            line-height: 1.35;
        }}
        .ags-count {{
            display: inline-block;
            margin-top: 0.75rem;
            padding: 0.2rem 0.5rem;
            border-radius: 999px;
            background: rgba(0, 43, 91, 0.08);
            color: var(--ags-navy);
            font-size: 0.78rem;
            font-weight: 800;
        }}
        div.stButton > button {{
            border-radius: 8px;
            border: 1px solid var(--ags-navy);
            background: var(--ags-navy);
            color: #ffffff;
            font-weight: 800;
            min-height: 2.55rem;
            box-shadow: none;
        }}
        div.stButton > button:hover {{
            border-color: var(--ags-red);
            background: var(--ags-red);
            color: #ffffff;
        }}
        div.stButton > button:disabled {{
            background: #d8d8d8;
            border-color: #d8d8d8;
            color: #777777;
        }}
        div[data-testid="stFileUploader"] section {{
            background: var(--ags-panel);
            border: 1px solid var(--ags-line);
            border-radius: 8px;
        }}
        div[data-testid="stExpander"] {{
            border-color: var(--ags-line);
            border-radius: 8px;
            background: #ffffff;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_home_screen() -> None:
    st.markdown(
        """
        <div class="ags-hero">
            <p class="ags-kicker">AGS data toolkit</p>
            <h1 class="ags-title">AGS Geotechnical Analysis</h1>
            <p class="ags-subtitle">Upload AGS data, inspect geotechnical tests, filter by geology, export scientific plots, and review mapped ground conditions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Upload AGS data",
        type=["ags", "csv", "txt", "xlsx"],
        accept_multiple_files=False,
    )

    local_path = render_local_file_loader(uploaded is None)

    if uploaded is not None:
        st.session_state["ags_source_name"] = uploaded.name
        st.session_state["ags_content"] = uploaded.getvalue()
    elif local_path is not None:
        st.session_state["ags_source_name"] = str(local_path)
        st.session_state["ags_content"] = local_path.read_bytes()

    if "ags_content" not in st.session_state:
        st.info("Upload an AGS file to begin. This first version also accepts AGS-style Excel exports.")
        return

    (
        parsed,
        spt,
        ivan,
        ucs,
        rqd,
        atterberg,
        pointload,
        psd,
        groundwater,
        spt_error,
        ivan_error,
        ucs_error,
        rqd_error,
        atterberg_error,
        pointload_error,
        psd_error,
        groundwater_error,
    ) = load_current_analysis_data()
    if (
        parsed is None
        or spt is None
        or ivan is None
        or ucs is None
        or rqd is None
        or atterberg is None
        or pointload is None
        or psd is None
        or groundwater is None
    ):
        return

    source_name = html.escape(str(parsed.source_name))
    st.markdown(
        f"""
        <div class="ags-status">
            <span class="ags-status-dot"></span>
            <div>
                <p class="ags-status-title">AGS data loaded</p>
                <p class="ags-status-path">{source_name}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if spt_error:
        st.warning(f"SPT module unavailable: {spt_error}")
    if ivan_error:
        st.warning(f"Hand Shear Vane module unavailable: {ivan_error}")
    if ucs_error:
        st.warning(f"UCS module unavailable: {ucs_error}")
    if rqd_error:
        st.warning(f"RQD module unavailable: {rqd_error}")
    if atterberg_error:
        st.warning(f"Atterberg Limits module unavailable: {atterberg_error}")
    if pointload_error:
        st.warning(f"Point Load Strength module unavailable: {pointload_error}")
    if psd_error:
        st.warning(f"Particle Size Distribution module unavailable: {psd_error}")
    if groundwater_error:
        st.warning(f"Groundwater Strike module unavailable: {groundwater_error}")
    bre_sulphate, bre_sulphate_error = build_optional_table(parsed.tables, build_bre_sulphate_table)
    if bre_sulphate_error:
        st.warning(f"BRE Sulphate Class module unavailable: {bre_sulphate_error}")

    st.markdown(
        """
        <h2 class="ags-section-title">Analysis modules</h2>
        <p class="ags-section-copy">Choose a module to inspect filtered data, plots, exports, and geology matching.</p>
        """,
        unsafe_allow_html=True,
    )
    modules = [
        {
            "screen": "spt",
            "title": "Standard Penetration Tests",
            "description": "SPT blow count against depth with geology filtering and design lines.",
            "count": f"{len(spt)} records",
            "disabled": spt.empty,
            "accent": False,
        },
        {
            "screen": "ivan",
            "title": "Hand Shear Vane",
            "description": "Hand vane readings by investigation or geological unit.",
            "count": f"{len(ivan)} records",
            "disabled": ivan.empty,
            "accent": False,
        },
        {
            "screen": "ucs",
            "title": "UCS",
            "description": "Unconfined compressive strength plots and matched source rows.",
            "count": f"{len(ucs)} records",
            "disabled": ucs.empty,
            "accent": True,
        },
        {
            "screen": "rqd",
            "title": "RQD",
            "description": "Rock quality designation against core run top depth.",
            "count": f"{len(rqd)} records",
            "disabled": rqd.empty,
            "accent": False,
        },
        {
            "screen": "atterberg",
            "title": "Atterberg Limits",
            "description": "Liquid limit, plastic limit, and plasticity index graphs.",
            "count": f"{len(atterberg)} records",
            "disabled": atterberg.empty,
            "accent": True,
        },
        {
            "screen": "pointload",
            "title": "Point Load Strength",
            "description": "Point load strength index by depth and geology.",
            "count": f"{len(pointload)} records",
            "disabled": pointload.empty,
            "accent": False,
        },
        {
            "screen": "psd",
            "title": "Particle Size Distribution",
            "description": "PSD curves with curve selection and statistical design curve.",
            "count": f"{psd['PSD_SAMPLE_ID'].nunique() if 'PSD_SAMPLE_ID' in psd.columns else 0} curves",
            "disabled": psd.empty,
            "accent": False,
        },
        {
            "screen": "groundwater",
            "title": "Groundwater Strike",
            "description": "Strike depths, post-strike readings, and geology match.",
            "count": f"{len(groundwater)} strikes",
            "disabled": groundwater.empty,
            "accent": True,
        },
        {
            "screen": "map",
            "title": "Map Viewer",
            "description": "Investigation map with selected geology ranges under each borehole.",
            "count": f"{len(table_ids(parsed.get('LOCA'), 'LOCA_ID')) or 0} locations",
            "disabled": False,
            "accent": False,
        },
        {
            "screen": "geological_model",
            "title": "Geological Model",
            "description": "Classify strata by unit, material type, model unit, and bedrock lithology.",
            "count": f"{len(parsed.get('GEOL')) if 'GEOL' in parsed.tables else 0} intervals",
            "disabled": "GEOL" not in parsed.tables,
            "accent": True,
        },
        {
            "screen": "summary_stats",
            "title": "Summary Stats",
            "description": "Mean and cautious estimates for filtered geotechnical test results.",
            "count": "statistical table",
            "disabled": all(
                table.empty
                for table in [spt, ivan, ucs, rqd, atterberg, pointload, psd, groundwater]
            ),
            "accent": False,
        },
        {
            "screen": "bre_sulphate",
            "title": "BRE Sulphate Class",
            "description": "Classify DS and ACEC classes from GCHM sulphate and pH chemistry.",
            "count": f"{len(bre_sulphate)} samples",
            "disabled": bre_sulphate.empty,
            "accent": True,
        },
    ]
    render_module_grid(modules)


def render_module_grid(modules: list[dict[str, object]]) -> None:
    for row_start in range(0, len(modules), 3):
        columns = st.columns(3)
        for column, module in zip(columns, modules[row_start : row_start + 3]):
            with column:
                render_module_card(module)


def render_module_card(module: dict[str, object]) -> None:
    card_class = "ags-card accent" if module["accent"] else "ags-card"
    title = html.escape(str(module["title"]))
    description = html.escape(str(module["description"]))
    count = html.escape(str(module["count"]))
    st.markdown(
        f"""
        <div class="{card_class}">
            <h3>{title}</h3>
            <p>{description}</p>
            <span class="ags-count">{count}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Open", key=f"open_{module['screen']}", disabled=bool(module["disabled"]), use_container_width=True):
        st.session_state["screen"] = str(module["screen"])
        st.rerun()


def render_spt_screen() -> None:
    header_col, action_col = st.columns([1, 0.18])
    with header_col:
        st.title("Standard Penetration Tests")
    with action_col:
        if st.button("Back"):
            st.session_state["screen"] = "home"
            st.rerun()

    if "ags_content" not in st.session_state:
        st.warning("Load AGS data before opening the SPT module.")
        if st.button("Go to upload"):
            st.session_state["screen"] = "home"
            st.rerun()
        return

    analysis = load_current_analysis_data()
    parsed, spt, spt_error = analysis[0], analysis[1], analysis[9]
    if parsed is None or spt is None:
        return
    if spt_error or spt.empty:
        st.warning(spt_error or "No valid SPT rows found after reading LOCA_ID, ISPT_TOP, and ISPT_MAIN.")
        return

    st.caption(f"Loaded {parsed.source_name}")

    render_spt_module(parsed, spt)


def render_ivan_screen() -> None:
    header_col, action_col = st.columns([1, 0.18])
    with header_col:
        st.title("Hand Shear Vane")
    with action_col:
        if st.button("Back"):
            st.session_state["screen"] = "home"
            st.rerun()

    if "ags_content" not in st.session_state:
        st.warning("Load AGS data before opening the Hand Shear Vane module.")
        if st.button("Go to upload"):
            st.session_state["screen"] = "home"
            st.rerun()
        return

    analysis = load_current_analysis_data()
    parsed, ivan, ivan_error = analysis[0], analysis[2], analysis[10]
    if parsed is None or ivan is None:
        return
    if ivan_error or ivan.empty:
        st.warning(ivan_error or "No valid hand shear vane rows found after reading LOCA_ID, IVAN_DPTH, and IVAN_IVAN.")
        return

    st.caption(f"Loaded {parsed.source_name}")

    render_ivan_module(parsed, ivan)


def render_ucs_screen() -> None:
    header_col, action_col = st.columns([1, 0.18])
    with header_col:
        st.title("Unconfined Compressive Strength")
    with action_col:
        if st.button("Back"):
            st.session_state["screen"] = "home"
            st.rerun()

    if "ags_content" not in st.session_state:
        st.warning("Load AGS data before opening the UCS module.")
        if st.button("Go to upload"):
            st.session_state["screen"] = "home"
            st.rerun()
        return

    analysis = load_current_analysis_data()
    parsed, ucs, ucs_error = analysis[0], analysis[3], analysis[11]
    if parsed is None or ucs is None:
        return
    if ucs_error or ucs.empty:
        st.warning(ucs_error or "No valid UCS rows found after reading LOCA_ID, SAMP_TOP, and RUCS_UCS.")
        return

    st.caption(f"Loaded {parsed.source_name}")

    render_ucs_module(parsed, ucs)


def render_rqd_screen() -> None:
    header_col, action_col = st.columns([1, 0.18])
    with header_col:
        st.title("Rock Quality Designation")
    with action_col:
        if st.button("Back"):
            st.session_state["screen"] = "home"
            st.rerun()

    if "ags_content" not in st.session_state:
        st.warning("Load AGS data before opening the RQD module.")
        if st.button("Go to upload"):
            st.session_state["screen"] = "home"
            st.rerun()
        return

    analysis = load_current_analysis_data()
    parsed, rqd, rqd_error = analysis[0], analysis[4], analysis[12]
    if parsed is None or rqd is None:
        return
    if rqd_error or rqd.empty:
        st.warning(rqd_error or "No valid RQD rows found after reading LOCA_ID, CORE_TOP, and CORE_RQD.")
        return

    st.caption(f"Loaded {parsed.source_name}")

    render_rqd_module(parsed, rqd)


def render_atterberg_screen() -> None:
    header_col, action_col = st.columns([1, 0.18])
    with header_col:
        st.title("Atterberg Limits")
    with action_col:
        if st.button("Back"):
            st.session_state["screen"] = "home"
            st.rerun()

    if "ags_content" not in st.session_state:
        st.warning("Load AGS data before opening the Atterberg Limits module.")
        if st.button("Go to upload"):
            st.session_state["screen"] = "home"
            st.rerun()
        return

    analysis = load_current_analysis_data()
    parsed, atterberg, atterberg_error = analysis[0], analysis[5], analysis[13]
    if parsed is None or atterberg is None:
        return
    if atterberg_error or atterberg.empty:
        st.warning(atterberg_error or "No valid Atterberg rows found after reading LOCA_ID, SAMP_TOP, LLPL_LL, and LLPL_PL.")
        return

    st.caption(f"Loaded {parsed.source_name}")

    render_atterberg_module(parsed, atterberg)


def render_pointload_screen() -> None:
    header_col, action_col = st.columns([1, 0.18])
    with header_col:
        st.title("Point Load Strength")
    with action_col:
        if st.button("Back"):
            st.session_state["screen"] = "home"
            st.rerun()

    if "ags_content" not in st.session_state:
        st.warning("Load AGS data before opening the Point Load Strength module.")
        if st.button("Go to upload"):
            st.session_state["screen"] = "home"
            st.rerun()
        return

    analysis = load_current_analysis_data()
    parsed, pointload, pointload_error = analysis[0], analysis[6], analysis[14]
    if parsed is None or pointload is None:
        return
    if pointload_error or pointload.empty:
        st.warning(pointload_error or "No valid point load rows found after reading LOCA_ID, depth, and RPLT_PLSI.")
        return

    st.caption(f"Loaded {parsed.source_name}")

    render_pointload_module(parsed, pointload)


def render_psd_screen() -> None:
    header_col, action_col = st.columns([1, 0.18])
    with header_col:
        st.title("Particle Size Distribution")
    with action_col:
        if st.button("Back"):
            st.session_state["screen"] = "home"
            st.rerun()

    if "ags_content" not in st.session_state:
        st.warning("Load AGS data before opening the Particle Size Distribution module.")
        if st.button("Go to upload"):
            st.session_state["screen"] = "home"
            st.rerun()
        return

    analysis = load_current_analysis_data()
    parsed, psd, psd_error = analysis[0], analysis[7], analysis[15]
    if parsed is None or psd is None:
        return
    if psd_error or psd.empty:
        st.warning(psd_error or "No valid PSD rows found after reading LOCA_ID, SAMP_TOP, GRAT_SIZE, and GRAT_PERP.")
        return

    st.caption(f"Loaded {parsed.source_name}")

    render_psd_module(parsed, psd)


def render_groundwater_screen() -> None:
    header_col, action_col = st.columns([1, 0.18])
    with header_col:
        st.title("Groundwater Strike")
    with action_col:
        if st.button("Back"):
            st.session_state["screen"] = "home"
            st.rerun()

    if "ags_content" not in st.session_state:
        st.warning("Load AGS data before opening the Groundwater Strike module.")
        if st.button("Go to upload"):
            st.session_state["screen"] = "home"
            st.rerun()
        return

    analysis = load_current_analysis_data()
    parsed, groundwater, groundwater_error = analysis[0], analysis[8], analysis[16]
    if parsed is None or groundwater is None:
        return
    if groundwater_error or groundwater.empty:
        st.warning(groundwater_error or "No valid groundwater strike rows found after reading LOCA_ID and WSTG_DPTH.")
        return

    st.caption(f"Loaded {parsed.source_name}")

    render_groundwater_module(parsed, groundwater)


def render_map_screen() -> None:
    header_col, action_col = st.columns([1, 0.18])
    with header_col:
        st.title("Map Viewer")
    with action_col:
        if st.button("Back"):
            st.session_state["screen"] = "home"
            st.rerun()

    if "ags_content" not in st.session_state:
        st.warning("Load AGS data before opening the Map Viewer.")
        if st.button("Go to upload"):
            st.session_state["screen"] = "home"
            st.rerun()
        return

    analysis = load_current_analysis_data()
    parsed = analysis[0]
    if parsed is None:
        return

    try:
        locations, x_column, y_column, x_label, y_label = build_map_locations(parsed.tables)
        geology = build_geology_intervals(parsed.tables)
    except ValueError as exc:
        st.warning(str(exc))
        return

    st.caption(f"Loaded {parsed.source_name}")
    render_map_module(locations, geology, x_column, y_column, x_label, y_label)


def render_geological_model_screen() -> None:
    header_col, action_col = st.columns([1, 0.18])
    with header_col:
        st.title("Geological Model")
    with action_col:
        if st.button("Back"):
            st.session_state["screen"] = "home"
            st.rerun()

    if "ags_content" not in st.session_state:
        st.warning("Load AGS data before opening the Geological Model module.")
        if st.button("Go to upload"):
            st.session_state["screen"] = "home"
            st.rerun()
        return

    analysis = load_current_analysis_data()
    parsed = analysis[0]
    if parsed is None:
        return

    try:
        model = build_geological_model(parsed.tables)
    except ValueError as exc:
        st.warning(str(exc))
        return

    st.caption(f"Loaded {parsed.source_name}")
    render_geological_model_module(model)


def render_summary_stats_screen() -> None:
    header_col, action_col = st.columns([1, 0.18])
    with header_col:
        st.title("Summary Stats")
    with action_col:
        if st.button("Back"):
            st.session_state["screen"] = "home"
            st.rerun()

    if "ags_content" not in st.session_state:
        st.warning("Load AGS data before opening the Summary Stats module.")
        if st.button("Go to upload"):
            st.session_state["screen"] = "home"
            st.rerun()
        return

    analysis = load_current_analysis_data()
    parsed = analysis[0]
    if parsed is None:
        return

    st.caption(f"Loaded {parsed.source_name}")
    render_summary_stats_module(
        {
            "SPT": analysis[1],
            "Hand Shear Vane": analysis[2],
            "UCS": analysis[3],
            "RQD": analysis[4],
            "Atterberg Limits": analysis[5],
            "Point Load Strength": analysis[6],
            "Particle Size Distribution": analysis[7],
            "Groundwater Strike": analysis[8],
        }
    )


def render_bre_sulphate_screen() -> None:
    header_col, action_col = st.columns([1, 0.18])
    with header_col:
        st.title("BRE Sulphate Class")
    with action_col:
        if st.button("Back"):
            st.session_state["screen"] = "home"
            st.rerun()

    if "ags_content" not in st.session_state:
        st.warning("Load AGS data before opening the BRE Sulphate Class module.")
        if st.button("Go to upload"):
            st.session_state["screen"] = "home"
            st.rerun()
        return

    analysis = load_current_analysis_data()
    parsed = analysis[0]
    if parsed is None:
        return

    try:
        bre_sulphate = build_bre_sulphate_table(parsed.tables)
    except ValueError as exc:
        st.warning(str(exc))
        return

    st.caption(f"Loaded {parsed.source_name}")
    render_bre_sulphate_module(bre_sulphate)


def load_current_analysis_data() -> tuple[
    Any | None,
    pd.DataFrame | None,
    pd.DataFrame | None,
    pd.DataFrame | None,
    pd.DataFrame | None,
    pd.DataFrame | None,
    pd.DataFrame | None,
    pd.DataFrame | None,
    pd.DataFrame | None,
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
]:
    try:
        source_name = st.session_state["ags_source_name"]
        content = st.session_state["ags_content"]

        with st.spinner(f"Reading {source_name} ({content_size(content)})"):
            (
                parsed,
                spt,
                ivan,
                ucs,
                rqd,
                atterberg,
                pointload,
                psd,
                groundwater,
                spt_error,
                ivan_error,
                ucs_error,
                rqd_error,
                atterberg_error,
                pointload_error,
                psd_error,
                groundwater_error,
            ) = load_analysis_data(source_name, content)
        return (
            parsed,
            spt,
            ivan,
            ucs,
            rqd,
            atterberg,
            pointload,
            psd,
            groundwater,
            spt_error,
            ivan_error,
            ucs_error,
            rqd_error,
            atterberg_error,
            pointload_error,
            psd_error,
            groundwater_error,
        )
    except Exception as exc:
        st.error(str(exc))
        return None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None


def build_optional_table(tables: dict[str, pd.DataFrame], builder) -> tuple[pd.DataFrame, str | None]:
    try:
        return builder(tables), None
    except ValueError as exc:
        return pd.DataFrame(), str(exc)


def render_bre_sulphate_module(bre_sulphate: pd.DataFrame) -> None:
    sample_count, investigation_count, sulphate_count, ph_count, matched_count = st.columns(5)
    sample_count.metric("Chemistry samples", len(bre_sulphate))
    investigation_count.metric("Investigations", bre_sulphate["LOCA_ID"].nunique())
    sulphate_count.metric("Sulphate results", int(bre_sulphate["WS_MG_L"].notna().sum()))
    ph_count.metric("pH results", int(bre_sulphate["PH_VALUE"].notna().sum()))
    matched_count.metric("Matched to GEOL", int(bre_sulphate["GEOLOGY_MATCHED"].sum()))

    st.caption(
        "This module implements the BRE SD1 Table C1/C2 non-pyrite route using GCHM water soluble sulphate "
        "and pH results. Brownfield magnesium, chloride, and nitrate indicators are shown where present."
    )

    option_col_1, option_col_2 = st.columns(2)
    with option_col_1:
        site_type = st.radio("Location type", ["Natural", "Brownfield"], horizontal=True)
    with option_col_2:
        water_mobility = st.radio("Groundwater mobility", ["Static", "Mobile"], horizontal=True)

    selected_loca, geology_mode, selected_units, selected_materials, selected_model_units, selected_bedrock = render_filters(
        bre_sulphate,
        "BRE chemistry samples",
    )
    filtered = bre_sulphate[bre_sulphate["LOCA_ID"].isin(selected_loca)].copy()
    filtered = apply_geology_filter(
        filtered,
        selected_units,
        geology_mode,
        selected_materials,
        selected_model_units,
        selected_bedrock,
    )
    if filtered.empty:
        st.warning("No BRE chemistry samples match the current filters.")
        return

    summary = pd.DataFrame([calculate_bre_summary(filtered, site_type, water_mobility)])
    classified = add_bre_sample_classes(filtered, site_type, water_mobility)

    if site_type == "Brownfield" and classified["MG_MG_L"].notna().sum() == 0:
        st.info("No magnesium results were found in `GCHM`; brownfield `m` suffix classes cannot be triggered.")
    if site_type == "Brownfield" and classified[["CL_MG_L", "NO3_MG_L"]].notna().sum().sum() == 0:
        st.info("No chloride or nitrate results were found for the filtered records; sulfate adjustment for these acids is not applied.")

    tab_summary, tab_samples = st.tabs(["Classification Summary", "Matched Chemistry"])
    with tab_summary:
        st.dataframe(summary, use_container_width=True, hide_index=True)
        st.caption(
            "The summary uses BRE characteristic values across the currently filtered records. "
            "For soil sulphate, small datasets use the maximum result; larger datasets use the highest results as described in SD1."
        )
        st.download_button(
            "Download BRE summary CSV",
            data=summary.to_csv(index=False).encode("utf-8"),
            file_name="bre_sulphate_summary.csv",
            mime="text/csv",
        )

    with tab_samples:
        columns = [
            "LOCA_ID",
            "BRE_DEPTH",
            "BRE_SAMPLE_TYPE",
            "WS_MG_L",
            "PH_VALUE",
            "AS_PERCENT",
            "TS_PERCENT",
            "TPS_PERCENT",
            "OS_PERCENT",
            "CL_MG_L",
            "NO3_MG_L",
            "MG_MG_L",
            "SAMPLE_DS_CLASS",
            "SAMPLE_ACEC_CLASS",
            "GEOL_GEOL",
            "MATERIAL_CLASS",
            "MODEL_UNIT",
            "GEOL_TOP",
            "GEOL_BASE",
            "GEOL_DESC",
            "GEOLOGY_MATCHED",
        ]
        st.dataframe(
            classified[[column for column in columns if column in classified.columns]],
            use_container_width=True,
            hide_index=True,
        )

    render_custom_bre_groups(
        bre_sulphate,
        selected_loca,
        selected_materials,
        selected_bedrock,
        site_type,
        water_mobility,
    )


def add_bre_sample_classes(data: pd.DataFrame, site_type: str, water_mobility: str) -> pd.DataFrame:
    classified = data.copy()
    sample_ds_classes: list[str | None] = []
    sample_acec_classes: list[str | None] = []
    for _, row in classified.iterrows():
        sample_type = "groundwater" if row.get("BRE_SAMPLE_TYPE") == "Groundwater" else "soil"
        ds_class = sulphate_class(row.get("WS_MG_L"), sample_type)
        if site_type == "Brownfield":
            magnesium = row.get("MG_MG_L")
            if not pd.isna(magnesium) and ds_class in {"DS-4", "DS-5"}:
                ds_class = f"{ds_class}m" if float(magnesium) > (1000 if sample_type == "groundwater" else 1200) else ds_class
        acec_class = classify_acec(
            ds_class,
            None if pd.isna(row.get("PH_VALUE")) else float(row.get("PH_VALUE")),
            site_type,
            water_mobility,
        )
        sample_ds_classes.append(ds_class)
        sample_acec_classes.append(acec_class)
    classified["SAMPLE_DS_CLASS"] = sample_ds_classes
    classified["SAMPLE_ACEC_CLASS"] = sample_acec_classes
    return classified


def render_custom_bre_groups(
    data: pd.DataFrame,
    selected_loca: list[str],
    selected_materials: list[str],
    selected_bedrock: list[str],
    site_type: str,
    water_mobility: str,
) -> None:
    st.subheader("Custom Grouped BRE Classification")
    st.caption(
        "Create a master list of combined and separate geology/model groups for DS and ACEC classifications. "
        "Use one group per line, for example `Alluvium = ALV, ALV(G)`."
    )
    group_fields = available_group_fields(data)
    if not group_fields:
        st.info("No geology or model fields are available for custom BRE grouping.")
        return
    group_field = st.selectbox("BRE group by field", group_fields, key="bre_group_field")
    available_values = sorted(data[group_field].dropna().astype(str).unique()) if group_field in data.columns else []
    with st.expander("Available BRE group values"):
        st.write(", ".join(available_values) if available_values else "No values available.")

    group_text = st.text_area(
        "BRE custom groups",
        value=default_custom_group_text(group_field, available_values),
        height=150,
        key="bre_group_text",
    )
    groups, errors = parse_custom_groups(group_text)
    for error in errors:
        st.warning(error)
    if not groups:
        st.info("Add at least one group line to create grouped BRE classifications.")
        return

    base = data[data["LOCA_ID"].isin(selected_loca)].copy()
    if selected_materials and "MATERIAL_CLASS" in base.columns:
        base = base[base["MATERIAL_CLASS"].isin(selected_materials)].copy()
    if selected_bedrock and "BEDROCK_TYPE" in base.columns:
        base = base[base["BEDROCK_TYPE"].isin(selected_bedrock)].copy()

    rows = build_custom_bre_group_rows(base, group_field, groups, site_type, water_mobility)
    if not rows:
        st.warning("No BRE chemistry samples matched the custom group definitions.")
        return

    grouped = pd.DataFrame(rows)
    st.dataframe(grouped, use_container_width=True, hide_index=True)
    st.download_button(
        "Download grouped BRE CSV",
        data=grouped.to_csv(index=False).encode("utf-8"),
        file_name="bre_custom_group_summary.csv",
        mime="text/csv",
    )


def build_custom_bre_group_rows(
    data: pd.DataFrame,
    group_field: str,
    groups: list[dict[str, object]],
    site_type: str,
    water_mobility: str,
) -> list[dict[str, object]]:
    if group_field not in data.columns:
        return []

    rows: list[dict[str, object]] = []
    for group in groups:
        members = [str(member) for member in group["members"]]
        group_data = data[data[group_field].astype(str).isin(members)].copy()
        if group_data.empty:
            continue
        rows.append(
            {
                "Group": group["name"],
                "Group Field": group_field,
                "Group Members": ", ".join(members),
                **calculate_bre_summary(group_data, site_type, water_mobility),
            }
        )
    return rows


def render_summary_stats_module(module_tables: dict[str, pd.DataFrame | None]) -> None:
    available = {
        name: table
        for name, table in module_tables.items()
        if table is not None and not table.empty and summary_parameter_definitions(name)
    }
    if not available:
        st.warning("No valid test data is available for summary statistics.")
        return

    selected_module = st.selectbox("Test module", list(available), index=0)
    data = available[selected_module].copy()

    selected_loca, geology_mode, selected_units, selected_materials, selected_model_units, selected_bedrock = render_filters(
        data,
        f"{selected_module.lower()} records",
    )
    filtered = data[data["LOCA_ID"].isin(selected_loca)].copy()
    filtered = apply_geology_filter(
        filtered,
        selected_units,
        geology_mode,
        selected_materials,
        selected_model_units,
        selected_bedrock,
    )

    rows = build_summary_stat_rows(selected_module, filtered)
    if not rows:
        st.warning("No numeric records match the current filters.")
        return

    summary = pd.DataFrame(rows)
    st.subheader("Filtered Summary")
    st.dataframe(summary, use_container_width=True, hide_index=True)
    st.caption(
        "Cautious estimates are calculated from all filtered values as one population. "
        "They are not depth trend estimates."
    )

    csv_bytes = summary.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download summary CSV",
        data=csv_bytes,
        file_name=f"{slugify(selected_module)}_summary_stats.csv",
        mime="text/csv",
    )

    with st.expander("Filtered records used in the summary"):
        display_columns = summary_source_columns(selected_module, filtered)
        st.dataframe(filtered[display_columns], use_container_width=True, hide_index=True)

    render_custom_summary_groups(selected_module, data, selected_loca, selected_materials, selected_bedrock)


def summary_parameter_definitions(module_name: str) -> list[dict[str, str]]:
    definitions = {
        "SPT": [
            {"label": "Raw SPT N", "column": "ISPT_MAIN_NUM", "unit": "blows", "side": "lower"},
            {"label": "Corrected SPT N60", "column": "ISPT_N60_NUM", "unit": "blows", "side": "lower"},
        ],
        "Hand Shear Vane": [
            {"label": "Hand shear vane", "column": "IVAN_IVAN_NUM", "unit": "", "side": "lower"},
        ],
        "UCS": [
            {"label": "UCS", "column": "RUCS_UCS_NUM", "unit": "", "side": "lower"},
        ],
        "RQD": [
            {"label": "RQD", "column": "CORE_RQD_NUM", "unit": "%", "side": "lower"},
        ],
        "Atterberg Limits": [
            {"label": "Liquid limit LL", "column": "LLPL_LL_NUM", "unit": "%", "side": "lower"},
            {"label": "Plastic limit PL", "column": "LLPL_PL_NUM", "unit": "%", "side": "lower"},
            {"label": "Plasticity index PI", "column": "LLPL_PI_NUM", "unit": "%", "side": "lower"},
        ],
        "Point Load Strength": [
            {"label": "Point load Is50", "column": "RPLT_PLSI_NUM", "unit": "", "side": "lower"},
        ],
        "Groundwater Strike": [
            {"label": "Strike depth", "column": "WSTG_DPTH_NUM", "unit": "m bgl", "side": "upper"},
            {"label": "Post-strike reading", "column": "WSTD_POST_NUM", "unit": "m bgl", "side": "upper"},
        ],
        "Particle Size Distribution": [
            {"label": "D10", "column": "PSD_D10_NUM", "unit": "mm", "side": "lower"},
            {"label": "D30", "column": "PSD_D30_NUM", "unit": "mm", "side": "lower"},
            {"label": "D60", "column": "PSD_D60_NUM", "unit": "mm", "side": "lower"},
        ],
    }
    return definitions.get(module_name, [])


def build_summary_stat_rows(module_name: str, data: pd.DataFrame) -> list[dict[str, object]]:
    summary_data = build_psd_summary_values(data) if module_name == "Particle Size Distribution" else data
    rows: list[dict[str, object]] = []
    for parameter in summary_parameter_definitions(module_name):
        column = parameter["column"]
        if column not in summary_data.columns:
            continue
        stats = calculate_scalar_summary(summary_data[column], parameter["side"])
        if stats is None:
            continue
        rows.append(
            {
                "Module": module_name,
                "Parameter": parameter["label"],
                "Unit": parameter["unit"],
                "Records": stats["count"],
                "Investigations": int(summary_data.loc[summary_data[column].notna(), "LOCA_ID"].nunique())
                if "LOCA_ID" in summary_data.columns
                else pd.NA,
                "Mean": stats["mean"],
                "Std Dev": stats["std_dev"],
                "Lower 95% Estimate": stats["lower_95"],
                "Upper 95% Estimate": stats["upper_95"],
                "Cautious Estimate": stats["cautious"],
                "Cautious Side": stats["side"],
            }
        )
    return rows


def render_custom_summary_groups(
    module_name: str,
    data: pd.DataFrame,
    selected_loca: list[str],
    selected_materials: list[str],
    selected_bedrock: list[str],
) -> None:
    st.subheader("Custom Grouped Summary")
    st.caption(
        "Create a master list of combined and separate geology/model groups. "
        "Use one group per line, for example `Alluvium = ALV, ALV(G)`."
    )
    group_fields = available_group_fields(data)
    if not group_fields:
        st.info("No geology or model fields are available for custom grouping.")
        return
    group_field = st.selectbox(
        "Group by field",
        group_fields,
        key=f"{slugify(module_name)}_summary_group_field",
    )
    available_values = sorted(data[group_field].dropna().astype(str).unique()) if group_field in data.columns else []
    with st.expander("Available group values"):
        st.write(", ".join(available_values) if available_values else "No values available.")

    group_text = st.text_area(
        "Custom groups",
        value=default_custom_group_text(group_field, available_values),
        height=150,
        key=f"{slugify(module_name)}_summary_group_text",
    )
    groups, errors = parse_custom_groups(group_text)
    for error in errors:
        st.warning(error)
    if not groups:
        st.info("Add at least one group line to create a grouped summary.")
        return

    base = data[data["LOCA_ID"].isin(selected_loca)].copy()
    if selected_materials and "MATERIAL_CLASS" in base.columns:
        base = base[base["MATERIAL_CLASS"].isin(selected_materials)].copy()
    if selected_bedrock and "BEDROCK_TYPE" in base.columns:
        base = base[base["BEDROCK_TYPE"].isin(selected_bedrock)].copy()

    grouped_rows = build_custom_group_summary_rows(module_name, base, group_field, groups)
    if not grouped_rows:
        st.warning("No records matched the custom group definitions.")
        return

    grouped_summary = pd.DataFrame(grouped_rows)
    st.dataframe(grouped_summary, use_container_width=True, hide_index=True)
    st.download_button(
        "Download grouped summary CSV",
        data=grouped_summary.to_csv(index=False).encode("utf-8"),
        file_name=f"{slugify(module_name)}_custom_group_summary.csv",
        mime="text/csv",
    )


def available_group_fields(data: pd.DataFrame) -> list[str]:
    candidates = ["GEOL_GEOL", "MODEL_UNIT", "MATERIAL_CLASS", "BEDROCK_TYPE"]
    return [column for column in candidates if column in data.columns and data[column].notna().any()]


def default_custom_group_text(group_field: str, values: list[str]) -> str:
    if group_field == "GEOL_GEOL":
        lines = []
        if "ALV" in values or "ALV(G)" in values:
            members = [value for value in ["ALV", "ALV(G)"] if value in values]
            lines.append(f"Alluvium = {', '.join(members)}")
        if "GT" in values:
            lines.append("Glacial Till = GT")
        return "\n".join(lines)
    return ""


def parse_custom_groups(text: str) -> tuple[list[dict[str, object]], list[str]]:
    groups: list[dict[str, object]] = []
    errors: list[str] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if "=" not in line:
            errors.append(f"Line {line_number} is ignored because it has no `=` separator.")
            continue
        name, raw_members = line.split("=", 1)
        members = [member.strip() for member in raw_members.split(",") if member.strip()]
        if not name.strip() or not members:
            errors.append(f"Line {line_number} is ignored because it needs a group name and at least one value.")
            continue
        groups.append({"name": name.strip(), "members": members})
    return groups, errors


def build_custom_group_summary_rows(
    module_name: str,
    data: pd.DataFrame,
    group_field: str,
    groups: list[dict[str, object]],
) -> list[dict[str, object]]:
    if group_field not in data.columns:
        return []

    rows: list[dict[str, object]] = []
    for group in groups:
        members = [str(member) for member in group["members"]]
        group_data = data[data[group_field].astype(str).isin(members)].copy()
        for row in build_summary_stat_rows(module_name, group_data):
            rows.append(
                {
                    "Group": group["name"],
                    "Group Field": group_field,
                    "Group Members": ", ".join(members),
                    **row,
                }
            )
    return rows


def calculate_scalar_summary(values: pd.Series, cautious_side: str = "lower") -> dict[str, object] | None:
    clean = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if clean.empty:
        return None

    count = int(len(clean))
    mean_value = float(clean.mean())
    if count == 1:
        std_dev = pd.NA
        lower = mean_value
        upper = mean_value
    else:
        std_dev_value = float(clean.std(ddof=1))
        standard_error = std_dev_value / math.sqrt(count)
        width = t_critical_one_sided_95(count - 1) * standard_error
        lower = mean_value - width
        upper = mean_value + width
        std_dev = std_dev_value

    cautious = upper if cautious_side == "upper" else lower
    return {
        "count": count,
        "mean": round(mean_value, 3),
        "std_dev": pd.NA if pd.isna(std_dev) else round(float(std_dev), 3),
        "lower_95": round(lower, 3),
        "upper_95": round(upper, 3),
        "cautious": round(cautious, 3),
        "side": "Upper" if cautious_side == "upper" else "Lower",
    }


def build_psd_summary_values(data: pd.DataFrame) -> pd.DataFrame:
    required = {"PSD_SAMPLE_ID", "GRAT_SIZE_NUM", "GRAT_PERP_NUM"}
    if data.empty or not required.issubset(data.columns):
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for curve_id, group in data.dropna(subset=["PSD_SAMPLE_ID"]).groupby("PSD_SAMPLE_ID"):
        sorted_group = group.dropna(subset=["GRAT_SIZE_NUM", "GRAT_PERP_NUM"]).sort_values("GRAT_PERP_NUM")
        if sorted_group.empty:
            continue
        row = sorted_group.iloc[0].to_dict()
        row["PSD_SAMPLE_ID"] = curve_id
        for percent in (10, 30, 60):
            row[f"PSD_D{percent}_NUM"] = interpolate_psd_d_value(sorted_group, percent)
        rows.append(row)
    return pd.DataFrame(rows)


def interpolate_psd_d_value(curve: pd.DataFrame, percent_passing: float) -> float | None:
    points = curve[["GRAT_SIZE_NUM", "GRAT_PERP_NUM"]].dropna().drop_duplicates("GRAT_PERP_NUM")
    points = points.sort_values("GRAT_PERP_NUM")
    if points.empty:
        return None

    passing = points["GRAT_PERP_NUM"].astype(float).tolist()
    sizes = points["GRAT_SIZE_NUM"].astype(float).tolist()
    if percent_passing < min(passing) or percent_passing > max(passing):
        return None
    for index, passing_value in enumerate(passing):
        if math.isclose(percent_passing, passing_value):
            return sizes[index]
        if index == 0:
            continue
        lower_passing = passing[index - 1]
        upper_passing = passing_value
        if lower_passing <= percent_passing <= upper_passing:
            lower_size = sizes[index - 1]
            upper_size = sizes[index]
            if lower_size <= 0 or upper_size <= 0 or math.isclose(lower_passing, upper_passing):
                return lower_size
            fraction = (percent_passing - lower_passing) / (upper_passing - lower_passing)
            log_size = math.log10(lower_size) + fraction * (math.log10(upper_size) - math.log10(lower_size))
            return 10**log_size
    return None


def summary_source_columns(module_name: str, data: pd.DataFrame) -> list[str]:
    base_columns = {
        "SPT": ["LOCA_ID", "ISPT_TOP", "ISPT_MAIN", "ISPT_ERAT", "ISPT_MAIN_NUM", "ISPT_N60_NUM"],
        "Hand Shear Vane": ["LOCA_ID", "IVAN_DPTH", "IVAN_IVAN", "IVAN_IVAN_NUM"],
        "UCS": ["LOCA_ID", "SAMP_TOP", "RUCS_UCS", "RUCS_UCS_NUM"],
        "RQD": ["LOCA_ID", "CORE_TOP", "CORE_RQD", "CORE_RQD_NUM"],
        "Atterberg Limits": ["LOCA_ID", "SAMP_TOP", "LLPL_LL", "LLPL_PL", "LLPL_PI"],
        "Point Load Strength": ["LOCA_ID", "SAMP_TOP", "SPEC_DPTH", "RPLT_PLSI", "RPLT_PLSI_NUM"],
        "Particle Size Distribution": ["LOCA_ID", "PSD_SAMPLE_ID", "SAMP_TOP", "GRAT_SIZE", "GRAT_PERP"],
        "Groundwater Strike": ["LOCA_ID", "WSTG_DPTH", "WSTD_POST", "WSTG_DPTH_NUM", "WSTD_POST_NUM"],
    }
    columns = base_columns.get(module_name, ["LOCA_ID"])
    columns.extend(["GEOL_GEOL", "MATERIAL_CLASS", "MODEL_UNIT", "BEDROCK_TYPE", "GEOL_DESC"])
    return [column for column in dict.fromkeys(columns) if column in data.columns]


def render_spt_module(parsed, spt: pd.DataFrame) -> None:
    loca_ids_all = table_ids(parsed.get("LOCA"), "LOCA_ID")
    spt_loca_ids = table_ids(spt, "LOCA_ID")
    loca_without_spt = [loca_id for loca_id in loca_ids_all if loca_id not in set(spt_loca_ids)]

    group_count, loca_count, spt_loca_count, spt_count, matched_count = st.columns(5)
    group_count.metric("AGS groups", len(parsed.tables))
    loca_count.metric("LOCA investigations", len(loca_ids_all) or "n/a")
    spt_loca_count.metric("SPT investigations", len(spt_loca_ids))
    spt_count.metric("SPT records", len(spt))
    matched_count.metric("Matched to GEOL", int(spt["GEOLOGY_MATCHED"].sum()))

    st.caption(
        "Graphs use investigations with valid `ISPT_TOP` and `ISPT_MAIN` records. "
        "`LOCA` can include trial pits or locations with no SPT data, so the plotted investigation count may be lower."
    )

    if loca_without_spt:
        with st.expander(f"{len(loca_without_spt)} LOCA investigations have no plottable SPT records"):
            st.dataframe(
                pd.DataFrame({"LOCA_ID": loca_without_spt}),
                use_container_width=True,
                hide_index=True,
            )

    if spt.empty:
        st.warning("No valid SPT rows found after reading LOCA_ID, ISPT_TOP, and ISPT_MAIN.")
        return

    value_mode = st.radio(
        "SPT value",
        ["Raw N", "Corrected N60"],
        horizontal=True,
        help="Corrected N60 is calculated as ISPT_MAIN x ISPT_ERAT / 60.",
    )
    if value_mode == "Corrected N60":
        spt_value_column = "ISPT_N60_NUM"
        spt_value_label = "Corrected SPT N60 (ISPT_MAIN x ISPT_ERAT / 60)"
        spt_title_value = "Corrected SPT N60"
        corrected_count = int(spt[spt_value_column].notna().sum()) if spt_value_column in spt.columns else 0
        if corrected_count == 0:
            st.warning("No corrected N60 values are available because `ISPT_ERAT` is missing or non-numeric.")
    else:
        spt_value_column = "ISPT_MAIN_NUM"
        spt_value_label = "SPT blow count, N (ISPT_MAIN)"
        spt_title_value = "SPT Blow Count"

    selected_loca, geology_mode, selected_units, selected_materials, selected_model_units, selected_bedrock = render_filters(spt, "SPT records")

    filtered = spt[spt["LOCA_ID"].isin(selected_loca)].copy()
    filtered_by_unit = apply_geology_filter(filtered, selected_units, geology_mode, selected_materials, selected_model_units, selected_bedrock)

    tab_all, tab_units, tab_data = st.tabs(
        ["All Investigations", "Geological Units", "Matched Data"]
    )

    with tab_all:
        render_spt_plot(
            filtered_by_unit,
            title=f"{spt_title_value} vs Depth by Investigation",
            color_by="LOCA_ID",
            x_column=spt_value_column,
            x_label=spt_value_label,
        )

    with tab_units:
        render_spt_plot(
            filtered_by_unit,
            title=f"{spt_title_value} vs Depth by Geological Unit",
            color_by="GEOL_GEOL",
            x_column=spt_value_column,
            x_label=spt_value_label,
        )

    with tab_data:
        columns = [
            "LOCA_ID",
            "ISPT_TOP",
            "ISPT_MAIN",
            "ISPT_ERAT",
            "ISPT_TOP_NUM",
            "ISPT_MAIN_NUM",
            "ISPT_ERAT_NUM",
            "ISPT_N60_NUM",
            "GEOL_GEOL",
            "GEOL_TOP",
            "GEOL_BASE",
            "GEOL_DESC",
            "GEOLOGY_MATCHED",
        ]
        st.dataframe(
            filtered_by_unit[matched_data_columns(filtered_by_unit, columns)],
            use_container_width=True,
            hide_index=True,
        )


def render_ivan_module(parsed, ivan: pd.DataFrame) -> None:
    loca_ids_all = table_ids(parsed.get("LOCA"), "LOCA_ID")
    ivan_loca_ids = table_ids(ivan, "LOCA_ID")
    loca_without_ivan = [loca_id for loca_id in loca_ids_all if loca_id not in set(ivan_loca_ids)]

    group_count, loca_count, ivan_loca_count, ivan_count, matched_count = st.columns(5)
    group_count.metric("AGS groups", len(parsed.tables))
    loca_count.metric("LOCA investigations", len(loca_ids_all) or "n/a")
    ivan_loca_count.metric("Hand vane investigations", len(ivan_loca_ids))
    ivan_count.metric("Hand vane records", len(ivan))
    matched_count.metric("Matched to GEOL", int(ivan["GEOLOGY_MATCHED"].sum()))

    st.caption(
        "Graphs use investigations with valid `IVAN_DPTH` and `IVAN_IVAN` records. "
        "`LOCA` can include locations with no hand shear vane data, so the plotted investigation count may be lower."
    )

    if loca_without_ivan:
        with st.expander(f"{len(loca_without_ivan)} LOCA investigations have no plottable hand shear vane records"):
            st.dataframe(
                pd.DataFrame({"LOCA_ID": loca_without_ivan}),
                use_container_width=True,
                hide_index=True,
            )

    selected_loca, geology_mode, selected_units, selected_materials, selected_model_units, selected_bedrock = render_filters(ivan, "hand shear vane records")

    filtered = ivan[ivan["LOCA_ID"].isin(selected_loca)].copy()
    filtered_by_unit = apply_geology_filter(filtered, selected_units, geology_mode, selected_materials, selected_model_units, selected_bedrock)

    tab_all, tab_units, tab_data = st.tabs(
        ["All Investigations", "Geological Units", "Matched Data"]
    )

    with tab_all:
        render_ivan_plot(
            filtered_by_unit,
            title="Hand Shear Vane vs Depth by Investigation",
            color_by="LOCA_ID",
        )

    with tab_units:
        render_ivan_plot(
            filtered_by_unit,
            title="Hand Shear Vane vs Depth by Geological Unit",
            color_by="GEOL_GEOL",
        )

    with tab_data:
        columns = [
            "LOCA_ID",
            "IVAN_DPTH",
            "IVAN_IVAN",
            "IVAN_DPTH_NUM",
            "IVAN_IVAN_NUM",
            "GEOL_GEOL",
            "GEOL_TOP",
            "GEOL_BASE",
            "GEOL_DESC",
            "GEOLOGY_MATCHED",
        ]
        st.dataframe(
            filtered_by_unit[matched_data_columns(filtered_by_unit, columns)],
            use_container_width=True,
            hide_index=True,
        )


def render_ucs_module(parsed, ucs: pd.DataFrame) -> None:
    loca_ids_all = table_ids(parsed.get("LOCA"), "LOCA_ID")
    ucs_loca_ids = table_ids(ucs, "LOCA_ID")
    loca_without_ucs = [loca_id for loca_id in loca_ids_all if loca_id not in set(ucs_loca_ids)]

    group_count, loca_count, ucs_loca_count, ucs_count, matched_count = st.columns(5)
    group_count.metric("AGS groups", len(parsed.tables))
    loca_count.metric("LOCA investigations", len(loca_ids_all) or "n/a")
    ucs_loca_count.metric("UCS investigations", len(ucs_loca_ids))
    ucs_count.metric("UCS records", len(ucs))
    matched_count.metric("Matched to GEOL", int(ucs["GEOLOGY_MATCHED"].sum()))

    st.caption(
        "Graphs use investigations with valid `SAMP_TOP` and `RUCS_UCS` records from the `RUCS` group. "
        "`LOCA` can include locations with no UCS data, so the plotted investigation count may be lower."
    )

    if loca_without_ucs:
        with st.expander(f"{len(loca_without_ucs)} LOCA investigations have no plottable UCS records"):
            st.dataframe(
                pd.DataFrame({"LOCA_ID": loca_without_ucs}),
                use_container_width=True,
                hide_index=True,
            )

    selected_loca, geology_mode, selected_units, selected_materials, selected_model_units, selected_bedrock = render_filters(ucs, "UCS records")

    filtered = ucs[ucs["LOCA_ID"].isin(selected_loca)].copy()
    filtered_by_unit = apply_geology_filter(filtered, selected_units, geology_mode, selected_materials, selected_model_units, selected_bedrock)

    tab_all, tab_units, tab_data = st.tabs(
        ["All Investigations", "Geological Units", "Matched Data"]
    )

    with tab_all:
        render_ucs_plot(
            filtered_by_unit,
            title="UCS vs Depth by Investigation",
            color_by="LOCA_ID",
        )

    with tab_units:
        render_ucs_plot(
            filtered_by_unit,
            title="UCS vs Depth by Geological Unit",
            color_by="GEOL_GEOL",
        )

    with tab_data:
        columns = [
            "LOCA_ID",
            "SAMP_TOP",
            "RUCS_UCS",
            "SAMP_TOP_NUM",
            "RUCS_UCS_NUM",
            "GEOL_GEOL",
            "GEOL_TOP",
            "GEOL_BASE",
            "GEOL_DESC",
            "GEOLOGY_MATCHED",
        ]
        st.dataframe(
            filtered_by_unit[matched_data_columns(filtered_by_unit, columns)],
            use_container_width=True,
            hide_index=True,
        )


def render_rqd_module(parsed, rqd: pd.DataFrame) -> None:
    loca_ids_all = table_ids(parsed.get("LOCA"), "LOCA_ID")
    rqd_loca_ids = table_ids(rqd, "LOCA_ID")
    loca_without_rqd = [loca_id for loca_id in loca_ids_all if loca_id not in set(rqd_loca_ids)]

    group_count, loca_count, rqd_loca_count, rqd_count, matched_count = st.columns(5)
    group_count.metric("AGS groups", len(parsed.tables))
    loca_count.metric("LOCA investigations", len(loca_ids_all) or "n/a")
    rqd_loca_count.metric("RQD investigations", len(rqd_loca_ids))
    rqd_count.metric("RQD records", len(rqd))
    matched_count.metric("Matched to GEOL", int(rqd["GEOLOGY_MATCHED"].sum()))

    st.caption(
        "Graphs use investigations with valid `CORE_TOP` and `CORE_RQD` records from the `CORE` group. "
        "`LOCA` can include locations with no core RQD data, so the plotted investigation count may be lower."
    )

    if loca_without_rqd:
        with st.expander(f"{len(loca_without_rqd)} LOCA investigations have no plottable RQD records"):
            st.dataframe(
                pd.DataFrame({"LOCA_ID": loca_without_rqd}),
                use_container_width=True,
                hide_index=True,
            )

    selected_loca, geology_mode, selected_units, selected_materials, selected_model_units, selected_bedrock = render_filters(rqd, "RQD records")

    filtered = rqd[rqd["LOCA_ID"].isin(selected_loca)].copy()
    filtered_by_unit = apply_geology_filter(filtered, selected_units, geology_mode, selected_materials, selected_model_units, selected_bedrock)

    tab_all, tab_units, tab_data = st.tabs(
        ["All Investigations", "Geological Units", "Matched Data"]
    )

    with tab_all:
        render_rqd_plot(
            filtered_by_unit,
            title="RQD vs Depth by Investigation",
            color_by="LOCA_ID",
        )

    with tab_units:
        render_rqd_plot(
            filtered_by_unit,
            title="RQD vs Depth by Geological Unit",
            color_by="GEOL_GEOL",
        )

    with tab_data:
        columns = [
            "LOCA_ID",
            "CORE_TOP",
            "CORE_BASE",
            "CORE_RQD",
            "CORE_TOP_NUM",
            "CORE_RQD_NUM",
            "GEOL_GEOL",
            "GEOL_TOP",
            "GEOL_BASE",
            "GEOL_DESC",
            "GEOLOGY_MATCHED",
        ]
        st.dataframe(
            filtered_by_unit[matched_data_columns(filtered_by_unit, columns)],
            use_container_width=True,
            hide_index=True,
        )


def render_atterberg_module(parsed, atterberg: pd.DataFrame) -> None:
    loca_ids_all = table_ids(parsed.get("LOCA"), "LOCA_ID")
    atterberg_loca_ids = table_ids(atterberg, "LOCA_ID")
    loca_without_atterberg = [loca_id for loca_id in loca_ids_all if loca_id not in set(atterberg_loca_ids)]

    (
        loca_count,
        atterberg_loca_count,
        atterberg_count,
        ll_count,
        pl_count,
        pi_count,
        matched_count,
    ) = st.columns(7)
    loca_count.metric("LOCA investigations", len(loca_ids_all) or "n/a")
    atterberg_loca_count.metric("Atterberg investigations", len(atterberg_loca_ids))
    atterberg_count.metric("Atterberg records", len(atterberg))
    ll_count.metric("LL values", int(atterberg["LLPL_LL_NUM"].notna().sum()))
    pl_count.metric("PL values", int(atterberg["LLPL_PL_NUM"].notna().sum()))
    pi_count.metric("PI values", int(atterberg["LLPL_PI_NUM"].notna().sum()))
    matched_count.metric("Matched to GEOL", int(atterberg["GEOLOGY_MATCHED"].sum()))

    st.caption(
        "Graphs use investigations with valid `SAMP_TOP` and Atterberg results from the `LLPL` group. "
        "Plasticity Index is taken from `LLPL_PI` where present and otherwise calculated as `LLPL_LL - LLPL_PL`."
    )

    if loca_without_atterberg:
        with st.expander(f"{len(loca_without_atterberg)} LOCA investigations have no plottable Atterberg records"):
            st.dataframe(
                pd.DataFrame({"LOCA_ID": loca_without_atterberg}),
                use_container_width=True,
                hide_index=True,
            )

    selected_loca, geology_mode, selected_units, selected_materials, selected_model_units, selected_bedrock = render_filters(atterberg, "Atterberg records")

    filtered = atterberg[atterberg["LOCA_ID"].isin(selected_loca)].copy()
    filtered_by_unit = apply_geology_filter(filtered, selected_units, geology_mode, selected_materials, selected_model_units, selected_bedrock)

    tab_ll, tab_pl, tab_pi, tab_data = st.tabs(
        ["Liquid Limit", "Plastic Limit", "Plasticity Index", "Matched Data"]
    )

    with tab_ll:
        render_atterberg_pair(
            filtered_by_unit,
            value_column="LLPL_LL_NUM",
            value_label="Liquid limit, LL (%) (LLPL_LL)",
            title_prefix="Liquid Limit",
        )

    with tab_pl:
        render_atterberg_pair(
            filtered_by_unit,
            value_column="LLPL_PL_NUM",
            value_label="Plastic limit, PL (%) (LLPL_PL)",
            title_prefix="Plastic Limit",
        )

    with tab_pi:
        render_atterberg_pair(
            filtered_by_unit,
            value_column="LLPL_PI_NUM",
            value_label="Plasticity index, PI (%)",
            title_prefix="Plasticity Index",
        )

    with tab_data:
        columns = [
            "LOCA_ID",
            "SAMP_TOP",
            "LLPL_LL",
            "LLPL_PL",
            "LLPL_PI",
            "LLPL_LL_NUM",
            "LLPL_PL_NUM",
            "LLPL_PI_NUM",
            "GEOL_GEOL",
            "GEOL_TOP",
            "GEOL_BASE",
            "GEOL_DESC",
            "GEOLOGY_MATCHED",
        ]
        st.dataframe(
            filtered_by_unit[matched_data_columns(filtered_by_unit, columns)],
            use_container_width=True,
            hide_index=True,
        )


def render_pointload_module(parsed, pointload: pd.DataFrame) -> None:
    loca_ids_all = table_ids(parsed.get("LOCA"), "LOCA_ID")
    pointload_loca_ids = table_ids(pointload, "LOCA_ID")
    loca_without_pointload = [loca_id for loca_id in loca_ids_all if loca_id not in set(pointload_loca_ids)]

    loca_count, pointload_loca_count, pointload_count, matched_count = st.columns(4)
    loca_count.metric("LOCA investigations", len(loca_ids_all) or "n/a")
    pointload_loca_count.metric("Point load investigations", len(pointload_loca_ids))
    pointload_count.metric("Point load records", len(pointload))
    matched_count.metric("Matched to GEOL", int(pointload["GEOLOGY_MATCHED"].sum()))

    st.caption(
        "Graphs use `RPLT_PLSI` from the `RPLT` group. Depth is taken from `SPEC_DPTH` where present, "
        "otherwise `SAMP_TOP` is used."
    )

    if loca_without_pointload:
        with st.expander(f"{len(loca_without_pointload)} LOCA investigations have no plottable point load records"):
            st.dataframe(
                pd.DataFrame({"LOCA_ID": loca_without_pointload}),
                use_container_width=True,
                hide_index=True,
            )

    selected_loca, geology_mode, selected_units, selected_materials, selected_model_units, selected_bedrock = render_filters(pointload, "point load records")

    filtered = pointload[pointload["LOCA_ID"].isin(selected_loca)].copy()
    filtered_by_unit = apply_geology_filter(filtered, selected_units, geology_mode, selected_materials, selected_model_units, selected_bedrock)

    tab_all, tab_units, tab_data = st.tabs(
        ["All Investigations", "Geological Units", "Matched Data"]
    )

    with tab_all:
        render_pointload_plot(
            filtered_by_unit,
            title="Point Load Strength Index vs Depth by Investigation",
            color_by="LOCA_ID",
        )

    with tab_units:
        render_pointload_plot(
            filtered_by_unit,
            title="Point Load Strength Index vs Depth by Geological Unit",
            color_by="GEOL_GEOL",
        )

    with tab_data:
        columns = [
            "LOCA_ID",
            "SAMP_TOP",
            "SPEC_DPTH",
            "RPLT_PLSI",
            "RPLT_PLS",
            "POINTLOAD_DEPTH_NUM",
            "RPLT_PLSI_NUM",
            "GEOL_GEOL",
            "GEOL_TOP",
            "GEOL_BASE",
            "GEOL_DESC",
            "GEOLOGY_MATCHED",
        ]
        st.dataframe(
            filtered_by_unit[matched_data_columns(filtered_by_unit, columns)],
            use_container_width=True,
            hide_index=True,
        )


def render_groundwater_module(parsed, groundwater: pd.DataFrame) -> None:
    loca_ids_all = table_ids(parsed.get("LOCA"), "LOCA_ID")
    groundwater_loca_ids = table_ids(groundwater, "LOCA_ID")
    loca_without_groundwater = [loca_id for loca_id in loca_ids_all if loca_id not in set(groundwater_loca_ids)]

    loca_count, groundwater_loca_count, groundwater_count, post_count, matched_count = st.columns(5)
    loca_count.metric("LOCA investigations", len(loca_ids_all) or "n/a")
    groundwater_loca_count.metric("Strike investigations", len(groundwater_loca_ids))
    groundwater_count.metric("Groundwater strikes", len(groundwater))
    post_count.metric("Post-strike readings", int(groundwater["WSTD_POST_NUM"].notna().sum()))
    matched_count.metric("Matched to GEOL", int(groundwater["GEOLOGY_MATCHED"].sum()))

    st.caption(
        "Graphs use groundwater strike depth from `WSTG_DPTH` in the `WSTG` group. "
        "`WSTD_POST` readings from `WSTD` are included in the data table where available."
    )

    if loca_without_groundwater:
        with st.expander(f"{len(loca_without_groundwater)} LOCA investigations have no plottable groundwater strike records"):
            st.dataframe(
                pd.DataFrame({"LOCA_ID": loca_without_groundwater}),
                use_container_width=True,
                hide_index=True,
            )

    selected_loca, geology_mode, selected_units, selected_materials, selected_model_units, selected_bedrock = render_filters(groundwater, "groundwater strike records")

    filtered = groundwater[groundwater["LOCA_ID"].isin(selected_loca)].copy()
    filtered_by_unit = apply_geology_filter(filtered, selected_units, geology_mode, selected_materials, selected_model_units, selected_bedrock)

    tab_all, tab_units, tab_data = st.tabs(
        ["All Investigations", "Geological Units", "Matched Data"]
    )

    with tab_all:
        render_groundwater_plot(
            filtered_by_unit,
            title="Groundwater Strike Depth by Investigation",
            color_by="LOCA_ID",
        )

    with tab_units:
        render_groundwater_plot(
            filtered_by_unit,
            title="Groundwater Strike Depth by Geological Unit",
            color_by="GEOL_GEOL",
        )

    with tab_data:
        columns = [
            "LOCA_ID",
            "WSTG_DPTH",
            "WSTG_DTIM",
            "WSTG_SEAL",
            "WSTG_CAS",
            "WSTD_NMIN",
            "WSTD_POST",
            "WSTG_DPTH_NUM",
            "WSTD_POST_NUM",
            "GEOL_GEOL",
            "GEOL_TOP",
            "GEOL_BASE",
            "GEOL_DESC",
            "GEOLOGY_MATCHED",
        ]
        st.dataframe(
            filtered_by_unit[matched_data_columns(filtered_by_unit, columns)],
            use_container_width=True,
            hide_index=True,
        )


def render_map_module(
    locations: pd.DataFrame,
    geology: pd.DataFrame,
    x_column: str,
    y_column: str,
    x_label: str,
    y_label: str,
) -> None:
    location_count, geology_count, unit_count = st.columns(3)
    location_count.metric("Mapped investigations", len(locations))
    geology_count.metric("Geology intervals", len(geology))
    unit_count.metric("Geological units", geology["GEOL_GEOL"].nunique())

    st.caption(
        "Map positions use the populated `LOCA` coordinate pair. The geology filter reports matching `GEOL` depth "
        "intervals for the investigations selected on the map."
    )

    filter_col_1, filter_col_2 = st.columns([1.4, 1])
    location_ids = sorted(locations["LOCA_ID"].dropna().unique())
    geology_units = sorted(geology["GEOL_GEOL"].dropna().unique())

    with filter_col_1:
        investigation_mode = st.radio(
            "Investigation filter",
            ["All investigations", "Choose investigations"],
            horizontal=True,
            key="map_investigation_filter",
        )
        if investigation_mode == "All investigations":
            selected_loca = location_ids
        else:
            selected_loca = st.multiselect(
                "Mapped investigations",
                location_ids,
                default=location_ids[: min(len(location_ids), 12)],
                key="map_selected_loca",
            )
        st.caption(f"Showing {len(selected_loca)} investigations on the map.")

    with filter_col_2:
        selected_units = st.multiselect(
            "Geology to report",
            geology_units,
            default=["PEAT"] if "PEAT" in geology_units else geology_units[:1],
            key="map_selected_geology",
        )

    selected_locations = locations[locations["LOCA_ID"].isin(selected_loca)].copy()
    selected_geology = geology[
        geology["LOCA_ID"].isin(selected_loca) & geology["GEOL_GEOL"].isin(selected_units)
    ].copy()
    merged_geology = merge_geology_intervals(selected_geology)

    map_display = st.radio(
        "Map display",
        ["Static plot", "Aerial basemap"],
        horizontal=True,
        key="map_display_mode",
    )
    if map_display == "Aerial basemap":
        render_aerial_map(
            selected_locations,
            merged_geology,
            x_column,
            y_column,
            x_label,
            y_label,
        )
    else:
        png_bytes = build_map_png(
            selected_locations,
            merged_geology,
            x_column,
            y_column,
            x_label,
            y_label,
            selected_units,
        )
        st.image(png_bytes, use_container_width=True)
        st.download_button(
            "Download map PNG",
            data=png_bytes,
            file_name="ags_map_viewer.png",
            mime="image/png",
        )

    tab_summary, tab_intervals, tab_locations = st.tabs(["Geology Summary", "Geology Intervals", "Mapped Locations"])

    with tab_summary:
        summary = build_geology_summary(merged_geology)
        if summary.empty:
            st.warning("No geology intervals match the selected investigations and geology units.")
        else:
            st.dataframe(summary, use_container_width=True, hide_index=True)

    with tab_intervals:
        columns = [
            "LOCA_ID",
            "GEOL_GEOL",
            "GEOL_TOP",
            "GEOL_BASE",
            "GEOL_TOP_NUM",
            "GEOL_BASE_NUM",
            "THICKNESS_NUM",
            "GEOL_DESC",
        ]
        st.dataframe(selected_geology[[column for column in columns if column in selected_geology.columns]], use_container_width=True, hide_index=True)

    with tab_locations:
        st.dataframe(selected_locations, use_container_width=True, hide_index=True)


def render_geological_model_module(model: pd.DataFrame) -> None:
    interval_count, investigation_count, unit_count, material_count = st.columns(4)
    interval_count.metric("Strata intervals", len(model))
    investigation_count.metric("Investigations", model["LOCA_ID"].nunique())
    unit_count.metric("Geological units", model["GEOL_GEOL"].nunique())
    material_count.metric("Material classes", model["MATERIAL_CLASS"].nunique())

    st.caption(
        "Material classes are derived from `GEOL_DESC` and `GEOL_GEOL`. Bedrock type is extracted from capitalised "
        "rock names in the description, such as `PSAMMITE`, `GRANITE`, or `GNEISS`."
    )

    filter_col_1, filter_col_2, filter_col_3 = st.columns(3)
    with filter_col_1:
        selected_loca = st.multiselect(
            "Investigations",
            sorted(model["LOCA_ID"].dropna().unique()),
            default=[],
            placeholder="All investigations",
        )
    with filter_col_2:
        selected_units = st.multiselect(
            "Geological units",
            sorted(model["GEOL_GEOL"].dropna().unique()),
            default=[],
            placeholder="All geological units",
        )
    with filter_col_3:
        selected_materials = st.multiselect(
            "Material classes",
            sorted(model["MATERIAL_CLASS"].dropna().unique()),
            default=[],
            placeholder="All material classes",
        )

    filter_col_4, filter_col_5 = st.columns(2)
    with filter_col_4:
        selected_model_units = st.multiselect(
            "Model units",
            sorted(model["MODEL_UNIT"].dropna().unique()),
            default=[],
            placeholder="All model units",
        )
    with filter_col_5:
        bedrock_options = sorted(model["BEDROCK_TYPE"].dropna().unique())
        selected_bedrock = st.multiselect(
            "Bedrock types",
            bedrock_options,
            default=[],
            placeholder="All bedrock types",
        )

    filtered = apply_geological_model_filters(
        model,
        selected_loca,
        selected_units,
        selected_materials,
        selected_model_units,
        selected_bedrock,
    )

    tab_summary, tab_investigations, tab_profile, tab_data = st.tabs(
        ["Model Summary", "Matching Investigations", "Investigation Profiles", "Filtered Strata"]
    )

    with tab_summary:
        summary = build_geological_model_summary(filtered)
        if summary.empty:
            st.warning("No strata match the current filters.")
        else:
            st.dataframe(summary, use_container_width=True, hide_index=True)

    with tab_investigations:
        investigation_summary = build_filtered_investigation_summary(filtered)
        if investigation_summary.empty:
            st.warning("No investigations match the current filters.")
        else:
            st.caption(f"{len(investigation_summary)} investigations match the current filters.")
            st.dataframe(investigation_summary, use_container_width=True, hide_index=True)

    with tab_profile:
        selected_profile_loca = st.multiselect(
            "Profile investigations",
            sorted(filtered["LOCA_ID"].dropna().unique()),
            default=sorted(filtered["LOCA_ID"].dropna().unique())[: min(8, filtered["LOCA_ID"].nunique())],
        )
        profile_data = filtered[filtered["LOCA_ID"].isin(selected_profile_loca)].copy()
        render_geological_profile_plot(profile_data)

    with tab_data:
        columns = [
            "LOCA_ID",
            "GEOL_TOP",
            "GEOL_BASE",
            "GEOL_TOP_NUM",
            "GEOL_BASE_NUM",
            "THICKNESS_NUM",
            "GEOL_GEOL",
            "MATERIAL_CLASS",
            "MODEL_UNIT",
            "BEDROCK_TYPE",
            "GEOL_DESC",
            "GEOL_FORM",
            "GEOL_LEG",
        ]
        st.dataframe(filtered[[column for column in columns if column in filtered.columns]], use_container_width=True, hide_index=True)


def render_psd_module(parsed, psd: pd.DataFrame) -> None:
    loca_ids_all = table_ids(parsed.get("LOCA"), "LOCA_ID")
    psd_loca_ids = table_ids(psd, "LOCA_ID")
    sample_count = psd["PSD_SAMPLE_ID"].nunique()
    loca_without_psd = [loca_id for loca_id in loca_ids_all if loca_id not in set(psd_loca_ids)]

    loca_count, psd_loca_count, psd_sample_count, point_count, matched_count = st.columns(5)
    loca_count.metric("LOCA investigations", len(loca_ids_all) or "n/a")
    psd_loca_count.metric("PSD investigations", len(psd_loca_ids))
    psd_sample_count.metric("PSD curves", sample_count)
    point_count.metric("Curve points", len(psd))
    matched_count.metric("Matched to GEOL", int(psd["GEOLOGY_MATCHED"].sum()))

    st.caption(
        "Curves use `GRAT_SIZE` and `GRAT_PERP` from the `GRAT` group. Each curve is grouped by investigation, "
        "sample depth, sample reference, sample type, sample ID, and specimen reference where present."
    )

    if loca_without_psd:
        with st.expander(f"{len(loca_without_psd)} LOCA investigations have no plottable PSD records"):
            st.dataframe(
                pd.DataFrame({"LOCA_ID": loca_without_psd}),
                use_container_width=True,
                hide_index=True,
            )

    selected_loca, geology_mode, selected_units, selected_materials, selected_model_units, selected_bedrock = render_filters(psd, "PSD curve points")

    filtered = psd[psd["LOCA_ID"].isin(selected_loca)].copy()
    filtered_by_unit = apply_geology_filter(filtered, selected_units, geology_mode, selected_materials, selected_model_units, selected_bedrock)

    curve_ids = sorted(filtered_by_unit["PSD_SAMPLE_ID"].dropna().unique())
    selected_curves = st.multiselect(
        "PSD curves",
        curve_ids,
        default=curve_ids[: min(len(curve_ids), 12)],
    )
    plot_data = filtered_by_unit[filtered_by_unit["PSD_SAMPLE_ID"].isin(selected_curves)].copy()
    st.caption(f"Plotting {plot_data['PSD_SAMPLE_ID'].nunique()} curves. Select fewer curves for clearer comparison.")

    tab_curve, tab_data = st.tabs(["PSD Curves", "Matched Data"])

    with tab_curve:
        render_psd_plot(plot_data)

    with tab_data:
        columns = [
            "LOCA_ID",
            "SAMP_TOP",
            "SAMP_REF",
            "SAMP_TYPE",
            "SAMP_ID",
            "SPEC_REF",
            "GRAT_SIZE",
            "GRAT_PERP",
            "GRAT_TYPE",
            "PSD_SAMPLE_ID",
            "PSD_DEPTH_NUM",
            "GEOL_GEOL",
            "GEOL_TOP",
            "GEOL_BASE",
            "GEOL_DESC",
            "GEOLOGY_MATCHED",
        ]
        st.dataframe(
            filtered_by_unit[matched_data_columns(filtered_by_unit, columns)],
            use_container_width=True,
            hide_index=True,
        )


def render_filters(data: pd.DataFrame, record_label: str) -> tuple[list[str], str, list[str], list[str], list[str], list[str]]:
    st.subheader("Filters")
    loca_ids = sorted(data["LOCA_ID"].dropna().unique())
    units = sorted(data["GEOL_GEOL"].fillna("Unmatched").unique())
    material_classes = sorted(data["MATERIAL_CLASS"].dropna().unique()) if "MATERIAL_CLASS" in data.columns else []
    model_units = sorted(data["MODEL_UNIT"].dropna().unique()) if "MODEL_UNIT" in data.columns else []
    bedrock_types = sorted(data["BEDROCK_TYPE"].dropna().unique()) if "BEDROCK_TYPE" in data.columns else []
    st.caption(f"Investigation filters operate on {len(loca_ids)} locations with valid {record_label}.")

    filter_col_1, filter_col_2 = st.columns([1.4, 1])
    with filter_col_1:
        investigation_mode = st.radio(
            "Investigation filter",
            ["All investigations", "Choose investigations"],
            horizontal=True,
        )
        if investigation_mode == "All investigations":
            selected_loca = loca_ids
            st.caption(f"Plotting all {len(selected_loca)} investigations with valid {record_label}.")
        else:
            selected_loca = st.multiselect(f"Investigations with valid {record_label}", loca_ids, default=[])
            st.caption(f"Plotting {len(selected_loca)} selected investigations.")

    with filter_col_2:
        geology_mode = st.radio(
            "Geology filter",
            ["All geology", "Include selected"],
            horizontal=True,
        )
        selected_units = []
        if geology_mode == "Include selected":
            selected_units = st.multiselect(
                "Geological units",
                units,
                default=default_geology_selection(units, geology_mode),
            )

    st.caption("Material filters are derived from the matched GEOL description. Leave blank to include all.")
    material_col_1, material_col_2, material_col_3 = st.columns(3)
    with material_col_1:
        selected_materials = st.multiselect(
            "Material classes",
            material_classes,
            default=[],
            placeholder="All material classes",
        )
    with material_col_2:
        selected_model_units = st.multiselect(
            "Model units",
            model_units,
            default=[],
            placeholder="All model units",
        )
    with material_col_3:
        selected_bedrock = st.multiselect(
            "Bedrock types",
            bedrock_types,
            default=[],
            placeholder="All bedrock types",
        )

    return selected_loca, geology_mode, selected_units, selected_materials, selected_model_units, selected_bedrock


def table_ids(table: pd.DataFrame, column: str) -> list[str]:
    if table.empty or column not in table.columns:
        return []
    return sorted(table[column].dropna().astype(str).str.strip().unique())


def matched_data_columns(data: pd.DataFrame, columns: list[str]) -> list[str]:
    model_columns = ["MATERIAL_CLASS", "MODEL_UNIT", "BEDROCK_TYPE"]
    output: list[str] = []
    for column in columns:
        if column == "GEOL_DESC":
            output.extend([model_column for model_column in model_columns if model_column in data.columns])
        output.append(column)
    return [column for column in dict.fromkeys(output) if column in data.columns]


def render_local_file_loader(show_loader: bool) -> Path | None:
    if not show_loader:
        return None

    with st.expander("Load a local file path"):
        with st.form("local_file_loader"):
            candidates = recent_input_files()
            selected = st.selectbox(
                "Recent files",
                [""] + [str(path) for path in candidates],
                format_func=lambda value: Path(value).name if value else "",
            )
            typed_path = st.text_input("Path override", placeholder="Paste a full file path if needed")
            chosen_path = typed_path.strip() or selected
            load_clicked = st.form_submit_button("Load file", type="primary")

    if not load_clicked:
        return None

    if not chosen_path.strip():
        st.error("Choose a recent file or paste a full file path.")
        return None

    path = Path(chosen_path.strip().strip('"'))
    if not path.exists():
        st.error(f"File not found: {path}")
        return None
    if path.suffix.lower() not in {".ags", ".csv", ".txt", ".xlsx"}:
        st.error("Supported file types are .ags, .csv, .txt, and .xlsx.")
        return None
    return path


@st.cache_data(ttl=10, show_spinner=False)
def recent_input_files() -> list[Path]:
    folders = [Path.home() / "Downloads", Path.cwd()]
    files: list[Path] = []
    for folder in folders:
        if not folder.exists():
            continue
        for suffix in ("*.ags", "*.xlsx", "*.csv", "*.txt"):
            files.extend(folder.glob(suffix))
    return sorted(files, key=lambda path: path.stat().st_mtime, reverse=True)[:20]


def default_geology_selection(units: list[str], geology_mode: str) -> list[str]:
    if geology_mode == "All geology":
        return []
    return ["GT"] if "GT" in units else []


def apply_geology_filter(
    data: pd.DataFrame,
    selected_units: list[str],
    geology_mode: str,
    selected_materials: list[str] | None = None,
    selected_model_units: list[str] | None = None,
    selected_bedrock: list[str] | None = None,
) -> pd.DataFrame:
    if geology_mode == "All geology":
        filtered = data.copy()
    if geology_mode == "Include selected":
        if not selected_units:
            filtered = data.iloc[0:0].copy()
        else:
            filtered = data[data["GEOL_GEOL"].isin(selected_units)].copy()
    if geology_mode not in {"All geology", "Include selected"}:
        filtered = data.copy()

    if selected_materials and "MATERIAL_CLASS" in filtered.columns:
        filtered = filtered[filtered["MATERIAL_CLASS"].isin(selected_materials)].copy()
    if selected_model_units and "MODEL_UNIT" in filtered.columns:
        filtered = filtered[filtered["MODEL_UNIT"].isin(selected_model_units)].copy()
    if selected_bedrock and "BEDROCK_TYPE" in filtered.columns:
        filtered = filtered[filtered["BEDROCK_TYPE"].isin(selected_bedrock)].copy()
    return filtered


def content_size(content: bytes) -> str:
    size_mb = len(content) / (1024 * 1024)
    if size_mb >= 1:
        return f"{size_mb:.1f} MB"
    return f"{len(content) / 1024:.1f} KB"


def render_spt_plot(
    data: pd.DataFrame,
    title: str,
    color_by: str,
    x_column: str = "ISPT_MAIN_NUM",
    x_label: str = "SPT blow count, N (ISPT_MAIN)",
) -> None:
    render_depth_scatter_plot(
        data=data,
        title=title,
        color_by=color_by,
        x_column=x_column,
        y_column="ISPT_TOP_NUM",
        x_label=x_label,
        y_label="Depth below ground level (m)",
    )


def render_ivan_plot(
    data: pd.DataFrame,
    title: str,
    color_by: str,
) -> None:
    render_depth_scatter_plot(
        data=data,
        title=title,
        color_by=color_by,
        x_column="IVAN_IVAN_NUM",
        y_column="IVAN_DPTH_NUM",
        x_label="Hand shear vane reading (IVAN_IVAN)",
        y_label="Depth below ground level (m)",
    )


def render_ucs_plot(
    data: pd.DataFrame,
    title: str,
    color_by: str,
) -> None:
    render_depth_scatter_plot(
        data=data,
        title=title,
        color_by=color_by,
        x_column="RUCS_UCS_NUM",
        y_column="SAMP_TOP_NUM",
        x_label="Unconfined compressive strength (RUCS_UCS)",
        y_label="Sample top depth below ground level (m)",
    )


def render_rqd_plot(
    data: pd.DataFrame,
    title: str,
    color_by: str,
) -> None:
    render_depth_scatter_plot(
        data=data,
        title=title,
        color_by=color_by,
        x_column="CORE_RQD_NUM",
        y_column="CORE_TOP_NUM",
        x_label="Rock quality designation, RQD (%) (CORE_RQD)",
        y_label="Core run top depth below ground level (m)",
    )


def render_pointload_plot(
    data: pd.DataFrame,
    title: str,
    color_by: str,
) -> None:
    render_depth_scatter_plot(
        data=data,
        title=title,
        color_by=color_by,
        x_column="RPLT_PLSI_NUM",
        y_column="POINTLOAD_DEPTH_NUM",
        x_label="Point load strength index, Is50 (RPLT_PLSI)",
        y_label="Depth below ground level (m)",
    )


def render_groundwater_plot(
    data: pd.DataFrame,
    title: str,
    color_by: str,
) -> None:
    render_depth_scatter_plot(
        data=data,
        title=title,
        color_by=color_by,
        x_column="GROUNDWATER_PLOT_NUM",
        y_column="WSTG_DPTH_NUM",
        x_label="Investigation order",
        y_label="Groundwater strike depth below ground level (m)",
    )


def render_atterberg_pair(
    data: pd.DataFrame,
    value_column: str,
    value_label: str,
    title_prefix: str,
) -> None:
    plot_data = data.dropna(subset=[value_column]).copy()
    if plot_data.empty:
        st.warning(f"No {title_prefix.lower()} records match the current filters.")
        return

    st.subheader("By Investigation")
    render_atterberg_plot(
        plot_data,
        title=f"{title_prefix} vs Depth by Investigation",
        color_by="LOCA_ID",
        value_column=value_column,
        value_label=value_label,
    )

    st.subheader("By Geological Unit")
    render_atterberg_plot(
        plot_data,
        title=f"{title_prefix} vs Depth by Geological Unit",
        color_by="GEOL_GEOL",
        value_column=value_column,
        value_label=value_label,
    )


def render_atterberg_plot(
    data: pd.DataFrame,
    title: str,
    color_by: str,
    value_column: str,
    value_label: str,
) -> None:
    render_depth_scatter_plot(
        data=data,
        title=title,
        color_by=color_by,
        x_column=value_column,
        y_column="SAMP_TOP_NUM",
        x_label=value_label,
        y_label="Sample top depth below ground level (m)",
    )


def render_psd_plot(data: pd.DataFrame) -> None:
    if data.empty:
        st.warning("No PSD curves match the current filters.")
        return

    title = "Particle Size Distribution Curves"
    design_line = st.selectbox(
        "Design line",
        DESIGN_LINE_OPTIONS,
        key="design_line_particle_size_distribution_curves",
        help=(
            "Calculates a statistical curve from the selected PSD curves using a one-sided 95% "
            "confidence bound at each particle size."
        ),
    )
    show_design_line = design_line != "Off"
    if show_design_line and data["PSD_SAMPLE_ID"].nunique() < 3:
        st.warning("At least three selected PSD curves are needed for a statistical design curve.")
        show_design_line = False

    png_bytes = build_psd_png(data, title, design_line if show_design_line else "Off")
    st.image(png_bytes, use_container_width=True)
    st.download_button(
        "Download graph PNG",
        data=png_bytes,
        file_name=f"{slugify(title)}.png",
        mime="image/png",
    )
    if show_design_line:
        st.caption(
            "The PSD design curve is recalculated from the selected curves and uses a one-sided 95% "
            "confidence bound at each particle size."
        )


@st.cache_data(show_spinner=False)
def build_map_png(
    locations: pd.DataFrame,
    geology: pd.DataFrame,
    x_column: str,
    y_column: str,
    x_label: str,
    y_label: str,
    selected_units: list[str],
) -> bytes:
    fig, ax = plt.subplots(figsize=(9.4, 7.0), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    if locations.empty:
        ax.text(0.5, 0.5, "No selected investigations", ha="center", va="center", transform=ax.transAxes)
    else:
        highlighted = set(geology["LOCA_ID"].dropna().astype(str).unique())
        geology_labels = build_map_geology_labels(geology)
        base = locations[~locations["LOCA_ID"].isin(highlighted)]
        matching = locations[locations["LOCA_ID"].isin(highlighted)]

        if not base.empty:
            ax.scatter(
                base[x_column],
                base[y_column],
                s=42,
                c="#6B6D76",
                edgecolors="#2f2f2f",
                linewidths=0.35,
                alpha=0.7,
                label="Selected investigation",
            )
        if not matching.empty:
            ax.scatter(
                matching[x_column],
                matching[y_column],
                s=64,
                c=SCIENTIFIC_PALETTE[2],
                edgecolors="#1f1f1f",
                linewidths=0.5,
                alpha=0.95,
                label="Contains selected geology",
            )

        for _, row in locations.iterrows():
            borehole_label = str(row["LOCA_ID"])
            geology_label = geology_labels.get(borehole_label, "")
            label = borehole_label if not geology_label else f"{borehole_label}\n{geology_label}"
            ax.annotate(
                label,
                (row[x_column], row[y_column]),
                textcoords="offset points",
                xytext=(5, -4),
                fontsize=6.5,
                color="#222222",
                va="top",
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "white",
                    "edgecolor": "#d0d0d0",
                    "linewidth": 0.4,
                    "alpha": 0.82,
                } if geology_label else None,
            )

    unit_label = ", ".join(selected_units) if selected_units else "No geology selected"
    ax.set_title(f"Investigation Map - {unit_label}", fontsize=11, weight="semibold", color="#222222", pad=10)
    ax.set_xlabel(x_label, fontsize=10, color="#333333")
    ax.set_ylabel(y_label, fontsize=10, color="#333333")
    ax.tick_params(axis="both", colors="#444444", labelsize=8)
    ax.grid(True, which="major", color="#d8d8d8", linewidth=0.7)
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(x=0.06, y=0.08)

    for spine in ax.spines.values():
        spine.set_color("#555555")
        spine.set_linewidth(0.9)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            frameon=True,
            facecolor="white",
            edgecolor="#c0c0c0",
            fontsize=8,
            loc="best",
        )

    fig.tight_layout()
    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buffer.getvalue()


def render_aerial_map(
    locations: pd.DataFrame,
    geology: pd.DataFrame,
    x_column: str,
    y_column: str,
    x_label: str,
    y_label: str,
) -> None:
    aerial_data = build_aerial_map_data(locations, geology, x_column, y_column, x_label, y_label)
    if aerial_data.empty:
        st.warning("The selected coordinates cannot be converted to latitude/longitude for an aerial basemap.")
        return

    center_lat = float(aerial_data["LATITUDE"].mean())
    center_lon = float(aerial_data["LONGITUDE"].mean())
    highlighted = aerial_data[aerial_data["HAS_SELECTED_GEOLOGY"]]
    unhighlighted = aerial_data[~aerial_data["HAS_SELECTED_GEOLOGY"]]

    tile_layer = pdk.Layer(
        "TileLayer",
        data="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        min_zoom=0,
        max_zoom=19,
        tile_size=256,
        render_sub_layers={
            "@@type": "BitmapLayer",
            "data": None,
            "image": "@@=data",
            "bounds": "@@=bbox",
        },
    )
    layers = [tile_layer]
    if not unhighlighted.empty:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=unhighlighted,
                get_position="[LONGITUDE, LATITUDE]",
                get_fill_color=[107, 109, 118, 190],
                get_line_color=[255, 255, 255, 230],
                get_radius=8,
                radius_units="meters",
                pickable=True,
            )
        )
    if not highlighted.empty:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=highlighted,
                get_position="[LONGITUDE, LATITUDE]",
                get_fill_color=[76, 149, 108, 235],
                get_line_color=[255, 255, 255, 255],
                get_radius=12,
                radius_units="meters",
                pickable=True,
            )
        )
    layers.append(
        pdk.Layer(
            "TextLayer",
            data=aerial_data,
            get_position="[LONGITUDE, LATITUDE]",
            get_text="MAP_LABEL",
            get_size=13,
            get_color=[255, 255, 255, 255],
            get_angle=0,
            get_text_anchor="start",
            get_alignment_baseline="top",
            get_pixel_offset=[8, 8],
            background=True,
            get_background_color=[0, 43, 91, 210],
            pickable=False,
        )
    )

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=13,
            pitch=0,
            bearing=0,
        ),
        tooltip={
            "html": "<b>{LOCA_ID}</b><br/>{GEOLOGY_LABEL}",
            "style": {"backgroundColor": "#002b5b", "color": "white"},
        },
        map_provider=None,
    )
    st.pydeck_chart(deck, use_container_width=True)
    st.caption(
        "Aerial imagery is loaded from Esri World Imagery. Labels show the investigation name and selected geology ranges."
    )


@st.cache_data(show_spinner=False)
def build_aerial_map_data(
    locations: pd.DataFrame,
    geology: pd.DataFrame,
    x_column: str,
    y_column: str,
    x_label: str,
    y_label: str,
) -> pd.DataFrame:
    if locations.empty:
        return pd.DataFrame()

    converted = locations.copy()
    if x_label == "Longitude" and y_label == "Latitude":
        converted["LONGITUDE"] = pd.to_numeric(converted[x_column], errors="coerce")
        converted["LATITUDE"] = pd.to_numeric(converted[y_column], errors="coerce")
    elif x_label == "Easting" and y_label == "Northing":
        transformer = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
        longitudes, latitudes = transformer.transform(
            converted[x_column].astype(float).to_numpy(),
            converted[y_column].astype(float).to_numpy(),
        )
        converted["LONGITUDE"] = longitudes
        converted["LATITUDE"] = latitudes
    else:
        return pd.DataFrame()

    converted = converted.dropna(subset=["LONGITUDE", "LATITUDE"]).copy()
    converted = converted[
        converted["LONGITUDE"].between(-180, 180) & converted["LATITUDE"].between(-90, 90)
    ].copy()
    if converted.empty:
        return converted

    geology_labels = build_map_geology_labels(geology)
    highlighted = set(geology["LOCA_ID"].dropna().astype(str).unique())
    converted["GEOLOGY_LABEL"] = converted["LOCA_ID"].astype(str).map(geology_labels).fillna("")
    converted["HAS_SELECTED_GEOLOGY"] = converted["LOCA_ID"].astype(str).isin(highlighted)
    converted["MAP_LABEL"] = converted.apply(
        lambda row: str(row["LOCA_ID"])
        if not row["GEOLOGY_LABEL"]
        else f"{row['LOCA_ID']}\n{row['GEOLOGY_LABEL']}",
        axis=1,
    )
    return converted


def build_geology_summary(geology: pd.DataFrame) -> pd.DataFrame:
    if geology.empty:
        return pd.DataFrame()

    summary = merge_geology_intervals(geology)
    summary["Depth range"] = summary["GEOL_TOP_NUM"].map(format_depth) + "-" + summary["GEOL_BASE_NUM"].map(format_depth) + " m"
    summary = summary.rename(columns={"GEOL_GEOL": "Geology", "THICKNESS_NUM": "Total thickness (m)"})
    summary["Total thickness (m)"] = summary["Total thickness (m)"].round(2)
    return summary[
        ["LOCA_ID", "Geology", "Depth range", "Total thickness (m)"]
    ].sort_values(["Geology", "LOCA_ID", "Depth range"]).reset_index(drop=True)


def merge_geology_intervals(geology: pd.DataFrame) -> pd.DataFrame:
    if geology.empty:
        return geology.copy()

    merged_rows: list[dict[str, object]] = []
    sorted_geology = geology.dropna(subset=["LOCA_ID", "GEOL_GEOL", "GEOL_TOP_NUM", "GEOL_BASE_NUM"]).sort_values(
        ["LOCA_ID", "GEOL_GEOL", "GEOL_TOP_NUM", "GEOL_BASE_NUM"]
    )
    for (loca_id, unit), group in sorted_geology.groupby(["LOCA_ID", "GEOL_GEOL"], sort=False):
        current_top: float | None = None
        current_base: float | None = None
        descriptions: list[str] = []

        for _, row in group.iterrows():
            top = float(row["GEOL_TOP_NUM"])
            base = float(row["GEOL_BASE_NUM"])
            description = row.get("GEOL_DESC", pd.NA)
            touches_current = current_base is not None and top <= current_base + 1e-9

            if current_top is None or current_base is None or not touches_current:
                if current_top is not None and current_base is not None:
                    merged_rows.append(
                        build_merged_geology_row(loca_id, unit, current_top, current_base, descriptions)
                    )
                current_top = top
                current_base = base
                descriptions = []
            else:
                current_base = max(current_base, base)

            if not pd.isna(description):
                descriptions.append(str(description))

        if current_top is not None and current_base is not None:
            merged_rows.append(build_merged_geology_row(loca_id, unit, current_top, current_base, descriptions))

    return pd.DataFrame(merged_rows)


def build_merged_geology_row(
    loca_id: str,
    unit: str,
    top: float,
    base: float,
    descriptions: list[str],
) -> dict[str, object]:
    unique_descriptions = list(dict.fromkeys(descriptions))
    return {
        "LOCA_ID": loca_id,
        "GEOL_GEOL": unit,
        "GEOL_TOP": format_depth(top),
        "GEOL_BASE": format_depth(base),
        "GEOL_TOP_NUM": top,
        "GEOL_BASE_NUM": base,
        "THICKNESS_NUM": base - top,
        "GEOL_DESC": "; ".join(unique_descriptions),
    }


def build_map_geology_labels(geology: pd.DataFrame) -> dict[str, str]:
    if geology.empty:
        return {}

    labels: dict[str, str] = {}
    for loca_id, group in geology.groupby("LOCA_ID", sort=False):
        parts = [
            f"{row['GEOL_GEOL']} {format_depth(row['GEOL_TOP_NUM'])}-{format_depth(row['GEOL_BASE_NUM'])}m"
            for _, row in group.sort_values(["GEOL_GEOL", "GEOL_TOP_NUM"]).iterrows()
        ]
        labels[str(loca_id)] = "\n".join(parts[:3])
        if len(parts) > 3:
            labels[str(loca_id)] += f"\n+{len(parts) - 3} more"
    return labels


def format_depth(value: object) -> str:
    return f"{float(value):g}"


def apply_geological_model_filters(
    model: pd.DataFrame,
    selected_loca: list[str],
    selected_units: list[str],
    selected_materials: list[str],
    selected_model_units: list[str],
    selected_bedrock: list[str],
) -> pd.DataFrame:
    filtered = model.copy()
    if selected_loca:
        filtered = filtered[filtered["LOCA_ID"].isin(selected_loca)]
    if selected_units:
        filtered = filtered[filtered["GEOL_GEOL"].isin(selected_units)]
    if selected_materials:
        filtered = filtered[filtered["MATERIAL_CLASS"].isin(selected_materials)]
    if selected_model_units:
        filtered = filtered[filtered["MODEL_UNIT"].isin(selected_model_units)]
    if selected_bedrock:
        filtered = filtered[filtered["BEDROCK_TYPE"].isin(selected_bedrock)]
    return filtered.copy()


def build_geological_model_summary(model: pd.DataFrame) -> pd.DataFrame:
    if model.empty:
        return pd.DataFrame()

    summary = (
        model.groupby(["MODEL_UNIT", "GEOL_GEOL", "MATERIAL_CLASS", "BEDROCK_TYPE"], dropna=False, as_index=False)
        .agg(
            Intervals=("LOCA_ID", "count"),
            Investigations=("LOCA_ID", "nunique"),
            MinTop=("GEOL_TOP_NUM", "min"),
            MaxBase=("GEOL_BASE_NUM", "max"),
            TotalThickness=("THICKNESS_NUM", "sum"),
        )
        .rename(
            columns={
                "MODEL_UNIT": "Model unit",
                "GEOL_GEOL": "Geological unit",
                "MATERIAL_CLASS": "Material class",
                "BEDROCK_TYPE": "Bedrock type",
                "MinTop": "Min top (m)",
                "MaxBase": "Max base (m)",
                "TotalThickness": "Total thickness (m)",
            }
        )
    )
    summary["Bedrock type"] = summary["Bedrock type"].fillna("")
    summary["Min top (m)"] = summary["Min top (m)"].round(2)
    summary["Max base (m)"] = summary["Max base (m)"].round(2)
    summary["Total thickness (m)"] = summary["Total thickness (m)"].round(2)
    return summary.sort_values(["Material class", "Geological unit", "Model unit"]).reset_index(drop=True)


def build_filtered_investigation_summary(model: pd.DataFrame) -> pd.DataFrame:
    if model.empty:
        return pd.DataFrame()

    summary = model.copy()
    summary["Depth range"] = summary["GEOL_TOP_NUM"].map(format_depth) + "-" + summary["GEOL_BASE_NUM"].map(format_depth) + " m"
    grouped = (
        summary.groupby("LOCA_ID", as_index=False)
        .agg(
            Intervals=("LOCA_ID", "count"),
            GeologicalUnits=("GEOL_GEOL", lambda values: ", ".join(sorted(set(map(str, values))))),
            MaterialClasses=("MATERIAL_CLASS", lambda values: ", ".join(sorted(set(map(str, values))))),
            ModelUnits=("MODEL_UNIT", lambda values: ", ".join(sorted(set(map(str, values))))),
            BedrockTypes=("BEDROCK_TYPE", lambda values: ", ".join(sorted({str(value) for value in values if not pd.isna(value)}))),
            DepthRanges=("Depth range", lambda values: "; ".join(values)),
            TotalThickness=("THICKNESS_NUM", "sum"),
            MinTop=("GEOL_TOP_NUM", "min"),
            MaxBase=("GEOL_BASE_NUM", "max"),
        )
        .rename(
            columns={
                "LOCA_ID": "Investigation",
                "GeologicalUnits": "Geological units",
                "MaterialClasses": "Material classes",
                "ModelUnits": "Model units",
                "BedrockTypes": "Bedrock types",
                "DepthRanges": "Matching depth ranges",
                "TotalThickness": "Total matching thickness (m)",
                "MinTop": "Shallowest match (m)",
                "MaxBase": "Deepest match (m)",
            }
        )
    )
    grouped["Total matching thickness (m)"] = grouped["Total matching thickness (m)"].round(2)
    grouped["Shallowest match (m)"] = grouped["Shallowest match (m)"].round(2)
    grouped["Deepest match (m)"] = grouped["Deepest match (m)"].round(2)
    return grouped.sort_values("Investigation").reset_index(drop=True)


def render_geological_profile_plot(data: pd.DataFrame) -> None:
    if data.empty:
        st.warning("No profile strata match the current filters.")
        return

    title = "Geological Model Profiles"
    png_bytes = build_geological_profile_png(data, title)
    st.image(png_bytes, use_container_width=True)
    st.download_button(
        "Download profile PNG",
        data=png_bytes,
        file_name=f"{slugify(title)}.png",
        mime="image/png",
    )


@st.cache_data(show_spinner=False)
def build_geological_profile_png(data: pd.DataFrame, title: str) -> bytes:
    plot_data = data.dropna(subset=["LOCA_ID", "GEOL_TOP_NUM", "GEOL_BASE_NUM"]).copy()
    locas = sorted(plot_data["LOCA_ID"].dropna().unique())
    model_units = sorted(plot_data["MODEL_UNIT"].dropna().unique())
    color_lookup = {
        unit: SCIENTIFIC_PALETTE[index % len(SCIENTIFIC_PALETTE)]
        for index, unit in enumerate(model_units)
    }

    fig_width = max(8.2, min(14, 1.1 * len(locas)))
    fig, ax = plt.subplots(figsize=(fig_width, 6.2), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for index, loca_id in enumerate(locas):
        group = plot_data[plot_data["LOCA_ID"] == loca_id].sort_values("GEOL_TOP_NUM")
        for _, row in group.iterrows():
            top = float(row["GEOL_TOP_NUM"])
            base = float(row["GEOL_BASE_NUM"])
            height = max(base - top, 0.01)
            model_unit = str(row["MODEL_UNIT"])
            ax.bar(
                index,
                height,
                bottom=top,
                width=0.62,
                color=color_lookup.get(model_unit, DEFAULT_POINT_COLOR),
                edgecolor="#2f2f2f",
                linewidth=0.35,
                label=model_unit,
            )
            if height >= 0.7:
                ax.text(
                    index,
                    top + height / 2,
                    str(row["GEOL_GEOL"]),
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="#1f1f1f",
                    rotation=90,
                )

    ax.invert_yaxis()
    ax.set_title(title, fontsize=11, weight="semibold", color="#222222", pad=10)
    ax.set_ylabel("Depth below ground level (m)", fontsize=10, color="#333333")
    ax.set_xticks(range(len(locas)))
    ax.set_xticklabels(locas, rotation=45, ha="right", fontsize=8)
    ax.tick_params(axis="y", colors="#444444", labelsize=9)
    ax.grid(True, axis="y", color="#d8d8d8", linewidth=0.7)
    ax.set_axisbelow(True)

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    if len(unique) <= 14:
        ax.legend(
            unique.values(),
            unique.keys(),
            frameon=True,
            facecolor="white",
            edgecolor="#c0c0c0",
            fontsize=7,
            loc="best",
        )

    fig.tight_layout()
    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buffer.getvalue()


def render_depth_scatter_plot(
    data: pd.DataFrame,
    title: str,
    color_by: str,
    x_column: str,
    y_column: str,
    x_label: str,
    y_label: str,
) -> None:
    if data.empty:
        st.warning("No records match the current filters.")
        return

    design_line = st.selectbox(
        "Design line",
        DESIGN_LINE_OPTIONS,
        key=f"design_line_{slugify(title)}_{color_by}",
        help=(
            "Fits a linear trend to the currently plotted records and can show a one-sided 95% "
            "confidence bound as a cautious estimate."
        ),
    )
    show_design_line = design_line != "Off"
    if show_design_line and len(data.dropna(subset=[x_column, y_column])) < 3:
        st.warning("At least three plotted records are needed for a statistical design line.")
        show_design_line = False

    png_bytes = build_depth_scatter_png(
        data,
        title,
        color_by,
        x_column,
        y_column,
        x_label,
        y_label,
        design_line if show_design_line else "Off",
    )
    st.image(
        png_bytes,
        use_container_width=True,
    )
    st.download_button(
        "Download graph PNG",
        data=png_bytes,
        file_name=f"{slugify(title)}.png",
        mime="image/png",
    )

    if color_by == "LOCA_ID" and data[color_by].nunique() > 12:
        st.caption(
            "A single colour is used when more than 12 investigations are plotted. "
            "Using a separate colour for every investigation would make the graph and legend unreadable."
        )
    if show_design_line:
        st.caption(
            "The design line is recalculated from the records visible in this plot. "
            "The cautious estimate uses a one-sided 95% confidence bound on the fitted mean trend."
        )


@st.cache_data(show_spinner=False)
def build_depth_scatter_png(
    data: pd.DataFrame,
    title: str,
    color_by: str,
    x_column: str,
    y_column: str,
    x_label: str,
    y_label: str,
    design_line: str = "Off",
) -> bytes:
    fig, ax = plt.subplots(figsize=(8.2, 5.8), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    categories = [category for category in sorted(data[color_by].dropna().unique())]
    show_legend = 1 < len(categories) <= 12

    if color_by == "LOCA_ID" and len(categories) > 12:
        ax.scatter(
            data[x_column],
            data[y_column],
            s=32,
            c=DEFAULT_POINT_COLOR,
            edgecolors="none",
            linewidths=0,
            alpha=1.0,
        )
    else:
        for index, category in enumerate(categories):
            group = data[data[color_by] == category]
            ax.scatter(
                group[x_column],
                group[y_column],
                s=34,
                c=SCIENTIFIC_PALETTE[index % len(SCIENTIFIC_PALETTE)],
                edgecolors="#2f2f2f",
                linewidths=0.35,
                alpha=0.86,
                label=str(category),
            )

    line = calculate_design_line(data, x_column, y_column, design_line)
    if line is not None:
        line_x, line_y, label = line
        ax.plot(
            line_x,
            line_y,
            color=DESIGN_LINE_COLOR,
            linewidth=2.0,
            linestyle="--",
            label=label,
            zorder=5,
        )
        show_legend = True

    ax.invert_yaxis()
    ax.set_title(title, fontsize=11, weight="semibold", color="#222222", pad=10)
    ax.set_xlabel(x_label, fontsize=10, color="#333333")
    ax.set_ylabel(y_label, fontsize=10, color="#333333")
    ax.tick_params(axis="both", colors="#444444", labelsize=9)
    ax.grid(True, which="major", color="#d8d8d8", linewidth=0.7)
    ax.grid(True, which="minor", color="#eeeeee", linewidth=0.45)
    ax.minorticks_on()
    ax.set_axisbelow(True)
    ax.margins(x=0.04, y=0.04)

    for spine in ax.spines.values():
        spine.set_color("#555555")
        spine.set_linewidth(0.9)

    if show_legend:
        ax.legend(
            title=color_by,
            frameon=True,
            facecolor="white",
            edgecolor="#c0c0c0",
            fontsize=8,
            title_fontsize=8,
            loc="best",
        )

    fig.tight_layout()
    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buffer.getvalue()


def calculate_design_line(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    design_line: str,
) -> tuple[list[float], list[float], str] | None:
    if design_line == "Off":
        return None

    clean = data[[x_column, y_column]].dropna().copy()
    if len(clean) < 3:
        return None

    x_values = clean[x_column].astype(float).tolist()
    y_values = clean[y_column].astype(float).tolist()
    n = len(clean)
    mean_x = sum(x_values) / n
    mean_y = sum(y_values) / n
    sxx = sum((depth - mean_y) ** 2 for depth in y_values)

    min_depth = min(y_values)
    max_depth = max(y_values)
    if math.isclose(min_depth, max_depth):
        line_y = [min_depth, max_depth]
    else:
        step_count = 79
        line_y = [min_depth + (max_depth - min_depth) * index / step_count for index in range(step_count + 1)]

    if math.isclose(sxx, 0.0):
        sample_variance = sum((value - mean_x) ** 2 for value in x_values) / (n - 1)
        standard_error = math.sqrt(sample_variance / n)
        t_value = t_critical_one_sided_95(n - 1)
        mean_line = [mean_x for _ in line_y]
        if design_line == "Lower cautious estimate":
            line_x = [value - t_value * standard_error for value in mean_line]
        elif design_line == "Upper cautious estimate":
            line_x = [value + t_value * standard_error for value in mean_line]
        else:
            line_x = mean_line
        return line_x, line_y, design_line_label(design_line)

    slope = sum((depth - mean_y) * (value - mean_x) for depth, value in zip(y_values, x_values)) / sxx
    intercept = mean_x - slope * mean_y
    fitted = [intercept + slope * depth for depth in y_values]
    residual_sum_squares = sum((value - fit) ** 2 for value, fit in zip(x_values, fitted))
    degrees_freedom = n - 2
    residual_standard_error = math.sqrt(residual_sum_squares / degrees_freedom) if degrees_freedom > 0 else 0.0
    t_value = t_critical_one_sided_95(degrees_freedom)

    mean_line = [intercept + slope * depth for depth in line_y]
    if design_line == "Mean trend":
        return mean_line, line_y, design_line_label(design_line)

    confidence_width = [
        t_value * residual_standard_error * math.sqrt((1 / n) + ((depth - mean_y) ** 2 / sxx))
        for depth in line_y
    ]
    if design_line == "Lower cautious estimate":
        line_x = [value - width for value, width in zip(mean_line, confidence_width)]
    elif design_line == "Upper cautious estimate":
        line_x = [value + width for value, width in zip(mean_line, confidence_width)]
    else:
        return None

    return line_x, line_y, design_line_label(design_line)


def design_line_label(design_line: str) -> str:
    if design_line == "Lower cautious estimate":
        return "Lower 95% cautious line"
    if design_line == "Upper cautious estimate":
        return "Upper 95% cautious line"
    return "Mean trend"


def t_critical_one_sided_95(degrees_freedom: int) -> float:
    if degrees_freedom <= 0:
        return 0.0

    table = {
        1: 6.314,
        2: 2.920,
        3: 2.353,
        4: 2.132,
        5: 2.015,
        6: 1.943,
        7: 1.895,
        8: 1.860,
        9: 1.833,
        10: 1.812,
        11: 1.796,
        12: 1.782,
        13: 1.771,
        14: 1.761,
        15: 1.753,
        16: 1.746,
        17: 1.740,
        18: 1.734,
        19: 1.729,
        20: 1.725,
        21: 1.721,
        22: 1.717,
        23: 1.714,
        24: 1.711,
        25: 1.708,
        26: 1.706,
        27: 1.703,
        28: 1.701,
        29: 1.699,
        30: 1.697,
        40: 1.684,
        60: 1.671,
        120: 1.658,
    }
    if degrees_freedom in table:
        return table[degrees_freedom]
    if degrees_freedom > 120:
        return 1.645

    larger_keys = [key for key in table if key > degrees_freedom]
    return table[min(larger_keys)] if larger_keys else 1.645


@st.cache_data(show_spinner=False)
def build_psd_png(data: pd.DataFrame, title: str, design_line: str = "Off") -> bytes:
    fig, ax = plt.subplots(figsize=(8.2, 5.8), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    curve_ids = [curve_id for curve_id in sorted(data["PSD_SAMPLE_ID"].dropna().unique())]
    show_legend = 1 < len(curve_ids) <= 12

    for index, curve_id in enumerate(curve_ids):
        group = data[data["PSD_SAMPLE_ID"] == curve_id].sort_values("GRAT_SIZE_NUM")
        ax.plot(
            group["GRAT_SIZE_NUM"],
            group["GRAT_PERP_NUM"],
            marker="o",
            markersize=3.2,
            linewidth=1.25,
            color=SCIENTIFIC_PALETTE[index % len(SCIENTIFIC_PALETTE)],
            alpha=0.9,
            label=str(curve_id),
        )

    statistical_curve = calculate_psd_design_line(data, design_line)
    if statistical_curve is not None:
        line_x, line_y, label = statistical_curve
        ax.plot(
            line_x,
            line_y,
            color=DESIGN_LINE_COLOR,
            linewidth=2.2,
            linestyle="--",
            label=label,
            zorder=6,
        )
        show_legend = True

    ax.set_xscale("log")
    ax.set_xlim(left=max(data["GRAT_SIZE_NUM"].min() * 0.75, 0.0005), right=data["GRAT_SIZE_NUM"].max() * 1.25)
    ax.set_ylim(0, 100)
    ax.set_title(title, fontsize=11, weight="semibold", color="#222222", pad=10)
    ax.set_xlabel("Particle size (mm)", fontsize=10, color="#333333")
    ax.set_ylabel("Percentage passing (%)", fontsize=10, color="#333333")
    ax.tick_params(axis="both", colors="#444444", labelsize=9)
    ax.grid(True, which="major", color="#d8d8d8", linewidth=0.7)
    ax.grid(True, which="minor", color="#eeeeee", linewidth=0.45)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_color("#555555")
        spine.set_linewidth(0.9)

    if show_legend:
        ax.legend(
            title="PSD curve",
            frameon=True,
            facecolor="white",
            edgecolor="#c0c0c0",
            fontsize=7,
            title_fontsize=8,
            loc="best",
        )

    fig.tight_layout()
    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buffer.getvalue()


def calculate_psd_design_line(
    data: pd.DataFrame,
    design_line: str,
) -> tuple[list[float], list[float], str] | None:
    if design_line == "Off":
        return None

    clean = data[["PSD_SAMPLE_ID", "GRAT_SIZE_NUM", "GRAT_PERP_NUM"]].dropna().copy()
    if clean["PSD_SAMPLE_ID"].nunique() < 3:
        return None

    line_x: list[float] = []
    line_y: list[float] = []
    for size, group in clean.groupby("GRAT_SIZE_NUM"):
        values = group.drop_duplicates("PSD_SAMPLE_ID")["GRAT_PERP_NUM"].astype(float).tolist()
        n = len(values)
        if n < 3:
            continue

        mean_value = sum(values) / n
        if design_line == "Mean trend":
            estimate = mean_value
        else:
            variance = sum((value - mean_value) ** 2 for value in values) / (n - 1)
            standard_error = math.sqrt(variance / n)
            width = t_critical_one_sided_95(n - 1) * standard_error
            if design_line == "Lower cautious estimate":
                estimate = mean_value - width
            elif design_line == "Upper cautious estimate":
                estimate = mean_value + width
            else:
                return None

        line_x.append(float(size))
        line_y.append(min(max(estimate, 0.0), 100.0))

    if len(line_x) < 2:
        return None

    ordered = sorted(zip(line_x, line_y), key=lambda pair: pair[0])
    return [pair[0] for pair in ordered], [pair[1] for pair in ordered], design_line_label(design_line)


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return slug or "graph"


if __name__ == "__main__":
    main()
