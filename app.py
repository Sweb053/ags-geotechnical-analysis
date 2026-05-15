from __future__ import annotations

import base64
import html
from io import BytesIO
import json
import math
from pathlib import Path
import re
import uuid

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import transforms
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
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
    "Custom line",
]
UI_NAVY = "#002b5b"
UI_WHITE = "#ffffff"
UI_PANEL = "#f6f7f9"
UI_LINE = "#e0e0e0"
UI_INK = "#2b2d42"
UI_RED = "#d90429"
RUNTIME_DATA_DIR = Path(".ags_runtime")
LOGO_PATH = Path("assets/ags-logo.png")


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
    restore_cached_ags_from_query()
    requested_screen = st.query_params.get("screen")
    valid_screens = {
        "home",
        "spt",
        "ivan",
        "ucs",
        "rqd",
        "atterberg",
        "pointload",
        "psd",
        "groundwater",
        "map",
        "geological_model",
        "summary_stats",
        "bre_sulphate",
    }
    if requested_screen in valid_screens:
        st.session_state["screen"] = requested_screen
        st.query_params.clear()

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


def persist_ags_session(source_name: str, content: bytes) -> None:
    token = st.session_state.get("ags_data_token")
    if not token:
        token = uuid.uuid4().hex
        st.session_state["ags_data_token"] = token

    RUNTIME_DATA_DIR.mkdir(exist_ok=True)
    (RUNTIME_DATA_DIR / f"{token}.bin").write_bytes(content)
    (RUNTIME_DATA_DIR / f"{token}.json").write_text(
        json.dumps({"source_name": source_name}),
        encoding="utf-8",
    )


def restore_cached_ags_from_query() -> None:
    if "ags_content" in st.session_state:
        return

    token = st.query_params.get("data_token")
    if not token or not re.fullmatch(r"[0-9a-f]{32}", token):
        return

    data_path = RUNTIME_DATA_DIR / f"{token}.bin"
    meta_path = RUNTIME_DATA_DIR / f"{token}.json"
    if not data_path.exists() or not meta_path.exists():
        return

    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        metadata = {}
    st.session_state["ags_data_token"] = token
    st.session_state["ags_source_name"] = str(metadata.get("source_name") or data_path.name)
    st.session_state["ags_content"] = data_path.read_bytes()


@st.cache_data(show_spinner=False)
def load_logo_data_url(path: str) -> str:
    logo_path = Path(path)
    if not logo_path.exists():
        return ""
    encoded = base64.b64encode(logo_path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def inject_custom_css() -> None:
    st.markdown(
        f"""
        <style>
        :root {{
            --ags-navy: #3a3a3a;
            --ags-white: {UI_WHITE};
            --ags-panel: #f4f4f3;
            --ags-soft: #f8f8f7;
            --ags-line: #dedede;
            --ags-ink: #222222;
            --ags-muted: #6f6f6f;
            --ags-active: #3a3a3a;
        }}
        .stApp {{
            background: #f5f5f4;
            color: var(--ags-ink);
        }}
        .block-container {{
            max-width: 1220px;
            padding-top: 1.1rem;
            padding-bottom: 3rem;
        }}
        h1, h2, h3 {{
            color: var(--ags-ink);
            letter-spacing: 0;
        }}
        .ags-hero {{
            margin: 3rem 0 2rem 0;
            display: flex;
            align-items: center;
            gap: 1.35rem;
        }}
        .ags-logo {{
            width: clamp(82px, 9vw, 116px);
            height: clamp(82px, 9vw, 116px);
            border-radius: 999px;
            border: 1px solid #dedede;
            background: #ffffff;
            object-fit: cover;
            padding: 0.35rem;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.035);
            flex: 0 0 auto;
        }}
        .ags-hero-copy {{
            min-width: 0;
        }}
        .ags-kicker {{
            margin: 0 0 0.4rem 0;
            color: var(--ags-muted);
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }}
        .ags-title {{
            margin: 0;
            font-size: clamp(2rem, 3.2vw, 3rem);
            line-height: 1.08;
            font-weight: 760;
            color: #161616;
        }}
        .ags-subtitle {{
            margin: 0.7rem 0 0 0;
            color: var(--ags-muted);
            max-width: 760px;
            font-size: 1rem;
            line-height: 1.6;
        }}
        .ags-status {{
            border: 1px solid var(--ags-line);
            border-radius: 16px;
            padding: 0.85rem 1rem;
            background: #ffffff;
            display: flex;
            gap: 0.75rem;
            align-items: center;
            margin: 1rem 0 2.8rem 0;
        }}
        .ags-status-dot {{
            width: 0.75rem;
            height: 0.75rem;
            border-radius: 999px;
            background: #3f7d5b;
            box-shadow: 0 0 0 4px rgba(63, 125, 91, 0.12);
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
            color: var(--ags-muted);
            font-size: 0.88rem;
            overflow-wrap: anywhere;
        }}
        .ags-section-title {{
            color: #161616;
            font-weight: 760;
            margin: 2.5rem 0 1.6rem 0;
            font-size: 1.7rem;
        }}
        .ags-section-copy {{
            color: var(--ags-muted);
            margin: 0 0 0.9rem 0;
        }}
        .ags-module-panel {{
            border: 1px solid var(--ags-line);
            border-radius: 26px;
            padding: 2.15rem 2.25rem;
            background: #ffffff;
            min-height: 360px;
            display: grid;
            grid-template-columns: 1fr 0.95fr;
            gap: 2.2rem;
            align-items: center;
        }}
        .ags-module-panel h3 {{
            margin: 0 0 1rem 0;
            font-size: 1.35rem;
            font-weight: 760;
            color: #161616;
        }}
        .ags-module-panel p {{
            color: #343434;
            font-size: 1rem;
            line-height: 1.55;
            margin: 0 0 1.15rem 0;
        }}
        .ags-module-meta {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
        }}
        .ags-count {{
            display: inline-block;
            padding: 0.32rem 0.7rem;
            border-radius: 999px;
            background: #f0f0ef;
            border: 1px solid #e2e2e0;
            color: #4c4c4c;
            font-size: 0.82rem;
            font-weight: 650;
        }}
        .ags-software-visual {{
            border: 1px solid #e4e4e2;
            border-radius: 18px;
            background: #fafafa;
            padding: 1rem;
        }}
        .ags-window-bar {{
            height: 1.15rem;
            display: flex;
            gap: 0.32rem;
            margin-bottom: 0.85rem;
        }}
        .ags-window-dot {{
            width: 0.45rem;
            height: 0.45rem;
            border-radius: 999px;
            background: #c9c9c7;
        }}
        .ags-plot-preview {{
            height: 210px;
            border: 1px solid #dfdfdc;
            border-radius: 12px;
            background:
                linear-gradient(#eeeeeb 1px, transparent 1px),
                linear-gradient(90deg, #eeeeeb 1px, transparent 1px),
                #ffffff;
            background-size: 36px 36px;
            position: relative;
            overflow: hidden;
        }}
        .ags-plot-line {{
            position: absolute;
            left: 12%;
            right: 10%;
            top: 45%;
            height: 2px;
            background: #555555;
            transform: rotate(-12deg);
            transform-origin: center;
        }}
        .ags-plot-dot {{
            position: absolute;
            width: 0.55rem;
            height: 0.55rem;
            border-radius: 999px;
            background: #4c4c4c;
        }}
        .ags-module-sidebar-note {{
            color: var(--ags-muted);
            font-size: 0.9rem;
            margin-bottom: 0.9rem;
        }}
        .ags-workspace {{
            display: grid;
            grid-template-columns: minmax(230px, 0.31fr) minmax(0, 0.69fr);
            gap: 2rem;
            align-items: start;
        }}
        .ags-workspace-list {{
            display: grid;
            gap: 0.85rem;
        }}
        .ags-workspace-item {{
            display: block;
            border: 1px solid #e4e4e2;
            border-radius: 12px;
            background: #ffffff;
            color: #222222;
            min-height: 3.05rem;
            padding: 0.88rem 1rem;
            text-align: center;
            text-decoration: none;
            font-weight: 650;
        }}
        .ags-workspace-item:link,
        .ags-workspace-item:visited {{
            color: #222222;
            text-decoration: none;
        }}
        .ags-workspace-item:hover {{
            border-color: var(--ags-active);
            background: var(--ags-active);
            color: #ffffff;
            text-decoration: none;
        }}
        .ags-workspace-item.disabled {{
            pointer-events: none;
            background: #eeeeec;
            border-color: #eeeeec;
            color: #9a9a98;
        }}
        .ags-workspace-panels {{
            position: relative;
        }}
        .ags-hover-panel {{
            display: none;
        }}
        .ags-hover-panel.panel-0 {{
            display: grid;
        }}
        .ags-workspace:has(.ags-workspace-item:hover) .ags-hover-panel {{
            display: none;
        }}
        .ags-workspace:has(.item-0:hover) .panel-0,
        .ags-workspace:has(.item-1:hover) .panel-1,
        .ags-workspace:has(.item-2:hover) .panel-2,
        .ags-workspace:has(.item-3:hover) .panel-3,
        .ags-workspace:has(.item-4:hover) .panel-4,
        .ags-workspace:has(.item-5:hover) .panel-5,
        .ags-workspace:has(.item-6:hover) .panel-6,
        .ags-workspace:has(.item-7:hover) .panel-7,
        .ags-workspace:has(.item-8:hover) .panel-8,
        .ags-workspace:has(.item-9:hover) .panel-9,
        .ags-workspace:has(.item-10:hover) .panel-10,
        .ags-workspace:has(.item-11:hover) .panel-11 {{
            display: grid;
        }}
        @media (max-width: 820px) {{
            .ags-hero {{
                align-items: flex-start;
                gap: 1rem;
            }}
            .ags-workspace,
            .ags-module-panel {{
                grid-template-columns: 1fr;
            }}
        }}
        div.stButton > button {{
            border-radius: 12px;
            border: 1px solid #e4e4e2;
            background: #ffffff;
            color: #222222;
            font-weight: 650;
            min-height: 3.05rem;
            box-shadow: none;
        }}
        div.stButton > button:hover {{
            border-color: var(--ags-active);
            background: var(--ags-active);
            color: #ffffff;
        }}
        div.stButton > button[kind="primary"] {{
            border-color: var(--ags-active);
            background: var(--ags-active);
            color: #ffffff;
        }}
        div.stButton > button:disabled {{
            background: #eeeeec;
            border-color: #eeeeec;
            color: #9a9a98;
        }}
        div[data-testid="stFileUploader"] section {{
            background: var(--ags-soft);
            border: 1px solid var(--ags-line);
            border-radius: 16px;
        }}
        div[data-testid="stExpander"] {{
            border-color: var(--ags-line);
            border-radius: 16px;
            background: #ffffff;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_home_screen() -> None:
    logo_data_url = load_logo_data_url(str(LOGO_PATH))
    logo_markup = f'<img class="ags-logo" src="{logo_data_url}" alt="AGS Geotechnical Analysis logo" />' if logo_data_url else ""
    st.markdown(
        f"""
        <div class="ags-hero">
            {logo_markup}
            <div class="ags-hero-copy">
                <p class="ags-kicker">AGS data toolkit</p>
                <h1 class="ags-title">AGS Geotechnical Analysis</h1>
                <p class="ags-subtitle">Upload AGS data, inspect geotechnical tests, filter by geology, export scientific plots, and review mapped ground conditions.</p>
            </div>
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
        persist_ags_session(uploaded.name, uploaded.getvalue())
    elif local_path is not None:
        st.session_state["ags_source_name"] = str(local_path)
        content = local_path.read_bytes()
        st.session_state["ags_content"] = content
        persist_ags_session(str(local_path), content)

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
    available_modules = [module for module in modules if not bool(module["disabled"])]
    featured = available_modules[0] if available_modules else modules[0]
    if "ags_content" in st.session_state and not st.session_state.get("ags_data_token"):
        persist_ags_session(
            str(st.session_state.get("ags_source_name", "uploaded_ags_data")),
            st.session_state["ags_content"],
        )
    token = st.session_state.get("ags_data_token", "")
    sidebar_items = []
    panels = []
    for index, module in enumerate(modules):
        item_class = f"ags-workspace-item item-{index}"
        title = html.escape(str(module["title"]))
        if bool(module["disabled"]):
            sidebar_items.append(f'<span class="{item_class} disabled">{title}</span>')
        else:
            screen = html.escape(str(module["screen"]))
            token_query = f"&data_token={html.escape(str(token))}" if token else ""
            sidebar_items.append(f'<a class="{item_class}" href="?screen={screen}{token_query}">{title}</a>')
        panels.append(render_module_overview_panel(module, modules, index))

    st.html(
        f"""
        <div class="ags-workspace">
            <div>
                <p class="ags-module-sidebar-note">Open an analysis workspace</p>
                <div class="ags-workspace-list">
                    {''.join(sidebar_items)}
                </div>
            </div>
            <div class="ags-workspace-panels">
                {''.join(panels)}
            </div>
        </div>
        """
    )


def render_module_overview_panel(module: dict[str, object], modules: list[dict[str, object]], index: int) -> str:
    title = html.escape(str(module["title"]))
    description = html.escape(str(module["description"]))
    summary = html.escape(module_summary(str(module["screen"])))
    count = html.escape(str(module["count"]))
    available_count = sum(1 for candidate in modules if not bool(candidate["disabled"]))
    return f"""
        <div class="ags-module-panel ags-hover-panel panel-{index}">
            <div>
                <h3>{title}</h3>
                <p>{description}</p>
                <p>{summary}</p>
                <div class="ags-module-meta">
                    <span class="ags-count">{count}</span>
                    <span class="ags-count">{available_count} modules available</span>
                    <span class="ags-count">AGS linked analysis</span>
                </div>
            </div>
            <div class="ags-software-visual">
                <div class="ags-window-bar">
                    <span class="ags-window-dot"></span>
                    <span class="ags-window-dot"></span>
                    <span class="ags-window-dot"></span>
                </div>
                <div class="ags-plot-preview">
                    <span class="ags-plot-line"></span>
                    <span class="ags-plot-dot" style="left: 18%; top: 64%;"></span>
                    <span class="ags-plot-dot" style="left: 30%; top: 52%;"></span>
                    <span class="ags-plot-dot" style="left: 43%; top: 46%;"></span>
                    <span class="ags-plot-dot" style="left: 56%; top: 39%;"></span>
                    <span class="ags-plot-dot" style="left: 70%; top: 31%;"></span>
                    <span class="ags-plot-dot" style="left: 82%; top: 24%;"></span>
                </div>
            </div>
        </div>
        """


def module_summary(screen: str) -> str:
    summaries = {
        "spt": "Review raw or corrected N60 values against depth, isolate material classes or model units, and export presentation-ready SPT plots.",
        "ivan": "Plot undrained shear strength, Cu, from hand shear vane records and compare selected investigations against matched geological units.",
        "ucs": "Assess UCS results in MPa by depth, filter by geology or material class, and review the linked sample rows behind each plotted value.",
        "rqd": "Inspect RQD variation through rock core runs, separating bedrock units and matched strata for rock mass quality review.",
        "atterberg": "Compare liquid limit, plastic limit, and plasticity index datasets with consistent filtering across investigations and geology.",
        "pointload": "Plot point load strength index by sample depth and use geology filters to develop rock strength summaries.",
        "psd": "Review particle size distribution curves with soil fraction bands, selected sample curves, and optional statistical design curves.",
        "groundwater": "Review groundwater strike depths by investigation without geology filters, focused on strike and post-strike observations.",
        "map": "Map selected investigations over aerial or static basemaps and label boreholes with merged geology depth ranges.",
        "geological_model": "Build a searchable geological model from GEOL strata, material classes, model units, and bedrock lithology descriptions.",
        "summary_stats": "Create cautious estimates and means for selected modules, including custom combined geology groups for reporting tables.",
        "bre_sulphate": "Calculate BRE DS and ACEC classes from GCHM chemistry and summarise filtered sulphate design classifications.",
    }
    return summaries.get(screen, "Open this workspace to review filtered AGS data, linked geology, plots, and exports.")


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
        spt_value_label = "Corrected SPT N60"
        spt_title_value = "Corrected SPT N60"
        corrected_count = int(spt[spt_value_column].notna().sum()) if spt_value_column in spt.columns else 0
        if corrected_count == 0:
            st.warning("No corrected N60 values are available because `ISPT_ERAT` is missing or non-numeric.")
    else:
        spt_value_column = "ISPT_MAIN_NUM"
        spt_value_label = "SPT blow count, N"
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
            value_label="Liquid limit, LL (%)",
            title_prefix="Liquid Limit",
        )

    with tab_pl:
        render_atterberg_pair(
            filtered_by_unit,
            value_column="LLPL_PL_NUM",
            value_label="Plastic limit, PL (%)",
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
    selected_loca = render_investigation_filter(groundwater, "groundwater strike records")
    filtered = groundwater[groundwater["LOCA_ID"].isin(selected_loca)].copy()

    tab_all, tab_data = st.tabs(["All Investigations", "Matched Data"])

    with tab_all:
        render_groundwater_plot(
            filtered,
            title="Groundwater Strike Depth by Investigation",
            color_by="LOCA_ID",
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
            filtered[matched_data_columns(filtered, columns)],
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
    sample_count = psd["PSD_SAMPLE_ID"].nunique()

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
    material_classes = sorted(data["MATERIAL_CLASS"].dropna().unique()) if "MATERIAL_CLASS" in data.columns else []
    model_units = sorted(data["MODEL_UNIT"].dropna().unique()) if "MODEL_UNIT" in data.columns else []
    bedrock_types = sorted(data["BEDROCK_TYPE"].dropna().unique()) if "BEDROCK_TYPE" in data.columns else []

    investigation_mode = st.radio(
        "Investigation filter",
        ["All investigations", "Choose investigations"],
        horizontal=True,
    )
    if investigation_mode == "All investigations":
        selected_loca = loca_ids
    else:
        selected_loca = st.multiselect(f"Investigations with valid {record_label}", loca_ids, default=[])

    geology_mode = "All geology"
    selected_units: list[str] = []

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


def render_investigation_filter(data: pd.DataFrame, record_label: str) -> list[str]:
    st.subheader("Filters")
    loca_ids = sorted(data["LOCA_ID"].dropna().unique())
    investigation_mode = st.radio(
        "Investigation filter",
        ["All investigations", "Choose investigations"],
        horizontal=True,
        key=f"{slugify(record_label)}_investigation_filter",
    )
    if investigation_mode == "All investigations":
        return loca_ids
    return st.multiselect(
        f"Investigations with valid {record_label}",
        loca_ids,
        default=[],
        key=f"{slugify(record_label)}_selected_loca",
    )


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
    x_label: str = "SPT blow count, N",
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
        x_label="Undrained shear strength, Cu",
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
        x_label="Unconfined compressive strength (MPa)",
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
        x_label="Rock quality designation, RQD (%)",
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
        x_label="Point load strength index, Is50",
        y_label="Depth below ground level (m)",
    )


def render_groundwater_plot(
    data: pd.DataFrame,
    title: str,
    color_by: str,
) -> None:
    plot_data = data.copy()
    loca_ids = sorted(plot_data["LOCA_ID"].dropna().unique())
    tick_labels = {index + 1: loca_id for index, loca_id in enumerate(loca_ids)}
    loca_to_plot_num = {loca_id: index for index, loca_id in tick_labels.items()}
    plot_data["GROUNDWATER_PLOT_NUM"] = plot_data["LOCA_ID"].map(loca_to_plot_num)
    render_depth_scatter_plot(
        data=plot_data,
        title=title,
        color_by=color_by,
        x_column="GROUNDWATER_PLOT_NUM",
        y_column="WSTG_DPTH_NUM",
        x_label="Investigation",
        y_label="Groundwater strike depth below ground level (m)",
        x_tick_labels=tick_labels,
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
    custom_lines: tuple[tuple[str, tuple[tuple[float, float], ...]], ...] = tuple()
    if design_line == "Custom line":
        custom_lines = render_custom_design_line_editor(
            title,
            "Particle size (mm)",
            "Percentage passing (%)",
            positive_x=True,
        )
        if not custom_lines:
            st.warning("Add at least two valid positive particle-size rows with the same line name to plot a custom design line.")
    show_design_line = design_line != "Off" and (design_line != "Custom line" or bool(custom_lines))
    if design_line != "Custom line" and show_design_line and data["PSD_SAMPLE_ID"].nunique() < 3:
        st.warning("At least three selected PSD curves are needed for a statistical design curve.")
        show_design_line = False

    png_bytes = build_psd_png(
        data,
        title,
        design_line if show_design_line else "Off",
        custom_lines,
    )
    st.image(png_bytes, use_container_width=True)
    st.download_button(
        "Download graph PNG",
        data=png_bytes,
        file_name=f"{slugify(title)}.png",
        mime="image/png",
    )
    if show_design_line and design_line != "Custom line":
        st.caption(
            "The PSD design curve is recalculated from the selected curves and uses a one-sided 95% "
            "confidence bound at each particle size."
        )
    if show_design_line and design_line == "Custom line":
        st.caption("The custom design line is drawn from the coordinate table above and is included in the PNG export.")


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

    map_html = build_leaflet_aerial_map_html(aerial_data)
    components.html(map_html, height=560)
    st.caption(
        "Aerial imagery is loaded from Esri World Imagery. Labels show the investigation name and selected geology ranges."
    )


def build_leaflet_aerial_map_html(aerial_data: pd.DataFrame) -> str:
    records = []
    for _, row in aerial_data.iterrows():
        records.append(
            {
                "id": str(row["LOCA_ID"]),
                "lat": float(row["LATITUDE"]),
                "lon": float(row["LONGITUDE"]),
                "label": str(row["MAP_LABEL"]),
                "geology": str(row["GEOLOGY_LABEL"]),
                "highlighted": bool(row["HAS_SELECTED_GEOLOGY"]),
            }
        )
    payload = json.dumps(records).replace("</", "<\\/")
    return f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <style>
    html, body, #map {{ height: 100%; margin: 0; background: #ffffff; }}
    .ags-map-label {{
      color: #1f2933;
      background: rgba(255, 255, 255, 0.86);
      border: 1px solid rgba(31, 41, 51, 0.28);
      border-radius: 4px;
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.18);
      font: 12px/1.25 Arial, sans-serif;
      padding: 3px 5px;
      white-space: pre-line;
    }}
    .ags-map-label::before {{ display: none; }}
  </style>
</head>
<body>
  <div id="map"></div>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    const points = {payload};
    const map = L.map("map", {{ scrollWheelZoom: true }});
    L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}", {{
      maxZoom: 19,
      attribution: "Tiles &copy; Esri"
    }}).addTo(map);

    const bounds = [];
    points.forEach((point) => {{
      const latLng = [point.lat, point.lon];
      bounds.push(latLng);
      const fillColor = point.highlighted ? "#4C956C" : "#6B6D76";
      const radius = point.highlighted ? 7 : 5;
      const marker = L.circleMarker(latLng, {{
        radius,
        color: "#ffffff",
        weight: 1.4,
        fillColor,
        fillOpacity: 0.95
      }}).addTo(map);
      const popupText = point.geology ? `<strong>${{point.id}}</strong><br>${{point.geology}}` : `<strong>${{point.id}}</strong>`;
      marker.bindPopup(popupText);
      marker.bindTooltip(point.label, {{
        permanent: true,
        direction: "bottom",
        offset: [0, 10],
        opacity: 1,
        className: "ags-map-label"
      }});
    }});

    if (bounds.length === 1) {{
      map.setView(bounds[0], 17);
    }} else {{
      map.fitBounds(bounds, {{ padding: [40, 40], maxZoom: 18 }});
    }}
  </script>
</body>
</html>
"""


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
    x_tick_labels: dict[int, str] | None = None,
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
    custom_lines: tuple[tuple[str, tuple[tuple[float, float], ...]], ...] = tuple()
    if design_line == "Custom line":
        custom_lines = render_custom_design_line_editor(title, x_label, y_label)
        if not custom_lines:
            st.warning("Add at least two valid coordinate rows with the same line name to plot a custom design line.")
    show_design_line = design_line != "Off" and (design_line != "Custom line" or bool(custom_lines))
    if design_line != "Custom line" and show_design_line and len(data.dropna(subset=[x_column, y_column])) < 3:
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
        x_tick_labels,
        custom_lines,
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
    if show_design_line and design_line != "Custom line":
        st.caption(
            "The design line is recalculated from the records visible in this plot. "
            "The cautious estimate uses a one-sided 95% confidence bound on the fitted mean trend."
        )
    if show_design_line and design_line == "Custom line":
        st.caption("The custom design line is drawn from the coordinate table above and is included in the PNG export.")


def render_custom_design_line_editor(
    title: str,
    x_label: str,
    y_label: str,
    positive_x: bool = False,
) -> tuple[tuple[str, tuple[tuple[float, float], ...]], ...]:
    key = f"custom_design_line_{slugify(title)}"
    default_rows = pd.DataFrame(
        [
            {"Line": "Line 1", "X": None, "Y": None},
            {"Line": "Line 1", "X": None, "Y": None},
        ]
    )
    edited = st.data_editor(
        default_rows,
        key=key,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "Line": st.column_config.TextColumn("Line", help="Rows with the same line name are connected together."),
            "X": st.column_config.NumberColumn(f"X - {x_label}", format="%.3f"),
            "Y": st.column_config.NumberColumn(f"Y - {y_label}", format="%.3f"),
        },
    )
    return parse_custom_design_lines(edited, positive_x=positive_x)


def parse_custom_design_lines(
    rows: pd.DataFrame,
    positive_x: bool = False,
) -> tuple[tuple[str, tuple[tuple[float, float], ...]], ...]:
    if rows.empty or not {"Line", "X", "Y"}.issubset(rows.columns):
        return tuple()

    prepared = rows.copy()
    prepared["Line"] = prepared["Line"].fillna("Line 1").astype(str).str.strip()
    prepared.loc[prepared["Line"] == "", "Line"] = "Line 1"
    prepared["X"] = pd.to_numeric(prepared["X"], errors="coerce")
    prepared["Y"] = pd.to_numeric(prepared["Y"], errors="coerce")
    prepared = prepared.dropna(subset=["X", "Y"]).copy()
    if positive_x:
        prepared = prepared[prepared["X"] > 0].copy()
    if prepared.empty:
        return tuple()

    parsed_lines: list[tuple[str, tuple[tuple[float, float], ...]]] = []
    for line_name, group in prepared.groupby("Line", sort=False):
        points = tuple((float(row["X"]), float(row["Y"])) for _, row in group.iterrows())
        if len(points) >= 2:
            parsed_lines.append((str(line_name), points))
    return tuple(parsed_lines)


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
    x_tick_labels: dict[int, str] | None = None,
    custom_lines: tuple[tuple[str, tuple[tuple[float, float], ...]], ...] = tuple(),
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
    if custom_lines:
        plot_custom_design_lines(ax, custom_lines)
        show_legend = True

    ax.invert_yaxis()
    ax.set_title(title, fontsize=11, weight="semibold", color="#222222", pad=10)
    ax.set_xlabel(x_label, fontsize=10, color="#333333")
    ax.set_ylabel(y_label, fontsize=10, color="#333333")
    if x_tick_labels:
        ticks = sorted(x_tick_labels)
        rotation = 90 if len(ticks) > 24 else 45
        ax.set_xticks(ticks)
        ax.set_xticklabels([x_tick_labels[tick] for tick in ticks], rotation=rotation, ha="right")
    ax.tick_params(axis="both", colors="#444444", labelsize=9)
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


def plot_custom_design_lines(
    ax: plt.Axes,
    custom_lines: tuple[tuple[str, tuple[tuple[float, float], ...]], ...],
) -> None:
    for index, (line_name, points) in enumerate(custom_lines):
        if len(points) < 2:
            continue
        line_x = [point[0] for point in points]
        line_y = [point[1] for point in points]
        ax.plot(
            line_x,
            line_y,
            color=DESIGN_LINE_COLOR,
            linewidth=2.0,
            linestyle="-" if index == 0 else "--",
            marker="s",
            markersize=3.0,
            label=line_name,
            zorder=6,
        )


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
def build_psd_png(
    data: pd.DataFrame,
    title: str,
    design_line: str = "Off",
    custom_lines: tuple[tuple[str, tuple[tuple[float, float], ...]], ...] = tuple(),
) -> bytes:
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
    if custom_lines:
        plot_custom_design_lines(ax, custom_lines)
        show_legend = True

    ax.set_xscale("log")
    ax.set_xlim(
        left=min(max(data["GRAT_SIZE_NUM"].min() * 0.75, 0.0005), 0.001),
        right=max(data["GRAT_SIZE_NUM"].max() * 1.25, 100),
    )
    ax.set_ylim(0, 100)
    ax.set_title(title, fontsize=11, weight="semibold", color="#222222", pad=10)
    ax.set_xlabel("Particle size (mm)", fontsize=10, color="#333333")
    ax.set_ylabel("Percentage passing (%)", fontsize=10, color="#333333")
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", which="both", colors="#444444", labelsize=9, top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.tick_params(axis="y", colors="#444444", labelsize=9)
    ax.set_axisbelow(True)
    add_soil_fraction_axis(ax)

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

    fig.subplots_adjust(left=0.11, right=0.98, top=0.86, bottom=0.23)
    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buffer.getvalue()


def add_soil_fraction_axis(ax: plt.Axes) -> None:
    x_min, x_max = ax.get_xlim()
    major_bounds = [0.063, 2.0, 63.0]
    subdivision_bounds = [0.002, 0.2, 0.63, 6.3, 20.0]

    for boundary in major_bounds + subdivision_bounds:
        if x_min < boundary < x_max:
            line_width = 0.9 if boundary in major_bounds else 0.55
            ax.axvline(boundary, color="#5f5f5f", linewidth=line_width, alpha=0.75, zorder=0)

    transform = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    major_bands = [
        ("Clay and silt", 0.0005, 0.063),
        ("Sand", 0.063, 2.0),
        ("Gravel", 2.0, 63.0),
        ("Cobbles", 63.0, 200.0),
    ]
    sub_bands = [
        ("Fine", 0.063, 0.2),
        ("Medium", 0.2, 0.63),
        ("Coarse", 0.63, 2.0),
        ("Fine", 2.0, 6.3),
        ("Medium", 6.3, 20.0),
        ("Coarse", 20.0, 63.0),
    ]
    draw_fraction_bands(ax, major_bands, transform, y_bottom=-0.145, height=0.055, fontsize=7.0)
    draw_fraction_bands(ax, sub_bands, transform, y_bottom=-0.205, height=0.055, fontsize=6.5)


def draw_fraction_bands(
    ax: plt.Axes,
    bands: list[tuple[str, float, float]],
    transform,
    y_bottom: float,
    height: float,
    fontsize: float,
) -> None:
    x_min, x_max = ax.get_xlim()
    for label, start, end in bands:
        left = max(start, x_min)
        right = min(end, x_max)
        if left >= right:
            continue
        rect = plt.Rectangle(
            (left, y_bottom),
            right - left,
            height,
            transform=transform,
            facecolor="white",
            edgecolor="#5f5f5f",
            linewidth=0.65,
            clip_on=False,
        )
        ax.add_patch(rect)
        midpoint = math.sqrt(left * right)
        ax.text(
            midpoint,
            y_bottom + height / 2,
            label,
            transform=transform,
            ha="center",
            va="center",
            fontsize=fontsize,
            color="#222222",
            clip_on=False,
        )


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
