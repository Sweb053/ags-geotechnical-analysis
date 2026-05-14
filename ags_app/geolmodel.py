from __future__ import annotations

import re

import pandas as pd

from ags_app.common import required_table, to_number


REQUIRED_GEOL_COLUMNS = {"LOCA_ID", "GEOL_TOP", "GEOL_BASE", "GEOL_GEOL", "GEOL_DESC"}
BEDROCK_UNITS = {"ANHGS", "GLEN", "L", "MALP", "MORR"}
ROCK_TERMS = {
    "AMPHIBOLITE",
    "DIORITE",
    "GNEISS",
    "GRANITE",
    "GRANODIORITE",
    "MIGMATITE",
    "ORTHOGNEISS",
    "PELITE",
    "PSAMMITE",
    "QUARTZITE",
    "SCHIST",
    "SEMIPelite".upper(),
}
NON_BEDROCK_CAPITAL_WORDS = {
    "AND",
    "BEDROCK",
    "BOULDER",
    "BOULDERS",
    "CLAY",
    "COBBLE",
    "COBBLES",
    "GRAVEL",
    "PEAT",
    "SAND",
    "SILT",
    "TOPSOIL",
}


def build_geological_model(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    geol = required_table(tables, "GEOL", REQUIRED_GEOL_COLUMNS).copy()
    geol["LOCA_ID"] = geol["LOCA_ID"].astype(str).str.strip()
    geol["GEOL_GEOL"] = geol["GEOL_GEOL"].fillna("Unspecified").astype(str).str.strip()
    geol["GEOL_DESC"] = geol["GEOL_DESC"].fillna("").astype(str).str.strip()
    geol["GEOL_TOP_NUM"] = to_number(geol["GEOL_TOP"])
    geol["GEOL_BASE_NUM"] = to_number(geol["GEOL_BASE"])
    geol["THICKNESS_NUM"] = geol["GEOL_BASE_NUM"] - geol["GEOL_TOP_NUM"]
    geol = geol.dropna(subset=["LOCA_ID", "GEOL_TOP_NUM", "GEOL_BASE_NUM"]).copy()

    geol["MATERIAL_CLASS"] = geol.apply(
        lambda row: classify_material(row["GEOL_GEOL"], row["GEOL_DESC"]),
        axis=1,
    )
    geol["BEDROCK_TYPE"] = geol["GEOL_DESC"].map(extract_bedrock_type)
    geol.loc[geol["MATERIAL_CLASS"] != "Rock / Bedrock", "BEDROCK_TYPE"] = pd.NA
    geol["MODEL_UNIT"] = geol.apply(build_model_unit, axis=1)
    return geol.sort_values(["LOCA_ID", "GEOL_TOP_NUM"]).reset_index(drop=True)


def add_geological_model_fields(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    if "GEOL_GEOL" not in data.columns:
        data["GEOL_GEOL"] = "Unmatched"
    if "GEOL_DESC" not in data.columns:
        data["GEOL_DESC"] = ""

    data["MATERIAL_CLASS"] = data.apply(
        lambda row: "Unmatched"
        if str(row["GEOL_GEOL"]) == "Unmatched"
        else classify_material(row["GEOL_GEOL"], row["GEOL_DESC"]),
        axis=1,
    )
    data["BEDROCK_TYPE"] = data["GEOL_DESC"].map(extract_bedrock_type)
    data.loc[data["MATERIAL_CLASS"] != "Rock / Bedrock", "BEDROCK_TYPE"] = pd.NA
    data["MODEL_UNIT"] = data.apply(
        lambda row: "Unmatched" if str(row["GEOL_GEOL"]) == "Unmatched" else build_model_unit(row),
        axis=1,
    )
    return data


def classify_material(unit: object, description: object) -> str:
    unit_text = str(unit).upper()
    desc = str(description).upper()

    if "PEAT" in unit_text or "PEAT" in desc:
        return "Peat"
    if "TOP" in unit_text or "TOPSOIL" in desc or "MADE GROUND" in desc:
        return "Organic / Made Ground"
    if unit_text in BEDROCK_UNITS or (has_bedrock_language(desc) and extract_bedrock_type(desc)):
        return "Rock / Bedrock"

    primary_soil = find_primary_soil(desc)
    if primary_soil in {"SAND", "GRAVEL"}:
        return "Granular"
    if primary_soil in {"CLAY", "SILT"}:
        return "Cohesive"
    if "SAND" in desc or "GRAVEL" in desc:
        return "Granular"
    if "CLAY" in desc or "SILT" in desc:
        return "Cohesive"
    return "Unclassified"


def has_bedrock_language(description: str) -> bool:
    return any(
        token in description
        for token in (
            "DISCONTINUIT",
            "RECOVERED AS",
            "WEATHERED",
            "MEDIUM STRONG",
            "STRONG",
            "VERY STRONG",
            "WEAK ROCK",
        )
    )


def find_primary_soil(description: str) -> str | None:
    primary_description = description.split(".")[0]
    matches = list(re.finditer(r"\b(CLAYS?|SILTS?|SANDS?|GRAVELS?)\b", primary_description))
    if not matches:
        return None
    word = matches[-1].group(1)
    if word.startswith("CLAY"):
        return "CLAY"
    if word.startswith("SILT"):
        return "SILT"
    if word.startswith("SAND"):
        return "SAND"
    if word.startswith("GRAVEL"):
        return "GRAVEL"
    return None


def extract_bedrock_type(description: object) -> str | None:
    desc = str(description).upper()
    words = re.findall(r"\b[A-Z]{3,}\b", desc)
    rock_words = [word for word in words if word in ROCK_TERMS and word not in NON_BEDROCK_CAPITAL_WORDS]
    if rock_words:
        return " / ".join(dict.fromkeys(rock_words))

    return None


def build_model_unit(row: pd.Series) -> str:
    unit = str(row["GEOL_GEOL"])
    material = str(row["MATERIAL_CLASS"])
    bedrock = row.get("BEDROCK_TYPE")
    if material == "Rock / Bedrock" and not pd.isna(bedrock):
        return f"{unit} - {bedrock}"
    if material in {"Granular", "Cohesive", "Peat", "Organic / Made Ground"}:
        return f"{unit} - {material}"
    return unit
