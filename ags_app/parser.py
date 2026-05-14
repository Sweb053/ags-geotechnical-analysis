from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path
from typing import BinaryIO

import pandas as pd


@dataclass(frozen=True)
class ParsedAGS:
    """Parsed AGS group tables keyed by upper-case group name."""

    tables: dict[str, pd.DataFrame]
    source_name: str

    def get(self, group: str) -> pd.DataFrame:
        return self.tables.get(group.upper(), pd.DataFrame())


def parse_uploaded_file(file_name: str, content: bytes) -> ParsedAGS:
    suffix = Path(file_name).suffix.lower()
    if suffix == ".xlsx":
        return ParsedAGS(parse_ags_xlsx(content), file_name)
    return ParsedAGS(parse_ags_text(content), file_name)


def parse_ags_file(path: str | Path) -> ParsedAGS:
    file_path = Path(path)
    content = file_path.read_bytes()
    return parse_uploaded_file(file_path.name, content)


def parse_ags_text(content: bytes) -> dict[str, pd.DataFrame]:
    text = _decode_text(content)
    _raise_csv_field_limit()
    reader = _csv_reader(text)

    tables: dict[str, list[dict[str, str | None]]] = {}
    headings: dict[str, list[str]] = {}
    current_group: str | None = None

    for raw_row in reader:
        if not raw_row:
            continue

        row = [cell.strip() if isinstance(cell, str) else cell for cell in raw_row]
        record_type = (row[0] or "").strip().upper()

        if record_type == "GROUP" and len(row) > 1:
            current_group = row[1].strip().upper()
            tables.setdefault(current_group, [])
            continue

        if current_group is None:
            continue

        if record_type == "HEADING":
            headings[current_group] = [cell.strip() for cell in row[1:] if cell.strip()]
            continue

        if record_type != "DATA":
            continue

        header = headings.get(current_group)
        if not header:
            continue

        values = _pad_or_trim(row[1:], len(header))
        tables[current_group].append(dict(zip(header, values)))

    parsed = {}
    for group, rows in tables.items():
        parsed[group] = pd.DataFrame(rows, columns=headings.get(group))
    return parsed


def parse_ags_xlsx(content: bytes | BinaryIO) -> dict[str, pd.DataFrame]:
    data = content if hasattr(content, "read") else BytesIO(content)
    workbook = pd.read_excel(data, sheet_name=None, header=None, dtype=str, engine="openpyxl")
    parsed: dict[str, pd.DataFrame] = {}

    for sheet_name, raw in workbook.items():
        raw = raw.dropna(how="all").dropna(axis=1, how="all")
        if raw.empty:
            continue

        marker_col = raw.iloc[:, 0].astype(str).str.strip().str.upper()
        heading_rows = raw.index[marker_col == "HEADING"].tolist()
        if not heading_rows:
            continue

        heading_idx = heading_rows[0]
        heading_values = raw.loc[heading_idx].tolist()
        headers = [str(value).strip() for value in heading_values[1:] if pd.notna(value) and str(value).strip()]
        if not headers:
            continue

        data_rows = raw.loc[raw.index > heading_idx]
        data_rows = data_rows[data_rows.iloc[:, 0].astype(str).str.strip().str.upper() == "DATA"]
        if data_rows.empty:
            parsed[sheet_name.upper()] = pd.DataFrame(columns=headers)
            continue

        records = []
        for _, row in data_rows.iterrows():
            values = _pad_or_trim(row.tolist()[1:], len(headers))
            records.append(dict(zip(headers, values)))

        parsed[sheet_name.upper()] = pd.DataFrame(records)

    return parsed


def _decode_text(content: bytes) -> str:
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="replace")


def _csv_reader(text: str) -> csv.reader:
    sample = text[:8192]
    first_line = next((line for line in sample.splitlines() if line.strip()), "")
    for delimiter in (",", "\t", ";", "|"):
        try:
            first_row = next(csv.reader([first_line], delimiter=delimiter))
        except csv.Error:
            continue
        if len(first_row) > 1 and first_row[0].strip().upper() == "GROUP":
            return csv.reader(StringIO(text), delimiter=delimiter)

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", "|"])
    except csv.Error:
        dialect = csv.excel
    return csv.reader(StringIO(text), dialect)


def _raise_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit = int(limit / 10)


def _pad_or_trim(values: list[object], size: int) -> list[str | None]:
    cleaned = [None if pd.isna(value) else str(value).strip() for value in values]
    if len(cleaned) < size:
        cleaned.extend([None] * (size - len(cleaned)))
    return cleaned[:size]
