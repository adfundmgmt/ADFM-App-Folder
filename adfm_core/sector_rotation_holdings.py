"""Holdings helpers for sector breadth."""

from __future__ import annotations

from io import BytesIO
from typing import List
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import pandas as pd


def xlsx_first_sheet_to_frame(content: bytes) -> pd.DataFrame:
    """Read the first worksheet of a simple XLSX file using the stdlib only."""
    if not content:
        return pd.DataFrame()

    with ZipFile(BytesIO(content)) as archive:
        shared: List[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root:
                pieces = [node.text or "" for node in item.iter() if node.tag.endswith("}t")]
                shared.append("".join(pieces))

        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        rel_map = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in rels
        }
        first_sheet = next(node for node in workbook.iter() if node.tag.endswith("}sheet"))
        rel_id = first_sheet.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
        target = rel_map[rel_id]
        sheet_path = target if target.startswith("xl/") else f"xl/{target.lstrip('/')}"
        sheet = ET.fromstring(archive.read(sheet_path))

        rows = []
        max_col = -1
        for row in (node for node in sheet.iter() if node.tag.endswith("}row")):
            values = {}
            for cell in (node for node in row if node.tag.endswith("}c")):
                ref = cell.attrib.get("r", "A1")
                letters = "".join(ch for ch in ref if ch.isalpha())
                col = 0
                for ch in letters:
                    col = col * 26 + (ord(ch.upper()) - 64)
                col -= 1
                max_col = max(max_col, col)
                cell_type = cell.attrib.get("t")
                value_node = next((node for node in cell if node.tag.endswith("}v")), None)
                inline_nodes = [node for node in cell.iter() if node.tag.endswith("}t")]
                value = None
                if cell_type == "inlineStr" and inline_nodes:
                    value = "".join(node.text or "" for node in inline_nodes)
                elif value_node is not None:
                    raw = value_node.text or ""
                    if cell_type == "s":
                        try:
                            value = shared[int(raw)]
                        except (ValueError, IndexError):
                            value = raw
                    else:
                        try:
                            value = float(raw)
                        except ValueError:
                            value = raw
                values[col] = value
            rows.append(values)

    width = max_col + 1
    return pd.DataFrame([[row.get(col) for col in range(width)] for row in rows])


def parse_spdr_holdings_table(raw: pd.DataFrame) -> List[str]:
    """Extract equity tickers from a raw SPDR holdings worksheet.

    The parser keys off the actual Ticker/Name header and excludes only explicit
    cash rows, so company names containing the word ``Cash`` remain valid.
    """
    if raw is None or raw.empty:
        return []

    header_idx = None
    ticker_col = None
    name_col = None
    for i, row in raw.iterrows():
        values = [str(v).strip() if pd.notna(v) else "" for v in row.tolist()]
        lowered = [v.lower() for v in values]
        if "ticker" in lowered:
            header_idx = i
            ticker_col = lowered.index("ticker")
            name_col = lowered.index("name") if "name" in lowered else None
            break

    if header_idx is None or ticker_col is None:
        return []

    out: List[str] = []
    for _, row in raw.loc[header_idx + 1 :].iterrows():
        ticker = (
            str(row.iloc[ticker_col]).strip().upper()
            if pd.notna(row.iloc[ticker_col])
            else ""
        )
        name = ""
        if name_col is not None and name_col < len(row) and pd.notna(row.iloc[name_col]):
            name = str(row.iloc[name_col]).strip().upper()

        if not ticker or ticker in {"NAN", "NONE", "-"}:
            continue
        if (
            ticker.startswith("CASH_")
            or ticker in {"USD", "CASH"}
            or name in {"US DOLLAR", "U.S. DOLLAR", "CASH"}
        ):
            continue
        if ticker.replace(".", "").replace("-", "").isalnum():
            out.append(ticker)

    return list(dict.fromkeys(out))
