"""Refresh the Streamlit fallback list from the official SEC Form 13F data page.

Streamlit Cloud can occasionally receive a 403 when it tries to scrape the SEC
landing page at request time.  This script is intended to run in GitHub Actions
around each quarterly SEC bulk-data publication window.  It snapshots the
latest official release metadata into the repository and rewrites the fallback
list used by the hosted app, so no manual code update is required each quarter.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from adfm_core.sec_13f import QuarterDataset, discover_quarter_datasets

ROOT = Path(__file__).resolve().parents[1]
BROWSER_PATH = ROOT / "adfm_core" / "sec_13f_browser.py"
MANIFEST_PATH = ROOT / "data" / "13f" / "releases.json"
MAX_FALLBACK_RELEASES = 12

os.environ.setdefault(
    "ADFM_SEC_USER_AGENT",
    "AD Fund Management LP aryadeniz@adfundmgmt.com",
)


def _python_block(releases: list[QuarterDataset]) -> str:
    lines = ["OFFICIAL_RELEASE_FALLBACKS = ("]
    for release in releases:
        lines.extend(
            [
                "    QuarterDataset(",
                f"        slug={release.slug!r},",
                f"        label={release.label!r},",
                f"        url={release.url!r},",
                f"        size_label={release.size_label!r},",
                "    ),",
            ]
        )
    lines.append(")")
    return "\n".join(lines)


def _manifest_payload(releases: list[QuarterDataset]) -> str:
    payload = [
        {
            "slug": release.slug,
            "label": release.label,
            "url": release.url,
            "size_label": release.size_label,
        }
        for release in releases
    ]
    return json.dumps(payload, indent=2) + "\n"


def refresh() -> bool:
    releases = discover_quarter_datasets()[:MAX_FALLBACK_RELEASES]
    if not releases:
        raise RuntimeError("SEC discovery returned no Form 13F releases")

    source = BROWSER_PATH.read_text(encoding="utf-8")
    replacement = _python_block(releases)
    pattern = re.compile(
        r"OFFICIAL_RELEASE_FALLBACKS = \(\n.*?\n\)\n\nos\.environ\.setdefault",
        flags=re.DOTALL,
    )
    updated, count = pattern.subn(
        replacement + "\n\nos.environ.setdefault",
        source,
        count=1,
    )
    if count != 1:
        raise RuntimeError("Could not locate OFFICIAL_RELEASE_FALLBACKS in browser source")

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest = _manifest_payload(releases)

    changed = updated != source or not MANIFEST_PATH.exists()
    if MANIFEST_PATH.exists():
        changed = changed or MANIFEST_PATH.read_text(encoding="utf-8") != manifest

    BROWSER_PATH.write_text(updated, encoding="utf-8")
    MANIFEST_PATH.write_text(manifest, encoding="utf-8")
    print(f"Latest SEC 13F bulk release: {releases[0].label}")
    print(f"Fallback releases stored: {len(releases)}")
    return changed


if __name__ == "__main__":
    refresh()
