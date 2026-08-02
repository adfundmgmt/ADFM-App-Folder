"""Repository-wide standards that protect every cataloged Streamlit page."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

from adfm_core.catalog import TOOL_CATALOG

ROOT = Path(__file__).resolve().parents[1]


class RepositoryStandardsTests(unittest.TestCase):
    def test_every_cataloged_page_has_a_standard_footer(self) -> None:
        for tool in TOOL_CATALOG:
            source = (ROOT / "pages" / tool.page_filename).read_text(encoding="utf-8")
            self.assertIn(
                "render_footer",
                source,
                msg=f"{tool.page_filename} is missing the shared application footer.",
            )

    def test_every_cataloged_page_parses_and_configures_streamlit(self) -> None:
        for tool in TOOL_CATALOG:
            path = ROOT / "pages" / tool.page_filename
            source = path.read_text(encoding="utf-8")
            ast.parse(source, filename=str(path))
            self.assertIn(
                "st.set_page_config",
                source,
                msg=f"{tool.page_filename} is missing page configuration.",
            )
            self.assertIn(
                "adfm_core",
                source,
                msg=f"{tool.page_filename} does not load the shared ADFM theme.",
            )
            self.assertIn(
                "About This Tool",
                source,
                msg=f"{tool.page_filename} is missing the standardized About This Tool section.",
            )

    def test_runtime_requirements_are_unique_and_use_one_pdf_library(self) -> None:
        requirements = (
            (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        )
        packages = [
            line.strip().lower()
            for line in requirements
            if line.strip() and not line.startswith("#") and not line.startswith("-")
        ]

        self.assertEqual(len(packages), len(set(packages)))
        self.assertIn("pypdf", packages)
        self.assertNotIn("pypdf2", packages)

    def test_governance_and_reproducibility_files_exist(self) -> None:
        for relative_path in (
            ".gitignore",
            "constraints.txt",
            "CHANGELOG.md",
            ".github/CODEOWNERS",
            ".github/dependabot.yml",
            ".github/pull_request_template.md",
            "SECURITY.md",
            "docs/ARCHITECTURE.md",
            "docs/DEVELOPMENT.md",
        ):
            self.assertTrue(
                (ROOT / relative_path).is_file(), msg=f"Missing {relative_path}"
            )


if __name__ == "__main__":
    unittest.main()
