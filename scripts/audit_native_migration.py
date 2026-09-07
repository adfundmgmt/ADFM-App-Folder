"""Inventory the original application without importing or executing UI code.

Run from the repository root. The JSON contains every source's SHA, imports,
function boundaries, control expressions, caches, callbacks, and I/O references.
The Markdown summarizes the page graph; it does not claim runtime verification.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/native-migration"
CONTROLS = {"text_input", "text_area", "number_input", "selectbox", "multiselect", "slider", "select_slider", "checkbox", "toggle", "radio", "date_input", "time_input", "file_uploader", "button", "form_submit_button", "download_button", "data_editor", "pills", "segmented_control"}
OUTPUTS = {"plotly_chart", "dataframe", "table", "metric", "pyplot", "altair_chart", "line_chart", "bar_chart", "area_chart", "html", "tabs", "expander"}

def audit() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    files = [ROOT / "Home.py", ROOT / "adfm_sector_rotation_config.py"]
    for directory in ("pages", "adfm_core", "cte", "scripts"):
        files.extend(sorted((ROOT / directory).rglob("*.py")))
    files = [p for p in files if p.name != Path(__file__).name]
    modules = {p.relative_to(ROOT).with_suffix("").as_posix().replace("/", "."): p.relative_to(ROOT).as_posix() for p in files}
    records = {}
    for path in files:
        source = path.read_text()
        tree = ast.parse(source)
        rel = path.relative_to(ROOT).as_posix()
        record = {"lines": len(source.splitlines()), "sha256": hashlib.sha256(source.encode()).hexdigest(), "imports": [], "local_dependencies": [], "functions": [], "controls": [], "outputs": [], "caches": [], "state_and_io": [], "urls": []}
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                record["imports"].append({"line": node.lineno, "expression": ast.unparse(node)})
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                if isinstance(node, ast.ImportFrom):
                    if node.level:
                        prefix = rel.replace("/", ".").rsplit(".", 2)[0].split(".")
                        base = ".".join(prefix[:len(prefix) - node.level + 1])
                        names = [base + ("." + node.module if node.module else "")]
                    names += [names[0] + "." + a.name for a in node.names]
                for name in names:
                    if name in modules:
                        record["local_dependencies"].append(modules[name])
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                record["functions"].append({"name": node.name, "line": node.lineno, "end": node.end_lineno, "decorators": [ast.unparse(d) for d in node.decorator_list]})
            if isinstance(node, ast.Call):
                fn = ast.unparse(node.func)
                item = {"line": node.lineno, "expression": ast.unparse(node)}
                if fn.rsplit(".", 1)[-1] in CONTROLS:
                    record["controls"].append(item)
                if fn.rsplit(".", 1)[-1] in OUTPUTS:
                    record["outputs"].append(item)
                if "cache_data" in fn or "cache_resource" in fn or fn.endswith("fragment"):
                    record["caches"].append(item)
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                record["urls"].extend(re.findall(r"https?://[^\s\"<>]+", node.value))
        for line, value in enumerate(source.splitlines(), 1):
            if re.search(r"session_state|st\.secrets|on_change|on_click|os\.environ|os\.getenv|read_(csv|parquet|json|text)|to_(csv|parquet)|write_text|requests\.|http_get\(|exec\(|importlib", value):
                record["state_and_io"].append({"line": line, "expression": value.strip()})
        record["local_dependencies"] = sorted(set(record["local_dependencies"]))
        record["urls"] = sorted(set(record["urls"]))
        records[rel] = record
    # Runtime source loading is invisible to the import graph.
    records["pages/3_Liquidity_Conditions_Monitor.py"]["local_dependencies"].append("adfm_core/_liquidity_tracker_base.py")
    def closure(path, seen=None):
        seen = set() if seen is None else seen
        for dep in records[path]["local_dependencies"]:
            if dep not in seen:
                seen.add(dep)
                closure(dep, seen)
        return sorted(seen)
    pages = sorted((ROOT / "pages").glob("*.py"), key=lambda p: int(p.name.split("_", 1)[0]))
    catalog_tree = ast.parse((ROOT / "adfm_core/catalog.py").read_text())
    catalog = {}
    for node in ast.walk(catalog_tree):
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "ToolDefinition":
            values = [ast.literal_eval(a) for a in node.args]
            catalog[values[2]] = dict(zip(("number", "title", "filename", "group", "description", "data_sources"), values))
    summary = []
    for p in pages:
        rel = p.relative_to(ROOT).as_posix()
        info = dict(catalog[p.name], path=rel, dependencies=closure(rel))
        active = [records[x] for x in [rel] + info["dependencies"] if x != "adfm_core/ui.py"]
        info.update(lines=records[rel]["lines"], control_calls=sum(len(r["controls"]) for r in active), chart_table_calls=sum(len(r["outputs"]) for r in active))
        summary.append(info)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    result = {"baseline_commit": sha, "method": "AST and text audit of every Python source; static counts include conditional/dead branches. Package-init UI side effects documented separately.", "pages": summary, "files": records}
    (OUT / "source-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Source inventory", "", f"Baseline: `{sha}`. {len(pages)} pages plus Home. {len(records)} Python files inspected.", "", "Static call counts are audit clues, not the number of rendered charts or controls. Conditional paths and imported renderers must be checked during each page migration.", "", "| Page | File lines | Python dependencies (transitive) | Data sources |", "|---|---:|---|---|"]
    for p in summary:
        deps = ", ".join("`" + d.removeprefix("adfm_core/").removeprefix("cte/") + "`" for d in p["dependencies"] if d not in {"adfm_core/ui.py", "adfm_core/palette.py", "adfm_core/catalog.py", "adfm_core/observability.py"})
        lines.append(f"| {p['number']}. {p['title']} | {p['lines']} | {deps or 'Page-local functions'} | {p['data_sources']} |")
    lines += ["", "See `source-audit.json` for full control/default expressions, functions, decorator TTLs, state/callback references, URLs, and hashes for **every** file. Presentation imports also execute `adfm_core/__init__.py`; its global patches are not included in the transitive counts above.", ""]
    (OUT / "INVENTORY.md").write_text("\n".join(lines))
    print(f"Audited {len(records)} Python files; {len(pages)} catalogued pages.")

if __name__ == "__main__":
    audit()
